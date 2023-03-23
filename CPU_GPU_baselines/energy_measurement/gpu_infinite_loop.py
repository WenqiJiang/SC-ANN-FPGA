"""

This script is used to measure the GPU energy consumption. 
Give a search setting, the script will iterate on that setting forever. 
In the meantime, use `nvidia-smi -l 1 > out_energy` to print the energy consumption per second.
Use another script to read the nvidia-smi log, average the consumption of the given GPU

Usage:
python gpu_infinite_loop.py -dbname SIFT100M -index_key IVF65536,PQ16 -topK 100 -ngpu 1 -startgpu 0 -tempmem $[1536*1024*1024] -nprobe 32 -qbs 10000

"""

from __future__ import print_function
import numpy as np
import time
import os
import sys
import faiss
import inspect
import re
import pickle

from multiprocessing.dummy import Pool as ThreadPool

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# from .. import datasets
from datasets import ivecs_read
from datasets import read_deep_fbin, read_deep_ibin, mmap_bvecs_SBERT, mmap_bvecs_GNN

####################################################################
# Parse command line
####################################################################

cur_script_dir = os.path.dirname(os.path.abspath(__file__))

def usage():
    print("""

Besides training / searching, 
this script can also be used to automatically search the min(nprobe) that can achieve
the target recall R@k using dataset D and index I.

Usage: bench_gpu_1bn.py dataset indextype [options]

dataset: set of vectors to operate on.
   Supported: SIFT1M, SIFT2M, ..., SIFT1000M or Deep1B

indextype: any index type supported by index_factory that runs on GPU.

    General options

-ngpu ngpu         nb of GPUs to use (default = all)
-tempmem N         use N bytes of temporary GPU memory
-nocache           do not read or write intermediate files
-float16           use 16-bit floats on the GPU side

-recall_goal       this option evaluates recall

    Add options

-abs N             split adds in blocks of no more than N vectors
-max_add N         copy sharded dataset to CPU each max_add additions
                   (to avoid memory overflows with geometric reallocations)
-altadd            Alternative add function, where the index is not stored
                   on GPU during add. Slightly faster for big datasets on
                   slow GPUs

    Search options

-R R:              nb of replicas of the same dataset (the dataset
                   will be copied across ngpu/R, default R=1)
-noptables         do not use precomputed tables in IVFPQ.
-qbs N             split queries in blocks of no more than N vectors
-topK N             search N neighbors for each query
-nprobe 4,16,64    try this number of probes
-knngraph          instead of the standard setup for the dataset,
                   compute a k-nn graph with topK neighbors per element
-oI xx%d.npy       output the search result indices to this numpy file,
                   %d will be replaced with the nprobe
-oD xx%d.npy       output the search result distances to this file

""", file=sys.stderr)
    sys.exit(1)

query_num_factor = 1 # the default query num = 10K, can set query_num_factor as 10 to run 100K query to get a more stable performance

# default values

dbname = None
index_key = None

ngpu = faiss.get_num_gpus()

replicas = 1  # nb of replicas of sharded dataset
add_batch_size = 32768
query_batch_size = 10000

nprobes = [1 << l for l in range(10)]
knngraph = False
# Wenqi edited, origin use_precomputed_tables=True
#use_precomputed_tables = False
use_precomputed_tables = True
tempmem = -1  # if -1, use system default
max_add = -1
use_float16 = False
use_cache = True
startgpu=0
topK = 10
altadd = False
I_fname = None
D_fname = None
recall_goal = None

load_from_dict = None 
overwrite = 0
nprobe_dict_dir = None
throughput_dict_dir = None
response_time_dict_dir = None

args = sys.argv[1:]

while args:
    a = args.pop(0)
    if a == '-h': usage()
    elif a == '-ngpu':      ngpu = int(args.pop(0))
    elif a == '-startgpu':  startgpu = int(args.pop(0)) # Wenqi, the id the first GPU used, e.g., 1 -> skip GPU0
    elif a == '-R':         replicas = int(args.pop(0))
    elif a == '-noptables': use_precomputed_tables = False
    elif a == '-abs':       add_batch_size = int(args.pop(0))
    elif a == '-qbs':       query_batch_size = int(args.pop(0))
    elif a == '-topK':       topK = int(args.pop(0))
    elif a == '-tempmem':   tempmem = int(args.pop(0))
    elif a == '-nocache':   use_cache = False
    elif a == '-knngraph':  knngraph = True
    elif a == '-altadd':    altadd = True
    elif a == '-float16':   use_float16 = True
    elif a == '-nprobe':    nprobes = [int(x) for x in args.pop(0).split(',')]
    elif a == '-max_add':   max_add = int(args.pop(0))
    elif a == '-recall_goal': recall_goal = int(args.pop(0)) / 100.0
    elif a == '-load_from_dict': load_from_dict = int(args.pop(0))
    elif a == '-overwrite': overwrite = int(args.pop(0))
    elif a == '-nprobe_dict_dir': nprobe_dict_dir = str(args.pop(0))
    elif a == '-throughput_dict_dir': throughput_dict_dir = str(args.pop(0))
    elif a == '-response_time_dict_dir': response_time_dict_dir = str(args.pop(0))
    elif a == '-dbname': dbname = str(args.pop(0))
    elif a == '-index_key': index_key = str(args.pop(0))
    else:
        print("argument %s unknown" % a, file=sys.stderr)
        sys.exit(1)

print("query_batch_size: ", query_batch_size)

#################################################################
# Small Utility Functions
#################################################################

# we mem-map the biggest files to avoid having them in memory all at
# once

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


class IdentPreproc:
    """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x


def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))


def dataset_iterator(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    def prepare_block(i01):
        i0, i1 = i01
        xb = sanitize(x[i0:i1])
        return i0, preproc.apply_py(xb)

    return rate_limited_imap(prepare_block, block_ranges)


def eval_intersection_measure(gt_I, I):
    """ measure intersection measure (used for knngraph)"""
    inter = 0
    rank = I.shape[1]
    assert gt_I.shape[1] >= rank
    for q in range(nq_gt):
        inter += faiss.ranklist_intersection_size(
            rank, faiss.swig_ptr(gt_I[q, :]),
            rank, faiss.swig_ptr(I[q, :].astype('int64')))
    return inter / float(rank * nq_gt)


#################################################################
# Prepare dataset
#################################################################

cacheroot = None
dbsize = None
xb = None
xq = None
xt = None
gt_I = None
preproc_str = None
ivf_str = None
pqflat_str = None
ncent = None
prefix = None
gt_cachefile = None
cent_cachefile = None
index_cachefile = None
preproc_cachefile = None

nq_gt = None
gt_sl = None

if dbname: 
    print("Preparing dataset", dbname)

    cacheroot = os.path.abspath(os.path.join(cur_script_dir, '../trained_CPU_indexes/bench_cpu_{}_{}'.format(dbname, index_key)))
    #cacheroot = os.path.relpath('./trained_GPU_indexes/bench_gpu_{}_{}'.format(dbname, index_key))

    if not os.path.isdir(cacheroot):
        print("%s does not exist, creating it" % cacheroot)
        os.mkdir(cacheroot)

    if dbname.startswith('SIFT'):
        # SIFT1M to SIFT1000M
        dbsize = int(dbname[4:-1])
        # xb = mmap_bvecs(os.path.abspath(os.path.join(cur_script_dir, '../bigann/bigann_base.bvecs')))
        xq = mmap_bvecs(os.path.abspath(os.path.join(cur_script_dir, '../bigann/bigann_query.bvecs')))

        # Wenqi adjusted, only use the first 1000 queries
        # print(xq.shape)
        # xq = xq[:1000]
        print(xq.shape)

        # xt = mmap_bvecs(os.path.abspath(os.path.join(cur_script_dir, '../bigann/bigann_learn.bvecs')))

        # trim xb to correct size
        # xb = xb[:dbsize * 1000 * 1000]

        gt_I = ivecs_read(os.path.abspath(os.path.join(cur_script_dir, '../bigann/gnd/idx_%dM.ivecs' % dbsize)))

    elif dbname.startswith('Deep'):

        assert dbname[:4] == 'Deep' 
        assert dbname[-1] == 'M'
        dbsize = int(dbname[4:-1]) # in million
        # xb = read_deep_fbin('deep1b/base.1B.fbin')[:dbsize * 1000 * 1000]
        xq = read_deep_fbin('../deep1b/query.public.10K.fbin')
        # xt = read_deep_fbin('deep1b/learn.350M.fbin')

        gt_I = read_deep_ibin('../deep1b/gt_idx_{}M.ibin'.format(dbsize))

        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)
    elif dbname.startswith('SBERT'):
        # FB1M to FB1000M
        dataset_dir = '../sbert'
        assert dbname[:5] == 'SBERT' 
        assert dbname[-1] == 'M'
        dbsize = int(dbname[5:-1]) # in million
        # xb = mmap_bvecs_SBERT('sbert/sbert3B.fvecs', num_vec=int(dbsize * 1e6))
        xq = mmap_bvecs_SBERT('../sbert/query_10K.fvecs', num_vec=10 * 1000)
        # xt = xb

        # trim to correct size
        # xb = xb[:dbsize * 1000 * 1000]
        
        gt = read_deep_ibin('../sbert/gt_idx_{}M.ibin'.format(dbsize), dtype='uint32')

        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)

        query_num = xq.shape[0]
        print('query shape: ', xq.shape)
        # Wenqi: use true for >= 64 byte PQ code
        # https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
        # RuntimeError: Error in void faiss::gpu::GpuIndexIVFPQ::verifySettings_() const at 
        #   /root/miniconda3/conda-bld/faiss-pkg_1641228905850/work/faiss/gpu/GpuIndexIVFPQ.cu:443: 
        #   Error: 'requiredSmemSize <= getMaxSharedMemPerBlock(config_.device)' failed: Device 0 has 
        #   49152 bytes of shared memory, while 8 bits per code and 64 sub-quantizers requires 65536 bytes. 
        #   Consider useFloat16LookupTables and/or reduce parameters
        use_float16 = True 

    elif dbname.startswith('GNN'):
        # FB1M to FB1000M
        dataset_dir = '../MariusGNN/'
        assert dbname[:3] == 'GNN' 
        assert dbname[-1] == 'M'
        dbsize = int(dbname[3:-1]) # in million
        # xb = mmap_bvecs_GNN('MariusGNN/embeddings.bin', num_vec=int(dbsize * 1e6))
        xq = mmap_bvecs_GNN('../MariusGNN/query_10K.fvecs', num_vec=10 * 1000)
        # xt = xb

        # trim to correct size
        # xb = xb[:dbsize * 1000 * 1000]

        gt = read_deep_ibin('../MariusGNN/gt_idx_{}M.ibin'.format(dbsize), dtype='uint32') 
        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)
        # The dataset is highly skewed (imbalance factor > 30), only search a subset to speedup the test
        num_query_for_eval = 1000
        xq = xq[:num_query_for_eval]
        gt = gt[:num_query_for_eval]

        query_num = xq.shape[0]
        print('query shape: ', xq.shape)

        # Wenqi: use true for >= 64 byte PQ code
        # https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
        # RuntimeError: Error in void faiss::gpu::GpuIndexIVFPQ::verifySettings_() const at 
        #   /root/miniconda3/conda-bld/faiss-pkg_1641228905850/work/faiss/gpu/GpuIndexIVFPQ.cu:443: 
        #   Error: 'requiredSmemSize <= getMaxSharedMemPerBlock(config_.device)' failed: Device 0 has 
        #   49152 bytes of shared memory, while 8 bits per code and 64 sub-quantizers requires 65536 bytes. 
        #   Consider useFloat16LookupTables and/or reduce parameters
        use_float16 = True 

        # 4 GPU for 1400M: RuntimeError: Exception thrown from index 0: Error in virtual 
        # void* faiss::gpu::StandardGpuResourcesImpl::allocMemory(const faiss::gpu::AllocRequest&) 
        # at /root/miniconda3/conda-bld/faiss-pkg_1641228905850/work/faiss/gpu/StandardGpuResources.cpp:452: 
        # Error: 'err == cudaSuccess' failed: StandardGpuResources: alloc fail type TemporaryMemoryOverflow 
        # dev 0 space Device stream 0x7effcc5f47a0 size 4513349632 bytes (cudaMalloc error out of memory [2])

    else:
        print('unknown dataset', dbname, file=sys.stderr)
        sys.exit(1)


    if knngraph:
        # convert to knn-graph dataset
        xq = xb
        xt = xb
        # we compute the ground-truth on this number of queries for validation
        nq_gt = 10000
        gt_sl = 100

        # ground truth will be computed below
        gt_I = None


    # print("sizes: B %s Q %s T %s gt %s" % (
    #     xb.shape, xq.shape, xt.shape,
    #     gt_I.shape if gt_I is not None else None))



    #################################################################
    # Parse index_key and set cache files
    #
    # The index_key is a valid factory key that would work, but we
    # decompose the training to do it faster
    #################################################################


    pat = re.compile('(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,)?' +
                    '(IVF[0-9]+),' +
                    '(PQ[0-9]+|Flat)')

    matchobject = pat.match(index_key)

    assert matchobject, 'could not parse ' + index_key

    mog = matchobject.groups()

    preproc_str = mog[0]
    ivf_str = mog[2]
    pqflat_str = mog[3]

    ncent = int(ivf_str[3:])

    prefix = ''

    if knngraph:
        gt_cachefile = '%s/BK_gt_%s.npy' % (cacheroot, dbname)
        prefix = 'BK_'
        # files must be kept distinct because the training set is not the
        # same for the knngraph

    if preproc_str:
        preproc_cachefile = '%s/%spreproc_%s_%s.vectrans' % (
            cacheroot, prefix, dbname, preproc_str[:-1])
    else:
        preproc_cachefile = None
        preproc_str = ''

    cent_cachefile = '%s/%scent_%s_%s%s.npy' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str)

    index_cachefile = '%s/%s%s_%s%s,%s_populated.index' % (
        cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)


    if not use_cache:
        preproc_cachefile = None
        cent_cachefile = None
        index_cachefile = None

    print("cachefiles:")
    print(preproc_cachefile)
    print(cent_cachefile)
    print(index_cachefile)


#################################################################
# Wake up GPUs
#################################################################

print("preparing resources for %d GPUs" % ngpu)

gpu_resources = []

###### hardcode here #######
total_gpu = 8 # spaceml 1, select the last GPU for our use

# skip the first 7
#for i in range(total_gpu):
#    res = faiss.StandardGpuResources()
#    if tempmem >= 0:
#        res.setTempMemory(tempmem)
#    gpu_resources.append(res)
#print("GPU resourses:", gpu_resources)
#gpu_resources = [gpu_resources[-1]]
#print("Select the last one:\nGPU resourses:", gpu_resources)
for i in range(ngpu):
    res = faiss.StandardGpuResources()
    if tempmem >= 0:
        res.setTempMemory(tempmem)
    gpu_resources.append(res)


# Wenqi: I guess this is where we can adjust GPU resources
def make_vres_vdev(i0=0, i1=-1):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = ngpu
#    for i in range(i0, i1):
        #vdev.push_back(i)
        #vres.push_back(gpu_resources[i])
    # WENQI: Start from assigned GPU
    for i in range(i0, i1):
        vdev.push_back(i + startgpu)
        vres.push_back(gpu_resources[i])
    return vres, vdev


#################################################################
# Prepare ground truth (for the knngraph)
#################################################################


def compute_GT():
    print("compute GT")
    t0 = time.time()

    gt_I = np.zeros((nq_gt, gt_sl), dtype='int64')
    gt_D = np.zeros((nq_gt, gt_sl), dtype='float32')
    heaps = faiss.float_maxheap_array_t()
    heaps.k = gt_sl
    heaps.nh = nq_gt
    heaps.val = faiss.swig_ptr(gt_D)
    heaps.ids = faiss.swig_ptr(gt_I)
    heaps.heapify()
    bs = 10 ** 5

    n, d = xb.shape
    xqs = sanitize(xq[:nq_gt])

    db_gt = faiss.IndexFlatL2(d)
    vres, vdev = make_vres_vdev()
    db_gt_gpu = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, db_gt)

    # compute ground-truth by blocks of bs, and add to heaps
    for i0, xsl in dataset_iterator(xb, IdentPreproc(d), bs):
        db_gt_gpu.add(xsl)
        D, I = db_gt_gpu.search(xqs, gt_sl)
        I += i0
        heaps.addn_with_ids(
            gt_sl, faiss.swig_ptr(D), faiss.swig_ptr(I), gt_sl)
        db_gt_gpu.reset()
        print("\r   %d/%d, %.3f s" % (i0, n, time.time() - t0), end=' ')
    print()
    heaps.reorder()

    print("GT time: %.3f s" % (time.time() - t0))
    return gt_I

if dbname: 
    if knngraph:

        if gt_cachefile and os.path.exists(gt_cachefile):
            print("load GT", gt_cachefile)
            gt_I = np.load(gt_cachefile)
        else:
            gt_I = compute_GT()
            if gt_cachefile:
                print("store GT", gt_cachefile)
                np.save(gt_cachefile, gt_I)

#################################################################
# Prepare the vector transformation object (pure CPU)
#################################################################


def get_preprocessor():
    
    if preproc_str:
        if not preproc_cachefile or not os.path.exists(preproc_cachefile):
            raise ValueError
        else:
            print("load", preproc_cachefile)
            preproc = faiss.read_VectorTransform(preproc_cachefile)
    else:
        d = xq.shape[1]
        preproc = IdentPreproc(d)
    return preproc


#################################################################
# Prepare the coarse quantizer
#################################################################

def prepare_coarse_quantizer(preproc):

    if cent_cachefile and os.path.exists(cent_cachefile):
        print("load centroids", cent_cachefile)
        centroids = np.load(cent_cachefile)
    else:
        raise ValueError

    coarse_quantizer = faiss.IndexFlatL2(preproc.d_out)
    coarse_quantizer.add(centroids)

    return coarse_quantizer


#################################################################
# Make index and add elements to it
#################################################################


def prepare_trained_index(preproc):

    coarse_quantizer = prepare_coarse_quantizer(preproc)
    d = preproc.d_out
    if pqflat_str == 'Flat':
        print("making an IVFFlat index")
        idx_model = faiss.IndexIVFFlat(coarse_quantizer, d, ncent,
                                       faiss.METRIC_L2)
    else:
        m = int(pqflat_str[2:])
        assert m < 56 or use_float16, "PQ%d will work only with -float16" % m
        print("making an IVFPQ index, m = ", m)
        idx_model = faiss.IndexIVFPQ(coarse_quantizer, d, ncent, m, 8)

    coarse_quantizer.this.disown()
    idx_model.own_fields = True

    # finish training on CPU
    t0 = time.time()
    print("Training vector codes")
    x = preproc.apply_py(sanitize(xt[:1000000]))
    idx_model.train(x)
    print("  done %.3f s" % (time.time() - t0))

    return idx_model


def compute_populated_index(preproc):
    """Add elements to a sharded index. Return the index and if available
    a sharded gpu_index that contains the same data. """

    indexall = prepare_trained_index(preproc)

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = use_precomputed_tables
    co.indicesOptions = faiss.INDICES_CPU
    co.verbose = True
    co.reserveVecs = max_add if max_add > 0 else xb.shape[0]
    co.shard = True
    assert co.shard_type in (0, 1, 2)
    vres, vdev = make_vres_vdev()
    gpu_index = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, indexall, co)

    print("add...")
    t0 = time.time()
    nb = xb.shape[0]
    for i0, xs in dataset_iterator(xb, preproc, add_batch_size):
        i1 = i0 + xs.shape[0]
        gpu_index.add_with_ids(xs, np.arange(i0, i1))
        if max_add > 0 and gpu_index.ntotal > max_add:
            print("Flush indexes to CPU")
            for i in range(ngpu):
                index_src_gpu = faiss.downcast_index(gpu_index.at(i))
                index_src = faiss.index_gpu_to_cpu(index_src_gpu)
                print("  index %d size %d" % (i, index_src.ntotal))
                index_src.copy_subset_to(indexall, 0, 0, nb)
                index_src_gpu.reset()
                index_src_gpu.reserveMemory(max_add)
            gpu_index.sync_with_shard_indexes()

        print('\r%d/%d (%.3f s)  ' % (
            i0, nb, time.time() - t0), end=' ')
        sys.stdout.flush()
    print("Add time: %.3f s" % (time.time() - t0))

    print("Aggregate indexes to CPU")
    t0 = time.time()

    if hasattr(gpu_index, 'at'):
        # it is a sharded index
        for i in range(ngpu):
            index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
            print("  index %d size %d" % (i, index_src.ntotal))
            index_src.copy_subset_to(indexall, 0, 0, nb)
    else:
        # simple index
        index_src = faiss.index_gpu_to_cpu(gpu_index)
        index_src.copy_subset_to(indexall, 0, 0, nb)

    print("  done in %.3f s" % (time.time() - t0))

    if max_add > 0:
        # it does not contain all the vectors
        gpu_index = None

    return gpu_index, indexall

def compute_populated_index_2(preproc):

    indexall = prepare_trained_index(preproc)

    # set up a 3-stage pipeline that does:
    # - stage 1: load + preproc
    # - stage 2: assign on GPU
    # - stage 3: add to index

    stage1 = dataset_iterator(xb, preproc, add_batch_size)

    vres, vdev = make_vres_vdev()
    coarse_quantizer_gpu = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, indexall.quantizer)

    def quantize(args):
        (i0, xs) = args
        _, assign = coarse_quantizer_gpu.search(xs, 1)
        return i0, xs, assign.ravel()

    stage2 = rate_limited_imap(quantize, stage1)

    print("add...")
    t0 = time.time()
    nb = xb.shape[0]

    for i0, xs, assign in stage2:
        i1 = i0 + xs.shape[0]
        if indexall.__class__ == faiss.IndexIVFPQ:
            indexall.add_core_o(i1 - i0, faiss.swig_ptr(xs),
                                None, None, faiss.swig_ptr(assign))
        elif indexall.__class__ == faiss.IndexIVFFlat:
            indexall.add_core(i1 - i0, faiss.swig_ptr(xs), None,
                              faiss.swig_ptr(assign))
        else:
            assert False

        print('\r%d/%d (%.3f s)  ' % (
            i0, nb, time.time() - t0), end=' ')
        sys.stdout.flush()
    print("Add time: %.3f s" % (time.time() - t0))

    return None, indexall



def get_populated_index(preproc):

    if not index_cachefile or not os.path.exists(index_cachefile):
        raise ValueError
    else:
        print("load", index_cachefile)
        indexall = faiss.read_index(index_cachefile)
        gpu_index = None

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = use_precomputed_tables
    co.indicesOptions = 0
    co.verbose = True
    co.shard = True    # the replicas will be made "manually"
    t0 = time.time()
    print("CPU index contains %d vectors, move to GPU" % indexall.ntotal)
    if replicas == 1:

        if not gpu_index:
            print("copying loaded index to GPUs")
            vres, vdev = make_vres_vdev()
            index = faiss.index_cpu_to_gpu_multiple(
                vres, vdev, indexall, co)
        else:
            index = gpu_index

    else:
        del gpu_index # We override the GPU index

        print("Copy CPU index to %d sharded GPU indexes" % replicas)

        index = faiss.IndexReplicas()

        for i in range(replicas):
            gpu0 = ngpu * i / replicas
            gpu1 = ngpu * (i + 1) / replicas
            vres, vdev = make_vres_vdev(gpu0, gpu1)

            print("   dispatch to GPUs %d:%d" % (gpu0, gpu1))

            index1 = faiss.index_cpu_to_gpu_multiple(
                vres, vdev, indexall, co)
            index1.this.disown()
            index.addIndex(index1)
        index.own_fields = True
    del indexall
    print("move to GPU done in %.3f s" % (time.time() - t0))
    return index



#################################################################
# Perform search
#################################################################


def eval_dataset(index, preproc):

    ps = faiss.GpuParameterSpace()
    ps.initialize(index)

    nq_gt = gt_I.shape[0]
    print("search...")
    sl = query_batch_size

    nq = xq.shape[0]
    print(nq)

    for nprobe in nprobes:
        ps.set_index_parameter(index, 'nprobe', nprobe)
        t0 = time.time()

        if sl == 0:
            D, I = index.search(preproc.apply_py(sanitize(xq)), topK)
        else:
            I = np.empty((nq, topK), dtype='int32')
            D = np.empty((nq, topK), dtype='float32')

            inter_res = ''

            # Wenqi: run this setting forever...
            print("Start the while loop for search, ready for energy measurement...")
            while True:
                for i0, xs in dataset_iterator(xq, preproc, sl):
                    # print('\r%d/%d (%.3f s%s)   ' % (
                    #     i0, nq, time.time() - t0, inter_res), end=' ')
                    # sys.stdout.flush()

                    i1 = i0 + xs.shape[0]
                    # Wenqi: debugging memory overflow
                    # print(xs.shape)
                    Di, Ii = index.search(xs, topK)

                    I[i0:i1] = Ii
                    D[i0:i1] = Di

                    if knngraph and not inter_res and i1 >= nq_gt:
                        ires = eval_intersection_measure(
                            gt_I[:, :topK], I[:nq_gt])
                        inter_res = ', %.4f' % ires

        t1 = time.time()
        if knngraph:
            ires = eval_intersection_measure(gt_I[:, :topK], I[:nq_gt])
            print("  probe=%-3d: %.3f s rank-%d intersection results: %.4f" % (
                nprobe, t1 - t0, topK, ires))
        else:
            print("  probe=%-3d: %.3f s (QPS=%.3f)" % (nprobe, t1 - t0, nq / (t1 - t0)), end=' ')
            gtc = gt_I[:, :1]
            nq = xq.shape[0]
            # WENQI modified, when only using 1000 query, comment below
            # because groud truth verification have problems with shape
            rank_list = [1, 10, 100]
            if topK not in rank_list:
                rank_list.append(topK)
            for rank in rank_list:
                if rank > topK: continue
                nok = (I[:, :rank] == gtc).sum()
                print("1-R@%d: %.4f" % (rank, nok / float(nq)), end=' ')
            print()
        if I_fname:
            I_fname_i = I_fname % I
            print("storing", I_fname_i)
            np.save(I, I_fname_i)
        if D_fname:
            D_fname_i = I_fname % I
            print("storing", D_fname_i)
            np.save(D, D_fname_i)


def eval_dataset_from_dict():

    # load recall dictionary
    d_nprobes = None
    if os.path.exists(nprobe_dict_dir):
        with open(nprobe_dict_dir, 'rb') as f:
            d_nprobes = pickle.load(f)
    else:
        print("ERROR! input dictionary does not exists")
        raise ValueError

    d_throughput = None
    if os.path.exists(throughput_dict_dir):
        with open(throughput_dict_dir, 'rb') as f:
            d_throughput = pickle.load(f)
    else:
        d_throughput = dict()

    if os.path.exists(response_time_dict_dir):
        with open(response_time_dict_dir, 'rb') as f:
            d_response_time = pickle.load(f)
    else:
        d_response_time = dict()

    for dbname in d_nprobes:

        if dbname.startswith('SIFT'):
            # SIFT1M to SIFT1000M
            global dbsize
            global xb
            global xq
            global xt
            global gt_I

            dbsize = int(dbname[4:-1])
            xb = mmap_bvecs(os.path.abspath(os.path.join(cur_script_dir, 'bigann/bigann_base.bvecs')))
            xq = mmap_bvecs(os.path.abspath(os.path.join(cur_script_dir, 'bigann/bigann_query.bvecs')))
            xt = mmap_bvecs(os.path.abspath(os.path.join(cur_script_dir, 'bigann/bigann_learn.bvecs')))

            gt = ivecs_read(os.path.abspath(os.path.join(cur_script_dir, 'bigann/gnd/idx_%dM.ivecs' % dbsize)))

            # Wenqi: load xq to main memory and reshape
            xq = xq.astype('float32').copy()
            xq = np.array(xq, dtype=np.float32)
            xq = np.tile(xq, (query_num_factor, 1)) # replicate the 10K queries to 100K queries to get a more stable performance
            gt = np.array(gt, dtype=np.int32)
            gt_I = ivecs_read(os.path.abspath(os.path.join(cur_script_dir, 'bigann/gnd/idx_%dM.ivecs' % dbsize)))
            gt = np.tile(gt, (query_num_factor, 1))
            gt_I = np.tile(gt_I, (query_num_factor, 1))
        else:
            print('unknown dataset', dbname, file=sys.stderr)
            sys.exit(1)

        nq, d = xq.shape
        assert gt.shape[0] == nq
        
        if dbname not in d_throughput:
            d_throughput[dbname] = dict()
        if dbname not in d_response_time:
            d_response_time[dbname] = dict()

        for index_key in d_nprobes[dbname]:

            if index_key not in d_throughput[dbname]:
                d_throughput[dbname][index_key] = dict()
            if index_key not in d_response_time[dbname]:
                d_response_time[dbname][index_key] = dict()

            cacheroot = os.path.abspath(os.path.join(cur_script_dir, './trained_GPU_indexes/bench_gpu_{}_{}'.format(dbname, index_key)))
            pat = re.compile('(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,)?' +
                            '(IVF[0-9]+),' +
                            '(PQ[0-9]+|Flat)')
            matchobject = pat.match(index_key)
            assert matchobject, 'could not parse ' + index_key
            mog = matchobject.groups()
            global preproc_str
            global ivf_str
            global pqflat_str
            global ncent
            global prefix
            preproc_str = mog[0]
            ivf_str = mog[2]
            pqflat_str = mog[3]

            ncent = int(ivf_str[3:])

            prefix = ''

            global gt_cachefile
            global preproc_cachefile
            global cent_cachefile
            global index_cachefile

            global gt_cachefile 
            global cent_cachefile 
            global index_cachefile 
            global preproc_cachefile 

            global nq_gt 
            global gt_sl 
            
            if knngraph:
                gt_cachefile = '%s/BK_gt_%s.npy' % (cacheroot, dbname)
                prefix = 'BK_'
                # files must be kept distinct because the training set is not the
                # same for the knngraph

            if preproc_str:
                preproc_cachefile = '%s/%spreproc_%s_%s.vectrans' % (
                    cacheroot, prefix, dbname, preproc_str[:-1])
            else:
                preproc_cachefile = None
                preproc_str = ''

            cent_cachefile = '%s/%scent_%s_%s%s.npy' % (
                cacheroot, prefix, dbname, preproc_str, ivf_str)

            index_cachefile = '%s/%s%s_%s%s,%s.index' % (
                cacheroot, prefix, dbname, preproc_str, ivf_str, pqflat_str)


            if not use_cache:
                preproc_cachefile = None
                cent_cachefile = None
                index_cachefile = None

            print("cachefiles:")
            print(preproc_cachefile)
            print(cent_cachefile)
            print(index_cachefile)

            preproc = get_preprocessor()
            index = get_populated_index(preproc)

            ps = faiss.GpuParameterSpace()
            ps.initialize(index)

            nq_gt = gt_I.shape[0]
            sl = query_batch_size

            nq = xq.shape[0]

            for topK in d_nprobes[dbname][index_key]:

                if topK not in d_throughput[dbname][index_key]:
                    d_throughput[dbname][index_key][topK] = dict()
                if topK not in d_response_time[dbname][index_key]:
                    d_response_time[dbname][index_key][topK] = dict()

                for recall_goal in d_nprobes[dbname][index_key][topK]:

                    if recall_goal not in d_throughput[dbname][index_key][topK]:
                        d_throughput[dbname][index_key][topK][recall_goal] = dict()
                    if recall_goal not in d_response_time[dbname][index_key][topK]:
                        d_response_time[dbname][index_key][topK][recall_goal] = dict()
                        
                    # skip if there's already a QPS
                    if d_throughput[dbname][index_key][topK][recall_goal] and d_response_time[dbname][index_key][topK] and (not overwrite): 
                        print("SKIP TEST.\tDB: {}\tindex: {}\ttopK: {}\trecall goal: {}\t".format(
                            dbname, index_key, topK, recall_goal))
                        continue

                    if d_nprobes[dbname][index_key][topK][recall_goal] is not None:

                        nprobe = d_nprobes[dbname][index_key][topK][recall_goal]

                        ps.set_index_parameter(index, 'nprobe', nprobe)
                        t0 = time.time()

                        if sl == 0:
                            D, I = index.search(preproc.apply_py(sanitize(xq)), topK)
                        else:
                            I = np.empty((nq, topK), dtype='int32')
                            D = np.empty((nq, topK), dtype='float32')

                            inter_res = ''

                            response_time = [] # in terms of ms
                            for i0, xs in dataset_iterator(xq, preproc, sl):
                                # print('\r%d/%d (%.3f s%s)   ' % (
                                #     i0, nq, time.time() - t0, inter_res), end=' ')
                                # sys.stdout.flush()

                                i1 = i0 + xs.shape[0]
                                # Wenqi: debugging memory overflow
                                # print(xs.shape)
                                t_RT_start = time.time()
                                Di, Ii = index.search(xs, topK)
                                t_RT_end = time.time()
                                response_time.append(1000 * (t_RT_end - t_RT_start)) 

                                I[i0:i1] = Ii
                                D[i0:i1] = Di

                                if knngraph and not inter_res and i1 >= nq_gt:
                                    ires = eval_intersection_measure(
                                        gt_I[:, :topK], I[:nq_gt])
                                    inter_res = ', %.4f' % ires

                        t1 = time.time()
                        throughput = nq / (t1 - t0)
                        print("DB: {}\tindex: {}\ttopK: {}\trecall goal: {}\tnprobe: {}\tQPS = {}".format(
                            dbname, index_key, topK, recall_goal, nprobe, throughput))
                        if query_batch_size not in d_throughput[dbname][index_key][topK][recall_goal]:
                            d_throughput[dbname][index_key][topK][recall_goal][query_batch_size] = None
                        d_throughput[dbname][index_key][topK][recall_goal][query_batch_size] = throughput

                        response_time = np.array(response_time, dtype=np.float32)
                        if query_batch_size not in d_response_time[dbname][index_key][topK][recall_goal]:
                            d_response_time[dbname][index_key][topK][recall_goal][query_batch_size] = None 
                        d_response_time[dbname][index_key][topK][recall_goal][query_batch_size] = response_time 

                        with open(throughput_dict_dir, 'wb') as f:
                            # dictionary format:
                            #   d[dbname (str)][index_key (str)][topK (int)][recall_goal (float, 0~1)] = QPS
                            #   e.g., d["SIFT100M"]["IVF4096,PQ16"][10][0.7]
                            pickle.dump(d_throughput, f, protocol=4)
                            
                        with open(response_time_dict_dir, 'wb') as f:
                            # dictionary format:
                            #   d[dbname (str)][index_key (str)][topK (int)][recall_goal (float, 0~1)] = response time array (np array)
                            #   e.g., d["SIFT100M"]["IVF4096,PQ16"][10][0.7]
                            pickle.dump(d_response_time, f, protocol=4)

                        if knngraph:
                            ires = eval_intersection_measure(gt_I[:, :topK], I[:nq_gt])
                            print("  probe=%-3d: %.3f s rank-%d intersection results: %.4f" % (
                                nprobe, t1 - t0, topK, ires))
                        else:
                            print("  probe=%-3d: %.3f s" % (nprobe, t1 - t0), end=' ')
                            gtc = gt_I[:, :1]
                            nq = xq.shape[0]
                            # WENQI modified, when only using 1000 query, comment below
                            # because groud truth verification have problems with shape
                            rank_list = [1, 10, 100]
                            if topK not in rank_list:
                                rank_list.append(topK)
                            for rank in rank_list:
                                if rank > topK: continue
                                nok = (I[:, :rank] == gtc).sum()
                                print("1-R@%d: %.4f" % (rank, nok / float(nq)), end=' ')
                            print()
                        if I_fname:
                            I_fname_i = I_fname % I
                            print("storing", I_fname_i)
                            np.save(I, I_fname_i)
                        if D_fname:
                            D_fname_i = I_fname % I
                            print("storing", D_fname_i)
                            np.save(D, D_fname_i)

            del index

def recall_eval(index, preproc):
    """
    This script is used to automatically search the min(nprobe) that can achieve
    the target recall R@k using dataset D and index I.
    """
    nlist = None
    index_array = index_key.split(",")
    if len(index_array) == 2: # "IVF4096,PQ16" 
        s = index_array[0]
        if s[:3]  == "IVF":
            nlist = int(s[3:])
        else:
            raise ValueError
    elif len(index_array) == 3: # "OPQ16,IVF4096,PQ16"
        s = index_array[1]
        if s[:3]  == "IVF":
            nlist = int(s[3:])
        else:
            raise ValueError
    else:
        raise ValueError

    threshold_nlist = nlist 
    if nlist <= 128:
        pass
    elif nlist <= 256:
        threshold_nlist = nlist / 2
    elif nlist <= 512:
        threshold_nlist = nlist / 4
    elif nlist <= 1024:
        threshold_nlist = nlist / 8
    elif nlist > 1024:
        threshold_nlist = nlist / 16

    ps = faiss.GpuParameterSpace()
    ps.initialize(index)

    nq_gt = gt_I.shape[0]
    sl = query_batch_size
    nq = xq.shape[0]

    nprobe = 1 # start nprobe
    min_range = 1
    max_range = None
    fail = False

    while True:
        ps.set_index_parameter(index, 'nprobe', nprobe)
        t0 = time.time()

        if sl == 0:
            D, I = index.search(preproc.apply_py(sanitize(xq)), topK)
        else:
            I = np.empty((nq, topK), dtype='int32')
            D = np.empty((nq, topK), dtype='float32')

            inter_res = ''

            for i0, xs in dataset_iterator(xq, preproc, sl):

                i1 = i0 + xs.shape[0]
                Di, Ii = index.search(xs, topK)

                I[i0:i1] = Ii
                D[i0:i1] = Di

                if knngraph and not inter_res and i1 >= nq_gt:
                    ires = eval_intersection_measure(
                        gt_I[:, :topK], I[:nq_gt])
                    inter_res = ', %.4f' % ires

        t1 = time.time()
        
        print("  probe=%-3d: %.3f s" % (nprobe, t1 - t0), end=' ')
        gtc = gt_I[:, :1]
        nq = xq.shape[0]
        nok = (I[:, :topK] == gtc).sum()
        recall = nok / float(nq)
        print("1-R@%d: %.4f" % (topK, recall), end='\n')

        if recall >= recall_goal:
            max_range = nprobe # max range is used when recall goal is achieved
            nprobe = int((min_range + nprobe) / 2.0)
            if nprobe == min_range:
                break
        else:
            min_range = nprobe  # to achieve target recall, need larger than this nprobe
            if nprobe == threshold_nlist:
                print("ERROR! Search failed: cannot reach expected recall on given dataset and index")
                break
            elif max_range:
                if nprobe  ==  max_range - 1:
                    break
                nprobe = int((max_range + nprobe) / 2.0)
            else:
                nprobe = nprobe * 2
                if nprobe > threshold_nlist:
                    nprobe = threshold_nlist
    
    if not fail:
        min_nprobe = max_range
        print("The minimum nprobe to achieve R@{topK}={recall_goal} on {dbname} {index_key} is {nprobe}".format(
            topK=topK, recall_goal=recall_goal, dbname=dbname, index_key=index_key, nprobe=min_nprobe))

        fname = os.path.abspath(os.path.join(cur_script_dir, './recall_info/gpu_recall_index_nprobe_pairs_{}.pkl'.format(dbname)))
        if os.path.exists(fname) and os.path.getsize(fname) > 0: # load and write
            d = None
            with open(fname, 'rb') as f:
                d = pickle.load(f)

            with open(fname, 'wb') as f:
                # dictionary format:
                #   d[dbname (str)][index_key (str)][topK (int)][recall_goal (float, 0~1)] = nprobe
                #   e.g., d["SIFT100M"]["IVF4096,PQ16"][10][0.7]
                if dbname not in d:
                    d[dbname] = dict()
                if index_key not in d[dbname]:
                    d[dbname][index_key] = dict()
                if topK not in d[dbname][index_key]:
                    d[dbname][index_key][topK] = dict()
                d[dbname][index_key][topK][recall_goal] = min_nprobe
                pickle.dump(d, f, protocol=4)

        else: # write new file
            with open(fname, 'wb') as f:
                # dictionary format:
                #   d[dbname (str)][index_key (str)][topK (int)][recall_goal (float, 0~1)] = nprobe
                #   e.g., d["SIFT100M"]["IVF4096,PQ16"][10][0.7]
                d = dict()
                d[dbname] = dict()
                d[dbname][index_key] = dict()
                d[dbname][index_key][topK] = dict()
                d[dbname][index_key][topK][recall_goal] = min_nprobe
                pickle.dump(d, f, protocol=4)




#################################################################
# Driver
#################################################################



if recall_goal: # usage 3
    preproc = get_preprocessor()
    index = get_populated_index(preproc)
    # test the min nprobe to achieve certain recall
    recall_eval(index, preproc)
    # make sure index is deleted before the resources
    del index
elif load_from_dict:  # usage 2
    eval_dataset_from_dict()
else: # usage 1
    # replicate the 10K queries to 100K to evaluate the more stable version of the GPU performance
    xq = np.tile(xq, (query_num_factor, 1))
    gt_I = np.tile(gt_I, (query_num_factor, 1))
    preproc = get_preprocessor()
    index = get_populated_index(preproc)
    # test throughput, nprobe recall, etc.
    eval_dataset(index, preproc)
    # make sure index is deleted before the resources
    del index
