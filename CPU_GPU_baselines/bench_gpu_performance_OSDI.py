"""
Benchmarking the trade-off between recall, QPS, and latency using 10,000 queries (it could take hours to run):

It evaluates different combinations of batch size (qbs) and nprobe, then evaluate (a) R1@K, R@K (b) QPS (c) 50%/95% tail latency

To use the script:

nrun = number of runs to average the performance (the GPU runtime is not very stable)

e.g., measure the performance of a single server
python bench_gpu_performance_OSDI.py -dbname SIFT1000M -index_key IVF32768,PQ32  -ngpu 4 -nrun 5 -performance_dict_dir './gpu_performance_result/Titan_X_gpu_performance_trade_off.pkl' -record_latency_distribution 0 -overwrite 0
python bench_gpu_performance_OSDI.py -dbname SBERT3000M -index_key IVF65536,PQ64  -ngpu 8 -nrun 5 -performance_dict_dir './gpu_performance_result/V100_32GB_gpu_performance_trade_off.pkl' -record_latency_distribution 0 -overwrite 0
python bench_gpu_performance_OSDI.py -dbname GNN1400M -index_key IVF32768,PQ64  -ngpu 5 -nrun 5 -performance_dict_dir './gpu_performance_result/V100_32GB_gpu_performance_trade_off.pkl' -record_latency_distribution 0 -overwrite 0

e.g., measure the latency distribution
python bench_gpu_performance_OSDI.py -dbname SIFT1000M -index_key IVF32768,PQ32  -ngpu 4 -nrun 5 -performance_dict_dir './gpu_performance_result/Titan_X_gpu_performance_latency_distribution.pkl' -record_latency_distribution 1 -overwrite 0

The results are saved as an dictionary which has the following format:
    dict[dbname][index_key][ngpu][qbs][nprobe] contains several components:
    dict[dbname][index_key][ngpu][qbs][nprobe]["R1@1"]
    dict[dbname][index_key][ngpu][qbs][nprobe]["R1@10"]
    dict[dbname][index_key][ngpu][qbs][nprobe]["R1@100"]
    dict[dbname][index_key][ngpu][qbs][nprobe]["R@1"]
    dict[dbname][index_key][ngpu][qbs][nprobe]["R@10"]
    dict[dbname][index_key][ngpu][qbs][nprobe]["R@100"]
    dict[dbname][index_key][ngpu][qbs][nprobe]["QPS"]
    dict[dbname][index_key][ngpu][qbs][nprobe]["latency@50"] in ms
    dict[dbname][index_key][ngpu][qbs][nprobe]["latency@95"] in ms

    optional (record_latency_distribution == 1): 
    dict[dbname][index_key][ngpu][qbs][nprobe]["latency_distribution"] -> a list of latency (of batches) in ms

"""

from __future__ import print_function
import numpy as np
import time
import os
import sys
import faiss
import pickle
import re
import pickle

from multiprocessing.dummy import Pool as ThreadPool
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
-nprobe 4,16,64    try this number of probesd
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
nrun = 1

replicas = 1  # nb of replicas of sharded dataset
# qbs_list = [1]
qbs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
qbs_list.reverse() # using large batches first since they are faster

# Wenqi edited, origin use_precomputed_tables=True
#use_precomputed_tables = False
use_precomputed_tables = True
tempmem = -1  # if -1, use system default
max_add = -1
use_float16 = False
use_cache = True
startgpu=0
topK = 100
altadd = False
I_fname = None
D_fname = None
recall_goal = None

load_from_dict = None 
overwrite = 0
nprobe_dict_dir = None
throughput_dict_dir = None
response_time_dict_dir = None

performance_dict_dir=None 
record_latency_distribution=None 
overwrite=None

args = sys.argv[1:]

while args:
    a = args.pop(0)
    if a == '-h': usage()
    elif a == '-ngpu':      ngpu = int(args.pop(0))
    elif a == '-nrun':      nrun = int(args.pop(0))
    elif a == '-startgpu':  startgpu = int(args.pop(0)) # Wenqi, the id the first GPU used, e.g., 1 -> skip GPU0
    elif a == '-R':         replicas = int(args.pop(0))
    elif a == '-noptables': use_precomputed_tables = False
    elif a == '-qbs':       qbs = int(args.pop(0))
    elif a == '-topK':       topK = int(args.pop(0))
    elif a == '-tempmem':   tempmem = int(args.pop(0))
    elif a == '-nocache':   use_cache = False
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
    elif a == '-performance_dict_dir': performance_dict_dir = str(args.pop(0))
    elif a == '-record_latency_distribution': record_latency_distribution = int(args.pop(0))
    elif a == '-overwrite': overwrite = int(args.pop(0))
    else:
        print("argument %s unknown" % a, file=sys.stderr)
        sys.exit(1)


dict_perf = None
if os.path.exists(performance_dict_dir):
    with open(performance_dict_dir, 'rb') as f:
        dict_perf = pickle.load(f)
else:
    dict_perf = dict()

if dbname not in dict_perf:
    dict_perf[dbname] = dict()

if index_key not in dict_perf[dbname]:
    dict_perf[dbname][index_key] = dict()

if ngpu not in dict_perf[dbname][index_key]:
    dict_perf[dbname][index_key][ngpu] = dict()

#################################################################
# Small Utility Functions
#################################################################


def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

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


#################################################################
# Prepare dataset
#################################################################

cacheroot = None
dbsize = None
xb = None
xq = None
xt = None
gt = None
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

if dbname: 
    print("Preparing dataset", dbname)

    cacheroot = os.path.abspath(os.path.join(cur_script_dir, './trained_CPU_indexes/bench_cpu_{}_{}'.format(dbname, index_key)))
    print("Using CPU trained indexes...")
    #cacheroot = os.path.relpath('./trained_GPU_indexes/bench_gpu_{}_{}'.format(dbname, index_key))

    if not os.path.isdir(cacheroot):
        print("%s does not exist, creating it" % cacheroot)
        os.mkdir(cacheroot)

    if dbname.startswith('SIFT'):
        # SIFT1M to SIFT1000M
        dbsize = int(dbname[4:-1])
        # xb = mmap_bvecs(os.path.abspath(os.path.join(cur_script_dir, 'bigann/bigann_base.bvecs')))
        xq = mmap_bvecs(os.path.abspath(os.path.join(cur_script_dir, 'bigann/bigann_query.bvecs')))

        # Wenqi adjusted, only use the first 1000 queries
        # print(xq.shape)
        # xq = xq[:1000]
        print(xq.shape)

        # xt = mmap_bvecs(os.path.abspath(os.path.join(cur_script_dir, 'bigann/bigann_learn.bvecs')))

        # trim xb to correct size
        # xb = xb[:dbsize * 1000 * 1000]

        gt = ivecs_read(os.path.abspath(os.path.join(cur_script_dir, 'bigann/gnd/idx_%dM.ivecs' % dbsize)))

        nprobes = [1, 2, 4, 8, 16, 32, 64, 128] 

    elif dbname.startswith('Deep'):

        assert dbname[:4] == 'Deep' 
        assert dbname[-1] == 'M'
        dbsize = int(dbname[4:-1]) # in million
        # xb = read_deep_fbin('deep1b/base.1B.fbin')[:dbsize * 1000 * 1000]
        xq = read_deep_fbin('deep1b/query.public.10K.fbin')
        # xt = read_deep_fbin('deep1b/learn.350M.fbin')

        gt = read_deep_ibin('deep1b/gt_idx_{}M.ibin'.format(dbsize))

        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)

        nprobes = [1, 2, 4, 8, 16, 32, 64, 128] 
        
    elif dbname.startswith('SBERT'):
        # FB1M to FB1000M
        dataset_dir = './sbert'
        assert dbname[:5] == 'SBERT' 
        assert dbname[-1] == 'M'
        dbsize = int(dbname[5:-1]) # in million
        # xb = mmap_bvecs_SBERT('sbert/sbert3B.fvecs', num_vec=int(dbsize * 1e6))
        xq = mmap_bvecs_SBERT('sbert/query_10K.fvecs', num_vec=10 * 1000)
        # xt = xb

        # trim to correct size
        # xb = xb[:dbsize * 1000 * 1000]
        
        gt = read_deep_ibin('sbert/gt_idx_{}M.ibin'.format(dbsize), dtype='uint32')

        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)

        query_num = xq.shape[0]
        print('query shape: ', xq.shape)

        nprobes = [1, 2, 4, 8, 16, 32, 64] # 1 to 64, nprobe=128 for SBERT3000M -> cudaMalloc Fail
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
        dataset_dir = './MariusGNN/'
        assert dbname[:3] == 'GNN' 
        assert dbname[-1] == 'M'
        dbsize = int(dbname[3:-1]) # in million
        # xb = mmap_bvecs_GNN('MariusGNN/embeddings.bin', num_vec=int(dbsize * 1e6))
        xq = mmap_bvecs_GNN('MariusGNN/query_10K.fvecs', num_vec=10 * 1000)
        # xt = xb

        # trim to correct size
        # xb = xb[:dbsize * 1000 * 1000]
    
        gt = read_deep_ibin('MariusGNN/gt_idx_{}M.ibin'.format(dbsize), dtype='uint32') 
        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)
        # The dataset is highly skewed (imbalance factor > 30), only search a subset to speedup the test
        num_query_for_eval = 1000
        xq = xq[:num_query_for_eval]
        gt = gt[:num_query_for_eval]

        query_num = xq.shape[0]
        print('query shape: ', xq.shape)

        nprobes = [1, 2, 4, 8, 16, 32] # > 32 no recall improvement

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


    # print("sizes: B %s Q %s T %s gt %s" % (
    #     xb.shape, xq.shape, xt.shape,
    #     gt.shape if gt is not None else None))



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


def get_populated_index(preproc):

    if not index_cachefile or not os.path.exists(index_cachefile):
        raise ValueError
    else:
        print("load", index_cachefile)
        indexall = faiss.read_index(index_cachefile)
        gpu_index = None

    print("Setting differnt GPUs holding different shards by faiss.GpuMultipleClonerOptions() shard=True")
    co = faiss.GpuMultipleClonerOptions()
    co.shard=True

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

    nq_gt = gt.shape[0]
    print("search...")

    nq = xq.shape[0]
    print(nq)

    for qbs in qbs_list:
        
        print("batch size: ", qbs)
        sys.stdout.flush()
        if qbs not in dict_perf[dbname][index_key][ngpu]:
            dict_perf[dbname][index_key][ngpu][qbs] = dict()

        for nprobe in nprobes:

            ps.set_index_parameter(index, 'nprobe', nprobe)
            print("nprobe: ", nprobe)

            if nprobe not in dict_perf[dbname][index_key][ngpu][qbs] or overwrite:
                dict_perf[dbname][index_key][ngpu][qbs][nprobe] = dict()
            else:
                continue

            t_query_list = [] # in sec, for all runs (e.g., 5 runs per nprobe)
            
            I = np.empty((nq, topK), dtype='int64')
            D = np.empty((nq, topK), dtype='float32')

            for run_iter in range(nrun):

                for i0, xs in dataset_iterator(xq, preproc, qbs):
                    # print('\r%d/%d (%.3f s%s)   ' % (
                    #     i0, nq, time.time() - t0, inter_res), end=' ')
                    # sys.stdout.flush()

                    i1 = i0 + xs.shape[0]
                    # Wenqi: debugging memory overflow
                    # print(xs.shape)
                    t_q_start = time.time()
                    Di, Ii = index.search(xs, topK)
                    t_q_end = time.time()
                    t_query_list.append(t_q_end - t_q_start)

                    I[i0:i1] = Ii
                    D[i0:i1] = Di
                    
            n_ok = (I[:, :topK] == gt[:, :1]).sum()
            for rank in 1, 10, 100: # R1@K
                n_ok = (I[:, :rank] == gt[:, :1]).sum()
                R1_at_K = n_ok / float(nq)
                if rank == 1:
                    print("R1@1 = %.4f" % (R1_at_K), end='\t')
                    dict_perf[dbname][index_key][ngpu][qbs][nprobe]["R1@1"] = R1_at_K
                elif rank == 10:
                    print("R1@10 = %.4f" % (R1_at_K), end='\t')
                    dict_perf[dbname][index_key][ngpu][qbs][nprobe]["R1@10"] = R1_at_K
                elif rank == 100:
                    print("R1@100 = %.4f" % (R1_at_K), end='\t')
                    dict_perf[dbname][index_key][ngpu][qbs][nprobe]["R1@100"] = R1_at_K
            for rank in 1, 10, 100: # R@K
                R_at_K = compute_recall(I[:,:rank], gt[:, :rank])
                if rank == 1:
                    print("R@1 = %.4f" % R_at_K, end='\t')
                    dict_perf[dbname][index_key][ngpu][qbs][nprobe]["R@1"] = R_at_K
                elif rank == 10:
                    print("R@10 = %.4f" % R_at_K, end='\t')
                    dict_perf[dbname][index_key][ngpu][qbs][nprobe]["R@10"] = R_at_K
                elif rank == 100:
                    print("R@100 = %.4f" % R_at_K, end='\t')
                    dict_perf[dbname][index_key][ngpu][qbs][nprobe]["R@100"] = R_at_K

            if record_latency_distribution: 
                dict_perf[dbname][index_key][ngpu][qbs][nprobe]["latency_distribution"] = np.array(t_query_list) * 1000

            total_time = np.sum(np.array(t_query_list)) 
            QPS = nrun * nq / total_time
            print("QPS = {:.4f}".format(QPS), end='\t')
            dict_perf[dbname][index_key][ngpu][qbs][nprobe]["QPS"] = QPS
            
            sorted_t_query_list = np.sort(np.array(t_query_list))
            latency_50 = sorted_t_query_list[np.amin([len(sorted_t_query_list) - 1, int(np.ceil(len(sorted_t_query_list) * 0.5))])] * 1000
            print("latency(50%)/ms = {:.4f}".format(latency_50), end='\t')
            dict_perf[dbname][index_key][ngpu][qbs][nprobe]["latency@50"] = latency_50

            latency_95 = sorted_t_query_list[np.amin([len(sorted_t_query_list) - 1, int(np.ceil(len(sorted_t_query_list) * 0.95))])] * 1000
            print("latency(95%)/ms = {:.4f}".format(latency_95))
            dict_perf[dbname][index_key][ngpu][qbs][nprobe]["latency@95"] = latency_95

            with open(performance_dict_dir, 'wb') as f:
                pickle.dump(dict_perf, f, protocol=4)

#################################################################
# Driver
#################################################################


# replicate the 10K queries to 100K to evaluate the more stable version of the GPU performance
xq = np.tile(xq, (query_num_factor, 1))
gt = np.tile(gt, (query_num_factor, 1))
preproc = get_preprocessor()
index = get_populated_index(preproc)
# test throughput, nprobe recall, etc.
eval_dataset(index, preproc)
# make sure index is deleted before the resources
del index
