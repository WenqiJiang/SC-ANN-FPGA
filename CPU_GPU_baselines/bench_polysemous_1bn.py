"""
Example usage:
    <on_disk> -> 0 = in memory; 1 = on disk
    <dbnam> e.g., SIFT100M
    <index_key> e.g., IVF4096,PQ16
    <parametersets>, e.g., 'nprobe=1 nprobe=32
    python bench_polysemous_1bn.py --on_disk 0 --dbname SIFT100M --index_key IVF4096,PQ16 --parametersets 'nprobe=1 nprobe=32'
    python bench_polysemous_1bn.py --on_disk 0 --dbname SIFT1M --index_key IVF4096,Flat --parametersets 'nprobe=1 nprobe=32'
For large dataset that needs multiple servers / FPGAs for the search, using the shard option (need to add populated index shard by shard), e.g.:
    python bench_polysemous_1bn.py --on_disk 0 --dbname GNN1400M --index_key IVF32768,PQ64 --n_shards 2 --shard_id 1 --parametersets 'nprobe=1 nprobe=32'
    python bench_polysemous_1bn.py --on_disk 0 --dbname SBERT3000M --index_key IVF65536,PQ64 --n_shards 4 --shard_id 3 --parametersets 'nprobe=1 nprobe=32'

Note! Use on_disk = 1 only when
    (1) the trained index is built in memory (not by ondisk merge)
    (2) you want to use mmap for the index during the search instead of loading in memory
    For a disk-merged index, never use the on_disk=1. For those indexes,
    populated index & the ivflist are separate (the later always use mmap),
    in this case, using io_flags=1 to populated_index will result in segfault during search
More details:
    https://github.com/facebookresearch/faiss/blob/main/contrib/ondisk.py
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import faiss
from multiprocessing.dummy import Pool as ThreadPool
from datasets import ivecs_read
from datasets import read_deep_fbin, read_deep_ibin, mmap_bvecs_FB, \
    mmap_bvecs_SBERT, mmap_bvecs_GNN, mmap_bvecs_Journal

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--on_disk', type=int, default=0, help="0 -> search in memory; 1 -> search on disk based on mmap")
parser.add_argument('--dbname', type=str, default='SIFT100M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default='IVF4096,PQ16', help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--n_shards', type=int, default=None, help="e.g., can use 2 or 4 shards for large datasets")
parser.add_argument('--shard_id', type=int, default=None, help="shard id, cooperate with n_shards")
parser.add_argument('--parametersets', type=str, default='nprobe=1', help="a string of nprobes, e.g., 'nprobe=1 nprobe=32'")

args = parser.parse_args()
on_disk = args.on_disk
dbname = args.dbname
index_key = args.index_key
n_shards = args.n_shards
shard_id = args.shard_id
parametersets = args.parametersets.split() # split nprobe argument string by space

### Wenqi: when loading the index, save it to numpy array, default: False
save_numpy_index = False
# save_numpy_index = False 
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


if not on_disk:
    io_flags = 0
else:
    io_flags = faiss.IO_FLAG_MMAP
print("io_flags: ", io_flags)


if n_shards is not None and shard_id is not None:
    tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}_{}shards'.format(dbname, index_key, n_shards)
else:
    tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}'.format(dbname, index_key)

if not os.path.isdir(tmpdir):
    print("%s does not exist, creating it" % tmpdir)
    os.mkdir(tmpdir)


#################################################################
# Prepare dataset
#################################################################


print("Preparing dataset", dbname)

if dbname.startswith('SIFT'):
    # SIFT1M to SIFT1000M
    dbsize = int(dbname[4:-1])
    xb = mmap_bvecs('bigann/bigann_base.bvecs')
    xq = mmap_bvecs('bigann/bigann_query.bvecs')
    xt = mmap_bvecs('bigann/bigann_learn.bvecs')

    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]

    gt = ivecs_read('bigann/gnd/idx_%dM.ivecs' % dbsize)

elif dbname.startswith('Deep'):

    assert dbname[:4] == 'Deep' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[4:-1]) # in million
    xb = read_deep_fbin('deep1b/base.1B.fbin')[:dbsize * 1000 * 1000]
    xq = read_deep_fbin('deep1b/query.public.10K.fbin')
    xt = read_deep_fbin('deep1b/learn.350M.fbin')

    gt = read_deep_ibin('deep1b/gt_idx_{}M.ibin'.format(dbsize))

    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

elif dbname.startswith('FB'):

    assert dbname[:2] == 'FB' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[2:-1]) # in million
    xb = mmap_bvecs_FB('Facebook_SimSearchNet++/FB_ssnpp_database.u8bin', num_vec=dbsize * 1000 * 1000)
    xq = mmap_bvecs_FB('Facebook_SimSearchNet++/FB_ssnpp_public_queries.u8bin', num_vec=10 * 1000)
    xt = xb

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]

    gt = read_deep_ibin('Facebook_SimSearchNet++/gt_idx_{}M.ibin'.format(dbsize), dtype='int32')

    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

elif dbname.startswith('SBERT'):
    # FB1M to FB1000M
    dataset_dir = './sbert'
    assert dbname[:5] == 'SBERT' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[5:-1]) # in million
    xb = mmap_bvecs_SBERT('sbert/sbert3B.fvecs', num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_SBERT('sbert/query_10K.fvecs', num_vec=10 * 1000)
    xt = xb

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
    
    gt = read_deep_ibin('sbert/gt_idx_{}M.ibin'.format(dbsize), dtype='uint32')

    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    query_num = xq.shape[0]
    print('query shape: ', xq.shape)

elif dbname.startswith('GNN'):
    # FB1M to FB1000M
    dataset_dir = './MariusGNN/'
    assert dbname[:3] == 'GNN' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[3:-1]) # in million
    xb = mmap_bvecs_GNN('MariusGNN/embeddings.bin', num_vec=int(dbsize * 1e6))
    
    xq = mmap_bvecs_GNN('MariusGNN/query_10K.fvecs', num_vec=10 * 1000)
    xt = xb

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
   
    gt = read_deep_ibin('MariusGNN/gt_idx_{}M.ibin'.format(dbsize), dtype='uint32') 
    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    nb, D = xb.shape # same as SIFT
    query_num = xq.shape[0]
    print('query shape: ', xq.shape)

elif dbname.startswith('Journal'): # Roger's 4M sample
    # FB1M to FB1000M
    dataset_dir = './Journal/'
    assert dbname[:7] == 'Journal' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[7:-1]) # in million
    xb = mmap_bvecs_Journal('Journal/livejournal_embeddings.bin', num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_Journal('Journal/query_10K.fvecs', num_vec=10 * 1000)
    xt = xb

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
    
    gt = read_deep_ibin('Journal/gt_idx_{}M.ibin'.format(dbsize), dtype='uint32') 
    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    nb, D = xb.shape # same as SIFT
    query_num = xq.shape[0]
    print('query shape: ', xq.shape)

else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)


if n_shards is not None and shard_id is not None:
    assert int(dbsize * 1e6) % n_shards == 0
    size_per_shard = int(int(dbsize * 1e6) / n_shards)
    xb = xb[shard_id * size_per_shard: (shard_id + 1) * size_per_shard]

print("sizes: B %s Q %s T %s gt %s" % (
    xb.shape, xq.shape, xt.shape, gt.shape))

nq, d = xq.shape
nb, d = xb.shape
assert gt.shape[0] == nq
print("nb: {}\tnq: {}\td: {}".format(nb, nq, d))


#################################################################
# Training
#################################################################


def choose_train_size(index_key):

    # some training vectors for PQ and the PCA
    n_train = 256 * 1000

    if "IVF" in index_key:
        matches = re.findall('IVF([0-9]+)', index_key)
        ncentroids = int(matches[0])
        n_train = max(n_train, 100 * ncentroids)
    elif "IMI" in index_key:
        matches = re.findall('IMI2x([0-9]+)', index_key)
        nbit = int(matches[0])
        n_train = max(n_train, 256 * (1 << nbit))
    return n_train


def get_trained_index():
    filename = "%s/%s_%s_trained.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = faiss.index_factory(d, index_key)

        n_train = choose_train_size(index_key)

        xtsub = xt[:n_train]
        print("Keeping %d train vectors" % xtsub.shape[0])
        # make sure the data is actually in RAM and in float
        xtsub = xtsub.astype('float32').copy()
        index.verbose = True

        t0 = time.time()
        index.train(xtsub)
        index.verbose = False
        print("train done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename, io_flags)
    return index


#################################################################
# Adding vectors to dataset
#################################################################

def rate_limited_imap(f, l):
    'a thread pre-processes the next element'
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


def matrix_slice_iterator(x, bs):
    " iterate over the lines of x in blocks of size bs"
    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    return rate_limited_imap(
        lambda i01: x[i01[0]:i01[1]].astype('float32').copy(),
        block_ranges)


def get_populated_index():

    if n_shards is not None and shard_id is not None:
        print("n_shards: {}\tshard_id".format(n_shards, shard_id))
        filename = "%s/%s_%s_populated_shard_%s.index" % (
            tmpdir, dbname, index_key, str(shard_id))
    else:
        filename = "%s/%s_%s_populated.index" % (
            tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = get_trained_index()
        if n_shards is not None and shard_id is not None:
            i0 = size_per_shard * shard_id
        else:
            i0 = 0
        t0 = time.time()
        for xs in matrix_slice_iterator(xb, 100000):
            i1 = i0 + xs.shape[0]
            print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
            sys.stdout.flush()
            index.add_with_ids(xs, np.arange(i0, i1))
            i0 = i1
        print()
        print("Add done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename, io_flags)
        if save_numpy_index:
            print("Saving index to numpy array...")
            chunk = faiss.serialize_index(index)
            np.save("{}.npy".format(filename), chunk)
            print("Finish saving numpy index")
    return index


#################################################################
# Perform searches
#################################################################

index = get_populated_index()
if "IVF" in index_key:
    print("IVF imbalanced factor: ", index.invlists.imbalance_factor())
    
ps = faiss.ParameterSpace()
ps.initialize(index)

# make sure queries are in RAM
xq = xq.astype('float32').copy()

# a static C++ object that collects statistics about searches
ivfpq_stats = faiss.cvar.indexIVFPQ_stats
ivf_stats = faiss.cvar.indexIVF_stats


if parametersets == ['autotune'] or parametersets == ['autotuneMT']:

    if parametersets == ['autotune']:
        faiss.omp_set_num_threads(1)

    # setup the Criterion object: optimize for 1-R@1
    crit = faiss.OneRecallAtRCriterion(nq, 1)
    # by default, the criterion will request only 1 NN
    crit.nnn = 100
    crit.set_groundtruth(None, gt.astype('int64'))

    # then we let Faiss find the optimal parameters by itself
    print("exploring operating points")

    t0 = time.time()
    op = ps.explore(index, xq, crit)
    print("Done in %.3f s, available OPs:" % (time.time() - t0))

    # opv is a C++ vector, so it cannot be accessed like a Python array
    opv = op.optimal_pts
    print("%-40s  1-R@1     time" % "Parameters")
    for i in range(opv.size()):
        opt = opv.at(i)
        print("%-40s  %.4f  %7.3f" % (opt.key, opt.perf, opt.t))

else:

    # we do queries in a single thread
    # faiss.omp_set_num_threads(1)

    print(' ' * len(parametersets[0]), '\t', 'R@1    R@10   R@100     time    %pass')

    for param in parametersets:
        print(param, '\t', end=' ')
        sys.stdout.flush()
        ps.set_index_parameters(index, param)
        t0 = time.time()
        ivfpq_stats.reset()
        ivf_stats.reset()
        D, I = index.search(xq, 100)
        t1 = time.time()
        for rank in 1, 10, 100:
            n_ok = (I[:, :rank] == gt[:, :1]).sum()
            print("%.4f" % (n_ok / float(nq)), end=' ')
        print("%8.3f  " % ((t1 - t0) * 1000.0 / nq), end=' ')
        print("%5.2f" % (ivfpq_stats.n_hamming_pass * 100.0 / ivf_stats.ndis))
