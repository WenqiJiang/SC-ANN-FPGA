"""
Usage:

For a single FPGA/CPU server:
    python train_SYN_dataset.py --dbname SYN1M --index_key IVF1024,PQ16 --topK 100 --qbs 1 --parametersets 'nprobe=1 nprobe=32'

For multiple FPGA/CPU servers:
    python train_SYN_dataset.py --dbname SYN1M --index_key IVF1024,PQ16 --topK 100 --qbs 1 --total_index_num 4 --index_ID 0 --parametersets 'nprobe=1 nprobe=32'
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import faiss
import dask.array as da
import pickle
from multiprocessing.dummy import Pool as ThreadPool
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SYN1M', help="dataset name, e.g., SYN1000M")
parser.add_argument('--index_key', type=str, default='IVF4096,PQ16', help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--qbs', type=int, default=1, help="query batch size")
parser.add_argument('--total_index_num', type=int, help="For large indexes in a distributed seach scenario, the total number of server == number of index")
parser.add_argument('--index_ID', type=int, help="For large indexes in a distributed seach scenario, the ID of server == ID of index")
parser.add_argument('--parametersets', type=str, default='nprobe=1', help="a string of nprobes, e.g., 'nprobe=1 nprobe=32'")


args = parser.parse_args()

def SYN_mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = 128
    return x.reshape(-1, d)

def SYN_ivecs_read(fname):
    a = np.fromfile(fname, dtype='int64')
    d = 1000 # topK=1000 results
    return a.reshape(-1, d).copy()



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


def get_trained_index(load_only=False):
    filename = "%s/%s_%s_trained.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        if load_only:
            print("Set load vector quantizer only, but the quantizer is not trained")
            raise ValueError
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
        index = faiss.read_index(filename)
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
        lambda i01: np.array(x[i01[0]:i01[1]]).astype('float32').copy(),
        block_ranges)


def get_populated_index():

    filename = "%s/%s_%s_populated.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = get_trained_index()
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
        index = faiss.read_index(filename)
    return index


def get_populated_index_multi_server(total_index_num, index_ID, xb_part, start_vec_ID):

    filename = "%s/%s_%s_populated_%s_of_%s.index" % (
        tmpdir, dbname, index_key, index_ID, total_index_num)


    if not os.path.exists(filename):
        if index_ID == 0: 
            index = get_trained_index(load_only=False)
        else:
            print("Warning: when doing multi-partition training, the vector quantizer"
             " must be trained using the first partition before other indexes are constructed")
            index = get_trained_index(load_only=True)
        i0 = start_vec_ID
        t0 = time.time()
        for xs in matrix_slice_iterator(xb_part, 100000):
            i1 = i0 + xs.shape[0]
            print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
            sys.stdout.flush()
            # index.add(xs)
            index.add_with_ids(xs, np.arange(i0, i1))
            i0 = i1
        print()
        print("Add done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index

#################################################################
# Perform searches
#################################################################

dbname = args.dbname
index_key = args.index_key
topK = args.topK
parametersets = args.parametersets.split() # split nprobe argument string by space

if args.total_index_num is None:    
    tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}'.format(dbname, index_key)
else:
    tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}_{}_servers'.format(dbname, index_key, args.total_index_num)

if not os.path.isdir(tmpdir):
    print("%s does not exist, creating it" % tmpdir)
    os.mkdir(tmpdir)


print("Preparing dataset", dbname)

if dbname.startswith('SYN'):
    dbsize = int(dbname[3:-1])

    if dbsize < 100:
        dataset_dir = './SYN_dataset/SYN{}M'.format(dbsize)
        xb = SYN_mmap_bvecs(os.path.join(dataset_dir, 'base.bvecs'))
        xt = SYN_mmap_bvecs(os.path.join(dataset_dir, 'learn.bvecs'))
        xq = SYN_mmap_bvecs(os.path.join(dataset_dir, 'query.bvecs')).astype('float32')
        gt = SYN_ivecs_read(os.path.join(dataset_dir, 'idx_10000_by_1000_int64.lvecs'))
    else:
        # must be the multiple of 100M
        assert dbsize % 100 == 0
        dataset_dir = 'SYN_dataset/SYN{}M'.format(dbsize)
        db_batches = int(dbsize / 100)
        xb_partitions = []
        for batch_id in range(db_batches):
            base_vec_dir = os.path.join(dataset_dir, "base_{}_of_{}.bvecs".format(batch_id, db_batches))
            xb_partitions.append(SYN_mmap_bvecs(base_vec_dir))

        # dask.array object, should be converted to numpy array when used
        xb = da.concatenate(xb_partitions, axis=0) 

        xt = SYN_mmap_bvecs(os.path.join(dataset_dir, 'learn.bvecs'))
        xq = SYN_mmap_bvecs(os.path.join(dataset_dir, 'query.bvecs')).astype('float32')
        gt = SYN_ivecs_read(os.path.join(dataset_dir, 'idx_10000_by_1000_int64.lvecs'))

    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)
    gt = np.array(gt, dtype=np.int64)
else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)

nq, d = xq.shape
assert gt.shape[0] == nq


if args.total_index_num is None: # single server setting
    index = get_populated_index()
else: # multiple server setting
    if args.index_ID is None:
        print("index_ID is needed for multi index training")
        raise ValueError
    assert  xb.shape[0] % args.total_index_num == 0
    partition_size = int(xb.shape[0] / args.total_index_num)
    start_vec_ID = partition_size * args.index_ID
    xb_part = xb[partition_size * args.index_ID: partition_size * (1 + args.index_ID)]
    index = get_populated_index_multi_server(args.total_index_num, args.index_ID, xb_part, start_vec_ID)

ps = faiss.ParameterSpace()
ps.initialize(index)

# a static C++ object that collects statistics about searches
ivfpq_stats = faiss.cvar.indexIVFPQ_stats
ivf_stats = faiss.cvar.indexIVF_stats

# we do queries in a single thread
# faiss.omp_set_num_threads(1)

print(' ' * len(parametersets[0]), '\t', 'R@{}     time'.format(topK))

query_vecs = np.reshape(xq, (nq,1,128))

for param in parametersets:
    print(param, '\t', end=' ')
    sys.stdout.flush()
    ps.set_index_parameters(index, param)
    

    I = np.empty((nq, topK), dtype='int32')
    D = np.empty((nq, topK), dtype='float32')

    ivfpq_stats.reset()
    ivf_stats.reset()

    t0 = time.time()

    i0 = 0
    while i0 < nq:
        if i0 + args.qbs < nq:
            i1 = i0 + args.qbs
        else:
            i1 = nq
        Di, Ii = index.search(xq[i0:i1], topK)
        I[i0:i1] = Ii
        D[i0:i1] = Di
        i0 = i1

    t1 = time.time()

    n_ok = (I[:, :topK] == gt[:, :1]).sum()
    for rank in 1, 10, 100:
        n_ok = (I[:, :rank] == gt[:, :1]).sum()
        print("%.4f" % (n_ok / float(nq)), end=' ')
    print("QPS = {}".format(nq / (t1 - t0)))
    #print("%8.3f  " % ((t1 - t0) * 1000.0 / nq), end=' ms')
    # print("%5.2f" % (ivfpq_stats.n_hamming_pass * 100.0 / ivf_stats.ndis))
