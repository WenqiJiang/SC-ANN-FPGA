"""
Benchmarking the CPU's throughput using 10,000 queries (takes several minutes),

python bench_on_disk_performance.py --dbname SIFT100M --index_key IVF4096,Flat --topK 100 --qbs 100 --query_num 100 --nprobe 1 
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import faiss
import pickle
from multiprocessing.dummy import Pool as ThreadPool
from datasets import ivecs_read
import argparse 
parser = argparse.ArgumentParser()
#parser.add_argument('--on_disk', type=int, default=0, help="0 -> search in memory; 1 -> search on disk based on mmap")
parser.add_argument('--dbname', type=str, default='SIFT100M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default='IVF4096,Flat', help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--nprobe', type=int, default='1', help="a single nprobe value")
parser.add_argument('--qbs', type=int, default=1, help="query batch size")
parser.add_argument('--query_num', type=int, default=100, help="query batch size")

print("Info: you are using the on_disk search script, several thing to keep in mind\n" \
	"1. The input index should be trained by build_index_on_disk.py, the index folder contains 3 rather than 2 files, e.g., merged_index.ivfdata  SIFT100M_IVF4096,Flat_populated.index  SIFT100M_IVF4096,Flat_trained.index\n" \
	"2. don't use too many query number, otherwise differnt query may overlap the scanned region, thus they are simply accessing the memory. We force num_query * nprobe < nlist\n" \
	"3. After running the script, make sure that you flush the page cache, otherwise the OS may just cache the content in memory, and the next run is actually in-memory performance\n" \
	"    page cache flush: (1) sudo su (2) sync; echo 1 > /proc/sys/vm/drop_caches (3) free (see whether flushed)\n" \
	"4. The recall only reflects the <query_num> query, for the precise recall please use the full 10000 queries")

args = parser.parse_args()
dbname = args.dbname
index_key = args.index_key
assert index_key[:len("IVF")] == "IVF" and index_key[-len(",Flat"):] == ",Flat", "Index not supported"
nlist = int(index_key[len("IVF"):-len(",Flat")])
topK = args.topK
nprobe = args.nprobe
qbs = args.qbs
query_num = args.query_num
if query_num < qbs:
    qbs = query_num
assert query_num * nprobe <= nlist, \
    "too many querys and/or nprobe, the disk content could be reused and the performance won't be accurate"
print("index: {} nlist: {} nprobe: {} query_num: {} qbs: {} topK: {}".format(
    index_key, nlist, nprobe, query_num, qbs, topK))


# https://github.com/facebookresearch/faiss/blob/main/faiss/index_io.h
# https://www.programcreek.com/python/example/112290/faiss.write_index
io_flags = 0
#if not args.on_disk:
#    io_flags = 0
#else:
#    io_flags = faiss.IO_FLAG_MMAP

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


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
            index.add(xs)
            i0 = i1
        print()
        print("Add done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename, io_flags)
    return index


#################################################################
# Perform searches
#################################################################


tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}'.format(dbname, index_key)

if not os.path.isdir(tmpdir):
    print("%s does not exist, creating it" % tmpdir)
    os.mkdir(tmpdir)


print("Preparing dataset", dbname)

if dbname.startswith('SIFT'):
    # SIFT1M to SIFT1000M
    dbsize = int(dbname[4:-1])
    xq = mmap_bvecs('bigann/bigann_query.bvecs')

    gt = ivecs_read('bigann/gnd/idx_%dM.ivecs' % dbsize)

    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)
    gt = np.array(gt, dtype=np.int32)

else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)

nq, d = xq.shape
assert gt.shape[0] == nq
query_vecs = np.reshape(xq, (nq,1,128))
query_vecs = query_vecs[:query_num]
print("query_vector_shape", query_vecs.shape)
nq = query_num
gt = gt[:nq]

index = get_populated_index()

ps = faiss.ParameterSpace()
ps.initialize(index)

# a static C++ object that collects statistics about searches
ivfpq_stats = faiss.cvar.indexIVFPQ_stats
ivf_stats = faiss.cvar.indexIVF_stats

# we do queries in a single thread
# faiss.omp_set_num_threads(1)

param = "nprobe={}".format(nprobe)
print(' ' * len(param), '\t', 'R@{}     time'.format(topK))

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
    if i0 + qbs < nq:
        i1 = i0 + qbs
    else:
        i1 = nq
    Di, Ii = index.search(xq[i0:i1], topK)
    I[i0:i1] = Ii
    D[i0:i1] = Di
    i0 = i1

t1 = time.time()

n_ok = (I[:, :topK] == gt[:, :1]).sum()
print("The recall only reflects the <query_num> query, for the precise recall please use the full 10000 queries")
for rank in 1, 10, 100:
    n_ok = (I[:, :rank] == gt[:, :1]).sum()
    print("%.4f" % (n_ok / float(nq)), end=' ')
QPS = nq / (t1 - t0)
print("QPS = {}".format(QPS))
DB_size = int(dbname[len('SIFT'):-len('M')])
print("Disk throughput (bytes/sec) = QPS * nprobe / nlist * xxx Million * 512")
print("QPS ({}) * nprobe ({}) / nlist ({}) * ({} * 1e6) * 512".format(QPS, nprobe, nlist, DB_size))
bytes_per_sec = QPS * nprobe / nlist * DB_size * 1e6 * 512
print("= {} bytes/sec = {} GB/sec".format(bytes_per_sec, bytes_per_sec / 1e9))
#print("%8.3f  " % ((t1 - t0) * 1000.0 / nq), end=' ms')
# print("%5.2f" % (ivfpq_stats.n_hamming_pass * 100.0 / ivf_stats.ndis))

