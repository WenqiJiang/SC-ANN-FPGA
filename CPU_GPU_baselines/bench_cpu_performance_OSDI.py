"""
Benchmarking the trade-off between recall, QPS, and latency using 10,000 queries (it could take hours to run):

It evaluates different combinations of batch size (qbs) and nprobe, then evaluate (a) R1@K, R@K (b) QPS (c) 50%/95% tail latency

To use the script:

e.g., measure the performance of a single server
python bench_cpu_performance_OSDI.py --dbname SIFT1000M --index_key IVF32768,PQ32  --performance_dict_dir './cpu_performance_result/r630_cpu_performance_trade_off.pkl' --record_latency_distribution 0 --overwrite 0

e.g., measure the latency distribution (for distributed search QPS measurement)
python bench_cpu_performance_OSDI.py --dbname SIFT1000M --index_key IVF32768,PQ32  --performance_dict_dir './cpu_performance_result/r630_cpu_performance_latency_distribution.pkl' --record_latency_distribution 1 --overwrite 0

e.g., measure the performance of one server of distributed search 
python bench_cpu_performance_OSDI.py --dbname SBERT3000M --index_key IVF65536,PQ64 --n_shards 4 --shard_id 0 --performance_dict_dir './cpu_performance_result/r630_cpu_performance_latency_distribution_server0.pkl' --record_latency_distribution 1 --overwrite 0


The results are saved as an dictionary which has the following format:
    dict[dbname][index_key][qbs][nprobe] contains several components:
    dict[dbname][index_key][qbs][nprobe]["R1@1"]
    dict[dbname][index_key][qbs][nprobe]["R1@10"]
    dict[dbname][index_key][qbs][nprobe]["R1@100"]
    dict[dbname][index_key][qbs][nprobe]["R@1"]
    dict[dbname][index_key][qbs][nprobe]["R@10"]
    dict[dbname][index_key][qbs][nprobe]["R@100"]
    dict[dbname][index_key][qbs][nprobe]["QPS"]
    dict[dbname][index_key][qbs][nprobe]["latency@50"] in ms
    dict[dbname][index_key][qbs][nprobe]["latency@95"] in ms

    optional (record_latency_distribution == 1): 
    dict[dbname][index_key][qbs][nprobe]["latency_distribution"] -> a list of latency (of batches) in ms

    optional (record_computed_results == 1):
    dict[dbname][index_key][ngpu][qbs][nprobe]["I"] -> idx, shape = np.empty((nq, topK), dtype='int64')
    dict[dbname][index_key][ngpu][qbs][nprobe]["D"] -> dist, shape = np.empty((nq, topK), dtype='float32')
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
from datasets import read_deep_fbin, read_deep_ibin, mmap_bvecs_SBERT, mmap_bvecs_GNN
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SIFT100M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default='IVF4096,PQ16', help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--n_shards', type=int, default=None, help="e.g., can use 2 or 4 shards for large datasets")
parser.add_argument('--shard_id', type=int, default=None, help="shard id, cooperate with n_shards")
parser.add_argument('--overwrite', type=int, default=0, help="whether to overwrite existed performance, by default, skip existed settings")
parser.add_argument('--record_latency_distribution', type=int, default=0, help="whether to measure")
parser.add_argument('--record_computed_results', type=int, default=0, help="whether to measure")
parser.add_argument('--performance_dict_dir', type=str, default='./cpu_performance_result/cpu_throughput_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> throughput (QPS)")

topK = 100
qbs_list = [1, 2, 4, 8, 16, 32, 64]
qbs_list.reverse() # using large batches first since they are faster

### Wenqi: when loading the index, save it to numpy array, default: False
save_numpy_index = False
# save_numpy_index = False 
# we mem-map the biggest files to avoid having them in memory all at
# once

args = parser.parse_args()

dbname = args.dbname
index_key = args.index_key
n_shards = args.n_shards
shard_id = args.shard_id
overwrite = args.overwrite
performance_dict_dir = args.performance_dict_dir

# https://github.com/facebookresearch/faiss/blob/main/faiss/index_io.h
# https://www.programcreek.com/python/example/112290/faiss.write_index
io_flags = 0

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

    if n_shards is not None and shard_id is not None:
        print("n_shards: {}\tshard_id: {}".format(n_shards, shard_id))
        filename = "%s/%s_%s_populated_shard_%s.index" % (
            tmpdir, dbname, index_key, str(shard_id))
    else:
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
        if save_numpy_index:
            print("Saving index to numpy array...")
            chunk = faiss.serialize_index(index)
            np.save("{}.npy".format(filename), chunk)
            print("Finish saving numpy index")
    return index


#################################################################
# Perform searches
#################################################################

"""
The results are saved as an dictionary which has the following format:
    dict[dbname][index_key][qbs][nprobe] contains several components:
    dict[dbname][index_key][qbs][nprobe]["R1@1"]
    dict[dbname][index_key][qbs][nprobe]["R1@10"]
    dict[dbname][index_key][qbs][nprobe]["R1@100"]
    dict[dbname][index_key][qbs][nprobe]["R@1"]
    dict[dbname][index_key][qbs][nprobe]["R@10"]
    dict[dbname][index_key][qbs][nprobe]["R@100"]
    dict[dbname][index_key][qbs][nprobe]["QPS"]
    dict[dbname][index_key][qbs][nprobe]["latency@50"] in ms
    dict[dbname][index_key][qbs][nprobe]["latency@95"] in ms
"""

dict_perf = None
if os.path.exists(args.performance_dict_dir):
    with open(args.performance_dict_dir, 'rb') as f:
        dict_perf = pickle.load(f)
else:
    dict_perf = dict()

if dbname not in dict_perf:
    dict_perf[dbname] = dict()

if index_key not in dict_perf[dbname]:
    dict_perf[dbname][index_key] = dict()


if n_shards is not None and shard_id is not None:
    tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}_{}shards'.format(dbname, index_key, n_shards)
else:
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
    
    nprobe_list = [1, 2, 4, 8, 16, 32, 64, 128]

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

    nprobe_list = [1, 2, 4, 8, 16, 32, 64, 128]

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

    nprobe_list = [1, 2, 4, 8, 16, 32, 64, 128]

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

    nprobe_list = [1, 2, 4, 8, 16, 32]
else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)

nq, d = xq.shape
assert gt.shape[0] == nq


index = get_populated_index()

ps = faiss.ParameterSpace()
ps.initialize(index)

# a static C++ object that collects statistics about searches
ivfpq_stats = faiss.cvar.indexIVFPQ_stats
ivf_stats = faiss.cvar.indexIVF_stats

# we do queries in a single thread
# faiss.omp_set_num_threads(1)

# print(' ' * len('nprobe=32'), '\t', 'R1@1\t R1@10\t R1@100\t R@1\t R@10\t R@100\t QPS\t latency(50%)/ms\t latency(95%)/ms\t')

query_vecs = np.reshape(xq, (nq,1,d))


def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

for qbs in qbs_list:

    print("batch size: ", qbs)
    sys.stdout.flush()
    if qbs not in dict_perf[dbname][index_key]:
        dict_perf[dbname][index_key][qbs] = dict()

    for nprobe in nprobe_list:


        param = 'nprobe={}'.format(nprobe)
        print(param, '\t', end=' ')
        sys.stdout.flush()

        if nprobe not in dict_perf[dbname][index_key][qbs]:
            dict_perf[dbname][index_key][qbs][nprobe] = dict()
        # skip if numbers already exists
        else:
            if not args.overwrite: 
                print("SKIP TEST.\tDB: {}\tindex: {}\tbatch_size: {}\tnprobe: {}\t".format(
                    dbname, index_key, qbs, nprobe))
                continue

        ps.set_index_parameters(index, param)
        
        I = np.empty((nq, topK), dtype='int64')
        D = np.empty((nq, topK), dtype='float32')

        ivfpq_stats.reset()
        ivf_stats.reset()

        t_query_list = [] # in sec

        i0 = 0
        while i0 < nq:
            if i0 + qbs < nq:
                i1 = i0 + qbs
            else:
                i1 = nq
            t_q_start = time.time()
            Di, Ii = index.search(xq[i0:i1], topK)
            t_q_end = time.time()
            t_query_list.append(t_q_end - t_q_start)
            I[i0:i1] = Ii
            D[i0:i1] = Di
            i0 = i1

        n_ok = (I[:, :topK] == gt[:, :1]).sum()
        for rank in 1, 10, 100: # R1@K
            n_ok = (I[:, :rank] == gt[:, :1]).sum()
            R1_at_K = n_ok / float(nq)
            if rank == 1:
                print("R1@1 = %.4f" % (R1_at_K), end='\t')
                dict_perf[dbname][index_key][qbs][nprobe]["R1@1"] = R1_at_K
            elif rank == 10:
                print("R1@10 = %.4f" % (R1_at_K), end='\t')
                dict_perf[dbname][index_key][qbs][nprobe]["R1@10"] = R1_at_K
            elif rank == 100:
                print("R1@100 = %.4f" % (R1_at_K), end='\t')
                dict_perf[dbname][index_key][qbs][nprobe]["R1@100"] = R1_at_K
        for rank in 1, 10, 100: # R@K
            R_at_K = compute_recall(I[:,:rank], gt[:, :rank])
            if rank == 1:
                print("R@1 = %.4f" % R_at_K, end='\t')
                dict_perf[dbname][index_key][qbs][nprobe]["R@1"] = R_at_K
            elif rank == 10:
                print("R@10 = %.4f" % R_at_K, end='\t')
                dict_perf[dbname][index_key][qbs][nprobe]["R@10"] = R_at_K
            elif rank == 100:
                print("R@100 = %.4f" % R_at_K, end='\t')
                dict_perf[dbname][index_key][qbs][nprobe]["R@100"] = R_at_K

        if args.record_latency_distribution: 
            dict_perf[dbname][index_key][qbs][nprobe]["latency_distribution"] = np.array(t_query_list) * 1000

        if args.record_computed_results:
            dict_perf[dbname][index_key][qbs][nprobe]["I"] = I
            dict_perf[dbname][index_key][qbs][nprobe]["D"] = D

        total_time = np.sum(np.array(t_query_list)) 
        QPS = nq / total_time
        print("QPS = {:.4f}".format(QPS), end='\t')
        dict_perf[dbname][index_key][qbs][nprobe]["QPS"] = QPS
        
        sorted_t_query_list = np.sort(np.array(t_query_list))
        latency_50 = sorted_t_query_list[np.amin([len(sorted_t_query_list) - 1, int(np.ceil(len(sorted_t_query_list) * 0.5))])] * 1000
        print("latency(50%)/ms = {:.4f}".format(latency_50), end='\t')
        dict_perf[dbname][index_key][qbs][nprobe]["latency@50"] = latency_50

        latency_95 = sorted_t_query_list[np.amin([len(sorted_t_query_list) - 1, int(np.ceil(len(sorted_t_query_list) * 0.95))])] * 1000
        print("latency(95%)/ms = {:.4f}".format(latency_95))
        dict_perf[dbname][index_key][qbs][nprobe]["latency@95"] = latency_95

        with open(args.performance_dict_dir, 'wb') as f:
            pickle.dump(dict_perf, f, protocol=4)