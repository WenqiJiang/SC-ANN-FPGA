"""
Benchmarking the CPU's throughput using 10,000 queries (takes several minutes),

There are 2 ways to use the script:

(1) Test the throughput of given DB & index & nprobe:

python bench_cpu_performance.py --on_disk 0 --dbname SIFT100M --index_key IVF4096,PQ16 --topK 10 --qbs 1 --parametersets 'nprobe=1 nprobe=32'

optional: --nthreads 1

(2) Load the dictionary that maps DB & index & topK & recall to nprobe, evaluate them all, then save the results

python bench_cpu_performance.py --on_disk 0 --qbs 1 --load_from_dict 1 --overwrite 0 --nprobe_dict_dir './recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --throughput_dict_dir './cpu_performance_result/cpu_throughput_SIFT100M.pkl' --response_time_dict_dir './cpu_performance_result/cpu_response_time_SIFT100M.pkl' 
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
from datasets import read_deep_fbin, read_deep_ibin
from datasets import ivecs_read
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--on_disk', type=int, default=0, help="0 -> search in memory; 1 -> search on disk based on mmap")
parser.add_argument('--dbname', type=str, default='SIFT100M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default='IVF4096,PQ16', help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--qbs', type=int, default=1, help="query batch size")
parser.add_argument('--nthreads', type=int, default=None, help="number of threads, if not set, use the max")
parser.add_argument('--parametersets', type=str, default='nprobe=1', help="a string of nprobes, e.g., 'nprobe=1 nprobe=32'")


parser.add_argument('--load_from_dict', type=int, default=0, help="whether to use Mode B: evluating throughput by using loaded settings")
parser.add_argument('--overwrite', type=int, default=0, help="whether to overwrite existed performance, by default, skip existed settings")
parser.add_argument('--nprobe_dict_dir', type=str, default='./recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> nprobe")
parser.add_argument('--throughput_dict_dir', type=str, default='./cpu_performance_result/cpu_throughput_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> throughput (QPS)")
parser.add_argument('--response_time_dict_dir', type=str, default='./cpu_performance_result/cpu_response_time_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> response_time (QPS)")

### Wenqi: when loading the index, save it to numpy array, default: False
save_numpy_index = False
# save_numpy_index = False 
# we mem-map the biggest files to avoid having them in memory all at
# once

args = parser.parse_args()

# https://github.com/facebookresearch/faiss/blob/main/faiss/index_io.h
# https://www.programcreek.com/python/example/112290/faiss.write_index
if not args.on_disk:
    io_flags = 0
else:
    io_flags = faiss.IO_FLAG_MMAP

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
        if save_numpy_index:
            print("Saving index to numpy array...")
            chunk = faiss.serialize_index(index)
            np.save("{}.npy".format(filename), chunk)
            print("Finish saving numpy index")
    return index


#################################################################
# Perform searches
#################################################################


# we do queries in a single thread
if args.nthreads:
    faiss.omp_set_num_threads(args.nthreads)

if not args.load_from_dict: # Mode A: using arguments passed by the arguments

    dbname = args.dbname
    index_key = args.index_key
    topK = args.topK
    parametersets = args.parametersets.split() # split nprobe argument string by space

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

    print(' ' * len(parametersets[0]), '\t', 'R@{}     time'.format(topK))

    query_vecs = np.reshape(xq, (nq,1,d))

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

else: # Mode B: using dictionary as input, save throughput to another dict

    d_nprobes = None
    if os.path.exists(args.nprobe_dict_dir):
        with open(args.nprobe_dict_dir, 'rb') as f:
            d_nprobes = pickle.load(f)
    else:
        print("ERROR! input dictionary does not exists")
        raise ValueError

    d_throughput = None
    if os.path.exists(args.throughput_dict_dir):
        with open(args.throughput_dict_dir, 'rb') as f:
            d_throughput = pickle.load(f)
    else:
        d_throughput = dict()
    d_response_time = None
    if os.path.exists(args.response_time_dict_dir):
        with open(args.response_time_dict_dir, 'rb') as f:
            d_response_time = pickle.load(f)
    else:
        d_response_time = dict()


    for dbname in d_nprobes:

        if dbname.startswith('SIFT'):
            # SIFT1M to SIFT1000M
            dbsize = int(dbname[4:-1])
            xq = mmap_bvecs('bigann/bigann_query.bvecs')

            gt = ivecs_read('bigann/gnd/idx_%dM.ivecs' % dbsize)

            # Wenqi: load xq to main memory and reshape
            xq = xq.astype('float32').copy()
            xq = np.array(xq, dtype=np.float32)
            gt = np.array(gt, dtype=np.int32)

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

            tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}'.format(dbname, index_key)
            index = get_populated_index()
            ps = faiss.ParameterSpace()
            ps.initialize(index)
            ivfpq_stats = faiss.cvar.indexIVFPQ_stats
            ivf_stats = faiss.cvar.indexIVF_stats
            # faiss.omp_set_num_threads(1)
            query_vecs = np.reshape(xq, (nq,1,d))

            for topK in d_nprobes[dbname][index_key]:

                if topK not in d_throughput[dbname][index_key]:
                    d_throughput[dbname][index_key][topK] = dict()
                if topK not in d_response_time[dbname][index_key]:
                    d_response_time[dbname][index_key][topK] = dict()

                for recall_goal in d_nprobes[dbname][index_key][topK]:

                    if recall_goal not in d_throughput[dbname][index_key][topK]:
                        d_throughput[dbname][index_key][topK][recall_goal] = None
                    if recall_goal not in d_response_time[dbname][index_key][topK]:
                        d_response_time[dbname][index_key][topK][recall_goal] = None
                        
                    # skip if there's already a QPS
                    if d_throughput[dbname][index_key][topK][recall_goal] and d_response_time[dbname][index_key][topK][recall] and (not args.overwrite): 
                        print("SKIP TEST.\tDB: {}\tindex: {}\ttopK: {}\trecall goal: {}\t".format(
                            dbname, index_key, topK, recall_goal))
                        continue

                    if d_nprobes[dbname][index_key][topK][recall_goal] is not None:

                        nprobe = d_nprobes[dbname][index_key][topK][recall_goal]
                        param = "nprobe={}".format(nprobe)

                        sys.stdout.flush()
                        ps.set_index_parameters(index, param)
                        
                        I = np.empty((nq, topK), dtype='int32')
                        D = np.empty((nq, topK), dtype='float32')

                        ivfpq_stats.reset()
                        ivf_stats.reset()

                        t0 = time.time()
                        response_time = [] # in terms of ms
                        i0 = 0
                        while i0 < nq:
                            if i0 + args.qbs < nq:
                                i1 = i0 + args.qbs
                            else:
                                i1 = nq
                            t_RT_start = time.time()
                            Di, Ii = index.search(xq[i0:i1], topK)
                            I[i0:i1] = Ii
                            D[i0:i1] = Di
                            i0 = i1
                            t_RT_end = time.time()
                            response_time.append(1000 * (t_RT_end - t_RT_start)) 

                        t1 = time.time()


                        #for rank in 1, 10:
                        #    n_ok = (I[:, :rank] == gt[:, :1]).sum()
                        #    print("%.4f" % (n_ok / float(nq)), end=' ')
                        throughput = nq / (t1 - t0)
                        print("DB: {}\tindex: {}\ttopK: {}\trecall goal: {}\tnprobe: {}\tQPS = {}".format(
                            dbname, index_key, topK, recall_goal, nprobe, throughput))
                        d_throughput[dbname][index_key][topK][recall_goal] = throughput

                        response_time = np.array(response_time, dtype=np.float32)
                        d_response_time[dbname][index_key][topK][recall_goal] = response_time


                        with open(args.throughput_dict_dir, 'wb') as f:
                            # dictionary format:
                            #   d[dbname (str)][index_key (str)][topK (int)][recall_goal (float, 0~1)] = QPS
                            #   e.g., d["SIFT100M"]["IVF4096,PQ16"][10][0.7]
                            pickle.dump(d_throughput, f, protocol=4)

                        with open(args.response_time_dict_dir, 'wb') as f:
                            # dictionary format:
                            #   d[dbname (str)][index_key (str)][topK (int)][recall_goal (float, 0~1)] = QPS
                            #   e.g., d["SIFT100M"]["IVF4096,PQ16"][10][0.7]
                            pickle.dump(d_response_time, f, protocol=4)
