#!/usr/bin/env python2

"""
This script is used to automatically search the min(nprobe) that can achieve
the target recall R@k using dataset D and index I.

Note: better make sure that the recall goal is achievable on the given index, 
    otherwise it may take a century to detect that it's not achievable

Example:

python bench_cpu_recall.py --dbname SIFT100M --index_key IVF4096,PQ16 --recall_goal 80 --topK 10
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
from datasets import read_deep_fbin, read_deep_ibin

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default=0, help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default=0, help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--recall_goal', type=float, default=50, help="target minimum recall, e.g., 50%")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
# parser.add_argument('--parametersets', type=str, default=0, help="nprobe series, e.g., 'nprobe=1 nprobe=2 nprobe=4'")

args = parser.parse_args()
dbname = args.dbname
index_key = args.index_key
recall_goal = args.recall_goal / 100.0
topK = args.topK

nlist = None
index_array = index_key.split(",")
index_type = None
if len(index_array) == 2: # "IVF4096,PQ16" or "IMI2x14,PQ16" 
    s = index_array[0]
    if s[:3]  == "IVF":
        index_type = "IVF"
        nlist = int(s[3:])
    elif s[:5]  == "IMI2x":
        nlist = int((2 ** int(s[5:])) ** 2)
        index_type = "IMI"
    else:
        raise ValueError
elif len(index_array) == 3: # "OPQ16,IVF4096,PQ16" or "OPQ16,IMI2x14,PQ16" 
    s = index_array[1]
    if s[:3]  == "IVF":
        nlist = int(s[3:])
        index_type = "IVF"
    elif s[:5]  == "IMI2x":
        nlist = int((2 ** int(s[5:])) ** 2)
        index_type = "IMI"
    else:
        raise ValueError
else:
    raise ValueError

threshold_nlist = nlist 
if index_type == "IVF":
    if nlist <= 64:
        pass
    elif nlist <= 128:
        threshold_nlist = nlist / 2
    elif nlist <= 256:
        threshold_nlist = nlist / 4
    elif nlist <= 512:
        threshold_nlist = nlist / 8
    elif nlist <= 1024:
        threshold_nlist = nlist / 16
    elif nlist > 1024:
        threshold_nlist = nlist / 32
elif index_type == "IMI":
    threshold_nlist = int(np.sqrt(nlist))
else:
    print("Unknown index type")
    raise(ValueError)
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


#################################################################
# Bookkeeping
#################################################################



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


else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)


print("sizes: B %s Q %s T %s gt %s" % (
    xb.shape, xq.shape, xt.shape, gt.shape))

nq, d = xq.shape
nb, d = xb.shape
assert gt.shape[0] == nq


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
        index = faiss.read_index(filename)
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

ps = faiss.ParameterSpace()
ps.initialize(index)

# make sure queries are in RAM
xq = xq.astype('float32').copy()

# a static C++ object that collects statistics about searches
ivfpq_stats = faiss.cvar.indexIVFPQ_stats
ivf_stats = faiss.cvar.indexIVF_stats

# we do queries in a single thread
# faiss.omp_set_num_threads(1)

print(' ' * len("nprobe=1024"), '\t', 'R@{}'.format(topK))

param = 1 # start nprobe
min_range = 1
max_range = None

while True:
    print("nprobe={}".format(param), '\t', end=' ')
    sys.stdout.flush()
    ps.set_index_parameters(index, "nprobe={}".format(param))
    t0 = time.time()
    ivfpq_stats.reset()
    ivf_stats.reset()
    D, I = index.search(xq, topK)
    t1 = time.time()
    if t1 - t0 >= 1800:
        print("spend more than 1800 seconds to run nprobe={}, quit", param)
        print("ERROR! Search failed: cannot reach expected recall on given dataset and index")
        break
    n_ok = (I[:, :topK] == gt[:, :1]).sum()
    recall = n_ok / float(nq)
    print("%.4f" % (recall), end='\n')
    if recall >= recall_goal:
        max_range = param # max range is used when recall goal is achieved
        param = int((min_range + param) / 2.0)
        if param == min_range:
            break
    else:
        min_range = param  # to achieve target recall, need larger than this nprobe
        if param == threshold_nlist:
            print("ERROR! Search failed: cannot reach expected recall on given dataset and index")
            break
        elif max_range:
            if param  ==  max_range - 1:
                break
            param = int((max_range + param) / 2.0)
        else:
            param = param * 2
            if param > threshold_nlist:
                param = threshold_nlist

min_nprobe = max_range
print("The minimum nprobe to achieve R@{topK}={recall_goal} on {dbname} {index_key} is {nprobe}".format(
    topK=topK, recall_goal=recall_goal, dbname=dbname, index_key=index_key, nprobe=min_nprobe))

fname = './recall_info/cpu_recall_index_nprobe_pairs_{}.pkl'.format(dbname)
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

