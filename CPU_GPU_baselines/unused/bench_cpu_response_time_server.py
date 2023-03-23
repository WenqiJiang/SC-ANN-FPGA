#!/usr/bin/env python3

"""
Benchmarking the CPU's response time using 10,000 queries (takes several minutes),
the batch size is set to 1

Start the server first (bench_cpu_response_time_server.py), then the client (bench_cpu_response_time_client.py),
    make sure they are using the same mode (two modes specified below)    

There are 2 ways to use the script:

(1) Test the response time of given DB & index & nprobe:

python bench_cpu_response_time_server.py --dbname SIFT100M --index_key OPQ16,IVF4096,PQ16 --topK 10 --param 'nprobe=32' --HOST 127.0.0.1 --PORT 65432

(2) Load the dictionary that maps DB & index & topK & recall to nprobe, evaluate them all, then save the results

python bench_cpu_response_time_server.py --load_from_dict 1 --overwrite 0 --nprobe_dict_dir './recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --performance_dict_dir './cpu_performance_result/cpu_response_time_SIFT100M.pkl' --HOST 10.1.212.76 --PORT 65432

"""


from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import faiss
import pickle
import struct
from multiprocessing.dummy import Pool as ThreadPool
from datasets import ivecs_read

import socket

# python socket tutorial: https://realpython.com/python-sockets/#socket-api-overview


import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SIFT100M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default='IVF4096,PQ16', help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--param', type=str, default='nprobe=1', help="a string of nprobes, e.g., 'nprobe=1 nprobe=32'")
parser.add_argument('--HOST', type=str, default='127.0.0.1', help="HOST IP")
parser.add_argument('--PORT', type=int, default=65432, help="HOST port")


parser.add_argument('--load_from_dict', type=int, default=0, help="whether to use Mode B: evluating throughput by using loaded settings")
parser.add_argument('--overwrite', type=int, default=0, help="whether to overwrite existed performance, by default, skip existed settings")
parser.add_argument('--nprobe_dict_dir', type=str, default='./recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> nprobe")
parser.add_argument('--performance_dict_dir', type=str, default='./cpu_performance_result/cpu_response_time_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> response time (10000-D numpy array)")

args = parser.parse_args()

# python socket tutorial: https://realpython.com/python-sockets/#socket-api-overview
BYTES_PER_QUERY = 128 * 4
QUERY_NUM = 10000

HOST          = args.HOST # e.g., '127.0.0.1', set to physical interface if needed
PORT          = args.PORT# e.g., 65432 Port to listen on


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

    if os.path.exists(filename):
        print("loading", filename)
        index = faiss.read_index(filename)
    else:
        raise ValueError("no index file")
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
        raise ValueError("index file not found")
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index


#################################################################
# Perform searches
#################################################################
if not args.load_from_dict: # Mode A: using arguments passed by the arguments

    dbname = args.dbname
    index_key = args.index_key
    topK = args.topK
    param = args.param

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


    index = get_populated_index()

    ps = faiss.ParameterSpace()
    ps.initialize(index)

    # a static C++ object that collects statistics about searches
    ivfpq_stats = faiss.cvar.indexIVFPQ_stats
    ivf_stats = faiss.cvar.indexIVF_stats

    # we do queries in a single thread
    faiss.omp_set_num_threads(1)

    # Load the database by performing 1 single search
    sample_vec = np.zeros((1,128), dtype=np.float32)

    print(param, '\t', end=' ')
    sys.stdout.flush()
    ps.set_index_parameters(index, param)
    ivfpq_stats.reset()
    ivf_stats.reset()

    D, I = index.search(sample_vec, topK)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        print("start listening")
        s.listen()
        conn, addr = s.accept()
        t0 = time.time()

        with conn:
            print('Connected by', addr)
            for i in range(QUERY_NUM):
                s = conn.recv(BYTES_PER_QUERY)
                query_vec = np.fromstring(s, dtype=np.float32)
                query_vec = np.reshape(query_vec, (1, 128))
                D, I = index.search(query_vec, topK)
                I = np.array(I, dtype=np.int32)
                I = I.tostring()
                conn.sendall(I)

        t1 = time.time()
        print("QPS = {} (using a single client)".format(QUERY_NUM / (t1 - t0)))

else: # Mode B: using dictionary as input, save throughput to another dict

    d_nprobes = None
    if os.path.exists(args.nprobe_dict_dir):
        with open(args.nprobe_dict_dir, 'rb') as f:
            d_nprobes = pickle.load(f)
    else:
        print("ERROR! input dictionary does not exists")
        raise ValueError

    d_response_time = None
    if os.path.exists(args.performance_dict_dir):
        with open(args.performance_dict_dir, 'rb') as f:
            d_response_time = pickle.load(f)
    else:
        d_response_time = dict()


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        print("start listening")
        s.listen()
        conn, addr = s.accept()
        t0 = time.time()

        with conn:
            print('Connected by', addr)

            # Load the database by performing 1 single search
            sample_vec = np.zeros((1,128), dtype=np.float32)

            # After the dictionary is traversed, the connection will automatically tear down
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

                else:
                    print('unknown dataset', dbname, file=sys.stderr)
                    sys.exit(1)

                nq, d = xq.shape
                assert gt.shape[0] == nq
                
                if dbname not in d_response_time:
                    d_response_time[dbname] = dict()
                
                for index_key in d_nprobes[dbname]:

                    if index_key not in d_response_time[dbname]:
                        d_response_time[dbname][index_key] = dict()

                    tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}'.format(dbname, index_key)
                    index = get_populated_index()
                    ps = faiss.ParameterSpace()
                    ps.initialize(index)
                    ivfpq_stats = faiss.cvar.indexIVFPQ_stats
                    ivf_stats = faiss.cvar.indexIVF_stats
                    faiss.omp_set_num_threads(1)
                    query_vecs = np.reshape(xq, (nq,1,128))

                    for topK in d_nprobes[dbname][index_key]:

                        if topK not in d_response_time[dbname][index_key]:
                            d_response_time[dbname][index_key][topK] = dict()

                        for recall_goal in d_nprobes[dbname][index_key][topK]:

                            if recall_goal not in d_response_time[dbname][index_key][topK]:
                                d_response_time[dbname][index_key][topK][recall_goal] = None
                                
                            # skip if it is already evaluated
                            if (d_response_time[dbname][index_key][topK][recall_goal] is not None) and (not args.overwrite): 
                                print("SKIP TEST.\tDB: {}\tindex: {}\ttopK: {}\trecall goal: {}\t".format(
                                    dbname, index_key, topK, recall_goal))
                                continue

                            if d_nprobes[dbname][index_key][topK][recall_goal] is not None:

                                nprobe = d_nprobes[dbname][index_key][topK][recall_goal]
                                param = "nprobe={}".format(nprobe)

                                sys.stdout.flush()
                                ps.set_index_parameters(index, param)
                                t0 = time.time()
                                ivfpq_stats.reset()
                                ivf_stats.reset()

                                D, I = index.search(sample_vec, topK)
                                
                                print("Start receiving data")
                                # sendback dbname, index_key, topK, recall_goal
                                dbname_str = dbname.encode()
                                conn.sendall(struct.pack('>I', len(dbname_str)))
                                conn.sendall(dbname_str)
                                index_key_str = index_key.encode()
                                conn.sendall(struct.pack('>I', len(index_key_str)))
                                conn.sendall(index_key.encode())
                                topK_str = str(topK).encode()
                                conn.sendall(struct.pack('>I', len(topK_str)))
                                conn.sendall(topK_str)
                                recall_goal_str = str(recall_goal).encode()
                                conn.sendall(struct.pack('>I', len(recall_goal_str)))
                                conn.sendall(recall_goal_str)

                                print("DB: {}\tindex_key: {}\ttopK: {}\trecall_goal: {}".format(
                                    dbname, index_key, topK, recall_goal))

                                for i in range(QUERY_NUM):
                                    s = conn.recv(BYTES_PER_QUERY)
                                    query_vec = np.fromstring(s, dtype=np.float32)
                                    query_vec = np.reshape(query_vec, (1, 128))
                                    D, I = index.search(query_vec, topK)
                                    I = np.array(I, dtype=np.int32)
                                    I = I.tostring()
                                    conn.sendall(I)
            
                                t1 = time.time()
                                #for rank in 1, 10:
                                #    n_ok = (I[:, :rank] == gt[:, :1]).sum()
                                #    print("%.4f" % (n_ok / float(nq)), end=' ')
                                throughput = nq / (t1 - t0)
                                print("DB: {}\tindex: {}\ttopK: {}\trecall goal: {}\tnprobe: {}\tQPS = {}".format(
                                    dbname, index_key, topK, recall_goal, nprobe, throughput))