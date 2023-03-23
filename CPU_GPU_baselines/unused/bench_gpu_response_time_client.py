#!/usr/bin/env python3

import socket
import numpy as np
import time
import sys
import os
import struct
import pickle

"""
Benchmarking the GPU's response time using 10,000 queries (takes several minutes),
the batch size is set to 1

Start the server first (bench_gpu_response_time_server.py), then the client (bench_gpu_response_time_client.py),
    make sure they are using the same mode (two modes specified below)

There are 2 ways to use the script:

(1) Test the response time of given DB & index & nprobe:

python bench_gpu_response_time_client.py --dbname SIFT100M --index_key OPQ16,IVF4096,PQ16 --topK 10 --param 'nprobe=32' --HOST 127.0.0.1 --PORT 65432

(2) Load the dictionary that maps DB & index & topK & recall to nprobe, evaluate them all, then save the results

python bench_gpu_response_time_client.py --load_from_dict 1 --overwrite 0 --nprobe_dict_dir './recall_info/gpu_recall_index_nprobe_pairs_SIFT100M.pkl' --performance_dict_dir './gpu_performance_result/gpu_response_time_SIFT100M.pkl' --HOST 10.1.212.76 --PORT 65432

"""

# python socket tutorial: https://realpython.com/python-sockets/#socket-api-overview

#### Change server's host IP when needed ####


import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SIFT100M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default='IVF4096,PQ16', help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--param', type=str, default='nprobe=1', help="a string of nprobes, e.g., 'nprobe=1 nprobe=32'")
parser.add_argument('--HOST', type=str, default='127.0.0.1', help="server HOST IP")
parser.add_argument('--PORT', type=int, default=65432, help="HOST port")


parser.add_argument('--load_from_dict', type=int, default=0, help="whether to use Mode B: evluating throughput by using loaded settings")
parser.add_argument('--overwrite', type=int, default=0, help="whether to overwrite existed performance, by default, skip existed settings")
parser.add_argument('--nprobe_dict_dir', type=str, default='./recall_info/gpu_recall_index_nprobe_pairs_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> nprobe")
parser.add_argument('--performance_dict_dir', type=str, default='./gpu_performance_result/gpu_response_time_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> response time (10000-D numpy array)")

args = parser.parse_args()

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


xq = mmap_bvecs('bigann/bigann_query.bvecs')
xq = xq.astype('float32').copy()
xq = np.array(xq, dtype=np.float32)
xq = np.tile(xq, (10, 1)) # replicate the 10K queries to 100K queries to get a more stable performance
# xq_list = []
# for i in range(10):
#     xq_list.append(xq)
# xq = np.array(xq_list)
# xq = np.reshape(xq, (xq.shape[0] * xq.shape[1], -1))

nq, d = xq.shape

query_vecs = []
for i in range(nq):
    query_vec = xq[i]
    query_vec = np.reshape(query_vec, (1,128))
    query_vec = query_vec.tostring()
    query_vecs.append(query_vec)

if not args.load_from_dict: # Mode A: using arguments passed by the arguments

    dbname = args.dbname
    index_key = args.index_key
    param = args.param

    response_time = [] # in terms of ms

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        rcvdData = c.recv(1024).decode()

        for i in range(nq):
            t0 = time.time()
            s.sendall(query_vecs[i])
            I = s.recv(BYTES_PER_RESULT) # top 10 * 4 bytes
            I = np.fromstring(I, dtype=np.int32)
            t1 = time.time()
            response_time.append(1000 * (t1 - t0)) 

        response_time = np.array(response_time, dtype=np.float32)
        
        np.save('./GPU_response_time/GPU_response_time_{}_{}_{}'.format(dbname, index_key, param), 
            response_time)

else: # Mode B: using dictionary as input, save throughput to another dict

    d_response_time = None
    if os.path.exists(args.performance_dict_dir):
        with open(args.performance_dict_dir, 'rb') as f:
            d_response_time = pickle.load(f)
    else:
        d_response_time = dict()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        s.connect((HOST, PORT))

        while True: # the loop will break when the server tears down the connecitoin

            print("Start sending data")
            dbname_len = struct.unpack('>I', s.recv(4))[0]
            dbname = s.recv(dbname_len).decode()
            index_key_len = struct.unpack('>I', s.recv(4))[0]
            index_key = s.recv(index_key_len).decode()
            topK_len = struct.unpack('>I', s.recv(4))[0]
            topK = int(s.recv(topK_len).decode())
            recall_goal_len = struct.unpack('>I', s.recv(4))[0]
            recall_goal = float(s.recv(recall_goal_len).decode())
            print("DB: {}\tindex_key: {}\ttopK: {}\trecall_goal: {}".format(
                dbname, index_key, topK, recall_goal))
            BYTES_PER_RESULT = 4 * topK # int32

            response_time = [] # in terms of ms
            for i in range(nq):
                t0 = time.time()
                s.sendall(query_vecs[i])
                I = s.recv(BYTES_PER_RESULT) # top 10 * 4 bytes
                I = np.fromstring(I, dtype=np.int32)
                t1 = time.time()
                response_time.append(1000 * (t1 - t0)) 

            response_time = np.array(response_time, dtype=np.float32)
            
            if dbname not in d_response_time:
                d_response_time[dbname] = dict()
            
            if index_key not in d_response_time[dbname]:
                d_response_time[dbname][index_key] = dict()

            if topK not in d_response_time[dbname][index_key]:
                d_response_time[dbname][index_key][topK] = dict()

            if recall_goal not in d_response_time[dbname][index_key][topK] or args.overwrite:
                d_response_time[dbname][index_key][topK][recall_goal] = response_time
                           
            print("Finish sending data, DB: {}\tindex_key: {}\ttopK: {}\trecall_goal: {}".format(
                dbname, index_key, topK, recall_goal))

            with open(args.performance_dict_dir, 'wb') as f:
                # dictionary format:
                #   d[dbname (str)][index_key (str)][topK (int)][recall_goal (float, 0~1)] = response time array (np array)
                #   e.g., d["SIFT100M"]["IVF4096,PQ16"][10][0.7]
                pickle.dump(d_response_time, f, pickle.HIGHEST_PROTOCOL)
