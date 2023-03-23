"""
Given the same m, and same the k=1,  evaluate the CPU performance (QPS) using different nbits

Example usage:
    python nbits_experiments_fix_m.py --m 16 --nb 1000000 --nq 10000
    
Goal:
    In principle, if there's not the instruction constraints on CPU, a lower nbit (given the same m) will:
        1. reduce the bytes needed for a vector (which might increase performance due to the lower required bytes to be scanned)
        2. reduce the time needed for constructing distance LUT (this is obvious when vectors in the cell is low, while hard to tell the difference when there are many vectors in the cell)
        3. have the same number of lookup operations and additions per ADC
        
    However, customized nbits per vector (other than nbits=8 or nbits=16), will lead to a number of bit shift operations during table lookups, thus the performance may actually be much lower than the standard nbits=8/16.
"""

import numpy as np
import time
import faiss

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--m', type=int, default=16, help="number of sub-quantizers per vector")

parser.add_argument('--nb', type=int, default=int(1e6), help="database size: number of vectors")
parser.add_argument('--nq', type=int, default=int(1e5), help="query size: number of vectors")


args = parser.parse_args()
m = args.m
nb = args.nb
nq = args.nq

if nq > 1e4: 
    print("largest supported query num = 10000, use 10000 instead of {}".format(nq))
    nq = int(1e4)   
    
# Wenqi: use k = 1 to eliminate the priority queue performance factor
k = 1
d = 128

print("Parameters in the experiments:")
print("nb (db vector num):", nb)
print("nq (num queries):", nq)
print("m (num sub-quantizers):", m)
print("d (dimension of vectors):", d)
print("k (topK):", k)

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # Wenqi: Format of ground truth (for 10000 query vectors):
    #   1000(topK), [1000 ids]
    #   1000(topK), [1000 ids]
    #        ...     ...
    #   1000(topK), [1000 ids]
    # 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    return a.reshape(-1, d + 1)[:, 1:].copy()

xb = mmap_bvecs('../bigann/bigann_base.bvecs')
# trim xb to correct size
xb = np.array(xb[:nb], dtype=np.float32)

xq = mmap_bvecs('../bigann/bigann_query.bvecs')
xq = np.array(xq[:nq], dtype=np.float32)

# use the same learning set (always use the first 1e6 vectors)
xt = mmap_bvecs('../bigann/bigann_learn.bvecs')
xt = np.array(xt[:int(1e6)], dtype=np.float32)

if nb == int(1e6):
    gt = ivecs_read('../bigann/gnd/idx_1M.ivecs')
elif nb == int(1e7):
    gt = ivecs_read('../bigann/gnd/idx_10M.ivecs')
elif nb == int(1e8):
    gt = ivecs_read('../bigann/gnd/idx_100M.ivecs')
elif nb == int(1e9):
    gt = ivecs_read('../bigann/gnd/idx_1000M.ivecs')

for nbits in range(1, 16 + 1):
    
    index = faiss.IndexPQ(d, m, nbits)
    index.train(xt) 
    index.add(xb)

    start = time.time()
    D, I = index.search(xq, k) # sanity check
    end = time.time()
    if nb == int(1e6) or nb == int(1e7) or nb == int(1e8) or nb == int(1e9):
        n_ok = (I[:, :k] == gt[:, :1]).sum()
        recall = n_ok / float(nq)
        print("nbits = {}\tQPS: {}\trecall: {}".format(nbits, nq / (end - start), recall))
    else:
        print("nbits = {}\tQPS: {}".format(nbits, nq / (end - start)))
