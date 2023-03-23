"""
Given the same bytes per vector, e.g., 16, evaluate the CPU recall (R@1) and throughput (QPS) using different nbits

Example usage:
    python nbits_experiments_fix_bytes_per_vec.py  --bytes_per_vec 16
    
Goal:
    In principle, we can construct the same bytes per vector by m and nbits. For example, 16 bytes per vector = 16 x 8 bits = 32 x 4 bits = 8 x 16 bits. Each of them have different tradeoffs:
        1. the larger the nbits === the smaller the m: 
            (a) (-) the more effort is required to construct distance LUTs, as the computation cost is O(d * 2^nbits); 
            (b) (-) the more likely cache miss will happen (distance LUT has d * 2 ^ nbits elements) 
            (c) (+) the less operations per ADC due (but large tables can result in cache misses)
            this setting should be better when there are many PQ codes in the table
        2. it is unclear what combination of (m, nbits) can yield the best QPS / recall
        
On CPU, though, nbits=8 is probably the only option one want to use, because setting nbits=4 can reduce the performance by almost 10x, and setting nbits=16 will lead to a unnecessarily large distance LUT that results in unacceptable performance (training the index takes UNKNOWN time on even a 1M dataset).

Note: d % m must be 0

Traceback (most recent call last):
  File "nbits_experiments_fix_bytes_per_vec.py", line 82, in <module>
    index = faiss.IndexPQ(d, m, nbits)
  File "/home/wejiang/anaconda3/envs/py37/lib/python3.7/site-packages/faiss/swigfaiss_avx2.py", line 2865, in __init__
    this = _swigfaiss_avx2.new_IndexPQ(*args)
RuntimeError: Error in void faiss::ProductQuantizer::set_derived_values() at impl/ProductQuantizer.cpp:189: Error: 'd % M == 0' failed
"""

import numpy as np
import time
import faiss

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--bytes_per_vec', type=int, default=16, help="number of bytes per vector")

args = parser.parse_args()
bytes_per_vec = args.bytes_per_vec
nb = int(1e6)
nq = int(1e4)

# Wenqi: use k = 1 to eliminate the priority queue performance factor
k = 1
d = 128

print("Parameters in the experiments:")
print("bytes_per_vec:", bytes_per_vec)
print("nb (db vector num):", nb)
print("nq (num queries):", nq)
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

gt = ivecs_read('../bigann/gnd/idx_1M.ivecs')


for nbits in range(15, 16 + 1):
    
    max_bits = bytes_per_vec * 8
    m = int(np.floor(max_bits / nbits))
    
    # downgrade m if d % m != 0
    if d % m != 0:
        max_exp = 0
        for i in range(7 + 1): # 128 = 2 ^ 7
            if m > 2 ** i:
                max_exp = i
            else:
                break
        m = int(2 ** max_exp)
          
    pad_bits = max_bits - m * nbits
    
    index = faiss.IndexPQ(d, m, nbits)
    index.train(xt) 
    index.add(xb)

    start = time.time()
    D, I = index.search(xq, k) # sanity check
    end = time.time()
    
    n_ok = (I[:, :k] == gt[:, :1]).sum()
    recall = n_ok / float(nq)
    print("nbits = {}\tm={}\tpad_bits={}\tQPS: {:.2f}\trecall: {}".format(nbits, m, pad_bits, nq / (end - start), recall))
