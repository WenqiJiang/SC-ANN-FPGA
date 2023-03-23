"""
Merge multi-shard index to a single big one.
Example usage:

python merge_indexes.py --dbname SBERT3000M --index_key IVF65536,PQ64 --n_shards 4

Reference: https://gist.github.com/mdouze/7331e6fc1da2334f30706b9b9962068b
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import faiss
from multiprocessing.dummy import Pool as ThreadPool

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SBERT3000M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default='IVF65536,PQ64', help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--n_shards', type=int, default=4, help="e.g., can use 2 or 4 shards for large datasets")

args = parser.parse_args()
dbname = args.dbname
index_key = args.index_key
n_shards = args.n_shards

tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}_{}shards'.format(dbname, index_key, n_shards)

filename_ls = []
for i in range(n_shards):
    filename_ls.append("%s/%s_%s_populated_shard_%s.index" % (
        tmpdir, dbname, index_key, str(i)))

print("load the first index")
index = faiss.read_index(filename_ls[0]) # the index to be add to 

def merge_invlists(il_src, il_dest):
    """ 
    merge inverted lists from two ArrayInvertedLists 
    add src index to dest index
    """
    assert il_src.nlist == il_dest.nlist
    assert il_src.code_size == il_dest.code_size
    
    for list_no in range(il_src.nlist): 
        
        il_dest.add_entries(
            list_no,
            il_src.list_size(list_no), 
            il_src.get_ids(list_no), 
            il_src.get_codes(list_no)
        )    

ntotal = index.ntotal 
for i in range(1, n_shards): 
    index_shard = faiss.read_index(filename_ls[i])
    print("Current shard num vec: ", index_shard.ntotal)
    merge_invlists(
        faiss.extract_index_ivf(index_shard).invlists,
        faiss.extract_index_ivf(index).invlists 
    )
    ntotal += index_shard.ntotal
    del index_shard

index.ntotal = faiss.extract_index_ivf(index).ntotal = ntotal 
print("Merged index num vec: ", index.ntotal)
print("Finished merge... writing")
filename = "%s/%s_%s_populated.index" % (
            tmpdir, dbname, index_key)
faiss.write_index(index, filename)