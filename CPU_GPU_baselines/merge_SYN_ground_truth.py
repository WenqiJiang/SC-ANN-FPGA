"""
For the generated SYN dataset that is larger than 100M, the ground truth are stored in 
    100-M-gt partitions. This script is used to merge them.

Usage: 
    python merge_SYN_ground_truth.py --dbname SYN10000M
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import pickle
from multiprocessing.dummy import Pool as ThreadPool
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SYN1M', help="dataset name, e.g., SYN1000M")

args = parser.parse_args()
dbname = args.dbname

topK = 1000 # gt == 1000 results per query
query_num = 10000

def SYN_ivecs_read(fname):
    a = np.fromfile(fname, dtype='int64')
    d = 1000 # topK=1000 results
    return a.reshape(-1, d).copy()

def SYN_fvecs_read(fname):
    a = np.fromfile(fname, dtype='float32')
    d = 1000 # topK=1000 results
    return a.reshape(-1, d).copy()

dataset_dir = os.path.join('SYN_dataset', dbname)

if dbname.startswith('SYN'):
    dbsize = int(dbname[3:-1])
    if dbsize < 100:
        print("db size < 100M, need not merging results")
        sys.exit(0)
    else:
        # must be the multiple of 100M
        assert dbsize % 100 == 0
else:
    print("Unknown dataset")
    sys.exit(1)

print("loading ground truth...")
db_batches = int(dbsize / 100)
print("number of batches: ", db_batches)
gt_ID_all_list = []
gt_dist_all_list = []
for batch_id in range(db_batches):
    gt_ID_dir = os.path.join(dataset_dir, "idx_10000_by_1000_int64_{}_of_{}.lvecs".format(batch_id, db_batches))
    gt_dist_dir = os.path.join(dataset_dir, "dis_10000_by_1000_float32_{}_of_{}.fvecs".format(batch_id, db_batches))
    gt_ID_all_list.append(SYN_ivecs_read(gt_ID_dir))
    gt_dist_all_list.append(SYN_fvecs_read(gt_dist_dir))
    assert gt_ID_all_list[-1].shape == (query_num, topK)
    assert gt_dist_all_list[-1].shape == (query_num, topK)

print("merging ground truth...")

gt_ID_merged = np.zeros((query_num, topK), dtype='int64')
gt_dist_merged = np.zeros((query_num, topK), dtype='float32')

for query_id in range(query_num):
    print("query ID: {}".format(query_id))
    ID_list = []
    dist_list = []
    for batch_id in range(db_batches):
        ID_list.append(gt_ID_all_list[batch_id][query_id])
        dist_list.append(gt_dist_all_list[batch_id][query_id])
    ID_array = np.concatenate(ID_list)
    dist_array = np.concatenate(dist_list)
    
    topK_indices = np.argsort(dist_array)[:topK]
    selected_ID = np.take(ID_array, topK_indices)
    selected_dist = np.take(dist_array, topK_indices)

    gt_ID_merged[query_id] = selected_ID
    gt_dist_merged[query_id] = selected_dist


gt_dist_dir = os.path.join(dataset_dir, "dis_{}_by_{}_float32.fvecs".format(query_num, topK))
gt_id_dir = os.path.join(dataset_dir, "idx_{}_by_{}_int64.lvecs".format(query_num, topK))

gt_ID_merged.tofile(gt_id_dir)
gt_dist_merged.tofile(gt_dist_dir)
