"""
This script evaluate the recall and performance of multi-shard indexes.

It first run the program on each CPU server, then gather the results to compute recall, and measure the performance.

Example: 
python bench_multi_cpu_performance_OSDI.py --dbname SBERT3000M --index_key IVF65536,PQ64 --n_shards 4 --performance_dict_dir './cpu_performance_result/m5.4xlarge_cpu_performance.pkl' --overwrite 0

"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import pickle
import argparse 
from datasets import read_deep_fbin, read_deep_ibin, mmap_bvecs_SBERT, mmap_bvecs_GNN

parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SIFT100M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default='IVF4096,PQ16', help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--n_shards', type=int, default=None, help="e.g., can use 2 or 4 shards for large datasets")
parser.add_argument('--overwrite', type=int, default=0, help="whether to overwrite existed performance, by default, skip existed settings")
parser.add_argument('--performance_dict_dir', type=str, default='./cpu_performance_result/cpu_throughput_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> throughput (QPS)")

args = parser.parse_args()

dbname = args.dbname
index_key = args.index_key
n_shards = args.n_shards
overwrite = args.overwrite
performance_dict_dir = args.performance_dict_dir

record_latency_distribution = 1 # always record latency distribution
record_computed_results = 1 # always record results
assert performance_dict_dir[-4:] == '.pkl'


# evaluate the performance results
performance_dict_list = []
for shard_id in range(n_shards):
    performance_dict_dir_shard = performance_dict_dir[:-4] + '_shard_{}'.format(shard_id)
    cmd = "python bench_cpu_performance_OSDI.py --dbname {} --index_key {} --n_shards {} --shard_id {} --performance_dict_dir {} --record_latency_distribution {} --record_computed_results {} --overwrite {}".format(
        dbname, index_key, n_shards, shard_id, performance_dict_dir_shard, record_latency_distribution, record_computed_results, overwrite)
    print("Running command:\n{}".format(cmd))
    os.system(cmd)
    with open(performance_dict_dir_shard, 'rb') as f:
        dict_perf = pickle.load(f)
        performance_dict_list.append(dict_perf)

# Merge the performance results
# Make sure these numbers to be the same with bench_cpu_performance_OSDI.py
topK = 100
qbs_list = [1, 2, 4, 8, 16, 32, 64]
qbs_list.reverse() # using large batches first since they are faster

# load the ground truth
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

nq = query_num

def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

"""
Input dict format

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
    dict[dbname][index_key][qbs][nprobe]["I"] -> idx, shape = np.empty((nq, topK), dtype='int64')
    dict[dbname][index_key][qbs][nprobe]["D"] -> dist, shape = np.empty((nq, topK), dtype='float32')
"""

dict_perf_merged = None
if os.path.exists(args.performance_dict_dir):
    with open(args.performance_dict_dir, 'rb') as f:
        dict_perf_merged = pickle.load(f)
else:
    dict_perf_merged = dict()

if dbname not in dict_perf_merged:
    dict_perf_merged[dbname] = dict()

if index_key not in dict_perf_merged[dbname]:
    dict_perf_merged[dbname][index_key] = dict()

for qbs in qbs_list:

    print("\nbatch size: ", qbs)
    sys.stdout.flush()

    if qbs not in dict_perf_merged[dbname][index_key]:
        dict_perf_merged[dbname][index_key][qbs] = dict()

    for nprobe in nprobe_list:

        print("\nnprobe: ", nprobe)
        if nprobe not in dict_perf_merged[dbname][index_key][qbs]:
            dict_perf_merged[dbname][index_key][qbs][nprobe] = dict()

        I = np.zeros((query_num, topK), dtype='int64')
        D = np.zeros((query_num, topK), dtype='float32')

        # Compute recall
        for query_id in range(query_num):
            
            ID_list = []
            dist_list = []
            for shard_id in range(n_shards):
                dict_perf = performance_dict_list[shard_id]
                ID_list.append(dict_perf[dbname][index_key][qbs][nprobe]["I"][query_id])
                dist_list.append(dict_perf[dbname][index_key][qbs][nprobe]["D"][query_id])
            ID_array = np.concatenate(ID_list)
            dist_array = np.concatenate(dist_list)
            
            topK_indices = np.argsort(dist_array)[:topK]
            selected_ID = np.take(ID_array, topK_indices)
            selected_dist = np.take(dist_array, topK_indices)

            I[query_id] = selected_ID
            D[query_id] = selected_dist


        n_ok = (I[:, :topK] == gt[:, :1]).sum()
        for rank in 1, 10, 100: # R1@K
            n_ok = (I[:, :rank] == gt[:, :1]).sum()
            R1_at_K = n_ok / float(nq)
            if rank == 1:
                print("R1@1 = %.4f" % (R1_at_K), end='\t')
                dict_perf_merged[dbname][index_key][qbs][nprobe]["R1@1"] = R1_at_K
            elif rank == 10:
                print("R1@10 = %.4f" % (R1_at_K), end='\t')
                dict_perf_merged[dbname][index_key][qbs][nprobe]["R1@10"] = R1_at_K
            elif rank == 100:
                print("R1@100 = %.4f" % (R1_at_K), end='\t')
                dict_perf_merged[dbname][index_key][qbs][nprobe]["R1@100"] = R1_at_K
        for rank in 1, 10, 100: # R@K
            R_at_K = compute_recall(I[:,:rank], gt[:, :rank])
            if rank == 1:
                print("R@1 = %.4f" % R_at_K, end='\t')
                dict_perf_merged[dbname][index_key][qbs][nprobe]["R@1"] = R_at_K
            elif rank == 10:
                print("R@10 = %.4f" % R_at_K, end='\t')
                dict_perf_merged[dbname][index_key][qbs][nprobe]["R@10"] = R_at_K
            elif rank == 100:
                print("R@100 = %.4f" % R_at_K, end='\t')
                dict_perf_merged[dbname][index_key][qbs][nprobe]["R@100"] = R_at_K

        # Compute performance
        latency_tmp_list = []
        for dict_perf in performance_dict_list: 
            latency_tmp_list.append(dict_perf[dbname][index_key][qbs][nprobe]["latency_distribution"])
        latency_array = np.array(latency_tmp_list)
        tail_latency_ms = latency_array.max(axis=0) # decides by the highest latency per batch
        total_sec = np.sum(tail_latency_ms) / 1000
        real_QPS = query_num / total_sec
        sorted_t_query_list = np.sort(np.array(tail_latency_ms))
        latency_50 = sorted_t_query_list[np.amin([len(sorted_t_query_list) - 1, int(np.ceil(len(sorted_t_query_list) * 0.5))])]
        latency_95 = sorted_t_query_list[np.amin([len(sorted_t_query_list) - 1, int(np.ceil(len(sorted_t_query_list) * 0.95))])]

        dict_perf_merged[dbname][index_key][qbs][nprobe]["QPS"] = real_QPS
        dict_perf_merged[dbname][index_key][qbs][nprobe]["latency@50"] = latency_50 # in ms
        dict_perf_merged[dbname][index_key][qbs][nprobe]["latency@95"] = latency_95 # in ms
        dict_perf_merged[dbname][index_key][qbs][nprobe]["latency_distribution"] = list(tail_latency_ms)
        print("QPS: {}\t50% latency: {}\t95% latency: {}\t".format(real_QPS, latency_50, latency_95))

        with open(args.performance_dict_dir, 'wb') as f:
            pickle.dump(dict_perf_merged, f, protocol=4)