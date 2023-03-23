"""
This script is used to compute the ground truth for datasets, e.g., Deep100M

Usage:
    python compute_ground_truth.py --dbname Deep1M 
    python compute_ground_truth.py --dbname GNN1M 
    python compute_ground_truth.py --dbname SBERT_NQ1M  
    python compute_ground_truth.py --dbname SBERT500M  # compute all results linearly, or merge results after individual batch search
    python compute_ground_truth.py --dbname SBERT500M --batch_ID 1 # only compute 100M dataset of batch_ID=1

For example, compute the gt on 5 servers in parallel for SBERT500M:
    run these commands on 5 servers, wait for finish:
        python compute_ground_truth.py --dbname SBERT500M --batch_ID 0
        python compute_ground_truth.py --dbname SBERT500M --batch_ID 1
        python compute_ground_truth.py --dbname SBERT500M --batch_ID 2
        python compute_ground_truth.py --dbname SBERT500M --batch_ID 3
        python compute_ground_truth.py --dbname SBERT500M --batch_ID 4
    then run the final command to merge the results:
        python compute_ground_truth.py --dbname SBERT500M 
"""

from __future__ import print_function
from email.errors import InvalidBase64PaddingDefect
import os
import sys
import time
from tracemalloc import start
import numpy as np
import re
import argparse 
import gc
import faiss
from datasets import read_deep_fbin, read_deep_ibin, write_deep_fbin, \
    write_deep_ibin, mmap_bvecs_FB, mmap_bvecs_SBERT, mmap_bvecs_GNN, mmap_bvecs_Journal

from multiprocessing.dummy import Pool as ThreadPool

parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='Deep1M', help='Deep1M, Deep1000M, FB1000M')
parser.add_argument('--batch_ID', type=int, default=None, help='generated 100M results, only go for batch_ID, e.g., batch_ID=1 -> 100~200M')

args = parser.parse_args()

dbname = args.dbname
batch_ID = args.batch_ID

ID_dtype = 'uint32' # works for dataset < 2B vectors
gt_topK = 1000


if dbname.startswith('Deep'):
    # Deep1M to Deep1000M
    dataset_dir = './deep1b'
    assert dbname[:4] == 'Deep' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[4:-1]) # in million
    xb = read_deep_fbin('deep1b/base.1B.fbin')[:dbsize * 1000 * 1000]
    xq = read_deep_fbin('deep1b/query.public.10K.fbin')

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
    
    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    nb, D = xb.shape # same as SIFT
    query_num = xq.shape[0]
    print('query shape: ', xq.shape)
elif dbname.startswith('FB'):
    # FB1M to FB1000M
    dataset_dir = './Facebook_SimSearchNet++'
    assert dbname[:2] == 'FB' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[2:-1]) # in million
    xb = mmap_bvecs_FB('Facebook_SimSearchNet++/FB_ssnpp_database.u8bin', num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_FB('Facebook_SimSearchNet++/FB_ssnpp_public_queries.u8bin', num_vec=10 * 1000)

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
    
    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    nb, D = xb.shape # same as SIFT
    query_num = xq.shape[0]
    print('query shape: ', xq.shape)
elif dbname.startswith('SBERT') and not dbname.startswith('SBERT_NQ'):
    # FB1M to FB1000M
    dataset_dir = './sbert'
    assert dbname[:5] == 'SBERT' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[5:-1]) # in million
    xb = mmap_bvecs_SBERT('sbert/sbert3B.fvecs', num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_SBERT('sbert/query_10K.fvecs', num_vec=10 * 1000)

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
    
    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    nb, D = xb.shape # same as SIFT
    query_num = xq.shape[0]
    print('query shape: ', xq.shape)

elif dbname.startswith('SBERT_NQ'):
    # FB1M to FB1000M
    dataset_dir = './sbert_natural_question_with_frequency'
    assert dbname[:8] == 'SBERT_NQ' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[8:-1]) # in million
    xb = mmap_bvecs_SBERT('sbert/sbert3B.fvecs', num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_SBERT('sbert_natural_question_with_frequency/sbert_natural_question_with_frequency_10K.fvecs', num_vec=10 * 1000)

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
    
    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    nb, D = xb.shape # same as SIFT
    query_num = xq.shape[0]
    print('query shape: ', xq.shape)

elif dbname.startswith('GNN'):
    # FB1M to FB1000M
    dataset_dir = './MariusGNN/'
    assert dbname[:3] == 'GNN' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[3:-1]) # in million
    xb = mmap_bvecs_GNN('MariusGNN/embeddings.bin', num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_GNN('MariusGNN/query_10K.fvecs', num_vec=10 * 1000)

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
    
    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    nb, D = xb.shape # same as SIFT
    query_num = xq.shape[0]
    print('query shape: ', xq.shape)

elif dbname.startswith('Journal'): # Roger's 4M sample
    # FB1M to FB1000M
    dataset_dir = './Journal/'
    assert dbname[:7] == 'Journal' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[7:-1]) # in million
    xb = mmap_bvecs_Journal('Journal/livejournal_embeddings.bin', num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_Journal('Journal/query_10K.fvecs', num_vec=10 * 1000)

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
    
    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    nb, D = xb.shape # same as SIFT
    query_num = xq.shape[0]
    print('query shape: ', xq.shape)

else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)

nq, d = xq.shape
nb, d = xb.shape
print("nb: {}\tnq: {}\td: {}".format(nb, nq, d))

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


""" compute ground truth """
gt_dist_dir = os.path.join(dataset_dir, "gt_dis_{}M.fbin".format(dbsize))
gt_id_dir = os.path.join(dataset_dir, "gt_idx_{}M.ibin".format(dbsize))

if os.path.exists(gt_dist_dir) or os.path.exists(gt_id_dir):
    print("Ground truth data already exist, skip...")
else:
    print("Computing ground truth...")

    size_per_iter = 100 # no more than 100M vectors per iteration to prevent thrushing
    db_batches = int(np.ceil(dbsize / size_per_iter))


    gt_ID_all_list = []
    gt_dist_all_list = []
    # Build index & search
    for iter_id in range(db_batches):
        if batch_ID is not None and iter_id != batch_ID:
            print("Skip batch {} of 100M vec".format(iter_id))
            continue

        """ add data to index """
        start_ID = iter_id * size_per_iter * int(1e6)
        end_ID = int(np.amin([dbsize * int(1e6), (iter_id + 1) * size_per_iter * int(1e6)]))
        print("Adding data to Flat index...\tIter {}\tVec from {} to {}".format(
            iter_id, start_ID, end_ID))

        partial_gt_dist_dir = gt_dist_dir + '_partial_{}_to_{}'.format(start_ID, end_ID)
        partial_gt_id_dir = gt_id_dir + '_partial_{}_to_{}'.format(start_ID, end_ID)
        if os.path.exists(partial_gt_dist_dir) or os.path.exists(partial_gt_id_dir):
            print("Ground truth data already exist, skip...")
            partial_gt_id = read_deep_ibin(partial_gt_id_dir, dtype=ID_dtype)
            partial_gt_dist = read_deep_fbin(partial_gt_dist_dir)
        else:
            index = faiss.index_factory(D, "IVF1,Flat")
            index.train(np.zeros(D, dtype='float32').reshape(1,D))
            # index = faiss.IndexFlatL2(D) # IndexFlat does not support add_with_ids

            xb_partial = xb[start_ID: end_ID]
            t0 = time.time()
            i0 = start_ID
            add_batch_size = 100000
            for xs in matrix_slice_iterator(xb_partial, add_batch_size):
                i1 = i0 + xs.shape[0]
                print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
                sys.stdout.flush()
                index.add_with_ids(xs, np.arange(i0, i1))
                i0 = i1
            
            """ search """
            print("Searching")
            partial_gt_dist, partial_gt_id = index.search(xq, gt_topK)

            partial_gt_dist = np.array(partial_gt_dist, dtype='float32')
            partial_gt_id = np.array(partial_gt_id, dtype='int32')
            print("dist shape: ", partial_gt_dist.shape)
            print("ID shape", partial_gt_id.shape)

            # save intermediate results
            write_deep_fbin(partial_gt_dist_dir, partial_gt_dist)
            write_deep_ibin(partial_gt_id_dir, partial_gt_id, dtype=ID_dtype)

        gt_ID_all_list.append(partial_gt_id)
        gt_dist_all_list.append(partial_gt_dist)
        assert gt_ID_all_list[-1].shape == (query_num, gt_topK)
        assert gt_dist_all_list[-1].shape == (query_num, gt_topK)

    if batch_ID is None: # merge when sequential scan
        # merge results
        gt_ID_merged = np.zeros((query_num, gt_topK), dtype=ID_dtype)
        gt_dist_merged = np.zeros((query_num, gt_topK), dtype='float32')
        for query_id in range(query_num):
            print("query ID: {}".format(query_id))
            ID_list = []
            dist_list = []
            for batch_id in range(db_batches):
                ID_list.append(gt_ID_all_list[batch_id][query_id])
                dist_list.append(gt_dist_all_list[batch_id][query_id])
            ID_array = np.concatenate(ID_list)
            dist_array = np.concatenate(dist_list)
            
            topK_indices = np.argsort(dist_array)[:gt_topK]
            selected_ID = np.take(ID_array, topK_indices)
            selected_dist = np.take(dist_array, topK_indices)

            gt_ID_merged[query_id] = selected_ID
            gt_dist_merged[query_id] = selected_dist

        write_deep_fbin(gt_dist_dir, gt_dist_merged, )
        write_deep_ibin(gt_id_dir, gt_ID_merged, dtype=ID_dtype)

