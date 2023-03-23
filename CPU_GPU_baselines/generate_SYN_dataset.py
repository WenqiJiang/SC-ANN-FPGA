"""
Generate dummy dataset like SIFT1B

The generation is based on the SIFT1B dataset distribution, because 
    random distribution can cause very low recall as PQ cannot encode data effectively
    
Since the data generation and topK computation is expensive, the script can by parallelized
    by running the script of different batches on several machines

Usage:
    python generate_SYN_dataset.py --dbname SYN1M
    python generate_SYN_dataset.py --dbname SYN10000M --start_batch 80 --end_batch 100
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import argparse 
import gc
import faiss

from multiprocessing.dummy import Pool as ThreadPool

parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SYN1M', help="SYN1M, SYN1000M, SYN10000M")
parser.add_argument('--start_batch', type=int, help="start_batch ID (inclusive), each batch = 100M vecs")
parser.add_argument('--end_batch', type=int, help="end_batch ID (exclusive)")

args = parser.parse_args()

dbname = args.dbname

assert dbname[:3] == 'SYN' 
assert dbname[-1] == 'M'
dbsize = int(dbname[3:-1]) # in million

dataset_dir = './SYN_dataset/{}'.format(dbname)
if not os.path.isdir(dataset_dir):
    print("%s does not exist, creating it" % dataset_dir)
    os.mkdir(dataset_dir)

def mmap_SYN_bvecs(fname, d=128):
    # vectors only, no vec_ID
    x = np.memmap(fname, dtype='uint8', mode='r')
    return x.reshape(-1, d)

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

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

gt_topK = 1000
D = 128 # same as SIFT
query_num = 10000

CHANGE_RATIO = 0.1 # select 10% of bigann elements to adjust

xb_bigann = mmap_bvecs('bigann/bigann_base.bvecs')
xq_bigann = mmap_bvecs('bigann/bigann_query.bvecs')
xt_bigann = mmap_bvecs('bigann/bigann_learn.bvecs')


if dbsize < 100:

    """ Generate the dataset """
    print("Generating base data...")
    base_vec_dir = os.path.join(dataset_dir, "base.bvecs")
    if os.path.exists(base_vec_dir):
        print("Base data already exist, skip...")
    else:
        num_vec = dbsize * 1000 * 1000 
        num_elements = num_vec * D
        mask = np.zeros(num_elements, dtype='uint8')
        sub_mask_elements = 1000 * D # shuffling an big matrix is too expensive, thus copy this small mask
        sub_mask = np.zeros(sub_mask_elements)
        sub_mask[:int(CHANGE_RATIO * sub_mask_elements)] = 1
        np.random.seed(seed=0)
        np.random.shuffle(sub_mask) 
        for i in range(int(num_elements / sub_mask_elements)):
            mask[i * sub_mask_elements: (i + 1) * sub_mask_elements] = sub_mask

        # small delta, only 0~5
        np.random.seed(seed=10000)
        delta = (np.random.randint(0, 5, size=num_elements, dtype='uint8') * mask).reshape(-1, D).astype('uint8')
        xb = xb_bigann[:num_vec] + delta

        xb.tofile(base_vec_dir)

        del mask
        del xb
        gc.collect()

    # 10% traning set
    print("Generating learning data...")
    learn_vec_dir = os.path.join(dataset_dir, "learn.bvecs")
    if os.path.exists(learn_vec_dir):
        print("Learning data already exist, skip...")
    else:
        np.random.seed(seed=1)
        num_vec = dbsize * 100 * 1000 # 10% of base vec
        xt = delta[:num_vec] + xt_bigann[:num_vec]
        xt.tofile(learn_vec_dir)
        del xt
        gc.collect()

    # 10000 query
    print("Generating query data...")
    query_vec_dir = os.path.join(dataset_dir, "query.bvecs")
    if os.path.exists(query_vec_dir):
        print("Query data already exist, skip...")
    else:
        np.random.seed(seed=2)
        xq = delta[:query_num] + xq_bigann
        xq.tofile(query_vec_dir)
        del xq
        gc.collect()

    """ compute ground truth """
    gt_dist_dir = os.path.join(dataset_dir, "dis_{}_by_{}_float32.fvecs".format(query_num, gt_topK))
    gt_id_dir = os.path.join(dataset_dir, "idx_{}_by_{}_int64.lvecs".format(query_num, gt_topK))

    if os.path.exists(gt_dist_dir) and os.path.exists(gt_id_dir):
        print("Ground truth data already exist, skip...")
    else:
        print("Computing ground truth...")
        xb = mmap_SYN_bvecs(base_vec_dir)
        xt = mmap_SYN_bvecs(learn_vec_dir)
        xq = mmap_SYN_bvecs(query_vec_dir)
        xq = xq.astype('float32').copy()

        """ add data to index """
        print("Adding data to Flat index...")
        index = faiss.index_factory(D, "IVF1,Flat")
        index.train(np.zeros(D, dtype='float32').reshape(1,D))
        # index = faiss.IndexFlatL2(D) # IndexFlat does not support add_with_ids

        i0 = 0
        t0 = time.time()
        add_batch_size = 100000
        for xs in matrix_slice_iterator(xb, add_batch_size):
            i1 = i0 + xs.shape[0]
            print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
            sys.stdout.flush()
            index.add_with_ids(xs, np.arange(i0, i1))
            i0 = i1
        
        """ search """
        print("Searching")
        gt_dist, gt_id = index.search(xq, gt_topK)

        gt_dist = np.array(gt_dist, dtype='float32').reshape(-1)
        gt_dist.tofile(gt_dist_dir)
        gt_id = np.array(gt_id, dtype='int64').reshape(-1)
        gt_id.tofile(gt_id_dir)

else: # dbsize >= 100
    assert dbsize % 100 == 0

    db_batches = int(dbsize / 100)
    dbsize_per_batch = 100 * 1000 * 1000

    """ Generate the dataset """

    # mask for training set & query
    num_vec =  100 * 1000 * 1000 # 100M at most
    num_elements = num_vec * D
    mask = np.zeros(num_elements, dtype='uint8')
    sub_mask_elements = 1000 * D # shuffling an big matrix is too expensive, thus copy this small mask
    sub_mask = np.zeros(sub_mask_elements)
    sub_mask[:int(CHANGE_RATIO * sub_mask_elements)] = 1
    np.random.seed(seed=0)
    np.random.shuffle(sub_mask) 
    for i in range(int(num_elements / sub_mask_elements)):
        mask[i * sub_mask_elements: (i + 1) * sub_mask_elements] = sub_mask
    # small delta, only 0~5
    np.random.seed(seed=10000)
    delta = (np.random.randint(0, 5, size=num_elements, dtype='uint8') * mask).reshape(-1, D).astype('uint8')


    # 10% traning set
    print("Generating learning data...")
    learn_vec_dir = os.path.join(dataset_dir, "learn.bvecs")
    if os.path.exists(learn_vec_dir):
        print("Learning data already exist, skip...")
    else:
        np.random.seed(seed=1)
        if dbsize * 100 * 1000 > 100 * 1000 * 1000: # 100M at most
            num_vec = 100 * 1000 * 1000
        else:
            num_vec = dbsize * 100 * 1000
        xt = delta[:num_vec] + xt_bigann[:num_vec]
        xt.tofile(learn_vec_dir)
        del xt
        gc.collect()

    # 10000 query
    print("Generating query data...")
    query_vec_dir = os.path.join(dataset_dir, "query.bvecs")
    if os.path.exists(query_vec_dir):
        print("Query data already exist, skip...")
    else:
        np.random.seed(seed=2)
        xq = delta[:query_num] + xq_bigann
        xq.tofile(query_vec_dir)
        del xq
        gc.collect()

    if args.start_batch is not None: # inclusive
        start_batch_id = args.start_batch
    else:
        start_batch_id = 0
    
    if args.end_batch is not None: # exclusive
        end_batch_id = args.end_batch
    else:
        end_batch_id = db_batches

    for batch_id in range(start_batch_id, end_batch_id):

        print("Generating base data... batch {} out of {}".format(batch_id, db_batches))
        base_vec_dir = os.path.join(dataset_dir, "base_{}_of_{}.bvecs".format(batch_id, db_batches))
        if os.path.exists(base_vec_dir):
            print("Training data already exist, skip...")
        else:
            num_elements = dbsize_per_batch * D
            mask = np.zeros(num_elements, dtype='uint8')
            sub_mask_elements = 1000 * D # shuffling an big matrix is too expensive, thus copy this small mask
            sub_mask = np.zeros(sub_mask_elements)
            sub_mask[:int(CHANGE_RATIO * sub_mask_elements)] = 1
            np.random.seed(seed=batch_id)
            np.random.shuffle(sub_mask) 
            for i in range(int(num_elements / sub_mask_elements)):
                mask[i * sub_mask_elements: (i + 1) * sub_mask_elements] = sub_mask

            # small delta, only 0~5
            np.random.seed(seed=10000 + batch_id)
            delta = (np.random.randint(0, 5, size=num_elements, dtype='uint8') * mask).reshape(-1, D).astype('uint8')
            xb = xb_bigann[:dbsize_per_batch] + delta

            xb.tofile(base_vec_dir)
            del mask
            del delta
            del xb
            gc.collect()

        """ compute ground truth """
        gt_dist_dir = os.path.join(dataset_dir, "dis_{}_by_{}_float32_{}_of_{}.fvecs".format(query_num, gt_topK, batch_id, db_batches))
        gt_id_dir = os.path.join(dataset_dir, "idx_{}_by_{}_int64_{}_of_{}.lvecs".format(query_num, gt_topK, batch_id, db_batches))

        if os.path.exists(gt_dist_dir) and os.path.exists(gt_id_dir):
            print("Ground truth data already exist, skip...")
        else:
            print("Computing ground truth...")
            xb = mmap_SYN_bvecs(base_vec_dir)
            xt = mmap_SYN_bvecs(learn_vec_dir)
            xq = mmap_SYN_bvecs(query_vec_dir)
            xq = xq.astype('float32').copy()

            """ add data to index """
            print("Adding data to Flat index...")
            index = faiss.index_factory(D, "IVF1,Flat")
            index.train(np.zeros(D, dtype='float32').reshape(1,D))
            # index = faiss.IndexFlatL2(D) # IndexFlat does not support add_with_ids
            
            i0 = batch_id * dbsize_per_batch
            t0 = time.time()
            add_batch_size = 100000
            for xs in matrix_slice_iterator(xb, add_batch_size):
                i1 = i0 + xs.shape[0]
                print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
                sys.stdout.flush()
                index.add_with_ids(xs, np.arange(i0, i1))
                i0 = i1
            
            """ search """
            print("Searching")
            gt_dist, gt_id = index.search(xq, gt_topK)

            gt_dist = np.array(gt_dist, dtype='float32').reshape(-1)
            gt_dist.tofile(gt_dist_dir)
            gt_id = np.array(gt_id, dtype='int64').reshape(-1)
            gt_id.tofile(gt_id_dir)


