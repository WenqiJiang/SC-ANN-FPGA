"""
OPQ not supported 

Usage e.g.: 
python extract_Enzian_U250_required_data.py --dbname SIFT100M --index_key IVF8192,PQ16 --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_IVF8192,PQ16/SIFT100M_IVF8192,PQ16_populated.index' --DDR_bank_num 4 --output_dir '/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT100M_IVF8192,PQ16'
python extract_Enzian_U250_required_data.py --dbname SBERT1000M --index_key IVF32768,PQ64 --index_dir '../trained_CPU_indexes/bench_cpu_SBERT1000M_IVF32768,PQ64_2shards/SBERT1000M_IVF32768,PQ64_populated_shard_0.index' --DDR_bank_num 4 --output_dir '/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT1000M_IVF32768,PQ64_2shards/shard_0'
python extract_Enzian_U250_required_data.py --dbname SBERT100M --index_key IVF65536,PQ64 --index_dir '../trained_CPU_indexes/bench_cpu_SBERT100M_IVF65536,PQ64/SBERT100M_IVF65536,PQ64_populated.index' --DDR_bank_num 3 --output_dir '/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT100M_IVF65536,PQ64_3_banks'
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import faiss
from multiprocessing.dummy import Pool as ThreadPool
from matplotlib import pyplot
import argparse 
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from datasets import read_deep_fbin, read_deep_ibin, mmap_bvecs_SBERT, mmap_bvecs_GNN


parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default=0, help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default=0, help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--index_dir', type=str, default=0, help="the directory of the stored index, e.g., ../trained_CPU_indexes/bench_cpu_SIFT100M_IVF1024,PQ16/SIFT100M_IVF1024,PQ16_populated.index")
parser.add_argument('--DDR_bank_num', type=int, default=4, help="number of banks used per FPGA")
parser.add_argument('--output_dir', type=str, default=0, help="where to output the generated FPGA data, e.g., /home/wejiang/Faiss_Enzian_U250_index/FPGA_data_SIFT100M_IVF1024,PQ16_DDR_10_banks")

args = parser.parse_args()
dbname = args.dbname
index_key = args.index_key
DDR_bank_num = args.DDR_bank_num

nlist = None
PQ_bytes = None
index_array = index_key.split(",")
if len(index_array) == 2: # "IVF4096,PQ16" 
    s = index_array[0]
    if s[:3]  == "IVF":
        nlist = int(s[3:])
    else:
        raise ValueError
    p = index_array[1]
    PQ_bytes = int(p[2:])
else:
    raise ValueError

if args.index_dir:
    index_dir = args.index_dir
else:
    index_dir = '../trained_CPU_indexes/bench_cpu_{}_{}/{}_{}_populated.index'.format(
        dbname, index_key, dbname, index_key)
if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = os.path.join('/home/wejiang/Faiss_Enzian_U250_index/', 
        'FPGA_data_{}_{}'.format(dbname, index_key))

# if os.path.exists(output_dir):
#     print("Warning: the output directory already exists, stop generating data...")
#     exit(1)

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


if not os.path.exists(index_dir):
    raise("%s does not exist")


#################################################################
# Prepare dataset
#################################################################


print("Preparing dataset", dbname)

if dbname.startswith('SIFT'):
    # SIFT1M to SIFT1000M
    dbsize = int(dbname[4:-1])
    xb = mmap_bvecs('../bigann/bigann_base.bvecs')
    xq = mmap_bvecs('../bigann/bigann_query.bvecs')
    xt = mmap_bvecs('../bigann/bigann_learn.bvecs')

    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]

    gt = ivecs_read('../bigann/gnd/idx_%dM.ivecs' % dbsize)

elif dbname.startswith('Deep'):

    assert dbname[:4] == 'Deep' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[4:-1]) # in million
    xb = read_deep_fbin('../deep1b/base.1B.fbin')[:dbsize * 1000 * 1000]
    xq = read_deep_fbin('../deep1b/query.public.10K.fbin')
    xt = read_deep_fbin('../deep1b/learn.350M.fbin')

    gt = read_deep_ibin('../deep1b/gt_idx_{}M.ibin'.format(dbsize))

    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)
elif dbname.startswith('SBERT'):
    # FB1M to FB1000M
    assert dbname[:5] == 'SBERT' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[5:-1]) # in million
    xb = mmap_bvecs_SBERT('../sbert/sbert3B.fvecs', num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_SBERT('../sbert/query_10K.fvecs', num_vec=10 * 1000)
    xt = xb
    
    gt = read_deep_ibin('../sbert/gt_idx_{}M.ibin'.format(dbsize), dtype='uint32')

    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    query_num = xq.shape[0]
    print('query shape: ', xq.shape)

elif dbname.startswith('GNN'):
    # FB1M to FB1000M
    assert dbname[:3] == 'GNN' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[3:-1]) # in million
    xb = mmap_bvecs_GNN('../MariusGNN/embeddings.bin', num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_GNN('../MariusGNN/query_10K.fvecs', num_vec=10 * 1000)
    xt = xb

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
   
    gt = read_deep_ibin('../MariusGNN/gt_idx_{}M.ibin'.format(dbsize), dtype='uint32') 
    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)

    nb, D = xb.shape # same as SIFT
    query_num = xq.shape[0]
    print('query shape: ', xq.shape)
else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)

xq = np.array(xq, dtype=np.float32)

print("sizes: B %s Q %s T %s gt %s" % (
    xb.shape, xq.shape, xt.shape, gt.shape))

nq, d = xq.shape
nb, d = xb.shape
assert gt.shape[0] == nq


#################################################################
# Load Index
#################################################################

def get_populated_index():

    # filename = "%s/%s_%s_populated.index" % (
    #     index_dir, dbname, index_key)
    filename = index_dir

    if not os.path.exists(filename):
        raise("Index does not exist!")
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index


index = get_populated_index()

def get_sub_quantizer_centroids(index):
    """
    return the sub-quantizer centroids, 
    shape = (m, 256, d / m)
    e.g., d=128, m=16 -> (16, 256, 8)
    """
    pq = index.pq
    cen = faiss.vector_to_array(pq.centroids)
    cen = cen.reshape(pq.M, pq.ksub, pq.dsub)
    
    return cen

def get_coarse_quantizer_centroids(index):
    """
    return the coarse-grained quantizer centroids,
    shape = (nlist, d),
    e.g., nlist=1024, d=128 -> (1024, 128)
    """
    coarse_quantizer = faiss.downcast_index(index.quantizer)
    coarse_cen = faiss.vector_to_array(coarse_quantizer.xb)

    coarse_cen = coarse_cen.reshape(coarse_quantizer.ntotal, coarse_quantizer.d)
    return coarse_cen

# Get Sub quantizer info
sub_cen = get_sub_quantizer_centroids(index)
print("==== Sub-quantizer ====\n{}\n\nshape:{}\n".format(sub_cen, sub_cen.shape))

# Get Coarse quantizer info
coarse_cen = get_coarse_quantizer_centroids(index)
print("==== Coarse-quantizer ====\n{}\n\nshape:{}\n".format(coarse_cen, coarse_cen.shape))

PQ_quantizer = np.array(sub_cen, dtype=np.float32)
coarse_cen = np.array(coarse_cen, dtype=np.float32)
# 16, 256, 8 -> (0,0,0:8) the first row of the subquantizer of the first sub-vector
print("Shape PQ quantizer: {}\n".format(PQ_quantizer.shape))
print("Shape coarse quantizer: {}\n".format(coarse_cen.shape))


""" Get contents in a single Voronoi Cell """
invlists = index.invlists

def get_invlist(invlists, l):
    """ 
    returns the (vector IDs set, PQ cose set) of list ID "l"
    list_ids: (#vec_in_list, ), e.g., #vec_in_list=10 -> (10, )
    list_codes: (#vec_in_list, m), e.g., #vec_in_list=10, m=16 -> (10, 16)
    
    That the data is *NOT* copied: if the inverted index is deallocated or changes, accessing the array may crash.
    To avoid this, just clone the output arrays on output. 
    """
    ls = invlists.list_size(l)
    list_vec_ids = faiss.rev_swig_ptr(invlists.get_ids(l), ls)
    list_PQ_codes = faiss.rev_swig_ptr(invlists.get_codes(l), ls * invlists.code_size)
    # list_PQ_codes = list_PQ_codes.reshape(-1, invlists.code_size)
    return list_vec_ids, list_PQ_codes

# Example of using function "get_invlist"
list_id = 123
list_vec_ids, list_PQ_codes = get_invlist(invlists, list_id)
print("Contents of a single cluster:")
print("==== Vector IDs ====\n{}\n\nshape: {}\n".format(list_vec_ids, list_vec_ids.shape))
print("==== PQ codes ====\n{}\n\nshape: {}\n".format(list_PQ_codes, list_PQ_codes.shape))

def get_contents_to_DDR(invlists, cluster_id, DDR_bank_num=4):
    """
    For a single cluster (list), extract the contents in the format that DDR loads
      inputs:
        invlists: the Faiss index.invlists object
        cluster_id: e.g., 0~8191 for nlist=8192
        DDR_bank_num: 4 for default, athough there are 4 banks on U250 / Enzian 
      outputs:
        DDR_bank_contents_PQ: PQ codes, split in 4 banks, padded
        DDR_bank_contents_vec_ids: vec IDs respective to the PQ codes, split in 4 banks, padded
        num_vec: int, number of vectors in this Voronoi cell 
    """

    assert 64 % PQ_bytes == 0
    vec_per_channel_entry = int(64 / PQ_bytes)
    vectors_per_entry = int(DDR_bank_num * 64 / PQ_bytes) # number of vectors per N DDR channels
    
    list_vec_ids, list_PQ_codes = get_invlist(invlists, cluster_id)
    list_vec_ids = np.array(list_vec_ids, dtype=np.int64)
    list_PQ_codes = np.array(list_PQ_codes, dtype=np.uint8)
    
    # print("list_vec_ids", list_vec_ids.shape)
    # print("list_PQ_codes", list_PQ_codes.shape)
    num_vec = list_vec_ids.shape[0]
    assert list_vec_ids.shape[0] == list_PQ_codes.shape[0] / PQ_bytes
    if num_vec == 0:
        DDR_bank_contents_PQ = [bytes()] * DDR_bank_num
        DDR_bank_contents_vec_ids = [bytes()] * DDR_bank_num
        return DDR_bank_contents_PQ, DDR_bank_contents_vec_ids, 0, 0, 0
    
    size_vec_ID = 8 # faiss use 8-byte int

    print("num_vec", num_vec)
    
    DDR_bank_contents_PQ = []
    DDR_bank_contents_vec_ids = []

    # convert to bytes: bytes() must be in range(0, 256)
    if num_vec % vectors_per_entry == 0 and num_vec > 0:
        # no padding
        num_vec_per_bank = int(num_vec / DDR_bank_num)
        for bank_id in range(DDR_bank_num):
            DDR_bank_contents_PQ.append(bytes(list_PQ_codes[bank_id * PQ_bytes * num_vec_per_bank: (bank_id + 1) * PQ_bytes * num_vec_per_bank].copy()))
            DDR_bank_contents_vec_ids.append(bytes(list_vec_ids[bank_id * num_vec_per_bank: (bank_id + 1) * num_vec_per_bank].copy()))

    else:
        # with padding

        # first handling the non-pad elements
        num_entries_lower_bound = int(np.floor(num_vec / vectors_per_entry))
        num_vec_per_bank_lower_bound = num_entries_lower_bound * vec_per_channel_entry

        # print('num_entries_lower_bound', num_entries_lower_bound)
        # print('num_vec_per_bank_lower_bound', num_vec_per_bank_lower_bound)

        for bank_id in range(DDR_bank_num):
            DDR_bank_contents_PQ.append(bytes(np.array(list_PQ_codes[bank_id * PQ_bytes * num_vec_per_bank_lower_bound: (bank_id + 1) * PQ_bytes * num_vec_per_bank_lower_bound], dtype=np.uint8)))
            DDR_bank_contents_vec_ids.append(bytes(list_vec_ids[bank_id * num_vec_per_bank_lower_bound: (bank_id + 1) * num_vec_per_bank_lower_bound].copy()))

        # then handling the last row
        zero = int(0)
        empty_byte = zero.to_bytes(1, "little", signed=True)

        last_PQ_codes = list_PQ_codes[DDR_bank_num * PQ_bytes * num_vec_per_bank_lower_bound: ].copy()
        last_vec_ids = list_vec_ids[DDR_bank_num * num_vec_per_bank_lower_bound: ].copy()

        # print(len(last_vec_ids) / vec_per_channel_entry) 
        channel_mix_id = int(np.ceil(len(last_vec_ids) / vec_per_channel_entry - 1))  # full + empty

        # all full
        for bank_id in range(channel_mix_id):
            DDR_bank_contents_PQ[bank_id] += bytes(last_PQ_codes[bank_id * vec_per_channel_entry * PQ_bytes : (bank_id + 1) * vec_per_channel_entry * PQ_bytes].copy())
            DDR_bank_contents_vec_ids[bank_id] += bytes(last_vec_ids[bank_id * vec_per_channel_entry: (bank_id + 1) * vec_per_channel_entry].copy())

        # full + empty
        bank_id = channel_mix_id

        DDR_bank_contents_PQ[bank_id] += bytes(last_PQ_codes[bank_id * vec_per_channel_entry * PQ_bytes: ].copy())
        DDR_bank_contents_vec_ids[bank_id] += bytes(last_vec_ids[bank_id * vec_per_channel_entry: ].copy())
        
        rest_vec = (bank_id + 1) * vec_per_channel_entry - len(last_vec_ids)
        DDR_bank_contents_PQ[bank_id] +=  empty_byte * rest_vec * PQ_bytes
        DDR_bank_contents_vec_ids[bank_id] += empty_byte * rest_vec * size_vec_ID

        # empty
        for bank_id in range(channel_mix_id + 1, DDR_bank_num):
            DDR_bank_contents_PQ[bank_id] +=  empty_byte * vec_per_channel_entry * PQ_bytes
            DDR_bank_contents_vec_ids[bank_id] += empty_byte * vec_per_channel_entry * size_vec_ID

        # if len(last_vec_ids) / vec_per_channel_entry <= 1: # only bank 0 has real contents
        #     rest_vec = vec_per_channel_entry - len(last_vec_ids)
        #     bank_0_PQ += bytes(last_PQ_codes)
        #     bank_0_vec_ids += bytes(last_vec_ids)
        #     bank_0_PQ +=  empty_byte * rest_vec * PQ_bytes
        #     bank_0_vec_ids += empty_byte * rest_vec * size_vec_ID

        #     bank_1_PQ += empty_byte * vec_per_channel_entry * PQ_bytes
        #     bank_2_PQ += empty_byte * vec_per_channel_entry * PQ_bytes
        #     bank_3_PQ += empty_byte * vec_per_channel_entry * PQ_bytes

        #     bank_1_vec_ids += empty_byte * vec_per_channel_entry * size_vec_ID
        #     bank_2_vec_ids += empty_byte * vec_per_channel_entry * size_vec_ID
        #     bank_3_vec_ids += empty_byte * vec_per_channel_entry * size_vec_ID

        # elif len(last_vec_ids) / vec_per_channel_entry <= 2: # only bank 0+1 has real contents

        #     bank_0_PQ += bytes(last_PQ_codes[0: vec_per_channel_entry * PQ_bytes].copy())
        #     bank_0_vec_ids += bytes(last_vec_ids[0: vec_per_channel_entry].copy())

        #     bank_1_PQ += bytes(last_PQ_codes[vec_per_channel_entry * PQ_bytes: ].copy())
        #     bank_1_vec_ids += bytes(last_vec_ids[vec_per_channel_entry: ].copy())

        #     rest_vec = 2 * vec_per_channel_entry - len(last_vec_ids)
        #     bank_1_PQ +=  empty_byte * rest_vec * PQ_bytes
        #     bank_1_vec_ids += empty_byte * rest_vec * size_vec_ID

        #     bank_2_PQ += empty_byte * vec_per_channel_entry * PQ_bytes
        #     bank_3_PQ += empty_byte * vec_per_channel_entry * PQ_bytes

        #     bank_2_vec_ids += empty_byte * vec_per_channel_entry * size_vec_ID
        #     bank_3_vec_ids += empty_byte * vec_per_channel_entry * size_vec_ID

        # elif len(last_vec_ids) / vec_per_channel_entry <= 3: # only bank 0+1+2 has real contents

        #     bank_0_PQ += bytes(last_PQ_codes[0: vec_per_channel_entry * PQ_bytes].copy())
        #     bank_0_vec_ids += bytes(last_vec_ids[0: vec_per_channel_entry].copy())

        #     bank_1_PQ += bytes(last_PQ_codes[vec_per_channel_entry * PQ_bytes: 2 * vec_per_channel_entry * PQ_bytes].copy())
        #     bank_1_vec_ids += bytes(last_vec_ids[vec_per_channel_entry: 2 * vec_per_channel_entry].copy())

        #     bank_2_PQ += bytes(last_PQ_codes[2 * vec_per_channel_entry * PQ_bytes : ].copy())
        #     bank_2_vec_ids += bytes(last_vec_ids[2 * vec_per_channel_entry : ].copy())

        #     rest_vec = 3 * vec_per_channel_entry - len(last_vec_ids)
        #     bank_2_PQ +=  empty_byte * rest_vec * PQ_bytes
        #     bank_2_vec_ids += empty_byte * rest_vec * size_vec_ID

        #     bank_3_PQ += empty_byte * vec_per_channel_entry * PQ_bytes
        #     bank_3_vec_ids += empty_byte * vec_per_channel_entry * size_vec_ID

        # elif len(last_vec_ids) / vec_per_channel_entry <= 4:

        #     bank_0_PQ += bytes(last_PQ_codes[0: vec_per_channel_entry * PQ_bytes].copy())
        #     bank_0_vec_ids += bytes(last_vec_ids[0: vec_per_channel_entry].copy())

        #     bank_1_PQ += bytes(last_PQ_codes[vec_per_channel_entry * PQ_bytes: 2 * vec_per_channel_entry * PQ_bytes].copy())
        #     bank_1_vec_ids += bytes(last_vec_ids[vec_per_channel_entry: 2 * vec_per_channel_entry].copy())

        #     bank_2_PQ += bytes(last_PQ_codes[2 * vec_per_channel_entry * PQ_bytes : 3 * vec_per_channel_entry * PQ_bytes].copy())
        #     bank_2_vec_ids += bytes(last_vec_ids[2 * vec_per_channel_entry : 3 * vec_per_channel_entry].copy())

        #     bank_3_PQ += bytes(last_PQ_codes[3 * vec_per_channel_entry * PQ_bytes : ].copy())
        #     bank_3_vec_ids += bytes(last_vec_ids[3 * vec_per_channel_entry : ].copy())

        #     rest_vec = 4 * vec_per_channel_entry - len(last_vec_ids)
        #     bank_3_PQ +=  empty_byte * rest_vec * PQ_bytes
        #     bank_3_vec_ids += empty_byte * rest_vec * size_vec_ID

        # else:
        #     raise ValueError

    # print(len(DDR_bank_contents_PQ[0]), len(DDR_bank_contents_PQ[1]), len(DDR_bank_contents_PQ[2]), len(DDR_bank_contents_PQ[3]))
    # print(len(DDR_bank_contents_vec_ids[0]), len(DDR_bank_contents_vec_ids[1]), len(DDR_bank_contents_vec_ids[2]), len(DDR_bank_contents_vec_ids[3]))

    # assert len
    for i in range(DDR_bank_num):
        assert len(DDR_bank_contents_PQ[i]) % 64 == 0
        assert len(DDR_bank_contents_PQ[i]) == len(DDR_bank_contents_PQ[0])
        assert len(DDR_bank_contents_vec_ids[i]) == len(DDR_bank_contents_vec_ids[0])
        # print(len(DDR_bank_contents_PQ[i]), len(DDR_bank_contents_vec_ids[i]))
        assert len(DDR_bank_contents_PQ[i]) / len(DDR_bank_contents_vec_ids[i]) == PQ_bytes / size_vec_ID

    num_vec_in_cell = num_vec
    num_entry_PQ_codes = len(DDR_bank_contents_PQ[0]) / 64 # number of 64-byte entry per bank
    num_entry_vec_ID = len(DDR_bank_contents_vec_ids[0]) / size_vec_ID # number of 8-byte int vec ID per bank
    
    return DDR_bank_contents_PQ, DDR_bank_contents_vec_ids, num_vec_in_cell, int(num_entry_PQ_codes), int(num_entry_vec_ID)

# Get DDR contents from all clusters

list_num_vecs = [] # number of vectors per nlist, array of nlist elements
list_num_entry_PQ_codes = []
list_num_entry_vec_ID = []

list_DDR_bank_contents_PQ = []
list_DDR_bank_contents_vec_ids = []

for bank_id in range(DDR_bank_num):
    list_DDR_bank_contents_PQ.append([])
    list_DDR_bank_contents_vec_ids.append([])


for c in range(nlist):
    print("generating contents in cluster {}".format(c))
    DDR_bank_contents_PQ, DDR_bank_contents_vec_ids, num_vec_in_cell, num_entry_PQ_codes, num_entry_vec_ID = get_contents_to_DDR(invlists, c, DDR_bank_num)
    
    for bank_id in range(DDR_bank_num):
        list_DDR_bank_contents_PQ[bank_id].append(DDR_bank_contents_PQ[bank_id])
        list_DDR_bank_contents_vec_ids[bank_id].append(DDR_bank_contents_vec_ids[bank_id])

    list_num_vecs.append(num_vec_in_cell)
    list_num_entry_PQ_codes.append(num_entry_PQ_codes)
    list_num_entry_vec_ID.append(num_entry_vec_ID)
    
print("Concatenating data...")
DDR_bank_contents_PQ = []
DDR_bank_contents_vec_ids = []


for bank_id in range(DDR_bank_num):
    DDR_bank_contents_PQ.append(bytes(b"".join(list_DDR_bank_contents_PQ[bank_id]))) # array of nlist elements
    DDR_bank_contents_vec_ids.append(bytes(b"".join(list_DDR_bank_contents_vec_ids[bank_id]))) # array of nlist elements


# Reorder list_DDR_bank_contents_PQ
print("list_num_vecs:\n", list_num_vecs)
print("bytes of PQ codes per channel: ", len(DDR_bank_contents_PQ[0]))
print("bytes of vec IDs per channel: ", len(DDR_bank_contents_vec_ids[0]))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

xq.tofile(os.path.join(output_dir, "query_vectors_float32_{}_{}_raw".format(
    xq.shape[0], xq.shape[1])))
PQ_quantizer.tofile(os.path.join(output_dir, "product_quantizer_float32_{}_{}_{}_raw".format(
    PQ_quantizer.shape[0], PQ_quantizer.shape[1], PQ_quantizer.shape[2])))
coarse_cen.tofile(os.path.join(output_dir, "vector_quantizer_float32_{}_{}_raw".format(
    coarse_cen.shape[0], coarse_cen.shape[1])))

# Save DDR contents 
for bank_id in range(DDR_bank_num):
    with open (os.path.join(output_dir, "DDR_bank_{}_PQ_raw".format(bank_id)), 'wb') as f:
        f.write(DDR_bank_contents_PQ[bank_id])
    with open (os.path.join(output_dir, "DDR_bank_{}_vec_ID_raw".format(bank_id)), 'wb') as f:
        f.write(DDR_bank_contents_vec_ids[bank_id])


# Save control contents
#  The format of storing nlist_init: 
# // int* nlist_PQ_codes_start_addr, -> start entry ID of a cell (for uint_512 type codes), same in 4 channels
#    width = 64 * 4 = 256 byte per entry
# // int* nlist_vec_ID_start_addr, -> start entry ID of a cell (for int format vec ID), same in 4 channels
#    here, DRAM store int directly, width = 4 ints per entry
# // int* nlist_num_vecs -> num vec per cell

list_nlist_PQ_codes_start_addr = [0]
list_nlist_vec_ID_start_addr = [0]

for c in range(1, nlist):
    num_vec = list_num_vecs[c]

    # vec_per_channel_entry = int(64 / PQ_bytes)
    # vectors_per_entry = int(4 * 64 / PQ_bytes) # number of vectors per 4 DDR channels
    # num_entries_upper_bound = int(np.ceil(num_vec / vectors_per_entry))

    PQ_codes_start_addr = int(list_nlist_PQ_codes_start_addr[-1] + list_num_entry_PQ_codes[c - 1])
    vec_ID_start_addr = int(list_nlist_vec_ID_start_addr[-1] + list_num_entry_vec_ID[c - 1])

    list_nlist_PQ_codes_start_addr.append(PQ_codes_start_addr)
    list_nlist_vec_ID_start_addr.append(vec_ID_start_addr)


size_vec_ID = 8 # faiss use 8-byte int
assert len(list_nlist_PQ_codes_start_addr) == len(list_num_vecs) and\
    len(list_nlist_vec_ID_start_addr) == len(list_num_vecs)
assert (list_nlist_vec_ID_start_addr[-1] + list_num_entry_vec_ID[-1]) * size_vec_ID == len(DDR_bank_contents_vec_ids[0])
assert (list_nlist_PQ_codes_start_addr[-1] + list_num_entry_PQ_codes[-1]) * 64 == len(DDR_bank_contents_PQ[0]) 

list_nlist_PQ_codes_start_addr = np.array(list_nlist_PQ_codes_start_addr, dtype=np.int32)
list_nlist_vec_ID_start_addr = np.array(list_nlist_vec_ID_start_addr, dtype=np.int32)
list_num_vecs = np.array(list_num_vecs, dtype=np.int32)

list_nlist_PQ_codes_start_addr.tofile(
    os.path.join(output_dir, 'nlist_PQ_codes_start_addr'))
list_nlist_vec_ID_start_addr.tofile(
    os.path.join(output_dir, 'nlist_vec_ID_start_addr'))
list_num_vecs.tofile(
    os.path.join(output_dir, 'nlist_num_vecs'))