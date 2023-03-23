"""
Usage e.g.: 
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key OPQ16,IVF65536,PQ16 --FPGA_num 4 --HBM_bank_num 16 --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_OPQ16,IVF65536,PQ16' --output_dir '/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_OPQ16,IVF65536,PQ16_4_FPGA_16_banks'
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

parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default=0, help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default=0, help="index parameters, e.g., IVF4096,PQ16 or OPQ16,IVF4096,PQ16")
parser.add_argument('--FPGA_num', type=int, default=4, help="partition the data to N FPGA")
parser.add_argument('--HBM_bank_num', type=int, default=10, help="partition the data to N HBM banks per FPGA")
parser.add_argument('--index_dir', type=str, default=0, help="the directory of the stored index, e.g., ../trained_CPU_indexes/bench_cpu_SIFT100M_IVF1024,PQ16")
parser.add_argument('--output_dir', type=str, default=0, help="where to output the generated FPGA data, e.g., /home/wejiang/saved_npy_data/FPGA_data_SIFT100M_IVF1024,PQ16_HBM_10_banks")

args = parser.parse_args()
dbname = args.dbname
index_key = args.index_key
FPGA_num = args.FPGA_num
HBM_bank_num = args.HBM_bank_num

nlist = None
PQ_bytes = None
OPQ_enable = None
index_array = index_key.split(",")
if len(index_array) == 2: # "IVF4096,PQ16" 
    OPQ_enable = False
    s = index_array[0]
    if s[:3]  == "IVF":
        nlist = int(s[3:])
    else:
        raise ValueError
    p = index_array[1]
    PQ_bytes = int(p[2:])
elif len(index_array) == 3: # "OPQ16,IVF4096,PQ16" 
    OPQ_enable = True
    s = index_array[1]
    if s[:3]  == "IVF":
        nlist = int(s[3:])
    else:
        raise ValueError
    p = index_array[2]
    PQ_bytes = int(p[2:])
else:
    raise ValueError

if args.index_dir:
    index_dir = args.index_dir
else:
    index_dir = '../trained_CPU_indexes/bench_cpu_{}_{}'.format(dbname, index_key)
if args.output_dir:
    output_parent_dir = args.output_dir
else:
    output_parent_dir = os.path.join('/home/wejiang/saved_npy_data/', 
        'FPGA_data_{}_{}_HBM_{}_banks'.format(dbname, index_key, HBM_bank_num))
if not os.path.exists(output_parent_dir):
    os.mkdir(output_parent_dir)
else:
    print("Warning: the output directory already exists, stop generating data...")
    exit(1)

output_dir_set = []
for i in range(FPGA_num):
    output_dir_set.append(os.path.join(output_parent_dir, 'FPGA_{}'.format(i)))
for output_dir in output_dir_set:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir) 

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


if not os.path.isdir(index_dir):
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

elif dbname == 'Deep1B':
    xb = mmap_fvecs('../deep1b/base.fvecs')
    xq = mmap_fvecs('../deep1b/deep1B_queries.fvecs')
    xt = mmap_fvecs('../deep1b/learn.fvecs')
    # deep1B's train is is outrageously big
    xt = xt[:10 * 1000 * 1000]
    gt = ivecs_read('../deep1b/deep1B_groundtruth.ivecs')

else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)

xq = np.array(xq, dtype=np.float32)
for output_dir in output_dir_set:
    xq.tofile(os.path.join(output_dir, "query_vectors_float32_{}_{}_raw".format(
        xq.shape[0], xq.shape[1])))

print("sizes: B %s Q %s T %s gt %s" % (
    xb.shape, xq.shape, xt.shape, gt.shape))

nq, d = xq.shape
nb, d = xb.shape
assert gt.shape[0] == nq


#################################################################
# Load Index
#################################################################

def get_populated_index():

    filename = "%s/%s_%s_populated.index" % (
        index_dir, dbname, index_key)

    if not os.path.exists(filename):
        raise("Index does not exist!")
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index


index = get_populated_index()

if OPQ_enable:
    """ Get OPQ Matrix """
    linear_trans = faiss.downcast_VectorTransform(index.chain.at(0))
    OPQ_mat = faiss.vector_to_array(linear_trans.A)
    OPQ_mat = OPQ_mat.reshape((128,128))
    OPQ_mat = np.array(OPQ_mat, dtype=np.float32)
    print("OPQ mat: {}\nshape: {}\n".format(OPQ_mat, OPQ_mat.shape))

""" Get IVF index (coarse quantizer) and product quantizer"""
if OPQ_enable:
    downcasted_index = faiss.downcast_index(index.index)

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
if OPQ_enable: 
    sub_cen = get_sub_quantizer_centroids(downcasted_index)
else:
    sub_cen = get_sub_quantizer_centroids(index)
print("==== Sub-quantizer ====\n{}\n\nshape:{}\n".format(sub_cen, sub_cen.shape))

# Get Coarse quantizer info
if OPQ_enable: 
    coarse_cen = get_coarse_quantizer_centroids(downcasted_index)
else:
    coarse_cen = get_coarse_quantizer_centroids(index)
print("==== Coarse-quantizer ====\n{}\n\nshape:{}\n".format(coarse_cen, coarse_cen.shape))

# Save the OPQ matrix, coarse quantizer, and the product quantizer
if OPQ_enable:
    OPQ_mat = np.array(OPQ_mat, dtype=np.float32)
    for output_dir in output_dir_set:
        OPQ_mat.tofile(os.path.join(output_dir, "OPQ_matrix_float32_{}_{}_raw".format(
            OPQ_mat.shape[0], OPQ_mat.shape[1]))) 
    print("Shape OPQ: {}\n".format(OPQ_mat.shape))
        
PQ_quantizer = np.array(sub_cen, dtype=np.float32)
coarse_cen = np.array(coarse_cen, dtype=np.float32)
# 16, 256, 8 -> (0,0,0:8) the first row of the subquantizer of the first sub-vector
print("Shape PQ quantizer: {}\n".format(PQ_quantizer.shape))
print("Shape coarse quantizer: {}\n".format(coarse_cen.shape))

for output_dir in output_dir_set:
    PQ_quantizer.tofile(os.path.join(output_dir, "product_quantizer_float32_{}_{}_{}_raw".format(
        PQ_quantizer.shape[0], PQ_quantizer.shape[1], PQ_quantizer.shape[2])))
    coarse_cen.tofile(os.path.join(output_dir, "vector_quantizer_float32_{}_{}_raw".format(
        coarse_cen.shape[0], coarse_cen.shape[1])))


""" Get contents in a single Voronoi Cell """
if OPQ_enable:
    invlists = downcasted_index.invlists
else:
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
    list_PQ_codes = list_PQ_codes.reshape(-1, invlists.code_size)
    return list_vec_ids, list_PQ_codes

# Example of using function "get_invlist"
list_id = 123
list_vec_ids, list_PQ_codes = get_invlist(invlists, list_id)
print("Contents of a single cluster:")
print("==== Vector IDs ====\n{}\n\nshape: {}\n".format(list_vec_ids, list_vec_ids.shape))
print("==== PQ codes ====\n{}\n\nshape: {}\n".format(list_PQ_codes, list_PQ_codes.shape))

def get_contents_to_HBM(invlists, cluster_id, FPGA_num=4, HBM_bank_num=int(21)):
    """
    For a single cluster (list), extract the contents in the format that HBM loads
      inputs:
        invlists: the Faiss index.invlists object
        cluster_id: e.g., 0~8191 for nlist=8192
        FPGA_num: number of FPGA
        HBM_bank_num: per FPGA, 21 for default, athough there are 32 banks on U280, 
                    we don't have enough hardware logic to load and compute at that rate
      outputs:
        HBM_bank_contents_list( content of 21 banks): a list of 21 element
            list (A) of list (B)
            list(A) has a length of FPGA_num
            in each list(B)
                each element is as byte object with a set of contents
                the size of the content is m * 64 bytes
                the contents includes (3 * (int32 vector ID) (16 byte PQ code)) + 4byte padding
        entries_per_bank_list: 
            list of FPGA_num int (identical ints), all HBM shares the same number of 512-bit items to scan
        last_valid_element_list: 
            list, [63, 20, -1, -1] given FPGA_num=4
            int from -1 to 63 (63 numbers in total given 21 HBM channels)
            some of the elements in the last row are paddings, which of them is the last non-padding (valid) 
            
      term:
        entry: a 512-bit entry containing 3 PQ codes
        vector: a 20-byte vector containing 4 byte vector ID + 16 byte PQ code
    """
    
    total_HBM_bank_num = FPGA_num * HBM_bank_num

    list_vec_ids, list_PQ_codes = get_invlist(invlists, cluster_id)
#     print("list_vec_ids", list_vec_ids.shape)
#     print("list_PQ_codes", list_PQ_codes.shape)
    num_vec = list_vec_ids.shape[0]
    assert list_vec_ids.shape[0] == list_PQ_codes.shape[0]
    
#     print("num_vec", num_vec)
    
    if num_vec % (total_HBM_bank_num * 3) == 0:
        # no padding
        entries_per_bank = num_vec / (total_HBM_bank_num * 3)
        last_valid_element = total_HBM_bank_num * 3 - 1
        num_vec_per_HBM = [int(num_vec / total_HBM_bank_num)] * total_HBM_bank_num
        num_pad_per_HBM = [0] * total_HBM_bank_num
    else:
        # with padding
        entries_per_bank = int(num_vec / (total_HBM_bank_num * 3)) + 1
        last_valid_element = num_vec % (total_HBM_bank_num * 3) - 1
        num_vec_per_HBM = []
        num_pad_per_HBM = []
        
        counted_banks = 0
        # bank with full valid elements
        for i in range(int((last_valid_element + 1) / 3)):
            num_vec_per_HBM += [entries_per_bank * 3]
            num_pad_per_HBM += [0]
        counted_banks += int((last_valid_element + 1) / 3)
        
        # (optional) bank with some valid elements and some padding in the last entry
        if (last_valid_element + 1) % 3 != 0:
            num_vec_per_HBM += [(entries_per_bank - 1) * 3 + (last_valid_element + 1) % 3]
            num_pad_per_HBM += [3 - (last_valid_element + 1) % 3]
            counted_banks += 1
        
        # (optional) bank with full padding in the last entry
        for i in range(total_HBM_bank_num - counted_banks):
            num_vec_per_HBM += [int((entries_per_bank - 1) * 3)]
            num_pad_per_HBM += [3]
            
    assert np.sum(np.array(num_vec_per_HBM)) == num_vec
    assert entries_per_bank * total_HBM_bank_num * 3 - np.sum(np.array(num_pad_per_HBM)) == num_vec
    
    HBM_bank_contents = []
    
    start = int(0)
    
    zero = int(0)
    empty_byte = zero.to_bytes(1, "little", signed=True)
    
#     print("num_vec_per_HBM:", num_vec_per_HBM)
#     print("num_pad_per_HBM:", num_pad_per_HBM)
    
    for i in range(total_HBM_bank_num):
        
        # add valid vectors first
        end = start + num_vec_per_HBM[i]
        vec_per_bank_count = 0
        byte_obj = bytes()
        
#         print(start, end)
        
        for vec_id_per_bank in range(start, end):
            
            # Vec ID = signed int
            vec_id = int(list_vec_ids[vec_id_per_bank])
            # Xilinx's ap int use little endian
            # Linux on X86 use little endian
            # https://serverfault.com/questions/163487/how-to-tell-if-a-linux-system-is-big-endian-or-little-endian
            byte_obj += vec_id.to_bytes(4, "little", signed=True)
            
            # PQ code = unsigned char
            PQ_codes = list_PQ_codes[vec_id_per_bank]
            for code in PQ_codes:
                code = int(code)
                # Xilinx's ap int use little endian
                byte_obj += code.to_bytes(1, "little", signed=False)
            
            vec_per_bank_count += 1
            if vec_per_bank_count % 3 == 0:
                byte_obj += empty_byte * 4
        
        start = end
        
        # then add paddings
        if num_pad_per_HBM[i] > 0:
            for pad_id in range(num_pad_per_HBM[i]):
                byte_obj += empty_byte * 20
            byte_obj += empty_byte * 4
        
        HBM_bank_contents += [byte_obj]
       
    for i in range(total_HBM_bank_num):
        assert len(HBM_bank_contents[i]) == len(HBM_bank_contents[0])
        assert len(HBM_bank_contents[i]) == 64 * entries_per_bank
    
    HBM_bank_contents_list = []
    for i in range(FPGA_num):
        HBM_bank_contents_list.append([])

    for i in range(FPGA_num):
        for j in range(HBM_bank_num):
            HBM_bank_contents_list[i].append(HBM_bank_contents[i * HBM_bank_num + j])
    
    entries_per_bank_list = []
    for i in range(FPGA_num):
        entries_per_bank_list.append(entries_per_bank)

    last_valid_element_list = []
    last_valid_element_count = last_valid_element
    for i in range(FPGA_num):
        if last_valid_element_count >= 63: 
            last_valid_element_count -= 63
            last_valid_element_list.append(63)
        elif last_valid_element_count < 63 and last_valid_element >= 0:
            last_valid_element_list.append(last_valid_element_count)
            last_valid_element_count = -1
        else:
            last_valid_element_list.append(-1)

    return HBM_bank_contents_list, entries_per_bank_list, last_valid_element_list

# Get HBM contents from all clusters
list_HBM_bank_contents_all_FPGA = [] # array of nlist * HBM_bank_num elements
list_entries_per_bank_all_FPGA = []
list_last_valid_element_all_FPGA = []

for FPGA_id in range(FPGA_num):
    list_HBM_bank_contents_all_FPGA.append([])
    list_entries_per_bank_all_FPGA.append([])
    list_last_valid_element_all_FPGA.append([])

for c in range(nlist):
    print("generating contents in cluster {}".format(c))
    HBM_bank_contents_list, entries_per_bank_list, last_valid_element_list = get_contents_to_HBM(invlists, c, FPGA_num, HBM_bank_num)

    for FPGA_id in range(FPGA_num):
        list_HBM_bank_contents_all_FPGA[FPGA_id] += HBM_bank_contents_list[FPGA_id]
        list_entries_per_bank_all_FPGA[FPGA_id] += [entries_per_bank_list[FPGA_id]]
        list_last_valid_element_all_FPGA[FPGA_id] += [last_valid_element_list[FPGA_id]]

for FPGA_id in range(FPGA_num):

    list_HBM_bank_contents = list_HBM_bank_contents_all_FPGA[FPGA_id]
    list_entries_per_bank = list_entries_per_bank_all_FPGA[FPGA_id]
    list_last_valid_element = list_last_valid_element_all_FPGA[FPGA_id]

    # Reorder list_HBM_bank_contents
    print(len(list_HBM_bank_contents))
    print("list_entries_per_bank:\n", list_entries_per_bank)
    print("list_last_valid_element:\n", list_last_valid_element)

    list_HBM_bank_contents_reordered = [] # put all contents of the same HBM bank together

    for b in range(HBM_bank_num):
        sub_list = []
        for c in range(nlist):
            sub_list += [list_HBM_bank_contents[c * HBM_bank_num + b]]
        print(len(sub_list))
        list_HBM_bank_contents_reordered += [sub_list]
        
    print("list_HBM_bank_contents_reordered:", len(list_HBM_bank_contents_reordered), len(list_HBM_bank_contents_reordered[0]))

    # Concatenate 
    HBM_bank_contents_all = [bytes()] * HBM_bank_num # contents of each bank
    for b in range(HBM_bank_num):
        HBM_bank_contents_all[b] = HBM_bank_contents_all[b].join(list_HBM_bank_contents_reordered[b])
        
    total_size = np.sum(np.array([len(h) for h in HBM_bank_contents_all]))
    print("HBM_bank_contents_all: shape: {}\tsize: {}".format(len(HBM_bank_contents_all), total_size))

    # Save HBM contents 
    for b in range(HBM_bank_num):
        assert len(HBM_bank_contents_all[b]) == len(HBM_bank_contents_all[0])

    for b in range(HBM_bank_num):
        with open (os.path.join(output_dir_set[FPGA_id], "HBM_bank_{}_raw".format(b)), 'wb') as f:
            f.write(HBM_bank_contents_all[b])

    # Save control contents

    #  The format of storing HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid: 
    #     8192 start_addr, then 8192 scanned_entries_every_cell, then 8192 last_valid_element
    #     int start_addr_LUT[nlist];
    #     int scanned_entries_every_cell_LUT[nlist];
    #     int last_valid_channel_LUT[nlist];  

    list_start_addr_every_cell = [0]
    for c in range(nlist - 1):
        list_start_addr_every_cell.append(list_start_addr_every_cell[c] + list_entries_per_bank[c])

    assert len(list_start_addr_every_cell) == len(list_entries_per_bank) and\
        len(list_start_addr_every_cell) == len(list_last_valid_element)

    print(list_start_addr_every_cell[-1])

    HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid = \
        list_start_addr_every_cell + list_entries_per_bank + list_last_valid_element

    HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid = np.array(
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid, dtype=np.int32)

    HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid.tofile(
        os.path.join(output_dir_set[FPGA_id], 'HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_3_by_{}_raw'.format(nlist)))
