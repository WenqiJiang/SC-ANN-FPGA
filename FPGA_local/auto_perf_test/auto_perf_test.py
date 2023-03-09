# This script can automatically evaluate the FPGA performance by loading a set of parameter stored in a dict,
#   dict format: d[dbname][index_key][topK][recall_goal] = min_nprobe
# user can either specify a single recall goal, or evaluate all the recall goal stored in the dictionary
# The output performance dict format:
#   p[dbname][index_key][topK][recall_goal] = QPS
# Note! The bitstream folder should contain xrt.ini that enables profiling storing profile_summary.csv
# Copy this script to the FPGA bitstream folder to use it
# Usage 1, specify recall:
#    python auto_perf_test.py --dbname SIFT100M --topK 10 --use_recall_goal 1 --recall_goal 0.8 --nlist_min 1024 --nlist_max 16384 --FPGA_num 1 --bank_num 16 --recall_dict_dir './cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --FPGA_perf_dict_dir './FPGA_perf_dict_SIFT100M.pkl' --overwrite 0
# Usage 2, evaluate all recall:
#    python auto_perf_test.py --dbname SIFT100M --topK 10 --use_recall_goal 0 --nlist_min 1024 --nlist_max 65536 --FPGA_num 1 --bank_num 16 --recall_dict_dir './cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --FPGA_perf_dict_dir './FPGA_perf_dict_SIFT100M.pkl' --overwrite 0
# An example to evaluate the performance of bitstream K=1,10,100 on several dataset
#   Folder organization
#   --- auto_perf_test.py
#    |_ cpu_recall_index_nprobe_pairs_SIFT100M.pkl
#    |_ cpu_recall_index_nprobe_pairs_SIFT500M.pkl
#    |_ cpu_recall_index_nprobe_pairs_SIFT1000M.pkl
#    |_ bitstream_K_1
#    |_ bitstream_K_10
#    |_ bitstream_K_100
#  In each bitstream folder, e.g., bitstream_K_1, run the evaluate all command, e.g.,
#    python ../auto_perf_test.py --dbname SIFT100M --topK 1 --use_recall_goal 0 --nlist_min 1024 --nlist_max 65536 --FPGA_num 1 --bank_num 12 --recall_dict_dir '../cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --FPGA_perf_dict_dir '../FPGA_perf_dict_SIFT100M.pkl' --overwrite 1 > log_SIFT100M
#    python ../auto_perf_test.py --dbname SIFT500M --topK 1 --use_recall_goal 0 --nlist_min 1024 --nlist_max 65536 --FPGA_num 4 --bank_num 12 --recall_dict_dir '../cpu_recall_index_nprobe_pairs_SIFT500M.pkl' --FPGA_perf_dict_dir '../FPGA_perf_dict_SIFT500M.pkl' --overwrite 1 > log_SIFT500M
#    python ../auto_perf_test.py --dbname SIFT1000M --topK 1 --use_recall_goal 0 --nlist_min 1024 --nlist_max 65536 --FPGA_num 8 --bank_num 12 --recall_dict_dir '../cpu_recall_index_nprobe_pairs_SIFT1000M.pkl' --FPGA_perf_dict_dir '../FPGA_perf_dict_SIFT1000M.pkl' --overwrite 1 > log_SIFT1000M

# Note! The recalls of the SITF500M & SIFT1000M are wrong, because
#   1. the host verify the ground truth with the SIFT100M dataset (I hardcoded it)
#   2. the results returned by this script only uses the data partition of 1 single FPGA, 
#         we need to gather the partition of multiple FPGAs to verify the correctness

import csv
import os
import pickle
import sys
import time
import numpy as np
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default=0, help="dataset name, e.g., SIFT100M")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--use_recall_goal', type=int, default=0, help="whether to evaluate the single recall goal")
parser.add_argument('--recall_goal', type=float, default=0.5, help="target minimum recall, e.g., 50% -> 0.5, if not specified, evaluate all recall goal")
parser.add_argument('--nlist_min', type=int, default=1024, help="the minimum nprobe to evluate")
parser.add_argument('--nlist_max', type=int, default=16384, help="the minimum nprobe to evluate")
parser.add_argument('--FPGA_num', type=int, default=1, help="the number of FPGAs to execute the data (we only run test on 1 FPGA to evaluate the performance)")
parser.add_argument('--bank_num', type=int, default=16, help="the FPGA bank number used to store PQ codes")
parser.add_argument('--recall_dict_dir', type=str, default='./cpu_recall_index_nprobe_pairs_SIFT100M.pkl', help="recall dictionary directory")
parser.add_argument('--FPGA_perf_dict_dir', type=str, default='./FPGA_perf_dict_SIFT100M.pkl', help="FPGA performance dictionary directory")
parser.add_argument('--overwrite', type=int, default=0, help="whether to overwrite the previous value in the perf dict")

args = parser.parse_args()
dbname = args.dbname
topK = args.topK
use_recall_goal = args.use_recall_goal
if use_recall_goal:
	recall_goal = args.recall_goal
nlist_min = args.nlist_min 
nlist_max = args.nlist_max 
FPGA_num = args.FPGA_num
bank_num = args.bank_num
recall_dict_dir = args.recall_dict_dir
FPGA_perf_dict_dir = args.FPGA_perf_dict_dir
overwrite = args.overwrite

d_recall = None # recall dictionary
if os.path.exists(recall_dict_dir) and os.path.getsize(recall_dict_dir) > 0: # load and write
    with open(recall_dict_dir, 'rb') as f:
        d_recall = pickle.load(f)

def load_perf_from_profile_summary():
    """
    CSV sample (the kernel time part, use 1481.36 as the number)

    Kernel Execution
    Kernel,Number Of Enqueues,Total Time (ms),Minimum Time (ms),Average Time (ms),Maximum Time (ms),
    vadd,1,1481.36,1481.36,1481.36,1481.36,
    """

    with open('profile_summary.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next_line_perf = False # whether the next line is the perf number
        for i, row in enumerate(csv_reader):
            if next_line_perf:
                time_ms = row[2]
                QPS = 10000 / (float(time_ms) / 1000.0)
                return QPS
            if 'Number Of Enqueues' in row:
                next_line_perf = True
        # print(csv_reader[i+1])

def write_perf(dbname, index_key, topK, recall, QPS, overwrite):

    fname = FPGA_perf_dict_dir
    if os.path.exists(fname) and os.path.getsize(fname) > 0: # load and write
        d = None
        with open(fname, 'rb') as f:
            d = pickle.load(f)

        with open(fname, 'wb') as f:
            # dictionary format:
            #   d[dbname (str)][index_key (str)][topK (int)][recall_goal (float, 0~1)] = nprobe
            #   e.g., d["SIFT100M"]["IVF4096,PQ16"][10][0.7]
            if dbname not in d:
                d[dbname] = dict()
            if index_key not in d[dbname]:
                d[dbname][index_key] = dict()
            if topK not in d[dbname][index_key]:
                d[dbname][index_key][topK] = dict()
            if recall_goal in d[dbname][index_key][topK]:
                if overwrite:
                    d[dbname][index_key][topK][recall_goal] = QPS
                    pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
            else:
                d[dbname][index_key][topK][recall_goal] = QPS
                pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    else: # write new file
        with open(fname, 'wb') as f:
            # dictionary format:
            #   d[dbname (str)][index_key (str)][topK (int)][recall_goal (float, 0~1)] = nprobe
            #   e.g., d["SIFT100M"]["IVF4096,PQ16"][10][0.7]
            d = dict()
            d[dbname] = dict()
            d[dbname][index_key] = dict()
            d[dbname][index_key][topK] = dict()
            d[dbname][index_key][topK][recall_goal] = QPS
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

for index_key in d_recall[dbname]:

    if "OPQ" in index_key:
        OPQ_enable = 1
    else:
        OPQ_enable = 0

    nlist = None
    index_array = index_key.split(",")
    if len(index_array) == 2: # "IVF4096,PQ16" or "IMI2x14,PQ16" 
        s = index_array[0]
        if s[:3]  == "IVF":
            nlist = int(s[3:])
        else:
            continue
    elif len(index_array) == 3: # "OPQ16,IVF4096,PQ16" or "OPQ16,IMI2x14,PQ16" 
        s = index_array[1]
        if s[:3]  == "IVF":
            nlist = int(s[3:])
        else:
            continue
    else:
        continue
    if nlist < nlist_min or nlist > nlist_max:
        continue

    if use_recall_goal:
        if d_recall[dbname][index_key][topK][recall_goal] is not None:
            nprobe = d_recall[dbname][index_key][topK][recall_goal]
            if FPGA_num == 1:
                data_dir = "/mnt/scratch/wenqi/saved_npy_data/FPGA_data_{dbname}_{index_key}_{bank_num}_banks".format(
                    dbname=dbname, index_key=index_key, bank_num=bank_num)
            else:
                data_dir = "/mnt/scratch/wenqi/saved_npy_data/FPGA_data_{dbname}_{index_key}_{FPGA_num}_FPGA_{bank_num}_banks/FPGA_0".format(
                    dbname=dbname, index_key=index_key, FPGA_num=FPGA_num, bank_num=bank_num)
            cmd = "./host vadd.xclbin {nlist} {nprobe} {OPQ_enable} {data_dir} /mnt/scratch/wenqi/saved_npy_data/gnd".format(
                nlist=nlist, nprobe=nprobe, OPQ_enable=OPQ_enable, data_dir=data_dir)
            print("Executing command:\n{}".format(cmd))
            os.system(cmd)
            QPS = load_perf_from_profile_summary()
            print("\n\ndbname={} index_key={} topK={} recall_goal={} QPS={}\n\n".format(dbname, index_key, topK, recall_goal, QPS))
            write_perf(dbname, index_key, topK, recall_goal, QPS, overwrite)
    else:
        for recall_goal in d_recall[dbname][index_key][topK]:
            if d_recall[dbname][index_key][topK][recall_goal] is not None:
                nprobe = d_recall[dbname][index_key][topK][recall_goal]
                if FPGA_num == 1:
                    data_dir = "/mnt/scratch/wenqi/saved_npy_data/FPGA_data_{dbname}_{index_key}_{bank_num}_banks".format(
                        dbname=dbname, index_key=index_key, bank_num=bank_num)
                else:
                    data_dir = "/mnt/scratch/wenqi/saved_npy_data/FPGA_data_{dbname}_{index_key}_{FPGA_num}_FPGA_{bank_num}_banks/FPGA_0".format(
                        dbname=dbname, index_key=index_key, FPGA_num=FPGA_num, bank_num=bank_num)
                cmd = "./host vadd.xclbin {nlist} {nprobe} {OPQ_enable} {data_dir} /mnt/scratch/wenqi/saved_npy_data/gnd".format(
                    dbname=dbname, nlist=nlist, nprobe=nprobe, OPQ_enable=OPQ_enable, data_dir=data_dir)
                print("Executing command:\n{}".format(cmd))
                os.system(cmd)
                QPS = load_perf_from_profile_summary()
                print("\n\ndbname={} index_key={} topK={} recall_goal={} QPS={}\n\n".format(dbname, index_key, topK, recall_goal, QPS))
                write_perf(dbname, index_key, topK, recall_goal, QPS, overwrite)



