"""
Usage: copy the config.yaml and perf_test.py to the bitstream folder,
    the bitstream folder should also contain: host vadd.xclbin xrt.ini
Then, execute the following command, specify data/gt/config dir if needed:
    python perf_test.py <--config_dir ... --data_parent_dir ... --bitstream_dir ... --gt_dir ...>
"""

import os
import sys
import csv
import time
import yaml
import numpy as np
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str, default='./config.yaml', help="the parent data dir")
parser.add_argument('--bitstream_dir', type=str, default='./xclbin/vadd.hw.xclbin', help="e.g., ./xclbin/vadd.hw.xclbin")
parser.add_argument('--data_parent_dir', type=str, default='/mnt/scratch/wenqi/saved_npy_data', help="the parent data dir")
parser.add_argument('--gt_dir', type=str, default='/mnt/scratch/wenqi/saved_npy_data/Deep_gnd', help="the ground truth dir")

# ./host <XCLBIN File> <data directory> <ground truth dir>

args = parser.parse_args()
config_dir = args.config_dir
data_parent_dir = args.data_parent_dir
gt_dir = args.gt_dir

# Load YAML configurations
config_file = open(config_dir, "r")
config = yaml.load(config_file)

dbname = config["DB_NAME"]
topK = config["TOPK"]
nlist = config["NLIST"]
OPQ_enable = config["OPQ_ENABLE"]
bank_num = config["HBM_CHANNEL_NUM"]
FPGA_num = config["FPGA_NUM"]

index_key = None
if OPQ_enable:
    index_key = 'OPQ16,IVF{},PQ16'.format(nlist)
else:
    index_key = 'IVF{},PQ16'.format(nlist)

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

if FPGA_num == 1:
    data_dir = "{data_parent_dir}/FPGA_data_{dbname}_{index_key}_{bank_num}_banks".format(
        data_parent_dir=data_parent_dir, dbname=dbname, index_key=index_key, bank_num=bank_num)
else:
    data_dir = "{data_parent_dir}/FPGA_data_{dbname}_{index_key}_{FPGA_num}_FPGA_{bank_num}_banks/FPGA_0".format(
        data_parent_dir=data_parent_dir, dbname=dbname, index_key=index_key, FPGA_num=FPGA_num, bank_num=bank_num)
cmd = "./host {bitstream_dir} {data_dir} {gt_dir}".format(bitstream_dir=args.bitstream_dir, data_dir=data_dir, gt_dir=gt_dir)
print("Executing command:\n{}".format(cmd))
os.system(cmd)
QPS = load_perf_from_profile_summary()
print("\n\ndbname={} index_key={} topK={} QPS={}\n\n".format(dbname, index_key, topK, QPS))