"""
Evaluating the effect of nlist given the fixed nprobe

Example Usage:
    python experiment_3_nlist.py --dbname SIFT100M --topK 10 --nprobe 16 --ngpu 1 --startgpu 0 --qbs 10000 --nsys_enable 1
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import faiss
import pickle
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SIFT100M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--nprobe', type=int, default=1, help="num of cells to scan, e.g. 32")
parser.add_argument('--ngpu', type=int, default=1, help="number of gpus to use")
parser.add_argument('--startgpu', type=int, default=0, help="id of the first GPU")
parser.add_argument('--qbs', type=int, default=10000, help="batch size, 1~10000")
parser.add_argument('--nsys_enable', type=int, default=1, help="whether to profile by nsys")

args = parser.parse_args()
dbname = args.dbname
nprobe = args.nprobe
topK = args.topK
ngpu = args.ngpu
startgpu = args.startgpu
qbs = args.qbs
nsys_enable = args.nsys_enable

out_dir = "result_experiment_3_nlist"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

logname = "./{out_dir}/out_{dbname}_K_{topK}_nprobe_{nprobe}_ngpu_{ngpu}".format(
    out_dir=out_dir, dbname=dbname, topK=topK, nprobe=nprobe, ngpu=ngpu)
if os.path.exists(logname):
    os.remove(logname)

index_keys = ['IVF1024,PQ16', 'IVF2048,PQ16', 'IVF4096,PQ16', 'IVF8192,PQ16', 'IVF16384,PQ16', 'IVF32768,PQ16', 'IVF65536,PQ16', \
    'OPQ16,IVF1024,PQ16', 'OPQ16,IVF2048,PQ16', 'OPQ16,IVF4096,PQ16', 'OPQ16,IVF8192,PQ16', 'OPQ16,IVF16384,PQ16', 'OPQ16,IVF32768,PQ16', 'OPQ16,IVF65536,PQ16']

for index_key in index_keys:

    # Example command:
    # python ../bench_gpu_1bn.py -dbname SIFT100M -index_key OPQ16,IVF262144,PQ16 -topK 100 -ngpu 1 -startgpu 1 -tempmem $[1536*1024*1024] -nprobe 32 -qbs 512
    cmd = "python ../bench_gpu_1bn.py -dbname {dbname} -index_key {index_key} -topK {topK} -ngpu {ngpu} -startgpu {startgpu} -nprobe {nprobe} -qbs {qbs} >> {logname}".format(
        dbname=dbname, index_key=index_key, topK=topK, ngpu=ngpu, startgpu=startgpu, nprobe=nprobe, qbs=qbs, logname=logname)

    if not nsys_enable:
        os.system(cmd)
    else:
        print("WARNING: nsys may cause memory bug and failed profiling by using small batches, according to experiments on 16GB V100")
        reportname = "./{out_dir}/nsys_report_{dbname}_{index_key}_K_{topK}_nprobe_{nprobe}_ngpu_{ngpu}_batchsize_{qbs}".format(
            out_dir=out_dir, dbname=dbname, index_key=index_key, topK=topK, nprobe=nprobe, ngpu=ngpu, qbs=qbs)
        cmd_prefix = "nsys profile --output {reportname} --force-overwrite true --trace=cuda,cudnn,cublas,osrt,nvtx ".format(reportname=reportname) # overwrite if the profile already exists
        cmd_prof = cmd_prefix + cmd
        os.system(cmd_prof)

        # generate the gpuevent.csv & gpukrnlsum.csv
        # the gpukrnl sum file does not naturally support long file name, thus name is as tmp, then rename
        cmd_stats = "nsys stats --report gputrace --report gpukernsum {reportname}.qdrep --output ./{out_dir}/report_tmp".format(
            reportname=reportname, out_dir=out_dir)
        os.system(cmd_stats)
        os.system("mv ./{out_dir}/report_tmp_gpukernsum.csv {reportname}_gpukernsum.csv".format(
            out_dir=out_dir, reportname=reportname))
        os.system("mv ./{out_dir}/report_tmp_gputrace.csv {reportname}_gputrace.csv".format(
            out_dir=out_dir, reportname=reportname))
