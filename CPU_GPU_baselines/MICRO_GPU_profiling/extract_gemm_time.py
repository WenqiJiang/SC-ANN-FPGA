"""
analyze the gemm in stage 1~3 versus stage 4~5 from a GPU event trace file
"""

import csv
import os
import pickle
import sys
import time
import numpy as np
import argparse 

# WENQI: TODO: add assertion (from gpukrnlsum.csv)
parser = argparse.ArgumentParser()
parser.add_argument('--gputrace_csv_dir', type=str, default='./report_OPQ16,IVF8192,PQ16_nprobe_17_K_100_B_512_gputrace.csv', help="GPU event trace csv file")
parser.add_argument('--verify', type=int, default=0, help="0: do not verify result; 1: verify total gemm ms using gpukernsum_csv_dir")
parser.add_argument('--gpukernsum_csv_dir', type=str, default='./report_OPQ16,IVF8192,PQ16_nprobe_17_K_100_B_512_gpukernsum.csv', help="GPU event trace csv file")

args = parser.parse_args()
gputrace_csv_dir = args.gputrace_csv_dir
verify = args.verify
gpukernsum_csv_dir = args.gpukernsum_csv_dir

# GPU can invoke multiple types of gemm kernel in a single run, e.g., volta_sgemm_32x128_tn and volta_sgemm_32x32_sliced1x4_tn
gemm_krnl_name = 'gemm' 

def gemm_stats(gputrace_csv_dir, verify=0, gpukernsum_csv_dir=gpukernsum_csv_dir):
    """
    the first 4 transposeAny are called during init
    then, the pattern is 
        N * stage 2 gemm 
        transposeAny
        N * stage 4~5 gemm
        transposeAny
        N * stage 2 gemm 
        transposeAny
        N * stage 4~5 gemm
        transposeAny
        ...
    then
    """

    transpose_count = 0
    stage_1_3_time = 0 
    stage_4_6_time = 0
    stage_all_include_init = 0
    stage_all_exclude_init = 0

    with open(gputrace_csv_dir) as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for i, row in enumerate(csv_reader):
            if 'transposeAny' in row['Name']:
                transpose_count += 1
                continue

            if gemm_krnl_name in row['Name']: 

                stage_all_include_init += int(row['Duration (ns)'])

                if transpose_count < 4:
                    continue
                elif (transpose_count - 4) % 2 == 0:
                    # stage 1~3 gemm
                    stage_1_3_time += int(row['Duration (ns)'])
                    stage_all_exclude_init += int(row['Duration (ns)'])
                    continue
                elif (transpose_count - 4) % 2 == 1:
                    # stage 4~6 gemm
                    stage_4_6_time += int(row['Duration (ns)'])
                    stage_all_exclude_init += int(row['Duration (ns)'])
                    continue
                else:
                    continue

    assert (transpose_count - 4) % 2 == 0
    batch_count = int((transpose_count - 4) / 2)
    print("Stage 1~3 GEMM: {} ns".format(stage_1_3_time))
    print("Stage 4~6 GEMM: {} ns".format(stage_4_6_time))
    print("Stage 1~6 GEMM: {} ns".format(stage_all_exclude_init))
    print("Init + Stage 1~6 GEMM: {} ns".format(stage_all_include_init))
    print("Total batch number: {}".format(batch_count))

    if verify:
        total_gemm_time = 0
        with open(gpukernsum_csv_dir) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for i, row in enumerate(csv_reader):
                if gemm_krnl_name in row['Name']:
                    total_gemm_time += int(row['Total Time (ns)'])
        
        if total_gemm_time == stage_all_include_init:
            print("Verification SUCCESSFUL: total GEMM duration is correct.")
        else:
            print("ERROR: total GEMM duration is wrong. computed: {}\tgpukernsum: {}".format(
                stage_all_include_init, total_gemm_time))


if __name__ == '__main__':
    gemm_stats(gputrace_csv_dir, verify, gpukernsum_csv_dir)