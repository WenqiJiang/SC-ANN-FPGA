"""
Classify the time consumption by stages, it requires two files as input, 
    the gputrace (event database) and gpukernel sum (time per kernel), e.g.:
        nsys_report_SIFT100M_IVF1024,PQ16_R@10=0.8_nprobe_39_ngpu_1_batchsize_10000_gputrace.csv
        nsys_report_SIFT100M_IVF1024,PQ16_R@10=0.8_nprobe_39_ngpu_1_batchsize_10000_gpukernsum.csv

Example Usage:
    python classify_stages.py --file_prefix ./nsys_report_SIFT100M_IVF1024,PQ16_R@10=0.8_nprobe_39_ngpu_1_batchsize_10000 
"""

import csv
import os
import pickle
import sys
import time
import numpy as np
import argparse 

def transpose_gemm_stats(gputrace_csv_dir, gpukernsum_csv_dir):
    """
    part of the transeposeAny are initialization; other are in stage 4~5
    part of the gemms are initialization; other are split across stage 1~2 and stage 4~5
    return transposeAny_stage_4_5, gemm_stage_1_2, gemm_stage_4_5  (all in ns)

    the first 4 transposeAny are called during init
    then, the pattern is 
        N * stage 1~2 gemm 
        transposeAny
        N * stage 4~5 gemm
        transposeAny
        N * stage 1~2 gemm 
        transposeAny
        N * stage 4~5 gemm
        transposeAny
        ...
    """

    transpose_count = 0
    transpose_all_time = 0
    transpose_init_time = 0
    transpose_4_5_time = 0

    gemm_stage_1_2_time = 0 
    gemm_stage_4_5_time = 0
    gemm_stage_all_include_init = 0
    gemm_stage_all_exclude_init = 0

    # GPU can invoke multiple types of gemm kernel in a single run, e.g., volta_sgemm_32x128_tn and volta_sgemm_32x32_sliced1x4_tn
    gemm_krnl_name = 'gemm' 

    with open(gputrace_csv_dir) as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for i, row in enumerate(csv_reader):
            if 'transposeAny' in row['Name'] or 'transposeOuter' in row['Name']:
                transpose_count += 1
                transpose_all_time += int(row['Duration(nsec)'])
                if transpose_count <= 4:
                    transpose_init_time += int(row['Duration(nsec)'])
                else:
                    transpose_4_5_time += int(row['Duration(nsec)'])
                continue

            if gemm_krnl_name in row['Name']: 

                gemm_stage_all_include_init += int(row['Duration(nsec)'])

                if transpose_count < 4:
                    continue
                elif (transpose_count - 4) % 2 == 0:
                    # stage 1~3 gemm
                    gemm_stage_1_2_time += int(row['Duration(nsec)'])
                    gemm_stage_all_exclude_init += int(row['Duration(nsec)'])
                    continue
                elif (transpose_count - 4) % 2 == 1:
                    # stage 4~6 gemm
                    gemm_stage_4_5_time += int(row['Duration(nsec)'])
                    gemm_stage_all_exclude_init += int(row['Duration(nsec)'])
                    continue
                else:
                    continue

    assert (transpose_count - 4) % 2 == 0
    batch_count = int((transpose_count - 4) / 2)

    total_gemm_time_truth = 0
    total_transpose_time_truth = 0
    with open(gpukernsum_csv_dir) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for i, row in enumerate(csv_reader):
            if gemm_krnl_name in row['Name']:
                total_gemm_time_truth += int(row['Total Time (ns)'])
            elif "transposeAny" in row['Name'] or 'transposeOuter' in row['Name']:
                total_transpose_time_truth += int(row['Total Time (ns)'])
    
    if total_gemm_time_truth == gemm_stage_all_include_init:
        # print("Verification SUCCESSFUL: total GEMM duration is correct.")
        pass
    else:
        print("ERROR: total GEMM duration is wrong. computed: {}\tgpukernsum: {}".format(
            gemm_stage_all_include_init, total_gemm_time_truth))
        raise ValueError
    if total_transpose_time_truth == transpose_all_time:
        # print("Verification SUCCESSFUL: total transposeAny duration is correct.")
        pass
    else:
        print("ERROR: total transposeAny duration is wrong. computed: {}\tgpukernsum: {}".format(
            transpose_all_time, total_transpose_time_truth))
        raise ValueError

    return transpose_4_5_time, gemm_stage_1_2_time, gemm_stage_4_5_time

def classify_stages(gputrace_csv_dir, gpukernsum_csv_dir):
    """
    classify the kernel time consumption by stages:
        stage 1~2
        stage 3
        stage 4~5
        stage 6
        other functions like thrust::cuda_cub::core::_kernel_agent

    return t_1_2, t_3, t_4_5, t_6, t_other, t_transpose_4_5  (all in ns)
    """

    """
    Complete function list according to gpukernsum:
        transposeAny / transposeOuter # stage 4~5
        pass1SelectLists # stage 6
        pqScanPrecomputedMultiPass # stage 4~5
        gemm # stage 1~2 or stage 4~5
        pass2SelectLists # stage  6
        l2SelectMinK # stage 3
        getResultLengths # stage 4~5
        sumAlongColumns # init
        sumAlongRows # stage 3
        l2NormRowMajor # stage 1~2


        Other non-faiss function:
            thrust::cuda_cub::core::_kernel_agent

    """

    transpose_4_5_time, gemm_stage_1_2_time, gemm_stage_4_5_time = \
        transpose_gemm_stats(gputrace_csv_dir, gpukernsum_csv_dir)
    # print("transpose_4_5_time: {} ns".format(transpose_4_5_time))
    # print("gemm_stage_1_2_time: {} ns".format(gemm_stage_1_2_time))
    # print("gemm_stage_4_5_time: {} ns".format(gemm_stage_4_5_time))

    t_1_2 = 0 + gemm_stage_1_2_time
    t_3 = 0
    t_4_5 = 0 + transpose_4_5_time + gemm_stage_4_5_time
    t_6 = 0
    t_total = transpose_4_5_time + gemm_stage_1_2_time + gemm_stage_4_5_time

    with open(gpukernsum_csv_dir) as csv_file:
        csv_reader = csv.DictReader(csv_file)

        # load time consumption except transposeAny, gemm, and the non-Faiss function _kernel_agent
        for i, row in enumerate(csv_reader):

            if ('transposeAny' not in row['Name']) and ('transposeOuter' not in row['Name']) and ('gemm' not in row['Name']):
                t_total += int(row['Total Time (ns)'])

            # stage 1 ~ 2
            if 'l2NormRowMajor' in row['Name']:
                t_1_2 += int(row['Total Time (ns)'])
            # stage 3
            elif ('l2SelectMin' in row['Name']) or ('sumAlongRows' in row['Name']):
                t_3 += int(row['Total Time (ns)'])
            # stage 4~5
            elif ('pqScanPrecomputedMultiPass' in row['Name']) or ('getResultLengths' in row['Name']):
                t_4_5 += int(row['Total Time (ns)'])
            elif ("pass1SelectLists" in row['Name']) or ("pass2SelectLists" in row['Name']):
                t_6 += int(row['Total Time (ns)'])

    assert (t_1_2 + t_3 + t_4_5 + t_6) / t_total >= 0.90, "Unknown function consumes > 10% total kernel time"
    t_other = t_total - (t_1_2 + t_3 + t_4_5 + t_6)
    t_transpose_4_5 = transpose_4_5_time # transpose is a unique bottleneck in many settings, take it as its own

    return t_1_2, t_3, t_4_5, t_6, t_other, t_transpose_4_5

def get_percentage(t_1_2, t_3, t_4_5, t_6, t_other, t_transpose_4_5=None):

    t_total = t_1_2 + t_3 + t_4_5 + t_6 + t_other

    # 0 ~ 100%
    p_1_2 = t_1_2 / t_total * 100
    p_3 = t_3 / t_total * 100
    p_4_5 = t_4_5 / t_total * 100
    p_6 = t_6 / t_total * 100
    p_other = t_other / t_total * 100

    if t_transpose_4_5 is not None:
        p_transpose_4_5 = t_transpose_4_5 / t_total * 100
        return p_1_2, p_3, p_4_5, p_6, p_other, p_transpose_4_5
    else:
        return p_1_2, p_3, p_4_5, p_6, p_other

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_prefix', type=str, 
        default='./nsys_report_SIFT100M_IVF1024,PQ16_R@10=0.8_nprobe_39_ngpu_1_batchsize_10000', 
        help="gputrace / gpukernsum csv file common prefix")

    args = parser.parse_args()
    gputrace_csv_dir = args.file_prefix + '_gputrace.csv'
    gpukernsum_csv_dir = args.file_prefix + "_gpukernsum.csv"

    print("=== extract transpose & GEMM time ===\n")
    transpose_4_5_time, gemm_stage_1_2_time, gemm_stage_4_5_time = \
        transpose_gemm_stats(gputrace_csv_dir, gpukernsum_csv_dir)
    print("transpose_4_5_time: {} ns".format(transpose_4_5_time))
    print("gemm_stage_1_2_time: {} ns".format(gemm_stage_1_2_time))
    print("gemm_stage_4_5_time: {} ns".format(gemm_stage_4_5_time))
    print("\n")

    print("=== classify time by stages ===\n")
    t_1_2, t_3, t_4_5, t_6, t_other, t_transpose_4_5 = classify_stages(gputrace_csv_dir, gpukernsum_csv_dir)
    p_1_2, p_3, p_4_5, p_6, p_other, p_transpose_4_5 = get_percentage(t_1_2, t_3, t_4_5, t_6, t_other, t_transpose_4_5=t_transpose_4_5)
    print("stage 1~2: {} ns\t{}%".format(t_1_2, p_1_2))
    print("stage 3: {} ns\t{}%".format(t_3, p_3))
    print("stage 4~5: {} ns\t{}%".format(t_4_5, p_4_5))
    print("stage 6: {} ns\t{}%".format(t_6, p_6))
    print("other helper kernels: {} ns\t{}%".format(t_other, p_other))
    print("transposeAny/transposeOuter kernel: {} ns\t{}%".format(t_transpose_4_5, p_transpose_4_5))



