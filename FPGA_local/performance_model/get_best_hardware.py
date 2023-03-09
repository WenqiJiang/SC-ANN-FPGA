"""
Usage 1: Given a database, an FPGA device, a resource constraint, and a frequency setting,
    search the parameter space (nlist, nprobe, OPQ_enable) and find the best
    hardware solution
Example usage:
    python get_best_hardware.py --mode 1 --dbname SIFT100M --FPGA_num 1 --topK 10 --recall_goal 0.8 --nprobe_dict_dir './recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --device U280 --max_utilization_rate 60 --freq 140 > out

Usage 2: Same as usage1, but specify a single nlist, also specify with/without OPQ
    python get_best_hardware.py --mode 2 --dbname SIFT100M --FPGA_num 1 --topK 10 --nlist 8192 --OPQ_enable 1 --recall_goal 0.8 --nprobe_dict_dir './recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --device U280 --max_utilization_rate 60 --freq 140 > out

Usage 3: specify nlist, nprobe, with/without OPQ, the recall is decided by these variables, so do not load the recall dictionary.
    python get_best_hardware.py --mode 3 --dbname SIFT100M --FPGA_num 1 --topK 10 --nlist 8192 --nprobe 17 --OPQ_enable 1 --device U280 --max_utilization_rate 60 --freq 140 > out
"""

import numpy as np
import os 
import pickle
import io

from contextlib import redirect_stdout

from constants import * 
from queue_and_sorting import *
from stages import *
from utils import *
import argparse 

parser = argparse.ArgumentParser()
# DB-related parameters
parser.add_argument('--mode', type=int, default=0, help="mode 1 / 2 / 3, see comments at the beginning of the py file")
parser.add_argument('--dbname', type=str, default='', help="e.g., SIFT100M")
parser.add_argument('--FPGA_num', type=int, default=1, help="number of FPGAs to serve a dataset, e.g., may use 8 to serve SIFT1B")
parser.add_argument('--topK', type=int, default=10, help="return the topK results")
parser.add_argument('--nlist', type=int, default=0, help="mode 2 / 3 specify nlist")
parser.add_argument('--nprobe', type=int, default=0, help="mode 3 specify nprobe")
parser.add_argument('--OPQ_enable', type=int, default=0, help="mode 2 / 3 specify whether enable OPQ")
parser.add_argument('--recall_goal', type=float, default=0.8, help="recall goal, e.g., 0.8 (80%)")
parser.add_argument('--nprobe_dict_dir', type=str, default='./recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl', help="a dictionary of d[dbname][index_key][topK][recall_goal] -> nprobe")
# FPGA-related parameters
parser.add_argument('--device', type=str, default='U280', help="U280/U250/U50")
parser.add_argument('--max_utilization_rate', type=int, default=80, help="in percentage")
parser.add_argument('--freq', type=int, default=140, help="FPGA frequency in MHz")

args = parser.parse_args()
print("""
Arguments:
    dbname: {dbname}
    topK: {topK}
    device: {device}
    max_utilization_rate: {max_utilization_rate}%
    freq: {freq} MHz
""".format(dbname=args.dbname, topK=args.topK, device=args.device, 
max_utilization_rate=args.max_utilization_rate, freq=args.freq))

assert args.dbname != '', "Please fill the DB name, e.g., SITF100M"
if args.dbname == 'SIFT100M' or args.dbname == 'Deep100M':
    TOTAL_VECTORS = int(1e8 / args.FPGA_num)
elif args.dbname == 'SIFT500M':
    TOTAL_VECTORS = int(5e8 / args.FPGA_num)
elif args.dbname == 'SIFT1000M':
    TOTAL_VECTORS = int(1e9 / args.FPGA_num)
else:
    print("Unsupported dataset")
    raise ValueError

"""
WENQI: the numbers below are got from the 100M dataset, more precise estimation TBD
An example of expected scanned ratio of a single index

e.g., suppose the query vectors has the same distribution as the trained vectors, 
    then the larger a Voronoi cell, the more likely they will be searched
e.g., searching 32 cells over 8192 in 100 M dataset will not scan 32 / 8192 * 1e8 entries on average,
    we need to scan more
"""
scan_ratio_with_OPQ = {
    1024: 1.102495894347366,
    2048: 1.12463916710666,
    4096: 1.12302396550103,
    8192: 1.135891773928242,
    16384: 1.1527141392580655,
    32768: 1.1441353378627621,
    65536: 1.1411144965226643,
    131072: 1.1476783059960072,
    262144: 1.1543383003102523
}
scan_ratio_without_OPQ = {
    1024: 1.1023307648983034,
    2048: 1.1245342465011723,
    4096: 1.1230564521721877,
    8192: 1.135866022841546, 
    16384: 1.1523836603564073, 
    32768: 1.1440334275739672,
    65536: 1.1410689577844846,
    131072: 1.1476378583040157,
    262144: 1.1543274466049378
}

topK = args.topK
recall_goal = args.recall_goal
MAX_UTIL_PERC = args.max_utilization_rate / 100.0
FREQ = args.freq * 1e6

if args.device == 'U280':
    """ Resource related constants """
    TOTAL_BRAM_18K = 4032 
    TOTAL_DSP48E = 9024
    TOTAL_FF = 2607360 
    TOTAL_LUT = 1303680
    TOTAL_URAM = 960

    # MAX_HBM_bank = 32 - 2 - 2 - 1 # reserve 30, 31 unused due to their overlap with PCIe; 2 for Network; 1 for value init
    MAX_HBM_bank = 20 # use 20 as a conservative constraints, as larger values can lead to routing error
    MAX_BRAM_18K = TOTAL_BRAM_18K * MAX_UTIL_PERC
    MAX_DSP48E = TOTAL_DSP48E * MAX_UTIL_PERC
    MAX_FF = TOTAL_FF * MAX_UTIL_PERC
    MAX_LUT = TOTAL_LUT * MAX_UTIL_PERC
    MAX_URAM = TOTAL_URAM * MAX_UTIL_PERC

    if args.dbname == 'SIFT100M' or args.dbname == 'Deep100M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 100 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / args.FPGA_num)) # at least 9 banks to hold PQ16 version
    elif args.dbname == 'SIFT500M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 500 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / args.FPGA_num))
    elif args.dbname == 'SIFT1000M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 1000 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / args.FPGA_num))
    else:
        print("Unsupported dataset")
        raise ValueError
    
    #####     Shell     #####

    resource_hmss = Resource()
    resource_hmss.LUT = 55643 
    resource_hmss.FF = 103037
    resource_hmss.BRAM_18K = 2 * 4
    resource_hmss.URAM = 0
    resource_hmss.DSP48E = 0
    resource_hmss.HBM_bank = 0

    resource_System_DPA = Resource()
    resource_System_DPA.LUT = 35738
    resource_System_DPA.FF = 76789
    resource_System_DPA.BRAM_18K = 2 * 16
    resource_System_DPA.URAM = 0
    resource_System_DPA.DSP48E = 0
    resource_System_DPA.HBM_bank = 0

    resource_xdma = Resource()
    resource_xdma.LUT = 9100
    resource_xdma.FF = 15572
    resource_xdma.BRAM_18K = 2 * 0
    resource_xdma.URAM = 0
    resource_xdma.DSP48E = 0
    resource_xdma.HBM_bank = 0

    resourece_static_region = Resource()
    resourece_static_region.LUT = 93280
    resourece_static_region.FF = 128746
    resourece_static_region.BRAM_18K = 2 * 200
    resourece_static_region.URAM = 0
    resourece_static_region.DSP48E = 4
    resourece_static_region.HBM_bank = 0

    # component_list_shell = [resource_hmss, resource_System_DPA, resource_xdma, resourece_static_region]
    component_list_shell = [resource_hmss, resource_System_DPA, resource_xdma, resourece_static_region]
    shell_consumption = sum_resource(component_list_shell)

elif args.device == 'U50':
    """ Resource related constants """
    TOTAL_BRAM_18K = 2688 
    TOTAL_DSP48E = 5952
    TOTAL_FF = 1743360 
    TOTAL_LUT = 871680
    TOTAL_URAM = 640

    # MAX_HBM_bank = 32 - 2 # reserve 30, 31 unused due to their overlap with PCIe
    MAX_HBM_bank = 20 # use 20 as a conservative constraints, as larger values can lead to routing error
    MAX_BRAM_18K = TOTAL_BRAM_18K * MAX_UTIL_PERC
    MAX_DSP48E = TOTAL_DSP48E * MAX_UTIL_PERC
    MAX_FF = TOTAL_FF * MAX_UTIL_PERC
    MAX_LUT = TOTAL_LUT * MAX_UTIL_PERC
    MAX_URAM = TOTAL_URAM * MAX_UTIL_PERC


    if args.dbname == 'SIFT100M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 100 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / args.FPGA_num)) # at least 9 banks to hold PQ16 version
    elif args.dbname == 'SIFT500M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 500 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / args.FPGA_num))
    elif args.dbname == 'SIFT1000M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 1000 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / args.FPGA_num))
    else:
        print("Unsupported dataset")
        raise ValueError

    #####     Shell     ####

    resourece_dynamic_region = Resource()
    resourece_dynamic_region.LUT = 92244
    resourece_dynamic_region.FF = 175459
    resourece_dynamic_region.BRAM_18K = 2 * 20
    resourece_dynamic_region.URAM = 0
    resourece_dynamic_region.DSP48E = 0
    resourece_dynamic_region.HBM_bank = 0

    resourece_static_region = Resource()
    resourece_static_region.LUT = 89439
    resourece_static_region.FF = 106348
    resourece_static_region.BRAM_18K = 2 * 176
    resourece_static_region.URAM = 0
    resourece_static_region.DSP48E = 4
    resourece_static_region.HBM_bank = 0

    component_list_shell = [resourece_dynamic_region, resourece_static_region]
    # component_list_shell = [resourece_dynamic_region, resourece_static_region]
    shell_consumption = sum_resource(component_list_shell)

elif args.device == 'U250':
    """ Resource related constants """
    TOTAL_BRAM_18K = 5376 
    TOTAL_DSP48E = 12288
    TOTAL_FF = 3456000 
    TOTAL_LUT = 1728000
    TOTAL_URAM = 1280

    MAX_HBM_bank = 4
    MAX_BRAM_18K = TOTAL_BRAM_18K * MAX_UTIL_PERC
    MAX_DSP48E = TOTAL_DSP48E * MAX_UTIL_PERC
    MAX_FF = TOTAL_FF * MAX_UTIL_PERC
    MAX_LUT = TOTAL_LUT * MAX_UTIL_PERC
    MAX_URAM = TOTAL_URAM * MAX_UTIL_PERC

    # 1 DRAM bank = 16 GB
    if args.dbname == 'SIFT100M':
        padding_factor = 1.05
        total_size = 100 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 16 * 1024 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / args.FPGA_num)) # at least 9 banks to hold PQ16 version
    elif args.dbname == 'SIFT500M':
        padding_factor = 1.05
        total_size = 500 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 16 * 1024 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / args.FPGA_num))
    elif args.dbname == 'SIFT1000M':
        padding_factor = 1.05
        total_size = 1000 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 16 * 1024 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / args.FPGA_num))
    else:
        print("Unsupported dataset")
        raise ValueError

    #####     Shell     #####
    resourece_dynamic_region = Resource()
    resourece_dynamic_region.LUT = 145032
    resourece_dynamic_region.FF = 219919
    resourece_dynamic_region.BRAM_18K = 2 * 377
    resourece_dynamic_region.URAM = 0
    resourece_dynamic_region.DSP48E = 12
    resourece_dynamic_region.HBM_bank = 0

    resourece_static_region = Resource()
    resourece_static_region.LUT = 104112
    resourece_static_region.FF = 160859
    resourece_static_region.BRAM_18K = 2 * 165
    resourece_static_region.URAM = 0
    resourece_static_region.DSP48E = 4
    resourece_static_region.HBM_bank = 0

    component_list_shell = [resourece_dynamic_region, resourece_static_region]
    shell_consumption = sum_resource(component_list_shell)

else:
    print("Unsupported device")
    raise ValueError


total_valid_design = 0 # design that is within resource consumption range

def get_best_hardware(nlist, nprobe, OPQ_enable=True):
    """
    given a single algorithm setting, return the hardware combination that achieves
        the highest performance
    """

    best_solution_QPS = 0
    best_solution_stage_option_list = []
    best_solution_PE_num_list = []
    global total_valid_design

    options_stage_1_OPQ = get_options_stage_1_OPQ(FREQ)
    options_stage_2_cluster_distance_computation = \
        get_options_stage_2_cluster_distance_computation(nlist, FREQ, MAX_URAM)
    options_stage_3_select_Voronoi_cells = get_options_stage_3_select_Voronoi_cells(nlist, nprobe, FREQ)
    options_stage_4_distance_LUT_construction = get_options_stage_4_distance_LUT_construction(nlist, nprobe, FREQ)
    if topK == 1:
        max_PE_num = MAX_HBM_bank * 3
    else: # K=10,100
        # prevent routing error
        max_PE_num = 36
    options_stage_5_distance_estimation_by_LUT = get_options_stage_5_distance_estimation_by_LUT(nlist, nprobe, FREQ, MIN_HBM_bank, MAX_HBM_bank, TOTAL_VECTORS, scan_ratio_with_OPQ, scan_ratio_without_OPQ, OPQ_enable, max_PE_num)

    if OPQ_enable:

        for option_stage_1_OPQ in options_stage_1_OPQ:
            for option_stage_2 in options_stage_2_cluster_distance_computation:
                for option_stage_3 in options_stage_3_select_Voronoi_cells:
                    for option_stage_4 in options_stage_4_distance_LUT_construction:
                        for option_stage_5 in options_stage_5_distance_estimation_by_LUT:

                            stage_5_PE_num = option_stage_5.STAGE5_COMP_PE_NUM
                            if OPQ_enable:
                                N_compute_per_nprobe = int(scan_ratio_with_OPQ[nlist] * TOTAL_VECTORS / nlist / stage_5_PE_num) + 1
                            else:
                                N_compute_per_nprobe = int(scan_ratio_without_OPQ[nlist] * TOTAL_VECTORS / nlist / stage_5_PE_num) + 1
                            N_insertion_per_stream = int(nprobe * N_compute_per_nprobe)
                            
                            options_stage_6_sort_reduction = get_options_stage6_select_topK(
                                stage_5_PE_num, N_insertion_per_stream, topK, FREQ)

                            for option_stage_6 in options_stage_6_sort_reduction:

                                option_list = \
                                    [option_stage_1_OPQ, option_stage_2, option_stage_3, \
                                        option_stage_4, option_stage_5, option_stage_6]
                                PE_num_list = [1, 1, 1, 1, 1, 1]

                                if fit_resource_constraints(
                                    option_list, 
                                    PE_num_list, 
                                    MAX_HBM_bank,
                                    MAX_BRAM_18K,
                                    MAX_DSP48E,
                                    MAX_FF,
                                    MAX_LUT,
                                    MAX_URAM,
                                    shell_consumption,
                                    count_shell=True):

                                    total_valid_design = total_valid_design + 1 # each valide design counts

                                    bottleneck_ID, accelerator_QPS = get_bottleneck(option_list)
                                    if accelerator_QPS > best_solution_QPS:
                                        best_solution_QPS = accelerator_QPS
                                        best_solution_stage_option_list = option_list
                                        best_solution_PE_num_list = PE_num_list
                                    elif accelerator_QPS == best_solution_QPS:
                                        if resource_consumption_A_less_than_B(
                                            option_list, PE_num_list,
                                            best_solution_stage_option_list, best_solution_PE_num_list):
                                                best_solution_QPS = accelerator_QPS
                                                best_solution_stage_option_list = option_list
                                                best_solution_PE_num_list = PE_num_list
    else: # no OPQ
        for option_stage_2 in options_stage_2_cluster_distance_computation:
            for option_stage_3 in options_stage_3_select_Voronoi_cells:
                for option_stage_4 in options_stage_4_distance_LUT_construction:
                    for option_stage_5 in options_stage_5_distance_estimation_by_LUT:

                        stage_5_PE_num = option_stage_5.STAGE5_COMP_PE_NUM
                        if OPQ_enable:
                            N_compute_per_nprobe = int(scan_ratio_with_OPQ[nlist] * TOTAL_VECTORS / nlist / stage_5_PE_num) + 1
                        else:
                            N_compute_per_nprobe = int(scan_ratio_without_OPQ[nlist] * TOTAL_VECTORS / nlist / stage_5_PE_num) + 1
                        N_insertion_per_stream = int(nprobe * N_compute_per_nprobe)
                        
                        options_stage_6_sort_reduction = get_options_stage6_select_topK(
                            stage_5_PE_num, N_insertion_per_stream, topK, FREQ)

                        for option_stage_6 in options_stage_6_sort_reduction:

                            option_list = \
                                [option_stage_2, option_stage_3, \
                                    option_stage_4, option_stage_5, option_stage_6]
                            PE_num_list = [1, 1, 1, 1, 1]

                            if fit_resource_constraints(
                                option_list, 
                                PE_num_list, 
                                MAX_HBM_bank,
                                MAX_BRAM_18K,
                                MAX_DSP48E,
                                MAX_FF,
                                MAX_LUT,
                                MAX_URAM,
                                shell_consumption,
                                count_shell=True):

                                total_valid_design = total_valid_design + 1 # each valide design counts

                                bottleneck_ID, accelerator_QPS = get_bottleneck(option_list)
                                if accelerator_QPS > best_solution_QPS:
                                    best_solution_QPS = accelerator_QPS
                                    best_solution_stage_option_list = option_list
                                    best_solution_PE_num_list = PE_num_list
                                elif accelerator_QPS == best_solution_QPS:
                                    if resource_consumption_A_less_than_B(
                                        option_list, PE_num_list,
                                        best_solution_stage_option_list, best_solution_PE_num_list):
                                            best_solution_QPS = accelerator_QPS
                                            best_solution_stage_option_list = option_list
                                            best_solution_PE_num_list = PE_num_list

    return best_solution_QPS, best_solution_stage_option_list, best_solution_PE_num_list

if __name__ == "__main__":

    # unit_test(FREQ, MAX_URAM, MAX_HBM_bank)
    best_solution_name = None
    best_solution_nlist = None
    best_solution_nprobe = None
    best_solution_OPQ_enable = None
    best_solution_QPS = 0 
    best_solution_stage_option_list = None 
    best_solution_PE_num_list = None


    if args.mode == 1 or args.mode == 2:
        
        d_nprobes = None
        if os.path.exists(args.nprobe_dict_dir):
            with open(args.nprobe_dict_dir, 'rb') as f:
                d_nprobes = pickle.load(f)
        else:
            print("ERROR! input dictionary does not exists")
            raise ValueError

        for index_key in d_nprobes[args.dbname]:

            index_array = index_key.split(",")
            if len(index_array) == 2: # "IVF4096,PQ16" 
                s = index_array[0]
                if s[:3]  == "IVF":
                    nlist = int(s[3:])
                    OPQ_enable = False
                elif s[:3] == "IMI":
                    continue
                else:
                    raise ValueError
            elif len(index_array) == 3: # "OPQ16,IVF4096,PQ16"
                s = index_array[1]
                if s[:3]  == "IVF":
                    nlist = int(s[3:])
                    OPQ_enable = True
                elif s[:3] == "IMI":
                    continue
                else:
                    raise ValueError
            else:
                raise ValueError

            if args.mode == 2 and (nlist != args.nlist or OPQ_enable != args.OPQ_enable):
                continue

            nprobe = d_nprobes[args.dbname][index_key][topK][recall_goal]
            if not nprobe: # nprobe can be None if it cannot reach the recall goal
                continue

            print(index_key, nlist, nprobe, OPQ_enable)
            current_solution_QPS, current_solution_stage_option_list, current_solution_PE_num_list = \
                get_best_hardware(nlist=nlist, nprobe=nprobe, OPQ_enable=OPQ_enable)

            if current_solution_QPS > best_solution_QPS:
                best_solution_name = index_key
                best_solution_nlist = nlist
                best_solution_nprobe = nprobe
                best_solution_OPQ_enable=OPQ_enable
                best_solution_QPS = current_solution_QPS
                best_solution_stage_option_list = current_solution_stage_option_list
                best_solution_PE_num_list = current_solution_PE_num_list
                
            print("index key", index_key)
            print("QPS", current_solution_QPS)
            print("stage_option_list")
            for option in current_solution_stage_option_list:
                option.print_attributes()
    elif args.mode == 3:
        current_solution_QPS, current_solution_stage_option_list, current_solution_PE_num_list = \
            get_best_hardware(nlist=args.nlist, nprobe=args.nprobe, OPQ_enable=args.OPQ_enable)

        if current_solution_QPS > best_solution_QPS:
            best_solution_name = "nlist_{}_nprobe_{}_OPQ_{}".format(args.nlist, args.nprobe, args.OPQ_enable)
            best_solution_nlist = args.nlist
            best_solution_nprobe = args.nprobe
            best_solution_OPQ_enable = args.OPQ_enable
            best_solution_QPS = current_solution_QPS
            best_solution_stage_option_list = current_solution_stage_option_list
            best_solution_PE_num_list = current_solution_PE_num_list
            
        print("QPS", current_solution_QPS)
        print("stage_option_list")
        for option in current_solution_stage_option_list:
            option.print_attributes()
    else:
        print("Supported mode: 1 / 2 / 3")
        raise ValueError

    # Redirect the best option print to config, also show in stdout
    # https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call

    f = io.StringIO()
    with redirect_stdout(f):
        print("\n\n======== Result =========\n")
        print("best option name", best_solution_name)
        if args.mode == 1 or args.mode == 2:
            print("nprobe: {}".format(d_nprobes[args.dbname][best_solution_name][topK][recall_goal]))
        else:
            print("nprobe: {}".format(args.nprobe))
        print("QPS", best_solution_QPS)
        print("stage_option_list")
        for option in best_solution_stage_option_list:
            option.print_attributes()
        print("total_valid_design:", total_valid_design)

        if len(best_solution_stage_option_list) == 5: # without OPQ
            total_consumption_obj = get_resource_consumption(
                best_solution_stage_option_list, 
                TOTAL_BRAM_18K, 
                TOTAL_DSP48E, 
                TOTAL_FF, 
                TOTAL_LUT, 
                TOTAL_URAM,
                PE_num_list=[1,1,1,1,1], 
                shell_consumption=shell_consumption,
                count_shell=True)
            total_consumption_obj_accelerator_kernel = get_resource_consumption(
                best_solution_stage_option_list, 
                TOTAL_BRAM_18K, 
                TOTAL_DSP48E, 
                TOTAL_FF, 
                TOTAL_LUT, 
                TOTAL_URAM,
                PE_num_list=[1,1,1,1,1], 
                shell_consumption=shell_consumption,
                count_shell=False)
            print("Total resource consumption: (with shell)")
            total_consumption_obj.print_resource()
            print("Total resource consumption: (accelerator kernel without shell)")
            total_consumption_obj_accelerator_kernel.print_resource()
            print("Per Stage Utilization rate (Total = FPGA):\n")
            for i in range(len(best_solution_stage_option_list)):
                print("Stage {}: {}".format(
                    i + 2, 
                    get_utilization_rate(
                        best_solution_stage_option_list[i],
                        TOTAL_BRAM_18K, 
                        TOTAL_DSP48E, 
                        TOTAL_FF, 
                        TOTAL_LUT, 
                        TOTAL_URAM)))
            print("Per Stage Utilization rate (Total = Accelerator Kernel (without shell)):\n")
            for i in range(len(best_solution_stage_option_list)):
                print("Stage {}: {}".format(
                    i + 2, 
                    get_utilization_rate(
                        best_solution_stage_option_list[i],
                        total_consumption_obj_accelerator_kernel.BRAM_18K, 
                        total_consumption_obj_accelerator_kernel.DSP48E, 
                        total_consumption_obj_accelerator_kernel.FF, 
                        total_consumption_obj_accelerator_kernel.LUT, 
                        total_consumption_obj_accelerator_kernel.URAM)))
                print("LUT only:")
                print("Stage {}: {}".format(
                    i + 2, 
                    get_utilization_rate(
                        best_solution_stage_option_list[i],
                        total_consumption_obj_accelerator_kernel.BRAM_18K, 
                        total_consumption_obj_accelerator_kernel.DSP48E, 
                        total_consumption_obj_accelerator_kernel.FF, 
                        total_consumption_obj_accelerator_kernel.LUT, 
                        total_consumption_obj_accelerator_kernel.URAM)['LUT'])) 
            print("Total Utilization rate:\n{}".format(get_utilization_rate(
                total_consumption_obj,
                TOTAL_BRAM_18K, 
                TOTAL_DSP48E, 
                TOTAL_FF, 
                TOTAL_LUT, 
                TOTAL_URAM)))
        elif len(best_solution_stage_option_list) == 6: # with OPQ
            total_consumption_obj = get_resource_consumption(
                best_solution_stage_option_list, 
                TOTAL_BRAM_18K, 
                TOTAL_DSP48E, 
                TOTAL_FF, 
                TOTAL_LUT, 
                TOTAL_URAM,
                PE_num_list=[1,1,1,1,1,1], 
                shell_consumption=shell_consumption,
                count_shell=True)
            total_consumption_obj_accelerator_kernel = get_resource_consumption(
                best_solution_stage_option_list, 
                TOTAL_BRAM_18K, 
                TOTAL_DSP48E, 
                TOTAL_FF, 
                TOTAL_LUT, 
                TOTAL_URAM,
                PE_num_list=[1,1,1,1,1,1], 
                shell_consumption=shell_consumption,
                count_shell=False)
            print("Total resource consumption: (with shell)")
            total_consumption_obj.print_resource()
            print("Total resource consumption: (accelerator kernel without shell)")
            total_consumption_obj_accelerator_kernel.print_resource()
            print("\nPer Stage Utilization rate (Total = FPGA):")
            for i in range(len(best_solution_stage_option_list)):
                print("Stage {}: {}".format(
                    i + 1, 
                    get_utilization_rate(
                        best_solution_stage_option_list[i],
                        TOTAL_BRAM_18K, 
                        TOTAL_DSP48E, 
                        TOTAL_FF, 
                        TOTAL_LUT, 
                        TOTAL_URAM)))
            print("\nPer Stage Utilization rate (Total = Accelerator Kernel (without shell)):")
            for i in range(len(best_solution_stage_option_list)):
                print("Stage {}: {}".format(
                    i + 1, 
                    get_utilization_rate(
                        best_solution_stage_option_list[i],
                        total_consumption_obj_accelerator_kernel.BRAM_18K, 
                        total_consumption_obj_accelerator_kernel.DSP48E, 
                        total_consumption_obj_accelerator_kernel.FF, 
                        total_consumption_obj_accelerator_kernel.LUT, 
                        total_consumption_obj_accelerator_kernel.URAM))) 
                print("LUT only:")
                print("Stage {}: {}".format(
                    i + 1, 
                    get_utilization_rate(
                        best_solution_stage_option_list[i],
                        total_consumption_obj_accelerator_kernel.BRAM_18K, 
                        total_consumption_obj_accelerator_kernel.DSP48E, 
                        total_consumption_obj_accelerator_kernel.FF, 
                        total_consumption_obj_accelerator_kernel.LUT, 
                        total_consumption_obj_accelerator_kernel.URAM)['LUT'])) 
            print("\nTotal Utilization rate:\n{}".format(get_utilization_rate(
                total_consumption_obj,
                TOTAL_BRAM_18K, 
                TOTAL_DSP48E, 
                TOTAL_FF, 
                TOTAL_LUT, 
                TOTAL_URAM)))
    s_best_option = f.getvalue()
    print(s_best_option)

    s_best_option_comment = ""
    for line in s_best_option.splitlines():
        s_best_option_comment += "  # {}\n".format(line) 

    """ 
    Write config file 
    """
    template_dir = './config_template.yaml'
    template_str = None
    with open(template_dir) as f:
        template_str = f.read()

    template_fill_dict = dict()
    template_fill_dict["best_option"] = s_best_option_comment
    template_fill_dict["nlist"] = best_solution_nlist
    template_fill_dict["nprobe"] = best_solution_nprobe
    template_fill_dict["topk"] = topK
    # stage 1
    template_fill_dict["OPQ_enable"] = best_solution_OPQ_enable
    if best_solution_OPQ_enable:
        array_offset = -1
        template_fill_dict["OPQ_unroll_factor"] = best_solution_stage_option_list[1 + array_offset].OPQ_UNROLL_FACTOR
    else:
        array_offset = -2
    # stage 2
    template_fill_dict["STAGE2_ON_CHIP"] = best_solution_stage_option_list[2 + array_offset].STAGE2_ON_CHIP
    template_fill_dict["PE_NUM_CENTER_DIST_COMP"] = best_solution_stage_option_list[2 + array_offset].PE_NUM_CENTER_DIST_COMP
    # stage 3
    template_fill_dict["STAGE_3_PRIORITY_QUEUE_LEVEL"] = best_solution_stage_option_list[3 + array_offset].STAGE_3_PRIORITY_QUEUE_LEVEL
    template_fill_dict["STAGE_3_PRIORITY_QUEUE_L1_NUM"] = best_solution_stage_option_list[3 + array_offset].STAGE_3_PRIORITY_QUEUE_L1_NUM
    # stage 4
    template_fill_dict["PE_NUM_TABLE_CONSTRUCTION"] = best_solution_stage_option_list[4 + array_offset].PE_NUM_TABLE_CONSTRUCTION
    # stage 5
    template_fill_dict["HBM_CHANNEL_NUM"] = best_solution_stage_option_list[5 + array_offset].HBM_CHANNEL_NUM
    template_fill_dict["STAGE5_COMP_PE_NUM"] = int(best_solution_stage_option_list[5 + array_offset].STAGE5_COMP_PE_NUM)
    # stage 6
    template_fill_dict["SORT_GROUP_ENABLE"] = best_solution_stage_option_list[6 + array_offset].SORT_GROUP_ENABLE
    if best_solution_stage_option_list[6 + array_offset].SORT_GROUP_ENABLE:
        template_fill_dict["SORT_GROUP_NUM"] = best_solution_stage_option_list[6 + array_offset].SORT_GROUP_NUM
    else:
        template_fill_dict["SORT_GROUP_NUM"] = 0
    template_fill_dict["STAGE_6_PRIORITY_QUEUE_LEVEL"] = best_solution_stage_option_list[6 + array_offset].STAGE_6_PRIORITY_QUEUE_LEVEL


    template_fill_dict["DB_NAME"] = args.dbname
    if args.dbname.startswith('SIFT'):
        template_fill_dict["D"] = 128
    elif args.dbname.startswith('Deep'):
        template_fill_dict["D"] = 96
    else:
        print('unknown dataset', args.dbname)
        raise ValueError

    if args.dbname == 'SIFT100M' or args.dbname == 'Deep100M':
        template_fill_dict["DB_SCALE"] = "100M"
    elif args.dbname == 'SIFT500M':
        template_fill_dict["DB_SCALE"] = "500M"
    elif args.dbname == 'SIFT1000M':
        template_fill_dict["DB_SCALE"] = "1000M"
    else:
        print("Unknown db name, unable to write DB_SCALE to config")
        raise ValueError
    
    template_fill_dict["FPGA_NUM"] = args.FPGA_num
    template_fill_dict["DEVICE"] = args.device
    template_fill_dict["FREQ"] = args.freq

    for k in template_fill_dict:
        template_str = template_str.replace("<--{}-->".format(k), str(template_fill_dict[k]))
    output_str = template_str

    # Save generated file
    if args.mode == 1:
        output_dir = './config_outputs/{device}_{dbname}_K{topK}_R{recall}_util_{max_utilization_rate}_{freq}MHz.yaml'.format(
            device=args.device, dbname=args.dbname, topK=args.topK, recall=int(args.recall_goal * 100), 
            max_utilization_rate=args.max_utilization_rate, freq=args.freq)
    elif args.mode == 2:
        if args.OPQ_enable:
            OPQ_enable = 1
        else:
            OPQ_enable = 0
        output_dir = './config_outputs/{device}_{dbname}_K{topK}_R{recall}_nlist_{nlist}_OPQ_{OPQ}_util_{max_utilization_rate}_{freq}MHz.yaml'.format(
            device=args.device, dbname=args.dbname, topK=args.topK, recall=int(args.recall_goal * 100), 
            nlist=args.nlist, OPQ=OPQ_enable, max_utilization_rate=args.max_utilization_rate, freq=args.freq)
    elif args.mode == 3:
        if args.OPQ_enable:
            OPQ_enable = 1
        else:
            OPQ_enable = 0
        output_dir = './config_outputs/{device}_{dbname}_K{topK}_nlist_{nlist}_nprobe_{nprobe}_OPQ_{OPQ}_util_{max_utilization_rate}_{freq}MHz.yaml'.format(
            device=args.device, dbname=args.dbname, topK=args.topK,  nlist=args.nlist, nprobe=args.nprobe,
            OPQ=OPQ_enable, max_utilization_rate=args.max_utilization_rate, freq=args.freq)
    else:
        print("Supported mode: 1 / 2/ 3")
        raise ValueError

    with open(output_dir, "w+") as f:
        f.write(output_str)
