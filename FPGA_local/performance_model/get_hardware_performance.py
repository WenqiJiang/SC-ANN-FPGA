"""
Input a set of hardware settings (config.yaml), predict the performance and resource 
    consumption by the performance  model.

Example usage:
    python get_hardware_performance.py --config_dir './config.yaml' > out
"""

import numpy as np
import os 
import pickle
import yaml

from constants import * 
from queue_and_sorting import *
from stages import *
from utils import *

import argparse 

parser = argparse.ArgumentParser()
# DB-related parameters
parser.add_argument('--config_dir', type=str, default='./config.yaml', help="input config dir")
args = parser.parse_args()

# Load YAML configurations
config_file = open(args.config_dir, "r")
config = yaml.load(config_file)

nlist = config["NLIST"]
nprobe = config["NPROBE"]
OPQ_enable = config["OPQ_ENABLE"]
topK = config["TOPK"]
FREQ = config["FREQ"] * 1e6
device = config["DEVICE"]
DB_SCALE = config["DB_SCALE"]
FPGA_num = config["FPGA_NUM"]

MAX_UTIL_PERC = 1 # no resource constraint applied herer

if DB_SCALE == '100M':
    dbname = 'SIFT100M'
    TOTAL_VECTORS = int(1e8 / FPGA_num)
elif DB_SCALE == '500M':
    dbname = 'SIFT500M'
    TOTAL_VECTORS = int(5e8 / FPGA_num)
elif DB_SCALE == '1000M':
    dbname = 'SIFT1000M'
    TOTAL_VECTORS = int(1e9 / FPGA_num)
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

if device == 'U280':
    """ Resource related constants """
    TOTAL_BRAM_18K = 4032 
    TOTAL_DSP48E = 9024
    TOTAL_FF = 2607360 
    TOTAL_LUT = 1303680
    TOTAL_URAM = 960
    
    MAX_HBM_bank = 32 - 2 - 2 - 1 # reserve 30, 31 unused due to their overlap with PCIe; 2 for Network; 1 for value init
    MAX_BRAM_18K = TOTAL_BRAM_18K * MAX_UTIL_PERC
    MAX_DSP48E = TOTAL_DSP48E * MAX_UTIL_PERC
    MAX_FF = TOTAL_FF * MAX_UTIL_PERC
    MAX_LUT = TOTAL_LUT * MAX_UTIL_PERC
    MAX_URAM = TOTAL_URAM * MAX_UTIL_PERC

    if dbname == 'SIFT100M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 100 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / FPGA_num)) # at least 9 banks to hold PQ16 version
    elif dbname == 'SIFT500M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 500 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / FPGA_num))
    elif dbname == 'SIFT1000M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 1000 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / FPGA_num))
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

elif device == 'U50':
    """ Resource related constants """
    TOTAL_BRAM_18K = 2688 
    TOTAL_DSP48E = 5952
    TOTAL_FF = 1743360 
    TOTAL_LUT = 871680
    TOTAL_URAM = 640

    MAX_HBM_bank = 32 - 2 # reserve 30, 31 unused due to their overlap with PCIe
    MAX_BRAM_18K = TOTAL_BRAM_18K * MAX_UTIL_PERC
    MAX_DSP48E = TOTAL_DSP48E * MAX_UTIL_PERC
    MAX_FF = TOTAL_FF * MAX_UTIL_PERC
    MAX_LUT = TOTAL_LUT * MAX_UTIL_PERC
    MAX_URAM = TOTAL_URAM * MAX_UTIL_PERC

    if dbname == 'SIFT100M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 100 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / FPGA_num)) # at least 9 banks to hold PQ16 version
    elif dbname == 'SIFT500M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 500 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / FPGA_num))
    elif dbname == 'SIFT1000M':
        # 1 Bank = 256 MB = 4194304 512-bit = 4194304 * 3 = 12582912 vectors
        # 100M / 12582912 = 7.94 (without considering padding)
        padding_factor = 1.05
        total_size = 1000 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 256 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / FPGA_num))
    else:
        print("Unsupported dataset")
        raise ValueError

    #####     Shell     #####

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

elif device == 'U250':
    """ Resource related constants """
    TOTAL_BRAM_18K = 5376 
    TOTAL_DSP48E = 12288
    TOTAL_FF = 3456000 
    TOTAL_LUT = 1728000
    TOTAL_URAM = 1280

    MAX_HBM_bank = 4 # reserve 30, 31 unused due to their overlap with PCIe; 2 for Network; 1 for value init
    MAX_BRAM_18K = TOTAL_BRAM_18K * MAX_UTIL_PERC
    MAX_DSP48E = TOTAL_DSP48E * MAX_UTIL_PERC
    MAX_FF = TOTAL_FF * MAX_UTIL_PERC
    MAX_LUT = TOTAL_LUT * MAX_UTIL_PERC
    MAX_URAM = TOTAL_URAM * MAX_UTIL_PERC

    # 1 DRAM bank = 16 GB
    if dbname == 'SIFT100M':
        padding_factor = 1.05
        total_size = 100 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 16 * 1024 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / FPGA_num)) # at least 9 banks to hold PQ16 version
    elif dbname == 'SIFT500M':
        padding_factor = 1.05
        total_size = 500 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 16 * 1024 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / FPGA_num))
    elif dbname == 'SIFT1000M':
        padding_factor = 1.05
        total_size = 1000 * 1e6 * 20 * 64 / 60 * padding_factor
        per_bank_size = 16 * 1024 * 1024 * 1024
        MIN_HBM_bank = int(np.ceil(total_size / per_bank_size / FPGA_num))
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


def get_hardware_performance_resource(config):
    """
    given a single hardware setting, return the predicted performance & resource
    """

    options_stage_1_OPQ = get_options_stage_1_OPQ(FREQ)
    options_stage_2_cluster_distance_computation = \
        get_options_stage_2_cluster_distance_computation(nlist, FREQ, MAX_URAM)
    options_stage_3_select_Voronoi_cells = get_options_stage_3_select_Voronoi_cells(nlist, nprobe, FREQ)
    options_stage_4_distance_LUT_construction = get_options_stage_4_distance_LUT_construction(nlist, nprobe, FREQ)
    options_stage_5_distance_estimation_by_LUT = get_options_stage_5_distance_estimation_by_LUT(nlist, nprobe, FREQ, MIN_HBM_bank, MAX_HBM_bank, TOTAL_VECTORS, scan_ratio_with_OPQ, scan_ratio_without_OPQ, OPQ_enable)

    # stage 6 options depend on stage 5 PE num
    stage_5_PE_num = config["STAGE5_COMP_PE_NUM"]
    if OPQ_enable:
        N_compute_per_nprobe = int(scan_ratio_with_OPQ[nlist] * TOTAL_VECTORS / nlist / stage_5_PE_num) + 1
    else:
        N_compute_per_nprobe = int(scan_ratio_without_OPQ[nlist] * TOTAL_VECTORS / nlist / stage_5_PE_num) + 1
    N_insertion_per_stream = int(nprobe * N_compute_per_nprobe)
    
    options_stage_6_sort_reduction = get_options_stage6_select_topK(
        stage_5_PE_num, N_insertion_per_stream, topK, FREQ)


    stage_option_list = []
    PE_num_list = []

    if OPQ_enable:
        option_stage_1 = None
        for option in options_stage_1_OPQ:
            if option.OPQ_ENABLE == config["OPQ_ENABLE"] and \
                option.OPQ_UNROLL_FACTOR == config["OPQ_UNROLL_FACTOR"]:
                option_stage_1 = option
                break
        if not option_stage_1:
            print("Did not find matched option between input template and performance model function")
            raise ValueError
        stage_option_list.append(option_stage_1)
        PE_num_list.append(1)

    option_stage_2 = None
    for option in options_stage_2_cluster_distance_computation:
        if option.STAGE2_ON_CHIP == config["STAGE2_ON_CHIP"] and \
            option.PE_NUM_CENTER_DIST_COMP == config["PE_NUM_CENTER_DIST_COMP"]:
            option_stage_2 = option
            print("FIND SOLUTION")
            break
        # else:
        #     print(" ==  ")
        #     print(option.STAGE2_ON_CHIP, config["STAGE2_ON_CHIP"], option.STAGE2_ON_CHIP == config["STAGE2_ON_CHIP"])
        #     print(option.STAGE2_OFF_CHIP_START_CHANNEL, config["STAGE2_OFF_CHIP_START_CHANNEL"], option.STAGE2_OFF_CHIP_START_CHANNEL == config["STAGE2_OFF_CHIP_START_CHANNEL"])
        #     print(option.PE_NUM_CENTER_DIST_COMP, config["PE_NUM_CENTER_DIST_COMP"], option.PE_NUM_CENTER_DIST_COMP == config["PE_NUM_CENTER_DIST_COMP"])
        #     print(" ==  ")

    if not option_stage_2:
        print("Did not find matched option between input template and performance model function")
        raise ValueError
    stage_option_list.append(option_stage_2)
    PE_num_list.append(1)

    option_stage_3 = None
    for option in options_stage_3_select_Voronoi_cells:
        if option.STAGE_3_PRIORITY_QUEUE_LEVEL == config["STAGE_3_PRIORITY_QUEUE_LEVEL"] and \
            option.STAGE_3_PRIORITY_QUEUE_L1_NUM == config["STAGE_3_PRIORITY_QUEUE_L1_NUM"]:
            option_stage_3 = option
            break
    if not option_stage_3:
        print("Did not find matched option between input template and performance model function")
        raise ValueError
    stage_option_list.append(option_stage_3)
    PE_num_list.append(1)

    option_stage_4 = None
    for option in options_stage_4_distance_LUT_construction:
        if option.PE_NUM_TABLE_CONSTRUCTION == config["PE_NUM_TABLE_CONSTRUCTION"]:
            option_stage_4 = option
            break
    if not option_stage_4:
        print("Did not find matched option between input template and performance model function")
        raise ValueError
    stage_option_list.append(option_stage_4)
    PE_num_list.append(1)

    option_stage_5 = None
    for option in options_stage_5_distance_estimation_by_LUT:
        if option.HBM_CHANNEL_NUM == config["HBM_CHANNEL_NUM"] and \
            option.STAGE5_COMP_PE_NUM == config["STAGE5_COMP_PE_NUM"]:
            option_stage_5 = option
            break
    if not option_stage_5:
        print("Did not find matched option between input template and performance model function")
        raise ValueError
    stage_option_list.append(option_stage_5)
    PE_num_list.append(1)

    option_stage_6 = None
    for option in options_stage_6_sort_reduction:
        if option.SORT_GROUP_ENABLE == config["SORT_GROUP_ENABLE"] and \
            (option.SORT_GROUP_NUM == config["SORT_GROUP_NUM"] or (not option.SORT_GROUP_NUM and not config["SORT_GROUP_NUM"])) and \
            option.STAGE_6_PRIORITY_QUEUE_LEVEL == config["STAGE_6_PRIORITY_QUEUE_LEVEL"]:

            if option.STAGE_6_PRIORITY_QUEUE_LEVEL == 2:
                option_stage_6 = option
                break
            elif option.STAGE_6_PRIORITY_QUEUE_LEVEL == 3:
                if option.STAGE_6_PRIORITY_QUEUE_L2_NUM == config["STAGE_6_PRIORITY_QUEUE_L2_NUM"] and \
                    option.STAGE_6_STREAM_PER_L2_QUEUE_LARGER == config["STAGE_6_STREAM_PER_L2_QUEUE_LARGER"] and \
                    option.STAGE_6_STREAM_PER_L2_QUEUE_SMALLER == config["STAGE_6_STREAM_PER_L2_QUEUE_SMALLER"]:
                    option_stage_6 = option
                    break
    if not option_stage_6:
        print("Did not find matched option between input template and performance model function")
        raise ValueError
    stage_option_list.append(option_stage_6)
    PE_num_list.append(1)

    bottleneck_ID, accelerator_QPS = get_bottleneck(stage_option_list)
        
    return accelerator_QPS, stage_option_list, PE_num_list

if __name__ == "__main__":

    accelerator_QPS, stage_option_list, PE_num_list = \
        get_hardware_performance_resource(config)

    print("\n\n======== Result =========\n")
    print("QPS", accelerator_QPS)
    print("stage_option_list")
    for option in stage_option_list:
        option.print_attributes()

    total_consumption_obj = get_resource_consumption(
        stage_option_list, 
        TOTAL_BRAM_18K, 
        TOTAL_DSP48E, 
        TOTAL_FF, 
        TOTAL_LUT, 
        TOTAL_URAM,
        PE_num_list=PE_num_list, 
        shell_consumption=shell_consumption,
        count_shell=True)
    print("Total resource consumption:")
    total_consumption_obj.print_resource()
    print("Utilization rate:\n{}".format(get_utilization_rate(
        total_consumption_obj,
        TOTAL_BRAM_18K, 
        TOTAL_DSP48E, 
        TOTAL_FF, 
        TOTAL_LUT, 
        TOTAL_URAM)))
        