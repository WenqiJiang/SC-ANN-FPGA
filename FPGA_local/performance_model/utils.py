""" Helper functions & Unit test """

from constants import *

def max_of_two(a, b):
    if a >= b:
        return a
    else:
        return b

def max_of_three(a, b, c):
    if a >= b:
        if a >= c:
            return a
        else: 
            return c
    else:
        if b >= c:
            return b
        else:
            return c

def get_bottleneck(perf_resource_obj_list):
    """
    Given a list of stages (each stage is a perf_resource_obj),
       return (a) which stage (ID in the list) is the bottleneck
              (b) the overall accelerator QPS
    """
    min_QPS = 9999999999
    min_QPS_ID = 0

    for i, perf_resource_obj in enumerate(perf_resource_obj_list):
        if perf_resource_obj.QPS < min_QPS:
            min_QPS = perf_resource_obj.QPS 
            min_QPS_ID = i

    assert min_QPS != 9999999999
    accelerator_QPS = min_QPS

    return min_QPS_ID, accelerator_QPS

def resource_consumption_A_less_than_B(
    perf_resource_obj_list_A, PE_num_list_A,
    perf_resource_obj_list_B, PE_num_list_B):

    resource_A = Resource()
    resource_B = Resource()


    for i, perf_resource_obj in enumerate(perf_resource_obj_list_A):

        resource_A.HBM_bank = resource_A.HBM_bank + perf_resource_obj.HBM_bank * PE_num_list_A[i]
        resource_A.BRAM_18K = resource_A.BRAM_18K + perf_resource_obj.BRAM_18K * PE_num_list_A[i] 
        resource_A.DSP48E = resource_A.DSP48E + perf_resource_obj.DSP48E * PE_num_list_A[i] 
        resource_A.FF = resource_A.FF + perf_resource_obj.FF * PE_num_list_A[i] 
        resource_A.LUT = resource_A.LUT + perf_resource_obj.LUT * PE_num_list_A[i] 
        resource_A.URAM = resource_A.URAM + perf_resource_obj.URAM * PE_num_list_A[i] 
    
    for i, perf_resource_obj in enumerate(perf_resource_obj_list_B):

        resource_B.HBM_bank = resource_B.HBM_bank + perf_resource_obj.HBM_bank * PE_num_list_B[i]
        resource_B.BRAM_18K = resource_B.BRAM_18K + perf_resource_obj.BRAM_18K * PE_num_list_B[i] 
        resource_B.DSP48E = resource_B.DSP48E + perf_resource_obj.DSP48E * PE_num_list_B[i] 
        resource_B.FF = resource_B.FF + perf_resource_obj.FF * PE_num_list_B[i] 
        resource_B.LUT = resource_B.LUT + perf_resource_obj.LUT * PE_num_list_B[i] 
        resource_B.URAM = resource_B.URAM + perf_resource_obj.URAM * PE_num_list_B[i] 
    
    # Priority: LUT is the most important one, then HBM bank consumption
    if resource_A.LUT < resource_B.LUT:
        return True
    elif resource_A.LUT == resource_B.LUT:
        if resource_A.HBM_bank < resource_B.HBM_bank:
            return True
        else:
            return False
    else:
        return False

def fit_resource_constraints(
    perf_resource_obj_list, 
    PE_num_list, 
    MAX_HBM_bank,
    MAX_BRAM_18K,
    MAX_DSP48E,
    MAX_FF,
    MAX_LUT,
    MAX_URAM,
    shell_consumption=None,
    count_shell=False):
    """
    Given a list of stages (each stage is a perf_resource_obj),
       return whether it is within the resource constraint
    """
    resource_all = Resource()

    for i, perf_resource_obj in enumerate(perf_resource_obj_list):

        resource_all.HBM_bank += perf_resource_obj.HBM_bank * PE_num_list[i]
        resource_all.BRAM_18K += perf_resource_obj.BRAM_18K * PE_num_list[i] 
        resource_all.DSP48E += perf_resource_obj.DSP48E * PE_num_list[i] 
        resource_all.FF += perf_resource_obj.FF * PE_num_list[i] 
        resource_all.LUT += perf_resource_obj.LUT * PE_num_list[i] 
        resource_all.URAM += perf_resource_obj.URAM * PE_num_list[i] 

    if count_shell:
        resource_all.HBM_bank += shell_consumption.HBM_bank
        resource_all.BRAM_18K += shell_consumption.BRAM_18K
        resource_all.DSP48E += shell_consumption.DSP48E
        resource_all.FF += shell_consumption.FF
        resource_all.LUT += shell_consumption.LUT
        resource_all.URAM += shell_consumption.URAM

    if resource_all.HBM_bank <= MAX_HBM_bank and resource_all.BRAM_18K <= MAX_BRAM_18K and \
        resource_all.DSP48E <= MAX_DSP48E and resource_all.FF <= MAX_FF and \
        resource_all.LUT < MAX_LUT and resource_all.URAM < MAX_URAM:
        return True
    else: 
        return False


def get_resource_consumption(
    perf_resource_obj_list, 
    TOTAL_BRAM_18K, 
    TOTAL_DSP48E, 
    TOTAL_FF, 
    TOTAL_LUT, 
    TOTAL_URAM,
    PE_num_list, 
    shell_consumption=None,
    count_shell=False):
    """
    Given a list of stages (each stage is a perf_resource_obj),
       return the resource consumption dictionary
    """
    resource = Resource()

    for i, perf_resource_obj in enumerate(perf_resource_obj_list):

        resource.HBM_bank += perf_resource_obj.HBM_bank * PE_num_list[i]
        resource.BRAM_18K += perf_resource_obj.BRAM_18K * PE_num_list[i] 
        resource.DSP48E += perf_resource_obj.DSP48E * PE_num_list[i] 
        resource.FF += perf_resource_obj.FF * PE_num_list[i] 
        resource.LUT += perf_resource_obj.LUT * PE_num_list[i] 
        resource.URAM += perf_resource_obj.URAM * PE_num_list[i] 

    if count_shell:
        resource.HBM_bank += shell_consumption.HBM_bank
        resource.BRAM_18K += shell_consumption.BRAM_18K
        resource.DSP48E += shell_consumption.DSP48E
        resource.FF += shell_consumption.FF
        resource.LUT += shell_consumption.LUT
        resource.URAM += shell_consumption.URAM

    return resource

def get_utilization_rate(
    perf_resource_obj,
    TOTAL_BRAM_18K, 
    TOTAL_DSP48E, 
    TOTAL_FF, 
    TOTAL_LUT, 
    TOTAL_URAM):

    utilization_rate = dict()

    utilization_rate["BRAM_18K"] = "{}%".format(perf_resource_obj.BRAM_18K / TOTAL_BRAM_18K * 100)
    utilization_rate["DSP48E"] = "{}%".format(perf_resource_obj.DSP48E / TOTAL_DSP48E * 100)
    utilization_rate["FF"] = "{}%".format(perf_resource_obj.FF / TOTAL_FF * 100)
    utilization_rate["LUT"] = "{}%".format(perf_resource_obj.LUT / TOTAL_LUT * 100)
    utilization_rate["URAM"] = "{}%".format(perf_resource_obj.URAM / TOTAL_URAM * 100)

    return utilization_rate

def unit_test(FREQ, MAX_URAM, MAX_HBM_bank):
    """ Print the options of each function unit """

    print("\nget_priority_queue_info:\n")
    perf_resource_obj = get_priority_queue_info(queue_len=32, N_insertion=8192, FREQ=FREQ)
    print(perf_resource_obj)

    print("\nget_bitonic_sort_16_info:\n")
    perf_resource_obj = get_bitonic_sort_16_info(N_insertion=1e8/8192/64*32, FREQ=FREQ)
    print(perf_resource_obj)

    print("\nget_parallel_merge_32_to_16_info:\n")
    perf_resource_obj = get_parallel_merge_32_to_16_info(N_insertion=1e8/8192/64*32, FREQ=FREQ)
    print(perf_resource_obj)

    option_list = get_options_stage_1_OPQ(FREQ)
    print("\nget_options_stage_1_OPQ:\n")
    for option in option_list:
        option.print_attributes()

    print("\nget_options_stage_2_cluster_distance_computation:\n")
    nlist_options = [2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18]
    for nlist in nlist_options:
        option_list = get_options_stage_2_cluster_distance_computation(nlist, FREQ, MAX_URAM)
        print("nlist={}".format(nlist))
        for option in option_list:
            option.print_attributes()

    print("\nget_options_stage_3_select_Voronoi_cells:\n")
    option_list = get_options_stage_3_select_Voronoi_cells(nlist=8192, nprobe=32, FREQ=FREQ)
    for option in option_list:
        option.print_attributes()

    print("\nget_options_stage_4_distance_LUT_construction:\n")    
    option_list = get_options_stage_4_distance_LUT_construction(nlist=8192, nprobe=32, FREQ=FREQ)
    for option in option_list:
        option.print_attributes()

    print("\nget_options_stage_5_distance_estimation_by_LUT:\n")    
    option_list = get_options_stage_5_distance_estimation_by_LUT(
        nlist=8192, nprobe=32, FREQ=FREQ, MAX_HBM_bank=MAX_HBM_bank, OPQ_enable=True)
    for option in option_list:
        option.print_attributes()

    # for a small amount of number being scanned, hierachical priority queue is not
    #   really an option
    print("\nget_options_stage6_select_topK:\n")    
    # for small number of Voronoi cells, only 2 level is required
    print("nlist=8192, nprobe=32, nstreams=64")
    option_list = get_options_stage6_select_topK(
        input_stream_num=64, 
        N_insertion_per_stream=int(1e8/8192*32/64),
        topK=10, FREQ=FREQ)
    for option in option_list:
        option.print_attributes()
    # for large number of Voronoi cells, 4 level is required
    print("nlist=262144, nprobe=32, nstreams=64")
    option_list = get_options_stage6_select_topK(
        input_stream_num=64, 
        N_insertion_per_stream=int(1e8/262144*32/64),
        topK=10, FREQ=FREQ)
    for option in option_list:
        option.print_attributes()
    # try different stream num
    print("nlist=8192, nprobe=32, nstreams=48")
    option_list = get_options_stage6_select_topK(
        input_stream_num=48, 
        N_insertion_per_stream=int(1e8/8192*32/64),
        topK=10, FREQ=FREQ)
    for option in option_list:
        option.print_attributes()
    print("nlist=8192, nprobe=32, nstreams=32")
    option_list = get_options_stage6_select_topK(
        input_stream_num=32, 
        N_insertion_per_stream=int(1e8/8192*32/64),
        topK=10, FREQ=FREQ)
    for option in option_list:
        option.print_attributes()
    print("nlist=8192, nprobe=32, nstreams=16")
    option_list = get_options_stage6_select_topK(
        input_stream_num=16, 
        N_insertion_per_stream=int(1e8/8192*32/64),
        topK=10, FREQ=FREQ)
    for option in option_list:
        option.print_attributes()
