""" The 6 Stages (optional preprocessing + 5 stages) """

import numpy as np
import copy

from constants import * 
from common import *
from queue_and_sorting import * 
from utils import * 

def get_options_stage_1_OPQ(FREQ):
    
    option_list = []

    """ Option 1: UNROLL 8 """
    perf_resource_obj = Resource_Performance_Stage1()
    perf_resource_obj.OPQ_ENABLE = True
    perf_resource_obj.OPQ_UNROLL_FACTOR = 8

    L_load = 128
    L_write = 128
    L_compute = 37
    UNROLL_FACTOR = 8
    II_compute = 1
    cycles_per_query = L_load + (L_compute + (D * D) / UNROLL_FACTOR * II_compute) + L_write
    QPS = 1 / (cycles_per_query / FREQ)
    perf_resource_obj.cycles_per_query = cycles_per_query
    perf_resource_obj.QPS = QPS
    
    #####  HLS Prediction #####
    perf_resource_obj.HBM_bank = 0
    perf_resource_obj.BRAM_18K = 33
    perf_resource_obj.DSP48E = 40
    perf_resource_obj.FF = 5134
    perf_resource_obj.LUT = 3660 
    perf_resource_obj.URAM = 0

    option_list.append(perf_resource_obj)

    """ Option 2: UNROLL 4 """
    perf_resource_obj = Resource_Performance_Stage1()
    perf_resource_obj.OPQ_ENABLE = True
    perf_resource_obj.OPQ_UNROLL_FACTOR = 4

    L_load = 128
    L_write = 128
    L_compute = 21
    UNROLL_FACTOR = 4
    II_compute = 1
    cycles_per_query = L_load + (L_compute + (D * D) / UNROLL_FACTOR * II_compute) + L_write
    QPS = 1 / (cycles_per_query / FREQ)
    perf_resource_obj.cycles_per_query = cycles_per_query
    perf_resource_obj.QPS = QPS
   
    #####  HLS Prediction ##### 
    perf_resource_obj.HBM_bank = 0
    perf_resource_obj.BRAM_18K = 37
    perf_resource_obj.DSP48E = 20
    perf_resource_obj.FF = 2659
    perf_resource_obj.LUT = 2152 
    perf_resource_obj.URAM = 0

    option_list.append(perf_resource_obj)

    return option_list

def get_options_stage_2_cluster_distance_computation(nlist, FREQ, MAX_URAM):
    
    """ The performance / resource of a single PE
        currently only the most optimized option is included:
        distance_computation_PE_systolic_optimized_perfect_loop_2 """

    option_list = []

    """ Systolic array, currently support up to 16 PEs (otherwise communication is the bottleneck) """
    MIN_PE_NUM = 1 
    MAX_PE_NUM = 16
    for PE_num in range(MIN_PE_NUM, MAX_PE_NUM + 1):

        perf_resource_obj = Resource_Performance_Stage2()
        perf_resource_obj.PE_NUM_CENTER_DIST_COMP = PE_num

        centroids_Per_PE = int(np.ceil(nlist / PE_num))
        N_comp = int(np.ceil(nlist * 8 / PE_num)) # 8 = SIMD width
        L_load = 128
        L_comp = 10
        II_comp = 2
        cycles_per_query = L_load + (L_comp + N_comp * II_comp)
        QPS = 1 / (cycles_per_query / FREQ)
        
        perf_resource_obj.cycles_per_query = cycles_per_query
        perf_resource_obj.QPS = QPS

        #####  HLS Estimation #####
        perf_resource_obj.LUT = 8946 * PE_num
        perf_resource_obj.FF = 11505 * PE_num
        perf_resource_obj.BRAM_18K = 2 * 0 * PE_num
        perf_resource_obj.DSP48E = 58 * PE_num

        #####   FIFO Consumption (Vivado Measured)   #####
        perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=1)
        perf_resource_obj.add_resource(resource_FIFO_d2_w32, num=4)

        # on-chip or off-chip storage given different nlist size
        i = np.ceil(centroids_Per_PE / 512) # centroids_Per_PE <= 512, URAM = 8
        perf_resource_obj_on_chip = copy.deepcopy(perf_resource_obj)
        perf_resource_obj_off_chip = copy.deepcopy(perf_resource_obj)
        if i * 8 * PE_num < MAX_URAM: # on-chip option available 
            perf_resource_obj_on_chip.STAGE2_ON_CHIP = True
            #####  HLS Prediction & Vivado Measured #####
            perf_resource_obj_on_chip.HBM_bank = 0
            perf_resource_obj_on_chip.URAM = i * 8 * PE_num
            option_list.append(perf_resource_obj_on_chip)
        # off-chip version
        #####  HLS Prediction & Vivado Measured #####
        perf_resource_obj_off_chip.STAGE2_ON_CHIP = False
        perf_resource_obj_off_chip.HBM_bank = int(np.ceil(PE_num/2))
        perf_resource_obj_off_chip.URAM = 0
        option_list.append(perf_resource_obj_off_chip)

    return option_list

def get_options_stage_3_select_Voronoi_cells(nlist, nprobe, FREQ):
    
    """ 
    Insertion_per_cycle should equal to the PE num of stage 2, suppose
        it can output 1 distance per cycle  
    Here, we try insertion_per_cycle from 1 to 8, that should cover most cases
        (larger insertion per cycle will lead to overutilization of stage 2 PEs)
    """

    option_list = []

    """ Option 1: single priority queue """
    perf_resource_obj = Resource_Performance_Stage3()
    perf_resource_obj.STAGE_3_PRIORITY_QUEUE_LEVEL = 1
    perf_resource_obj.STAGE_3_PRIORITY_QUEUE_L1_NUM = 1

    queue_size = nprobe
    N_insertion_per_queue = nlist
    priority_queue_perf_resource_obj = get_priority_queue_info(queue_size, N_insertion_per_queue, FREQ)
    perf_resource_obj.copy_from_Performance_Resource(priority_queue_perf_resource_obj)
    #####   FIFO Consumption (Vivado Measured)   #####
    perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=1)

    option_list.append(perf_resource_obj)
    
    """ Option 2: hierachical priority queue """
    # Up to 16 priority queues in Level A, but can be further increased (probably not useful)
    for insertion_per_cycle in range(1, 8 + 1):
        perf_resource_obj = Resource_Performance_Stage3()
        perf_resource_obj.STAGE_3_PRIORITY_QUEUE_LEVEL = 2

        queue_size_level_A = nprobe
        queue_size_level_B = nprobe
        # 2 level of queues, 
        #  the first level collect the output of stage 2 in parallel
        #  the second level collect the result of level 1
        queue_num_level_A = int(insertion_per_cycle * 2)
        perf_resource_obj.STAGE_3_PRIORITY_QUEUE_L1_NUM = queue_num_level_A
        queue_num_level_B = 1
        N_insertion_per_queue_level_A = int(nlist / queue_num_level_A)
        N_insertion_level_B = int(queue_num_level_A * queue_size_level_A)

        perf_resource_obj_level_A = get_priority_queue_info(queue_size_level_A, N_insertion_per_queue_level_A, FREQ)
        perf_resource_obj_level_B = get_priority_queue_info(queue_size_level_B, N_insertion_level_B, FREQ)

        if perf_resource_obj_level_A.cycles_per_query > perf_resource_obj_level_B.cycles_per_query:
            perf_resource_obj.cycles_per_query = perf_resource_obj_level_A.cycles_per_query 
        else:
            perf_resource_obj.cycles_per_query = perf_resource_obj_level_B.cycles_per_query 
        perf_resource_obj.QPS = 1 / (perf_resource_obj.cycles_per_query / FREQ) 
        
        total_resource = sum_resource([perf_resource_obj_level_A, perf_resource_obj_level_B], [queue_num_level_A, queue_num_level_B])
        perf_resource_obj.copy_from_Resource(total_resource)

        #####   FIFO Consumption (Vivado Measured)   #####
        perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=(queue_num_level_A + queue_num_level_B) * 2)

        option_list.append(perf_resource_obj)

    return option_list

def get_options_stage_4_distance_LUT_construction(nlist, nprobe, FREQ):
    
    """ Now we only use the most optimized version, i.e.,
          multiple_lookup_table_construction_PEs_optimized_version4_systolic """

    option_list = []

    for PE_num in range(1, 128):

        perf_resource_obj = Resource_Performance_Stage4()
        perf_resource_obj.PE_NUM_TABLE_CONSTRUCTION = PE_num

        nprobe_per_PE_max = int(np.ceil(nprobe / PE_num))
        L_load_query = 128
        L_load_and_compute_residual = 132
        L_compute = 36
        N_comupte = 4096
        II_compute = 1
        cycles_per_query = L_load_query + nprobe_per_PE_max * (L_load_and_compute_residual + L_compute + N_comupte * II_compute)
        QPS = 1 / (cycles_per_query / FREQ)
        perf_resource_obj.cycles_per_query = cycles_per_query
        perf_resource_obj.QPS = QPS
        
        # make sure that forwarding isn't the bottleneck
        #  forwarding CC = nprobe * K
        if cycles_per_query <= nprobe * K:
            break

        #####  HLS Prediction #####
        perf_resource_obj.BRAM_18K = 17 * PE_num
        perf_resource_obj.DSP48E = 54 * PE_num
        perf_resource_obj.FF = 7825 * PE_num
        perf_resource_obj.LUT = 6403 * PE_num
        perf_resource_obj.URAM = 8 * PE_num

        #####   FIFO Consumption (Vivado Measured)   #####

        # PQ quantizer init
        perf_resource_obj.add_resource(resource_FIFO_d2_w32, num=PE_num)

        # forward query vector + center vector dispatcher
        perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=2 * PE_num)
        
        # output FIFO to stage 5
        perf_resource_obj.add_resource(resource_FIFO_d512_w512, num=1)

        # LUT forward between PEs
        # the forward FIFO of each PE = nprobe_per_PE * 256
        perf_resource_obj.add_resource(resource_FIFO_d512_w512, num=PE_num * int(nprobe_per_PE_max / 2))

        # extra storage for vector quantizer storage, on-chip or off-chip
        perf_resource_obj_on_chip = copy.deepcopy(perf_resource_obj)
        perf_resource_obj_off_chip = copy.deepcopy(perf_resource_obj)
        if nlist <= 1024:
            #####  HLS Prediction & Vivado Measured #####
            perf_resource_obj_on_chip.HBM_bank = 0
            perf_resource_obj_on_chip.URAM = perf_resource_obj.URAM + 16
            option_list.append(perf_resource_obj_on_chip)
        elif nlist <= 2048:
            #####  HLS Prediction & Vivado Measured #####
            perf_resource_obj_on_chip.HBM_bank = 0
            perf_resource_obj_on_chip.URAM = perf_resource_obj.URAM + 32
            option_list.append(perf_resource_obj_on_chip)
        elif nlist <= 4096:
            #####  HLS Prediction & Vivado Measured #####
            perf_resource_obj_on_chip.HBM_bank = 0
            perf_resource_obj_on_chip.URAM = perf_resource_obj.URAM + 64
            option_list.append(perf_resource_obj_on_chip)
        elif nlist <= 8192:
            #####  HLS Prediction & Vivado Measured #####
            perf_resource_obj_on_chip.HBM_bank = 0
            perf_resource_obj_on_chip.URAM = perf_resource_obj.URAM + 128
            option_list.append(perf_resource_obj_on_chip)
        elif nlist <= 16384:
            #####  HLS Prediction & Vivado Measured #####
            perf_resource_obj_on_chip.HBM_bank = 0
            perf_resource_obj_on_chip.URAM = perf_resource_obj.URAM + 256
            option_list.append(perf_resource_obj_on_chip)
        elif nlist <= 32768:
            #####  HLS Prediction & Vivado Measured #####
            perf_resource_obj_on_chip.HBM_bank = 0
            perf_resource_obj_on_chip.URAM = perf_resource_obj.URAM + 512
            option_list.append(perf_resource_obj_on_chip)
        else:
            pass # on-chip N/A

        perf_resource_obj_off_chip.HBM_bank = 1
        option_list.append(perf_resource_obj_off_chip)

    return option_list

def get_options_stage_5_distance_estimation_by_LUT(nlist, nprobe, FREQ, MIN_HBM_bank, MAX_HBM_bank, TOTAL_VECTORS, scan_ratio_with_OPQ, scan_ratio_without_OPQ, OPQ_enable=True, max_PE_num=100):
    
    """ this function returns a list of the performance and resource consumption of
          the entire systolic array """

    option_list = []

    for HBM_bank in range(MIN_HBM_bank, MAX_HBM_bank + 1):

        # several options: 1 HBM = 3 streams; 1 HBM = 1 streams; 2 HBM = 1 stream; 3 HBM = 1 stream; 4 HBM = 1 stream
        PE_num_list = []
        if 3 * HBM_bank <= max_PE_num:
            PE_num_list.append(3 * HBM_bank)
        for i in range(1, 10 + 1):  # n channels in 1 PE
            if HBM_bank % i == 0:
                if HBM_bank / i <= max_PE_num:
                    PE_num_list.append(int(HBM_bank / i))
        
        for PE_num in PE_num_list:

            perf_resource_obj = Resource_Performance_Stage5()
            perf_resource_obj.HBM_CHANNEL_NUM = HBM_bank
            perf_resource_obj.STAGE5_COMP_PE_NUM = PE_num

            perf_resource_obj.HBM_bank = HBM_bank

            if OPQ_enable:
                N_compute_per_nprobe = int(scan_ratio_with_OPQ[nlist] * TOTAL_VECTORS / nlist / PE_num) + 1
            else:
                N_compute_per_nprobe = int(scan_ratio_without_OPQ[nlist] * TOTAL_VECTORS / nlist / PE_num) + 1

            L_load = 2
            N_load = 256
            II_load = 1
            L_compute = 63
            II_compute = 1
            cycles_per_query = \
                nprobe * ((L_load + N_load * II_load + PE_num - 1) + \
                    (L_compute + N_compute_per_nprobe * II_compute))
            QPS = 1 / (cycles_per_query / FREQ)
            perf_resource_obj.cycles_per_query = cycles_per_query
            perf_resource_obj.QPS = QPS

            #####  HLS Prediction #####
            # perf_resource_obj.HBM_bank = 0 * PE_num 
            # perf_resource_obj.BRAM_18K = 16 * PE_num
            # perf_resource_obj.DSP48E = 30 * PE_num
            # perf_resource_obj.FF = 5437 * PE_num
            # perf_resource_obj.LUT = 5329 * PE_num
            # perf_resource_obj.URAM = 0 * PE_num

            #####   Vivado Measured   #####
            perf_resource_obj.LUT = 3937 * PE_num
            perf_resource_obj.FF = 3954 * PE_num
            perf_resource_obj.BRAM_18K = 2 * 8 * PE_num
            perf_resource_obj.URAM = 0 * PE_num
            perf_resource_obj.DSP48E = 30 * PE_num

            #####   FIFO Consumption (Vivado Measured)   #####
            perf_resource_obj.add_resource(resource_FIFO_d2_w8, num=16 * PE_num)
            perf_resource_obj.add_resource(resource_FIFO_d2_w32, num=19 * 3 * (PE_num / 3))
            perf_resource_obj.add_resource(resource_FIFO_d2_w512, num=PE_num / 3)


            #####   AXI interface & Parser (Vivado Measured)   #####
            # AXI interface
            perf_resource_obj.LUT += 1159 * (PE_num / 3)
            perf_resource_obj.FF += 3117 * (PE_num / 3)
            perf_resource_obj.BRAM_18K += 2 * 7.5 * (PE_num / 3)
            perf_resource_obj.URAM += 0 * (PE_num / 3)
            perf_resource_obj.DSP48E += 0 * (PE_num / 3)

            # Type conversion (axi512 -> tuples paser)
            perf_resource_obj.LUT += 290 * (PE_num / 3)
            perf_resource_obj.FF += 1070 * (PE_num / 3)
            perf_resource_obj.BRAM_18K += 2 * 0 * (PE_num / 3)
            perf_resource_obj.URAM += 0 * (PE_num / 3)
            perf_resource_obj.DSP48E += 0 * (PE_num / 3)

            option_list.append(perf_resource_obj)

    return option_list


def get_options_stage6_select_topK(input_stream_num, N_insertion_per_stream, topK, FREQ):

    """
        input_stream_num === stage 5 PE num
        return 1 or 2 options 
        TODO: constraint: HBM_channel_num_for_PQ_code * 3 ~= input_stream_num
    """

    option_list = []

    """ Option 1: hierachical priority queue """
    queue_num_level_A = int(input_stream_num * 2)
    N_insertion_per_stream_level_A = int(N_insertion_per_stream / 2)
    perf_resource_obj_level_A = get_priority_queue_info(
        queue_len=topK, N_insertion=N_insertion_per_stream_level_A, FREQ=FREQ)

    def find_num_queues(cycles_per_query_upper_level, N_insertion_total_level, upper_level_queue_num):
        """ find an option that the lower level can match the upper level's performance """
        queue_num_level_N = 1
        for i in range(1, upper_level_queue_num):
            perf_resource_obj_level_N = get_priority_queue_info(
                queue_len=topK, N_insertion=int(N_insertion_total_level / queue_num_level_N), FREQ=FREQ)
            if perf_resource_obj_level_N.cycles_per_query > cycles_per_query_upper_level:
                queue_num_level_N = queue_num_level_N + 1
            else:
                return (queue_num_level_N, perf_resource_obj_level_N)
        return False

    def find_hierachical_queue_structure_recursive(
            cycles_per_query_upper_level, N_insertion_total_level, upper_level_queue_num):

        if not find_num_queues(cycles_per_query_upper_level, N_insertion_total_level, upper_level_queue_num):
            return False
        else: 
            queue_num_level_N, perf_resource_obj_level_N = \
                find_num_queues(cycles_per_query_upper_level, N_insertion_total_level, upper_level_queue_num)

            if queue_num_level_N == 1:
                return [1]
            else: 
                queue_num_array = find_hierachical_queue_structure_recursive(
                    cycles_per_query_upper_level=perf_resource_obj_level_N.cycles_per_query,
                    N_insertion_total_level=int(queue_num_level_N * topK),
                    upper_level_queue_num=queue_num_level_N)
                if queue_num_array:
                    queue_num_array.append(queue_num_level_N)
                else:
                    return False

            return queue_num_array


    # if lower level cannot reduce the number of queues used to match upper level, 
    #   then hierarchical priority queue is not an option
    queue_num_array = find_hierachical_queue_structure_recursive(
        cycles_per_query_upper_level=perf_resource_obj_level_A.cycles_per_query, 
        N_insertion_total_level=int(queue_num_level_A * topK),
        upper_level_queue_num=queue_num_level_A)

    if queue_num_array:

        perf_resource_obj = Resource_Performance_Stage6()
        perf_resource_obj.SORT_GROUP_ENABLE = False

        queue_num_array.append(queue_num_level_A)
        total_priority_queue_num = np.sum(np.array(queue_num_array))
        
        solution_exist = True
        if len(queue_num_array) == 2:
            perf_resource_obj.STAGE_6_PRIORITY_QUEUE_LEVEL = 2
        elif len(queue_num_array) == 3:
            perf_resource_obj.STAGE_6_PRIORITY_QUEUE_LEVEL = 3
            perf_resource_obj.STAGE_6_PRIORITY_QUEUE_L2_NUM = queue_num_array[1]
            STAGE_6_STREAM_PER_L2_QUEUE_LARGER = int(np.ceil(queue_num_level_A / perf_resource_obj.STAGE_6_PRIORITY_QUEUE_L2_NUM))
            STAGE_6_STREAM_PER_L2_QUEUE_SMALLER = queue_num_level_A - \
                STAGE_6_STREAM_PER_L2_QUEUE_LARGER * (perf_resource_obj.STAGE_6_PRIORITY_QUEUE_L2_NUM - 1)
            perf_resource_obj.STAGE_6_STREAM_PER_L2_QUEUE_LARGER = STAGE_6_STREAM_PER_L2_QUEUE_LARGER
            perf_resource_obj.STAGE_6_STREAM_PER_L2_QUEUE_SMALLER = STAGE_6_STREAM_PER_L2_QUEUE_SMALLER
        else:
            # currently does not support #Level >= 4
            solution_exist = False

        if solution_exist:
            # all levels use the same priority queue of depth=topK
            perf_resource_obj.add_resource(perf_resource_obj_level_A, total_priority_queue_num)

            # lower level is faster than upper level
            perf_resource_obj.cycles_per_query = perf_resource_obj_level_A.cycles_per_query
            perf_resource_obj.QPS = 1 / (perf_resource_obj.cycles_per_query / FREQ)
    
            #####   FIFO Consumption (Vivado Measured)   #####
            # Note I use depth=512 here, in practical it could be smaller but will consume more LUT/FF
            perf_resource_obj.add_resource(resource_FIFO_d512_w32, np.sum(queue_num_array) * 2)

            option_list.append(perf_resource_obj) 

    """ Option 2: sort reduction network """
    """ WENQI: disabled sort-reduction network for the moment
            I found memory bug when using it (K=10 4 groups), Vitis keep asking memory
            300 + GB in 8 minutes, and keep growing, there must be something going wrong...
        last time I encounter this bug is when writing a PE using alternative loop order,
            there should be nothing wrong in the code, probably because Vitis is doing sth weird
    """
    if input_stream_num > topK:

        perf_resource_obj_bitonic_sort_16 = \
            get_bitonic_sort_16_info(N_insertion=N_insertion_per_stream, FREQ=FREQ)
        perf_resource_obj_parallel_merge_32_to_16 = \
            get_parallel_merge_32_to_16_info(N_insertion=N_insertion_per_stream, FREQ=FREQ)

        if input_stream_num <= 16:

            perf_resource_obj = Resource_Performance_Stage6()
            perf_resource_obj.SORT_GROUP_ENABLE = True
            perf_resource_obj.SORT_GROUP_NUM = 1
            perf_resource_obj.STAGE_6_PRIORITY_QUEUE_LEVEL = 2

            perf_resource_obj.add_resource(perf_resource_obj_level_A, num=21)
            perf_resource_obj.add_resource(perf_resource_obj_bitonic_sort_16, num=1)
            perf_resource_obj.add_resource(perf_resource_obj_parallel_merge_32_to_16, num=0)
            
            #####   FIFO Consumption (Vivado Measured)   #####
            perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=2)
            perf_resource_obj.add_resource(resource_FIFO_d2_w32, num=152)

            perf_resource_obj.cycles_per_query = N_insertion_per_stream
            perf_resource_obj.QPS = 1 / (perf_resource_obj.cycles_per_query / FREQ)

            option_list.append(perf_resource_obj) 

        elif input_stream_num > 16 and input_stream_num <= 32:

            perf_resource_obj = Resource_Performance_Stage6()
            perf_resource_obj.SORT_GROUP_ENABLE = True
            perf_resource_obj.SORT_GROUP_NUM = 2
            perf_resource_obj.STAGE_6_PRIORITY_QUEUE_LEVEL = 2

            perf_resource_obj.add_resource(perf_resource_obj_level_A, num=21)
            perf_resource_obj.add_resource(perf_resource_obj_bitonic_sort_16, num=2)
            perf_resource_obj.add_resource(perf_resource_obj_parallel_merge_32_to_16, num=1)
            
            #####   FIFO Consumption (Vivado Measured)   #####
            perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=2)
            perf_resource_obj.add_resource(resource_FIFO_d2_w32, num=216)

            perf_resource_obj.cycles_per_query = N_insertion_per_stream
            perf_resource_obj.QPS = 1 / (perf_resource_obj.cycles_per_query / FREQ)

            option_list.append(perf_resource_obj) 

        elif input_stream_num > 32 and input_stream_num <= 48:

            perf_resource_obj = Resource_Performance_Stage6()
            perf_resource_obj.SORT_GROUP_ENABLE = True
            perf_resource_obj.SORT_GROUP_NUM = 3
            perf_resource_obj.STAGE_6_PRIORITY_QUEUE_LEVEL = 2

            perf_resource_obj.add_resource(perf_resource_obj_level_A, num=21)
            perf_resource_obj.add_resource(perf_resource_obj_bitonic_sort_16, num=3)
            perf_resource_obj.add_resource(perf_resource_obj_parallel_merge_32_to_16, num=2)
            
            #####   FIFO Consumption (Vivado Measured)   #####
            perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=2)
            perf_resource_obj.add_resource(resource_FIFO_d2_w32, num=280)
    
            perf_resource_obj.cycles_per_query = N_insertion_per_stream
            perf_resource_obj.QPS = 1 / (perf_resource_obj.cycles_per_query / FREQ)

            option_list.append(perf_resource_obj) 

        elif input_stream_num > 48 and input_stream_num <= 64:

            perf_resource_obj = Resource_Performance_Stage6()
            perf_resource_obj.SORT_GROUP_ENABLE = True
            perf_resource_obj.SORT_GROUP_NUM = 4
            perf_resource_obj.STAGE_6_PRIORITY_QUEUE_LEVEL = 2

            perf_resource_obj.add_resource(perf_resource_obj_level_A, num=21)
            perf_resource_obj.add_resource(perf_resource_obj_bitonic_sort_16, num=4)
            perf_resource_obj.add_resource(perf_resource_obj_parallel_merge_32_to_16, num=3)
            
            #####   FIFO Consumption (Vivado Measured)   #####
            perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=2)
            perf_resource_obj.add_resource(resource_FIFO_d2_w32, num=344)
    
            perf_resource_obj.cycles_per_query = N_insertion_per_stream
            perf_resource_obj.QPS = 1 / (perf_resource_obj.cycles_per_query / FREQ)

            option_list.append(perf_resource_obj) 

        elif input_stream_num > 64 and input_stream_num <= 80:

            perf_resource_obj = Resource_Performance_Stage6()
            perf_resource_obj.SORT_GROUP_ENABLE = True
            perf_resource_obj.SORT_GROUP_NUM = 5
            perf_resource_obj.STAGE_6_PRIORITY_QUEUE_LEVEL = 2

            perf_resource_obj.add_resource(perf_resource_obj_level_A, num=21)
            perf_resource_obj.add_resource(perf_resource_obj_bitonic_sort_16, num=5)
            perf_resource_obj.add_resource(perf_resource_obj_parallel_merge_32_to_16, num=4)
            
            #####   FIFO Consumption (Vivado Measured)   #####
            perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=2)
            perf_resource_obj.add_resource(resource_FIFO_d2_w32, num=408)
    
            perf_resource_obj.cycles_per_query = N_insertion_per_stream
            perf_resource_obj.QPS = 1 / (perf_resource_obj.cycles_per_query / FREQ)

            option_list.append(perf_resource_obj) 

        elif input_stream_num > 80 and input_stream_num <= 96:

            perf_resource_obj = Resource_Performance_Stage6()
            perf_resource_obj.SORT_GROUP_ENABLE = True
            perf_resource_obj.SORT_GROUP_NUM = 6
            perf_resource_obj.STAGE_6_PRIORITY_QUEUE_LEVEL = 2

            perf_resource_obj.add_resource(perf_resource_obj_level_A, num=21)
            perf_resource_obj.add_resource(perf_resource_obj_bitonic_sort_16, num=6)
            perf_resource_obj.add_resource(perf_resource_obj_parallel_merge_32_to_16, num=5)
            
            #####   FIFO Consumption (Vivado Measured)   #####
            perf_resource_obj.add_resource(resource_FIFO_d512_w32, num=2)
            perf_resource_obj.add_resource(resource_FIFO_d2_w32, num=472)
        
            perf_resource_obj.cycles_per_query = N_insertion_per_stream
            perf_resource_obj.QPS = 1 / (perf_resource_obj.cycles_per_query / FREQ)

            option_list.append(perf_resource_obj) 

    return option_list
