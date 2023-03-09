from constants import * 

""" Basic building blocks, i.e., priority queue and sorting network """
def get_priority_queue_info(queue_len, N_insertion, FREQ):
    
    """ Return a single result (no multiple options) """

    perf_resource_obj = Resource_Performance()

    # Performance
    L_control = 3 
    L_insertion=5
    II_insertion=2
    L_perf_resource_objput=2
    N_output=queue_len
    II_output=1

    cycles_per_query = L_control + (L_insertion + N_insertion * II_insertion) + (L_perf_resource_objput + N_output * II_output)
    QPS = 1 / (cycles_per_query / FREQ)

    perf_resource_obj.cycles_per_query = cycles_per_query
    perf_resource_obj.QPS = QPS

    # The resource consumption is almost linear to the queue_len

    #####  HLS Prediction #####
    # perf_resource_obj.HBM_bank = 0
    # perf_resource_obj.BRAM_18K = 0
    # perf_resource_obj.DSP48E = 0
    # perf_resource_obj.FF = 2177 / 10 * queue_len
    # perf_resource_obj.LUT = 3597 / 10 * queue_len 
    # perf_resource_obj.URAM = 0

    #####  Vivado Measured #####
    perf_resource_obj.LUT = 21549 / 100 * queue_len
    perf_resource_obj.FF = 19607 / 100 * queue_len
    perf_resource_obj.BRAM_18K = 2 * 0
    perf_resource_obj.URAM = 0
    perf_resource_obj.DSP48E = 0
    perf_resource_obj.HBM_bank = 0

    return perf_resource_obj
    
def get_bitonic_sort_16_info(N_insertion, FREQ):

    """ Return a single result (no multiple options) """

    perf_resource_obj = Resource_Performance()

    # Performance
    L_insertion = 12
    II_insertion = 1
    cycles_per_query = L_insertion + N_insertion * II_insertion
    QPS = 1 / (cycles_per_query / FREQ)
    perf_resource_obj.cycles_per_query = cycles_per_query
    perf_resource_obj.QPS = QPS

    # Resource
    #####  HLS Prediction #####
    # perf_resource_obj.HBM_bank = 0
    # perf_resource_obj.BRAM_18K = 0
    # perf_resource_obj.DSP48E = 0
    # perf_resource_obj.FF = 15693
    # perf_resource_obj.LUT = 20373 
    # perf_resource_obj.URAM = 0

    #####  Vivado Measured #####
    perf_resource_obj.LUT = 10223
    perf_resource_obj.FF = 15561
    perf_resource_obj.BRAM_18K = 2 * 0
    perf_resource_obj.URAM = 0
    perf_resource_obj.DSP48E = 0
    perf_resource_obj.HBM_bank = 0

    return perf_resource_obj


def get_parallel_merge_32_to_16_info(N_insertion, FREQ):

    """ Return a single result (no multiple options) """

    perf_resource_obj = Resource_Performance()

    # Performance
    L_insertion = 7
    II_insertion = 1
    cycles_per_query = L_insertion + N_insertion * II_insertion
    QPS = 1 / (cycles_per_query / FREQ)
    perf_resource_obj.cycles_per_query = cycles_per_query
    perf_resource_obj.QPS = QPS

    # Resource
    #####  HLS Prediction #####
    # perf_resource_obj.HBM_bank = 0
    # perf_resource_obj.BRAM_18K = 0
    # perf_resource_obj.DSP48E = 0
    # perf_resource_obj.FF = 9480
    # perf_resource_obj.LUT = 11861 
    # perf_resource_obj.URAM = 0

    #####  Vivado Measured #####
    perf_resource_obj.LUT = 5588
    perf_resource_obj.FF = 9374
    perf_resource_obj.BRAM_18K = 2 * 0
    perf_resource_obj.URAM = 0
    perf_resource_obj.DSP48E = 0
    perf_resource_obj.HBM_bank = 0

    return perf_resource_obj
