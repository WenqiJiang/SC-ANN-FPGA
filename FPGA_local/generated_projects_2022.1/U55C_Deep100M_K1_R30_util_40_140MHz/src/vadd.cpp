/*
Variable to be replaced (<--variable_name-->):

    multiple lines (depends on HBM channel num):
        HBM_in_vadd_arg
        HBM_in_m_axi
        HBM_in_s_axilite
        load_and_split_PQ_codes_wrapper_arg

    multiple lines (depends on stage 2 PE num / on or off-chip):
        stage2_vadd_arg
        stage2_m_axi
        stage2_s_axilite

    single line:
        stage_1_OPQ_preprocessing
        stage_2_IVF_center_distance_computation
        stage6_sort_reduction
        stage6_priority_queue_group_L2_wrapper_stream_num
        stage6_priority_queue_group_L2_wrapper_arg
        
    basic constants:

*/

#include <stdio.h>
#include <hls_stream.h>

#include "constants.hpp"
#include "debugger.hpp"
#include "helpers.hpp"
#include "types.hpp"

// stage 1
#if OPQ_ENABLE
#include "OPQ_preprocessing.hpp"
#endif 

// stage 2
#include "cluster_distance_computation.hpp"

// stage 3
#include "priority_queue_vector_quantizer.hpp"
#include "select_Voronoi_cell.hpp"

// stage 4
#include "LUT_construction.hpp"

// stage 5
#include "distance_estimation_by_LUT.hpp"
#include "HBM_interconnections.hpp"

// stage 6
#include "priority_queue_distance_results_wrapper.hpp"
#if SORT_GROUP_NUM
#include "sort_reduction_with_vecID.hpp"
#endif

extern "C" {

// The argument of top-level function must not be too long,
//   otherwise there will be error when loading bitstreams.
// The safe number is <= 20 chararcters, but it could be longer
//   (the limit is not tested yet)
void vadd(  
    const ap_uint512_t* HBM_in0,
    const ap_uint512_t* HBM_in1,
    const ap_uint512_t* HBM_in2,
    const ap_uint512_t* HBM_in3,
    const ap_uint512_t* HBM_in4,
    const ap_uint512_t* HBM_in5,
    const ap_uint512_t* HBM_in6,
    const ap_uint512_t* HBM_in7,
    const ap_uint512_t* HBM_in8,
    const ap_uint512_t* HBM_in9,
    const ap_uint512_t* HBM_in10,
    const ap_uint512_t* HBM_in11,
    const ap_uint512_t* HBM_in12,
    const ap_uint512_t* HBM_in13,
    const ap_uint512_t* HBM_in14,
    const ap_uint512_t* HBM_in15,
    const ap_uint512_t* HBM_in16,


    // HBM_meta_info containing several parts:
    //   (1) HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid: size = 3 * nlist
    //   (2) HBM_product_quantizer: size = K * D
    //   (3) (optional) s_OPQ_init: D * D, if OPQ_enable = False, send nothing
    //   (4) HBM_query_vectors: size = query_num * D (send last, because the accelerator needs to send queries continuously)
    const float* HBM_meta_info, 
    // center vector table (Vector_quantizer)
    const float* HBM_vector_quantizer,
    const int query_num,
    // stage 4 parameters, if PE_NUM==1, set the same value
    //   nprobe_per_table_construction_pe_larger = nprobe_per_table_construction_pe_smaller
    const int np_per_pe_larger,
    const int np_per_pe_smaller,
    // HBM output (vector_ID, distance)
    ap_uint64_t* HBM_out
    )
{
#pragma HLS INTERFACE m_axi port=HBM_in0 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=HBM_in1 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=HBM_in2 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=HBM_in3 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=HBM_in4 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=HBM_in5 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=HBM_in6 offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=HBM_in7 offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=HBM_in8 offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=HBM_in9 offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=HBM_in10 offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=HBM_in11 offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=HBM_in12 offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=HBM_in13 offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=HBM_in14 offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=HBM_in15 offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=HBM_in16 offset=slave bundle=gmem16



#pragma HLS INTERFACE m_axi port=HBM_meta_info  offset=slave bundle=gmemA
#pragma HLS INTERFACE m_axi port=HBM_vector_quantizer  offset=slave bundle=gmemC

#pragma HLS INTERFACE m_axi port=HBM_out offset=slave bundle=gmemF

#pragma HLS INTERFACE s_axilite port=HBM_in0
#pragma HLS INTERFACE s_axilite port=HBM_in1
#pragma HLS INTERFACE s_axilite port=HBM_in2
#pragma HLS INTERFACE s_axilite port=HBM_in3
#pragma HLS INTERFACE s_axilite port=HBM_in4
#pragma HLS INTERFACE s_axilite port=HBM_in5
#pragma HLS INTERFACE s_axilite port=HBM_in6
#pragma HLS INTERFACE s_axilite port=HBM_in7
#pragma HLS INTERFACE s_axilite port=HBM_in8
#pragma HLS INTERFACE s_axilite port=HBM_in9
#pragma HLS INTERFACE s_axilite port=HBM_in10
#pragma HLS INTERFACE s_axilite port=HBM_in11
#pragma HLS INTERFACE s_axilite port=HBM_in12
#pragma HLS INTERFACE s_axilite port=HBM_in13
#pragma HLS INTERFACE s_axilite port=HBM_in14
#pragma HLS INTERFACE s_axilite port=HBM_in15
#pragma HLS INTERFACE s_axilite port=HBM_in16



#pragma HLS INTERFACE s_axilite port=HBM_meta_info 
#pragma HLS INTERFACE s_axilite port=HBM_vector_quantizer 

#pragma HLS INTERFACE s_axilite port=query_num
#pragma HLS INTERFACE s_axilite port=np_per_pe_larger
#pragma HLS INTERFACE s_axilite port=np_per_pe_smaller

#pragma HLS INTERFACE s_axilite port=HBM_out

#pragma HLS INTERFACE s_axilite port=return
    
#pragma HLS dataflow

    const bool OPQ_enable = true;

    const int nlist = NLIST;
    const int nprobe = NPROBE;

    // name the input argument to longer version
    int centroids_per_partition_even = CENTROIDS_PER_PARTITION_EVEN;
    int centroids_per_partition_last_PE = CENTROIDS_PER_PARTITION_LAST_PE;

    int nprobe_per_table_construction_pe_larger = np_per_pe_larger;
    int nprobe_per_table_construction_pe_smaller = np_per_pe_smaller;

    ////////////////////     Init     ////////////////////

    hls::stream<float> s_query_vectors;
#pragma HLS stream variable=s_query_vectors depth=512
// #pragma HLS resource variable=s_query_vectors core=FIFO_BRAM

    hls::stream<float> s_PQ_quantizer_init;
#pragma HLS stream variable=s_PQ_quantizer_init depth=4
// #pragma HLS resource variable=s_PQ_quantizer_init core=FIFO_SRL

    hls::stream<int> s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid;
#pragma HLS stream variable=s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid depth=8
// #pragma HLS resource variable=s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid core=FIFO_SRL

    hls::stream<float> s_OPQ_init;
#pragma HLS stream variable=s_OPQ_init depth=512
// #pragma HLS resource variable=s_OPQ_init core=FIFO_BRAM

    parse_HBM_meta_info(
        query_num,
        nlist,
        OPQ_enable,
        HBM_meta_info, 
        s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid, 
        s_PQ_quantizer_init,
        s_OPQ_init,
        s_query_vectors);

    hls::stream<float> s_center_vectors_init_lookup_PE;
#pragma HLS stream variable=s_center_vectors_init_lookup_PE depth=2
// #pragma HLS resource variable=s_center_vectors_init_lookup_PE core=FIFO_SRL

#ifdef STAGE2_ON_CHIP
    hls::stream<float> s_center_vectors_init_distance_computation_PE;
#pragma HLS stream variable=s_center_vectors_init_distance_computation_PE depth=8

    load_center_vectors(
        nlist,
        HBM_vector_quantizer,
        s_center_vectors_init_distance_computation_PE,
        s_center_vectors_init_lookup_PE);
#endif


    ////////////////////     Preprocessing    ////////////////////


    hls::stream<float> s_preprocessed_query_vectors;
#pragma HLS stream variable=s_preprocessed_query_vectors depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors core=FIFO_BRAM

    OPQ_preprocessing(
        query_num,
        s_OPQ_init,
        s_query_vectors,
        s_preprocessed_query_vectors);

    hls::stream<float> s_preprocessed_query_vectors_lookup_PE;
#pragma HLS stream variable=s_preprocessed_query_vectors_lookup_PE depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors_lookup_PE core=FIFO_BRAM

    hls::stream<float> s_preprocessed_query_vectors_distance_computation_PE;
#pragma HLS stream variable=s_preprocessed_query_vectors_distance_computation_PE depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors_distance_computation_PE core=FIFO_BRAM

    broadcast_preprocessed_query_vectors(
        query_num,
        s_preprocessed_query_vectors,
        s_preprocessed_query_vectors_distance_computation_PE,
        s_preprocessed_query_vectors_lookup_PE);

    ////////////////////      Center Distance Computation    ////////////////////

    hls::stream<dist_cell_ID_t> s_merged_cell_distance;
#pragma HLS stream variable=s_merged_cell_distance depth=512
// #pragma HLS resource variable=s_merged_cell_distance core=FIFO_BRAM


    compute_cell_distance_wrapper(
        query_num,
        centroids_per_partition_even, 
        centroids_per_partition_last_PE, 
        nlist,
        s_center_vectors_init_distance_computation_PE, 
        s_preprocessed_query_vectors_distance_computation_PE, 
        s_merged_cell_distance);

    ////////////////////     Select Scanned Cells     ////////////////////    

    hls::stream<dist_cell_ID_t> s_selected_distance_cell_ID;
#pragma HLS stream variable=s_selected_distance_cell_ID depth=512
// #pragma HLS resource variable=s_selected_distance_cell_ID core=FIFO_BRAM

    select_Voronoi_cell<STAGE_3_PRIORITY_QUEUE_LEVEL, STAGE_3_PRIORITY_QUEUE_L1_NUM, NPROBE>(
        query_num,
        nlist,
        s_merged_cell_distance,
        s_selected_distance_cell_ID);

    hls::stream<int> s_searched_cell_id_lookup_PE;
#pragma HLS stream variable=s_searched_cell_id_lookup_PE depth=512
// #pragma HLS resource variable=s_searched_cell_id_lookup_PE core=FIFO_BRAM

    hls::stream<int> s_searched_cell_id_scan_controller;
#pragma HLS stream variable=s_searched_cell_id_scan_controller depth=512
// #pragma HLS resource variable=s_searched_cell_id_scan_controller core=FIFO_BRAM

    //  dist struct to cell ID (int)
    split_cell_ID(
        query_num,
        nprobe,
        s_selected_distance_cell_ID, 
        s_searched_cell_id_lookup_PE, 
        s_searched_cell_id_scan_controller);

    ////////////////////     Center Vector Lookup     ////////////////////    

    hls::stream<float> s_center_vectors_lookup_PE;
#pragma HLS stream variable=s_center_vectors_lookup_PE depth=128
// #pragma HLS resource variable=s_center_vectors_lookup_PE core=FIFO_BRAM

#ifdef STAGE2_ON_CHIP
    lookup_center_vectors(
        query_num,
        nlist,
        nprobe,
        s_center_vectors_init_lookup_PE, 
        s_searched_cell_id_lookup_PE, 
        s_center_vectors_lookup_PE);
#else 
    // if stage 2 on-chip, HBM_vector_quantizer will be used by stage2 helper PE
    //   otherwise HBM_vector_quantizer is reserved for stage 4
    lookup_center_vectors(
        query_num,
        nprobe,
        HBM_vector_quantizer, 
        s_searched_cell_id_lookup_PE, 
        s_center_vectors_lookup_PE);
#endif

    ////////////////////     Distance Lookup Table Construction     ////////////////////    

    hls::stream<distance_LUT_PQ16_t> s_distance_LUT;
#pragma HLS stream variable=s_distance_LUT depth=512
// #pragma HLS resource variable=s_distance_LUT core=FIFO_BRAM

#if PE_NUM_TABLE_CONSTRUCTION == 1
    lookup_table_construction_wrapper(
        query_num,
        nprobe,
        s_PQ_quantizer_init, 
        s_center_vectors_lookup_PE, 
        s_preprocessed_query_vectors_lookup_PE, 
        s_distance_LUT);
#else
    lookup_table_construction_wrapper(
        query_num,
        nprobe,
        nprobe_per_table_construction_pe_larger,
        nprobe_per_table_construction_pe_smaller,
        s_PQ_quantizer_init, 
        s_center_vectors_lookup_PE, 
        s_preprocessed_query_vectors_lookup_PE, 
        s_distance_LUT);
#endif
    ////////////////////     Load PQ Codes     ////////////////////    

    hls::stream<int> s_scanned_entries_every_cell_Load_unit;
#pragma HLS stream variable=s_scanned_entries_every_cell_Load_unit depth=512
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_Load_unit core=FIFO_BRAM

    hls::stream<int> s_scanned_entries_every_cell_PQ_lookup_computation;
#pragma HLS stream variable=s_scanned_entries_every_cell_PQ_lookup_computation depth=512
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_PQ_lookup_computation core=FIFO_BRAM

    hls::stream<int> s_last_valid_channel;
#pragma HLS stream variable=s_last_valid_channel depth=512
// #pragma HLS RESOURCE variable=s_last_valid_channel core=FIFO_BRAM

    hls::stream<int> s_start_addr_every_cell;
#pragma HLS stream variable=s_start_addr_every_cell depth=512
// #pragma HLS RESOURCE variable=s_start_addr_every_cell core=FIFO_BRAM

#if SORT_GROUP_NUM
    hls::stream<int> s_scanned_entries_per_query_Sort_and_reduction;
#pragma HLS stream variable=s_scanned_entries_per_query_Sort_and_reduction depth=512
// #pragma HLS RESOURCE variable=s_scanned_entries_per_query_Sort_and_reduction core=FIFO_BRAM
#endif
    hls::stream<int> s_scanned_entries_per_query_Priority_queue;
#pragma HLS stream variable=s_scanned_entries_per_query_Priority_queue depth=512
// #pragma HLS RESOURCE variable=s_scanned_entries_per_query_Priority_queue core=FIFO_BRAM

    scan_controller(
        query_num,
        nlist, 
        nprobe,
        s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid,
        s_searched_cell_id_scan_controller, 
        s_start_addr_every_cell,
        s_scanned_entries_every_cell_Load_unit, 
        s_scanned_entries_every_cell_PQ_lookup_computation,
        s_last_valid_channel, 
#if SORT_GROUP_NUM
        s_scanned_entries_per_query_Sort_and_reduction,
#endif
        s_scanned_entries_per_query_Priority_queue);

    // each 512 bit can store 3 set of (vecID, PQ code)
    hls::stream<single_PQ> s_single_PQ[STAGE5_COMP_PE_NUM];
#pragma HLS stream variable=s_single_PQ depth=8
#pragma HLS array_partition variable=s_single_PQ complete
// #pragma HLS RESOURCE variable=s_single_PQ core=FIFO_SRL

    load_and_split_PQ_codes_wrapper(
        query_num,
        nprobe,
        HBM_in0,
        HBM_in1,
        HBM_in2,
        HBM_in3,
        HBM_in4,
        HBM_in5,
        HBM_in6,
        HBM_in7,
        HBM_in8,
        HBM_in9,
        HBM_in10,
        HBM_in11,
        HBM_in12,
        HBM_in13,
        HBM_in14,
        HBM_in15,
        HBM_in16,

        s_start_addr_every_cell,
        s_scanned_entries_every_cell_Load_unit,
        s_single_PQ);

#if SORT_GROUP_NUM
    hls::stream<single_PQ_result> s_single_PQ_result[SORT_GROUP_NUM][16];
#pragma HLS stream variable=s_single_PQ_result depth=8
#pragma HLS array_partition variable=s_single_PQ_result complete
// #pragma HLS RESOURCE variable=s_single_PQ_result core=FIFO_SRL
#else
    hls::stream<single_PQ_result> s_single_PQ_result[STAGE5_COMP_PE_NUM];
#pragma HLS stream variable=s_single_PQ_result depth=8
#pragma HLS array_partition variable=s_single_PQ_result complete
// #pragma HLS RESOURCE variable=s_single_PQ_result core=FIFO_SRL
#endif

    ////////////////////     Estimate Distance by LUT     ////////////////////    

    PQ_lookup_computation_wrapper<STAGE5_COMP_PE_NUM, PQ_CODE_CHANNELS_PER_STREAM>(
        query_num,
        nprobe,
        s_single_PQ, 
        s_distance_LUT, 
        s_scanned_entries_every_cell_PQ_lookup_computation,
        s_last_valid_channel,
        s_single_PQ_result);



    hls::stream<single_PQ_result> s_output; // the top 10 numbers
#pragma HLS stream variable=s_output depth=512
// #pragma HLS RESOURCE variable=s_output core=FIFO_BRAM

    stage6_priority_queue_group_L2_wrapper<STAGE5_COMP_PE_NUM>(
        query_num,
        s_scanned_entries_per_query_Priority_queue, 
        s_single_PQ_result,
        s_output);

    ////////////////////     Write Results     ////////////////////    
    write_result(
        query_num, 
        s_output, 
        HBM_out);
}

}