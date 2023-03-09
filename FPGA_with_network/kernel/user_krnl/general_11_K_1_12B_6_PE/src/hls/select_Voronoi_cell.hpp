#pragma once 

#include "constants.hpp"
#include "types.hpp"
#include "priority_queue_vector_quantizer.hpp"

////////////////////     Function to call in top-level     ////////////////////
template<const int level_num, const int L1_queue_num, const int nprobe_max>
void select_Voronoi_cell(
    const int nlist,
    const int nprobe,
    hls::stream<dist_cell_ID_t> &s_distance_cell_ID,
    hls::stream<dist_cell_ID_t> &s_selected_distance_cell_ID);
////////////////////     Function to call in top-level     ////////////////////

template<const int query_num>
void split_distance_cell_ID(
    const int nlist,
    hls::stream<dist_cell_ID_t> &s_distance_cell_ID,
    hls::stream<dist_cell_ID_t> (&s_distance_cell_ID_level_A)[2]) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int c = 0; c < nlist / 2; c++) {
#pragma HLS pipeline II=2
            s_distance_cell_ID_level_A[0].write(s_distance_cell_ID.read());
            s_distance_cell_ID_level_A[1].write(s_distance_cell_ID.read());
        }
    }
}

template<const int query_num, const int nprobe_max>
void merge_distance_cell_ID_level_A(
    hls::stream<dist_cell_ID_t> (&s_result_level_A)[2],
    hls::stream<dist_cell_ID_t> &s_distance_cell_ID_level_B) {
    
    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int c = 0; c < nprobe_max; c++) {
#pragma HLS pipeline II=2
            s_distance_cell_ID_level_B.write(s_result_level_A[0].read());
            s_distance_cell_ID_level_B.write(s_result_level_A[1].read());
        }
    }
}

template<>
void select_Voronoi_cell<1, 1, NPROBE_MAX>(
    const int nlist,
    const int nprobe,
    hls::stream<dist_cell_ID_t> &s_distance_cell_ID,
    hls::stream<dist_cell_ID_t> &s_selected_distance_cell_ID) {
#pragma HLS inline

    // the depth of this priority queue is nprobe
    Priority_queue<dist_cell_ID_t, NPROBE_MAX, Collect_smallest> priority_queue_level_B;

    const bool sort_all = true;
    const int read_iter_per_query = nlist;
    const int output_iter_per_query = nprobe;
    priority_queue_level_B.insert_wrapper<QUERY_NUM>(
        sort_all, 
        read_iter_per_query,
        output_iter_per_query,
        s_distance_cell_ID, 
        s_selected_distance_cell_ID); 

}

template<>
void select_Voronoi_cell<2, 2, NPROBE_MAX>(
    const int nlist,
    const int nprobe,
    hls::stream<dist_cell_ID_t> &s_distance_cell_ID,
    hls::stream<dist_cell_ID_t> &s_selected_distance_cell_ID) {
#pragma HLS inline

    hls::stream<dist_cell_ID_t> s_distance_cell_ID_level_A[2];
#pragma HLS stream variable=s_distance_cell_ID_level_A depth=2
#pragma HLS array_partition variable=s_distance_cell_ID_level_A complete

    split_distance_cell_ID<QUERY_NUM>(
        nlist,
        s_distance_cell_ID,
        s_distance_cell_ID_level_A);

    hls::stream<dist_cell_ID_t> s_result_level_A[2];
#pragma HLS stream variable=s_result_level_A depth=512
#pragma HLS array_partition variable=s_result_level_A complete

    // the depth of this priority queue is nprobe
    Priority_queue<dist_cell_ID_t, NPROBE_MAX, Collect_smallest> priority_queue_level_A[2];
#pragma HLS array_partition variable=priority_queue_level_A complete

    const bool sort_all_level_A = false;
    const int read_iter_per_query_level_A = nlist / 2;
    const int output_iter_per_query_level_A = NPROBE_MAX;
    for (int s = 0; s < 2; s++) {
#pragma HLS UNROLL
        priority_queue_level_A[s].insert_wrapper<QUERY_NUM>(
            sort_all_level_A,
            read_iter_per_query_level_A,
            output_iter_per_query_level_A,
            s_distance_cell_ID_level_A[s], 
            s_result_level_A[s]); 
    }

    hls::stream<dist_cell_ID_t> s_distance_cell_ID_level_B;
#pragma HLS stream variable=s_distance_cell_ID_level_B depth=512

    merge_distance_cell_ID_level_A<QUERY_NUM, NPROBE_MAX>(
        s_result_level_A,
        s_distance_cell_ID_level_B);

    // the depth of this priority queue is nprobe
    Priority_queue<dist_cell_ID_t, NPROBE_MAX, Collect_smallest> priority_queue_level_B;

    const bool sort_all_level_B = true;
    const int read_iter_per_query_level_B = NPROBE_MAX * 2;
    const int output_iter_per_query_level_B = nprobe;
    priority_queue_level_B.insert_wrapper<QUERY_NUM>(
        sort_all_level_B,
        read_iter_per_query_level_B,
        output_iter_per_query_level_B,
        s_distance_cell_ID_level_B, 
        s_selected_distance_cell_ID); 

}