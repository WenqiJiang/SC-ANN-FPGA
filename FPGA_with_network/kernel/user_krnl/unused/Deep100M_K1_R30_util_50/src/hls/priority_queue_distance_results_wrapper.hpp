#pragma once

#include "constants.hpp"
#include "types.hpp"
#include "priority_queue_distance_results.hpp"

template<const int stream_num>
void replicate_scanned_entries_per_query_Redirected_sorted_stream(
        const int query_num, 
        hls::stream<int> &s_scanned_entries_per_query_Priority_queue, 
        hls::stream<int> (&s_insertion_per_queue_L1)[stream_num]);

void consume_single_stream(
    const int query_num, 
    hls::stream<single_PQ_result> &input_stream,
    hls::stream<int> &s_scanned_entries_every_cell);

void split_single_stream(
    const int query_num, 
    hls::stream<single_PQ_result> &input_stream,
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_scanned_entries_every_cell_Out_priority_queue_A, 
    hls::stream<int> &s_scanned_entries_every_cell_Out_priority_queue_B, 
    hls::stream<single_PQ_result> &output_stream_A,
    hls::stream<single_PQ_result> &output_stream_B);

template<const int stream_num>
void split_single_PQ_result_wrapper(
    const int query_num, 
    hls::stream<single_PQ_result> (&s_input)[stream_num], 
    hls::stream<int> &s_scanned_entries_per_query_In_Priority_queue,
    hls::stream<int> (&s_scanned_entries_every_cell_Out_priority_queue)[2 * stream_num],
    hls::stream<single_PQ_result> (&s_single_PQ_result_splitted)[2 * stream_num]);

template<const int iter_num_per_query>
void send_iter_num(
    const int query_num, 
    hls::stream<int> &s_insertion_per_queue);

template<const int priority_queue_len, const int stream_num>
void merge_streams(
    const int query_num, 
    hls::stream<single_PQ_result> (&intermediate_result)[stream_num],
    hls::stream<single_PQ_result> &output_stream);

template<const int stream_num>
void stream_redirect_to_priority_queue_wrapper( 
    const int query_num, 
    hls::stream<int> &s_scanned_entries_per_query_Priority_queue,
    hls::stream<single_PQ_result> (&s_input)[stream_num], 
    hls::stream<single_PQ_result> &output_stream);



template<const int stream_num>
void replicate_scanned_entries_per_query_Redirected_sorted_stream(
    const int query_num, 
    hls::stream<int> &s_scanned_entries_per_query_Priority_queue, 
    hls::stream<int> (&s_insertion_per_queue_L1)[stream_num]) {
    
    for (int i = 0; i < query_num; i++) {

        int scanned_entries_per_query = s_scanned_entries_per_query_Priority_queue.read();
        
        for (int s = 0; s < stream_num; s++) {
#pragma HLS UNROLL
            s_insertion_per_queue_L1[s].write(scanned_entries_per_query);
        }
    }
}


void consume_single_stream(
    const int query_num, 
    hls::stream<single_PQ_result> &input_stream,
    hls::stream<int> &s_scanned_entries_every_cell) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        int scanned_entries_every_cell = s_scanned_entries_every_cell.read();

        for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline II=1
            input_stream.read();
        }
    }
}


void split_single_stream(
    const int query_num, 
    hls::stream<single_PQ_result> &input_stream,
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_scanned_entries_every_cell_Out_priority_queue_A, 
    hls::stream<int> &s_scanned_entries_every_cell_Out_priority_queue_B, 
    hls::stream<single_PQ_result> &output_stream_A,
    hls::stream<single_PQ_result> &output_stream_B) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        int scanned_entries_every_cell = s_scanned_entries_every_cell.read();
        int half_scanned_entries_every_cell = scanned_entries_every_cell / 2;

        if ((scanned_entries_every_cell - 2 * half_scanned_entries_every_cell) == 1) {
            s_scanned_entries_every_cell_Out_priority_queue_A.write(half_scanned_entries_every_cell + 1);
            output_stream_A.write(input_stream.read());
        }
        else {
            s_scanned_entries_every_cell_Out_priority_queue_A.write(half_scanned_entries_every_cell); 
        }
        s_scanned_entries_every_cell_Out_priority_queue_B.write(half_scanned_entries_every_cell);

        for (int entry_id = 0; entry_id < half_scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline II=2
            output_stream_A.write(input_stream.read());
            output_stream_B.write(input_stream.read());
        }
    }
}

template<const int stream_num>
void split_single_PQ_result_wrapper(
    const int query_num, 
    hls::stream<single_PQ_result> (&s_input)[stream_num], 
    hls::stream<int> &s_scanned_entries_per_query_In_Priority_queue,
    hls::stream<int> (&s_scanned_entries_every_cell_Out_priority_queue)[2 * stream_num],
    hls::stream<single_PQ_result> (&s_single_PQ_result_splitted)[2 * stream_num]) {
    
#pragma HLS inline
    // for the top 16 elements, discard the last 6 
    // for the rest 10 elements, split them to 2 streams, since Priority Queue's
    //   insertion takes 2 CC

    hls::stream<int> s_scanned_entries_every_cell_Replicated[stream_num];
#pragma HLS stream variable=s_scanned_entries_every_cell_Replicated depth=8
#pragma HLS array_partition variable=s_scanned_entries_every_cell_Replicated complete
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_Replicated core=FIFO_SRL

    replicate_scanned_entries_per_query_Redirected_sorted_stream<stream_num>(
        query_num,
        s_scanned_entries_per_query_In_Priority_queue, 
        s_scanned_entries_every_cell_Replicated);

    for (int i = 0; i < stream_num; i++) {
#pragma HLS UNROLL
        split_single_stream(
            query_num, 
            s_input[i], 
            s_scanned_entries_every_cell_Replicated[i],
            s_scanned_entries_every_cell_Out_priority_queue[2 * i],
            s_scanned_entries_every_cell_Out_priority_queue[2 * i + 1],
            s_single_PQ_result_splitted[2 * i], 
            s_single_PQ_result_splitted[2 * i + 1]);
    }
}

template<const int iter_num_per_query>
void send_iter_num(
    const int query_num,
    hls::stream<int> &s_insertion_per_queue) {

    for (int query_id = 0; query_id < query_num; query_id++) {
        s_insertion_per_queue.write(iter_num_per_query);
    }
}


template<const int priority_queue_len, const int stream_num>
void merge_streams(
    const int query_num, 
    hls::stream<single_PQ_result> (&intermediate_result)[stream_num],
    hls::stream<single_PQ_result> &output_stream) {
    
    for (int query_id = 0; query_id < query_num; query_id++) {
        for (int d = 0; d < priority_queue_len; d++) {
            for (int s = 0; s < stream_num; s++) {
#pragma HLS pipeline II=1
                output_stream.write(intermediate_result[s].read());
            }
        }
    }
}

template<const int stream_num>
void stage6_priority_queue_group_L2_wrapper( 
    const int query_num, 
    hls::stream<int> &s_scanned_entries_per_query_Priority_queue,
    hls::stream<single_PQ_result> (&s_input)[stream_num], 
    hls::stream<single_PQ_result> &output_stream) {

    // Given 16 input stream (last 6 streams are discarded), redirect them to 
    // 20 priority queues (because 2 CC per insertion), and then insert them to a final
    // priority queue, return the results of top 10
#pragma HLS inline

    hls::stream<int> s_insertion_per_queue_L1[STAGE_6_PRIORITY_QUEUE_L1_NUM];
#pragma HLS stream variable=s_insertion_per_queue_L1 depth=8
#pragma HLS array_partition variable=s_insertion_per_queue_L1 complete
// #pragma HLS RESOURCE variable=s_insertion_per_queue_L1 core=FIFO_SRL

    hls::stream<single_PQ_result> s_single_PQ_result_splitted[STAGE_6_PRIORITY_QUEUE_L1_NUM];
#pragma HLS stream variable=s_single_PQ_result_splitted depth=8
#pragma HLS array_partition variable=s_single_PQ_result_splitted complete
// #pragma HLS RESOURCE variable=s_single_PQ_result_splitted core=FIFO_SRL

    hls::stream<single_PQ_result> intermediate_result[STAGE_6_PRIORITY_QUEUE_L1_NUM];
#pragma HLS stream variable=intermediate_result depth=8
#pragma HLS array_partition variable=intermediate_result complete
// #pragma HLS RESOURCE variable=intermediate_result core=FIFO_SRL

    // collecting results from multiple sources need deeper FIFO
    const int intermediate_result_depth = STAGE_6_PRIORITY_QUEUE_L1_NUM * TOPK;
    hls::stream<single_PQ_result> merged_intermediate_result;
#pragma HLS stream variable=merged_intermediate_result depth=intermediate_result_depth

    // WENQI: Here, the number of priority queue must be a constant passed by macro,
    //   I have tried to use the template argument, i.e., STAGE_6_PRIORITY_QUEUE_L1_NUM, but HLS has bug on it
    Priority_queue<single_PQ_result, TOPK, Collect_smallest> priority_queue_intermediate[STAGE_6_PRIORITY_QUEUE_L1_NUM];
#pragma HLS array_partition variable=priority_queue_intermediate complete

    ////////////////////         Priority Queue Level 1          ////////////////////
    split_single_PQ_result_wrapper<stream_num>(
        query_num, 
        s_input, 
        s_scanned_entries_per_query_Priority_queue,
        s_insertion_per_queue_L1,
        s_single_PQ_result_splitted); 

    // 2 CC per insertion
    for (int i = 0; i < STAGE_6_PRIORITY_QUEUE_L1_NUM; i++) {
#pragma HLS UNROLL
        // for each individual query, output intermediate_result
        priority_queue_intermediate[i].insert_wrapper(
            query_num, 
            s_insertion_per_queue_L1[i], 
            s_single_PQ_result_splitted[i], intermediate_result[i]);
    }

    merge_streams<TOPK, STAGE_6_PRIORITY_QUEUE_L1_NUM>(
        query_num, intermediate_result, merged_intermediate_result);

    ////////////////////         Priority Queue Level 2          ////////////////////

    hls::stream<int> s_insertion_per_queue_L2;
#pragma HLS stream variable=s_insertion_per_queue_L2 depth=8
// #pragma HLS RESOURCE variable=s_insertion_per_queue_L2 core=FIFO_SRL

    Priority_queue<single_PQ_result, TOPK, Collect_smallest> priority_queue_final;

    send_iter_num<STAGE_6_PRIORITY_QUEUE_L1_NUM * TOPK>(
        query_num, 
        s_insertion_per_queue_L2);

    priority_queue_final.insert_wrapper(
        query_num,
        s_insertion_per_queue_L2,
        merged_intermediate_result, 
        output_stream); 
}
