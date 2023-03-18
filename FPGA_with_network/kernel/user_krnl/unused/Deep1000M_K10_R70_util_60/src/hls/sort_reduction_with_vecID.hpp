#pragma once

#include "constants.hpp"
#include "types.hpp"

// Note! Template Function Specialization is NOT allowed in class scope
namespace sort_reduction_with_vecID {

    ////////////////////     Sorting Network Start    ////////////////////
    void compare_swap(
        single_PQ_result* input_array, single_PQ_result* output_array, int idxA, int idxB) {
        // note: idxA must < idxB
#pragma HLS inline
        if (input_array[idxA].dist > input_array[idxB].dist) {
            output_array[idxA] = input_array[idxB];
            output_array[idxB] = input_array[idxA];
        }
        else {
            output_array[idxA] = input_array[idxA];
            output_array[idxB] = input_array[idxB];
        }
    }


    template<const int array_len, const int partition_num>
    void compare_swap_range_head_tail(
        single_PQ_result* input_array, single_PQ_result* output_array) {
        // e.g., in the image phase merge 4 -> 8, the 1st stage
        // Input these constants to make computation fast
#pragma HLS inline
    
        const int elements_per_partition = array_len / partition_num;
        const int operations_per_partition = elements_per_partition / 2;

        for (int i = 0; i < partition_num; i++) {
#pragma HLS UNROLL
            for (int j = 0; j < operations_per_partition; j++) {
#pragma HLS UNROLL
                compare_swap(input_array, output_array, 
                    i * elements_per_partition + j, (i + 1) * elements_per_partition - 1 - j);
            }
        }
    }

    template<const int array_len, const int partition_num>
    void compare_swap_range_interval(
        single_PQ_result* input_array, single_PQ_result* output_array) {
        // e.g., in the image phase merge 4 -> 8, the 2nd and 3rd stage
#pragma HLS inline
    
        const int elements_per_partition = array_len / partition_num;
        const int operations_per_partition = elements_per_partition / 2;
        const int interval = operations_per_partition;

        for (int i = 0; i < partition_num; i++) {
#pragma HLS UNROLL
            for (int j = 0; j < operations_per_partition; j++) {
#pragma HLS UNROLL
            compare_swap(input_array, output_array, 
                i * elements_per_partition + j, i * elements_per_partition + interval + j);
            }
        }
    }

    template<const int array_len>
    void load_input_stream(
        hls::stream<single_PQ_result> (&s_input)[array_len], 
        single_PQ_result input_array[array_len]) {
#pragma HLS inline 

        for (int s = 0; s < array_len; s++) {
#pragma HLS UNROLL 
            input_array[s] = s_input[s].read();
        }
    }

    template<const int array_len>
    void write_output_stream(
        single_PQ_result output_array[array_len], 
        hls::stream<single_PQ_result> (&s_output)[array_len]) {
#pragma HLS inline 

        for (int s = 0; s < array_len; s++) {
#pragma HLS UNROLL 
            s_output[s].write(output_array[s]);
        }
    }

    void bitonic_sort_16(
        const int query_num,
        hls::stream<int>& s_control_iter_num_per_query,
        hls::stream<single_PQ_result> (&s_input)[16],
        hls::stream<single_PQ_result> (&s_output)[16]) {

        single_PQ_result input_array[16];
#pragma HLS array_partition variable=input_array complete

        single_PQ_result out_stage1_0[16];
#pragma HLS array_partition variable=out_stage1_0 complete

        single_PQ_result out_stage2_0[16];
        single_PQ_result out_stage2_1[16];
#pragma HLS array_partition variable=out_stage2_0 complete
#pragma HLS array_partition variable=out_stage2_1 complete

        single_PQ_result out_stage3_0[16];
        single_PQ_result out_stage3_1[16];
        single_PQ_result out_stage3_2[16];
#pragma HLS array_partition variable=out_stage3_0 complete
#pragma HLS array_partition variable=out_stage3_1 complete
#pragma HLS array_partition variable=out_stage3_2 complete

        single_PQ_result out_stage4_0[16];
        single_PQ_result out_stage4_1[16];
        single_PQ_result out_stage4_2[16];
        single_PQ_result out_stage4_3[16];
#pragma HLS array_partition variable=out_stage4_0 complete
#pragma HLS array_partition variable=out_stage4_1 complete
#pragma HLS array_partition variable=out_stage4_2 complete
#pragma HLS array_partition variable=out_stage4_3 complete

        for (int query_id = 0; query_id < query_num; query_id++) {

            int iter_num = s_control_iter_num_per_query.read();

            for (int iter = 0; iter < iter_num; iter++) {
#pragma HLS pipeline II=1

                load_input_stream<16>(s_input, input_array);
                // Total: 15 sub-stages
                // Stage 1
                compare_swap_range_interval<16, 8>(input_array, out_stage1_0);

                // Stage 2: 2 -> 4
                compare_swap_range_head_tail<16, 4>(out_stage1_0, out_stage2_0);
                compare_swap_range_interval<16, 8>(out_stage2_0, out_stage2_1);

                // Stage 3: 4 -> 8
                compare_swap_range_head_tail<16, 2>(out_stage2_1, out_stage3_0);
                compare_swap_range_interval<16, 4>(out_stage3_0, out_stage3_1);
                compare_swap_range_interval<16, 8>(out_stage3_1, out_stage3_2);

                // Stage 4: 8 -> 16
                compare_swap_range_head_tail<16, 1>(out_stage3_2, out_stage4_0);
                compare_swap_range_interval<16, 2>(out_stage4_0, out_stage4_1);
                compare_swap_range_interval<16, 4>(out_stage4_1, out_stage4_2);
                compare_swap_range_interval<16, 8>(out_stage4_2, out_stage4_3);
                
                write_output_stream<16>(out_stage4_3, s_output);
            }
        }
    }
    ////////////////////     Sorting Network Ends    ////////////////////

    ////////////////////     Merge and Sort and Filter Network Starts    ////////////////////

    void compare_select(
        single_PQ_result* input_array_A, single_PQ_result* input_array_B, 
        single_PQ_result* output_array, int idxA, int idxB) {
        // note: idxOut = idxA
        // select the smallest of the two as output
#pragma HLS inline
        if (input_array_A[idxA].dist > input_array_B[idxB].dist) {
            output_array[idxA] = input_array_B[idxB];
        }
        else {
            output_array[idxA] = input_array_A[idxA];
        }
    }

    template<const int array_len>
    void compare_select_range_head_tail(
        single_PQ_result* input_array_A, single_PQ_result* input_array_B, 
        single_PQ_result* output_array) {
        // e.g., in the image phase merge 4 -> 8, the 1st stage
        // Input these constants to make computation fast
#pragma HLS inline
    
        // A[0] <-> B[127], A[1] <-> B[126], etc.
        for (int j = 0; j < array_len; j++) {
#pragma HLS UNROLL
            compare_select(
                input_array_A, input_array_B, output_array, 
                j, array_len - 1 - j);
        }
    }

    template<const int replicate_num>
    void replicate_s_control_iter_num_per_query(
        const int query_num, 
        hls::stream<int>& s_control_iter_num_per_query,
        hls::stream<int> (&s_control_iter_num_per_query_replicated)[replicate_num]) {

        for (int query_id = 0; query_id < query_num; query_id++) {

            int iter_num = s_control_iter_num_per_query.read();
        
            for (int s = 0; s < replicate_num; s++) {
#pragma HLS UNROLL
                s_control_iter_num_per_query_replicated[s].write(iter_num);
            }
        }
    }

    void parallel_merge_sort_16(
        const int query_num, 
        hls::stream<int>& s_control_iter_num_per_query,
        hls::stream<single_PQ_result> (&s_input_A)[16],
        hls::stream<single_PQ_result> (&s_input_B)[16],
        hls::stream<single_PQ_result> (&s_output)[16]) {

        // given 2 input sorted array A and B of len array_len, 
        // merge and sort and reduction to output array C of len array_len,
        // containing the smallest numbers among A and B. 

        single_PQ_result input_array_A[16];
        single_PQ_result input_array_B[16];
#pragma HLS array_partition variable=input_array_A complete
#pragma HLS array_partition variable=input_array_B complete

        single_PQ_result out_stage_0[16];
        single_PQ_result out_stage_1[16];
        single_PQ_result out_stage_2[16];
        single_PQ_result out_stage_3[16];
        single_PQ_result out_stage_4[16];
#pragma HLS array_partition variable=out_stage_0 complete
#pragma HLS array_partition variable=out_stage_1 complete
#pragma HLS array_partition variable=out_stage_2 complete
#pragma HLS array_partition variable=out_stage_3 complete
#pragma HLS array_partition variable=out_stage_4 complete


        for (int query_id = 0; query_id < query_num; query_id++) {

            int iter_num = s_control_iter_num_per_query.read();

            for (int iter = 0; iter < iter_num; iter++) {
#pragma HLS pipeline II=1

                load_input_stream<16>(s_input_A, input_array_A);
                load_input_stream<16>(s_input_B, input_array_B);

                // select the smallest 16 numbers
                compare_select_range_head_tail<16>(
                    input_array_A, input_array_B, out_stage_0);

                // sort the smallest 16 numbers
                /* Analogue to sorting 32 (a half of sorting 32) */
                compare_swap_range_interval<16, 1>(out_stage_0, out_stage_1);
                compare_swap_range_interval<16, 2>(out_stage_1, out_stage_2);
                compare_swap_range_interval<16, 4>(out_stage_2, out_stage_3);
                compare_swap_range_interval<16, 8>(out_stage_3, out_stage_4);

                write_output_stream<16>(out_stage_4, s_output);
            }
        }
    }

    void reduce_sorted_streams(
        const int query_num, 
        hls::stream<single_PQ_result> (&sorted_stream)[16], 
        hls::stream<int> &s_scanned_entries_per_query,
        hls::stream<single_PQ_result> (&reduced_sorted_stream)[TOPK]) {
        
    #pragma HLS inline
        // when topK=10, for the top 16 elements, discard the last 6 
        //   insertion takes 2 CC


        for (int query_id = 0; query_id < query_num; query_id++) {

            int scanned_entries_per_query = s_scanned_entries_per_query.read();

            for (int entry_id = 0; entry_id < scanned_entries_per_query; entry_id++) {
#pragma HLS pipeline II=1
                for (int s = 0; s < TOPK; s++) {
#pragma HLS UNROLL
                    reduced_sorted_stream[s].write(sorted_stream[s].read());
                }
                for (int s = TOPK; s < 16; s++) {
#pragma HLS UNROLL
                    sorted_stream[s].read();
                }
            }
        }
    }
};

template<typename T, const int input_stream_num, const int output_stream_num, Order order> 
class Sort_reduction;


template<> 
class Sort_reduction<single_PQ_result, 16, TOPK, Collect_smallest> {
    // input: 16 streams, 1 input per CC
    // output: sorted 16 streams 

    public:

        Sort_reduction() {
#pragma HLS inline
        }

        // Top-level function in this class: 32 (2 * 16) -> 16
        void sort_and_reduction(
            const int query_num, 
            hls::stream<int>& s_control_iter_num_per_query,
            hls::stream<single_PQ_result> (&s_input)[1][16],
            hls::stream<single_PQ_result> (&s_output)[TOPK]) {
#pragma HLS inline

            hls::stream<single_PQ_result> s_result_stage_0[16];
#pragma HLS array_partition variable=s_result_stage_0 complete
#pragma HLS stream variable=s_result_stage_0 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_0 core=FIFO_SRL

            hls::stream<int> s_control_iter_num_per_query_replicated[2];
#pragma HLS array_partition variable=s_control_iter_num_per_query_replicated complete
#pragma HLS stream variable=s_control_iter_num_per_query_replicated depth=8
// #pragma HLS RESOURCE variable=s_control_iter_num_per_query_replicated core=FIFO_SRL

            sort_reduction_with_vecID::replicate_s_control_iter_num_per_query<2>(
                query_num,
                s_control_iter_num_per_query,
                s_control_iter_num_per_query_replicated); 

            sort_reduction_with_vecID::bitonic_sort_16(
                query_num,
                s_control_iter_num_per_query_replicated[0],
                s_input[0], 
                s_result_stage_0);
                    
            sort_reduction_with_vecID::reduce_sorted_streams(
                query_num,
                s_result_stage_0,
                s_control_iter_num_per_query_replicated[1],
                s_output);
        }

    private:

};

template<> 
class Sort_reduction<single_PQ_result, 32, TOPK, Collect_smallest> {
    // input: 32 streams, 1 input per CC
    // output: 16 streams -> the top 16 smallest sorted numbers of the 32 inputs

    public:

        Sort_reduction() {
#pragma HLS inline
        }

        // Top-level function in this class: 32 (2 * 16) -> 16
        void sort_and_reduction(
            const int query_num,
            hls::stream<int>& s_control_iter_num_per_query,
            hls::stream<single_PQ_result> (&s_input)[2][16],
            hls::stream<single_PQ_result> (&s_output)[TOPK]) {
#pragma HLS inline

            hls::stream<single_PQ_result> s_result_stage_0[2][16];
#pragma HLS array_partition variable=s_result_stage_0 complete
#pragma HLS stream variable=s_result_stage_0 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_0 core=FIFO_SRL

            hls::stream<single_PQ_result> s_result_stage_1[16];
#pragma HLS array_partition variable=s_result_stage_1 complete
#pragma HLS stream variable=s_result_stage_1 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_1 core=FIFO_SRL

            hls::stream<int> s_control_iter_num_per_query_replicated[4];
#pragma HLS array_partition variable=s_control_iter_num_per_query_replicated complete
#pragma HLS stream variable=s_control_iter_num_per_query_replicated depth=8
// #pragma HLS RESOURCE variable=s_control_iter_num_per_query_replicated core=FIFO_SRL

            sort_reduction_with_vecID::replicate_s_control_iter_num_per_query<4>(
                query_num,
                s_control_iter_num_per_query,
                s_control_iter_num_per_query_replicated); 

            for (int s = 0; s < 2; s++) {
#pragma HLS UNROLL
                sort_reduction_with_vecID::bitonic_sort_16(
                    query_num,
                    s_control_iter_num_per_query_replicated[s],
                    s_input[s], 
                    s_result_stage_0[s]);
            }

            // merge result 0 and 1 
            sort_reduction_with_vecID::parallel_merge_sort_16(
                query_num,
                s_control_iter_num_per_query_replicated[2],
                s_result_stage_0[0], 
                s_result_stage_0[1], 
                s_result_stage_1);

            sort_reduction_with_vecID::reduce_sorted_streams(
                query_num,
                s_result_stage_1,
                s_control_iter_num_per_query_replicated[3],
                s_output);
        }

    private:

};

template<> 
class Sort_reduction<single_PQ_result, 48, TOPK, Collect_smallest> {
    // input: 48 streams, 1 input per CC
    // output: 16 streams -> the top 16 smallest sorted numbers of the 48 inputs

    public:

        Sort_reduction() {
#pragma HLS inline
        }

        // Top-level function in this class: 48 (3 * 16) -> 16
        void sort_and_reduction(
            const int query_num,
            hls::stream<int>& s_control_iter_num_per_query,
            hls::stream<single_PQ_result> (&s_input)[3][16],
            hls::stream<single_PQ_result> (&s_output)[TOPK]) {

#pragma HLS inline

            hls::stream<single_PQ_result> s_result_stage_0[3][16];
#pragma HLS array_partition variable=s_result_stage_0 complete
#pragma HLS stream variable=s_result_stage_0 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_0 core=FIFO_SRL

            hls::stream<single_PQ_result> s_result_stage_1[16];
#pragma HLS array_partition variable=s_result_stage_1 complete
#pragma HLS stream variable=s_result_stage_1 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_1 core=FIFO_SRL

            hls::stream<single_PQ_result> s_result_stage_2[16];
#pragma HLS array_partition variable=s_result_stage_2 complete
#pragma HLS stream variable=s_result_stage_2 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_2 core=FIFO_SRL

            hls::stream<int> s_control_iter_num_per_query_replicated[6];
#pragma HLS array_partition variable=s_control_iter_num_per_query_replicated complete
#pragma HLS stream variable=s_control_iter_num_per_query_replicated depth=8
// #pragma HLS RESOURCE variable=s_control_iter_num_per_query_replicated core=FIFO_SRL

            sort_reduction_with_vecID::replicate_s_control_iter_num_per_query<6>(
                query_num, 
                s_control_iter_num_per_query,
                s_control_iter_num_per_query_replicated); 

            for (int s = 0; s < 3; s++) {
#pragma HLS UNROLL
                sort_reduction_with_vecID::bitonic_sort_16(
                    query_num, 
                    s_control_iter_num_per_query_replicated[s],
                    s_input[s], 
                    s_result_stage_0[s]);
            }

            // merge result 0 and 1 
            sort_reduction_with_vecID::parallel_merge_sort_16(
                query_num, 
                s_control_iter_num_per_query_replicated[3],
                s_result_stage_0[0], 
                s_result_stage_0[1], 
                s_result_stage_1);

            // merge the partial result above and result 2
            sort_reduction_with_vecID::parallel_merge_sort_16(
                query_num, 
                s_control_iter_num_per_query_replicated[4],
                s_result_stage_1, 
                s_result_stage_0[2], 
                s_result_stage_2);

            sort_reduction_with_vecID::reduce_sorted_streams(
                query_num, 
                s_result_stage_2,
                s_control_iter_num_per_query_replicated[5],
                s_output);
        }

    private:

};

template<> 
class Sort_reduction<single_PQ_result, 64, TOPK, Collect_smallest> {
    // input: 64 streams, 1 input per CC
    // output: 16 streams -> the top 16 smallest sorted numbers of the 64 inputs

    public:

        Sort_reduction() {
#pragma HLS inline
        }

        // Top-level function in this class: 64 (4 * 16) -> 16
        void sort_and_reduction(
            const int query_num,
            hls::stream<int>& s_control_iter_num_per_query,
            hls::stream<single_PQ_result> (&s_input)[4][16],
            hls::stream<single_PQ_result> (&s_output)[TOPK]) {

#pragma HLS inline

            hls::stream<single_PQ_result> s_result_stage_0[4][16];
#pragma HLS array_partition variable=s_result_stage_0 complete
#pragma HLS stream variable=s_result_stage_0 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_0 core=FIFO_SRL

            hls::stream<single_PQ_result> s_result_stage_1[2][16];
#pragma HLS array_partition variable=s_result_stage_1 complete
#pragma HLS stream variable=s_result_stage_1 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_1 core=FIFO_SRL

            hls::stream<single_PQ_result> s_result_stage_2[16];
#pragma HLS array_partition variable=s_result_stage_2 complete
#pragma HLS stream variable=s_result_stage_2 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_2 core=FIFO_SRL

            hls::stream<int> s_control_iter_num_per_query_replicated[8];
#pragma HLS array_partition variable=s_control_iter_num_per_query_replicated complete
#pragma HLS stream variable=s_control_iter_num_per_query_replicated depth=8
// #pragma HLS RESOURCE variable=s_control_iter_num_per_query_replicated core=FIFO_SRL

            sort_reduction_with_vecID::replicate_s_control_iter_num_per_query<8>(
                query_num,
                s_control_iter_num_per_query,
                s_control_iter_num_per_query_replicated); 

            for (int s = 0; s < 4; s++) {
#pragma HLS UNROLL
                sort_reduction_with_vecID::bitonic_sort_16(
                    query_num,
                    s_control_iter_num_per_query_replicated[s],
                    s_input[s], 
                    s_result_stage_0[s]);
            }

            for (int s = 0; s < 2; s++) {
#pragma HLS UNROLL
                sort_reduction_with_vecID::parallel_merge_sort_16(
                    query_num,
                    s_control_iter_num_per_query_replicated[s + 4],
                    s_result_stage_0[2 * s], 
                    s_result_stage_0[2 * s + 1], 
                    s_result_stage_1[s]);
            }

            sort_reduction_with_vecID::parallel_merge_sort_16(
                query_num,
                s_control_iter_num_per_query_replicated[6],
                s_result_stage_1[0], 
                s_result_stage_1[1], 
                s_result_stage_2);

            sort_reduction_with_vecID::reduce_sorted_streams(
                query_num,
                s_result_stage_2,
                s_control_iter_num_per_query_replicated[7],
                s_output);
        }

    private:

};

template<> 
class Sort_reduction<single_PQ_result, 80, TOPK, Collect_smallest> {
    // input: 80 streams, 1 input per CC
    // output: 16 streams -> the top 16 smallest sorted numbers of the 64 inputs

    public:

        Sort_reduction() {
#pragma HLS inline
        }

        // Top-level function in this class: 64 (4 * 16) -> 16
        void sort_and_reduction(
            const int query_num,
            hls::stream<int>& s_control_iter_num_per_query,
            hls::stream<single_PQ_result> (&s_input)[5][16],
            hls::stream<single_PQ_result> (&s_output)[TOPK]) {

#pragma HLS inline

            hls::stream<single_PQ_result> s_result_stage_0[5][16];
#pragma HLS array_partition variable=s_result_stage_0 complete
#pragma HLS stream variable=s_result_stage_0 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_0 core=FIFO_SRL

            hls::stream<single_PQ_result> s_result_stage_1[2][16];
#pragma HLS array_partition variable=s_result_stage_1 complete
#pragma HLS stream variable=s_result_stage_1 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_1 core=FIFO_SRL

            hls::stream<single_PQ_result> s_result_stage_2[16];
#pragma HLS array_partition variable=s_result_stage_2 complete
#pragma HLS stream variable=s_result_stage_2 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_2 core=FIFO_SRL

            hls::stream<single_PQ_result> s_result_stage_3[16];
#pragma HLS array_partition variable=s_result_stage_3 complete
#pragma HLS stream variable=s_result_stage_3 depth=8
// #pragma HLS RESOURCE variable=s_result_stage_3 core=FIFO_SRL

            hls::stream<int> s_control_iter_num_per_query_replicated[10];
#pragma HLS array_partition variable=s_control_iter_num_per_query_replicated complete
#pragma HLS stream variable=s_control_iter_num_per_query_replicated depth=8
// #pragma HLS RESOURCE variable=s_control_iter_num_per_query_replicated core=FIFO_SRL


            sort_reduction_with_vecID::replicate_s_control_iter_num_per_query<10>(
                query_num, 
                s_control_iter_num_per_query,
                s_control_iter_num_per_query_replicated); 

            for (int s = 0; s < 5; s++) {
#pragma HLS UNROLL
                sort_reduction_with_vecID::bitonic_sort_16(
                    query_num, 
                    s_control_iter_num_per_query_replicated[s],
                    s_input[s], 
                    s_result_stage_0[s]);
            }

            // join the stream 0~3, left 4 for later use
            for (int s = 0; s < 2; s++) {
#pragma HLS UNROLL
                sort_reduction_with_vecID::parallel_merge_sort_16(
                    query_num, 
                    s_control_iter_num_per_query_replicated[s + 5],
                    s_result_stage_0[2 * s], 
                    s_result_stage_0[2 * s + 1], 
                    s_result_stage_1[s]);
            }

            sort_reduction_with_vecID::parallel_merge_sort_16(
                query_num, 
                s_control_iter_num_per_query_replicated[7],
                s_result_stage_1[0], 
                s_result_stage_1[1], 
                s_result_stage_2);

            sort_reduction_with_vecID::parallel_merge_sort_16(
                query_num, 
                s_control_iter_num_per_query_replicated[8],
                s_result_stage_0[4], 
                s_result_stage_2, 
                s_result_stage_3);

            sort_reduction_with_vecID::reduce_sorted_streams(
                query_num, 
                s_result_stage_3,
                s_control_iter_num_per_query_replicated[9],
                s_output);
        }

    private:

};