#pragma once 

#include "constants.hpp"
#include "types.hpp"

template<typename T, const int queue_size, Order order> 
class Priority_queue;

template<const int queue_size> 
class Priority_queue<single_PQ_result, queue_size, Collect_smallest> {

    public: 

        Priority_queue() {
#pragma HLS inline
        }

        template<const int query_num>
        void insert_wrapper(
            hls::stream<int> &s_control_iter_num_per_query,
            hls::stream<single_PQ_result> &s_input, 
            hls::stream<single_PQ_result> &s_output) {
            
            single_PQ_result queue[queue_size];
#pragma HLS array_partition variable=queue complete

            for (int query_id = 0; query_id < query_num; query_id++) {

                int iter_num = s_control_iter_num_per_query.read();

                // init
                for (int i = 0; i < queue_size; i++) {
#pragma HLS UNROLL
                    queue[i].vec_ID = -1;
                    queue[i].dist = LARGE_NUM;
                }

                // insert: 
                for (int i = 0; i < iter_num; i++) {
#pragma HLS pipeline II=1
                    single_PQ_result reg = s_input.read();
                    queue[0] = queue[0].dist < reg.dist? queue[0] : reg;

                    compare_swap_array_step_A(queue);

                    compare_swap_array_step_B(queue);
                }

                // write
                for (int i = 0; i < queue_size; i++) {
#pragma HLS pipeline II=1
                    s_output.write(queue[i]);
                }
            }
        }


    private:
    
        void compare_swap(single_PQ_result* array, int idxA, int idxB) {
            // if smaller -> swap to right
            // note: idxA must < idxB
#pragma HLS inline
            if (array[idxA].dist < array[idxB].dist) {
                single_PQ_result regA = array[idxA];
                single_PQ_result regB = array[idxB];
                array[idxA] = regB;
                array[idxB] = regA;
            }
        }

        void compare_swap_array_step_A(single_PQ_result* array) {
            // start from idx 0, odd-even swap
#pragma HLS inline
            for (int j = 0; j < queue_size / 2; j++) {
#pragma HLS UNROLL
                compare_swap(array, 2 * j, 2 * j + 1);
            }
        }
                    
        void compare_swap_array_step_B(single_PQ_result* array) {
            // start from idx 1, odd-even swap
#pragma HLS inline
            for (int j = 0; j < (queue_size - 1) / 2; j++) {
#pragma HLS UNROLL
                compare_swap(array, 2 * j + 1, 2 * j + 2);
            }
        }
};
