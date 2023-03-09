#pragma once 

#include "constants.hpp"
#include "types.hpp"

////////////////////     Function to call in top-level     ////////////////////

template<const int query_num>
void lookup_table_construction_wrapper(
    const int nprobe,
    const int nprobe_per_table_construction_pe_larger,
    const int nprobe_per_table_construction_pe_smaller,
    hls::stream<float> &s_PQ_quantizer_init,
    hls::stream<float> &s_center_vectors_lookup_PE,
    hls::stream<float> &s_query_vectors_lookup_PE,
    hls::stream<distance_LUT_PQ16_t> &s_distance_LUT);
    
////////////////////     Function to call in top-level     ////////////////////

////////////////////     Padding Logic     ////////////////////
// PE_NUM_TABLE_CONSTRUCTION = PE_NUM_TABLE_CONSTRUCTION_LARGER + 1
//   the first PE_NUM_TABLE_CONSTRUCTION_LARGER PEs construct nprobe_per_table_construction_pe_larger 
//     LUTs per query, while the last PE constructs nprobe_per_table_construction_pe_smaller
//   it could happen that PE_NUM_TABLE_CONSTRUCTION_LARGER * nprobe_per_table_construction_pe_larger > nprobe
//     such that nprobe_per_table_construction_pe_smaller is negative, and in this case we need pad it to 1
//     the padding happens on the host side, and here we need to send some dummy data to finish the pad (if any)
//   the order of table construction,  given 4 PEs, nprobe=14 (pad to 15 = 3 * 5 + 1):
//     PE0 (head): 1 5 8 11 14
//     PE1: 2 6 9 12 15
//     PE2: 3 7 10 13 16
//     PE3 (tail): 4
//   this order is preserved when forwarding the LUTs, and finally there's a consume unit to remove the dummy LUTs
////////////////////     Padding Logic     ////////////////////


template<const int query_num>
void center_vectors_padding(
    const int nprobe,
    const int nprobe_per_table_construction_pe_larger,
    const int nprobe_per_table_construction_pe_smaller,
    hls::stream<float>& s_center_vectors_lookup_PE,
    hls::stream<float>& s_center_vectors_lookup_PE_with_dummy) {

    int padded_nprobe = 
        nprobe_per_table_construction_pe_larger * PE_NUM_TABLE_CONSTRUCTION_LARGER +
        nprobe_per_table_construction_pe_smaller; 

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int i = 0; i < padded_nprobe; i++) {

            for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
                if (i < nprobe) {
                    s_center_vectors_lookup_PE_with_dummy.write(
                        s_center_vectors_lookup_PE.read());
                }
                else {
                    s_center_vectors_lookup_PE_with_dummy.write(0.0);
                }
            }
        }
    }
}

template<const int query_num>
void center_vectors_dispatcher(
    const int nprobe_per_table_construction_pe_larger,
    const int nprobe_per_table_construction_pe_smaller,
    hls::stream<float>& s_center_vectors_lookup_PE_with_dummy,
    hls::stream<float> (&s_center_vectors_table_construction_PE)[PE_NUM_TABLE_CONSTRUCTION]) {

    // Given an input stream of center vectors, interleave it to all 
    //   distance table construction PEs in a round-robin manner 
    //   e.g., 4 PEs, vector 0,4,8 -> PE0, 1,5,9 -> PE1, etc.
    for (int query_id = 0; query_id < query_num; query_id++) {

        // first, interleave the common part of all PEs (decided by the PE of smaller scanned cells)
        for (int interleave_iter = 0; interleave_iter < nprobe_per_table_construction_pe_larger; interleave_iter++) {

            for (int s = 0; s < PE_NUM_TABLE_CONSTRUCTION_LARGER; s++) {

                for (int n = 0; n < D; n++) {
#pragma HLS pipeline II=1
                    s_center_vectors_table_construction_PE[s].write(s_center_vectors_lookup_PE_with_dummy.read());
                }
            }
            if (interleave_iter < nprobe_per_table_construction_pe_smaller) {

                for (int n = 0; n < D; n++) {
#pragma HLS pipeline II=1
                    s_center_vectors_table_construction_PE[PE_NUM_TABLE_CONSTRUCTION_LARGER].write(s_center_vectors_lookup_PE_with_dummy.read());
                }
            }
        }
    }
}

template<const int query_num>
void gather_float_to_distance_LUT_PQ16(
    const int nprobe_per_PE,
    hls::stream<float>& s_partial_result_table_construction_individual,
    hls::stream<distance_LUT_PQ16_t>& s_partial_result_table_construction_PE) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe_per_PE; nprobe_id++) {

            distance_LUT_PQ16_t out;
            for (int k = 0; k < K; k++) {
#pragma HLS pipeline II=16
                out.dist_0 = s_partial_result_table_construction_individual.read();
                out.dist_1 = s_partial_result_table_construction_individual.read();
                out.dist_2 = s_partial_result_table_construction_individual.read();
                out.dist_3 = s_partial_result_table_construction_individual.read();
                out.dist_4 = s_partial_result_table_construction_individual.read();
                out.dist_5 = s_partial_result_table_construction_individual.read();
                out.dist_6 = s_partial_result_table_construction_individual.read();
                out.dist_7 = s_partial_result_table_construction_individual.read();
                out.dist_8 = s_partial_result_table_construction_individual.read();
                out.dist_9 = s_partial_result_table_construction_individual.read();
                out.dist_10 = s_partial_result_table_construction_individual.read();
                out.dist_11 = s_partial_result_table_construction_individual.read();
                out.dist_12 = s_partial_result_table_construction_individual.read();
                out.dist_13 = s_partial_result_table_construction_individual.read();
                out.dist_14 = s_partial_result_table_construction_individual.read();
                out.dist_15 = s_partial_result_table_construction_individual.read();

                s_partial_result_table_construction_PE.write(out);
            }
        }
    }
}

template<const int query_num>
void lookup_table_construction_compute_head(
    const int nprobe_per_PE,
    hls::stream<float>& s_PQ_quantizer_init_in,
    hls::stream<float>& s_PQ_quantizer_init_out,
    hls::stream<float>& s_center_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_out,
    hls::stream<float>& s_partial_result_table_construction_individual) {

    /* output format:
     *   lookup table dim: (K x M)
     *   sending first row, then second row, and so on...
     *   store in distance_LUT_PQ16_t, each represent an entire row (M=16)
     *   256 distance_LUT_PQ16_t is an entire lookup table
     */

    // local alignment: 16-sub quantizers
    //    each quantizer: 256 row, (128 / 16) col
    // [M][K][D/M] -> [16][256][8]
    float sub_quantizer[M * K * (D / M)];
#pragma HLS resource variable=sub_quantizer core=RAM_2P_URAM
#pragma HLS array_partition variable=sub_quantizer cyclic factor=8 dim=1

    // DRAM PQ quantizer format: 16 (M) x 256 (K) x 8 (D/M)
    for (int i = 0; i < M * K * D / M; i++) {
        float reg = s_PQ_quantizer_init_in.read();
        sub_quantizer[i] = reg;
        s_PQ_quantizer_init_out.write(reg);
    }

    float query_vector_local[D];
    float center_vector_local[D];
    float residual_center_vector[D]; // query_vector - center_vector
#pragma HLS array_partition variable=residual_center_vector cyclic factor=16

    for (int query_id = 0; query_id < query_num; query_id++) {

        // load query vector
        for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
            float reg = s_query_vectors_table_construction_PE_in.read();
            query_vector_local[d] = reg;
            s_query_vectors_table_construction_PE_out.write(reg);
        }

        for (int nprobe_id = 0; nprobe_id < nprobe_per_PE; nprobe_id++) {

            // load center vector
            residual_center_vectors:
            for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
                center_vector_local[d] = s_center_vectors_table_construction_PE_in.read();
                residual_center_vector[d] = query_vector_local[d] - center_vector_local[d];
            }


            // construct distance lookup table
            single_row_lookup_table_construction:
            for (int k = 0; k < K; k++) {

                for (int m = 0; m < M; m++) {
#pragma HLS pipeline II=1

                    // no need to init to 0, the following logic will overwrite them
                    float L1_dist[D / M];
#pragma HLS array_partition variable=L1_dist complete
                    for (int simd_i = 0; simd_i < D / M; simd_i++) {
#pragma HLS UNROLL
                        L1_dist[simd_i] = 
                            residual_center_vector[m * (D / M) + simd_i] - 
                            sub_quantizer[m * K * (D / M) + k * (D / M) + simd_i];
                    }
                    float LUT_val = 
                    (L1_dist[0] * L1_dist[0]) + (L1_dist[1] * L1_dist[1]) +
                    (L1_dist[2] * L1_dist[2]) + (L1_dist[3] * L1_dist[3]) +
                    (L1_dist[4] * L1_dist[4]) + (L1_dist[5] * L1_dist[5]) + 
                    (L1_dist[6] * L1_dist[6]) + (L1_dist[7] * L1_dist[7]);
                    
                    s_partial_result_table_construction_individual.write(LUT_val);
                }
            }
        }
    }
}

template<const int query_num>
void extra_FIFO_head_PE(
    const int nprobe_per_PE,
    hls::stream<distance_LUT_PQ16_t>& s_partial_result_table_construction_PE_in,
    hls::stream<distance_LUT_PQ16_t>& s_partial_result_table_construction_PE_out) {
    // Prevent compute stall:
    //   make sure that the results of head PE can accumulate if later forward FIFO stalls
        for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe_per_PE; nprobe_id++) {

            for (int k = 0; k < K; k++) {
#pragma HLS pipeline II=1
                s_partial_result_table_construction_PE_out.write(
                    s_partial_result_table_construction_PE_in.read());
            }
        }
    }
}

template<const int query_num>
void lookup_table_construction_head_PE(
    const int nprobe_per_PE,
    hls::stream<float>& s_PQ_quantizer_init_in,
    hls::stream<float>& s_PQ_quantizer_init_out,
    hls::stream<float>& s_center_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_out,
    hls::stream<distance_LUT_PQ16_t>& s_partial_result_table_construction_PE) {

#pragma HLS dataflow

    hls::stream<float> s_partial_result_table_construction_individual;
#pragma HLS stream variable=s_partial_result_table_construction_individual depth=512

    const int s_partial_result_table_construction_PE_extra_FIFO_depth = K * PE_NUM_TABLE_CONSTRUCTION_LARGER;
    hls::stream<distance_LUT_PQ16_t> s_partial_result_table_construction_PE_extra_FIFO;
#pragma HLS stream variable=s_partial_result_table_construction_PE depth=s_partial_result_table_construction_PE_extra_FIFO_depth

    lookup_table_construction_compute_head<query_num>(
        nprobe_per_PE,
        s_PQ_quantizer_init_in,
        s_PQ_quantizer_init_out,
        s_center_vectors_table_construction_PE_in,
        s_query_vectors_table_construction_PE_in,
        s_query_vectors_table_construction_PE_out,
        s_partial_result_table_construction_individual);

    gather_float_to_distance_LUT_PQ16<query_num>(
        nprobe_per_PE,
        s_partial_result_table_construction_individual,
        s_partial_result_table_construction_PE_extra_FIFO);

    extra_FIFO_head_PE<query_num>(
        nprobe_per_PE,
        s_partial_result_table_construction_PE_extra_FIFO,
        s_partial_result_table_construction_PE);

}

template<const int query_num>
void lookup_table_construction_compute_midlle(
    const int nprobe_per_PE,
    hls::stream<float>& s_PQ_quantizer_init_in,
    hls::stream<float>& s_PQ_quantizer_init_out,
    hls::stream<float>& s_center_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_out,
    hls::stream<float>& s_partial_result_table_construction_individual) {

    /* output format:
     *   lookup table dim: (K x M)
     *   sending first row, then second row, and so on...
     *   store in distance_LUT_PQ16_t, each represent an entire row (M=16)
     *   256 distance_LUT_PQ16_t is an entire lookup table
     */

    // local alignment: 16-sub quantizers
    //    each quantizer: 256 row, (128 / 16) col
    // [M][K][D/M] -> [16][256][8]
    float sub_quantizer[M * K * (D / M)];
#pragma HLS resource variable=sub_quantizer core=RAM_2P_URAM
#pragma HLS array_partition variable=sub_quantizer cyclic factor=8 dim=1

    // DRAM PQ quantizer format: 16 (M) x 256 (K) x 8 (D/M)
    for (int i = 0; i < M * K * D / M; i++) {
        float reg = s_PQ_quantizer_init_in.read();
        sub_quantizer[i] = reg;
        s_PQ_quantizer_init_out.write(reg);
    }


    float query_vector_local[D];
    float center_vector_local[D];
    float residual_center_vector[D]; // query_vector - center_vector
#pragma HLS array_partition variable=residual_center_vector cyclic factor=16

    for (int query_id = 0; query_id < query_num; query_id++) {

        // load query vector
        for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
            float reg = s_query_vectors_table_construction_PE_in.read();
            query_vector_local[d] = reg;
            s_query_vectors_table_construction_PE_out.write(reg);
        }

        for (int nprobe_id = 0; nprobe_id < nprobe_per_PE; nprobe_id++) {

            // load center vector
            residual_center_vectors:
            for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
                center_vector_local[d] = s_center_vectors_table_construction_PE_in.read();
                residual_center_vector[d] = query_vector_local[d] - center_vector_local[d];
            }

            // construct distance lookup table
            single_row_lookup_table_construction:
            for (int k = 0; k < K; k++) {

                for (int m = 0; m < M; m++) {
#pragma HLS pipeline II=1

                    // no need to init to 0, the following logic will overwrite them
                    float L1_dist[D / M];
#pragma HLS array_partition variable=L1_dist complete
                    for (int simd_i = 0; simd_i < D / M; simd_i++) {
#pragma HLS UNROLL
                        L1_dist[simd_i] = 
                            residual_center_vector[m * (D / M) + simd_i] - 
                            sub_quantizer[m * K * (D / M) + k * (D / M) + simd_i];
                    }
                    float LUT_val = 
                    (L1_dist[0] * L1_dist[0]) + (L1_dist[1] * L1_dist[1]) +
                    (L1_dist[2] * L1_dist[2]) + (L1_dist[3] * L1_dist[3]) +
                    (L1_dist[4] * L1_dist[4]) + (L1_dist[5] * L1_dist[5]) + 
                    (L1_dist[6] * L1_dist[6]) + (L1_dist[7] * L1_dist[7]);
                    
                    s_partial_result_table_construction_individual.write(LUT_val);
                }
            }
        }
    }
}


template<const int query_num>
void lookup_table_construction_forward_middle(
    const int systolic_array_id,
    const int nprobe_per_table_construction_pe_larger,
    hls::stream<distance_LUT_PQ16_t> &s_partial_result_table_construction_PE,
    hls::stream<distance_LUT_PQ16_t> &s_partial_result_table_construction_forward_in,
    hls::stream<distance_LUT_PQ16_t> &s_partial_result_table_construction_forward_out) {

    //////////  NOTE: the order of output LUT must be consistent of the center vector input  ///////// 
    // e.g., say nprobe=17, PE_num=4, then the first 3 PEs compute 5 tables while the last compute 2
    //  first 2 rounds 4 PEs, last 3 rounds 3 PEs
    // PE 0: 0, 4, 8, 11, 14
    // PE 1: 1, 5, 9, 12, 15
    // PE 2: 2, 6, 10, 13, 16
    // PE 3: 3, 7

    for (int query_id = 0; query_id < query_num; query_id++) {
        
        for (int interleave_iter = 0; interleave_iter < nprobe_per_table_construction_pe_larger; interleave_iter++) {

            // forward head / midlle PEs
            for (int s = 0; s < systolic_array_id; s++) {
                // each lookup table: K rows
                for (int t = 0; t < K; t++) {
#pragma HLS pipeline II=1
                    s_partial_result_table_construction_forward_out.write(s_partial_result_table_construction_forward_in.read());
                }
            }
            // result from the current PE
            for (int t = 0; t < K; t++) {
#pragma HLS pipeline II=1
                s_partial_result_table_construction_forward_out.write(s_partial_result_table_construction_PE.read());
            }
        }
    }
}



template<const int query_num>
void lookup_table_construction_middle_PE(
    const int systolic_array_id,
    const int nprobe_per_table_construction_pe_larger,
    hls::stream<float>& s_PQ_quantizer_init_in,
    hls::stream<float>& s_PQ_quantizer_init_out,
    hls::stream<float>& s_center_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_out,
    hls::stream<distance_LUT_PQ16_t> &s_partial_result_table_construction_forward_in,
    hls::stream<distance_LUT_PQ16_t> &s_partial_result_table_construction_forward_out) {

#pragma HLS dataflow

    hls::stream<float> s_partial_result_table_construction_individual;
#pragma HLS stream variable=s_partial_result_table_construction_individual depth=512

    const int s_partial_result_table_construction_PE_depth = K * PE_NUM_TABLE_CONSTRUCTION_LARGER;
    hls::stream<distance_LUT_PQ16_t> s_partial_result_table_construction_PE;
#pragma HLS stream variable=s_partial_result_table_construction_PE depth=s_partial_result_table_construction_PE_depth

    lookup_table_construction_compute_midlle<query_num>(
        nprobe_per_table_construction_pe_larger,
        s_PQ_quantizer_init_in,
        s_PQ_quantizer_init_out,
        s_center_vectors_table_construction_PE_in,
        s_query_vectors_table_construction_PE_in,
        s_query_vectors_table_construction_PE_out,
        s_partial_result_table_construction_individual);

    gather_float_to_distance_LUT_PQ16<query_num>(
        nprobe_per_table_construction_pe_larger,
        s_partial_result_table_construction_individual,
        s_partial_result_table_construction_PE);

    lookup_table_construction_forward_middle<query_num>(
        systolic_array_id,
        nprobe_per_table_construction_pe_larger,
        s_partial_result_table_construction_PE,
        s_partial_result_table_construction_forward_in,
        s_partial_result_table_construction_forward_out);
}

template<const int query_num>
void lookup_table_construction_compute_tail(
    const int nprobe_per_PE,
    hls::stream<float>& s_PQ_quantizer_init_in,
    hls::stream<float>& s_center_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_in,
    hls::stream<float>& s_partial_result_table_construction_individual) {

    /* output format:
     *   lookup table dim: (K x M)
     *   sending first row, then second row, and so on...
     *   store in distance_LUT_PQ16_t, each represent an entire row (M=16)
     *   256 distance_LUT_PQ16_t is an entire lookup table
     */

    // local alignment: 16-sub quantizers
    //    each quantizer: 256 row, (128 / 16) col
    // [M][K][D/M] -> [16][256][8]
    float sub_quantizer[M * K * (D / M)];
#pragma HLS resource variable=sub_quantizer core=RAM_2P_URAM
#pragma HLS array_partition variable=sub_quantizer cyclic factor=8 dim=1

    // DRAM PQ quantizer format: 16 (M) x 256 (K) x 8 (D/M)
    for (int i = 0; i < M * K * D / M; i++) {
        float reg = s_PQ_quantizer_init_in.read();
        sub_quantizer[i] = reg;
    }


    float query_vector_local[D];
    float center_vector_local[D];
    float residual_center_vector[D]; // query_vector - center_vector
#pragma HLS array_partition variable=residual_center_vector cyclic factor=16

    for (int query_id = 0; query_id < query_num; query_id++) {

        // load query vector
        for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
            float reg = s_query_vectors_table_construction_PE_in.read();
            query_vector_local[d] = reg;
        }

        for (int nprobe_id = 0; nprobe_id < nprobe_per_PE; nprobe_id++) {

            // load center vector
            residual_center_vectors:
            for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
                center_vector_local[d] = s_center_vectors_table_construction_PE_in.read();
                residual_center_vector[d] = query_vector_local[d] - center_vector_local[d];
            }

            // construct distance lookup table
            single_row_lookup_table_construction:
            for (int k = 0; k < K; k++) {

                for (int m = 0; m < M; m++) {
#pragma HLS pipeline II=1

                    // no need to init to 0, the following logic will overwrite them
                    float L1_dist[D / M];
#pragma HLS array_partition variable=L1_dist complete
                    for (int simd_i = 0; simd_i < D / M; simd_i++) {
#pragma HLS UNROLL
                        L1_dist[simd_i] = 
                            residual_center_vector[m * (D / M) + simd_i] - 
                            sub_quantizer[m * K * (D / M) + k * (D / M) + simd_i];
                    }
                    float LUT_val = 
                    (L1_dist[0] * L1_dist[0]) + (L1_dist[1] * L1_dist[1]) +
                    (L1_dist[2] * L1_dist[2]) + (L1_dist[3] * L1_dist[3]) +
                    (L1_dist[4] * L1_dist[4]) + (L1_dist[5] * L1_dist[5]) + 
                    (L1_dist[6] * L1_dist[6]) + (L1_dist[7] * L1_dist[7]);
                    
                    s_partial_result_table_construction_individual.write(LUT_val);
                }
            }
        }
    }
}

template<const int query_num>
void lookup_table_construction_forward_tail(
    const int nprobe_per_table_construction_pe_larger,
    const int nprobe_per_table_construction_pe_smaller,
    hls::stream<distance_LUT_PQ16_t> &s_partial_result_table_construction_PE,
    hls::stream<distance_LUT_PQ16_t> &s_partial_result_table_construction_forward_in,
    hls::stream<distance_LUT_PQ16_t> &s_result_all_distance_lookup_table) {

    //////////  NOTE: the order of output LUT must be consistent of the center vector input  ///////// 
    // e.g., say nprobe=17, PE_num=4, then the first 3 PEs compute 5 tables while the last compute 2
    //  first 2 rounds 4 PEs, last 3 rounds 3 PEs
    // PE 0: 0, 4, 8, 11, 14
    // PE 1: 1, 5, 9, 12, 15
    // PE 2: 2, 6, 10, 13, 16
    // PE 3: 3, 7

    for (int query_id = 0; query_id < query_num; query_id++) {
        
        for (int interleave_iter = 0; interleave_iter < nprobe_per_table_construction_pe_larger; interleave_iter++) {

            // forward head / midlle PEs
            for (int s = 0; s < PE_NUM_TABLE_CONSTRUCTION_LARGER; s++) {
                // each lookup table: K rows
                for (int t = 0; t < K; t++) {
#pragma HLS pipeline II=1
                    s_result_all_distance_lookup_table.write(s_partial_result_table_construction_forward_in.read());
                }
            }
            if (interleave_iter < nprobe_per_table_construction_pe_smaller) {
                // result from the current PE
                for (int t = 0; t < K; t++) {
#pragma HLS pipeline II=1
                    s_result_all_distance_lookup_table.write(s_partial_result_table_construction_PE.read());
                }
            }
        }
    }
}

template<const int query_num>
void lookup_table_construction_tail_PE(
    const int nprobe_per_table_construction_pe_larger,
    const int nprobe_per_table_construction_pe_smaller,
    hls::stream<float>& s_PQ_quantizer_init_in,
    hls::stream<float>& s_center_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_in,
    hls::stream<distance_LUT_PQ16_t> &s_partial_result_table_construction_forward_in,
    hls::stream<distance_LUT_PQ16_t> &s_result_all_distance_lookup_table) {

#pragma HLS dataflow

    hls::stream<float> s_partial_result_table_construction_individual;
#pragma HLS stream variable=s_partial_result_table_construction_individual depth=512

    const int s_partial_result_table_construction_PE_depth = K * PE_NUM_TABLE_CONSTRUCTION_SMALLER;
    hls::stream<distance_LUT_PQ16_t> s_partial_result_table_construction_PE;
#pragma HLS stream variable=s_partial_result_table_construction_PE depth=s_partial_result_table_construction_PE_depth

    lookup_table_construction_compute_tail<query_num>(
        nprobe_per_table_construction_pe_smaller,
        s_PQ_quantizer_init_in,
        s_center_vectors_table_construction_PE_in,
        s_query_vectors_table_construction_PE_in,
        s_partial_result_table_construction_individual);

    gather_float_to_distance_LUT_PQ16<query_num>(
        nprobe_per_table_construction_pe_smaller,
        s_partial_result_table_construction_individual,
        s_partial_result_table_construction_PE);

    lookup_table_construction_forward_tail<query_num>(
        nprobe_per_table_construction_pe_larger,
        nprobe_per_table_construction_pe_smaller,
        s_partial_result_table_construction_PE,
        s_partial_result_table_construction_forward_in,
        s_result_all_distance_lookup_table);
}

template<const int query_num>
void remove_dummy_LUTs(
    const int nprobe,
    const int nprobe_per_table_construction_pe_larger,
    const int nprobe_per_table_construction_pe_smaller,
    hls::stream<distance_LUT_PQ16_t>& s_distance_LUT_with_dummy,
    hls::stream<distance_LUT_PQ16_t>& s_distance_LUT) {

    int padded_nprobe = 
        nprobe_per_table_construction_pe_larger * PE_NUM_TABLE_CONSTRUCTION_LARGER +
        nprobe_per_table_construction_pe_smaller; 

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int i = 0; i < padded_nprobe; i++) {

            for (int t = 0; t < K; t++) {
#pragma HLS pipeline II=1
                distance_LUT_PQ16_t reg = s_distance_LUT_with_dummy.read();
                if (i < nprobe) {
                    s_distance_LUT.write(reg);
                }
            }
        }
    }
}

template<const int query_num>
void lookup_table_construction_wrapper(
    const int nprobe,
    const int nprobe_per_table_construction_pe_larger,
    const int nprobe_per_table_construction_pe_smaller,
    hls::stream<float> &s_PQ_quantizer_init,
    hls::stream<float> &s_center_vectors_lookup_PE,
    hls::stream<float> &s_query_vectors_lookup_PE,
    hls::stream<distance_LUT_PQ16_t> &s_distance_LUT) {

#pragma HLS inline


    hls::stream<float> s_center_vectors_lookup_PE_with_dummy;
#pragma HLS stream variable=s_center_vectors_lookup_PE_with_dummy depth=512

    center_vectors_padding<query_num>(
        nprobe,
        nprobe_per_table_construction_pe_larger,
        nprobe_per_table_construction_pe_smaller,
        s_center_vectors_lookup_PE,
        s_center_vectors_lookup_PE_with_dummy);

    hls::stream<float> s_center_vectors_table_construction_PE[PE_NUM_TABLE_CONSTRUCTION];
#pragma HLS stream variable=s_center_vectors_table_construction_PE depth=512
// #pragma HLS resource variable=s_center_vectors_table_construction_PE core=FIFO_BRAM
#pragma HLS array_partition variable=s_center_vectors_table_construction_PE complete

    center_vectors_dispatcher<query_num>(
        nprobe_per_table_construction_pe_larger,
        nprobe_per_table_construction_pe_smaller,
        s_center_vectors_lookup_PE_with_dummy, 
        s_center_vectors_table_construction_PE);

    hls::stream<float> s_PQ_quantizer_init_forward[PE_NUM_TABLE_CONSTRUCTION_LARGER];
#pragma HLS stream variable=s_PQ_quantizer_init_forward depth=8
#pragma HLS array_partition variable=s_PQ_quantizer_init_forward complete

    hls::stream<float> s_query_vectors_table_construction_PE_forward[PE_NUM_TABLE_CONSTRUCTION_LARGER];
#pragma HLS stream variable=s_query_vectors_table_construction_PE_forward depth=512
#pragma HLS array_partition variable=s_query_vectors_table_construction_PE_forward complete

    hls::stream<distance_LUT_PQ16_t> s_partial_result_table_construction_PE_forward[PE_NUM_TABLE_CONSTRUCTION_LARGER];
#pragma HLS stream variable=s_partial_result_table_construction_PE_forward depth=8
#pragma HLS array_partition variable=s_partial_result_table_construction_PE_forward complete

    lookup_table_construction_head_PE<query_num>(
        nprobe_per_table_construction_pe_larger,
        s_PQ_quantizer_init,
        s_PQ_quantizer_init_forward[0],
        s_center_vectors_table_construction_PE[0],
        s_query_vectors_lookup_PE,
        s_query_vectors_table_construction_PE_forward[0],
        s_partial_result_table_construction_PE_forward[0]);

    // systolic array ID: e.g., 5 PEs, head = 0, middle = 1, 2, 3, tail = 4
    for (int s = 1; s < PE_NUM_TABLE_CONSTRUCTION_LARGER; s++) {
#pragma HLS UNROLL

        lookup_table_construction_middle_PE<query_num>(
            s,
            nprobe_per_table_construction_pe_larger,
            s_PQ_quantizer_init_forward[s - 1],
            s_PQ_quantizer_init_forward[s],
            s_center_vectors_table_construction_PE[s],
            s_query_vectors_table_construction_PE_forward[s - 1],
            s_query_vectors_table_construction_PE_forward[s],
            s_partial_result_table_construction_PE_forward[s - 1],
            s_partial_result_table_construction_PE_forward[s]);
    }

    hls::stream<distance_LUT_PQ16_t> s_distance_LUT_with_dummy;
#pragma HLS stream variable=s_distance_LUT_with_dummy depth=8

    // NOTE! PE_NUM_TABLE_CONSTRUCTION_SMALLER must === 1
    lookup_table_construction_tail_PE<query_num>(
        nprobe_per_table_construction_pe_larger,
        nprobe_per_table_construction_pe_smaller,
        s_PQ_quantizer_init_forward[PE_NUM_TABLE_CONSTRUCTION_LARGER - 1],
        s_center_vectors_table_construction_PE[PE_NUM_TABLE_CONSTRUCTION_LARGER],
        s_query_vectors_table_construction_PE_forward[PE_NUM_TABLE_CONSTRUCTION_LARGER - 1],
        s_partial_result_table_construction_PE_forward[PE_NUM_TABLE_CONSTRUCTION_LARGER - 1],
        s_distance_LUT_with_dummy);
    
    remove_dummy_LUTs<query_num>(
        nprobe,
        nprobe_per_table_construction_pe_larger,
        nprobe_per_table_construction_pe_smaller,
        s_distance_LUT_with_dummy,
        s_distance_LUT);
}
