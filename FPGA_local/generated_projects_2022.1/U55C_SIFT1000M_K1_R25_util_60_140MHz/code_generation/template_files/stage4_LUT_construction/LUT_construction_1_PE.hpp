#pragma once 

#include "constants.hpp"
#include "types.hpp"

////////////////////     Function to call in top-level     ////////////////////

void lookup_table_construction_wrapper(
    const int query_num,
    hls::stream<float> &s_PQ_quantizer_init,
    hls::stream<float> &s_center_vectors_lookup_PE,
    hls::stream<float> &s_query_vectors_lookup_PE,
    hls::stream<distance_LUT_PQ16_t> &s_distance_LUT);
    
////////////////////     Function to call in top-level     ////////////////////

void lookup_table_construction_compute(
    const int query_num,
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
                    float L2_dist[D / M];
#pragma HLS array_partition variable=L2_dist complete
                    for (int simd_i = 0; simd_i < D / M; simd_i++) {
#pragma HLS UNROLL
                        float reg = 
                            residual_center_vector[m * (D / M) + simd_i] - 
                            sub_quantizer[m * K * (D / M) + k * (D / M) + simd_i];
                        L2_dist[simd_i] = reg * reg;
                    }
                    float LUT_val = 0;
                    for (int simd_i = 0; simd_i < D / M; simd_i++) {
#pragma HLS UNROLL
                        LUT_val += L2_dist[simd_i];
                    }
                    
                    s_partial_result_table_construction_individual.write(LUT_val);
                }
            }
        }
    }
}

void gather_float_to_distance_LUT_PQ16(
    const int query_num,
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


void lookup_table_construction_PE(
    const int query_num,
    const int nprobe_per_PE,
    hls::stream<float>& s_PQ_quantizer_init_in,
    hls::stream<float>& s_center_vectors_table_construction_PE_in,
    hls::stream<float>& s_query_vectors_table_construction_PE_in,
    hls::stream<distance_LUT_PQ16_t>& s_partial_result_table_construction_PE) {

#pragma HLS dataflow

    hls::stream<float> s_partial_result_table_construction_individual;
#pragma HLS stream variable=s_partial_result_table_construction_individual depth=512

    lookup_table_construction_compute(
        query_num,
        nprobe_per_PE,
        s_PQ_quantizer_init_in,
        s_center_vectors_table_construction_PE_in,
        s_query_vectors_table_construction_PE_in,
        s_partial_result_table_construction_individual);

    gather_float_to_distance_LUT_PQ16(
        query_num,
        nprobe_per_PE,
        s_partial_result_table_construction_individual,
        s_partial_result_table_construction_PE);

}

void lookup_table_construction_wrapper(
    const int query_num,
    const int nprobe_per_PE,
    hls::stream<float> &s_PQ_quantizer_init,
    hls::stream<float> &s_center_vectors_lookup_PE,
    hls::stream<float> &s_query_vectors_lookup_PE,
    hls::stream<distance_LUT_PQ16_t> &s_distance_LUT) {

#pragma HLS inline

    lookup_table_construction_PE(
        query_num,
        nprobe_per_PE,
        s_PQ_quantizer_init,
        s_center_vectors_lookup_PE,
        s_query_vectors_lookup_PE,
        s_distance_LUT);
}
