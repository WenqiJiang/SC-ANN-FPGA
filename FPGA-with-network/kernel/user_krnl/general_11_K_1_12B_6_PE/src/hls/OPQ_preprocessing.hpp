#pragma once

#include "constants.hpp"
#include "types.hpp"

const int opq_unroll_width = 4; //vectorized computation of 8 numbers

////////////////////    Function to call in top-level     //////////////////// 
template<const int query_num>
void OPQ_preprocessing(
    const bool OPQ_enable,
    hls::stream<float> &s_OPQ_init,
    hls::stream<float> &s_query_vectors,
    hls::stream<float> &s_preprocessed_query_vectors);

////////////////////    Function to call in top-level     //////////////////// 


template<const int query_num>
void OPQ_preprocessing(
    const bool OPQ_enable,
    hls::stream<float> &s_OPQ_init,
    hls::stream<float> &s_query_vectors,
    hls::stream<float> &s_preprocessed_query_vectors) {

    float OPQ_mat[D][D];
#pragma HLS array_partition variable=OPQ_mat cyclic factor=opq_unroll_width dim=2
#pragma HLS resource variable=OPQ_mat core=RAM_1P_BRAM

    // init, load D x D OPQ matrix to local
    for (int r = 0; r < D; r++) {
        for (int c = 0; c < D; c++) {
#pragma HLS pipeline II=1
            if (OPQ_enable) {
                OPQ_mat[r][c] = s_OPQ_init.read();
            }
        }
    }

    float intermediate_result[D];
// #pragma HLS resource variable=intermediate_result core=RAM_1P_BRAM

    float query_buffer[D];
#pragma HLS array_partition variable=query_buffer cyclic factor=opq_unroll_width dim=1

    // for a row of query vector, multiply it by a row of OPQ matrix
    for (int query_id = 0; query_id < query_num; query_id++) {

        // load query vector
        for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
            query_buffer[d] = s_query_vectors.read();
        }

        // compute
        for (int c = 0; c < D / opq_unroll_width; c++) {

            for (int r = 0; r < D; r++) {
#pragma HLS pipeline II=1

                float partial_result = 
                    query_buffer[c * opq_unroll_width + 0] * OPQ_mat[r][c * opq_unroll_width + 0] +
                    query_buffer[c * opq_unroll_width + 1] * OPQ_mat[r][c * opq_unroll_width + 1] +
                    query_buffer[c * opq_unroll_width + 2] * OPQ_mat[r][c * opq_unroll_width + 2] +
                    query_buffer[c * opq_unroll_width + 3] * OPQ_mat[r][c * opq_unroll_width + 3];

                if (c == 0) {
                    intermediate_result[r] = partial_result;
                } 
                else {
                    intermediate_result[r] = intermediate_result[r] + partial_result;
                }
            }
        }

        // write
        for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
            float processed_query_vector;
            if (OPQ_enable) {
                processed_query_vector = intermediate_result[d];
            }
            else {
                processed_query_vector = query_buffer[d];
            }
            s_preprocessed_query_vectors.write(processed_query_vector);
        }
    }
}
