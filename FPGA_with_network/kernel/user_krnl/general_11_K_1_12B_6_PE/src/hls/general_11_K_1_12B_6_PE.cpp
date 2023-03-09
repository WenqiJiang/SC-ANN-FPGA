/*
 * Copyright (c) 2020, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "ap_axi_sdata.h"
#include <ap_fixed.h>
#include "ap_int.h" 
#include "../../../../common/include/communication.hpp"
#include "hls_stream.h"


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

#define SESSION_NUM 16


template<const int query_num>
void network_output_converter_K_1(
    hls::stream<single_PQ_result> &s_tuple_results, 
    hls::stream<ap_uint512_t>& s_network_results) {

    // Output format per query:
    //   pkg -> (vector ID, distance) pairs
    // pkg num per query = ceil(1/8) = 1
    int vec_ID_mask = -1;
    float dist_mask = 9999999.0;
    ap_uint<32> tmp_vec_ID_mask = *((ap_uint<32>*) (&vec_ID_mask));
    ap_uint<32> tmp_dist_mask = *((ap_uint<32>*) (&dist_mask));
    ap_uint<512> pkg_out_mask;
    for (int i = 0; i < 8; i++) { // 8 * (32 * 2) = 512 bit
#pragma HLS UNROLL
        pkg_out_mask.range(i * 64 + 31, i * 64) = tmp_vec_ID_mask;
        pkg_out_mask.range(i * 64 + 63, i * 64 + 32) = tmp_dist_mask;
    }

    // Note! Here TOPK is hard-coded as 1s
    int send_buffer_vec_ID;
    float send_buffer_dist;

    int processed_query_num = 0;
    int out_per_query_counter = 0;

    do {

        if ((!s_tuple_results.empty()) && (out_per_query_counter < 1)) {
            single_PQ_result reg = s_tuple_results.read();
            send_buffer_vec_ID = reg.vec_ID;
            send_buffer_dist = reg.dist;
            out_per_query_counter++;
        }

        if (out_per_query_counter == 1) {
            ap_uint<512> pkg_out = pkg_out_mask;

            ap_uint<32> tmp_vec_ID = *((ap_uint<32>*) (&send_buffer_vec_ID));
            pkg_out.range(0 * 64 + 31, 0 * 64) = tmp_vec_ID;

            ap_uint<32> tmp_dist = *((ap_uint<32>*) (&send_buffer_dist));
            pkg_out.range((0 + 1) * 64 - 1, 0 * 64 + 32) = tmp_dist;

            s_network_results.write(pkg_out);

            out_per_query_counter = 0;
            processed_query_num++;
        }

    } while (processed_query_num < query_num);

}

template<const int query_num>
void network_output_converter_K_10(
    hls::stream<single_PQ_result> &s_tuple_results, 
    hls::stream<ap_uint512_t>& s_network_results) {

    // Output format per query:
    //   pkg -> (vector ID, distance) pairs
    // pkg num per query = ceil(10/8) = 2
    int vec_ID_mask = -1;
    float dist_mask = 9999999.0;
    ap_uint<32> tmp_vec_ID_mask = *((ap_uint<32>*) (&vec_ID_mask));
    ap_uint<32> tmp_dist_mask = *((ap_uint<32>*) (&dist_mask));
    ap_uint<512> pkg_out_mask;
    for (int i = 0; i < 8; i++) { // 8 * (32 * 2) = 512 bit
#pragma HLS UNROLL
        pkg_out_mask.range(i * 64 + 31, i * 64) = tmp_vec_ID_mask;
        pkg_out_mask.range(i * 64 + 63, i * 64 + 32) = tmp_dist_mask;
    }
    const int pkg_num_per_query = 2; // ceil(10/8) = 2

    // Note! Here TOPK is hard-coded as 10
    int send_buffer_vec_ID[10];
    float send_buffer_dist[10];
#pragma HLS array_partition variable=send_buffer_vec_ID complete
#pragma HLS array_partition variable=send_buffer_dist complete

    int processed_query_num = 0;
    int out_per_query_counter = 0;

    do {

        if ((!s_tuple_results.empty()) && (out_per_query_counter < 10)) {
            single_PQ_result reg = s_tuple_results.read();
            send_buffer_vec_ID[out_per_query_counter] = reg.vec_ID;
            send_buffer_dist[out_per_query_counter] = reg.dist;
            out_per_query_counter++;
        }

        if (out_per_query_counter == 10) {

            // all but the last packet (fill all bits)
            for (int p = 0; p < pkg_num_per_query - 1; p++) {

                ap_uint<512> pkg_out = pkg_out_mask;

                for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
                    int vec_ID_reg = send_buffer_vec_ID[p * 8 + i];
                    ap_uint<32> tmp_vec_ID = *((ap_uint<32>*) (&vec_ID_reg));
                    float dist_reg = send_buffer_dist[p * 8 + i];
                    ap_uint<32> tmp_dist = *((ap_uint<32>*) (&dist_reg));
                    pkg_out.range(i * 64 + 31, i * 64) = tmp_vec_ID;
                    pkg_out.range(i * 64 + 63, i * 64 + 32) = tmp_dist;
                }
                s_network_results.write(pkg_out);
            }
            // last packet (some bits are left default)
            ap_uint<512> pkg_out = pkg_out_mask;
            const int last_packet_entry_num = 8 - (pkg_num_per_query * 8 - 10);
            for (int i = 0; i < last_packet_entry_num; i++) {
#pragma HLS UNROLL
                int vec_ID_reg = send_buffer_vec_ID[(pkg_num_per_query - 1) * 8 + i];
                ap_uint<32> tmp_vec_ID = *((ap_uint<32>*) (&vec_ID_reg));
                float dist_reg = send_buffer_dist[(pkg_num_per_query - 1) * 8 + i];
                ap_uint<32> tmp_dist = *((ap_uint<32>*) (&dist_reg));
                pkg_out.range(i * 64 + 31, i * 64) = tmp_vec_ID;
                pkg_out.range(i * 64 + 63, i * 64 + 32) = tmp_dist;
            }
            s_network_results.write(pkg_out);



            out_per_query_counter = 0;
            processed_query_num++;
        }

    } while (processed_query_num < query_num);

}


template<const int query_num>
void network_output_converter_K_100(
    hls::stream<single_PQ_result> &s_tuple_results, 
    hls::stream<ap_uint512_t>& s_network_results) {

    // Output format per query:
    //   pkg -> (vector ID, distance) pairs
    // pkg num per query = ceil(10/8) = 2
    int vec_ID_mask = -1;
    float dist_mask = 9999999.0;
    ap_uint<32> tmp_vec_ID_mask = *((ap_uint<32>*) (&vec_ID_mask));
    ap_uint<32> tmp_dist_mask = *((ap_uint<32>*) (&dist_mask));
    ap_uint<512> pkg_out_mask;
    for (int i = 0; i < 8; i++) { // 8 * (32 * 2) = 512 bit
#pragma HLS UNROLL
        pkg_out_mask.range(i * 64 + 31, i * 64) = tmp_vec_ID_mask;
        pkg_out_mask.range(i * 64 + 63, i * 64 + 32) = tmp_dist_mask;
    }
    const int pkg_num_per_query = 13; // ceil(100/8) = 13

    // Note! Here TOPK is hard-coded as 10
    int send_buffer_vec_ID[100];
    float send_buffer_dist[100];
#pragma HLS array_partition variable=send_buffer_vec_ID complete
#pragma HLS array_partition variable=send_buffer_dist complete

    int processed_query_num = 0;
    int out_per_query_counter = 0;

    do {

        if ((!s_tuple_results.empty()) && (out_per_query_counter < 100)) {
            single_PQ_result reg = s_tuple_results.read();
            send_buffer_vec_ID[out_per_query_counter] = reg.vec_ID;
            send_buffer_dist[out_per_query_counter] = reg.dist;
            out_per_query_counter++;
        }

        if (out_per_query_counter == 100) {

            // all but the last packet (fill all bits)
            for (int p = 0; p < pkg_num_per_query - 1; p++) {

                ap_uint<512> pkg_out = pkg_out_mask;

                for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
                    int vec_ID_reg = send_buffer_vec_ID[p * 8 + i];
                    ap_uint<32> tmp_vec_ID = *((ap_uint<32>*) (&vec_ID_reg));
                    float dist_reg = send_buffer_dist[p * 8 + i];
                    ap_uint<32> tmp_dist = *((ap_uint<32>*) (&dist_reg));
                    pkg_out.range(i * 64 + 31, i * 64) = tmp_vec_ID;
                    pkg_out.range(i * 64 + 63, i * 64 + 32) = tmp_dist;
                }
                s_network_results.write(pkg_out);
            }
            // last packet (some bits are left default)
            ap_uint<512> pkg_out = pkg_out_mask;
            const int last_packet_entry_num = 8 - (pkg_num_per_query * 8 - 10);
            for (int i = 0; i < last_packet_entry_num; i++) {
#pragma HLS UNROLL
                int vec_ID_reg = send_buffer_vec_ID[(pkg_num_per_query - 1) * 8 + i];
                ap_uint<32> tmp_vec_ID = *((ap_uint<32>*) (&vec_ID_reg));
                float dist_reg = send_buffer_dist[(pkg_num_per_query - 1) * 8 + i];
                ap_uint<32> tmp_dist = *((ap_uint<32>*) (&dist_reg));
                pkg_out.range(i * 64 + 31, i * 64) = tmp_vec_ID;
                pkg_out.range(i * 64 + 63, i * 64 + 32) = tmp_dist;
            }
            s_network_results.write(pkg_out);



            out_per_query_counter = 0;
            processed_query_num++;
        }

    } while (processed_query_num < query_num);

}


void network_query_controller_push(
    ap_uint<64> expectedRxByteCnt,
    hls::stream<ap_uint<512> >& s_data_in,
    hls::stream<ap_uint<16> >& s_sessionID_in,
    hls::stream<ap_uint<16> >& s_nextRxPacketLength_in,
    // output
    hls::stream<float> (&s_query_vectors_per_session)[SESSION_NUM],
    hls::stream<int> &s_session_entry, // which FIFO to pull
    hls::stream<ap_uint<16> >& s_sessionID_out
) {

    ap_uint<16> session_ID_array[SESSION_NUM];
    int remaining_floats_in_FIFO[SESSION_NUM]; // how many floats in each FIFO
    for (int i = 0; i < SESSION_NUM; i++) {
        remaining_floats_in_FIFO[i] = 0;
    }

    int existing_session_count = 0; 

    ap_uint<64> rxByteCnt = 0;

    do{
        // WENQI: The length here is in terms of bytes,
        //    hopefully the network stack has handled it 
        //    such that the bytes is always a multiple of 512-bit (64byte)
        ap_uint<16> byte_length = s_nextRxPacketLength_in.read();
        ap_uint<16> session_ID = s_sessionID_in.read();

        // identify which FIFO to write into
        int session_entry = -1;
        bool session_in_array = false;
        for (int i = 0; i < existing_session_count; i++) {
            if (session_ID == session_ID_array[i]) {
                session_entry = i;
                session_in_array = true;
                break;
            }
        }
        if (!session_in_array) {
            session_entry = existing_session_count;
            session_ID_array[session_entry] = session_ID;
            existing_session_count++;
        }

        // byte counts
        rxByteCnt = rxByteCnt + byte_length;
        int pkt_len = byte_length / 64; // 1 512 bit packet = 64 bytes
        int float_len = pkt_len * 16; // 1 float = 4 bytes

        // should add counter first, send enable signal out, then write 
        //     to prevent deadlock in the case that the puller doesnt know it should pull without the signal
        remaining_floats_in_FIFO[session_entry] += float_len;
        if (remaining_floats_in_FIFO[session_entry] >= 128) {
            s_session_entry.write(session_entry);
            s_sessionID_out.write(session_ID);
            remaining_floats_in_FIFO[session_entry] -= 128;
        }

        // write data to the respective FIFO
        for (int j = 0; j < pkt_len; j++) {
            ap_uint<512> pkt_data = s_data_in.read();
            for (int k = 0; k < 16; k++) {
                ap_uint<32> tmp = pkt_data.range(31 + 32 * k, 32 * k);
                float content = *((float*) (&tmp));
                s_query_vectors_per_session[session_entry].write(content);
            }
        }
    } while(rxByteCnt < expectedRxByteCnt);

}

template<const int query_num>
void network_query_controller_pull(
    // input
    hls::stream<float> (&s_query_vectors_per_session)[SESSION_NUM],
    hls::stream<int> &s_session_entry, 
    // output
    hls::stream<float>& s_query_vectors
) {

    int processed_query_num = 0;
    int ele_per_query_counter = 0;
    int session_entry = 0;
    bool consume_data = false; // whether to read data FIFO
    bool session_lock = false;  // read from one session consecutively


    do {

        if (!s_session_entry.empty() & !session_lock) {
            session_entry = s_session_entry.read();
            consume_data = true;
            session_lock = true;
        }

        // read input & write output
        if (consume_data && (ele_per_query_counter < 128)) {
            if (!s_query_vectors_per_session[session_entry].empty()) {
                s_query_vectors.write(s_query_vectors_per_session[session_entry].read());
                ele_per_query_counter++;
            }
        }

        if (ele_per_query_counter == 128) {
            ele_per_query_counter = 0;
            consume_data = false;
            session_lock = false;
            processed_query_num++;
        }

    } while (processed_query_num < query_num);   
}


extern "C" {

    void general_11_K_1_12B_6_PE(

        // Internal Stream, arg 0~3
        hls::stream<pkt512>& s_axis_udp_rx, 
        hls::stream<pkt512>& m_axis_udp_tx, 
        hls::stream<pkt256>& s_axis_udp_rx_meta, 
        hls::stream<pkt256>& m_axis_udp_tx_meta, 
        
        // arg 4~15
        hls::stream<pkt16>& m_axis_tcp_listen_port, 
        hls::stream<pkt8>& s_axis_tcp_port_status, 
        hls::stream<pkt64>& m_axis_tcp_open_connection, 
        hls::stream<pkt32>& s_axis_tcp_open_status, 
        hls::stream<pkt16>& m_axis_tcp_close_connection, 
        hls::stream<pkt128>& s_axis_tcp_notification, 
        hls::stream<pkt32>& m_axis_tcp_read_pkg, 
        hls::stream<pkt16>& s_axis_tcp_rx_meta, 
        hls::stream<pkt512>& s_axis_tcp_rx_data, 
        hls::stream<pkt32>& m_axis_tcp_tx_meta, 
        hls::stream<pkt512>& m_axis_tcp_tx_data, 
        hls::stream<pkt64>& s_axis_tcp_tx_status,

        // 16~18
        int useConn, 
        int listenPort, 
        int expectedRxByteCnt,

        // ANNS kernel, start from 19
        // ANNS kernel options
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

        const ap_uint512_t* HBM_centroid_vectors_stage2_0,
        const ap_uint512_t* HBM_centroid_vectors_stage2_1,
        const ap_uint512_t* HBM_centroid_vectors_stage2_2,

        // HBM_meta_info containing several parts:
        //   (1) HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid: size = 3 * nlist
        //   (2) HBM_product_quantizer: size = K * D
        //   (3) (optional) s_OPQ_init: D * D, if OPQ_enable = False, send nothing
        //   (4) HBM_query_vectors: size = query_num * D (send last, because the accelerator needs to send queries continuously)
        const float* HBM_meta_info, 
        // center vector table (Vector_quantizer)
        const float* HBM_vector_quantizer,
        const int nlist,
        const int nprobe,
        // stage 1 parameter
        const bool OPQ_enable,
        // stage 2 parameters
        const int c_per_part_even, 
        const int c_per_part_last, 
        // stage 4 parameters, if PE_NUM==1, set the same value
        //   nprobe_per_table_construction_pe_larger = nprobe_per_table_construction_pe_smaller
        const int np_per_pe_larger,
        const int np_per_pe_smaller
            ) {

#pragma HLS INTERFACE m_axi port=HBM_in0  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=HBM_in1  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=HBM_in2  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=HBM_in3  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=HBM_in4  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=HBM_in5  offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=HBM_in6  offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=HBM_in7  offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=HBM_in8  offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=HBM_in9  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=HBM_in10  offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=HBM_in11  offset=slave bundle=gmem11

#pragma HLS INTERFACE m_axi port=HBM_centroid_vectors_stage2_0  offset=slave bundle=gmemC0
#pragma HLS INTERFACE m_axi port=HBM_centroid_vectors_stage2_1  offset=slave bundle=gmemC1
#pragma HLS INTERFACE m_axi port=HBM_centroid_vectors_stage2_2  offset=slave bundle=gmemC2

#pragma HLS INTERFACE m_axi port=HBM_meta_info  offset=slave bundle=gmemA
#pragma HLS INTERFACE m_axi port=HBM_vector_quantizer  offset=slave bundle=gmemC

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

#pragma HLS INTERFACE s_axilite port=HBM_centroid_vectors_stage2_0
#pragma HLS INTERFACE s_axilite port=HBM_centroid_vectors_stage2_1
#pragma HLS INTERFACE s_axilite port=HBM_centroid_vectors_stage2_2


#pragma HLS INTERFACE s_axilite port=HBM_meta_info 
#pragma HLS INTERFACE s_axilite port=HBM_vector_quantizer 

#pragma HLS INTERFACE s_axilite port=nlist
#pragma HLS INTERFACE s_axilite port=nprobe
#pragma HLS INTERFACE s_axilite port=OPQ_enable
#pragma HLS INTERFACE s_axilite port=c_per_part_even
#pragma HLS INTERFACE s_axilite port=c_per_part_last
#pragma HLS INTERFACE s_axilite port=np_per_pe_larger
#pragma HLS INTERFACE s_axilite port=np_per_pe_smaller

#pragma HLS INTERFACE axis port = s_axis_udp_rx
#pragma HLS INTERFACE axis port = m_axis_udp_tx
#pragma HLS INTERFACE axis port = s_axis_udp_rx_meta
#pragma HLS INTERFACE axis port = m_axis_udp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_listen_port
#pragma HLS INTERFACE axis port = s_axis_tcp_port_status
#pragma HLS INTERFACE axis port = m_axis_tcp_open_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_open_status
#pragma HLS INTERFACE axis port = m_axis_tcp_close_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_notification
#pragma HLS INTERFACE axis port = m_axis_tcp_read_pkg
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_meta
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_data
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_data
#pragma HLS INTERFACE axis port = s_axis_tcp_tx_status
#pragma HLS INTERFACE s_axilite port=useConn 
#pragma HLS INTERFACE s_axilite port=listenPort 
#pragma HLS INTERFACE s_axilite port=expectedRxByteCnt
#pragma HLS INTERFACE s_axilite port = return

// #pragma HLS INTERFACE ap_control_non

#pragma HLS dataflow

        // name the input argument to longer version
        int centroids_per_partition_even = c_per_part_even;
        int centroids_per_partition_last_PE = c_per_part_last;

        int nprobe_per_table_construction_pe_larger = np_per_pe_larger;
        int nprobe_per_table_construction_pe_smaller = np_per_pe_smaller;

        listenPorts (listenPort, useConn, m_axis_tcp_listen_port, 
            s_axis_tcp_port_status);


        hls::stream<ap_uint<512> >    s_data_in;
#pragma HLS STREAM variable=s_data_in depth=512

        hls::stream<ap_uint<16> > s_sessionID_in;
#pragma HLS STREAM variable=s_sessionID_in depth=512

        hls::stream<ap_uint<16> > s_nextRxPacketLength_in;
#pragma HLS STREAM variable=s_nextRxPacketLength_in depth=512

        recvData(
            expectedRxByteCnt, 
            s_axis_tcp_notification, 
            m_axis_tcp_read_pkg, 
            s_axis_tcp_rx_meta, 
            s_axis_tcp_rx_data,
            // output, including the sessionID and length of each packet
            s_data_in,
            s_sessionID_in,
            s_nextRxPacketLength_in);

        hls::stream<float> s_query_vectors;
#pragma HLS STREAM variable=s_query_vectors depth=512

        hls::stream<ap_uint<16> > s_sessionID_out;
#pragma HLS STREAM variable=s_sessionID_out depth=512


        hls::stream<float> s_query_vectors_per_session[SESSION_NUM];
#pragma HLS STREAM variable=s_query_vectors_per_session depth=512
#pragma HLS array_partition variable=s_query_vectors_per_session complete

        hls::stream<int> s_session_entry;
#pragma HLS STREAM variable=s_session_entry depth=512

        network_query_controller_push(
            // input
            QUERY_NUM * 512,
            s_data_in,
            s_sessionID_in,
            s_nextRxPacketLength_in,
            // output
            s_query_vectors_per_session,
            s_session_entry, // which FIFO to pull
            s_sessionID_out);

        network_query_controller_pull<QUERY_NUM>(
            // input
            s_query_vectors_per_session,
            s_session_entry, 
            // output
            s_query_vectors);

        ///////// Network Recv Ends /////////

        ////////////////////     Init     ////////////////////
    
    hls::stream<float> s_PQ_quantizer_init;
#pragma HLS stream variable=s_PQ_quantizer_init depth=4
// #pragma HLS resource variable=s_PQ_quantizer_init core=FIFO_SRL

    hls::stream<int> s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid;
#pragma HLS stream variable=s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid depth=8
// #pragma HLS resource variable=s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid core=FIFO_SRL

    hls::stream<float> s_OPQ_init;
#pragma HLS stream variable=s_OPQ_init depth=512
// #pragma HLS resource variable=s_OPQ_init core=FIFO_BRAM

    parse_HBM_meta_info<QUERY_NUM>(
        nlist,
        OPQ_enable,
        HBM_meta_info, 
        s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid, 
        s_PQ_quantizer_init,
        s_OPQ_init);
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

    OPQ_preprocessing<QUERY_NUM>(
        OPQ_enable,
        s_OPQ_init,
        s_query_vectors,
        s_preprocessed_query_vectors);

    hls::stream<float> s_preprocessed_query_vectors_lookup_PE;
#pragma HLS stream variable=s_preprocessed_query_vectors_lookup_PE depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors_lookup_PE core=FIFO_BRAM

    hls::stream<float> s_preprocessed_query_vectors_distance_computation_PE;
#pragma HLS stream variable=s_preprocessed_query_vectors_distance_computation_PE depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors_distance_computation_PE core=FIFO_BRAM

    broadcast_preprocessed_query_vectors<QUERY_NUM>(
        s_preprocessed_query_vectors,
        s_preprocessed_query_vectors_distance_computation_PE,
        s_preprocessed_query_vectors_lookup_PE);

    ////////////////////      Center Distance Computation    ////////////////////

    hls::stream<dist_cell_ID_t> s_merged_cell_distance;
#pragma HLS stream variable=s_merged_cell_distance depth=512
// #pragma HLS resource variable=s_merged_cell_distance core=FIFO_BRAM


    compute_cell_distance_wrapper<QUERY_NUM>(
        centroids_per_partition_even, 
        centroids_per_partition_last_PE, 
        nlist,
        HBM_centroid_vectors_stage2_0,
        HBM_centroid_vectors_stage2_1,
        HBM_centroid_vectors_stage2_2,

        s_preprocessed_query_vectors_distance_computation_PE,
        s_merged_cell_distance);

    ////////////////////     Select Scanned Cells     ////////////////////    

    hls::stream<dist_cell_ID_t> s_selected_distance_cell_ID;
#pragma HLS stream variable=s_selected_distance_cell_ID depth=512
// #pragma HLS resource variable=s_selected_distance_cell_ID core=FIFO_BRAM

    select_Voronoi_cell<STAGE_3_PRIORITY_QUEUE_LEVEL, STAGE_3_PRIORITY_QUEUE_L1_NUM, NPROBE_MAX>(
        nlist,
        nprobe,
        s_merged_cell_distance,
        s_selected_distance_cell_ID);

    hls::stream<int> s_searched_cell_id_lookup_PE;
#pragma HLS stream variable=s_searched_cell_id_lookup_PE depth=512
// #pragma HLS resource variable=s_searched_cell_id_lookup_PE core=FIFO_BRAM

    hls::stream<int> s_searched_cell_id_scan_controller;
#pragma HLS stream variable=s_searched_cell_id_scan_controller depth=512
// #pragma HLS resource variable=s_searched_cell_id_scan_controller core=FIFO_BRAM

    //  dist struct to cell ID (int)
    split_cell_ID<QUERY_NUM>(
        nprobe,
        s_selected_distance_cell_ID, 
        s_searched_cell_id_lookup_PE, 
        s_searched_cell_id_scan_controller);

    ////////////////////     Center Vector Lookup     ////////////////////    

    hls::stream<float> s_center_vectors_lookup_PE;
#pragma HLS stream variable=s_center_vectors_lookup_PE depth=128
// #pragma HLS resource variable=s_center_vectors_lookup_PE core=FIFO_BRAM

#ifdef STAGE2_ON_CHIP
    lookup_center_vectors<QUERY_NUM>(
        nlist,
        nprobe,
        s_center_vectors_init_lookup_PE, 
        s_searched_cell_id_lookup_PE, 
        s_center_vectors_lookup_PE);
#else 
    // if stage 2 on-chip, HBM_vector_quantizer will be used by stage2 helper PE
    //   otherwise HBM_vector_quantizer is reserved for stage 4
    lookup_center_vectors<QUERY_NUM>(
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
    lookup_table_construction_wrapper<QUERY_NUM>(
        nprobe,
        s_PQ_quantizer_init, 
        s_center_vectors_lookup_PE, 
        s_preprocessed_query_vectors_lookup_PE, 
        s_distance_LUT);
#else
    lookup_table_construction_wrapper<QUERY_NUM>(
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

    scan_controller<QUERY_NUM>(
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

    load_and_split_PQ_codes_wrapper<QUERY_NUM>(
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

    PQ_lookup_computation_wrapper<QUERY_NUM, STAGE5_COMP_PE_NUM, PQ_CODE_CHANNELS_PER_STREAM>(
        nprobe,
        s_single_PQ, 
        s_distance_LUT, 
        s_scanned_entries_every_cell_PQ_lookup_computation,
        s_last_valid_channel,
        s_single_PQ_result);



    hls::stream<single_PQ_result> s_output; // the top 10 numbers
#pragma HLS stream variable=s_output depth=512
// #pragma HLS RESOURCE variable=s_output core=FIFO_BRAM

    stage6_priority_queue_group_L2_wrapper<QUERY_NUM, STAGE5_COMP_PE_NUM>(
        s_scanned_entries_per_query_Priority_queue, 
        s_single_PQ_result,
        s_output);
        ///////// Network Send Starts /////////

        hls::stream<ap_uint512_t> s_network_results;
#pragma HLS stream variable=s_network_results depth=512
// #pragma HLS RESOURCE variable=s_output core=FIFO_BRAM

        network_output_converter_K_1<QUERY_NUM>(
            s_output, 
            s_network_results);

        int pkgWordCountOut = 1;
        if (TOPK == 1) {
            pkgWordCountOut = 1;
        }
        else if (TOPK == 10) {
            pkgWordCountOut = 2;
        }
        else if (TOPK == 100) {
            pkgWordCountOut = 13;
        }
        int expectedTxByteCnt = QUERY_NUM * 64 * pkgWordCountOut;

        sendData(
            m_axis_tcp_tx_meta, 
            m_axis_tcp_tx_data, 
            s_axis_tcp_tx_status,
            s_network_results,
            s_sessionID_out,
            expectedTxByteCnt, 
            pkgWordCountOut);
          
          
        tie_off_tcp_open_connection(m_axis_tcp_open_connection, 
            s_axis_tcp_open_status);


        // tie_off_tcp_tx(m_axis_tcp_tx_meta, 
        //                m_axis_tcp_tx_data, 
        //                s_axis_tcp_tx_status);
        tie_off_udp(s_axis_udp_rx, 
            m_axis_udp_tx, 
            s_axis_udp_rx_meta, 
            m_axis_udp_tx_meta);

        tie_off_tcp_close_con(m_axis_tcp_close_connection);
    }
}
