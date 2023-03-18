/*
Variable to be replaced (<--variable_name-->):
    load_and_split_PQ_codes_wrapper_arguments
    load_and_split_PQ_codes_wrapper_func_body
*/


#pragma once 

#include "constants.hpp"
#include "types.hpp"

////////////////////     Declaration     ////////////////////

void load_and_split_PQ_codes_wrapper(
    const int query_num,
    const int nprobe,
<--load_and_split_PQ_codes_wrapper_arguments-->
    hls::stream<int>& s_start_addr_every_cell,
    hls::stream<int>& s_scanned_entries_every_cell_Load_unit,
    hls::stream<single_PQ> (&s_single_PQ)[STAGE5_COMP_PE_NUM]);

////////////////////     Definition     ////////////////////

void load_PQ_codes(
    const int query_num,
    const int nprobe,
    const ap_uint512_t* src,
    hls::stream<int>& s_scanned_entries_every_cell,
    hls::stream<int>& s_start_addr_every_cell,
    hls::stream<ap_uint512_t>& s_raw_input) {

    // s_scanned_entries_every_cell -> 
    //   number of axi width of scanned PQ code per Voronoi cell, 
    //   e.g. AXI width = 512 -> 64 bytes = 20 byte * 3 = 3 PQ code
    //      need scan 299 PQ code ->  axi_num_per_scanned_cell = 100
    //   read number = query_num * nprobe

    for (int query_id = 0; query_id < query_num; query_id++) {
#pragma HLS loop_flatten

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell.read();
            int start_addr = s_start_addr_every_cell.read();

            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline II=1
                s_raw_input.write(src[start_addr + entry_id]);
            }
        }
    }
}

three_PQ_codes ap_uint512_to_three_PQ_codes(ap_uint<512> in) {
// AXI datawidth of 480 is banned, must use 2^n, e.g., 512
#pragma HLS pipeline
#pragma HLS inline off
#pragma HLS interface port=return register
    three_PQ_codes out;

    ap_uint<32> tmp_int0 = in.range(31 + 0, 0 + 0);
    out.PQ_0.vec_ID = *((int*)(&tmp_int0));
    for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
        ap_uint<8> tmp_char = in.range(0 + 7 + 32 + i * 8, 0 + 32 + i * 8);
        out.PQ_0.PQ_code[i] = *((unsigned char*)(&tmp_char));
    }
    ap_uint<32> tmp_int1 = in.range(31 + 160, 0 + 160);
    out.PQ_1.vec_ID = *((int*)(&tmp_int1));
    for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
        ap_uint<8> tmp_char = in.range(160 + 7 + 32 + i * 8, 160 + 32 + i * 8);
        out.PQ_1.PQ_code[i] = *((unsigned char*)(&tmp_char));
    }
    ap_uint<32> tmp_int2 = in.range(31 + 320, 0 + 320);
    out.PQ_2.vec_ID = *((int*)(&tmp_int2));
    for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
        ap_uint<8> tmp_char = in.range(320 + 7 + 32 + i * 8, 320 + 32 + i * 8);
        out.PQ_2.PQ_code[i] = *((unsigned char*)(&tmp_char));
    }

    return out;
}


void type_conversion_and_split(
    const int query_num,
    const int nprobe,
    hls::stream<int>& s_scanned_entries_every_cell,
    hls::stream<ap_uint512_t>& s_raw_input,
    hls::stream<single_PQ>& s_single_PQ) {


    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline II=3
                ap_uint512_t in = s_raw_input.read();
                three_PQ_codes out = ap_uint512_to_three_PQ_codes(in);
                s_single_PQ.write(out.PQ_0);
                s_single_PQ.write(out.PQ_1);
                s_single_PQ.write(out.PQ_2);
            }
        }
    }
}

void type_conversion_and_split(
    const int query_num,
    const int nprobe,
    hls::stream<int>& s_scanned_entries_every_cell,
    hls::stream<ap_uint512_t>& s_raw_input,
    hls::stream<single_PQ>& s_single_PQ_0,
    hls::stream<single_PQ>& s_single_PQ_1,
    hls::stream<single_PQ>& s_single_PQ_2) {


    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline II=1
                ap_uint512_t in = s_raw_input.read();
                three_PQ_codes out = ap_uint512_to_three_PQ_codes(in);
                s_single_PQ_0.write(out.PQ_0);
                s_single_PQ_1.write(out.PQ_1);
                s_single_PQ_2.write(out.PQ_2);
            }
        }
    }
}

void load_and_split_PQ_codes(
    const int query_num,
    const int nprobe,
    const ap_uint512_t* HBM_in, // HBM for PQ code + vecID storage
    hls::stream<int>& s_start_addr_every_cell,
    hls::stream<int>& s_scanned_entries_every_cell_Load_unit,
    hls::stream<int>& s_scanned_entries_every_cell_Split_unit,
    hls::stream<single_PQ>& s_single_PQ) {

#pragma HLS inline

    hls::stream<ap_uint512_t> s_raw_input; // raw AXI width input

    load_PQ_codes(
        query_num,
        nprobe,
        HBM_in, 
        s_scanned_entries_every_cell_Load_unit, 
        s_start_addr_every_cell, 
        s_raw_input);
    type_conversion_and_split(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Split_unit,
        s_raw_input, 
        s_single_PQ);
}

void load_and_split_PQ_codes(
    const int query_num,
    const int nprobe,
    const ap_uint512_t* HBM_in, // HBM for PQ code + vecID storage
    hls::stream<int>& s_start_addr_every_cell,
    hls::stream<int>& s_scanned_entries_every_cell_Load_unit,
    hls::stream<int>& s_scanned_entries_every_cell_Split_unit,
    hls::stream<single_PQ>& s_single_PQ_0,
    hls::stream<single_PQ>& s_single_PQ_1,
    hls::stream<single_PQ>& s_single_PQ_2) {

#pragma HLS inline

    hls::stream<ap_uint512_t> s_raw_input; // raw AXI width input

    load_PQ_codes(
        query_num,
        nprobe,
        HBM_in, 
        s_scanned_entries_every_cell_Load_unit, 
        s_start_addr_every_cell, 
        s_raw_input);
    type_conversion_and_split(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Split_unit,
        s_raw_input, s_single_PQ_0, s_single_PQ_1, s_single_PQ_2);
}

void replicate_s_start_addr_every_cell(
    const int query_num,
    const int nprobe,
    hls::stream<int>& s_start_addr_every_cell,
    hls::stream<int> (&s_start_addr_every_cell_replicated)[HBM_CHANNEL_NUM]) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int start_addr_every_cell = s_start_addr_every_cell.read();

            for (int s = 0; s < HBM_CHANNEL_NUM; s++) {
#pragma HLS UNROLL
                s_start_addr_every_cell_replicated[s].write(start_addr_every_cell);
            }
        }
    }
}

void replicate_s_scanned_entries_every_cell(
    const int query_num,
    const int nprobe,
    hls::stream<int>& s_scanned_entries_every_cell_in,
    hls::stream<int> (&s_scanned_entries_every_cell_Load_unit_replicated)[HBM_CHANNEL_NUM],
    hls::stream<int> (&s_scanned_entries_every_cell_Split_unit_replicated)[HBM_CHANNEL_NUM]) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_in.read();

            for (int s = 0; s < HBM_CHANNEL_NUM; s++) {
#pragma HLS UNROLL
                s_scanned_entries_every_cell_Load_unit_replicated[s].write(
                    scanned_entries_every_cell);
            }
            for (int s = 0; s < HBM_CHANNEL_NUM; s++) {
#pragma HLS UNROLL
                s_scanned_entries_every_cell_Split_unit_replicated[s].write(
                    scanned_entries_every_cell);
            }
        }
    }
}

void replicate_s_scanned_entries_every_cell(
    const int query_num,
    const int nprobe,
    hls::stream<int>& s_scanned_entries_every_cell_in,
    hls::stream<int> (&s_scanned_entries_every_cell_Load_unit_replicated)[HBM_CHANNEL_NUM],
    hls::stream<int> (&s_scanned_entries_every_cell_Split_unit_replicated)[HBM_CHANNEL_NUM],
    hls::stream<int> (&s_scanned_entries_every_cell_Merge_unit_replicated)[STAGE5_COMP_PE_NUM]) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_in.read();

            for (int s = 0; s < HBM_CHANNEL_NUM; s++) {
#pragma HLS UNROLL
                s_scanned_entries_every_cell_Load_unit_replicated[s].write(
                    scanned_entries_every_cell);
            }
            for (int s = 0; s < HBM_CHANNEL_NUM; s++) {
#pragma HLS UNROLL
                s_scanned_entries_every_cell_Split_unit_replicated[s].write(
                    scanned_entries_every_cell);
            }
            for (int s = 0; s < STAGE5_COMP_PE_NUM; s++) {
#pragma HLS UNROLL
                s_scanned_entries_every_cell_Merge_unit_replicated[s].write(
                    scanned_entries_every_cell);
            }
        }
    }
}

void merge_HBM_channel_PQ_codes_2_in_1(
    const int query_num,
    const int nprobe,
    hls::stream<int> &s_scanned_entries_every_cell_Merge_unit_replicated,
    hls::stream<single_PQ> &s_single_PQ_A,
    hls::stream<single_PQ> &s_single_PQ_B,
    hls::stream<single_PQ> &s_single_PQ_out) {
    // NOTE! Must use 2 consecutive channel as input
    //  in order to preserve the order for padding detection

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_Merge_unit_replicated.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline

                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());

                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
            }
        }
    }
}

void merge_HBM_channel_PQ_codes_3_in_1(
    const int query_num,
    const int nprobe,
    hls::stream<int> &s_scanned_entries_every_cell_Merge_unit_replicated,
    hls::stream<single_PQ> &s_single_PQ_A,
    hls::stream<single_PQ> &s_single_PQ_B,
    hls::stream<single_PQ> &s_single_PQ_C,
    hls::stream<single_PQ> &s_single_PQ_out) {
    // NOTE! Must use 3 consecutive channels as input
    //  in order to preserve the order for padding detection

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_Merge_unit_replicated.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline

                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());

                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());

                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());
            }
        }
    }
}

void merge_HBM_channel_PQ_codes_4_in_1(
    const int query_num,
    const int nprobe,
    hls::stream<int> &s_scanned_entries_every_cell_Merge_unit_replicated,
    hls::stream<single_PQ> &s_single_PQ_A,
    hls::stream<single_PQ> &s_single_PQ_B,
    hls::stream<single_PQ> &s_single_PQ_C,
    hls::stream<single_PQ> &s_single_PQ_D,
    hls::stream<single_PQ> &s_single_PQ_out) {
    // NOTE! Must use 4 consecutive channels as input
    //  in order to preserve the order for padding detection

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_Merge_unit_replicated.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline

                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());

                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());

                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());

                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());
            }
        }
    }
}

void merge_HBM_channel_PQ_codes_5_in_1(
    const int query_num,
    const int nprobe,
    hls::stream<int> &s_scanned_entries_every_cell_Merge_unit_replicated,
    hls::stream<single_PQ> &s_single_PQ_A,
    hls::stream<single_PQ> &s_single_PQ_B,
    hls::stream<single_PQ> &s_single_PQ_C,
    hls::stream<single_PQ> &s_single_PQ_D,
    hls::stream<single_PQ> &s_single_PQ_E,
    hls::stream<single_PQ> &s_single_PQ_out) {
    // NOTE! Must use 4 consecutive channels as input
    //  in order to preserve the order for padding detection

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_Merge_unit_replicated.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline

                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());

                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());

                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());

                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());

                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());
            }
        }
    }
}

void merge_HBM_channel_PQ_codes_6_in_1(
    const int query_num,
    const int nprobe,
    hls::stream<int> &s_scanned_entries_every_cell_Merge_unit_replicated,
    hls::stream<single_PQ> &s_single_PQ_A,
    hls::stream<single_PQ> &s_single_PQ_B,
    hls::stream<single_PQ> &s_single_PQ_C,
    hls::stream<single_PQ> &s_single_PQ_D,
    hls::stream<single_PQ> &s_single_PQ_E,
    hls::stream<single_PQ> &s_single_PQ_F,
    hls::stream<single_PQ> &s_single_PQ_out) {
    // NOTE! Must use 4 consecutive channels as input
    //  in order to preserve the order for padding detection

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_Merge_unit_replicated.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline

                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());

                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());

                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());

                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());

                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());

                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());
            }
        }
    }
}

void merge_HBM_channel_PQ_codes_7_in_1(
    const int query_num,
    const int nprobe,
    hls::stream<int> &s_scanned_entries_every_cell_Merge_unit_replicated,
    hls::stream<single_PQ> &s_single_PQ_A,
    hls::stream<single_PQ> &s_single_PQ_B,
    hls::stream<single_PQ> &s_single_PQ_C,
    hls::stream<single_PQ> &s_single_PQ_D,
    hls::stream<single_PQ> &s_single_PQ_E,
    hls::stream<single_PQ> &s_single_PQ_F,
    hls::stream<single_PQ> &s_single_PQ_G,
    hls::stream<single_PQ> &s_single_PQ_out) {
    // NOTE! Must use 4 consecutive channels as input
    //  in order to preserve the order for padding detection

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_Merge_unit_replicated.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline

                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());

                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());

                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());

                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());

                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());

                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());

                s_single_PQ_out.write(s_single_PQ_G.read());
                s_single_PQ_out.write(s_single_PQ_G.read());
                s_single_PQ_out.write(s_single_PQ_G.read());
            }
        }
    }
}

void merge_HBM_channel_PQ_codes_8_in_1(
    const int query_num,
    const int nprobe,
    hls::stream<int> &s_scanned_entries_every_cell_Merge_unit_replicated,
    hls::stream<single_PQ> &s_single_PQ_A,
    hls::stream<single_PQ> &s_single_PQ_B,
    hls::stream<single_PQ> &s_single_PQ_C,
    hls::stream<single_PQ> &s_single_PQ_D,
    hls::stream<single_PQ> &s_single_PQ_E,
    hls::stream<single_PQ> &s_single_PQ_F,
    hls::stream<single_PQ> &s_single_PQ_G,
    hls::stream<single_PQ> &s_single_PQ_H,
    hls::stream<single_PQ> &s_single_PQ_out) {
    // NOTE! Must use 4 consecutive channels as input
    //  in order to preserve the order for padding detection

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_Merge_unit_replicated.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline

                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());

                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());

                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());

                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());

                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());

                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());

                s_single_PQ_out.write(s_single_PQ_G.read());
                s_single_PQ_out.write(s_single_PQ_G.read());
                s_single_PQ_out.write(s_single_PQ_G.read());

                s_single_PQ_out.write(s_single_PQ_H.read());
                s_single_PQ_out.write(s_single_PQ_H.read());
                s_single_PQ_out.write(s_single_PQ_H.read());
            }
        }
    }
}

void merge_HBM_channel_PQ_codes_9_in_1(
    const int query_num,
    const int nprobe,
    hls::stream<int> &s_scanned_entries_every_cell_Merge_unit_replicated,
    hls::stream<single_PQ> &s_single_PQ_A,
    hls::stream<single_PQ> &s_single_PQ_B,
    hls::stream<single_PQ> &s_single_PQ_C,
    hls::stream<single_PQ> &s_single_PQ_D,
    hls::stream<single_PQ> &s_single_PQ_E,
    hls::stream<single_PQ> &s_single_PQ_F,
    hls::stream<single_PQ> &s_single_PQ_G,
    hls::stream<single_PQ> &s_single_PQ_H,
    hls::stream<single_PQ> &s_single_PQ_I,
    hls::stream<single_PQ> &s_single_PQ_out) {
    // NOTE! Must use 4 consecutive channels as input
    //  in order to preserve the order for padding detection

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_Merge_unit_replicated.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline

                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());

                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());

                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());

                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());

                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());

                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());

                s_single_PQ_out.write(s_single_PQ_G.read());
                s_single_PQ_out.write(s_single_PQ_G.read());
                s_single_PQ_out.write(s_single_PQ_G.read());

                s_single_PQ_out.write(s_single_PQ_H.read());
                s_single_PQ_out.write(s_single_PQ_H.read());
                s_single_PQ_out.write(s_single_PQ_H.read());

                s_single_PQ_out.write(s_single_PQ_I.read());
                s_single_PQ_out.write(s_single_PQ_I.read());
                s_single_PQ_out.write(s_single_PQ_I.read());
            }
        }
    }
}

void merge_HBM_channel_PQ_codes_10_in_1(
    const int query_num,
    const int nprobe,
    hls::stream<int> &s_scanned_entries_every_cell_Merge_unit_replicated,
    hls::stream<single_PQ> &s_single_PQ_A,
    hls::stream<single_PQ> &s_single_PQ_B,
    hls::stream<single_PQ> &s_single_PQ_C,
    hls::stream<single_PQ> &s_single_PQ_D,
    hls::stream<single_PQ> &s_single_PQ_E,
    hls::stream<single_PQ> &s_single_PQ_F,
    hls::stream<single_PQ> &s_single_PQ_G,
    hls::stream<single_PQ> &s_single_PQ_H,
    hls::stream<single_PQ> &s_single_PQ_I,
    hls::stream<single_PQ> &s_single_PQ_J,
    hls::stream<single_PQ> &s_single_PQ_out) {
    // NOTE! Must use 4 consecutive channels as input
    //  in order to preserve the order for padding detection

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = s_scanned_entries_every_cell_Merge_unit_replicated.read();
            
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline

                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());
                s_single_PQ_out.write(s_single_PQ_A.read());

                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());
                s_single_PQ_out.write(s_single_PQ_B.read());

                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());
                s_single_PQ_out.write(s_single_PQ_C.read());

                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());
                s_single_PQ_out.write(s_single_PQ_D.read());

                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());
                s_single_PQ_out.write(s_single_PQ_E.read());

                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());
                s_single_PQ_out.write(s_single_PQ_F.read());

                s_single_PQ_out.write(s_single_PQ_G.read());
                s_single_PQ_out.write(s_single_PQ_G.read());
                s_single_PQ_out.write(s_single_PQ_G.read());

                s_single_PQ_out.write(s_single_PQ_H.read());
                s_single_PQ_out.write(s_single_PQ_H.read());
                s_single_PQ_out.write(s_single_PQ_H.read());

                s_single_PQ_out.write(s_single_PQ_I.read());
                s_single_PQ_out.write(s_single_PQ_I.read());
                s_single_PQ_out.write(s_single_PQ_I.read());

                s_single_PQ_out.write(s_single_PQ_J.read());
                s_single_PQ_out.write(s_single_PQ_J.read());
                s_single_PQ_out.write(s_single_PQ_J.read());
            }
        }
    }
}

void load_and_split_PQ_codes_wrapper(
    const int query_num,
    const int nprobe,
<--load_and_split_PQ_codes_wrapper_arguments-->
    hls::stream<int>& s_start_addr_every_cell,
    hls::stream<int>& s_scanned_entries_every_cell_Load_unit,
    hls::stream<single_PQ> (&s_single_PQ)[STAGE5_COMP_PE_NUM]) {

#pragma HLS inline

    hls::stream<int> s_start_addr_every_cell_replicated[HBM_CHANNEL_NUM];
#pragma HLS stream variable=s_start_addr_every_cell_replicated depth=8
#pragma HLS array_partition variable=s_start_addr_every_cell_replicated complete
// #pragma HLS RESOURCE variable=s_start_addr_every_cell_replicated core=FIFO_SRL

    replicate_s_start_addr_every_cell(
        query_num,
        nprobe,
        s_start_addr_every_cell, 
        s_start_addr_every_cell_replicated); 

    hls::stream<int> s_scanned_entries_every_cell_Load_unit_replicated[HBM_CHANNEL_NUM];
#pragma HLS stream variable=s_scanned_entries_every_cell_Load_unit_replicated depth=8
#pragma HLS array_partition variable=s_scanned_entries_every_cell_Load_unit_replicated complete
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_Load_unit_replicated core=FIFO_SRL

    hls::stream<int> s_scanned_entries_every_cell_Split_unit_replicated[HBM_CHANNEL_NUM];
#pragma HLS stream variable=s_scanned_entries_every_cell_Split_unit_replicated depth=8
#pragma HLS array_partition variable=s_scanned_entries_every_cell_Split_unit_replicated complete
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_Split_unit_replicated core=FIFO_SRL

<--load_and_split_PQ_codes_wrapper_func_body-->
}
