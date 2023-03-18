/*
Variable to be replaced (<--variable_name-->):
    PQ_lookup_computation_wrapper_arguments
    PQ_lookup_computation_wrapper_replicate_signal
*/

#pragma once 

#include "constants.hpp"
#include "types.hpp"

template<const int stream_num, const int PQ_code_channels_per_stream>
void PQ_lookup_computation_wrapper(
    const int query_num, 
    const int nprobe, 
    hls::stream<single_PQ> (&s_single_PQ)[stream_num], 
    hls::stream<distance_LUT_PQ16_t>& s_distance_LUT,
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation, 
    hls::stream<int>& s_last_valid_channel, 
    hls::stream<single_PQ_result> (&s_single_PQ_result)[SORT_GROUP_NUM][16]);


void dummy_distance_LUT_sender(
    const int query_num, 
    const int nprobe,
    hls::stream<distance_LUT_PQ16_t>& s_distance_LUT) {

    distance_LUT_PQ16_t dist_row;

    dist_row.dist_0 = 0; 
    dist_row.dist_1 = 1; 
    dist_row.dist_2 = 2;
    dist_row.dist_3 = 3; 
    dist_row.dist_4 = 4; 
    dist_row.dist_5 = 5; 
    dist_row.dist_6 = 6; 
    dist_row.dist_7 = 7; 
    dist_row.dist_8 = 8; 
    dist_row.dist_9 = 9; 
    dist_row.dist_10 = 10; 
    dist_row.dist_11 = 11; 
    dist_row.dist_12 = 12; 
    dist_row.dist_13 = 13; 
    dist_row.dist_14 = 14; 
    dist_row.dist_15 = 15;

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int row_id = 0; row_id < K; row_id++) {
#pragma HLS pipeline II=1
                s_distance_LUT.write(dist_row);
            }

        }
    }
    
}

void PQ_lookup_computation(
    const int query_num, 
    const int nprobe,
    // input streams
    hls::stream<distance_LUT_PQ16_t>& s_distance_LUT_in,
    hls::stream<single_PQ>& s_single_PQ,
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation,
    hls::stream<int>& s_last_element_valid_PQ_lookup_computation, 
    // output streams
    hls::stream<distance_LUT_PQ16_t>& s_distance_LUT_out,
    hls::stream<single_PQ_result>& s_single_PQ_result) {

    float distance_LUT[16][256];
#pragma HLS array_partition variable=distance_LUT dim=1
#pragma HLS resource variable=distance_LUT core=RAM_2P_BRAM

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = 
                s_scanned_entries_every_cell_PQ_lookup_computation.read();
            int last_element_valid = 
                s_last_element_valid_PQ_lookup_computation.read();

            // Stage A: init distance LUT
            for (int row_id = 0; row_id < K; row_id++) {
#pragma HLS pipeline II=1

                // without duplication, HLS cannot achieve II=1
                distance_LUT_PQ16_t dist_row = s_distance_LUT_in.read();
                s_distance_LUT_out.write(dist_row);
                
                // col 0 ~ 7
                distance_LUT[0][row_id] = dist_row.dist_0; 
                distance_LUT[1][row_id] = dist_row.dist_1; 
                distance_LUT[2][row_id] = dist_row.dist_2;
                distance_LUT[3][row_id] = dist_row.dist_3; 
                distance_LUT[4][row_id] = dist_row.dist_4; 
                distance_LUT[5][row_id] = dist_row.dist_5; 
                distance_LUT[6][row_id] = dist_row.dist_6; 
                distance_LUT[7][row_id] = dist_row.dist_7; 
                distance_LUT[8][row_id] = dist_row.dist_8; 
                distance_LUT[9][row_id] = dist_row.dist_9; 
                distance_LUT[10][row_id] = dist_row.dist_10; 
                distance_LUT[11][row_id] = dist_row.dist_11; 
                distance_LUT[12][row_id] = dist_row.dist_12; 
                distance_LUT[13][row_id] = dist_row.dist_13; 
                distance_LUT[14][row_id] = dist_row.dist_14; 
                distance_LUT[15][row_id] = dist_row.dist_15;
            }

            // Stage B: compute estimated distance
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline II=1

                single_PQ PQ_local = s_single_PQ.read();

                single_PQ_result out; 
                out.vec_ID = PQ_local.vec_ID;
                
                out.dist = 
                    distance_LUT[0][PQ_local.PQ_code[0]] + 
                    distance_LUT[1][PQ_local.PQ_code[1]] + 
                    distance_LUT[2][PQ_local.PQ_code[2]] + 
                    distance_LUT[3][PQ_local.PQ_code[3]] + 
                    distance_LUT[4][PQ_local.PQ_code[4]] + 
                    distance_LUT[5][PQ_local.PQ_code[5]] + 
                    distance_LUT[6][PQ_local.PQ_code[6]] + 
                    distance_LUT[7][PQ_local.PQ_code[7]] + 
                    distance_LUT[8][PQ_local.PQ_code[8]] + 
                    distance_LUT[9][PQ_local.PQ_code[9]] + 
                    distance_LUT[10][PQ_local.PQ_code[10]] + 
                    distance_LUT[11][PQ_local.PQ_code[11]] + 
                    distance_LUT[12][PQ_local.PQ_code[12]] + 
                    distance_LUT[13][PQ_local.PQ_code[13]] + 
                    distance_LUT[14][PQ_local.PQ_code[14]] + 
                    distance_LUT[15][PQ_local.PQ_code[15]];

                // for padded element, replace its distance by large number
                // judge the last 3 distances
                if (entry_id >= scanned_entries_every_cell - last_element_valid) {
                    out.vec_ID = -1;
                    out.dist = LARGE_NUM;
                }
                s_single_PQ_result.write(out);
            }
        }
    }
}


void dummy_PQ_result_sender(
    const int query_num, 
    const int nprobe,
    hls::stream<int>& s_scanned_entries_every_cell_Dummy,
    hls::stream<single_PQ_result> &s_single_PQ_result) {

    single_PQ_result out; 
    out.vec_ID = -1;
    out.dist = LARGE_NUM;
    
    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell = 
                s_scanned_entries_every_cell_Dummy.read();

            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline II=1
                s_single_PQ_result.write(out);
            }
        }
    }
}

template<const int stream_num>
void replicate_s_scanned_entries_every_cell_PQ_lookup_computation(
    const int query_num, 
    const int nprobe,
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation,
    hls::stream<int> (&s_scanned_entries_every_cell_PQ_lookup_computation_replicated)[stream_num]) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell_PQ_lookup_computation = 
                s_scanned_entries_every_cell_PQ_lookup_computation.read();

            for (int s = 0; s < stream_num; s++) {
#pragma HLS UNROLL
                s_scanned_entries_every_cell_PQ_lookup_computation_replicated[s].write(
                    scanned_entries_every_cell_PQ_lookup_computation);
            }
        }
    }
}

template<const int stream_num, const int dummy_stream_num>
void replicate_s_scanned_entries_every_cell_PQ_lookup_computation(
    const int query_num, 
    const int nprobe, 
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation,
    hls::stream<int> (&s_scanned_entries_every_cell_PQ_lookup_computation_replicated)[stream_num],
    hls::stream<int> (&s_scanned_entries_every_cell_Dummy_replicated)[dummy_stream_num]) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int scanned_entries_every_cell_PQ_lookup_computation = 
                s_scanned_entries_every_cell_PQ_lookup_computation.read();

            for (int s = 0; s < stream_num; s++) {
#pragma HLS UNROLL
                s_scanned_entries_every_cell_PQ_lookup_computation_replicated[s].write(
                    scanned_entries_every_cell_PQ_lookup_computation);
            }
            for (int s = 0; s < dummy_stream_num; s++) {
#pragma HLS UNROLL
                s_scanned_entries_every_cell_Dummy_replicated[s].write(
                    scanned_entries_every_cell_PQ_lookup_computation);
            }
        }
    }
}

template<const int stream_num, const int PQ_code_channels_per_stream>
void send_s_last_element_valid_PQ_lookup_computation(
    const int query_num, 
    const int nprobe, 
    hls::stream<int>& s_last_valid_channel,
    hls::stream<int> (&s_last_element_valid_PQ_lookup_computation)[stream_num]) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int last_valid_channel = s_last_valid_channel.read(); 

            for (int s = 0; s < stream_num; s++) {
#pragma HLS UNROLL
                // send ONE signal to the compute PE to indicate which of the last elements are pads
                // For example, each compute unit is connected to 1 HBM channels (3 PQ codes / CC), 
                //   then, PQ_code_channels_per_stream = 3
                // the output signal=X means the last X elements are pads
                if (s * PQ_code_channels_per_stream + PQ_code_channels_per_stream - 1 <= last_valid_channel) {
                    // last stream <= valid channel ID, nothing is pad
                    s_last_element_valid_PQ_lookup_computation[s].write(0); 
                }
                else if (s * PQ_code_channels_per_stream > last_valid_channel) {
                    // first stream > valid channel ID, all last values are pad
                    s_last_element_valid_PQ_lookup_computation[s].write(PQ_code_channels_per_stream);
                }
                else {
                    // in the middle
                    int pad_num = s * PQ_code_channels_per_stream + PQ_code_channels_per_stream - 1 - last_valid_channel;
                    s_last_element_valid_PQ_lookup_computation[s].write(pad_num);
                }
            }
        }
    }
}


void dummy_distance_LUT_consumer(
    const int query_num, 
    const int nprobe,
    hls::stream<distance_LUT_PQ16_t>& s_distance_LUT) {

    distance_LUT_PQ16_t dist_row;

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int row_id = 0; row_id < K; row_id++) {
#pragma HLS pipeline II=1
                dist_row = s_distance_LUT.read();
            }

        }
    }
}

template<const int stream_num, const int PQ_code_channels_per_stream>
void PQ_lookup_computation_wrapper(
    const int query_num, 
    const int nprobe, 
    hls::stream<single_PQ> (&s_single_PQ)[stream_num], 
    hls::stream<distance_LUT_PQ16_t>& s_distance_LUT,
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation, 
    hls::stream<int>& s_last_valid_channel, 
    hls::stream<single_PQ_result> (&s_single_PQ_result)[SORT_GROUP_NUM][16]) {
#pragma HLS inline

    hls::stream<distance_LUT_PQ16_t> s_distance_LUT_systolic[stream_num];
#pragma HLS stream variable=s_distance_LUT_systolic depth=8
#pragma HLS array_partition variable=s_distance_LUT_systolic complete
// #pragma HLS RESOURCE variable=s_distance_LUT_systolic core=FIFO_SRL

    hls::stream<int> s_scanned_entries_every_cell_PQ_lookup_computation_replicated[stream_num];
#pragma HLS stream variable=s_scanned_entries_every_cell_PQ_lookup_computation_replicated depth=8
#pragma HLS array_partition variable=s_scanned_entries_every_cell_PQ_lookup_computation_replicated complete
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_PQ_lookup_computation_replicated core=FIFO_SRL


    hls::stream<int> s_scanned_entries_every_cell_Dummy_replicated[SORT_GROUP_NUM * 16 - stream_num]; // 32 - 3 * 10 = 2
#pragma HLS stream variable=s_scanned_entries_every_cell_Dummy_replicated depth=8
#pragma HLS array_partition variable=s_scanned_entries_every_cell_Dummy_replicated complete
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_Dummy_replicated core=FIFO_SRL

    replicate_s_scanned_entries_every_cell_PQ_lookup_computation<stream_num>(
        query_num,
        nprobe, 
        s_scanned_entries_every_cell_PQ_lookup_computation, 
        s_scanned_entries_every_cell_PQ_lookup_computation_replicated,
        s_scanned_entries_every_cell_Dummy_replicated);


    hls::stream<int> s_last_element_valid_PQ_lookup_computation[stream_num];
#pragma HLS stream variable=s_last_element_valid_PQ_lookup_computation depth=8
#pragma HLS array_partition variable=s_last_element_valid_PQ_lookup_computation complete
// #pragma HLS RESOURCE variable=s_last_element_valid_PQ_lookup_computation core=FIFO_SRL

    // Note, here interpret the last valid element, rather than simply replicate
    send_s_last_element_valid_PQ_lookup_computation<stream_num, PQ_code_channels_per_stream>(
        query_num, 
        nprobe,
        s_last_valid_channel, 
        s_last_element_valid_PQ_lookup_computation);

#if SORT_GROUP_NUM == 0
    // head PE, get input from s_distance_LUT
    PQ_lookup_computation(
        query_num, 
        nprobe,
        // input streams
        s_distance_LUT,
        s_single_PQ[0],
        s_scanned_entries_every_cell_PQ_lookup_computation_replicated[0],
        s_last_element_valid_PQ_lookup_computation[0], 
        // output streams
        s_distance_LUT_systolic[0],
        s_single_PQ_result[0]);

    // remaining PEs
    for (int j = 1; j < stream_num; j++) {
#pragma HLS UNROLL
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT_systolic[j - 1],
            s_single_PQ[j],
            s_scanned_entries_every_cell_PQ_lookup_computation_replicated[j],
            s_last_element_valid_PQ_lookup_computation[j], 
            // output streams
            s_distance_LUT_systolic[j],
            s_single_PQ_result[j]);
    }
#elif SORT_GROUP_NUM == 1
    // PE 0, get input from s_distance_LUT
    PQ_lookup_computation(
        query_num, 
        nprobe,
        // input streams
        s_distance_LUT,
        s_single_PQ[0],
        s_scanned_entries_every_cell_PQ_lookup_computation_replicated[0],
        s_last_element_valid_PQ_lookup_computation[0], 
        // output streams
        s_distance_LUT_systolic[0],
        s_single_PQ_result[0][0]);

    // head PE 1~15 
    for (int j = 1; j < stream_num; j++) {
#pragma HLS UNROLL
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT_systolic[j - 1],
            s_single_PQ[j],
            s_scanned_entries_every_cell_PQ_lookup_computation_replicated[j],
            s_last_element_valid_PQ_lookup_computation[j], 
            // output streams
            s_distance_LUT_systolic[j],
            s_single_PQ_result[0][j]);
    }


    for (int j = stream_num; j < 16; j++) {
#pragma HLS UNROLL
        dummy_PQ_result_sender(
            query_num,
            nprobe,
            s_scanned_entries_every_cell_Dummy_replicated[j - stream_num], 
            s_single_PQ_result[0][j]);
    }
    

#elif SORT_GROUP_NUM == 2
    // PE 0, get input from s_distance_LUT
    PQ_lookup_computation(
        query_num, 
        nprobe,
        // input streams
        s_distance_LUT,
        s_single_PQ[0],
        s_scanned_entries_every_cell_PQ_lookup_computation_replicated[0],
        s_last_element_valid_PQ_lookup_computation[0], 
        // output streams
        s_distance_LUT_systolic[0],
        s_single_PQ_result[0][0]);

    // head PE 1~15 
    for (int j = 1; j < 16; j++) {
#pragma HLS UNROLL
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT_systolic[j - 1],
            s_single_PQ[j],
            s_scanned_entries_every_cell_PQ_lookup_computation_replicated[j],
            s_last_element_valid_PQ_lookup_computation[j], 
            // output streams
            s_distance_LUT_systolic[j],
            s_single_PQ_result[0][j]);
    }

    // tail PE 16~26 
    for (int j = 0; j < stream_num - 16 * (SORT_GROUP_NUM - 1); j++) {
#pragma HLS UNROLL
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT_systolic[(SORT_GROUP_NUM - 1) * 16 + j - 1],
            s_single_PQ[(SORT_GROUP_NUM - 1) * 16 + j],
            s_scanned_entries_every_cell_PQ_lookup_computation_replicated[(SORT_GROUP_NUM - 1) * 16 + j],
            s_last_element_valid_PQ_lookup_computation[(SORT_GROUP_NUM - 1) * 16 + j], 
            // output streams
            s_distance_LUT_systolic[(SORT_GROUP_NUM - 1) * 16 + j],
            s_single_PQ_result[(SORT_GROUP_NUM - 1)][j]);
    }


    for (int j = stream_num - 16 * (SORT_GROUP_NUM - 1); j < 16; j++) {
#pragma HLS UNROLL
        dummy_PQ_result_sender(
            query_num,
            nprobe,
            s_scanned_entries_every_cell_Dummy_replicated[j - (stream_num - 16 * (SORT_GROUP_NUM - 1))], 
            s_single_PQ_result[SORT_GROUP_NUM - 1][j]);
    }
    

#elif SORT_GROUP_NUM >= 3
    // PE 0, get input from s_distance_LUT
    PQ_lookup_computation(
        query_num, 
        nprobe,
        // input streams
        s_distance_LUT,
        s_single_PQ[0],
        s_scanned_entries_every_cell_PQ_lookup_computation_replicated[0],
        s_last_element_valid_PQ_lookup_computation[0], 
        // output streams
        s_distance_LUT_systolic[0],
        s_single_PQ_result[0][0]);

    // head PE 1~15 
    for (int j = 1; j < 16; j++) {
#pragma HLS UNROLL
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT_systolic[j - 1],
            s_single_PQ[j],
            s_scanned_entries_every_cell_PQ_lookup_computation_replicated[j],
            s_last_element_valid_PQ_lookup_computation[j], 
            // output streams
            s_distance_LUT_systolic[j],
            s_single_PQ_result[0][j]);
    }

    // middle PE
    for (int i = 1; i < SORT_GROUP_NUM - 1; i++) {
        for (int j = 0; j < 16; j++) {
#pragma HLS UNROLL
            PQ_lookup_computation(
                query_num, 
                nprobe,
                // input streams
                s_distance_LUT_systolic[i * 16 + j - 1],
                s_single_PQ[i * 16 + j],
                s_scanned_entries_every_cell_PQ_lookup_computation_replicated[i * 16 + j],
                s_last_element_valid_PQ_lookup_computation[i * 16 + j], 
                // output streams
                s_distance_LUT_systolic[i * 16 + j],
                s_single_PQ_result[i][j]);
        }
    }

    // tail PE  
    for (int j = 0; j < stream_num - 16 * (SORT_GROUP_NUM - 1); j++) {
#pragma HLS UNROLL
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT_systolic[(SORT_GROUP_NUM - 1) * 16 + j - 1],
            s_single_PQ[(SORT_GROUP_NUM - 1) * 16 + j],
            s_scanned_entries_every_cell_PQ_lookup_computation_replicated[(SORT_GROUP_NUM - 1) * 16 + j],
            s_last_element_valid_PQ_lookup_computation[(SORT_GROUP_NUM - 1) * 16 + j], 
            // output streams
            s_distance_LUT_systolic[(SORT_GROUP_NUM - 1) * 16 + j],
            s_single_PQ_result[(SORT_GROUP_NUM - 1)][j]);
    }


    for (int j = stream_num - 16 * (SORT_GROUP_NUM - 1); j < 16; j++) {
#pragma HLS UNROLL
        dummy_PQ_result_sender(
            query_num,
            nprobe,
            s_scanned_entries_every_cell_Dummy_replicated[j - (stream_num - 16 * (SORT_GROUP_NUM - 1))], 
            s_single_PQ_result[SORT_GROUP_NUM - 1][j]);
    }
    

#endif

    // consume the systolic output of the last PE
    dummy_distance_LUT_consumer(
        query_num, 
        nprobe,
        s_distance_LUT_systolic[stream_num - 1]);
}

