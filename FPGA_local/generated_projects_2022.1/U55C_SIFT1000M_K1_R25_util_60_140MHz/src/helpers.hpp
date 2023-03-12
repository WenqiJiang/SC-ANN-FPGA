/*
Variable to be replaced (<--variable_name-->):

    single line:
        scan_controller_arg_s_scanned_entries_per_query_Sort_and_reduction
        scan_controller_body_s_scanned_entries_per_query_Sort_and_reduction

*/

#pragma once 

#include "constants.hpp"
#include "types.hpp"


void parse_HBM_meta_info(
    const int query_num,
    const int nlist,
    const bool OPQ_enable,
    const float* HBM_meta_info, 
    hls::stream<int>& s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid, 
    hls::stream<float>& s_product_quantizer,
    hls::stream<float>& s_OPQ_init,
    hls::stream<float>& s_query_vectors
    );

void load_query_vectors(
    const int query_num,
    const float* DRAM_query_vector,
    hls::stream<float>& s_query_vectors);

void broadcast_preprocessed_query_vectors(
    const int query_num,
    hls::stream<float>& s_preprocessed_query_vectors,
    hls::stream<float> &s_preprocessed_query_vectors_distance_computation_PE,
    hls::stream<float>& s_preprocessed_query_vectors_lookup_PE);

void load_center_vectors(
    const int nlist,
    const float* table_HBM1, 
    hls::stream<float> &s_center_vectors_init_distance_computation_PE,
    hls::stream<float> &s_center_vectors_init_lookup_PE);

void load_PQ_quantizer(
    const float* DRAM_PQ_quantizer,
    hls::stream<float> &s_PQ_quantizer_init);

void load_OPQ_matrix(
    const bool OPQ_enable,
    const float* HBM_OPQ_matrix, 
    hls::stream<float> &s_OPQ_init);

void split_cell_ID(
    const int query_num,
    const int nprobe,
    hls::stream<dist_cell_ID_t>& s_merge_output, 
    hls::stream<int>& s_searched_cell_id_lookup_PE, 
    hls::stream<int>& s_searched_cell_id_scan_controller);
    
void lookup_center_vectors(
    const int query_num,
    hls::stream<float> &s_center_vectors_init_lookup_PE,
    hls::stream<int>& s_searched_cell_id_lookup_PE,
    hls::stream<float>& s_center_vectors_lookup_PE);

void scan_controller(
    const int query_num,
    const int nlist, 
    const int nprobe,
    const int* HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid,
    hls::stream<int> &s_scanned_cell_id_Input, // from the cluster selection unit
    hls::stream<int> &s_start_addr_every_cell,
    hls::stream<int> &s_scanned_entries_every_cell_Load_unit,
    hls::stream<int> &s_scanned_entries_every_cell_PQ_lookup_computation,
    hls::stream<int> &s_last_valid_channel,

    hls::stream<int> &s_scanned_entries_per_query_Priority_queue);

void write_result(
    const int query_num,
    hls::stream<single_PQ_result> &output_stream, 
    ap_uint64_t* output);


void parse_HBM_meta_info(
    const int query_num,
    const int nlist,
    const bool OPQ_enable,
    const float* HBM_meta_info, 
    hls::stream<int>& s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid, 
    hls::stream<float>& s_product_quantizer,
    hls::stream<float>& s_OPQ_init,
    hls::stream<float>& s_query_vectors
    ) {

    // the storage format of the meta info:
    //   (1) HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid: size = 3 * nlist
    //   (2) HBM_product_quantizer: size = K * D
    //   (3) (optional) s_OPQ_init: D * D, if OPQ_enable = False, send nothing
    //   (4) HBM_query_vectors: size = query_num * D (send last, because the accelerator needs to send queries continuously)

    int start_addr_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid = 0;
    int size_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid = 3 * nlist;
    for (int addr = start_addr_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid; 
        addr < start_addr_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid + 
            size_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid
        ; addr++) {
#pragma HLS pipeline II=1
        float reg_float = HBM_meta_info[addr];
        int reg_int = *((int*) (&reg_float));
        s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid.write(reg_int);
    }

    int start_addr_HBM_product_quantizer = 
        start_addr_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid + 
        size_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid;
    int size_HBM_product_quantizer = K * D;
    for (int addr = start_addr_HBM_product_quantizer; 
        addr < start_addr_HBM_product_quantizer + size_HBM_product_quantizer; addr++) {
#pragma HLS pipeline II=1
        float reg_float = HBM_meta_info[addr];
        s_product_quantizer.write(reg_float);
    }

    int start_addr_OPQ_init;
    int size_OPQ_init;
    if (OPQ_enable) {
        start_addr_OPQ_init = start_addr_HBM_product_quantizer + size_HBM_product_quantizer;
        size_OPQ_init = D * D;
        for (int addr = start_addr_OPQ_init; 
            addr < start_addr_OPQ_init + size_OPQ_init; addr++) {
    #pragma HLS pipeline II=1
            float reg_float = HBM_meta_info[addr];
            s_OPQ_init.write(reg_float);
        }
    }

    int start_addr_HBM_query_vectors;
    if (OPQ_enable) {
        start_addr_HBM_query_vectors = start_addr_OPQ_init + size_OPQ_init;
    }
    else { 
        start_addr_HBM_query_vectors = start_addr_HBM_product_quantizer + size_HBM_product_quantizer;
    }
    int size_HBM_query_vectors = query_num * D;
    for (int addr = start_addr_HBM_query_vectors; 
        addr < start_addr_HBM_query_vectors + size_HBM_query_vectors; addr++) {
#pragma HLS pipeline II=1
        float reg_float = HBM_meta_info[addr];
        s_query_vectors.write(reg_float);
    }
}

void load_query_vectors(
    const int query_num,
    const float* DRAM_query_vector,
    hls::stream<float>& s_query_vectors) {

    // Data type: suppose each vector = 128 D, FPGA freq = 200 MHz
    //   then it takes 640 + 200 ns < 1 us to load a query vector, 
    //   much faster than computing distance and constructing LUT (> 10 us)

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
            float reg = DRAM_query_vector[query_id * D + d];
            s_query_vectors.write(reg);
        }
    }
}

void broadcast_preprocessed_query_vectors(
    const int query_num,
    hls::stream<float>& s_preprocessed_query_vectors,
    hls::stream<float> &s_preprocessed_query_vectors_distance_computation_PE,
    hls::stream<float>& s_preprocessed_query_vectors_lookup_PE) {

    // Data type: suppose each vector = 128 D, FPGA freq = 200 MHz
    //   then it takes 640 + 200 ns < 1 us to load a query vector, 
    //   much faster than computing distance and constructing LUT (> 10 us)

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int d = 0; d < D; d++) {
#pragma HLS pipeline II=1
            float reg = s_preprocessed_query_vectors.read();
            s_preprocessed_query_vectors_lookup_PE.write(reg);
            s_preprocessed_query_vectors_distance_computation_PE.write(reg);
        }
    }
}


void load_center_vectors(
    const int nlist,
    const float* table_HBM1, 
    hls::stream<float> &s_center_vectors_init_distance_computation_PE,
    hls::stream<float> &s_center_vectors_init_lookup_PE) {

    // e.g., CENTROIDS_PER_PARTITION = 256, CENTROID_PARTITIONS = 32
    //    first 256 elements -> stream 0
    //    second 256 elements -> stream 1, so on and so forth
    for (int i = 0; i < nlist * D; i++) {
        float reg = table_HBM1[i];
        s_center_vectors_init_distance_computation_PE.write(reg);
        s_center_vectors_init_lookup_PE.write(reg);
    }
}

void load_PQ_quantizer(
    const float* DRAM_PQ_quantizer,
    hls::stream<float> &s_PQ_quantizer_init) {

    // load PQ quantizer centers from HBM
    for (int i = 0; i < K * D; i++) {
#pragma HLS pipeline II=1
        float reg = DRAM_PQ_quantizer[i];
        s_PQ_quantizer_init.write(reg);
    }
}

void load_OPQ_matrix(
    const bool OPQ_enable,
    const float* HBM_OPQ_matrix, 
    hls::stream<float> &s_OPQ_init) {

    for (int i = 0; i < D * D; i++) {
#pragma HLS pipeline II=1
        if (OPQ_enable) {
            float reg = HBM_OPQ_matrix[i];
            s_OPQ_init.write(reg);
        }
    }
}

void split_cell_ID(
    const int query_num,
    const int nprobe,
    hls::stream<dist_cell_ID_t>& s_merge_output, 
    hls::stream<int>& s_searched_cell_id_lookup_PE, 
    hls::stream<int>& s_searched_cell_id_scan_controller) {

    for (int query_id = 0; query_id < query_num; query_id++) {
        
        dist_cell_ID_t tmp;
        int searched_cell_id_local[NPROBE];
#pragma HLS array_partition variable=searched_cell_id_local dim=1

        for (int i = 0; i < nprobe; i++) {
#pragma HLS pipeline II=1
            tmp = s_merge_output.read();
            searched_cell_id_local[i] = tmp.cell_ID;
            s_searched_cell_id_lookup_PE.write(searched_cell_id_local[i]);
            s_searched_cell_id_scan_controller.write(searched_cell_id_local[i]);
        }
    }
}

void lookup_center_vectors(
    const int query_num,
    const int nprobe,
    const float* HBM_vector_quantizer,
    hls::stream<int>& s_searched_cell_id_lookup_PE,
    hls::stream<float>& s_center_vectors_lookup_PE) {

    //  lookup center vectors given ID
    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int vec_id = s_searched_cell_id_lookup_PE.read();
            int start_addr = vec_id * D;

            for (int i = 0; i < D; i++) {
#pragma HLS pipeline II=1
                s_center_vectors_lookup_PE.write(HBM_vector_quantizer[start_addr + i]);
            }
        }
    }
}

void lookup_center_vectors(
    const int query_num,
    const int nlist,
    const int nprobe,
    hls::stream<float> &s_center_vectors_init_lookup_PE,
    hls::stream<int>& s_searched_cell_id_lookup_PE,
    hls::stream<float>& s_center_vectors_lookup_PE) {

    ap_uint<64> center_vector_local[NLIST * D / 2];
#pragma HLS resource variable=center_vector_local core=RAM_2P_URAM

    // init: load center vectors from DRAM 
    for (int i = 0; i < nlist * D / 2; i++) {
#pragma HLS pipeline II=1
        float val_A = s_center_vectors_init_lookup_PE.read();
        float val_B = s_center_vectors_init_lookup_PE.read();
        ap_uint<32> val_A_uint = *((ap_uint<32> *) (&val_A));
        ap_uint<32> val_B_uint = *((ap_uint<32> *) (&val_B));
        center_vector_local[i].range(31, 0) = val_A_uint;
        center_vector_local[i].range(63, 32) = val_B_uint;
    }

    //  lookup center vectors given ID
    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int vec_id = s_searched_cell_id_lookup_PE.read();
            int start_addr = vec_id * (D / 2);

            for (int i = 0; i < D / 2; i++) {
#pragma HLS pipeline II=2
                ap_uint<64> val_uint = center_vector_local[start_addr + i];
                ap_uint<32> val_A_uint = val_uint.range(31, 0);
                ap_uint<32> val_B_uint = val_uint.range(63, 32);
                float val_A = *((float*) (&val_A_uint));
                float val_B = *((float*) (&val_B_uint));
                s_center_vectors_lookup_PE.write(val_A);
                s_center_vectors_lookup_PE.write(val_B);
            }
        }
    }
}

void scan_controller(
    const int query_num,
    const int nlist, 
    const int nprobe,
    hls::stream<int> &s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid,
    hls::stream<int> &s_scanned_cell_id_Input, // from the cluster selection unit
    hls::stream<int> &s_start_addr_every_cell,
    hls::stream<int> &s_scanned_entries_every_cell_Load_unit,
    hls::stream<int> &s_scanned_entries_every_cell_PQ_lookup_computation,
    hls::stream<int> &s_last_valid_channel,

    hls::stream<int> &s_scanned_entries_per_query_Priority_queue) {
   
    // s_last_element_valid_PQ_lookup_computation -> last element of a channel can 
    //   be padded or not, 1 means valid (not padded), 0 means padded, should be discarded
    // last_valid_channel_LUT -> for each Voronoi cell, the last line in HBM may contain 
    //   padding, this is for storing where the last non-padded element id, ranges from 0~62
    //   e.g., all 63 elements store valid element, then last_valid_channel_LUT[x] = 62
    //   e.g., only the first channels's first element is valid, then last_valid_channel_LUT[x] = 0 
    int start_addr_LUT[NLIST];
    int scanned_entries_every_cell_LUT[NLIST];
    int last_valid_channel_LUT[NLIST];  
#pragma HLS resource variable=start_addr_LUT core=RAM_2P_URAM
#pragma HLS resource variable=scanned_entries_every_cell_LUT core=RAM_2P_URAM
#pragma HLS resource variable=last_valid_channel_LUT core=RAM_2P_URAM

    // init LUTs
    for (int i = 0; i < nlist; i++) {
#pragma HLS pipeline II=1
        start_addr_LUT[i] = s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid.read();
    }
    for (int i = 0; i < nlist; i++) {
#pragma HLS pipeline II=1
        scanned_entries_every_cell_LUT[i] = 
            s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid.read();
    }
    // ---- Fixed ----
    for (int i = 0; i < nlist; i++) {
#pragma HLS pipeline II=1
        last_valid_channel_LUT[i] = 
            s_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid.read();
    }

    // send control signals
    for (int query_id = 0; query_id < query_num; query_id++) {
        
        int accumulated_scanned_entries_per_query = 0;

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int cell_id = s_scanned_cell_id_Input.read();

            int start_addr = start_addr_LUT[cell_id];
            int scanned_entries_every_cell = scanned_entries_every_cell_LUT[cell_id];
            int last_valid_channel = last_valid_channel_LUT[cell_id];

            // each distance compute unit takes all 3 streams in from HBM
            int scanned_entries_every_cell_compute_unit = scanned_entries_every_cell * PQ_CODE_CHANNELS_PER_STREAM;

            s_start_addr_every_cell.write(start_addr);
            s_scanned_entries_every_cell_Load_unit.write(scanned_entries_every_cell);
            s_scanned_entries_every_cell_PQ_lookup_computation.write(scanned_entries_every_cell_compute_unit);
            s_last_valid_channel.write(last_valid_channel);

            accumulated_scanned_entries_per_query += scanned_entries_every_cell_compute_unit;
        }

        s_scanned_entries_per_query_Priority_queue.write(accumulated_scanned_entries_per_query);
    }
}

void write_result(
    const int query_num,
    hls::stream<single_PQ_result> &output_stream, 
    ap_uint64_t* output) {

    for (int i = 0; i < query_num * TOPK; i++) {
#pragma HLS pipeline II=1
        single_PQ_result raw_output = output_stream.read();
        ap_uint<64> reg;
        int vec_ID = raw_output.vec_ID;
        float dist = raw_output.dist;
        reg.range(31, 0) = *((ap_uint<32>*) (&vec_ID));
        reg.range(63, 32) = *((ap_uint<32>*) (&dist));
        output[i] = reg;
    }
}
