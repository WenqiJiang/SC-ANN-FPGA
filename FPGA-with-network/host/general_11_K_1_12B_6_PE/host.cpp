/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
#include "xcl2.hpp"
#include <vector>
#include <chrono>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "host.hpp"

#define DATA_SIZE 62500000

//Set IP address of FPGA
#define IP_ADDR 0x0A01D498
#define BOARD_NUMBER 0
#define ARP 0x0A01D498

#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
/* for U280 specifically */
const int bank[40] = {
    /* 0 ~ 31 HBM */
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
    BANK_NAME(30), BANK_NAME(31), 
    /* 32, 33 DDR */ 
    BANK_NAME(32), BANK_NAME(33), 
    /* 34 ~ 39 PLRAM */ 
    BANK_NAME(34), BANK_NAME(35), BANK_NAME(36), BANK_NAME(37), 
    BANK_NAME(38), BANK_NAME(39)};


void wait_for_enter(const std::string &msg) {
    std::cout << msg << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

// boost::filesystem does not compile well, so implement this myself
std::string dir_concat(std::string dir1, std::string dir2) {
    if (dir1.back() != '/') {
        dir1 += '/';
    }
    return dir1 + dir2;
}

int main(int argc, char **argv) {

    // Example Usage:
    // ./host ./network.xclbin 5120000 8888 10.1.212.153 1 8192 17 1 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF8192,PQ16_16_banks /mnt/scratch/wenqi/saved_npy_data/gnd
    
    // 11 arguments in total
    // arg 0 -> host exe
    // arg 1 -> bitstream file

    // FPGA network settings
    // arg 2 -> receive byte count
    // arg 3 -> FPGA port
    // arg 4 -> FPGA IP
    // arg 5 -> FPGA board num

    // Index settings
    // arg 6 -> nlist
    // arg 7 -> nprobe
    // arg 8 -> OPQ_enable
    // arg 9 -> FPGA index/data directory
    // arg 10 -> ground truth directory

    if (argc != 11) {
        std::cout << "Usage: " << argv[0] << 
            " <XCLBIN File> [<#RxByte> <Port> <local_IP> <boardNum>] " <<
            "<nlist> <nprobe> <OPQ_enable> <data directory> <ground truth dir>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    // uint32_t basePort = 5001; 
    // uint32_t rxByteCnt = 320000;
    uint32_t rxByteCnt = strtol(argv[2], NULL, 10);
    uint32_t basePort = strtol(argv[3], NULL, 10);
    uint32_t connection = 16;

    printf("rxByteCnt:%d, listen Port:%d, connection:%d\n", rxByteCnt, basePort, connection);

    // local_IP
    // uint32_t local_IP = 0x0A01D498;
    std::string s = argv[4];
    std::string delimiter = ".";
    int ip [4];
    size_t pos = 0;
    std::string token;
    int i = 0;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        ip [i] = stoi(token);
        s.erase(0, pos + delimiter.length());
        i++;
    }
    ip[i] = stoi(s); 
    uint32_t local_IP = ip[3] | (ip[2] << 8) | (ip[1] << 16) | (ip[0] << 24);
    
    // uint32_t boardNum = 1;
    uint32_t boardNum = strtol(argv[5], NULL, 10);
    printf("local_IP:%x, boardNum:%d\n", local_IP, boardNum);


    int nlist = std::stoi(argv[6]);
    int nprobe = std::stoi(argv[7]);
    bool OPQ_enable = (bool) std::stoi(argv[8]);

    std::string data_dir_prefix = argv[9];
    std::string gnd_dir = argv[10];

    std::cout << "nlist: " << nlist << std::endl <<
        "nprobe: " << nprobe << std::endl <<
        "OPQ enable: " << OPQ_enable << std::endl <<
        "data directory" << data_dir_prefix << std::endl <<
        "ground truth directory" << gnd_dir << std::endl;

    // inferred parameters giving input parameters
    int centroids_per_partition_even = ceil(float(nlist) / float(PE_NUM_CENTER_DIST_COMP));
    int centroids_per_partition_last_PE = nlist - centroids_per_partition_even * (PE_NUM_CENTER_DIST_COMP - 1);

    int nprobe_stage4 = nprobe;
    int nprobe_per_table_construction_pe_larger = -1;
    int nprobe_per_table_construction_pe_smaller = -1;
    while (nprobe_per_table_construction_pe_smaller < 1) {
        nprobe_per_table_construction_pe_larger = ceil(float(nprobe_stage4) / float(PE_NUM_TABLE_CONSTRUCTION));
        nprobe_per_table_construction_pe_smaller = 
            nprobe_stage4 - PE_NUM_TABLE_CONSTRUCTION_LARGER * nprobe_per_table_construction_pe_larger;
        if (nprobe_per_table_construction_pe_smaller < 1) {
            nprobe_stage4++;
            std::cout << "Increasing nprobe_stage4 due to stage 4 hardware compatibility reason," <<
                "current nprobe_stage4: " << nprobe_stage4 << std::endl;
        }
    }
    if (PE_NUM_TABLE_CONSTRUCTION == 1) {
        nprobe_per_table_construction_pe_smaller = nprobe_per_table_construction_pe_larger;
    }

    std::cout << "Inferred parameters:" << std::endl <<
         "centroids_per_partition_even: " << centroids_per_partition_even << std::endl <<
         "centroids_per_partition_last_PE: " << centroids_per_partition_last_PE << std::endl <<
         "nprobe_per_table_construction_pe_larger: " << nprobe_per_table_construction_pe_larger << std::endl <<
         "nprobe_per_table_construction_pe_smaller: " << nprobe_per_table_construction_pe_smaller << std::endl;


    
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;

    cl::Kernel user_kernel;
    cl::Kernel network_kernel;

    auto size = DATA_SIZE;
    
    //Allocate Memory in Host Memory
    auto vector_size_bytes = sizeof(int) * size;
    std::vector<int, aligned_allocator<int>> network_ptr0(size);
    std::vector<int, aligned_allocator<int>> network_ptr1(size);


    //OPENCL HOST CODE AREA START
    //Create Program and Kernel
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context({device}, NULL, NULL, NULL, &err));
        OCL_CHECK(err,
                  q = cl::CommandQueue(
                      context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
                  cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err,
                      network_kernel = cl::Kernel(program, "network_krnl", &err));
            OCL_CHECK(err,
                      user_kernel = cl::Kernel(program, "general_11_K_1_12B_6_PE", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    
    // wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");





    // Set network kernel arguments
    OCL_CHECK(err, err = network_kernel.setArg(0, local_IP)); // Default IP address
    OCL_CHECK(err, err = network_kernel.setArg(1, boardNum)); // Board number
    OCL_CHECK(err, err = network_kernel.setArg(2, local_IP)); // ARP lookup

    OCL_CHECK(err,
              cl::Buffer buffer_r1(context,
                                   CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   vector_size_bytes,
                                   network_ptr0.data(),
                                   &err));
    OCL_CHECK(err,
            cl::Buffer buffer_r2(context,
                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                vector_size_bytes,
                                network_ptr1.data(),
                                &err));

    OCL_CHECK(err, err = network_kernel.setArg(3, buffer_r1));
    OCL_CHECK(err, err = network_kernel.setArg(4, buffer_r2));

    printf("enqueue network kernel...\n");
    OCL_CHECK(err, err = q.enqueueTask(network_kernel));
    OCL_CHECK(err, err = q.finish());
    

    OCL_CHECK(err, err = user_kernel.setArg(16, connection));
    OCL_CHECK(err, err = user_kernel.setArg(17, basePort));
    OCL_CHECK(err, err = user_kernel.setArg(18, rxByteCnt));




    ////////////////////    User Kernel Starts     //////////////////// 
    
    std::string HBM_embedding0_dir_suffix("HBM_bank_0_raw");
    std::string HBM_embedding0_dir = dir_concat(data_dir_prefix, HBM_embedding0_dir_suffix);
    std::ifstream HBM_embedding0_fstream(
        HBM_embedding0_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding1_dir_suffix("HBM_bank_1_raw");
    std::string HBM_embedding1_dir = dir_concat(data_dir_prefix, HBM_embedding1_dir_suffix);
    std::ifstream HBM_embedding1_fstream(
        HBM_embedding1_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding2_dir_suffix("HBM_bank_2_raw");
    std::string HBM_embedding2_dir = dir_concat(data_dir_prefix, HBM_embedding2_dir_suffix);
    std::ifstream HBM_embedding2_fstream(
        HBM_embedding2_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding3_dir_suffix("HBM_bank_3_raw");
    std::string HBM_embedding3_dir = dir_concat(data_dir_prefix, HBM_embedding3_dir_suffix);
    std::ifstream HBM_embedding3_fstream(
        HBM_embedding3_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding4_dir_suffix("HBM_bank_4_raw");
    std::string HBM_embedding4_dir = dir_concat(data_dir_prefix, HBM_embedding4_dir_suffix);
    std::ifstream HBM_embedding4_fstream(
        HBM_embedding4_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding5_dir_suffix("HBM_bank_5_raw");
    std::string HBM_embedding5_dir = dir_concat(data_dir_prefix, HBM_embedding5_dir_suffix);
    std::ifstream HBM_embedding5_fstream(
        HBM_embedding5_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding6_dir_suffix("HBM_bank_6_raw");
    std::string HBM_embedding6_dir = dir_concat(data_dir_prefix, HBM_embedding6_dir_suffix);
    std::ifstream HBM_embedding6_fstream(
        HBM_embedding6_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding7_dir_suffix("HBM_bank_7_raw");
    std::string HBM_embedding7_dir = dir_concat(data_dir_prefix, HBM_embedding7_dir_suffix);
    std::ifstream HBM_embedding7_fstream(
        HBM_embedding7_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding8_dir_suffix("HBM_bank_8_raw");
    std::string HBM_embedding8_dir = dir_concat(data_dir_prefix, HBM_embedding8_dir_suffix);
    std::ifstream HBM_embedding8_fstream(
        HBM_embedding8_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding9_dir_suffix("HBM_bank_9_raw");
    std::string HBM_embedding9_dir = dir_concat(data_dir_prefix, HBM_embedding9_dir_suffix);
    std::ifstream HBM_embedding9_fstream(
        HBM_embedding9_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding10_dir_suffix("HBM_bank_10_raw");
    std::string HBM_embedding10_dir = dir_concat(data_dir_prefix, HBM_embedding10_dir_suffix);
    std::ifstream HBM_embedding10_fstream(
        HBM_embedding10_dir, 
        std::ios::in | std::ios::binary);
    
    std::string HBM_embedding11_dir_suffix("HBM_bank_11_raw");
    std::string HBM_embedding11_dir = dir_concat(data_dir_prefix, HBM_embedding11_dir_suffix);
    std::ifstream HBM_embedding11_fstream(
        HBM_embedding11_dir, 
        std::ios::in | std::ios::binary);


    HBM_embedding0_fstream.seekg(0, HBM_embedding0_fstream.end);
    size_t HBM_embedding0_size =  HBM_embedding0_fstream.tellg();
    if (!HBM_embedding0_size) std::cout << "HBM_embedding0_size is 0!";
    HBM_embedding0_fstream.seekg(0, HBM_embedding0_fstream.beg);
    HBM_embedding1_fstream.seekg(0, HBM_embedding1_fstream.end);
    size_t HBM_embedding1_size =  HBM_embedding1_fstream.tellg();
    if (!HBM_embedding1_size) std::cout << "HBM_embedding1_size is 0!";
    HBM_embedding1_fstream.seekg(0, HBM_embedding1_fstream.beg);
    HBM_embedding2_fstream.seekg(0, HBM_embedding2_fstream.end);
    size_t HBM_embedding2_size =  HBM_embedding2_fstream.tellg();
    if (!HBM_embedding2_size) std::cout << "HBM_embedding2_size is 0!";
    HBM_embedding2_fstream.seekg(0, HBM_embedding2_fstream.beg);
    HBM_embedding3_fstream.seekg(0, HBM_embedding3_fstream.end);
    size_t HBM_embedding3_size =  HBM_embedding3_fstream.tellg();
    if (!HBM_embedding3_size) std::cout << "HBM_embedding3_size is 0!";
    HBM_embedding3_fstream.seekg(0, HBM_embedding3_fstream.beg);
    HBM_embedding4_fstream.seekg(0, HBM_embedding4_fstream.end);
    size_t HBM_embedding4_size =  HBM_embedding4_fstream.tellg();
    if (!HBM_embedding4_size) std::cout << "HBM_embedding4_size is 0!";
    HBM_embedding4_fstream.seekg(0, HBM_embedding4_fstream.beg);
    HBM_embedding5_fstream.seekg(0, HBM_embedding5_fstream.end);
    size_t HBM_embedding5_size =  HBM_embedding5_fstream.tellg();
    if (!HBM_embedding5_size) std::cout << "HBM_embedding5_size is 0!";
    HBM_embedding5_fstream.seekg(0, HBM_embedding5_fstream.beg);
    HBM_embedding6_fstream.seekg(0, HBM_embedding6_fstream.end);
    size_t HBM_embedding6_size =  HBM_embedding6_fstream.tellg();
    if (!HBM_embedding6_size) std::cout << "HBM_embedding6_size is 0!";
    HBM_embedding6_fstream.seekg(0, HBM_embedding6_fstream.beg);
    HBM_embedding7_fstream.seekg(0, HBM_embedding7_fstream.end);
    size_t HBM_embedding7_size =  HBM_embedding7_fstream.tellg();
    if (!HBM_embedding7_size) std::cout << "HBM_embedding7_size is 0!";
    HBM_embedding7_fstream.seekg(0, HBM_embedding7_fstream.beg);
    HBM_embedding8_fstream.seekg(0, HBM_embedding8_fstream.end);
    size_t HBM_embedding8_size =  HBM_embedding8_fstream.tellg();
    if (!HBM_embedding8_size) std::cout << "HBM_embedding8_size is 0!";
    HBM_embedding8_fstream.seekg(0, HBM_embedding8_fstream.beg);
    HBM_embedding9_fstream.seekg(0, HBM_embedding9_fstream.end);
    size_t HBM_embedding9_size =  HBM_embedding9_fstream.tellg();
    if (!HBM_embedding9_size) std::cout << "HBM_embedding9_size is 0!";
    HBM_embedding9_fstream.seekg(0, HBM_embedding9_fstream.beg);
    HBM_embedding10_fstream.seekg(0, HBM_embedding10_fstream.end);
    size_t HBM_embedding10_size =  HBM_embedding10_fstream.tellg();
    if (!HBM_embedding10_size) std::cout << "HBM_embedding10_size is 0!";
    HBM_embedding10_fstream.seekg(0, HBM_embedding10_fstream.beg);
    HBM_embedding11_fstream.seekg(0, HBM_embedding11_fstream.end);
    size_t HBM_embedding11_size =  HBM_embedding11_fstream.tellg();
    if (!HBM_embedding11_size) std::cout << "HBM_embedding11_size is 0!";
    HBM_embedding11_fstream.seekg(0, HBM_embedding11_fstream.beg);
    

    size_t HBM_embedding0_len = (int) (HBM_embedding0_size / sizeof(ap_uint512_t));
    size_t HBM_embedding1_len = (int) (HBM_embedding1_size / sizeof(ap_uint512_t));
    size_t HBM_embedding2_len = (int) (HBM_embedding2_size / sizeof(ap_uint512_t));
    size_t HBM_embedding3_len = (int) (HBM_embedding3_size / sizeof(ap_uint512_t));
    size_t HBM_embedding4_len = (int) (HBM_embedding4_size / sizeof(ap_uint512_t));
    size_t HBM_embedding5_len = (int) (HBM_embedding5_size / sizeof(ap_uint512_t));
    size_t HBM_embedding6_len = (int) (HBM_embedding6_size / sizeof(ap_uint512_t));
    size_t HBM_embedding7_len = (int) (HBM_embedding7_size / sizeof(ap_uint512_t));
    size_t HBM_embedding8_len = (int) (HBM_embedding8_size / sizeof(ap_uint512_t));
    size_t HBM_embedding9_len = (int) (HBM_embedding9_size / sizeof(ap_uint512_t));
    size_t HBM_embedding10_len = (int) (HBM_embedding10_size / sizeof(ap_uint512_t));
    size_t HBM_embedding11_len = (int) (HBM_embedding11_size / sizeof(ap_uint512_t));

    size_t HBM_centroid_vectors0_len = 2 * centroids_per_partition_even * D * sizeof(float) / sizeof(ap_uint512_t);
    size_t HBM_centroid_vectors1_len = 2 * centroids_per_partition_even * D * sizeof(float) / sizeof(ap_uint512_t);
    size_t HBM_centroid_vectors2_len = (centroids_per_partition_even + centroids_per_partition_last_PE) * D * sizeof(float) / sizeof(ap_uint512_t);


    int query_num = 10000;
    size_t HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len = nlist * 3;
    size_t HBM_vector_quantizer_len = nlist * 128;
    size_t HBM_product_quantizer_len = 16 * 256 * (128 / 16);
    size_t HBM_OPQ_matrix_len = 128 * 128;
    size_t HBM_out_len = TOPK * query_num; 

    // the storage format of the meta info:
    //   (1) HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid: size = 3 * nlist
    //   (2) HBM_product_quantizer: size = K * D
    //   (3) (optional) s_OPQ_init: D * D, if OPQ_enable = False, send nothing
    //   (4) HBM_query_vectors: size = query_num * D (send last, because the accelerator needs to send queries continuously)
    size_t HBM_meta_info_len;
    if (OPQ_enable) {
        HBM_meta_info_len = HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len + 
            HBM_product_quantizer_len + HBM_OPQ_matrix_len;
    } else {
        HBM_meta_info_len = HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len + 
            HBM_product_quantizer_len;
    }

    // the raw ground truth size is the same for idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs
    size_t raw_gt_vec_ID_len = 10000 * 1001; 
    // recall counts the very first nearest neighbor only
    size_t gt_vec_ID_len = 10000;

    size_t HBM_centroid_vectors0_size =  HBM_centroid_vectors0_len * sizeof(ap_uint512_t);
    size_t HBM_centroid_vectors1_size =  HBM_centroid_vectors1_len * sizeof(ap_uint512_t);
    size_t HBM_centroid_vectors2_size =  HBM_centroid_vectors2_len * sizeof(ap_uint512_t);

    size_t HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_size = 
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len * sizeof(int);
    size_t HBM_vector_quantizer_size = HBM_vector_quantizer_len * sizeof(float);
    size_t HBM_product_quantizer_size = HBM_product_quantizer_len * sizeof(float);
    size_t HBM_OPQ_matrix_size = HBM_OPQ_matrix_len * sizeof(float);
    size_t HBM_meta_info_size = HBM_meta_info_len * sizeof(float);

    size_t raw_gt_vec_ID_size = raw_gt_vec_ID_len * sizeof(int);
    size_t gt_vec_ID_size = gt_vec_ID_len * sizeof(int);

//////////////////////////////   TEMPLATE END  //////////////////////////////

    unsigned fileBufSize;

    // allocate aligned 2D vectors
//////////////////////////////   TEMPLATE START  //////////////////////////////
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding0(HBM_embedding0_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding1(HBM_embedding1_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding2(HBM_embedding2_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding3(HBM_embedding3_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding4(HBM_embedding4_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding5(HBM_embedding5_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding6(HBM_embedding6_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding7(HBM_embedding7_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding8(HBM_embedding8_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding9(HBM_embedding9_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding10(HBM_embedding10_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding11(HBM_embedding11_len, 0);

    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_centroid_vectors0(HBM_centroid_vectors0_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_centroid_vectors1(HBM_centroid_vectors1_len, 0);
    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_centroid_vectors2(HBM_centroid_vectors2_len, 0);

    std::vector<int, aligned_allocator<int>> HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid(
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len, 0);
    std::vector<float, aligned_allocator<float>> HBM_vector_quantizer(HBM_vector_quantizer_len, 0);
    std::vector<float, aligned_allocator<float>> HBM_product_quantizer(HBM_product_quantizer_len, 0);
    std::vector<float, aligned_allocator<float>> HBM_OPQ_matrix(HBM_OPQ_matrix_len, 0);
    std::vector<float, aligned_allocator<float>> HBM_meta_info(HBM_meta_info_len, 0);
    
    std::vector<int, aligned_allocator<int>> raw_gt_vec_ID(raw_gt_vec_ID_len, 0);
    std::vector<int, aligned_allocator<int>> gt_vec_ID(gt_vec_ID_len, 0);

//////////////////////////////   TEMPLATE END  //////////////////////////////

    char* HBM_embedding0_char = (char*) malloc(HBM_embedding0_size);
    char* HBM_embedding1_char = (char*) malloc(HBM_embedding1_size);
    char* HBM_embedding2_char = (char*) malloc(HBM_embedding2_size);
    char* HBM_embedding3_char = (char*) malloc(HBM_embedding3_size);
    char* HBM_embedding4_char = (char*) malloc(HBM_embedding4_size);
    char* HBM_embedding5_char = (char*) malloc(HBM_embedding5_size);
    char* HBM_embedding6_char = (char*) malloc(HBM_embedding6_size);
    char* HBM_embedding7_char = (char*) malloc(HBM_embedding7_size);
    char* HBM_embedding8_char = (char*) malloc(HBM_embedding8_size);
    char* HBM_embedding9_char = (char*) malloc(HBM_embedding9_size);
    char* HBM_embedding10_char = (char*) malloc(HBM_embedding10_size);
    char* HBM_embedding11_char = (char*) malloc(HBM_embedding11_size);


    char* HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_char = 
        (char*) malloc(HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_size);
    char* HBM_vector_quantizer_char = (char*) malloc(HBM_vector_quantizer_size);
    char* HBM_product_quantizer_char = (char*) malloc(HBM_product_quantizer_size);
    char* HBM_OPQ_matrix_char = (char*) malloc(HBM_OPQ_matrix_size);

    char* raw_gt_vec_ID_char = (char*) malloc(raw_gt_vec_ID_size);

  
    std::string HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir_suffix = 
        "HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_3_by_" + std::to_string(nlist) + "_raw";
    std::string HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir = 
        dir_concat(data_dir_prefix, HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir_suffix);
    std::ifstream HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream(
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir, 
        std::ios::in | std::ios::binary);


    std::string HBM_vector_quantizer_dir_suffix = "vector_quantizer_float32_" + std::to_string(nlist) + "_128_raw";
    std::string HBM_vector_quantizer_dir = dir_concat(data_dir_prefix, HBM_vector_quantizer_dir_suffix);
    std::ifstream HBM_vector_quantizer_fstream(
        HBM_vector_quantizer_dir, 
        std::ios::in | std::ios::binary);

    
    std::string HBM_product_quantizer_suffix_dir = "product_quantizer_float32_16_256_8_raw";
    std::string HBM_product_quantizer_dir = dir_concat(data_dir_prefix, HBM_product_quantizer_suffix_dir);
    std::ifstream HBM_product_quantizer_fstream(
        HBM_product_quantizer_dir,
        std::ios::in | std::ios::binary);


    if (OPQ_enable) {
        std::string HBM_OPQ_matrix_suffix_dir = "OPQ_matrix_float32_128_128_raw";
        std::string HBM_OPQ_matrix_dir = dir_concat(data_dir_prefix, HBM_OPQ_matrix_suffix_dir);
        std::ifstream HBM_OPQ_matrix_fstream(
            HBM_OPQ_matrix_dir,
            std::ios::in | std::ios::binary);
        HBM_OPQ_matrix_fstream.read(HBM_OPQ_matrix_char, HBM_OPQ_matrix_size);
        if (!HBM_OPQ_matrix_fstream) {
            std::cout << "error: only " << HBM_OPQ_matrix_fstream.gcount() << " could be read";
            exit(1);
        }
        memcpy(&HBM_OPQ_matrix[0], HBM_OPQ_matrix_char, HBM_OPQ_matrix_size);
    }


    std::string raw_gt_vec_ID_suffix_dir = "idx_100M.ivecs";
    std::string raw_gt_vec_ID_dir = dir_concat(gnd_dir, raw_gt_vec_ID_suffix_dir);
    std::ifstream raw_gt_vec_ID_fstream(
        raw_gt_vec_ID_dir,
        std::ios::in | std::ios::binary);
    if (!raw_gt_vec_ID_fstream) {
        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
        exit(1);
}

        
    HBM_embedding0_fstream.read(HBM_embedding0_char, HBM_embedding0_size);
    if (!HBM_embedding0_fstream) {
            std::cout << "error: only " << HBM_embedding0_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding1_fstream.read(HBM_embedding1_char, HBM_embedding1_size);
    if (!HBM_embedding1_fstream) {
            std::cout << "error: only " << HBM_embedding1_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding2_fstream.read(HBM_embedding2_char, HBM_embedding2_size);
    if (!HBM_embedding2_fstream) {
            std::cout << "error: only " << HBM_embedding2_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding3_fstream.read(HBM_embedding3_char, HBM_embedding3_size);
    if (!HBM_embedding3_fstream) {
            std::cout << "error: only " << HBM_embedding3_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding4_fstream.read(HBM_embedding4_char, HBM_embedding4_size);
    if (!HBM_embedding4_fstream) {
            std::cout << "error: only " << HBM_embedding4_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding5_fstream.read(HBM_embedding5_char, HBM_embedding5_size);
    if (!HBM_embedding5_fstream) {
            std::cout << "error: only " << HBM_embedding5_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding6_fstream.read(HBM_embedding6_char, HBM_embedding6_size);
    if (!HBM_embedding6_fstream) {
            std::cout << "error: only " << HBM_embedding6_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding7_fstream.read(HBM_embedding7_char, HBM_embedding7_size);
    if (!HBM_embedding7_fstream) {
            std::cout << "error: only " << HBM_embedding7_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding8_fstream.read(HBM_embedding8_char, HBM_embedding8_size);
    if (!HBM_embedding8_fstream) {
            std::cout << "error: only " << HBM_embedding8_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding9_fstream.read(HBM_embedding9_char, HBM_embedding9_size);
    if (!HBM_embedding9_fstream) {
            std::cout << "error: only " << HBM_embedding9_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding10_fstream.read(HBM_embedding10_char, HBM_embedding10_size);
    if (!HBM_embedding10_fstream) {
            std::cout << "error: only " << HBM_embedding10_fstream.gcount() << " could be read";
        exit(1);
     }
    HBM_embedding11_fstream.read(HBM_embedding11_char, HBM_embedding11_size);
    if (!HBM_embedding11_fstream) {
            std::cout << "error: only " << HBM_embedding11_fstream.gcount() << " could be read";
        exit(1);
     }


    HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream.read(
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_char,
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_size);
    if (!HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream) {
        std::cout << "error: only " << HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream.gcount() << " could be read";
        exit(1);
    }
    HBM_vector_quantizer_fstream.read(HBM_vector_quantizer_char, HBM_vector_quantizer_size);
    if (!HBM_vector_quantizer_fstream) {
        std::cout << "error: only " << HBM_vector_quantizer_fstream.gcount() << " could be read";
        exit(1);
    }
    HBM_product_quantizer_fstream.read(HBM_product_quantizer_char, HBM_product_quantizer_size);
    if (!HBM_product_quantizer_fstream) {
        std::cout << "error: only " << HBM_product_quantizer_fstream.gcount() << " could be read";
        exit(1);
    }

    raw_gt_vec_ID_fstream.read(raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    if (!raw_gt_vec_ID_fstream) {
        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
        exit(1);
    }

    // std::cout << "HBM_vector_quantizer_fstream read bytes: " << HBM_vector_quantizer_fstream.gcount() << std::endl;
    // std::cout << "HBM_product_quantizer_fstream read bytes: " << HBM_product_quantizer_fstream.gcount() << std::endl;
 
    memcpy(&HBM_embedding0[0], HBM_embedding0_char, HBM_embedding0_size);
    memcpy(&HBM_embedding1[0], HBM_embedding1_char, HBM_embedding1_size);
    memcpy(&HBM_embedding2[0], HBM_embedding2_char, HBM_embedding2_size);
    memcpy(&HBM_embedding3[0], HBM_embedding3_char, HBM_embedding3_size);
    memcpy(&HBM_embedding4[0], HBM_embedding4_char, HBM_embedding4_size);
    memcpy(&HBM_embedding5[0], HBM_embedding5_char, HBM_embedding5_size);
    memcpy(&HBM_embedding6[0], HBM_embedding6_char, HBM_embedding6_size);
    memcpy(&HBM_embedding7[0], HBM_embedding7_char, HBM_embedding7_size);
    memcpy(&HBM_embedding8[0], HBM_embedding8_char, HBM_embedding8_size);
    memcpy(&HBM_embedding9[0], HBM_embedding9_char, HBM_embedding9_size);
    memcpy(&HBM_embedding10[0], HBM_embedding10_char, HBM_embedding10_size);
    memcpy(&HBM_embedding11[0], HBM_embedding11_char, HBM_embedding11_size);


    memcpy(&HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid[0], 
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_char, 
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_size);
    memcpy(&HBM_vector_quantizer[0], HBM_vector_quantizer_char, HBM_vector_quantizer_size);
    memcpy(&HBM_product_quantizer[0], HBM_product_quantizer_char, HBM_product_quantizer_size);

    int start_addr_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid = 0;
    int size_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid = 3 * nlist;
    memcpy(&HBM_meta_info[start_addr_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid], 
        &HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid[0],
        size_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid * sizeof(float));

    int start_addr_HBM_product_quantizer = 
        start_addr_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid + 
        size_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid;
    int size_HBM_product_quantizer = K * D;
    memcpy(&HBM_meta_info[start_addr_HBM_product_quantizer], 
        &HBM_product_quantizer[0],
        size_HBM_product_quantizer * sizeof(float));

    int start_addr_OPQ_init;
    int size_OPQ_init;
    if (OPQ_enable) {
        start_addr_OPQ_init = start_addr_HBM_product_quantizer + size_HBM_product_quantizer;
        size_OPQ_init = D * D;
        memcpy(&HBM_meta_info[start_addr_OPQ_init], 
            &HBM_OPQ_matrix[0],
            size_OPQ_init * sizeof(float));
    }


    int HBM_centroid_vectors_stage2_start_addr_0 = 2 * 0 * centroids_per_partition_even * D * sizeof(float);
    memcpy(&HBM_centroid_vectors0[0], HBM_vector_quantizer_char + HBM_centroid_vectors_stage2_start_addr_0, HBM_centroid_vectors0_size);

    int HBM_centroid_vectors_stage2_start_addr_1 = 2 * 1 * centroids_per_partition_even * D * sizeof(float);
    memcpy(&HBM_centroid_vectors1[0], HBM_vector_quantizer_char + HBM_centroid_vectors_stage2_start_addr_1, HBM_centroid_vectors1_size);

    int HBM_centroid_vectors_stage2_start_addr_2 = 2 * 2 * centroids_per_partition_even * D * sizeof(float);
    memcpy(&HBM_centroid_vectors2[0], HBM_vector_quantizer_char + HBM_centroid_vectors_stage2_start_addr_2, HBM_centroid_vectors2_size);

    memcpy(&raw_gt_vec_ID[0], raw_gt_vec_ID_char, raw_gt_vec_ID_size);

    free(HBM_embedding0_char);
    free(HBM_embedding1_char);
    free(HBM_embedding2_char);
    free(HBM_embedding3_char);
    free(HBM_embedding4_char);
    free(HBM_embedding5_char);
    free(HBM_embedding6_char);
    free(HBM_embedding7_char);
    free(HBM_embedding8_char);
    free(HBM_embedding9_char);
    free(HBM_embedding10_char);
    free(HBM_embedding11_char);


    free(HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_char);
    free(HBM_vector_quantizer_char);
    free(HBM_product_quantizer_char);
    free(HBM_OPQ_matrix_char);

    free(raw_gt_vec_ID_char);
    // free(sw_result_vec_ID_char);
    // free(sw_result_dist_char);

    // copy contents from raw ground truth to needed ones
    // Format of ground truth (for 10000 query vectors):
    //   1000(topK), [1000 ids]
    //   1000(topK), [1000 ids]
    //        ...     ...
    //   1000(topK), [1000 ids]
    // 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    for (int i = 0; i < 10000; i++) {
        gt_vec_ID[i] = raw_gt_vec_ID[i * 1001 + 1];
    }

// OPENCL HOST CODE AREA START
// ================================================================
// Step 2: Setup Buffers and run Kernels
// ================================================================
//   o) Allocate Memory to store the results 
//   o) Create Buffers in Global Memory to store data
// ================================================================

// ------------------------------------------------------------------
// Step 2: Create Buffers in Global Memory to store data
//             o) buffer_in1 - stores source_in1
//             o) buffer_in2 - stores source_in2
//             o) buffer_ouput - stores Results
// ------------------------------------------------------------------	

// .......................................................
// Allocate Global Memory for source_in1
// .......................................................	
//////////////////////////////   TEMPLATE START  //////////////////////////////

    std::cout << "Start to allocate device memory..." << std::endl;
    cl_mem_ext_ptr_t 
        HBM_embedding0Ext,
        HBM_embedding1Ext,
        HBM_embedding2Ext,
        HBM_embedding3Ext,
        HBM_embedding4Ext,
        HBM_embedding5Ext,
        HBM_embedding6Ext,
        HBM_embedding7Ext,
        HBM_embedding8Ext,
        HBM_embedding9Ext,
        HBM_embedding10Ext,
        HBM_embedding11Ext,

        HBM_centroid_vectors0Ext,
        HBM_centroid_vectors1Ext,
        HBM_centroid_vectors2Ext,

        HBM_meta_infoExt,
        HBM_vector_quantizerExt;
//////////////////////////////   TEMPLATE END  //////////////////////////////

//////////////////////////////   TEMPLATE START  //////////////////////////////
    HBM_embedding0Ext.obj = HBM_embedding0.data();
    HBM_embedding0Ext.param = 0;
    HBM_embedding0Ext.flags = bank[0];
    HBM_embedding1Ext.obj = HBM_embedding1.data();
    HBM_embedding1Ext.param = 0;
    HBM_embedding1Ext.flags = bank[1];
    HBM_embedding2Ext.obj = HBM_embedding2.data();
    HBM_embedding2Ext.param = 0;
    HBM_embedding2Ext.flags = bank[2];
    HBM_embedding3Ext.obj = HBM_embedding3.data();
    HBM_embedding3Ext.param = 0;
    HBM_embedding3Ext.flags = bank[4];
    HBM_embedding4Ext.obj = HBM_embedding4.data();
    HBM_embedding4Ext.param = 0;
    HBM_embedding4Ext.flags = bank[5];
    HBM_embedding5Ext.obj = HBM_embedding5.data();
    HBM_embedding5Ext.param = 0;
    HBM_embedding5Ext.flags = bank[6];
    HBM_embedding6Ext.obj = HBM_embedding6.data();
    HBM_embedding6Ext.param = 0;
    HBM_embedding6Ext.flags = bank[9];
    HBM_embedding7Ext.obj = HBM_embedding7.data();
    HBM_embedding7Ext.param = 0;
    HBM_embedding7Ext.flags = bank[10];
    HBM_embedding8Ext.obj = HBM_embedding8.data();
    HBM_embedding8Ext.param = 0;
    HBM_embedding8Ext.flags = bank[11];
    HBM_embedding9Ext.obj = HBM_embedding9.data();
    HBM_embedding9Ext.param = 0;
    HBM_embedding9Ext.flags = bank[14];
    HBM_embedding10Ext.obj = HBM_embedding10.data();
    HBM_embedding10Ext.param = 0;
    HBM_embedding10Ext.flags = bank[15];
    HBM_embedding11Ext.obj = HBM_embedding11.data();
    HBM_embedding11Ext.param = 0;
    HBM_embedding11Ext.flags = bank[16];
    
    HBM_centroid_vectors0Ext.obj = HBM_centroid_vectors0.data();
    HBM_centroid_vectors0Ext.param = 0;
    HBM_centroid_vectors0Ext.flags = bank[18];
    HBM_centroid_vectors1Ext.obj = HBM_centroid_vectors1.data();
    HBM_centroid_vectors1Ext.param = 0;
    HBM_centroid_vectors1Ext.flags = bank[20];
    HBM_centroid_vectors2Ext.obj = HBM_centroid_vectors2.data();
    HBM_centroid_vectors2Ext.param = 0;
    HBM_centroid_vectors2Ext.flags = bank[22];


    HBM_meta_infoExt.obj = HBM_meta_info.data();
    HBM_meta_infoExt.param = 0;
    HBM_meta_infoExt.flags = bank[25];

    HBM_vector_quantizerExt.obj = HBM_vector_quantizer.data();
    HBM_vector_quantizerExt.param = 0;
    HBM_vector_quantizerExt.flags = bank[26];


//////////////////////////////   TEMPLATE START  //////////////////////////////
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding0_size, &HBM_embedding0Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding1_size, &HBM_embedding1Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding2_size, &HBM_embedding2Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding3_size, &HBM_embedding3Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding4_size, &HBM_embedding4Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding5_size, &HBM_embedding5Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding6(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding6_size, &HBM_embedding6Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding7(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding7_size, &HBM_embedding7Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding8(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding8_size, &HBM_embedding8Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding9(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding9_size, &HBM_embedding9Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding10(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding10_size, &HBM_embedding10Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding11(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_embedding11_size, &HBM_embedding11Ext, &err));


    OCL_CHECK(err, cl::Buffer buffer_HBM_centroid_vectors0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_centroid_vectors0_size, &HBM_centroid_vectors0Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_centroid_vectors1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_centroid_vectors1_size, &HBM_centroid_vectors1Ext, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_centroid_vectors2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_centroid_vectors2_size, &HBM_centroid_vectors2Ext, &err));

    OCL_CHECK(err, cl::Buffer buffer_HBM_vector_quantizer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_vector_quantizer_size, &HBM_vector_quantizerExt, &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_meta_info(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            HBM_meta_info_size, &HBM_meta_infoExt, &err));
    
// .......................................................
// Allocate Global Memory for sourcce_hw_results
// .......................................................
    // OCL_CHECK(err, cl::Buffer buffer_output(
    //     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, 
    //     HBM_out_size, &HBM_outExt, &err));
    

// ============================================================================
// Step 2: Set Kernel Arguments and Run the Application
//         o) Set Kernel Arguments
//         o) Copy Input Data from Host to Global Memory on the device
//         o) Submit Kernels for Execution
//         o) Copy Results from Global Memory, device to Host
// ============================================================================	
    
//////////////////////////////   TEMPLATE START  //////////////////////////////
    int arg_counter = 19;
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding0));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding1));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding2));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding3));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding4));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding5));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding6));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding7));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding8));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding9));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding10));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_embedding11));

    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_centroid_vectors0));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_centroid_vectors1));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_centroid_vectors2));
    
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_meta_info));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_HBM_vector_quantizer));
    
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, nlist));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, nprobe));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, OPQ_enable));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, centroids_per_partition_even));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, centroids_per_partition_last_PE));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, nprobe_per_table_construction_pe_larger));
    OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, nprobe_per_table_construction_pe_smaller));

    // OCL_CHECK(err, err = user_kernel.setArg(arg_counter++, buffer_output));

//////////////////////////////   TEMPLATE END  //////////////////////////////
// ------------------------------------------------------
// Step 2: Copy Input data from Host to Global Memory on the device
// ------------------------------------------------------
//////////////////////////////   TEMPLATE START  //////////////////////////////
    std::cout << "Starting copy from Host to device..." << std::endl;
    OCL_CHECK(
        err, err = q.enqueueMigrateMemObjects({
        buffer_HBM_embedding0,
        buffer_HBM_embedding1,
        buffer_HBM_embedding2,
        buffer_HBM_embedding3,
        buffer_HBM_embedding4,
        buffer_HBM_embedding5,
        buffer_HBM_embedding6,
        buffer_HBM_embedding7,
        buffer_HBM_embedding8,
        buffer_HBM_embedding9,
        buffer_HBM_embedding10,
        buffer_HBM_embedding11,

        buffer_HBM_centroid_vectors0,
        buffer_HBM_centroid_vectors1,
        buffer_HBM_centroid_vectors2,

        buffer_HBM_meta_info,
        buffer_HBM_vector_quantizer
        }, 0/* 0 means from host*/));	
    std::cout << "Host to device finished..." << std::endl;

    //Launch the Kernel
    auto start = std::chrono::high_resolution_clock::now();
    printf("enqueue user kernel...\n");
    OCL_CHECK(err, err = q.enqueueTask(user_kernel));
    OCL_CHECK(err, err = q.finish());
    auto end = std::chrono::high_resolution_clock::now();
    double durationUs = 0.0;
    durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
    printf("durationUs:%f\n",durationUs);
    //OPENCL HOST CODE AREA END    

    std::cout << "EXIT recorded" << std::endl;
}
