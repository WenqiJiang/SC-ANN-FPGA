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
    // ./host ./network.xclbin 5120000 8888 10.1.212.153 1
    
    // 11 arguments in total
    // arg 0 -> host exe
    // arg 1 -> bitstream file

    // FPGA network settings
    // arg 2 -> receive byte count
    // arg 3 -> FPGA port
    // arg 4 -> FPGA IP
    // arg 5 -> FPGA board num


    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << 
            " <XCLBIN File> [<#RxByte> <Port> <local_IP> <boardNum>] " << std::endl;
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
                      user_kernel = cl::Kernel(program, "network_K_1", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    
    wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");





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
    
//     std::cout << "Starting copy from Host to device..." << std::endl;
//     OCL_CHECK(
//         err, err = q.enqueueMigrateMemObjects({
//         buffer_HBM_embedding0,
//         buffer_HBM_embedding1,
//         buffer_HBM_embedding2,
//         buffer_HBM_embedding3,
//         buffer_HBM_embedding4,
//         buffer_HBM_embedding5,
//         buffer_HBM_embedding6,
//         buffer_HBM_embedding7,
//         buffer_HBM_embedding8,
//         buffer_HBM_embedding9,
//         buffer_HBM_embedding10,
//         buffer_HBM_embedding11,
//         buffer_HBM_embedding12,
//         buffer_HBM_embedding13,
//         buffer_HBM_embedding14,
//         buffer_HBM_embedding15,

//         buffer_HBM_centroid_vectors0,
//         buffer_HBM_centroid_vectors1,
//         buffer_HBM_centroid_vectors2,
//         buffer_HBM_centroid_vectors3,
//         buffer_HBM_centroid_vectors4,
//         buffer_HBM_centroid_vectors5,

//         // buffer_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid,
//         // buffer_HBM_query_vectors,
//         buffer_HBM_meta_info,
//         buffer_HBM_vector_quantizer
// //         buffer_HBM_product_quantizer,
// // #ifdef OPQ_ENABLE
// //         buffer_HBM_OPQ_matrix
// // #endif
//         }, 0/* 0 means from host*/));	
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
