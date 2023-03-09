/*
Usage: 
    ./host <XCLBIN File> <data directory> <ground truth dir>
Example
    ./host vadd.xclbin /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF8192,PQ16_16_banks /mnt/scratch/wenqi/saved_npy_data/gnd
*/

/*
Variable to be replaced (<--variable_name-->):
    multiple lines (depends on HBM channel num):
        HBM_embedding_len    # number of 512-bit chunk in each bank
        HBM_embedding_size
        HBM_embedding_allocate
        HBM_embedding_char
        HBM_embedding_fstream
        HBM_embedding_memcpy
        HBM_embedding_char_free
        buffer_HBM_embedding

    multiple lines (depends on stage 2 PE num / on or off-chip):
        HBM_centroid_vectors_stage2_len
        HBM_centroid_vectors_stage2_size
        HBM_centroid_vectors_stage2_allocate
        HBM_centroid_vectors_stage2_memcpy
        buffer_HBM_centroid_vectors
        buffer_HBM_centroid_vectors_stage2_set_krnl_arg
        buffer_HBM_centroid_vectors_stage2_enqueueMigrateMemObjects

    single line:
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream
        HBM_query_vector_fstream
        HBM_vector_quantizer_fstream
        HBM_product_quantizer_fstream
        HBM_OPQ_matrix_fstream
        sw_result_vec_ID_fstream
        sw_result_dist_fstream

    basic constants:
        QUERY_NUM
        NLIST
        D
        M
*/

#include "host.hpp"
#include "constants.hpp"

#include <algorithm>
#include <vector>
#include <unistd.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <chrono>

#include <stdint.h>
#include <math.h>  



// boost::filesystem does not compile well, so implement this myself
std::string dir_concat(std::string dir1, std::string dir2) {
    if (dir1.back() != '/') {
        dir1 += '/';
    }
    return dir1 + dir2;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> <data directory> <ground truth dir>" << std::endl;
		return EXIT_FAILURE;
	}

    std::string binaryFile = argv[1];

    int nlist = NLIST;
    int nprobe = NPROBE;
    bool OPQ_enable = <--OPQ_ENABLE-->;

    std::string data_dir_prefix = argv[2];
    std::string gnd_dir = argv[3];

    std::cout << "nlist: " << nlist << std::endl <<
        "nprobe: " << nprobe << std::endl <<
        "OPQ enable: " << OPQ_enable << std::endl <<
        "data directory" << data_dir_prefix << std::endl <<
        "ground truth directory" << gnd_dir << std::endl;

    // inferred parameters giving input parameters
    int centroids_per_partition_even = CENTROIDS_PER_PARTITION_EVEN;
    int centroids_per_partition_last_PE = CENTROIDS_PER_PARTITION_LAST_PE;

    int nprobe_stage4 = nprobe;
    int nprobe_per_table_construction_pe_larger = ceil(float(nprobe) / float(PE_NUM_TABLE_CONSTRUCTION));
    int nprobe_per_table_construction_pe_smaller = -1;
    if (PE_NUM_TABLE_CONSTRUCTION == 1) {
        nprobe_per_table_construction_pe_smaller = nprobe_per_table_construction_pe_larger;
    }
    else {
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
    }

    std::cout << "Inferred parameters:" << std::endl <<
         "centroids_per_partition_even: " << centroids_per_partition_even << std::endl <<
         "centroids_per_partition_last_PE: " << centroids_per_partition_last_PE << std::endl <<
         "nprobe_per_table_construction_pe_larger: " << nprobe_per_table_construction_pe_larger << std::endl <<
         "nprobe_per_table_construction_pe_smaller: " << nprobe_per_table_construction_pe_smaller << std::endl;

//////////////////////////////   TEMPLATE START  //////////////////////////////
    
<--HBM_embedding_fstream-->
<--HBM_embedding_size-->

<--HBM_embedding_len-->
<--HBM_centroid_vectors_stage2_len-->

    int query_num = <--QUERY_NUM-->;
    size_t HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len = nlist * 3;
    size_t HBM_query_vector_len = query_num * <--D--> < <--QUERY_NUM--> * <--D-->? query_num * <--D-->: <--QUERY_NUM--> * <--D-->;
    size_t HBM_vector_quantizer_len = nlist * <--D-->;
    size_t HBM_product_quantizer_len = <--M--> * 256 * (<--D--> / <--M-->);
    size_t HBM_OPQ_matrix_len = <--D--> * <--D-->;
    size_t HBM_out_len = TOPK * query_num * 2; 

    // the storage format of the meta info:
    //   (1) HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid: size = 3 * nlist
    //   (2) HBM_product_quantizer: size = K * D
    //   (3) (optional) s_OPQ_init: D * D, if OPQ_enable = False, send nothing
    //   (4) HBM_query_vectors: size = query_num * D (send last, because the accelerator needs to send queries continuously)
    size_t HBM_meta_info_len;
    if (OPQ_enable) {
        HBM_meta_info_len = HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len + 
            HBM_product_quantizer_len + HBM_OPQ_matrix_len + HBM_query_vector_len;
    } else {
        HBM_meta_info_len = HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len + 
            HBM_product_quantizer_len + HBM_query_vector_len;
    }

    // the raw ground truth size is the same for idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs
    // size_t raw_gt_vec_ID_len = 10000 * 1001; 
<--RAW_gt_vec_ID_len-->
    // recall counts the very first nearest neighbor only
    size_t gt_vec_ID_len = 10000;

<--HBM_centroid_vectors_stage2_size-->

    size_t HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_size = 
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len * sizeof(int);
    size_t HBM_query_vector_size = HBM_query_vector_len * sizeof(float);
    size_t HBM_vector_quantizer_size = HBM_vector_quantizer_len * sizeof(float);
    size_t HBM_product_quantizer_size = HBM_product_quantizer_len * sizeof(float);
    size_t HBM_OPQ_matrix_size = HBM_OPQ_matrix_len * sizeof(float);
    size_t HBM_out_size = HBM_out_len * sizeof(uint32_t); 
    size_t HBM_meta_info_size = HBM_meta_info_len * sizeof(float);

    size_t raw_gt_vec_ID_size = raw_gt_vec_ID_len * sizeof(int);
    size_t gt_vec_ID_size = gt_vec_ID_len * sizeof(int);

//////////////////////////////   TEMPLATE END  //////////////////////////////

    cl_int err;
    unsigned fileBufSize;

    // allocate aligned 2D vectors
//////////////////////////////   TEMPLATE START  //////////////////////////////
<--HBM_embedding_allocate-->
<--HBM_centroid_vectors_stage2_allocate-->

    std::vector<int, aligned_allocator<int>> HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid(
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_len, 0);
    std::vector<float, aligned_allocator<float>> HBM_query_vectors(HBM_query_vector_len, 0);
    std::vector<float, aligned_allocator<float>> HBM_vector_quantizer(HBM_vector_quantizer_len, 0);
    std::vector<float, aligned_allocator<float>> HBM_product_quantizer(HBM_product_quantizer_len, 0);
    std::vector<float, aligned_allocator<float>> HBM_OPQ_matrix(HBM_OPQ_matrix_len, 0);
    std::vector<uint32_t, aligned_allocator<uint32_t>> HBM_out(HBM_out_len, 0);
    std::vector<float, aligned_allocator<float>> HBM_meta_info(HBM_meta_info_len, 0);
    
    std::vector<int, aligned_allocator<int>> raw_gt_vec_ID(raw_gt_vec_ID_len, 0);
    std::vector<int, aligned_allocator<int>> gt_vec_ID(gt_vec_ID_len, 0);

//////////////////////////////   TEMPLATE END  //////////////////////////////

<--HBM_embedding_char-->

    char* HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_char = 
        (char*) malloc(HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_size);
    char* HBM_query_vector_char = (char*) malloc(HBM_query_vector_size);
    char* HBM_vector_quantizer_char = (char*) malloc(HBM_vector_quantizer_size);
    char* HBM_product_quantizer_char = (char*) malloc(HBM_product_quantizer_size);
    char* HBM_OPQ_matrix_char = (char*) malloc(HBM_OPQ_matrix_size);

    char* raw_gt_vec_ID_char = (char*) malloc(raw_gt_vec_ID_size);

<--HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream-->
<--HBM_query_vector_fstream-->
<--HBM_vector_quantizer_fstream-->
<--HBM_product_quantizer_fstream-->

    if (OPQ_enable) {
        std::string HBM_OPQ_matrix_suffix_dir = "OPQ_matrix_float32_<--D-->_<--D-->_raw";
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

<--raw_gt_vec_ID_fstream-->
        
<--HBM_embedding_fstream_read-->

    HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream.read(
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_char,
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_size);
    if (!HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream) {
        std::cout << "error: only " << HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream.gcount() << " could be read";
        exit(1);
    }
    HBM_query_vector_fstream.read(HBM_query_vector_char, HBM_query_vector_size);
    if (!HBM_query_vector_fstream) {
        std::cout << "error: only " << HBM_query_vector_fstream.gcount() << " could be read";
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

    // std::cout << "HBM_query_vector_fstream read bytes: " << HBM_query_vector_fstream.gcount() << std::endl;
    // std::cout << "HBM_vector_quantizer_fstream read bytes: " << HBM_vector_quantizer_fstream.gcount() << std::endl;
    // std::cout << "HBM_product_quantizer_fstream read bytes: " << HBM_product_quantizer_fstream.gcount() << std::endl;
 
<--HBM_embedding_memcpy-->

    memcpy(&HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid[0], 
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_char, 
        HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_size);
    memcpy(&HBM_query_vectors[0], HBM_query_vector_char, HBM_query_vector_size);
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

    int start_addr_HBM_query_vectors;
    if (OPQ_enable) {
        start_addr_HBM_query_vectors = start_addr_OPQ_init + size_OPQ_init;
    }
    else { 
        start_addr_HBM_query_vectors = start_addr_HBM_product_quantizer + size_HBM_product_quantizer;
    }
    int size_HBM_query_vectors = query_num * D;
    memcpy(&HBM_meta_info[start_addr_HBM_query_vectors], 
        &HBM_query_vectors[0],
        size_HBM_query_vectors * sizeof(float));
    

<--HBM_centroid_vectors_stage2_memcpy-->

    memcpy(&raw_gt_vec_ID[0], raw_gt_vec_ID_char, raw_gt_vec_ID_size);

    // don't free, XRT will do it

    // free(HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_char);
    // free(HBM_query_vector_char);
    // free(HBM_vector_quantizer_char);
    // free(HBM_product_quantizer_char);
    // free(HBM_OPQ_matrix_char);

    // free(raw_gt_vec_ID_char);
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
        // gt_vec_ID[i] = raw_gt_vec_ID[i * 1001 + 1];
<--TOP1_ID-->
    }

// OPENCL HOST CODE AREA START
	
// ------------------------------------------------------------------------------------
// Step 1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
//	   Search for Platform: Xilinx 
// Check if the current platform matches Target_Platform_Vendor
// ------------------------------------------------------------------------------------	
    std::vector<cl::Device> devices = get_devices();
    devices.resize(1);
    cl::Device device = devices[0];
    std::cout << "Finished getting device..." << std::endl;
// ------------------------------------------------------------------------------------
// Step 1: Create Context
// ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	std::cout << "Finished creating context..." << std::endl;
// ------------------------------------------------------------------------------------
// Step 1: Create Command Queue
// ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
	std::cout << "Finished creating command queue..." << std::endl;
// ------------------------------------------------------------------
// Step 1: Load Binary File from disk
// ------------------------------------------------------------------		
    xclbin_file_name = argv[1];
    cl::Program::Binaries bins = import_binary_file();
    std::cout << "Finished loading binary..." << std::endl;
	
// -------------------------------------------------------------
// Step 1: Create the program object from the binary and program the FPGA device with it
// -------------------------------------------------------------	
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
	std::cout << "Finished creating program..." << std::endl;
// -------------------------------------------------------------
// Step 1: Create Kernels
// -------------------------------------------------------------
    OCL_CHECK(err, cl::Kernel krnl_vector_add(program,"vadd", &err));
    std::cout << "Finished creating kernel..." << std::endl;

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

//////////////////////////////   TEMPLATE END  //////////////////////////////

//////////////////////////////   TEMPLATE START  //////////////////////////////
<--buffer_HBM_embedding-->
<--buffer_HBM_centroid_vectors-->

//     OCL_CHECK(err, cl::Buffer buffer_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid(
//         context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
//         HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_size, 
//         HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid.data(), &err));

//     OCL_CHECK(err, cl::Buffer buffer_HBM_query_vectors(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
//             HBM_query_vector_size, HBM_query_vector.data(), &err));
//     OCL_CHECK(err, cl::Buffer buffer_HBM_product_quantizer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
//             HBM_product_quantizer_size, HBM_product_quantizer.data(), &err));
// #ifdef OPQ_ENABLE
//     OCL_CHECK(err, cl::Buffer buffer_HBM_OPQ_matrix(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
//             HBM_OPQ_matrix_size, HBM_OPQ_matrix.data(), &err));
// #endif

    OCL_CHECK(err, cl::Buffer buffer_HBM_vector_quantizer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            HBM_vector_quantizer_size, HBM_vector_quantizer.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_HBM_meta_info(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            HBM_meta_info_size, HBM_meta_info.data(), &err));
    
// .......................................................
// Allocate Global Memory for sourcce_hw_results
// .......................................................
    OCL_CHECK(err, cl::Buffer buffer_output(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
        HBM_out_size, HBM_out.data(), &err));

// ============================================================================
// Step 2: Set Kernel Arguments and Run the Application
//         o) Set Kernel Arguments
//         o) Copy Input Data from Host to Global Memory on the device
//         o) Submit Kernels for Execution
//         o) Copy Results from Global Memory, device to Host
// ============================================================================	
    
//////////////////////////////   TEMPLATE START  //////////////////////////////
    int arg_counter = 0;
<--buffer_HBM_embedding_set_krnl_arg-->
<--buffer_HBM_centroid_vectors_stage2_set_krnl_arg-->
    
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_query_vectors));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_meta_info));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_vector_quantizer));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_product_quantizer));
// #ifdef OPQ_ENABLE
//     OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_OPQ_matrix));
// #endif
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, query_num));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, nprobe_per_table_construction_pe_larger));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, nprobe_per_table_construction_pe_smaller));

    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_output));
    
//////////////////////////////   TEMPLATE END  //////////////////////////////
// ------------------------------------------------------
// Step 2: Copy Input data from Host to Global Memory on the device
// ------------------------------------------------------
//////////////////////////////   TEMPLATE START  //////////////////////////////
    std::cout << "Starting copy from Host to device... (wait for 10 sec to make sure wait finishes)" << std::endl;
    OCL_CHECK(
        err, err = q.enqueueMigrateMemObjects({
<--buffer_HBM_embedding_enqueueMigrateMemObjects-->
<--buffer_HBM_centroid_vectors_stage2_enqueueMigrateMemObjects-->
        // buffer_HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid,
        // buffer_HBM_query_vectors,
        buffer_HBM_meta_info,
        buffer_HBM_vector_quantizer
//         buffer_HBM_product_quantizer,
// #ifdef OPQ_ENABLE
//         buffer_HBM_OPQ_matrix
// #endif
        }, 0/* 0 means from host*/));	
    sleep(10);
    std::cout << "Host to device finished..." << std::endl;
//////////////////////////////   TEMPLATE END  //////////////////////////////
// ----------------------------------------
// Step 2: Submit Kernels for Execution
// ----------------------------------------
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));
// --------------------------------------------------
// Step 2: Copy Results from Device Global Memory to Host
// --------------------------------------------------
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST));

    q.finish();
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
// OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
    // only check the last batch (since other are not transfered back)
    std::cout << "Comparing Results..." << std::endl;
    bool match = true;
    int count = 0;
    int match_count = 0;


    for (int query_id = 0; query_id < query_num; query_id++) {

        std::vector<int> hw_result_vec_ID_partial(TOPK, 0);
        std::vector<float> hw_result_dist_partial(TOPK, 0);

        // Load data
        for (int k = 0; k < TOPK; k++) {

            uint32_t vec_ID = HBM_out[2 * (query_id * TOPK + k)];
            uint32_t raw_dist =HBM_out[2 * (query_id * TOPK + k) + 1];
            float dist = *((float*) (&raw_dist));
            
            hw_result_vec_ID_partial[k] = vec_ID;
            hw_result_dist_partial[k] = dist;
        }
        
        // Check correctness
        count++;
        // std::cout << "query id" << query_id << std::endl;
        for (int k = 0; k < TOPK; k++) {
            // std::cout << "hw: " << hw_result_vec_ID_partial[k] << "gt: " << gt_vec_ID[query_id] << std::endl;
            if (hw_result_vec_ID_partial[k] == gt_vec_ID[query_id]) {
                match_count++;
                break;
            }
        } 
    }

    float recall = ((float) match_count / (float) count);
    printf("\n=====  Recall: %.8f  =====\n", recall);
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    std::cout << "duration (sec), including dev->host cp, may have small difference with Vitis profiler:" << durationUs / 1000.0 / 1000.0 << std::endl;
    std::cout << "QPS: " << query_num / (durationUs / 1000.0 / 1000.0) << std::endl;

    // keep this, otherwise if return 0, XRT has memory free bugs ...
    exit(0);

    // std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 
    // return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
