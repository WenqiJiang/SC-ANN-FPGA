// Client side C/C++ program to demonstrate Socket programming 
#include <stdio.h> 
#include <stdlib.h> 
#include <stdint.h>
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#include <unistd.h>
#include <time.h>
#include <pthread.h> 
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

// #include <chrono>

// #define DEBUG
int topK;
int result_size;

#define QUERY_NUM 10000
#define QUERY_SIZE 512 // 128 D float vector 
#define RESULT_SIZE 128 // 10 * (float + int) = 80 bytes + padding = 128 bytes

#define SUPPORTED_THREADS 32

timespec diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

bool start_sending_array[SUPPORTED_THREADS];
bool start_sending = false;

double duration_ms[SUPPORTED_THREADS]; // transmission duration in milliseconds

struct Thread_info {
    char* IP; // server
    int port;
    int query_num;
    int thread_id;

    // copy query_num queries from the total 10000 queries
    char* global_query_vector_buf;
    int start_query_id; 
    char* global_result_buf;
    float* global_RT_ms_buf;
};

typedef struct Result{
    int vec_ID;
    float dist; 
} result_t;

void *check_start(void* vargp) 
{
    // int query_num = *((int*) vargp);
    while (true) {
        bool start_count = true;
        for (int i = 0; i < SUPPORTED_THREADS; i++) {
            if (!start_sending_array[i]) {
                start_count = false;
                break;
            }
        }
        if (start_count) {
            start_sending = true;
	    break;
        }
    }
}

// A normal C function that is executed as a thread  
void *thread_send_packets(void* vargp) 
{ 
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing Port from Thread %d\n", t_info -> port); 
    
    const int query_num = t_info -> query_num;
    const int start_query_id = t_info -> start_query_id;
    char* global_query_vector_buf = t_info -> global_query_vector_buf;
    char* global_result_buf = t_info -> global_result_buf;
    float* global_RT_ms_buf = t_info -> global_RT_ms_buf;

    int sock = 0;
    struct sockaddr_in serv_addr; 

    // char* send_buf = (char*) malloc(QUERY_SIZE * query_num);
    // char* recv_buf= (char*) malloc(result_size * query_num);
    char* send_buf = new char[QUERY_SIZE * query_num];
    char* recv_buf= new char[result_size * query_num];

    memcpy(send_buf, &global_query_vector_buf[QUERY_SIZE * start_query_id], QUERY_SIZE * query_num);

    //for (int i = 0; i < BLOCK_ENTRY_NUM; i++) {
    //    send_buf[i] = 1;
    //}

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return 0; 
    } 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(t_info -> port); 
       
    // Convert IPv4 and IPv6 addresses from text to binary form 
    if(inet_pton(AF_INET, t_info -> IP, &serv_addr.sin_addr)<=0)  
    // if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)  
    // if(inet_pton(AF_INET, "10.1.212.153", &serv_addr.sin_addr)<=0)  
    { 
        printf("\nInvalid address/ Address not supported \n"); 
        return 0; 
    } 
   
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr))<0) 
    { 
        printf("\nConnection Failed \n"); 
        return 0; 
    } 

    start_sending_array[t_info -> thread_id] = true;

    volatile int tmp_counter;
    do {
        // nothing
	tmp_counter++;
    } while(!start_sending);
    printf("Start sending data.\n");

    // //timespec* query_start_time_array = (timespec*) malloc(query_num * sizeof(timespec));
    // //timespec* query_end_time_array = (timespec*) malloc(query_num * sizeof(timespec));
    timespec* query_start_time_array = new timespec[query_num];
    timespec* query_end_time_array = new timespec[query_num];

    int total_sent_bytes = 0;
    int total_recv_bytes = 0;
    ////////////////   Data transfer   ////////////////
    // clock_t start = clock();

	timespec start, end;
	clock_gettime(CLOCK_BOOTTIME, &start);
	// clock_gettime(CLOCK_MONOTONIC  , &start);
	// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);

    // auto start = std::chrono::high_resolution_clock::now();


    for (int query_id = 0; query_id < query_num; query_id++) {

    	printf("thread_ID: %d\tquery_ID: %d\n", t_info -> thread_id, query_id);
        
	int current_query_sent_bytes = 0;
        int current_query_recv_bytes = 0;

	    clock_gettime(CLOCK_BOOTTIME, &query_start_time_array[query_id]);
	    // clock_gettime(CLOCK_MONOTONIC  , &query_start_time_array[query_id]);

        while (current_query_sent_bytes < QUERY_SIZE) {
            int sent_bytes = send(sock, send_buf + total_sent_bytes, QUERY_SIZE - current_query_sent_bytes, 0);
            total_sent_bytes += sent_bytes;
            current_query_sent_bytes += sent_bytes;
            if (sent_bytes == -1) {
                printf("Sending data UNSUCCESSFUL!\n");
                return 0;
            } 
#ifdef DEBUG
            else {
                printf("total sent bytes = %d\n", total_sent_bytes);
            }
#endif
        }

        while (current_query_recv_bytes < result_size) {
    	    int recv_bytes = read(sock, recv_buf + total_recv_bytes, result_size - current_query_recv_bytes);
            total_recv_bytes += recv_bytes;
            current_query_recv_bytes += total_recv_bytes;
            if (recv_bytes == -1) {
                printf("Receiving data UNSUCCESSFUL!\n");
                return 0;
            }
#ifdef DEBUG
            else {
                printf("thread %d totol received bytes: %d\n", t_info -> thread_id, total_recv_bytes);
            }
#endif
        }

	    clock_gettime(CLOCK_BOOTTIME, &query_end_time_array[query_id]);
	    // clock_gettime(CLOCK_MONOTONIC  , &query_end_time_array[query_id]);
    }

    if (total_sent_bytes != query_num * QUERY_SIZE) {
        printf("Sending error, sending more bytes than a block\n");
    }
    else {
	printf("Finish sending\n");
    }

    if (total_recv_bytes != query_num * result_size) {
        printf("Receiving error, receiving more bytes than a block\n");
    }
    else {
	printf("Finish receiving\n");
    }

    // clock_t end = clock();

	clock_gettime(CLOCK_BOOTTIME, &end);
	// clock_gettime(CLOCK_MONOTONIC  , &end);
	// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);

    timespec total_time = diff(start, end);

    float total_time_ms = 
        ((float) total_time.tv_sec) * 1000.0 + 
        ((float) total_time.tv_nsec) / 1000.0 / 1000.0;
    printf("\nThread %d duration: %f ms\n", t_info -> thread_id, total_time_ms);


    //float* response_time_ms_array = (float*) malloc(query_num * sizeof(float));
    float* response_time_ms_array = new float[query_num];
    float total_response_time_ms = 0.0;
    for (int query_id = 0; query_id < query_num; query_id++) {
        timespec diff_RT = diff(query_start_time_array[query_id], query_end_time_array[query_id]);
        response_time_ms_array[query_id] = 
            ((float) diff_RT.tv_sec) * 1000.0 + 
            ((float) diff_RT.tv_nsec) / 1000.0 / 1000.0;
        total_response_time_ms += response_time_ms_array[query_id];
    }
    memcpy(global_RT_ms_buf + start_query_id, response_time_ms_array, query_num * sizeof(float));
    // double average_response_time_ms = total_response_time_ms / query_num;
    // printf("\nAverage Response Time: %f ms\n", average_response_time_ms);

    duration_ms[t_info -> thread_id] = total_time_ms;

    // auto end = std::chrono::high_resolution_clock::now();
    // double durationUs = 0.0;
    // durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
    // printf("durationUs:%f\n",durationUs);

    memcpy(&global_result_buf[result_size * start_query_id], recv_buf, result_size * query_num);

    return NULL; 
} 

// boost::filesystem does not compile well, so implement this myself
std::string dir_concat(std::string dir1, std::string dir2) {
    if (dir1.back() != '/') {
        dir1 += '/';
    }
    return dir1 + dir2;
}

int main(int argc, char *argv[]) 
{ 

    printf("Several threads send queries to the server and receive results, sending start simultaneously, max supported connections = %d\n", SUPPORTED_THREADS);

    if (argc != 10) {
        printf("Usage: <executable> <topK> <nlist> <nprobe> <OPQ_enable> <IP> <port> <thread_num> <data directory> <ground truth dir> , e.g., ./anns_client 10 8192 17 1 127.0.0.1 8888 4 /mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF8192,PQ16_16_banks /mnt/scratch/wenqi/saved_npy_data/gnd\n");
        exit(1);
    }
    int topK = std::stoi(argv[1]);
    std::string s_nlist = argv[2];
    std::string s_nprobe = argv[3];
    std::string s_OPQ = argv[4];
    std::string s_IP = argv[5];
    std::string s_port = argv[6];
    std::string s_thread_num = argv[7];
    std::string data_dir_prefix = argv[8];
    std::string gnd_dir = argv[9];


    int n = s_IP.length();
    char IP[n + 1];
    strcpy(IP, s_IP.c_str());

    int nlist = stoi(s_nlist);
    int nprobe = stoi(s_nprobe);
    // int OPQ = stoi(s_OPQ);
    int port = stoi(s_port);
    int thread_num = stoi(s_thread_num);
    const int query_num = QUERY_NUM;
    printf("topK: %d nlist: %d, nprobe: %d, server IP: %s, port: %d, thread_num: %d\n", topK, nlist, nprobe, IP, port, thread_num);

    if (topK == 1) {
         // 1 single 512-bit packet
        result_size = 1 * 64;
        std::cout << "result size (per query) = " << result_size << " bytes" << std::endl;
    }
    if (topK == 10) {
         // 2 512-bit packets
        result_size = 2 * 64;
        std::cout << "result size (per query) = " << result_size << " bytes" << std::endl;
    }
    if (topK == 100) {
         // 13 512-bit packets
         // 100 * 8 byte per result = 800 bytes, ceil(800 / 64) = 13
        result_size = 13 * 64;
        std::cout << "result size (per query) = " << result_size << " bytes" << std::endl;
    }

    size_t query_vector_size = QUERY_NUM * QUERY_SIZE;
    char* global_query_vector_buf = new char[query_vector_size];
    char* global_result_buf = new char[QUERY_NUM * result_size];
    float* global_RT_ms_buf = new float[QUERY_NUM];


    // Load query vectors
    std::string query_vector_dir_suffix = "query_vectors_float32_10000_128_raw";
    std::string query_vector_path = dir_concat(data_dir_prefix, query_vector_dir_suffix);
    std::ifstream query_vector_fstream(
        query_vector_path,
        std::ios::in | std::ios::binary);
    query_vector_fstream.read(global_query_vector_buf, query_vector_size);
    if (!query_vector_fstream) {
        std::cout << "error: only " << query_vector_fstream.gcount() << " could be read";
        exit(1);
    }

    // Load ground truth
    // the raw ground truth size is the same for idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs
    size_t raw_gt_vec_ID_len = 10000 * 1001; 
    size_t raw_gt_vec_ID_size = raw_gt_vec_ID_len * sizeof(int);
    std::vector<int> raw_gt_vec_ID(raw_gt_vec_ID_len, 0);

    std::string raw_gt_vec_ID_suffix_dir = "idx_100M.ivecs";
    std::string raw_gt_vec_ID_dir = dir_concat(gnd_dir, raw_gt_vec_ID_suffix_dir);
    std::ifstream raw_gt_vec_ID_fstream(
        raw_gt_vec_ID_dir,
        std::ios::in | std::ios::binary);
    if (!raw_gt_vec_ID_fstream) {
        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
        exit(1);
    }
    char* raw_gt_vec_ID_char = (char*) malloc(raw_gt_vec_ID_size);
    raw_gt_vec_ID_fstream.read(raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    if (!raw_gt_vec_ID_fstream) {
        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&raw_gt_vec_ID[0], raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    free(raw_gt_vec_ID_char);
    size_t gt_vec_ID_len = 10000;
    std::vector<int> gt_vec_ID(gt_vec_ID_len, 0);
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


    std::cout << "finish loading" << std::endl;

    // Query allocate pattern: 
    //  e.g., 10000 queries in 4 threads -> 0~2499 thread 0; 2500~4999 thread 1; ...; 7500~9999 thread 3
    int query_num_per_thread[thread_num];
    int start_query_id_per_thread[thread_num];
    if (QUERY_NUM % thread_num == 0) {
        int query_per_thread = QUERY_NUM / thread_num;
        for (int i = 0; i < thread_num; i++) {
            query_num_per_thread[i] = query_per_thread;
            start_query_id_per_thread[i] = i * query_per_thread;
        }
    }
    else {
        int query_per_thread = QUERY_NUM / thread_num + 1;
        int query_last_thread = QUERY_NUM - query_per_thread * (thread_num - 1);
        for (int i = 0; i < thread_num - 1; i++) {
            query_num_per_thread[i] = query_per_thread;
            start_query_id_per_thread[i] = i * query_per_thread;
        }
        query_num_per_thread[thread_num - 1] = query_last_thread;
        start_query_id_per_thread[thread_num - 1] = (thread_num - 1) * query_per_thread;
    }

    pthread_t thread_id[thread_num]; 

    struct Thread_info t_info[thread_num];
    for (int i = 0; i < thread_num; i++) {
        t_info[i].IP = IP;
        t_info[i].port = port + i;
        t_info[i].query_num = query_num_per_thread[i];
        t_info[i].thread_id = i;
        t_info[i].global_query_vector_buf = global_query_vector_buf;
        t_info[i].start_query_id = start_query_id_per_thread[i]; 
        t_info[i].global_result_buf = global_result_buf;
        t_info[i].global_RT_ms_buf = global_RT_ms_buf;
    } 

    for (int i = 0; i < thread_num; i++) {
        start_sending_array[i] = false;
    }
    for (int i = thread_num; i < SUPPORTED_THREADS; i++) {
        start_sending_array[i] = true;
    }


    pthread_t check_start_thread; 
    pthread_create(&check_start_thread, NULL, check_start, (void*) &query_num); 

    printf("Before Thread\n");
    for (int i = 0; i < thread_num; i++) {
        pthread_create(&thread_id[i], NULL, thread_send_packets, (void*) &t_info[i]); 
	    sleep(0.1);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(thread_id[i], NULL); 
    }
    printf("After Thread\n"); 

    double total_duration_ms = 0.0;
    double max_duration_ms = 0.0;
    for (int i = 0; i < thread_num; i++) {
        total_duration_ms += duration_ms[i];
        max_duration_ms = max_duration_ms > duration_ms[i]?  max_duration_ms : duration_ms[i];
    }
    double average_duration_ms = total_duration_ms / thread_num;
    printf("Average duration per thread: %f ms\n", average_duration_ms);
    printf("Max duration per thread: %f ms\n", max_duration_ms);
    float QPS = 10000.0 / (max_duration_ms / 1000.0);
    printf("QPS: %f\n", QPS);

    ///// verify correctness /////

    // correctness verification
    std::cout << "Comparing Results..." << std::endl;
    int count = 0;
    int match_count = 0;

    int* hw_result_vec_ID_partial = (int*) malloc(topK * sizeof(int));
    float* hw_result_dist_partial = (float*) malloc(topK * sizeof(float));

    // Result format: <int vec_ID, float distance>
    for (int query_id = 0; query_id < query_num; query_id++) {


        int start_addr = result_size * query_id;
        // Load data
        for (int k = 0; k < topK; k++) {
            memcpy(&hw_result_vec_ID_partial[k], global_result_buf + start_addr + k * 8, 4);
            memcpy(&hw_result_dist_partial[k], global_result_buf + start_addr + k * 8 + 4, 4);
        }
        // for (int k = 0; k < topK; k++) {
        //     std::cout << "query id: " << query_id << "k = " << k << "vec ID = " << hw_result_vec_ID_partial[k] << std::endl;
        // }
        
        // Check correctness
        count++;
        // std::cout << "query id" << query_id << std::endl;
        for (int k = 0; k < topK; k++) {
            // std::cout << "hw: " << hw_result_vec_ID_partial[k] << "gt: " << gt_vec_ID[query_id] << std::endl;
            if (hw_result_vec_ID_partial[k] == gt_vec_ID[query_id]) {
                match_count++;
                break;
            }
        } 
    }
    free(hw_result_vec_ID_partial);
    free(hw_result_dist_partial);

    // for (int i = 0; i < QUERY_NUM * result_size / 8; i++) {
    //     char* start_addr = global_result_buf + 8 * i;
    //     int vec_ID;
    //     float dist;
    //     memcpy(&vec_ID, start_addr, 4);
    //     memcpy(&dist, start_addr + 4, 4);
    //     std::cout << "vecID = " << vec_ID << "dist = " << dist << std::endl;
    // }

    float recall = ((float) match_count / (float) count);
    printf("\n=====  Recall: %.8f  =====\n", recall);

    // Save RT distribution
    std::string RT_distribution = 
        "./RT_distribution/RT_distribution_10000_queries_nlist_" + std::to_string(nlist) + 
        "_nprobe_" + std::to_string(nprobe) + "_thread_num_" + std::to_string(thread_num);
    int char_len = RT_distribution.length();
    char RT_distribution_char[char_len + 1];
    strcpy(RT_distribution_char, RT_distribution.c_str());
    FILE *file = fopen(RT_distribution_char, "w");
    fwrite(global_RT_ms_buf, sizeof(float), QUERY_NUM, file);
    fclose(file);

    return 0; 
} 
