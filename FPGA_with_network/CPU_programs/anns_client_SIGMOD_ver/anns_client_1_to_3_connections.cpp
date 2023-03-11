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
#define TOPK 10
#define D 128
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
// whether to start per query, managed by the read sender threads
// as long as one real sender is ready, the dummy sender can start to follow up queries
bool start_sending_per_query[QUERY_NUM / 4];
bool start_sending = false;

double duration_ms[SUPPORTED_THREADS]; // transmission duration in milliseconds

struct Thread_info {
    char* IP; // server
    int port;
    int query_num;
    int thread_id;
    int real_thread_num;

    // copy query_num queries from the total 10000 queries
    char* global_query_vector_buf;
    int start_query_id; 
    char* global_result_buf;
    float* global_RT_ms_buf;
};

void *check_start(void* vargp) 
{
    int query_num = *((int*) vargp);
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
    const int real_thread_num = t_info -> real_thread_num;
    const int start_query_id = t_info -> start_query_id;
    const int thread_id = t_info -> thread_id;
    char* global_query_vector_buf = t_info -> global_query_vector_buf;
    char* global_result_buf = t_info -> global_result_buf;
    float* global_RT_ms_buf = t_info -> global_RT_ms_buf;

    int sock = 0, valread; 
    struct sockaddr_in serv_addr; 

    // char* send_buf = (char*) malloc(QUERY_SIZE * query_num);
    // char* recv_buf= (char*) malloc(RESULT_SIZE * query_num);
    char* send_buf = new char[QUERY_SIZE * query_num];
    char* recv_buf= new char[RESULT_SIZE * query_num];

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

        int current_query_sent_bytes = 0;
        int current_query_recv_bytes = 0;

	    clock_gettime(CLOCK_BOOTTIME, &query_start_time_array[query_id]);
	    // clock_gettime(CLOCK_MONOTONIC  , &query_start_time_array[query_id]);

        if (thread_id < real_thread_num) {
            start_sending_per_query[query_id] = true;
        }
        else {
            while (!start_sending_per_query[query_id]) {

            }
        }

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

        while (current_query_recv_bytes < RESULT_SIZE) {
    	    int recv_bytes = read(sock, recv_buf + total_recv_bytes, RESULT_SIZE - current_query_recv_bytes);
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

    if (total_recv_bytes != query_num * RESULT_SIZE) {
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

    memcpy(&global_result_buf[RESULT_SIZE * start_query_id], recv_buf, RESULT_SIZE * query_num);

    return NULL; 
} 

int main(int argc, char *argv[]) 
{ 

    printf("Several threads send queries to the server and receive results, sending start simultaneously, max supported connections = %d\n", SUPPORTED_THREADS);
    printf("There will be 4 threads sending data in total, some of them will be dummy senders to fill the FPGA pipeline\n");
    printf("Usage: executable nlist nprobe OPQ_enable IP port thread_num , e.g., ./anns_client_1_to_3_connections 8192 17 1 127.0.0.1 8888 4\n");

    std::string s_nlist = argv[1];
    std::string s_nprobe = argv[2];
    std::string s_OPQ = argv[3];
    std::string s_IP = argv[4];
    std::string s_port = argv[5];
    std::string s_thread_num = argv[6];

    int n = s_IP.length();
    char IP[n + 1];
    strcpy(IP, s_IP.c_str());

    int nlist = stoi(s_nlist);
    int nprobe = stoi(s_nprobe);
    int OPQ = stoi(s_OPQ);
    int port = stoi(s_port);
    int thread_num = stoi(s_thread_num);
    const int query_num = QUERY_NUM;
    if (thread_num > 3) {
        thread_num = 3;
        printf("This program only supports less than 3 threads, reset the thread num to 3\n");
    }
    printf("nlist: %d, nprobe: %d, server IP: %s, port: %d, thread_num: %d\n", nlist, nprobe, IP, port, thread_num);


    size_t query_vector_size = QUERY_NUM * QUERY_SIZE;
    char* global_query_vector_buf = new char[query_vector_size];
    char* global_result_buf = new char[QUERY_NUM * RESULT_SIZE];
    char* query_vector_char = (char*) malloc(query_vector_size);
    float* global_RT_ms_buf = new float[QUERY_NUM];
    std::string dir;
    if (OPQ) {
        dir = "/home/wejiang/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF";
    }
    else {
        dir = "/home/wejiang/saved_npy_data/FPGA_data_SIFT100M_IVF";
    }

    std::string query_vector_fname = 
        dir + std::to_string(nlist) + 
        ",PQ16_HBM_10_banks/query_vectors_float32_10000_128_raw";
    std::ifstream query_vector_fstream(
        query_vector_fname,
        std::ios::in | std::ios::binary);
    query_vector_fstream.read(query_vector_char, query_vector_size);
    memcpy(&global_query_vector_buf[0], query_vector_char, query_vector_size);

    size_t sw_result_vec_ID_size = QUERY_NUM * TOPK * sizeof(int);
    size_t sw_result_dist_size = QUERY_NUM * TOPK * sizeof(float);
    char* sw_result_vec_ID_char = (char*) malloc(sw_result_vec_ID_size);
    char* sw_result_dist_char = (char*) malloc(sw_result_dist_size);
    std::string sw_result_vec_ID_fname = 
        dir + std::to_string(nlist) + 
        ",PQ16_HBM_10_banks/result_nprobe_" + std::to_string(nprobe) + "_index_int32_10000_10_raw";
    std::string sw_result_dist_fname = 
        dir + std::to_string(nlist) + 
        ",PQ16_HBM_10_banks/result_nprobe_" + std::to_string(nprobe) + "_distance_float32_10000_10_raw";
    std::cout << sw_result_vec_ID_fname << " " << sw_result_dist_fname;
    std::ifstream sw_result_vec_ID_fstream(
        sw_result_vec_ID_fname,
        std::ios::in | std::ios::binary);
    std::ifstream sw_result_dist_fstream(
        sw_result_dist_fname,
        std::ios::in | std::ios::binary);
    sw_result_vec_ID_fstream.read(sw_result_vec_ID_char, sw_result_vec_ID_size);
    sw_result_dist_fstream.read(sw_result_dist_char, sw_result_dist_size);

    const int total_thread_num_include_dummy = 4;
    int query_num_per_thread[total_thread_num_include_dummy];
    int start_query_id_per_thread[total_thread_num_include_dummy];
    if (QUERY_NUM % total_thread_num_include_dummy == 0) {
        int query_per_thread = QUERY_NUM / total_thread_num_include_dummy;
        for (int i = 0; i < total_thread_num_include_dummy; i++) {
            query_num_per_thread[i] = query_per_thread;
            start_query_id_per_thread[i] = i * query_per_thread;
        }
    }
    else {
        int query_per_thread = QUERY_NUM / total_thread_num_include_dummy + 1;
        int query_last_thread = QUERY_NUM - query_per_thread * (total_thread_num_include_dummy - 1);
        for (int i = 0; i < total_thread_num_include_dummy - 1; i++) {
            query_num_per_thread[i] = query_per_thread;
            start_query_id_per_thread[i] = i * query_per_thread;
        }
        query_num_per_thread[total_thread_num_include_dummy - 1] = query_last_thread;
        start_query_id_per_thread[total_thread_num_include_dummy - 1] = (total_thread_num_include_dummy - 1) * query_per_thread;
    }

    // overwrite the dummy contents to 0s
    int dummy_start_query_id = 0;
    for (int i = 0; i < thread_num; i++) {
        dummy_start_query_id += query_num_per_thread[i];
    }
    for (int query_id = dummy_start_query_id; query_id < QUERY_NUM; query_id++) {
        for (int b = 0; b < sizeof(float) * D; b++) {
            global_query_vector_buf[query_id * sizeof(float) * D + b] = 0;
        }
    }


    pthread_t thread_id[total_thread_num_include_dummy]; 

    struct Thread_info t_info[total_thread_num_include_dummy];
    for (int i = 0; i < total_thread_num_include_dummy; i++) {
        t_info[i].IP = IP;
        t_info[i].port = port + i;
        t_info[i].query_num = query_num_per_thread[i];
        t_info[i].thread_id = i;
        t_info[i].real_thread_num = thread_num;
        t_info[i].global_query_vector_buf = global_query_vector_buf;
        t_info[i].start_query_id = start_query_id_per_thread[i]; 
        t_info[i].global_result_buf = global_result_buf;
        t_info[i].global_RT_ms_buf = global_RT_ms_buf;
    } 

    for (int i = 0; i < total_thread_num_include_dummy; i++) {
        start_sending_array[i] = false;
    }
    for (int i = total_thread_num_include_dummy; i < SUPPORTED_THREADS; i++) {
        start_sending_array[i] = true;
    }
    for (int i = 0; i < QUERY_NUM / 4; i++) {
        start_sending_per_query[i] = false;
    }


    pthread_t check_start_thread; 
    pthread_create(&check_start_thread, NULL, check_start, (void*) &query_num); 

    printf("Before Thread\n");
    for (int i = 0; i < total_thread_num_include_dummy; i++) {
        pthread_create(&thread_id[i], NULL, thread_send_packets, (void*) &t_info[i]); 
	    sleep(0.1);
    }

    for (int i = 0; i < total_thread_num_include_dummy; i++) {
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
    float QPS = QUERY_NUM / 4.0 * thread_num / (max_duration_ms / 1000.0);
    printf("QPS: %f\n", QPS);

    ///// verify correctness /////
    // process received result
    size_t hw_result_vec_ID_size = QUERY_NUM * TOPK * sizeof(int);
    size_t hw_result_dist_size = QUERY_NUM * TOPK * sizeof(float);
    char* hw_result_vec_ID_char = (char*) malloc(hw_result_vec_ID_size);
    char* hw_result_dist_char = (char*) malloc(hw_result_dist_size);

    // extract vecID / dist from network packet
    for (int query_id = 0; query_id < QUERY_NUM; query_id++) {
        // first 512-bit packet: vec_ID
        // second 512-bit packet: dist
        memcpy(&hw_result_vec_ID_char[query_id * TOPK * sizeof(int)], &global_result_buf[query_id * 128], TOPK * sizeof(int));
        memcpy(&hw_result_dist_char[query_id * TOPK * sizeof(float)], &global_result_buf[query_id * 128 + 64], TOPK * sizeof(float));
    }

    // correctness verification
    int count = 0;
    int mismatch_count = 0;
    
    for (int query_id = 0; query_id < dummy_start_query_id; query_id++) {

        std::vector<int> hw_result_vec_ID_partial(TOPK, 0);
        // std::vector<float> hw_result_dist_partial(TOPK, 0);

        std::vector<int> sw_result_vec_ID_partial(TOPK, 0);
        // std::vector<float> sw_result_dist_partial(TOPK, 0);

        // Load data
        for (int k = 0; k < TOPK; k++) {
            memcpy(&sw_result_vec_ID_partial[k], &sw_result_vec_ID_char[(TOPK * query_id + k) * sizeof(int)], sizeof(int));
            memcpy(&hw_result_vec_ID_partial[k], &hw_result_vec_ID_char[(TOPK * query_id + k) * sizeof(int)], sizeof(int));
        }

        std::sort(hw_result_vec_ID_partial.begin(), hw_result_vec_ID_partial.end());
        // std::sort(hw_result_dist_partial.begin(), hw_result_dist_partial.end());

        std::sort(sw_result_vec_ID_partial.begin(), sw_result_vec_ID_partial.end());
        // std::sort(sw_result_dist_partial.begin(), sw_result_dist_partial.end());

        // Check correctness
        for (int k = 0; k < TOPK; k++) {
            count++;
            if (hw_result_vec_ID_partial[k] != sw_result_vec_ID_partial[k]) {
                printf("query_id: %d\tk: %d\thw vec_ID: %d\t sw vec_ID:%d\n",
                    query_id, k, hw_result_vec_ID_partial[k], sw_result_vec_ID_partial[k]);
                mismatch_count++;
            }
        }
    }
    float mismatch_rate = ((float) mismatch_count / (float) count);
    printf("mismatch rate with CPU results: %.8f\n", mismatch_rate);
    if (mismatch_rate < 0.001) {
    printf("TEST PASS\n");
    } else {
    printf("TEST FAIL\n");
    }


    // replace the dummy RT by the real RT
    if (query_num == 1) {
        memcpy(global_RT_ms_buf + QUERY_NUM / 4, global_RT_ms_buf, QUERY_NUM / 4 * sizeof(float));
        memcpy(global_RT_ms_buf + 2 * QUERY_NUM / 4, global_RT_ms_buf, QUERY_NUM / 4 * sizeof(float));
        memcpy(global_RT_ms_buf + 3 * QUERY_NUM / 4, global_RT_ms_buf, QUERY_NUM / 4 * sizeof(float));
    } else if (query_num == 2) {
        memcpy(global_RT_ms_buf + QUERY_NUM / 2, global_RT_ms_buf, QUERY_NUM / 2 * sizeof(float));
    } else if (query_num == 3) {
        memcpy(global_RT_ms_buf + 3 * QUERY_NUM / 4, global_RT_ms_buf, QUERY_NUM / 4 * sizeof(float));
    }

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
