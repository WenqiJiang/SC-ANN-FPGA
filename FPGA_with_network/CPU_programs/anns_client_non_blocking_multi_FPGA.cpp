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

#define QUERY_NUM 10000
#define MAX_FPGA_NUM 16

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

bool first_time_earlier(timespec first, timespec second) {

    if (first.tv_sec < second.tv_sec) {
        return true;
    } else if (first.tv_sec > second.tv_sec) {
        return false;
    } else {
        if (first.tv_nsec < second.tv_nsec) {
            return true;
        } else {
            return false;
        }
    }
}

struct Thread_info {
    char* IP; // server
    int tid;
    int port;
    int query_num;
    int send_recv_gap; // e.g., send is only allow to send 5 queries before recv

    // copy query_num queries from the total 10000 queries
    char* global_query_vector_buf;
    char* global_result_buf;
    timespec* query_start_time_array;
    timespec* query_end_time_array;
};

typedef struct Result{
    int vec_ID;
    float dist; 
} result_t;

int num_FPGA;
int topK;
int result_size;

bool start_receiving = false;
bool finish_receiving = false;
double duration_ms; // transmission duration in milliseconds

int sock[MAX_FPGA_NUM];

int send_query_id[MAX_FPGA_NUM] = {-1};
int receive_query_id[MAX_FPGA_NUM] = {-1};

int query_size; // e.g., 512-byte 128 D float vector for SIFT 

void *thread_send_queries(void* vargp) 
{ 
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing from send thread...\n"); 
    
    const int tid = t_info -> tid;

    const int query_num = t_info -> query_num;

    char* global_query_vector_buf = t_info -> global_query_vector_buf;

    timespec* query_start_time_array = t_info -> query_start_time_array;

    int send_recv_gap = t_info -> send_recv_gap; 

    struct sockaddr_in serv_addr; 

    char* send_buf = new char[query_size * (query_num + send_recv_gap)];

    memcpy(send_buf, global_query_vector_buf, query_size * query_num);

    if ((sock[tid] = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return 0; 
    } 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(t_info -> port); 
       
    if(inet_pton(AF_INET, t_info -> IP, &serv_addr.sin_addr)<=0)  
    { 
        printf("\nInvalid address/ Address not supported \n"); 
        return 0; 
    } 
   
    if (connect(sock[tid], (struct sockaddr *)&serv_addr, sizeof(serv_addr))<0) 
    { 
        printf("\nConnection Failed \n"); 
        return 0; 
    } 

    printf("Start sending data.\n");
    start_receiving = true;

    int total_sent_bytes = 0;
    ////////////////   Data transfer   ////////////////
    // clock_t start = clock();

	timespec start, end;
	clock_gettime(CLOCK_BOOTTIME, &start);

    for (int query_id = 0; query_id < query_num; query_id++) {

        send_query_id[tid] = query_id;
        std::cout << "sending thread " << tid << " query id: " << send_query_id[tid] << std::endl;

        if (receive_query_id[tid] == query_num - 1) {
            printf("break send thread");
            break;
        }
        volatile int tmp_counter;
        do {
            // wait
            tmp_counter++;
        } while(send_query_id[tid] - receive_query_id[tid] >= send_recv_gap && !finish_receiving);
        do {
            // wait until other threads are on this query
            int count = 0;
            for (int i = 0; i < num_FPGA; i++) {
                if (send_query_id[i] >= send_query_id[tid]) {
                    count++;
                }
            }
            if (count == num_FPGA) {
                break;
            }
        } while (true);

	    int current_query_sent_bytes = 0;

	    clock_gettime(CLOCK_BOOTTIME, &query_start_time_array[query_id]);

        while (current_query_sent_bytes < query_size) {
            int sent_bytes = send(sock[tid], send_buf + total_sent_bytes, query_size - current_query_sent_bytes, 0);
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
    }

    // if (total_sent_bytes != query_num * query_size) {
    //     printf("Sending error, sending more bytes than a block\n");
    // }
    // else {
	printf("Finish sending\n");
    // }

	clock_gettime(CLOCK_BOOTTIME, &end);

    timespec total_time = diff(start, end);

    float total_time_ms = 
        ((float) total_time.tv_sec) * 1000.0 + 
        ((float) total_time.tv_nsec) / 1000.0 / 1000.0;
    printf("\nSend thread duration: %f ms\n", total_time_ms);
    float QPS = 10000.0 / (total_time_ms / 1000.0);
    printf("Send thread QPS: %f\n", QPS);

    return NULL; 
} 


void *thread_receive_results(void* vargp) 
{ 
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing from receive thread...\n"); 
    
    const int tid = t_info -> tid;
    const int query_num = t_info -> query_num;
    char* global_result_buf = t_info -> global_result_buf;
    timespec* query_end_time_array = t_info -> query_end_time_array;
    int send_recv_gap = t_info -> send_recv_gap; 

    char* recv_buf= new char[result_size * query_num];

    volatile int tmp_counter;
    do {
        // nothing
	    tmp_counter++;
    } while(!start_receiving);
    printf("Start receiving data.\n");

    int total_recv_bytes = 0;

    ////////////////   Data transfer   ////////////////
    

	timespec start, end;
	clock_gettime(CLOCK_BOOTTIME, &start);

    for (int query_id = 0; query_id < query_num - send_recv_gap; query_id++) {

        receive_query_id[tid] = query_id;
        std::cout << "receiving thread " << tid << " query id: " << receive_query_id[tid] << std::endl;
        
        volatile int tmp_counter;
        do {
            // wait
            tmp_counter++;
        } while(send_query_id[tid] < receive_query_id[tid]);

        int current_query_recv_bytes = 0;

        while (current_query_recv_bytes < result_size) {
    	    int recv_bytes = recv(sock[tid], recv_buf + total_recv_bytes, result_size - current_query_recv_bytes, 0);
            total_recv_bytes += recv_bytes;
            current_query_recv_bytes += total_recv_bytes;
            if (recv_bytes == -1) {
                printf("Receiving data UNSUCCESSFUL!\n");
                return 0;
            }
#ifdef DEBUG
            else {
                printf("totol received bytes: %d\n", total_recv_bytes);
            }
#endif
        }

	    clock_gettime(CLOCK_BOOTTIME, &query_end_time_array[query_id]);
    }

    // if (total_recv_bytes != query_num * result_size) {
    //     printf("Receiving error, receiving more bytes than a block\n");
    // }
    // else {
	printf("Finish receiving\n");
    finish_receiving = true;
    // }

	clock_gettime(CLOCK_BOOTTIME, &end);

    timespec total_time = diff(start, end);

    float total_time_ms = 
        ((float) total_time.tv_sec) * 1000.0 + 
        ((float) total_time.tv_nsec) / 1000.0 / 1000.0;
    printf("\nReceive thread duration: %f ms\n", total_time_ms);
    float QPS = 10000.0 / (total_time_ms / 1000.0);
    printf("Receive thread QPS: %f\n", QPS);

    memcpy(global_result_buf, recv_buf, query_num * result_size);

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

    // <data directory> is only used for loading query vector, can be any folder containing queries
    // <send_recv_gap> denotes the sender can be , e.g. x=5, queries in front of receiver, cannot be 100 queries in front which will influence performance
    printf("Usage: <executable> <num_FPGA> <Tx (FPGA) IP_addr (num_FPGA))> <Tx FPGA_port (num_FPGA)> <query vector dir> <DB name: SIFT or Deep> <topK> <RT_file_name> <send_recv_gap>\n");
    // e.g., ./anns_client_non_blocking_multi_FPGA 1 10.253.74.68 8888 /mnt/scratch/wenqi/saved_npy_data/query_vectors_float32_10000_128_raw SIFT 100 RT_1_server 5

    int arg_count = 1;
    num_FPGA = strtol(argv[arg_count++], NULL, 10);
    
    char** IP_addr = new char*[num_FPGA];
    for (int i = 0; i < num_FPGA; i++) {
        IP_addr[i] = argv[arg_count++];
    }

    unsigned int FPGA_port[num_FPGA] = { 8888 };
    for (int i = 0; i < num_FPGA; i++) {
        FPGA_port[i] = strtol(argv[arg_count++], NULL, 10);
    } 

    std::string query_vector_path = argv[arg_count++];
    std::string dbname = argv[arg_count++];

    if (strcmp(dbname.c_str(), "SIFT") == 0) {
        query_size = 512;
    } else if (strcmp(dbname.c_str(), "Deep") == 0) {
        query_size = 384;
    } else {
        printf("Unknown DB name, has to be SIFT/Deep");
        exit(1);
    }

    topK = std::stoi(argv[arg_count++]);
    std::string RT_file_name = argv[arg_count++];

    int send_recv_gap = std::stoi(argv[arg_count++]);
    
    for (int i = 0; i < num_FPGA; i++) {
        printf("FPGA IP: %s, FPGA port\n", IP_addr[i], FPGA_port[num_FPGA]);
    }
    // printf("Note: push <send_recv_gap> more queries to FPGA to get the final results out...\n");

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

    size_t query_vector_size = QUERY_NUM * query_size;
    char* global_query_vector_buf = new char[query_vector_size];

    char** global_result_buf = new char*[num_FPGA];
    timespec** query_start_time_array = new timespec*[num_FPGA];
    timespec** query_end_time_array = new timespec*[num_FPGA];
    for (int i = 0; i < num_FPGA; i++) {
        global_result_buf[i] = new char[QUERY_NUM * result_size];
        query_start_time_array[i] = new timespec[QUERY_NUM + send_recv_gap];
        query_end_time_array[i] = new timespec[QUERY_NUM];
    }

    float* global_RT_ms_buf = new float[QUERY_NUM];

    // Load query vectors
    std::ifstream query_vector_fstream(
        query_vector_path,
        std::ios::in | std::ios::binary);
    query_vector_fstream.read(global_query_vector_buf, query_vector_size);
    if (!query_vector_fstream) {
        std::cout << "error: only " << query_vector_fstream.gcount() << " could be read";
        exit(1);
    }

    std::cout << "finish loading" << std::endl;

    pthread_t thread_send[num_FPGA]; 
    pthread_t thread_recv[num_FPGA]; 

    struct Thread_info t_info[num_FPGA];

    for (int i = 0; i < num_FPGA; i++) {
        t_info[i].tid = i;
        t_info[i].IP = IP_addr[i];
        t_info[i].port = FPGA_port[i];
        t_info[i].query_num = QUERY_NUM;
        t_info[i].send_recv_gap = send_recv_gap;

        t_info[i].global_query_vector_buf = global_query_vector_buf;
        t_info[i].global_result_buf = global_result_buf[i];

        t_info[i].query_start_time_array = query_start_time_array[i];
        t_info[i].query_end_time_array = query_end_time_array[i];

    }
        
    printf("Before Thread\n");
    for (int i = 0; i < num_FPGA; i++) {
        pthread_create(&thread_send[i], NULL, thread_send_queries, (void*) &t_info[i]); 
    }
    sleep(0.1);
    for (int i = 0; i < num_FPGA; i++) {
        pthread_create(&thread_recv[i], NULL, thread_receive_results, (void*) &t_info[i]); 
    }

    for (int i = 0; i < num_FPGA; i++) {
        pthread_join(thread_send[i], NULL); 
        pthread_join(thread_recv[i], NULL); 
    }
    printf("After Thread\n"); 

    // TODO: global timing
    for (int query_id = 0; query_id < QUERY_NUM; query_id++) {
        // find the earliest send time 
        timespec first_send = query_start_time_array[0][query_id];
        for (int i = 0; i < num_FPGA; i++) {
            if (first_time_earlier(query_start_time_array[i][query_id], first_send)) {
                first_send = query_start_time_array[i][query_id];
            }
        }

        // find the last recv time
        timespec last_recv = query_end_time_array[0][query_id]; 
        for (int i = 0; i < num_FPGA; i++) {
            if (first_time_earlier(last_recv, query_end_time_array[i][query_id])) {
                last_recv = query_end_time_array[i][query_id];
            }
        }

        timespec diff_RT = diff(first_send, last_recv);
        global_RT_ms_buf[query_id] = 
            ((float) diff_RT.tv_sec) * 1000.0 + 
            ((float) diff_RT.tv_nsec) / 1000.0 / 1000.0;
    }

    for (int query_id = QUERY_NUM - send_recv_gap; query_id < QUERY_NUM; query_id++) {
        global_RT_ms_buf[query_id] = global_RT_ms_buf[QUERY_NUM / 2];
    }
    
    // Save RT distribution
    std::string RT_distribution = 
        "./RT_distribution/" + RT_file_name;
    int char_len = RT_distribution.length();
    char RT_distribution_char[char_len + 1];
    strcpy(RT_distribution_char, RT_distribution.c_str());
    FILE *file = fopen(RT_distribution_char, "w");
    fwrite(global_RT_ms_buf, sizeof(float), QUERY_NUM, file);
    fclose(file);

    return 0; 
} 
