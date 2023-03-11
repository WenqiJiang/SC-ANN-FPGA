// Client side C/C++ program to demonstrate Socket programming 
#include <stdio.h> 
#include <stdlib.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#include <unistd.h>
#include <time.h>
#include <pthread.h> 
#include <iostream>
#include <string>

// #include <chrono>

#define DEBUG

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
    
    const int query_num = (t_info -> query_num);

    int sock = 0, valread; 
    struct sockaddr_in serv_addr; 

    // char* send_buf = (char*) malloc(QUERY_SIZE * query_num);
    // char* recv_buf= (char*) malloc(RESULT_SIZE * query_num);
    char* send_buf = new char[QUERY_SIZE * query_num];
    char* recv_buf= new char[RESULT_SIZE * query_num];

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
    // timespec* query_start_time_array = new timespec[query_num];
    // timespec* query_end_time_array = new timespec[query_num];

    int total_sent_bytes = 0;
    int total_recv_bytes = 0;
    ////////////////   Data transfer   ////////////////
    // clock_t start = clock();

	timespec start, end;
	clock_gettime(CLOCK_MONOTONIC  , &start);
	// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);

    // auto start = std::chrono::high_resolution_clock::now();


    for (int query_id = 0; query_id < query_num; query_id++) {

        int current_query_sent_bytes = 0;
        int current_query_recv_bytes = 0;

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

	clock_gettime(CLOCK_MONOTONIC  , &end);
	// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);

    timespec total_time = diff(start, end);

    double total_time_ms = 
        ((double) total_time.tv_sec) * 1000.0 + 
        ((double) total_time.tv_nsec) / 1000.0 / 1000.0;
    printf("\nThread %d duration: %f ms\n", t_info -> thread_id, total_time_ms);


    // //double* response_time_ms_array = (double*) malloc(query_num * sizeof(double));
    // double* response_time_ms_array = new double[query_num];
    // double total_response_time_ms = 0.0;
    // for (int query_id = 0; query_id < query_num; query_id++) {
    //     timespec diff_RT = diff(query_start_time_array[query_id], query_end_time_array[query_id]);
    //     response_time_ms_array[query_id] = 
    //         ((double) diff_RT.tv_sec) * 1000.0 + 
    //         ((double) diff_RT.tv_nsec) / 1000.0 / 1000.0;
    //     total_response_time_ms += response_time_ms_array[query_id];
    // }
    // double average_response_time_ms = total_response_time_ms / query_num;
    // printf("\nAverage Response Time: %f ms\n", average_response_time_ms);

    duration_ms[t_info -> thread_id] = total_time_ms;

    // auto end = std::chrono::high_resolution_clock::now();
    // double durationUs = 0.0;
    // durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
    // printf("durationUs:%f\n",durationUs);

    return NULL; 
} 

int main(int argc, char *argv[]) 
{ 

    printf("Several threads send queries to the server and receive results, sending start simultaneously, max supported connections = %d\n", SUPPORTED_THREADS);
    printf("Usage: executable IP port query_num_per_thread thread_num , e.g., ./network_send_recv_client_sync 127.0.0.1 8888 10000 4\n");

    std::string s_IP = argv[1];
    std::string s_port = argv[2];
    std::string s_query_num = argv[3];
    std::string s_thread_num = argv[4];


    int n = s_IP.length();
    char IP[n + 1];
    strcpy(IP, s_IP.c_str());

    int port = stoi(s_port);
    int query_num = stoi(s_query_num);
    int thread_num = 1; // this file targets single client
    // int thread_num = stoi(s_thread_num);
    printf("server IP: %s, port: %d, query_num_per_thread: %d, thread_num: %d\n", IP, port, query_num, thread_num);

    pthread_t thread_id[thread_num]; 
    pthread_t thread_id_aux; 

    struct Thread_info t_info[thread_num];
    for (int i = 0; i < thread_num; i++) {
        t_info[i].IP = IP;
        t_info[i].port = port + i;
        t_info[i].query_num = query_num;
        t_info[i].thread_id = i;
    } 
    struct Thread_info t_info_aux;
    t_info_aux.IP = IP;
    t_info_aux.port = port + thread_num;
    t_info_aux.query_num = 10;
    t_info_aux.thread_id = thread_num;

    // count the aux thread
    for (int i = 0; i < thread_num + 1; i++) {
        start_sending_array[i] = false;
    }
    for (int i = thread_num + 1; i < SUPPORTED_THREADS; i++) {
        start_sending_array[i] = true;
    }


    pthread_t check_start_thread; 
    pthread_create(&check_start_thread, NULL, check_start, (void*) &query_num); 

    printf("Before Thread\n");
    for (int i = 0; i < thread_num; i++) {
        pthread_create(&thread_id[i], NULL, thread_send_packets, (void*) &t_info[i]); 
	sleep(0.1);
    }
    pthread_create(&thread_id_aux, NULL, thread_send_packets, (void*) &t_info_aux); 

    for (int i = 0; i < thread_num; i++) {
        pthread_join(thread_id[i], NULL); 
    }
    pthread_join(thread_id_aux, NULL); 
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

    return 0; 
} 
