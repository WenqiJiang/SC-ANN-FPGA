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

// #include <chrono>

#define QUERY_SIZE 512 // 128 D float vector 
#define RESULT_SIZE 128 // 10 * (float + int) = 80 bytes + padding = 128 bytes
// #define RESULT_SIZE 128 // 10 * (float + int) = 80 bytes + padding = 128 bytes
#define QUERY_NUM 10000
#define SEND_BYTES (QUERY_SIZE * QUERY_NUM) // the number of bytes to be send
#define RECV_BYTES (RESULT_SIZE * QUERY_NUM) // the number of bytes to be send

#define PORT 8888

typedef struct {
        time_t   tv_sec;        /* which seconds */
        long     tv_nsec;       /* which nanoseconds */
} timespec;


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


#define DEBUG

struct Thread_info {
    int port;
};

// A normal C function that is executed as a thread  
void *thread_send_packets(void* vargp) 
{ 
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing Port from Thread %d\n", t_info -> port); 
    

    int sock = 0, valread; 
    struct sockaddr_in serv_addr; 

    char* send_buf = (char*) malloc(SEND_BYTES);
    char* recv_buf= (char*) malloc(RECV_BYTES);
    //for (int i = 0; i < BLOCK_ENTRY_NUM; i++) {
    //    send_buf[i] = 1;
    //}

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return -1; 
    } 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(t_info -> port); 
       
    // Convert IPv4 and IPv6 addresses from text to binary form 
    // if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)  
    if(inet_pton(AF_INET, "10.1.212.153", &serv_addr.sin_addr)<=0)  
    { 
        printf("\nInvalid address/ Address not supported \n"); 
        return -1; 
    } 
   
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr))<0) 
    { 
        printf("\nConnection Failed \n"); 
        return -1; 
    } 

    printf("Start sending data.\n");
    ////////////////   Data transfer   ////////////////
    int i = 0;

    timespec* query_start_time_array = (timespec*) malloc(QUERY_NUM * sizeof(timespec));
    timespec* query_end_time_array = (timespec*) malloc(QUERY_NUM * sizeof(timespec));

    // clock_t start = clock();

	timespec start, end;
	clock_gettime(CLOCK_MONOTONIC  , &start);
	// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);

    // auto start = std::chrono::high_resolution_clock::now();

    int total_sent_bytes = 0;
    int total_recv_bytes = 0;

    for (int query_id = 0; query_id < QUERY_NUM; query_id++) {

        int current_query_sent_bytes = 0;
        int current_query_recv_bytes = 0;

	    clock_gettime(CLOCK_MONOTONIC  , &query_start_time_array[query_id]);

        while (current_query_sent_bytes < QUERY_SIZE) {
            int sent_bytes = send(sock, send_buf + total_sent_bytes, QUERY_SIZE - current_query_sent_bytes, 0);
            total_sent_bytes += sent_bytes;
            current_query_sent_bytes += sent_bytes;
            if (sent_bytes == -1) {
                printf("Sending data UNSUCCESSFUL!\n");
                return -1;
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
                return -1;
            }
#ifdef DEBUG
            else {
                printf("totol received bytes: %d\n", total_recv_bytes);
            }
#endif
        }

	    clock_gettime(CLOCK_MONOTONIC  , &query_end_time_array[query_id]);
    }

    if (total_sent_bytes != SEND_BYTES) {
        printf("Sending error, sending more bytes than a block\n");
    }
    else {
	printf("Finish sending\n");
    }

    if (total_recv_bytes != RECV_BYTES) {
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
    printf("\nConsumed time: %f ms\n", total_time_ms);


    double* response_time_ms_array = (double*) malloc(QUERY_NUM * sizeof(double));
    double total_response_time_ms = 0.0;
    for (int query_id = 0; query_id < QUERY_NUM; query_id++) {
        timespec diff_RT = diff(query_start_time_array[query_id], query_end_time_array[query_id]);
        response_time_ms_array[query_id] = 
            ((double) diff_RT.tv_sec) * 1000.0 + 
            ((double) diff_RT.tv_nsec) / 1000.0 / 1000.0;
        total_response_time_ms += response_time_ms_array[query_id];
    }
    double average_response_time_ms = total_response_time_ms / QUERY_NUM;
    printf("\nAverage Response Time: %f ms\n", average_response_time_ms);


    // auto end = std::chrono::high_resolution_clock::now();
    // double durationUs = 0.0;
    // durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
    // printf("durationUs:%f\n",durationUs);

//    float total_size = (float)LOOP_NUM * BATCH_NUM_PER_LOOP * SEND_BYTES;
//    printf("Data sent. Packet number:%d\tPacket size:%d bytes\tTotal data:%fGB\n",
//        LOOP_NUM * BATCH_NUM_PER_LOOP, SEND_BYTES, total_size / (1024 * 1024 * 1024));   
//    float elapsed_time = (end-start) / (float)CLOCKS_PER_SEC;
//    printf("\nConsumed time: %f seconds\n", elapsed_time);
//    printf("Transfer Throughput: %f GB / sec\n", total_size / elapsed_time / 1024 / 1024 / 1024); 

    return NULL; 
} 

int main(int argc, char const *argv[]) 
{ 

    pthread_t thread_id; 
    printf("Before Thread\n"); 

    struct Thread_info t_info_0;
    t_info_0.port = PORT;

    pthread_create(&thread_id, NULL, thread_send_packets, (void*) &t_info_0); 
    // pthread_create(&thread_id, NULL, thread_send_packets, NULL); 
    pthread_join(thread_id, NULL); 
    printf("After Thread\n"); 

    return 0; 
} 
