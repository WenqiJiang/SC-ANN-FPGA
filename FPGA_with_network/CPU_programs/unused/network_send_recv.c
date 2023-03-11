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

//#define SEND_BYTES (4 * 1024) // the number of bytes to be send
//#define RECV_BYTES (4 * 1024) // the number of bytes to be send
//#define SEND_BYTES (1024 * 1024) // the number of bytes to be send
//#define RECV_BYTES (1024 * 1024) // the number of bytes to be send
#define SEND_BYTES (1024 * 1024 * 1024) // the number of bytes to be send
#define RECV_BYTES (1024 * 1024 * 1024) // the number of bytes to be send

#define PORT 8888

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
    if(inet_pton(AF_INET, "10.1.212.152", &serv_addr.sin_addr)<=0)  
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

    clock_t start = clock();


    int total_sent_bytes = 0;
    int total_recv_bytes = 0;

    // NO, the send pkgWord should be 16 * 512 bit = 1024 bytes, thus always waiting for data to arrive
    const int segment_size = 512; // does not work,  packet size should be at least MTU? 
    // const int segment_size = 1024; // works, packet size should be at least MTU? (currennt MTU=512byte)
    // const int segment_size = 8192; // works
    // const int segment_size = 32768; // works

    while (total_sent_bytes < SEND_BYTES || total_recv_bytes < RECV_BYTES) {

	// also make sure that the sent and received bytes aren't very different
	if (total_sent_bytes - total_recv_bytes <= segment_size) {
      	    int send_bytes_this_iter = (SEND_BYTES - total_sent_bytes) < segment_size? (SEND_BYTES - total_sent_bytes) : segment_size;
            int sent_bytes = send(sock, send_buf + total_sent_bytes, send_bytes_this_iter, 0);
            total_sent_bytes += sent_bytes;
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

	if (total_sent_bytes > total_recv_bytes) {
            int recv_bytes_this_iter = (RECV_BYTES - total_recv_bytes) < segment_size? (RECV_BYTES - total_recv_bytes) : segment_size;
    	    int recv_bytes = read(sock, recv_buf + total_recv_bytes, recv_bytes_this_iter);
            total_recv_bytes += recv_bytes;
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
    }

    if (total_sent_bytes != SEND_BYTES) {
        printf("Sending error, sending more bytes than a block\n");
    }
    else {
	printf("Finish sending\n");
    }

    clock_t end = clock();

    if (total_recv_bytes != RECV_BYTES) {
        printf("Receiving error, receiving more bytes than a block\n");
    }
    else {
	printf("Finish receiving\n");
    }
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
