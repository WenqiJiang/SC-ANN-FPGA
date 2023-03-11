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

//#define SEND_BYTES 1
// #define SEND_BYTES (1024 * 1024) // the number of bytes to be send
#define SEND_BYTES (512 * 10000) // the number of bytes to be send

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

    char send_buf[SEND_BYTES];
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
    //if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)  
    //if(inet_pton(AF_INET, "10.1.212.153", &serv_addr.sin_addr)<=0)  
    if(inet_pton(AF_INET, "10.253.74.84", &serv_addr.sin_addr)<=0)  
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

    while (total_sent_bytes < SEND_BYTES) {
	    int send_bytes_this_iter = (SEND_BYTES - total_sent_bytes) < 4096? (SEND_BYTES - total_sent_bytes) : 4096;
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

    if (total_sent_bytes != SEND_BYTES) {
        printf("Sending error, sending more bytes than a block\n");
    }

    clock_t end = clock();

    return NULL; 
} 

int main(int argc, char const *argv[]) 
{ 

    pthread_t thread_id_0; 
    //pthread_t thread_id_1; 
    printf("Before Thread\n"); 

    struct Thread_info t_info_0;
    //struct Thread_info t_info_1;
    t_info_0.port = PORT;
    //t_info_1.port = PORT + 1;

    pthread_create(&thread_id_0, NULL, thread_send_packets, (void*) &t_info_0); 
    //pthread_create(&thread_id_1, NULL, thread_send_packets, (void*) &t_info_1); 

    pthread_join(thread_id_0, NULL); 
    //pthread_join(thread_id_1, NULL); 
    printf("After Thread\n"); 

    return 0; 
} 
