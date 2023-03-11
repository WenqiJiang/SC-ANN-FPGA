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

//#define RECV_BYTES 1
#define RECV_BYTES (1024 * 1024) // the number of bytes to be send

#define PORT 8889

#define DEBUG

struct Thread_info {
    int port;
};

// A normal C function that is executed as a thread  
void *thread_send_packets(void* vargp) 
{ 

    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing Port from Thread %d\n", t_info -> port); 

    int server_fd, sock;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char *finish = "Finish receiving.";

    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR , &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(t_info -> port);

    // Forcefully attaching socket to the port 8080 
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    if ((sock = accept(server_fd, (struct sockaddr *)&address,
                       (socklen_t*)&addrlen))<0)
    {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    printf("Successfully built connection.\n"); 


    printf("Start receiving data.\n");
    ////////////////   Data transfer   ////////////////
    char recv_buf[RECV_BYTES];

    clock_t start = clock();

    // Should wait until the server said all the data was sent correctly,
    // otherwise the sender may send packets yet the server did not receive.

    int total_recv_bytes = 0;
    while (total_recv_bytes < RECV_BYTES) {
        int recv_bytes_this_iter = (RECV_BYTES - total_recv_bytes) < 4096? (RECV_BYTES - total_recv_bytes) : 4096;
    	int recv_bytes = read(sock, recv_buf + total_recv_bytes, recv_bytes_this_iter);
    	//int recv_bytes = read(sock, recv_buf + total_recv_bytes, RECV_BYTES - total_recv_bytes);
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

    if (total_recv_bytes != RECV_BYTES) {
        printf("Receiving error, receiving more bytes than a block\n");
    }
    clock_t end = clock();
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
