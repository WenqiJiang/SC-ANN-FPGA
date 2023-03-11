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

// #define DEBUG

#define QUERY_SIZE 512 // 128 D float vector 
#define RESULT_SIZE 128 // 10 * (float + int) = 80 bytes + padding = 128 bytes

struct Thread_info {
    int port;
    int query_num;
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

    const int query_num = t_info -> query_num;
    // char* send_buf = (char*) malloc(query_num * RESULT_SIZE);
    // char* recv_buf= (char*) malloc(query_num * QUERY_SIZE);
    char* send_buf = new char[RESULT_SIZE * query_num];
    char* recv_buf= new char[QUERY_SIZE * query_num];

    clock_t start = clock();

    int total_recv_bytes = 0;
    int total_sent_bytes = 0;

    for (int query_id = 0; query_id < query_num; query_id++) {

        int current_query_sent_bytes = 0;
        int current_query_recv_bytes = 0;

        while (current_query_recv_bytes < QUERY_SIZE) {
    	    int recv_bytes = read(sock, recv_buf + total_recv_bytes, QUERY_SIZE - current_query_recv_bytes);
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

        while (current_query_sent_bytes < RESULT_SIZE) {
            int sent_bytes = send(sock, send_buf + total_sent_bytes, RESULT_SIZE - current_query_sent_bytes, 0);
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
    
    clock_t end = clock();

    if (total_sent_bytes != query_num * RESULT_SIZE) {
        printf("Sending error, sending more bytes than a block\n");
    }
    else {
	printf("Finish sending\n");
    }

    if (total_recv_bytes != query_num * QUERY_SIZE) {
        printf("Receiving error, receiving more bytes than a block\n");
    }
    else {
	printf("Finish receiving\n");
    }

    return NULL; 
} 

int main(int argc, char *argv[]) 
{ 

    printf("Usage: executable IP port query_num_per_thread thread_num, e.g., ./network_recv_send_server 8888 10000 4\n");

    std::string s_port = argv[1];
    std::string s_query_num = argv[2];
    std::string s_thread_num = argv[3];

    int port = stoi(s_port);
    int query_num = stoi(s_query_num);
    int thread_num = stoi(s_thread_num);
    printf("listen port: %d, query_num: %d, thread_num: %d\n", port, query_num, thread_num);

    pthread_t thread_id[thread_num];

    struct Thread_info t_info[thread_num];
    for (int i = 0; i < thread_num; i++) {
        t_info[i].port = port + i;
        t_info[i].query_num = query_num;
    }

    printf("Before Thread\n"); 
    for (int i = 0; i < thread_num; i++) {
        pthread_create(&thread_id[i], NULL, thread_send_packets, (void*) &t_info[i]); 
    }
    
    for (int i = 0; i < thread_num; i++) {
        pthread_join(thread_id[i], NULL); 
    }
    printf("After Thread\n"); 

    return 0; 
} 
