#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <unistd.h>
#include <time.h>
#include <pthread.h> 

#include <sys/types.h> 
#include <sys/ipc.h> 
#include <sys/shm.h>
#include <sys/socket.h> 
#include <netinet/in.h> 

#include "constant.h"

struct Thread_info {
    int port;
    float* data;
    int* control;
};

// A normal C function that is executed as a thread  
void *thread_receive_packets(void* vargp) 
{
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing Port from Thread %d\n", t_info -> port); 

    int* writer_block_id = t_info -> control + 0;
    int* writer_iteration = t_info -> control + 1;
    int* reader_block_id = t_info -> control + 2;
    int* reader_iteration = t_info -> control + 3;

    ////////////////////  Network Init  ////////////////////

    int server_fd, new_socket, valread; 
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
   // Forcefully attaching socket to the port 8080 
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR , &opt, sizeof(opt)))
    { 
        perror("setsockopt"); 
        exit(EXIT_FAILURE); 
    } 

    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons( t_info -> port ); 
       
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
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address,  
                       (socklen_t*)&addrlen))<0) 
    { 
        perror("accept"); 
        exit(EXIT_FAILURE); 
    } 
    printf("Successfully built connection.\n");

    ////////////////////  Network Receive -> Shared Memory  ////////////////////

    clock_t start = clock();
    // use loop instead of memset for floating point init
    for (int iter = 0; iter < LOOP_NUM; iter++) {

        *writer_iteration = iter;
        // id in the queue
        for (int block_id = 0; block_id < BATCH_NUM_PER_LOOP; block_id++) {

#ifdef DEBUG
            printf("iteration: %d\tblock_id: %d\n", iter, block_id);
#endif
            // NOTE: if testing performance, comment the control logic below
            // 2 cases where writer need to wait reader to finish read
            // Case 1: If (reader hasn’t finished read this region last round, then writer halt) 
            // -> almost in front of reader for an entire round
            while ((iter > *reader_iteration) && (block_id + 1 >= *reader_block_id)) {
                // halt until the reader proceed
            }
            // Case 2: the writer is near the end of the iteration while the 
            // reader just start, in this case the distance still need to be kept
            while ((block_id == BATCH_NUM_PER_LOOP - 1) 
                && (iter >= *reader_iteration) // consider the case when read_iter=-1
                && (*reader_block_id < 1)) { // consider the case when read_iter=-1
                // halt until the reader proceed
            }  

            // write block
            int block_addr = block_id * BLOCK_ENTRY_NUM;
            int total_recv_bytes = 0;
            // replace with network transfer
            while(total_recv_bytes < BLOCK_SIZE) {
                int recv_bytes = read(new_socket, t_info -> data + block_addr + total_recv_bytes, 
                    BLOCK_SIZE - total_recv_bytes);
                if (recv_bytes == -1) {
                    printf("Receiving data UNSUCCESSFUL!\n");
                    return -1;
                }
                total_recv_bytes += recv_bytes;
            }
            if (total_recv_bytes != BLOCK_SIZE) {
                printf("Receiving error, receiving more bytes than a block!\n");
            }

            *writer_block_id = block_id;
        }

    }
    clock_t end = clock() ;
    float elapsed_time = (end-start) / (float)CLOCKS_PER_SEC;
    printf("Consumed time: %f seconds, INCLUDING Waiting eader proceess\n", elapsed_time);
    printf("Throughput: %f GB / sec", (float)SHM_DATA_SIZE * LOOP_NUM / elapsed_time / 1024 / 1024 / 1024);

    return NULL; 
} 

void *thread_consume(void* vargp) {

    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing Port from Thread %d\n", t_info -> port); 

    int* writer_block_id = t_info -> control + 0;
    int* writer_iteration = t_info -> control + 1;
    int* reader_block_id = t_info -> control + 2;
    int* reader_iteration = t_info -> control + 3;

    float* input_feature = malloc(BLOCK_SIZE);

    for (int iter = 0; iter < LOOP_NUM; iter++) {

        *reader_iteration = iter;
        // id in the queue
        for (int block_id = 0; block_id < BATCH_NUM_PER_LOOP; block_id++) {

    #ifdef DEBUG
            printf("iteration: %d\tblock_id: %d\n", iter, block_id);
    #endif
            if (iter == 0 && block_id == 0) {
                clock_t start = clock();  
            }            
    #ifdef DEBUG
            printf("control signal: writer_iteration %d\twriter_block_id%d\n",
                *writer_iteration, *writer_block_id);
    #endif
            // NOTE: if testing performance, comment the control logic below
            // 2 cases where reader need to wait writer to finish write
            // Case 1: In the same w/r iteration, writer is just in front of reader
            //   for 1 blcok
            while ((iter >= *writer_iteration) && (block_id + 1 >= *writer_block_id)) {
                // halt until the reader proceed
                // if last iteration, go through
                if ((iter == LOOP_NUM - 1) 
                    && (*writer_iteration == LOOP_NUM - 1)
                    && (*writer_block_id == BATCH_NUM_PER_LOOP - 1)) {
                    break;
                }
            }
            // Case 2: the reader is near the end of the iteration while the 
            //   writer just start writing the next iteration, in this case 
            //   the distance still need to be kept
            while ((block_id == BATCH_NUM_PER_LOOP - 1) 
                && (iter == *writer_iteration - 1) // consider the case when read_iter=-1
                && (*writer_block_id < 1)) { // consider the case when read_iter=-1
                // halt until the reader proceed
                // if last iteration, go through
                if ((iter == LOOP_NUM - 1) 
                    && (*writer_iteration == LOOP_NUM - 1)
                    && (*writer_block_id == BATCH_NUM_PER_LOOP - 1)) { 
                    break;
                }
            }  

            // write block
            int block_addr = block_id * BLOCK_ENTRY_NUM;
            memcpy(input_feature, t_info -> data + block_addr, BLOCK_SIZE);
            *reader_block_id = block_id;
        }
    }
}

int main(int argc, char *argv[]) {

    ////////////////////  Shared Memory Init  ////////////////////
 
    float* data = malloc(SHM_DATA_SIZE); 
    int* control = malloc(4 * sizeof(int));

    // initialize data block, allocate consecutive memory
    memset(data, 0, SHM_DATA_SIZE);

    // initialize control signal
    control[0] = -1; // first value: block id of writer (client)
    control[1] = -1; // second value: iteration id of writer
    control[2] = -1; // third value: block id of reader (CUDA)
    control[3] = -1; // fourth value: iteration id of reader
    
    pthread_t thread_id_store; 
    pthread_t thread_id_consume; 
    printf("Before Thread\n"); 

    struct Thread_info t_info;
    t_info.port = PORT;
    t_info.data = data;
    t_info.control = control;

    pthread_create(&thread_id_store, NULL, thread_receive_packets, (void*) &t_info); 
    pthread_create(&thread_id_consume, NULL, thread_consume, (void*) &t_info); 
    // pthread_create(&thread_id, NULL, thread_receive_packets, NULL); 
    pthread_join(thread_id_store, NULL); 
    pthread_join(thread_id_consume, NULL); 
    printf("After Thread\n"); 
    
    return 0;
}