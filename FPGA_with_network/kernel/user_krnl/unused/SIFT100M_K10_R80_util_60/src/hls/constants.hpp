/*
Variable to be replaced (<--variable_name-->):

    NLIST
    NPROBE
    D
    M
    K
    TOPK
    
    OPQ_ENABLE

    QUERY_NUM

    LARGE_NUM

    // stage 2
    STAGE2_ON_CHIP
    PE_NUM_CENTER_DIST_COMP
    PE_NUM_CENTER_DIST_COMP_EVEN

    // stage 3
    STAGE_3_PRIORITY_QUEUE_LEVEL
    STAGE_3_PRIORITY_QUEUE_L1_NUM

    // stage 4
    PE_NUM_TABLE_CONSTRUCTION
    PE_NUM_TABLE_CONSTRUCTION_LARGER
    PE_NUM_TABLE_CONSTRUCTION_SMALLER

    // stage 5
    HBM_CHANNEL_NUM
    STAGE5_COMP_PE_NUM
    PQ_CODE_CHANNELS_PER_STREAM

    // stage 6
    SORT_GROUP_NUM
    STAGE_6_PRIORITY_QUEUE_LEVEL
    STAGE_6_PRIORITY_QUEUE_L1_NUM
    STAGE_6_L3_MACRO
*/

#pragma once

#define NLIST 8192
#define NPROBE 17
#define D 128
#define M 16
#define K 256
#define TOPK 10

#define OPQ_ENABLE 1

#define QUERY_NUM 10000

#define LARGE_NUM 99999999 // used to init the heap

// stage 2
// 16 = 15 equal one + 1 (all equal) diff must be 1!
#define STAGE2_ON_CHIP 1
#define PE_NUM_CENTER_DIST_COMP 11

#define PE_NUM_CENTER_DIST_COMP_EVEN 10
#define CENTROIDS_PER_PARTITION_EVEN 745
#define CENTROIDS_PER_PARTITION_LAST_PE 742


// stage 3
// 2 levels, first level 2 queue, second level 1 queue
#define STAGE_3_PRIORITY_QUEUE_LEVEL 2
#define STAGE_3_PRIORITY_QUEUE_L1_NUM 2

// stage 4
// first PE: construct 9 tables per query, last one construct 8
#define PE_NUM_TABLE_CONSTRUCTION 9

#define PE_NUM_TABLE_CONSTRUCTION_LARGER 8
#define PE_NUM_TABLE_CONSTRUCTION_SMALLER 1


// stage 5
#define HBM_CHANNEL_NUM 12
#define STAGE5_COMP_PE_NUM 36
#define PQ_CODE_CHANNELS_PER_STREAM 1


// number of 16 outputs per cycle, e.g., HBM channel num = 10, comp PE num = 30, then 
//   SORT_GROUP_NUM = 2; if HBM channel = 12, PE_num = 36, then SORT_GROUP_NUM = 3
#define SORT_GROUP_NUM 3
#define STAGE_6_PRIORITY_QUEUE_LEVEL 2
#define STAGE_6_PRIORITY_QUEUE_L1_NUM 20
