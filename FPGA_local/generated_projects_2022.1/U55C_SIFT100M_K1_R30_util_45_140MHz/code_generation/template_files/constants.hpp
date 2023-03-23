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

#define NLIST <--NLIST-->
#define NPROBE <--NPROBE-->
#define D <--D-->
#define M <--M-->
#define K <--K-->
#define TOPK <--TOPK-->

<--OPQ_ENABLE-->

#define QUERY_NUM <--QUERY_NUM-->

#define LARGE_NUM <--LARGE_NUM--> // used to init the heap

// stage 2
// 16 = 15 equal one + 1 (all equal) diff must be 1!
<--STAGE2_ON_CHIP-->
#define PE_NUM_CENTER_DIST_COMP <--PE_NUM_CENTER_DIST_COMP-->
<--stage_2_specification-->

// stage 3
// 2 levels, first level 2 queue, second level 1 queue
#define STAGE_3_PRIORITY_QUEUE_LEVEL <--STAGE_3_PRIORITY_QUEUE_LEVEL-->
#define STAGE_3_PRIORITY_QUEUE_L1_NUM <--STAGE_3_PRIORITY_QUEUE_L1_NUM-->

// stage 4
// first PE: construct 9 tables per query, last one construct 8
#define PE_NUM_TABLE_CONSTRUCTION <--PE_NUM_TABLE_CONSTRUCTION-->
<--stage_4_specification-->

// stage 5
#define HBM_CHANNEL_NUM <--HBM_CHANNEL_NUM-->
#define STAGE5_COMP_PE_NUM <--STAGE5_COMP_PE_NUM-->
#define PQ_CODE_CHANNELS_PER_STREAM <--PQ_CODE_CHANNELS_PER_STREAM-->


// number of 16 outputs per cycle, e.g., HBM channel num = 10, comp PE num = 30, then 
//   SORT_GROUP_NUM = 2; if HBM channel = 12, PE_num = 36, then SORT_GROUP_NUM = 3
#define SORT_GROUP_NUM <--SORT_GROUP_NUM-->
#define STAGE_6_PRIORITY_QUEUE_LEVEL <--STAGE_6_PRIORITY_QUEUE_LEVEL-->
#define STAGE_6_PRIORITY_QUEUE_L1_NUM <--STAGE_6_PRIORITY_QUEUE_L1_NUM-->
<--STAGE_6_L3_MACRO-->