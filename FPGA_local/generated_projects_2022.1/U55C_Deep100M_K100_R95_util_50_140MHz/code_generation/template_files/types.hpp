#pragma once

#include <ap_int.h>
#include <hls_stream.h>

enum Order { Collect_smallest, Collect_largest };

typedef struct {
    float dist;
    int cell_ID;
} dist_cell_ID_t;

typedef struct {
    int vec_ID;
    unsigned char PQ_code[M];
} single_PQ;
 
typedef struct {
    // a wrapper for single_PQ
    // used in the ap_uint<480> to 3 x PQ split function
    single_PQ PQ_0;
    single_PQ PQ_1;
    single_PQ PQ_2;
} three_PQ_codes;

typedef struct {
    int vec_ID;
    float dist;
} single_PQ_result; 

typedef struct {
    int vec_ID0;
    float dist0;
    int vec_ID1;
    float dist1;
    int vec_ID2;
    float dist2;

    // padd to 256 bits
    int vec_ID_dummy;
    float dist_dummy;
} host_PQ_results;

typedef struct {
    // each distance LUT has K=256 such row
    // each distance_LUT_PQ16_t is the content of a single row (16 floats)
    float dist_0; 
    float dist_1; 
    float dist_2; 
    float dist_3; 
    float dist_4; 
    float dist_5; 
    float dist_6;
    float dist_7; 
    float dist_8; 
    float dist_9; 
    float dist_10; 
    float dist_11; 
    float dist_12; 
    float dist_13;
    float dist_14; 
    float dist_15;
} distance_LUT_PQ16_t;

typedef ap_uint<64> ap_uint64_t;
typedef ap_uint<512> ap_uint512_t;
