#include <immintrin.h> // SIMD

#include <stdint.h>
#include <iostream>



float distance_single_code(
// float inline distance_single_code(
	const float* tab,
	const int M, 
    const int pq_ksub,
	const uint8_t* code) {

        float result = 0;

        __m256 collected = _mm256_setzero_ps();
        __m256 partialSum = _mm256_setzero_ps();
        partialSum = _mm256_add_ps(partialSum, collected);

        // horizontal sum for partialSum
        const __m256 h0 = _mm256_hadd_ps(partialSum, partialSum);
        const __m256 h1 = _mm256_hadd_ps(h0, h0);

        // extract high and low __m128 regs from __m256
        const __m128 h2 = _mm256_extractf128_ps(h1, 1);
        const __m128 h3 = _mm256_castps256_ps128(h1);

        // get a final hsum into all 4 regs
        const __m128 h4 = _mm_add_ss(h2, h3);

        // extract f[0] from __m128
        const float hsum = _mm_cvtss_f32(h4);
        result += hsum;
        
        return result;
    }

// float distance_single_code(
// // float inline distance_single_code(
// 	const float* tab,
// 	const int M, 
//     const int pq_ksub,
// 	const uint8_t* code) {

//         float result = 0;

//         const int pqM16 = M / 16; // Wenqi: 16 float in a 256-bit register 

//         if (pqM16 > 0) {
//             // process 16 values per loop

//             // const __m256i ksub = _mm256_set1_epi32(pq_ksub);
//             const __m256i ksub = _mm256_set1_epi32(0);
            
//             __m256i offsets_0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
//             offsets_0 = _mm256_mullo_epi32(offsets_0, ksub);

//             // accumulators of partial sums
//             __m256 partialSum = _mm256_setzero_ps();

//             // loop
//             for (int m = 0; m < pqM16 * 16; m += 16) {
              
              
//                 // Wenqi: first deal with the first 8 bytes
//                 const __m128i mm1 =
//                         _mm_loadu_si128((const __m128i_u*)(code + m));
//                 {
//                     // convert uint8 values (low part of __m128i) to int32
//                     // values
//                     const __m256i idx1 = _mm256_cvtepu8_epi32(mm1);

//                     // add offsets
//                     const __m256i indices_to_read_from =
//                             _mm256_add_epi32(idx1, offsets_0);

//                   	// Wenqi: it actually uses the gather operation!
//                     // gather 8 values, similar to 8 operations of tab[idx]
//                     __m256 collected = _mm256_i32gather_ps(
//                             tab, indices_to_read_from, sizeof(float));
//                     tab += pq_ksub * 8;

//                     // collect partial sums
//                     partialSum = _mm256_add_ps(partialSum, collected);
//                 }

//               	// Wenqi: then deal with the rest of 8 bytes
//                 // move high 8 uint8 to low ones
//                 const __m128i mm2 =
//                         _mm_unpackhi_epi64(mm1, _mm_setzero_si128());
//                 {
//                     // convert uint8 values (low part of __m128i) to int32
//                     // values
//                     const __m256i idx1 = _mm256_cvtepu8_epi32(mm2);

//                     // add offsets
//                     const __m256i indices_to_read_from =
//                             _mm256_add_epi32(idx1, offsets_0);

//                     // gather 8 values, similar to 8 operations of tab[idx]
//                     __m256 collected = _mm256_i32gather_ps(
//                             tab, indices_to_read_from, sizeof(float));
//                     tab += pq_ksub * 8;

//                     // collect partial sums
//                     partialSum = _mm256_add_ps(partialSum, collected);
//                 }
//             }

//             // horizontal sum for partialSum
//             const __m256 h0 = _mm256_hadd_ps(partialSum, partialSum);
//             const __m256 h1 = _mm256_hadd_ps(h0, h0);

//             // extract high and low __m128 regs from __m256
//             const __m128 h2 = _mm256_extractf128_ps(h1, 1);
//             const __m128 h3 = _mm256_castps256_ps128(h1);

//             // get a final hsum into all 4 regs
//             const __m128 h4 = _mm_add_ss(h2, h3);

//             // extract f[0] from __m128
//             const float hsum = _mm_cvtss_f32(h4);
//             result += hsum;
//         }

//         //
//         // if (m < M) {
//         //     // process leftovers
//         //     PQDecoder decoder(code + m, pq.nbits);

//         //     for (; m < M; m++) {
//         //         result += tab[decoder.decode()];
//         //         tab += pq.ksub;
//         //     }
//         // }

//         return result;
//     }

int main() {

    const int M = 16;
    const int pq_ksub = 256;
    float* tab = new float[M * 256]; 
    uint8_t* code = new uint8_t[10000];

    float result = distance_single_code(
        tab,
        M, 
        pq_ksub,
        code);

    std::cout << result << std::endl;

    return 0; 
}