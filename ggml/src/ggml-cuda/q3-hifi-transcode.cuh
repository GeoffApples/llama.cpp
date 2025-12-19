// Q3_HIFI GPU Transcoding - Separates Q3_K core from outliers for optimal kernel dispatch
// This allows running Q3_K's highly-optimized CUDA kernel for bulk computation,
// then applying outlier corrections in a separate, coalesced kernel.

#pragma once

#include "common.cuh"

// Outlier data structure for separated storage (24 bytes per block)
// Stored contiguously for better memory coalescing
struct q3_hifi_outliers {
    uint8_t  idx[Q3_HIFI_OUTLIERS];   // 8 bytes: outlier positions (0-255)
    uint16_t vals[Q3_HIFI_OUTLIERS];  // 16 bytes: FP16 outlier values (as uint16_t for alignment)
};
static_assert(sizeof(q3_hifi_outliers) == Q3_HIFI_OUTLIERS + Q3_HIFI_OUTLIERS * sizeof(uint16_t), "wrong q3_hifi_outliers size");

// Transcoding kernel: Extract Q3_K core and outliers into separate buffers
// This runs ONCE at model load time, so overhead is amortized over inference
__global__ void k_q3_hifi_transcode(
    const block_q3_hifi * __restrict__ src,
    block_q3_K * __restrict__ dst_q3k,
    q3_hifi_outliers * __restrict__ dst_outliers,
    const int nb
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;

    const block_q3_hifi & s = src[i];
    block_q3_K & d = dst_q3k[i];
    q3_hifi_outliers & o = dst_outliers[i];

    // Copy Q3_K-compatible core (110 bytes)
    // The memory layout is identical, so this is a direct copy
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        d.hmask[j] = s.hmask[j];
    }
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        d.qs[j] = s.qs[j];
    }
    #pragma unroll
    for (int j = 0; j < 12; ++j) {
        d.scales[j] = s.scales[j];
    }
    d.d = s.d;

    // Copy outliers to separate buffer
    #pragma unroll
    for (int k = 0; k < Q3_HIFI_OUTLIERS; ++k) {
        o.idx[k] = s.outlier_idx[k];
        o.vals[k] = reinterpret_cast<const uint16_t&>(s.outlier_vals[k]);
    }
}

// Outlier correction kernel - runs AFTER the main Q3_K matmul
// This kernel adds outlier contributions to the output vector
// 
// For matrix-vector multiply: y = W * x
//   - Each row of W corresponds to one output element
//   - Each row spans multiple Q3_K blocks
//   - Outliers within each block contribute: outlier_val * x[outlier_idx] * scale
//
// Parameters:
//   outliers: Outlier data for all blocks
//   q8_x: Q8-quantized input activations (from quantize_row_q8_1_cuda)
//   dst: Output vector to add corrections to
//   blocks_per_row: Number of Q3_K/Q3_HIFI blocks per output row
//   nrows: Number of output rows
__global__ void k_q3_hifi_outlier_correction(
    const q3_hifi_outliers * __restrict__ outliers,
    const block_q8_1 * __restrict__ q8_x,
    float * __restrict__ dst,
    const int blocks_per_row,
    const int nrows
) {
    // Each thread handles one output row
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    float correction = 0.0f;
    
    // Iterate over all blocks in this row
    const int block_offset = row * blocks_per_row;
    
    for (int b = 0; b < blocks_per_row; ++b) {
        const int block_idx = block_offset + b;
        const q3_hifi_outliers & out = outliers[block_idx];
        
        // Process all outliers in this block
        #pragma unroll
        for (int k = 0; k < Q3_HIFI_OUTLIERS; ++k) {
            const int idx = out.idx[k];                    // Position within block (0-255)
            const int global_idx = b * QK_K + idx;         // Global position in input
            
            // Decode FP16 outlier value
            const half outlier_half = reinterpret_cast<const half&>(out.vals[k]);
            const float outlier_val = __half2float(outlier_half);
            
            // Get corresponding Q8 activation
            const int q8_block = global_idx / QK8_1;       // Which Q8 block
            const int q8_idx = global_idx % QK8_1;         // Index within Q8 block
            
            const float q8_scale = __low2float(q8_x[q8_block].ds);
            const int8_t q8_val = q8_x[q8_block].qs[q8_idx];
            
            correction += outlier_val * q8_val * q8_scale;
        }
    }
    
    // Add correction to output
    dst[row] += correction;
}

// Alternative: Warp-coalesced outlier correction for better memory access
// Each warp processes multiple rows cooperatively
template <int ROWS_PER_WARP = 4>
__global__ void k_q3_hifi_outlier_correction_coalesced(
    const q3_hifi_outliers * __restrict__ outliers,
    const block_q8_1 * __restrict__ q8_x,
    float * __restrict__ dst,
    const int blocks_per_row,
    const int nrows
) {
    constexpr int WARP_SIZE = 32;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int base_row = warp_id * ROWS_PER_WARP;
    
    // Each lane processes a subset of blocks for multiple rows
    float corrections[ROWS_PER_WARP] = {0.0f};
    
    // Distribute blocks across lanes
    const int blocks_per_lane = (blocks_per_row + WARP_SIZE - 1) / WARP_SIZE;
    
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= nrows) continue;
        
        const int block_offset = row * blocks_per_row;
        
        for (int bl = 0; bl < blocks_per_lane; ++bl) {
            const int b = lane_id + bl * WARP_SIZE;
            if (b >= blocks_per_row) continue;
            
            const int block_idx = block_offset + b;
            const q3_hifi_outliers & out = outliers[block_idx];
            
            #pragma unroll
            for (int k = 0; k < Q3_HIFI_OUTLIERS; ++k) {
                const int idx = out.idx[k];
                const int global_idx = b * QK_K + idx;
                
                const half outlier_half = reinterpret_cast<const half&>(out.vals[k]);
                const float outlier_val = __half2float(outlier_half);
                
                const int q8_block = global_idx / QK8_1;
                const int q8_idx = global_idx % QK8_1;
                
                const float q8_scale = __low2float(q8_x[q8_block].ds);
                const int8_t q8_val = q8_x[q8_block].qs[q8_idx];
                
                corrections[r] += outlier_val * q8_val * q8_scale;
            }
        }
    }
    
    // Warp-level reduction and write
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= nrows) continue;
        
        // Reduce across warp
        float sum = corrections[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        
        // Lane 0 writes the result
        if (lane_id == 0) {
            dst[row] += sum;
        }
    }
}

// Host-side wrapper functions
inline void q3_hifi_transcode_cuda(
    const block_q3_hifi * src,
    block_q3_K * dst_q3k,
    q3_hifi_outliers * dst_outliers,
    int nb,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (nb + block_size - 1) / block_size;
    k_q3_hifi_transcode<<<grid_size, block_size, 0, stream>>>(src, dst_q3k, dst_outliers, nb);
}

inline void q3_hifi_outlier_correction_cuda(
    const q3_hifi_outliers * outliers,
    const block_q8_1 * q8_x,
    float * dst,
    int blocks_per_row,
    int nrows,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (nrows + block_size - 1) / block_size;
    k_q3_hifi_outlier_correction<<<grid_size, block_size, 0, stream>>>(
        outliers, q8_x, dst, blocks_per_row, nrows);
}

inline void q3_hifi_outlier_correction_coalesced_cuda(
    const q3_hifi_outliers * outliers,
    const block_q8_1 * q8_x,
    float * dst,
    int blocks_per_row,
    int nrows,
    cudaStream_t stream
) {
    constexpr int ROWS_PER_WARP = 4;
    constexpr int WARP_SIZE = 32;
    const int warps_needed = (nrows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int threads_needed = warps_needed * WARP_SIZE;
    const int block_size = 256;
    const int grid_size = (threads_needed + block_size - 1) / block_size;
    k_q3_hifi_outlier_correction_coalesced<ROWS_PER_WARP><<<grid_size, block_size, 0, stream>>>(
        outliers, q8_x, dst, blocks_per_row, nrows);
}

