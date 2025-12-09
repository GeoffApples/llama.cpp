#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

static __device__ __forceinline__ void dequantize_q3_hifi(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q3_hifi * x = (const block_q3_hifi *) vx;

    const float d = x[ib].d;
    const uint8_t * ql = x[ib].ql;
    const uint8_t * qh = x[ib].qh;

    // Extract two 3-bit values starting at iqs using split ql/qh layout
    // ql: 64 bytes, 4 values per byte (2-bit low parts)
    // qh: 32 bytes, 8 values per byte (1-bit high parts)
    const int idx0 = iqs;
    const int idx1 = iqs + 1;

    // Extract first value
    const int ql_byte0 = idx0 / 4;
    const int ql_shift0 = (idx0 % 4) * 2;
    const int qh_byte0 = idx0 / 8;
    const int qh_shift0 = idx0 % 8;
    const int low0 = (ql[ql_byte0] >> ql_shift0) & 0x03;
    const int high0 = (qh[qh_byte0] >> qh_shift0) & 0x01;
    const int quant_val0 = (low0 | (high0 << 2)) - 4; // [0,7] → [-4,3]

    // Extract second value
    const int ql_byte1 = idx1 / 4;
    const int ql_shift1 = (idx1 % 4) * 2;
    const int qh_byte1 = idx1 / 8;
    const int qh_shift1 = idx1 % 8;
    const int low1 = (ql[ql_byte1] >> ql_shift1) & 0x03;
    const int high1 = (qh[qh_byte1] >> qh_shift1) & 0x01;
    const int quant_val1 = (low1 | (high1 << 2)) - 4; // [0,7] → [-4,3]

    v.x = quant_val0 * d;
    v.y = quant_val1 * d;

    // Check if either index is an outlier and restore if so
    #pragma unroll
    for (int k = 0; k < Q3_HIFI_OUTFIERS_PER_BLOCK; ++k) {
        if (x[ib].outlier_idx[k] == idx0) {
            v.x = __half2float(x[ib].outlier_vals[k]);
        }
        if (x[ib].outlier_idx[k] == idx1) {
            v.y = __half2float(x[ib].outlier_vals[k]);
        }
    }
}
