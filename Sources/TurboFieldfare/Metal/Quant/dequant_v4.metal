#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_v4 — DeepSeek V4-Flash decode dequant kernels (V4F-02).
//
// Two formats, neither with a bias term:
//
//   FP4 routed experts: e2m1 codes packed two per byte (low nibble = element
//   2i, high = 2i+1) along K; one ue8m0 power-of-two scale byte per 32
//   elements along K. Value: w[i] = kV4E2M1Lut[code[i]] * 2^(scale[i/32]-127).
//
//   FP8 dense weights: one e4m3 byte per element; one ue8m0 scale byte per
//   128x128 2-D block, grid row-major over (ceil(M/128), N/128).
//   Value: w[r,c] = e4m3(code) * 2^(scale_grid[(r/128)][c/128]] - 127).
//
// Power-of-two scales are exact in FP32, so per-group accumulate-then-scale
// carries no rounding hazard beyond the FP32 accumulation itself.
//
// Alignment contract: the repack pads every FP4/FP8 sub-tensor to 4 bytes, so
// weight loads use aligned `uint`/`uchar4` (the fast path the int4 resident
// kernels cannot take). Wrappers precondition weightsOffset % 4 == 0 and
// x % 8-byte alignment for the half4 activation loads.
//
// All helpers are `static` and v4_-prefixed so this file can later merge into
// the shared runtime library without symbol collisions.
// ============================================================================

constant constexpr uint kV4FP4GroupSize = 32;    // elements per ue8m0 scale
constant constexpr uint kV4FP8BlockSize = 128;   // block-scale tile edge
constant constexpr uint kV4RowsPerTG = 8;        // one SIMD group per row

// e2m1 value table, index = 4-bit code (sign in bit 3). Exact in FP32.
constant float kV4E2M1Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static inline float v4_ue8m0(uint8_t b) {
    // 2^(b - 127); exact for every byte.
    return exp2(float(int(b) - 127));
}

static inline float v4_e2m1(uint code) {
    return kV4E2M1Lut[code & 0xFu];
}

static inline float v4_e4m3(uint8_t q) {
    const uint e = (uint(q) >> 3) & 0xFu;
    const uint m = uint(q) & 0x7u;
    float mag;
    if (e == 0u) {
        mag = float(m) * 0x1.0p-9f;                 // subnormal step 2^-9
    } else {
        mag = float(8u + m) * exp2(float(int(e) - 10));  // (8+m) * 2^(e-10)
    }
    return (q & 0x80u) ? -mag : mag;
}

// ----------------------------------------------------------------------------
// dequant_fp4_e2m1_gemv_simd
//
// y[m] = sum_n W[m,n] * x[n]. One SIMD group per row, eight rows per
// threadgroup (the dequant_int4_gemv_simd geometry). Main loop: each block is
// 8 groups = 256 elements = 128 packed bytes; each lane loads one aligned
// uint (4 bytes = 8 e2m1 codes, all inside one 32-element group) and two
// half4 activation chunks. Remainder covers group counts not divisible by 8
// with one byte per lane over lanes 0..15.
//
// Requires N % 32 == 0 and a 4-byte-aligned weights base (wrapper-enforced).
// ----------------------------------------------------------------------------
kernel void dequant_fp4_e2m1_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],   // [M, N/2] packed e2m1 pairs
    device const uint8_t* scales [[buffer(1)]],   // [M, N/32] ue8m0
    device const half*    x      [[buffer(2)]],
    device half*          y      [[buffer(3)]],
    constant uint&        M      [[buffer(4)]],
    constant uint&        N      [[buffer(5)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kV4RowsPerTG + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kV4FP4GroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const uint8_t* s_row = scales + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 8;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        // 8 elements per lane starting at blk*256 + lane*8 → group lane>>2.
        const uint g = blk * 8u + (lane >> 2);
        const float s = v4_ue8m0(s_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        float dot = 0.0f;
        dot = fma(v4_e2m1(b0 & 0xFu), float(xa.x), dot);
        dot = fma(v4_e2m1(b0 >> 4),   float(xa.y), dot);
        dot = fma(v4_e2m1(b1 & 0xFu), float(xa.z), dot);
        dot = fma(v4_e2m1(b1 >> 4),   float(xa.w), dot);
        dot = fma(v4_e2m1(b2 & 0xFu), float(xb.x), dot);
        dot = fma(v4_e2m1(b2 >> 4),   float(xb.y), dot);
        dot = fma(v4_e2m1(b3 & 0xFu), float(xb.z), dot);
        dot = fma(v4_e2m1(b3 >> 4),   float(xb.w), dot);
        acc = fma(s, dot, acc);
    }
    for (uint g = full_blocks * 8u; g < n_groups; ++g) {
        if (lane < 16u) {
            const float s = v4_ue8m0(s_row[g]);
            const uint8_t byte = W_row[g * (kV4FP4GroupSize / 2) + lane];
            const float x0 = float(x[g * kV4FP4GroupSize + lane * 2u]);
            const float x1 = float(x[g * kV4FP4GroupSize + lane * 2u + 1u]);
            float dot = fma(v4_e2m1(uint(byte & 0x0Fu)), x0, 0.0f);
            dot = fma(v4_e2m1(uint(byte >> 4)), x1, dot);
            acc = fma(s, dot, acc);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// ----------------------------------------------------------------------------
// dequant_fp8_e4m3_gemv_simd
//
// y[m] = sum_n W[m,n] * x[n] over FP8 e4m3 weights with 128x128 block scales.
// One SIMD group per row, eight rows per threadgroup. Each block iteration
// covers exactly one 128-column scale block: each lane loads one uchar4 (4
// e4m3 codes) and one half4, accumulates the 128-wide dot in FP32, then one
// exact power-of-two scale multiply per block.
//
// Requires N % 128 == 0 and a 4-byte-aligned weights base (wrapper-enforced).
// M is unconstrained: the grid row index is row/128 with ceil(M/128) rows.
// ----------------------------------------------------------------------------
kernel void dequant_fp8_e4m3_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],   // [M, N] e4m3
    device const uint8_t* scales [[buffer(1)]],   // [ceil(M/128), N/128] ue8m0
    device const half*    x      [[buffer(2)]],
    device half*          y      [[buffer(3)]],
    constant uint&        M      [[buffer(4)]],
    constant uint&        N      [[buffer(5)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kV4RowsPerTG + sg_idx;
    if (row >= M) return;
    const uint grid_w = N / kV4FP8BlockSize;
    device const uint8_t* W_row = W + uint(row) * N;
    device const uint8_t* s_row = scales + uint(row / kV4FP8BlockSize) * grid_w;

    float acc = 0.0f;
    for (uint blk = 0; blk < grid_w; ++blk) {
        const uint elem = blk * kV4FP8BlockSize + lane * 4u;
        const uchar4 q = *((device const uchar4*)(W_row + elem));
        const half4 xv = *((device const half4*)(x + elem));
        float dot = 0.0f;
        dot = fma(v4_e4m3(q.x), float(xv.x), dot);
        dot = fma(v4_e4m3(q.y), float(xv.y), dot);
        dot = fma(v4_e4m3(q.z), float(xv.z), dot);
        dot = fma(v4_e4m3(q.w), float(xv.w), dot);
        acc = fma(v4_ue8m0(s_row[blk]), dot, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// ----------------------------------------------------------------------------
// embed_lookup_fp8
//
// Untied FP8 embedding table lookup: out[i] = e4m3(table[token, i]) *
// block_scale * out_scale. One thread per output element. Requires D % 128 == 0.
// ----------------------------------------------------------------------------
kernel void embed_lookup_fp8(
    device const uint8_t* table     [[buffer(0)]],   // [V, D] e4m3
    device const uint8_t* scales    [[buffer(1)]],   // [ceil(V/128), D/128] ue8m0
    device half*          out       [[buffer(2)]],   // [D] FP16
    constant uint&        token_id  [[buffer(3)]],
    constant uint&        D         [[buffer(4)]],
    constant float&       out_scale [[buffer(5)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint grid_w = D / kV4FP8BlockSize;
    const uint8_t q = table[uint(token_id) * D + gid];
    const float s = v4_ue8m0(
        scales[uint(token_id / kV4FP8BlockSize) * grid_w + gid / kV4FP8BlockSize]);
    out[gid] = half(v4_e4m3(q) * s * out_scale);
}
