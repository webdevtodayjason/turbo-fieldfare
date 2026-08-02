#include <metal_stdlib>
using namespace metal;

// ============================================================================
// prefill_proj_v4 — DeepSeek V4-Flash chunked-prefill projection and window
// attention kernels (V4F-06c, work-order A2).
//
// Batched (row-parallel) variants of the decode projection path plus the
// window-branch prefill attention. All symbols are `v4pp_`-prefixed; the
// module compiles into the shared `MetalContext` library and collides with
// nothing (`v4_` decode, `v4b_` wave-2, `v4c_` indexer, `v4pf_` boundary).
//
// Kernel inventory:
//   v4pp_fp8_block_gemm     batched FP8 e4m3 + ue8m0 128x128 block GEMM:
//                           y[r, m] = sum_n W[m, n] * x[r, n]. Covers wq_a
//                           (M=1024, N=4096), wq_b (M=32768, N=1024), wkv
//                           (M=512, N=4096), indexer wq_b (M=8192, N=1024),
//                           and the o-proj up stage wo_b (M=4096, N=8192).
//   v4pp_bf16_gemm          batched BF16 GEMM, same contraction, no scales:
//                           compressor projections (M=1024 CSA / 512 HCA,
//                           N=4096) and router gate (M=256, N=4096).
//   v4pp_window_ring_write  batched KV rows -> window ring slots
//                           ((startPosition + i) % window), matching
//                           CompressedKVCacheManager.windowSlot addressing.
//   v4pp_window_mqa_prefill batched causal window MQA: 64 q heads x 512,
//                           per-head sinks in the DENOMINATOR only (not the
//                           max), row i attends ring[0..prefix) + chunk[0..i].
//                           FP16 in/out.
//   v4pp_fp8_grouped_gemm   grouped o-proj down stage: 8 groups, y[r, g*R+i]
//                           = dot(W[g*R+i, :], x[r, g*D .. +D]). NOT a flat
//                           GEMM (design-note pitfall, see 676731b).
//
// Correctness-over-speed v1: every GEMM stages the activation tile in
// threadgroup memory (2048 fp32 columns per iteration, 8 KB) and uses one
// SIMD group per output row, mirroring the decode GEMV geometry. All GEMMs
// require N % 128 == 0 (FP8) or N % 4 == 0 (BF16); M must be a multiple of
// 8 (every projection above is). FP8 weights offsets must be 4-byte aligned
// (wrapper-enforced) for the uchar4 fast loads.
// ============================================================================

constant constexpr uint kV4PPRowsPerTG = 8;      // one SIMD group per output row
constant constexpr uint kV4PPNChunk    = 2048;   // staged activation columns
constant constexpr uint kV4PPThreads   = 256;

static inline float v4pp_ue8m0(uint8_t b) {
    // 2^(b - 127); exact for every byte.
    return exp2(float(int(b) - 127));
}

static inline float v4pp_e4m3(uint8_t q) {
    const uint e = (uint(q) >> 3) & 0xFu;
    const uint m = uint(q) & 0x7u;
    float mag;
    if (e == 0u) {
        mag = float(m) * 0x1.0p-9f;                      // subnormal step 2^-9
    } else {
        mag = float(8u + m) * exp2(float(int(e) - 10));  // (8+m) * 2^(e-10)
    }
    return (q & 0x80u) ? -mag : mag;
}

static inline float v4pp_bf16(uint16_t b) {
    return as_type<float>(uint(b) << 16);
}

static inline float v4pp_block_reduce_sum(float v,
                                          uint simd_lane_id,
                                          uint simd_group_id,
                                          uint simdgroups,
                                          threadgroup float* scratch,
                                          threadgroup float* bcast) {
    float s = simd_sum(v);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { *bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *bcast;
}

// ----------------------------------------------------------------------------
// v4pp_fp8_block_gemm — y[r, m] = sum_n W[m, n] * x[r, n].
//
// W is [M, N] e4m3 row-major; scales are the [ceil(M/128), N/128] ue8m0 grid
// row-major (same layout as dequant_v4). x is [rows, N] fp16. Output buffer
// holds [rows, M]; outFP16 selects the fp16 or fp32 view of it (the wrapper
// binds the same buffer to both indices).
//
// Grid: rows * ceil(M/8) threadgroups of kV4PPThreads threads; threadgroup
// (row, tile) computes y[row, tile*8 .. tile*8+8). Requires N % 128 == 0.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(kV4PPThreads)]]
void v4pp_fp8_block_gemm(
    device const uint8_t* W       [[buffer(0)]],
    device const uint8_t* scales  [[buffer(1)]],
    device const half*    X       [[buffer(2)]],
    device float*         Yf      [[buffer(3)]],   // [rows, M] when outFP16 == 0
    device half*          Yh      [[buffer(4)]],   // [rows, M] when outFP16 == 1
    constant uint&        M       [[buffer(5)]],
    constant uint&        N       [[buffer(6)]],
    constant uint&        outFP16 [[buffer(7)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    const uint mTiles = (M + kV4PPRowsPerTG - 1u) / kV4PPRowsPerTG;
    const uint row = tg_id / mTiles;
    const uint m = (tg_id % mTiles) * kV4PPRowsPerTG + sg;

    threadgroup float xs[kV4PPNChunk];
    const uint grid_w = N / 128u;
    float acc = 0.0f;

    for (uint base = 0; base < N; base += kV4PPNChunk) {
        const uint width = min((uint)kV4PPNChunk, N - base);   // multiple of 128
        for (uint i = lid; i < width; i += lsize) {
            xs[i] = float(X[uint64_t(row) * N + base + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (m < M) {
            device const uint8_t* W_row = W + uint64_t(m) * N;
            device const uint8_t* s_row = scales + uint64_t(m / 128u) * grid_w;
            const uint blkEnd = (base + width) / 128u;
            for (uint blk = base / 128u; blk < blkEnd; ++blk) {
                const uint col = blk * 128u + lane * 4u;
                const uchar4 q = *((device const uchar4*)(W_row + col));
                const uint off = col - base;
                float dot = 0.0f;
                dot = fma(v4pp_e4m3(q.x), xs[off],      dot);
                dot = fma(v4pp_e4m3(q.y), xs[off + 1u], dot);
                dot = fma(v4pp_e4m3(q.z), xs[off + 2u], dot);
                dot = fma(v4pp_e4m3(q.w), xs[off + 3u], dot);
                acc = fma(v4pp_ue8m0(s_row[blk]), dot, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc = simd_sum(acc);
    if (m < M && lane == 0u) {
        const uint64_t idx = uint64_t(row) * M + m;
        if (outFP16 != 0u) { Yh[idx] = half(acc); } else { Yf[idx] = acc; }
    }
}

// ----------------------------------------------------------------------------
// v4pp_bf16_gemm — y[r, m] = sum_n W[m, n] * x[r, n] over bf16 weights
// (compressor CSA/HCA projections, router gate). No scale grid. Same grid
// geometry and output-view contract as v4pp_fp8_block_gemm. Requires
// N % 4 == 0 and a 4-byte-aligned weights base (wrapper-enforced).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(kV4PPThreads)]]
void v4pp_bf16_gemm(
    device const uint16_t* W      [[buffer(0)]],   // [M, N] bf16 bit patterns
    device const half*     X      [[buffer(1)]],
    device float*          Yf     [[buffer(2)]],
    device half*           Yh     [[buffer(3)]],
    constant uint&         M      [[buffer(4)]],
    constant uint&         N      [[buffer(5)]],
    constant uint&         outFP16 [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    const uint mTiles = (M + kV4PPRowsPerTG - 1u) / kV4PPRowsPerTG;
    const uint row = tg_id / mTiles;
    const uint m = (tg_id % mTiles) * kV4PPRowsPerTG + sg;

    threadgroup float xs[kV4PPNChunk];
    float acc = 0.0f;

    for (uint base = 0; base < N; base += kV4PPNChunk) {
        const uint width = min((uint)kV4PPNChunk, N - base);   // multiple of 4
        for (uint i = lid; i < width; i += lsize) {
            xs[i] = float(X[uint64_t(row) * N + base + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (m < M) {
            device const uint16_t* W_row = W + uint64_t(m) * N;
            for (uint off = lane * 4u; off < width; off += 128u) {
                const ushort4 w = *((device const ushort4*)(W_row + base + off));
                float dot = 0.0f;
                dot = fma(v4pp_bf16(w.x), xs[off],      dot);
                dot = fma(v4pp_bf16(w.y), xs[off + 1u], dot);
                dot = fma(v4pp_bf16(w.z), xs[off + 2u], dot);
                dot = fma(v4pp_bf16(w.w), xs[off + 3u], dot);
                acc += dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc = simd_sum(acc);
    if (m < M && lane == 0u) {
        const uint64_t idx = uint64_t(row) * M + m;
        if (outFP16 != 0u) { Yh[idx] = half(acc); } else { Yf[idx] = acc; }
    }
}

// ----------------------------------------------------------------------------
// v4pp_window_ring_write — land `rows` KV rows at ring slots
// (startPosition + i) % window, matching
// CompressedKVCacheManager.windowPhysicalSlot / windowSlot addressing.
// One thread per element.
// ----------------------------------------------------------------------------
kernel void v4pp_window_ring_write(
    device const half* src      [[buffer(0)]],   // [rows, headDim]
    device half*       ring     [[buffer(1)]],   // [window, headDim]
    constant uint&     startPos [[buffer(2)]],   // absolute position of row 0
    constant uint&     rows     [[buffer(3)]],
    constant uint&     headDim  [[buffer(4)]],
    constant uint&     window   [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows * headDim) { return; }
    const uint row = gid / headDim;
    const uint d = gid % headDim;
    const uint slot = (startPos + row) % window;
    ring[uint64_t(slot) * headDim + d] = src[uint64_t(row) * headDim + d];
}

// ----------------------------------------------------------------------------
// v4pp_window_mqa_prefill — batched causal window MQA with per-head sinks.
//
// Row i attends exactly ring[0 .. prefix) followed by chunk rows [0 .. i]
// (self included): the ring prefix holds the tokens already flushed by
// earlier chunks at slots 0..prefix-1, and the current chunk's KV stays in
// its staging rows so future rows are never visible. K == V (shared MQA),
// head_dim 512, 64 query heads. Sinks enter the DENOMINATOR only and are NOT
// in the running max, matching v4_sink_combine semantics (recon ambiguity
// #3): out = acc / (d_run + exp(sink[h] - m_run)).
//
// Grid: rows * 64 threadgroups of kV4PPThreads threads; tg = (row, head).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(kV4PPThreads)]]
void v4pp_window_mqa_prefill(
    device const half*  Q       [[buffer(0)]],   // [rows, 64, 512]
    device const half*  ringK   [[buffer(1)]],   // [window, 512], prefix at slots [0, prefix)
    device const half*  chunkKV [[buffer(2)]],   // [rows, 512]
    device const float* sinks   [[buffer(3)]],   // [64] fp32
    device half*        out     [[buffer(4)]],   // [rows, 64, 512]
    constant uint&      prefix  [[buffer(5)]],
    constant float&     scale   [[buffer(6)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    constexpr uint HD = 512;
    threadgroup float q_smem[HD];
    threadgroup float reduce_scratch[8];
    threadgroup float bcast;

    const uint head = tg_id & 63u;
    const uint row = tg_id >> 6u;

    device const half* q_row = Q + (uint64_t(row) * 64u + head) * HD;
    for (uint i = lid; i < HD; i += lsize) { q_smem[i] = float(q_row[i]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (HD + kV4PPThreads - 1) / kV4PPThreads;  // 2
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    const uint total = prefix + row + 1u;
    for (uint e = 0; e < total; ++e) {
        device const half* k_row = (e < prefix)
            ? ringK + uint64_t(e) * HD
            : chunkKV + uint64_t(e - prefix) * HD;
        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i], float(k_row[i]), partial);
        }
        const float s = v4pp_block_reduce_sum(partial, simd_lane_id,
                                              simd_group_id, simdgroups,
                                              reduce_scratch, &bcast) * scale;

        const float m_new = max(m_run, s);
        const float alpha = fast::exp(m_run - m_new);
        const float p_exp = fast::exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            o_local[slot] = o_local[slot] * alpha + p_exp * float(k_row[i]);
            slot += 1u;
        }
        m_run = m_new;
    }

    // Denominator-only sink (not in the max, not scaled). total >= 1 always,
    // so m_run is finite; a huge sink drives the output to ~0.
    const float denom = d_run + fast::exp(sinks[head] - m_run);
    const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    device half* out_row = out + (uint64_t(row) * 64u + head) * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        out_row[i] = half(o_local[slot] * inv);
        slot += 1u;
    }
}

// ----------------------------------------------------------------------------
// v4pp_fp8_grouped_gemm — grouped o-proj down stage (design-note pitfall:
// 8 separate group slices, NOT one flat GEMM).
//
// y[r, g*R + i] = sum_d W[g*R + i, d] * x[r, g*D + d]
//
// W is [G*R, D] e4m3 group-major (wo_a viewed as [8, 1024, 4096]) with the
// standard [G*R/128, D/128] ue8m0 grid; x is [rows, G*D] fp16; y is
// [rows, G*R] fp16 (the stage-2 up GEMM reads fp16 activations). Same grid
// geometry as v4pp_fp8_block_gemm. Requires D % 128 == 0, M % 8 == 0.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(kV4PPThreads)]]
void v4pp_fp8_grouped_gemm(
    device const uint8_t* W      [[buffer(0)]],   // [M, D] e4m3, M = G*R
    device const uint8_t* scales [[buffer(1)]],   // [M/128, D/128] ue8m0
    device const half*    X      [[buffer(2)]],   // [rows, (M/R)*D]
    device half*          Y      [[buffer(3)]],   // [rows, M] fp16
    constant uint&        M      [[buffer(4)]],   // G*R (8192)
    constant uint&        D      [[buffer(5)]],   // per-group input dim (4096)
    constant uint&        R      [[buffer(6)]],   // rows per group (1024)
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    const uint mTiles = (M + kV4PPRowsPerTG - 1u) / kV4PPRowsPerTG;
    const uint row = tg_id / mTiles;
    const uint m = (tg_id % mTiles) * kV4PPRowsPerTG + sg;
    // Staging runs on every thread; clamp the group for out-of-range m so
    // the (unused) staged values stay in bounds.
    const uint g = (m < M) ? (m / R) : 0u;
    const uint xStride = (M / R) * D;

    threadgroup float xs[kV4PPNChunk];
    const uint grid_w = D / 128u;
    float acc = 0.0f;

    for (uint base = 0; base < D; base += kV4PPNChunk) {
        const uint width = min((uint)kV4PPNChunk, D - base);
        const uint64_t xBase = uint64_t(row) * xStride + uint64_t(g) * D + base;
        for (uint i = lid; i < width; i += lsize) {
            xs[i] = float(X[xBase + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (m < M) {
            device const uint8_t* W_row = W + uint64_t(m) * D;
            device const uint8_t* s_row = scales + uint64_t(m / 128u) * grid_w;
            const uint blkEnd = (base + width) / 128u;
            for (uint blk = base / 128u; blk < blkEnd; ++blk) {
                const uint col = blk * 128u + lane * 4u;
                const uchar4 q = *((device const uchar4*)(W_row + col));
                const uint off = col - base;
                float dot = 0.0f;
                dot = fma(v4pp_e4m3(q.x), xs[off],      dot);
                dot = fma(v4pp_e4m3(q.y), xs[off + 1u], dot);
                dot = fma(v4pp_e4m3(q.z), xs[off + 2u], dot);
                dot = fma(v4pp_e4m3(q.w), xs[off + 3u], dot);
                acc = fma(v4pp_ue8m0(s_row[blk]), dot, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc = simd_sum(acc);
    if (m < M && lane == 0u) {
        Y[uint64_t(row) * M + m] = half(acc);
    }
}
