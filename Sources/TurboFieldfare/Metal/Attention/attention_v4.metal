#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention_v4 — DeepSeek V4-Flash decode-first attention kernels (V4F-03).
//
// Separate module from attention.metal (compiled standalone by V4Attention)
// so the production Gemma library is untouched.
//
// Geometry (Flash): head_dim 512, 64 query heads, shared-KV MQA (K == V),
// trailing 64 RoPE dims, 128-token uncompressed window branch, per-head
// attention sinks (denominator-only, NOT scaled, NOT in the running max),
// CSA compressed entries stored split-FP8 (7 x 64-dim e4m3 blocks with
// ue8m0 scales) + FP16 rope dims; window ring stored FP16.
//
// Kernel inventory:
//   v4_mqa_partial        split-KV partial over [sparse gather | window ring]
//                         in ONE online-softmax recurrence (sinks fold in at
//                         the combine). nSparse == 0 gives the ratio-0
//                         sliding-window MQA path (layers 0/1/42).
//   v4_sink_combine       partial merge + per-head sink in the denominator.
//   v4_indexer_score      CSA lightning indexer scores (ReLU-then-weighted
//                         -sum over heads, no softmax).
//   v4_topk_chunk         chunked bitonic top-512 selection (ping-pong).
//   v4_iota               dense gather list fill (HCA: all compressed
//                         entries, no indexer).
//   v4_csa_compress_group CSA pooling: overlapped channel-split softmax
//                         pooling + RMSNorm + group-start partial RoPE +
//                         FP8-split quantize, one threadgroup per entry.
// ============================================================================

constant constexpr uint kV4Threads       = 256;
constant constexpr uint kV4MaxHeadDim    = 512;
constant constexpr uint kV4MaxSimdGroups = 8;
constant constexpr uint kV4NonRopeDim    = 448;
constant constexpr uint kV4RopeDim       = 64;
constant constexpr uint kV4FP8Blocks     = 7;      // 448 / 64
constant constexpr uint kV4ScaleStride   = 8;      // 7 ue8m0 bytes + pad
constant constexpr uint kV4TopKChunk     = 2048;   // bitonic sort width
constant constexpr uint kV4TopK          = 512;

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

inline float v4_block_reduce_sum(float v,
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

inline float v4_softmax_exp(float x) { return fast::exp(x); }

/// e4m3 decode (bias 7; NaN payload never appears in the cache).
inline float v4_e4m3_decode(uchar b) {
    const uint e = (uint(b) >> 3) & 0xFu;
    const uint m = uint(b & 0x7u);
    float mag;
    if (e == 0u) {
        mag = float(m) * 0x1p-9f;
    } else {
        mag = (1.0f + float(m) * 0.125f) * exp2(float(int(e) - 7));
    }
    return (b & 0x80u) != 0u ? -mag : mag;
}

/// Round-to-nearest-even e4m3 encode, saturating at +/-448.
inline uchar v4_e4m3_encode(float x) {
    uchar sign = x < 0.0f ? 0x80u : 0x00u;
    float ax = min(fabs(x), 448.0f);
    if (ax < 0x1p-6f) {
        uint m = uint(rint(ax * 512.0f));          // subnormal grid m * 2^-9
        if (m >= 8u) { return sign | 0x08u; }
        return sign | uchar(m);
    }
    int e = int(floor(log2(ax)));
    float mant = ax * exp2(float(-e)) - 1.0f;      // in [0, 1)
    uint m = uint(rint(mant * 8.0f));
    if (m == 8u) { m = 0u; e += 1; }
    if (e > 8) { return sign | 0x7Eu; }            // clamp to 448
    return sign | uchar((uint(e + 7) << 3) | m);
}

inline uchar v4_ue8m0_encode(float scale) {
    int e = int(ceil(log2(scale)));
    e = clamp(e, -127, 128);
    return uchar(e + 127);
}

inline float v4_ue8m0_decode(uchar b) {
    return exp2(float(int(b) - 127));
}

/// DeepSeek YaRN-adjusted frequency for rope pair `pair` (of 32, rope dim
/// 64), matching the reference precompute_freqs_cis: correction range from
/// beta_fast/beta_slow against original_seq_len, linear ramp, blend of
/// freq/factor and freq.
inline float v4_yarn_freq(uint pair,
                          float theta,
                          float factor,
                          float orig_seq_len,
                          float beta_fast,
                          float beta_slow) {
    const float dim = 64.0f;
    const float log_base = log(theta);
    const float freq = pow(theta, -2.0f * float(pair) / dim);
    const float pi2 = 2.0f * M_PI_F;
    const float low_rot  = dim * log(orig_seq_len / (beta_fast * pi2)) / (2.0f * log_base);
    const float high_rot = dim * log(orig_seq_len / (beta_slow * pi2)) / (2.0f * log_base);
    float low  = max(floor(low_rot), 0.0f);
    float high = min(ceil(high_rot), dim - 1.0f);
    if (high == low) { high = low + 0.001f; }
    const float ramp = clamp((float(pair) - low) / (high - low), 0.0f, 1.0f);
    const float smooth = 1.0f - ramp;
    return (freq / factor) * (1.0f - smooth) + freq * smooth;
}

// ============================================================================
// v4_mqa_partial — merged sparse-gather + window-ring MQA partial.
//
// Virtual entry list: [0, n_sparse) indexes `gather` (compressed entries in
// the split FP8 store; -1 => skip), [n_sparse, n_sparse + n_window) indexes
// the FP16 window ring directly by slot. One online-softmax recurrence
// walks the chunk's slice of that list; sinks fold in at the combine.
//
// Grid: num_q_heads * num_chunks threadgroups, kV4Threads threads.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kV4Threads)]]
void v4_mqa_partial(
    device const half*  Q             [[buffer(0)]],   // [num_q_heads, 512]
    device const int*   gather        [[buffer(1)]],   // [n_sparse] compressed entry ids
    device const uchar* c_values      [[buffer(2)]],   // [n_entries * 448] e4m3
    device const uchar* c_scales      [[buffer(3)]],   // [n_entries * 8] ue8m0
    device const half*  c_rope        [[buffer(4)]],   // [n_entries * 64] fp16
    device const half*  window_k      [[buffer(5)]],   // [window, 512] fp16 ring
    device       float* m_out         [[buffer(6)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(7)]],
    device       float* o_out         [[buffer(8)]],   // [num_q_heads * num_chunks * 512]
    constant     uint&  num_q_heads   [[buffer(9)]],
    constant     uint&  n_sparse      [[buffer(10)]],
    constant     uint&  n_window      [[buffer(11)]],
    constant     uint&  chunk_len     [[buffer(12)]],
    constant     uint&  num_chunks    [[buffer(13)]],
    constant     float& scale         [[buffer(14)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    constexpr uint HD = kV4MaxHeadDim;
    threadgroup float q_smem[HD];
    threadgroup float reduce_scratch[kV4MaxSimdGroups];
    threadgroup float bcast;

    const uint q_head = tg_id / num_chunks;
    const uint chunk  = tg_id % num_chunks;
    const uint total  = n_sparse + n_window;
    const uint e_start = chunk * chunk_len;
    uint e_end = e_start + chunk_len;
    if (e_end > total) { e_end = total; }

    device const half* q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) { q_smem[i] = float(q_row[i]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (HD + kV4Threads - 1) / kV4Threads;  // 2
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint e = e_start; e < e_end; ++e) {
        float s;
        // Fetch K (== V) per thread-strided dim, computing the dot partial.
        float partial = 0.0f;
        if (e < n_sparse) {
            const int g = gather[e];
            if (g < 0) { continue; }     // padding slot: zero KV, -inf score
            device const uchar* vals = c_values + uint(g) * kV4NonRopeDim;
            device const uchar* scs  = c_scales + uint(g) * kV4ScaleStride;
            device const half*  rpe  = c_rope + uint(g) * kV4RopeDim;
            // Per-thread element mapping MUST match the writeback loop
            // (full HD stride), so branch per element instead of splitting
            // into nonrope/rope loops with divergent trip counts.
            for (uint i = lid; i < HD; i += lsize) {
                const float kv = (i < kV4NonRopeDim)
                    ? v4_e4m3_decode(vals[i]) * v4_ue8m0_decode(scs[i >> 6])
                    : float(rpe[i - kV4NonRopeDim]);
                partial = fma(q_smem[i], kv, partial);
            }
            s = v4_block_reduce_sum(partial, simd_lane_id, simd_group_id,
                                    simdgroups, reduce_scratch, &bcast) * scale;

            const float m_new = max(m_run, s);
            const float alpha = v4_softmax_exp(m_run - m_new);
            const float p_exp = v4_softmax_exp(s - m_new);
            d_run = d_run * alpha + p_exp;
            uint slot = 0;
            for (uint i = lid; i < HD; i += lsize) {
                const float kv = (i < kV4NonRopeDim)
                    ? v4_e4m3_decode(vals[i]) * v4_ue8m0_decode(scs[i >> 6])
                    : float(rpe[i - kV4NonRopeDim]);
                o_local[slot] = o_local[slot] * alpha + p_exp * kv;
                slot += 1;
            }
            m_run = m_new;
        } else {
            const uint w = e - n_sparse;
            device const half* k_row = window_k + w * HD;
            for (uint i = lid; i < HD; i += lsize) {
                partial = fma(q_smem[i], float(k_row[i]), partial);
            }
            s = v4_block_reduce_sum(partial, simd_lane_id, simd_group_id,
                                    simdgroups, reduce_scratch, &bcast) * scale;

            const float m_new = max(m_run, s);
            const float alpha = v4_softmax_exp(m_run - m_new);
            const float p_exp = v4_softmax_exp(s - m_new);
            d_run = d_run * alpha + p_exp;
            uint slot = 0;
            for (uint i = lid; i < HD; i += lsize) {
                o_local[slot] = o_local[slot] * alpha + p_exp * float(k_row[i]);
                slot += 1;
            }
            m_run = m_new;
        }
    }

    const uint base = uint(q_head) * num_chunks + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// ============================================================================
// v4_sink_combine — merge partials and fold the per-head sink into the
// denominator only: D = sum_c d_c e^{m_c - m*} + e^{sink[h] - m*}. The sink
// is NOT multiplied by softmax scale and does NOT enter the running max
// (recon ambiguity #3). Grid: num_q_heads threadgroups.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kV4Threads)]]
void v4_sink_combine(
    device const float* m_in         [[buffer(0)]],    // [num_q_heads * num_chunks]
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]],    // [num_q_heads * num_chunks * 512]
    device const float* sinks        [[buffer(3)]],    // [num_q_heads] fp32
    device       half*  out          [[buffer(4)]],    // [num_q_heads * 512]
    constant     uint&  num_chunks   [[buffer(5)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]
) {
    constexpr uint HD = kV4MaxHeadDim;
    const uint q_head = tg_id;
    device const float* m_row  = m_in + uint(q_head) * num_chunks;
    device const float* d_row  = d_in + uint(q_head) * num_chunks;
    device const float* o_base = o_in + uint(q_head) * num_chunks * HD;

    float m_glob = -INFINITY;
    for (uint c = 0; c < num_chunks; ++c) { m_glob = max(m_glob, m_row[c]); }
    if (m_glob == -INFINITY) { m_glob = 0.0f; }   // sink-only guard

    float D = v4_softmax_exp(sinks[q_head] - m_glob);
    for (uint c = 0; c < num_chunks; ++c) {
        D += d_row[c] * v4_softmax_exp(m_row[c] - m_glob);
    }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device half* out_row = out + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < num_chunks; ++c) {
            acc += o_base[c * HD + i] * v4_softmax_exp(m_row[c] - m_glob);
        }
        out_row[i] = half(acc * inv_d);
    }
}

// ============================================================================
// v4_indexer_score — CSA lightning indexer decode scores.
//
// score[b] = sum_h weights[h] * relu(dot(index_q[h], index_kv[b]))
// ReLU-then-weighted-sum over heads (NOT softmax). index_q is [64, 128]
// fp16, index_kv is [n_blocks, 128] fp16, weights [64] fp32 (pre-scaled by
// softmax_scale * n_heads^-0.5 write-side). Grid: n_blocks threadgroups.
// Causality at decode is handled by the caller bounding n_blocks to the
// visible groups; no mask needed here.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kV4Threads)]]
void v4_indexer_score(
    device const half*  index_q       [[buffer(0)]],   // [64, 128]
    device const half*  index_kv      [[buffer(1)]],   // [n_blocks, 128]
    device const float* weights       [[buffer(2)]],   // [64]
    device       float* scores        [[buffer(3)]],   // [n_blocks]
    constant     uint&  num_heads     [[buffer(4)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    constexpr uint ID = 128;
    threadgroup float reduce_scratch[kV4MaxSimdGroups];
    threadgroup float bcast;
    threadgroup float acc;

    if (lid == 0) { acc = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half* kv_row = index_kv + uint(tg_id) * ID;
    for (uint h = 0; h < num_heads; ++h) {
        device const half* q_row = index_q + h * ID;
        float partial = 0.0f;
        for (uint i = lid; i < ID; i += lsize) {
            partial = fma(float(q_row[i]), float(kv_row[i]), partial);
        }
        const float dot = v4_block_reduce_sum(partial, simd_lane_id,
                                              simd_group_id, simdgroups,
                                              reduce_scratch, &bcast);
        if (lid == 0) { acc += weights[h] * max(dot, 0.0f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) { scores[tg_id] = acc; }
}

// ============================================================================
// v4_topk_chunk — one threadgroup per 2048-candidate chunk: bitonic sort
// descending on (score, index) and emit the chunk's top kV4TopK. The Swift
// wrapper ping-pongs until a single chunk remains, whose first
// min(kV4TopK, n) entries are the selection. Ties break toward the lower
// block index (matches torch.topk ordering closely enough for the recall
// gate; distinct-score tests are exact).
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(1024)]]
void v4_topk_chunk(
    device const float* in_scores     [[buffer(0)]],   // [n]
    device const int*   in_index      [[buffer(1)]],   // [n] (ignored if implicit)
    device       float* out_scores    [[buffer(2)]],   // [n_chunks * kV4TopK]
    device       int*   out_index     [[buffer(3)]],   // [n_chunks * kV4TopK]
    constant     uint&  n             [[buffer(4)]],
    constant     uint&  index_implicit [[buffer(5)]],  // 1 => index = candidate id
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]
) {
    constexpr uint N = kV4TopKChunk;
    threadgroup float s_key[N];
    threadgroup int   s_idx[N];

    const uint base = tg_id * N;
    for (uint i = lid; i < N; i += lsize) {
        const uint g = base + i;
        if (g < n) {
            s_key[i] = in_scores[g];
            s_idx[i] = index_implicit != 0u ? int(g) : in_index[g];
        } else {
            s_key[i] = -INFINITY;
            s_idx[i] = -1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort, descending.
    for (uint k = 2; k <= N; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = lid; i < N; i += lsize) {
                const uint ixj = i ^ j;
                if (ixj > i) {
                    const bool desc = ((i & k) == 0u);
                    const float ki = s_key[i], kj = s_key[ixj];
                    const int   ii = s_idx[i], ij = s_idx[ixj];
                    const bool i_first = (ki > kj) || (ki == kj && ii < ij);
                    if (i_first != desc) {
                        s_key[i] = kj; s_key[ixj] = ki;
                        s_idx[i] = ij; s_idx[ixj] = ii;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    device float* os = out_scores + tg_id * kV4TopK;
    device int*   oi = out_index + tg_id * kV4TopK;
    for (uint i = lid; i < kV4TopK; i += lsize) {
        os[i] = s_key[i];
        oi[i] = s_idx[i];
    }
}

// ============================================================================
// v4_iota — dense gather list for HCA decode (all visible compressed
// entries, no indexer). Grid: threads over n.
// ============================================================================

kernel void v4_iota(
    device       int*   out           [[buffer(0)]],
    constant     uint&  n             [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) { out[gid] = int(gid); }
}

// ============================================================================
// v4_csa_compress_group — CSA pooling for ONE compressed entry (decode-time
// group flush; also usable per-group in prefill).
//
// Overlap channel-split pooling (recon note #7): the 1024-dim projections
// are split in half; the pooled 8-row window takes the PREVIOUS group's 4
// tokens on channels [0,512) and the CURRENT group's 4 tokens on channels
// [512,1024), for both the value (wkv) and gate (wgate) projections. ape is
// [4, 1024] and splits identically. Then:
//   sm[j, d] = softmax_j(gate[j, d] + ape[j, d])   (softmax over the 8 rows)
//   x[d]     = sum_j sm[j, d] * kv[j, d]
//   RMSNorm(512, gamma), partial RoPE on the trailing 64 dims at the
//   group-start position (compress theta + YaRN), FP8-split quantize the
//   non-rope dims, FP16 write for the rope dims.
//
// One threadgroup, 512 threads (thread d owns channel d).
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(512)]]
void v4_csa_compress_group(
    device const float* prev_kv       [[buffer(0)]],   // [4, 1024] fp32
    device const float* cur_kv        [[buffer(1)]],   // [4, 1024] fp32
    device const float* prev_gate     [[buffer(2)]],   // [4, 1024] fp32
    device const float* cur_gate      [[buffer(3)]],   // [4, 1024] fp32
    device const float* ape           [[buffer(4)]],   // [4, 1024] fp32
    device const float* gamma         [[buffer(5)]],   // [512] fp32
    device       uchar* out_values    [[buffer(6)]],   // [448] e4m3
    device       uchar* out_scales    [[buffer(7)]],   // [8] ue8m0 (7 used)
    device       half*  out_rope      [[buffer(8)]],   // [64] fp16
    constant     uint&  rope_position [[buffer(9)]],
    constant     float& rope_theta    [[buffer(10)]],
    constant     float& yarn_factor   [[buffer(11)]],
    constant     float& orig_seq_len  [[buffer(12)]],
    constant     float& beta_fast     [[buffer(13)]],
    constant     float& beta_slow     [[buffer(14)]],
    constant     uint&  use_yarn      [[buffer(15)]],
    constant     float& norm_eps      [[buffer(16)]],
    uint lid             [[thread_position_in_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    constexpr uint HD = 512;
    threadgroup float x[HD];
    // 512 threads = 16 simdgroups (vs 8 in the 256-thread kernels).
    threadgroup float reduce_scratch[16];
    threadgroup float bcast;
    threadgroup float block_scales[kV4FP8Blocks];

    const uint d = lid;

    // Pooled 8-row window for this channel (rows 0..3: prev group, first
    // half channels; rows 4..7: current group, second half channels).
    float kv_rows[8], gate_rows[8];
    for (uint j = 0; j < 4; ++j) {
        kv_rows[j]     = prev_kv[j * 1024 + d];
        gate_rows[j]   = prev_gate[j * 1024 + d] + ape[j * 1024 + d];
        kv_rows[4 + j] = cur_kv[j * 1024 + HD + d];
        gate_rows[4 + j] = cur_gate[j * 1024 + HD + d] + ape[j * 1024 + HD + d];
    }

    // Softmax over the 8 rows (max-subtracted).
    float mx = gate_rows[0];
    for (uint j = 1; j < 8; ++j) { mx = max(mx, gate_rows[j]); }
    float sum = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        gate_rows[j] = v4_softmax_exp(gate_rows[j] - mx);
        sum += gate_rows[j];
    }
    float acc = 0.0f;
    for (uint j = 0; j < 8; ++j) { acc = fma(gate_rows[j], kv_rows[j], acc); }
    acc /= sum;

    // RMSNorm over the 512 channels.
    const float sq = v4_block_reduce_sum(acc * acc, simd_lane_id,
                                         simd_group_id, simdgroups,
                                         reduce_scratch, &bcast);
    x[d] = acc * rsqrt(sq / float(HD) + norm_eps) * gamma[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Partial RoPE on the trailing 64 dims, interleaved (adjacent-pair)
    // convention matching the reference: pair i covers slice elements
    // (2i, 2i+1), i.e. channels (448+2i, 448+2i+1).
    if (d >= kV4NonRopeDim && d < kV4NonRopeDim + kV4RopeDim / 2) {
        const uint i = d - kV4NonRopeDim;
        const uint i0 = kV4NonRopeDim + 2u * i;
        const uint i1 = i0 + 1u;
        const float freq = use_yarn != 0u
            ? v4_yarn_freq(i, rope_theta, yarn_factor, orig_seq_len,
                           beta_fast, beta_slow)
            : pow(rope_theta, -2.0f * float(i) / float(kV4RopeDim));
        const float angle = float(rope_position) * freq;
        const float cs = cos(angle);
        const float sn = sin(angle);
        const float x0 = x[i0];
        const float x1 = x[i1];
        x[i0] = x0 * cs - x1 * sn;
        x[i1] = x0 * sn + x1 * cs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FP8 quantize the non-rope dims: 7 blocks of 64 channels. Thread b of
    // the first 7 computes its block's ue8m0 scale serially (cheap; this
    // kernel runs once per 4 tokens).
    if (d < kV4FP8Blocks) {
        float amax = 0.0f;
        for (uint i = 0; i < 64; ++i) {
            amax = max(amax, fabs(x[d * 64 + i]));
        }
        block_scales[d] = v4_ue8m0_decode(
            v4_ue8m0_encode(max(amax, 1e-4f) / 448.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d < kV4NonRopeDim) {
        const float scale = block_scales[d >> 6];
        out_values[d] = v4_e4m3_encode(x[d] / scale);
        if ((d & 63u) == 0u) {
            out_scales[d >> 6] = v4_ue8m0_encode(scale);
        }
    } else {
        out_rope[d - kV4NonRopeDim] = half(x[d]);
    }
}
