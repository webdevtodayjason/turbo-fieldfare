#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention_v4c — DeepSeek V4-Flash decode-graph glue kernels (V4F-04).
// Small connective kernels that the committed V4F-02/03 modules
// (dequant_v4, moe_v4, attention_v4, attention_v4b) do not cover, ported
// from the wave-3 integration draft onto the committed dtype conventions
// (fp32 mHC stream, fp16 hidden, fp32 norm gammas):
//
//   v4c_embed_broadcast            BF16 embedding row gathered and broadcast
//                                  into the 4-stream mHC residual (FP32).
//   v4c_rmsnorm_f32in              RMSNorm with FP32 input + FP32 gamma,
//                                  FP16 out (mHC branch input -> sublayer).
//   v4c_swiglu_act                 clamped SwiGLU for the FP8 shared expert:
//                                  act = silu(min(gate, limit)) *
//                                  clamp(up, +/-limit); gate has no lower
//                                  clamp, per the reference.
//   v4c_scale_f32                  constant scale (indexer per-head weights
//                                  pre-scale).
//   v4c_router_weights_at_indices  hash-layer routing weights (recon §6):
//                                  sqrt(softplus(logits)) gathered at the
//                                  tid2eid indices, L1-normalized, scaled by
//                                  route_scale. No bias (hash layers have
//                                  none); selection is the table's.
//   v4c_indexer_compress_group     CSA lightning-indexer compressed-entry
//                                  flush: overlapped 8-token softmax pooling
//                                  at 128 dims + RMSNorm + group-start
//                                  partial RoPE, FP16 out. The reference's
//                                  Hadamard + FP4 QAT sim on indexer entries
//                                  is NOT applied (parity seam recorded in
//                                  the V4F-03 report; storage is FP16 per
//                                  CompressedKVCacheManager).
// ============================================================================

constant constexpr uint kV4CThreads = 256;

inline float v4c_yarn_freq(uint pair,
                           float theta,
                           float factor,
                           float orig_seq_len,
                           float beta_fast,
                           float beta_slow,
                           uint rope_dim) {
    const float dim = float(rope_dim);
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

// BF16 embedding row gather broadcast into the fp32 hc-stream residual:
// out[s*dim + i] = table[token*dim + i] for s in 0..<streams.
kernel void v4c_embed_broadcast(
    device const bfloat* table   [[buffer(0)]],
    device       float*  out     [[buffer(1)]],
    constant     uint&   token   [[buffer(2)]],
    constant     uint&   dim     [[buffer(3)]],
    constant     uint&   streams [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint flat = streams * dim;
    if (gid >= flat) { return; }
    out[gid] = float(table[uint(token) * dim + (gid % dim)]);
}

// RMSNorm with FP32 input and FP32 gamma, FP16 output (V4 norm gammas are
// fp32 in the checkpoint, recon §7; the mHC branch input arrives fp32 from
// v4b_hc_pre). One threadgroup, decode is a single row:
// out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i].
kernel void v4c_rmsnorm_f32in(
    device const float* x     [[buffer(0)]],
    device const float* w     [[buffer(1)]],
    device       half*  out   [[buffer(2)]],
    constant     uint&  dim   [[buffer(3)]],
    constant     float& eps   [[buffer(4)]],
    uint lid            [[thread_position_in_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reduce_scratch[kV4CThreads / 32];
    threadgroup float bcast;
    float sq = 0.0f;
    for (uint i = lid; i < dim; i += kV4CThreads) {
        const float v = x[i];
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (simd_lane_id == 0) { reduce_scratch[simd_group_id] = sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < kV4CThreads / 32) ? reduce_scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { bcast = rsqrt(t / float(dim) + eps); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rstd = bcast;
    for (uint i = lid; i < dim; i += kV4CThreads) {
        out[i] = half(x[i] * rstd * w[i]);
    }
}

// Clamped SwiGLU for the FP8 shared expert (recon §5):
//   act[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, limit)
kernel void v4c_swiglu_act(
    device const half*  gate    [[buffer(0)]],
    device const half*  up      [[buffer(1)]],
    device       half*  act     [[buffer(2)]],
    constant     uint&  n       [[buffer(3)]],
    constant     float& limit   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    const float g = min(float(gate[gid]), limit);
    const float u = clamp(float(up[gid]), -limit, limit);
    const float sig = 1.0f / (1.0f + fast::exp(-g));
    act[gid] = half(g * sig * u);
}

// buf[i] *= scale (indexer per-head weights pre-scale:
// softmax_scale * n_heads^-0.5, recon §2).
kernel void v4c_scale_f32(
    device       float* buf   [[buffer(0)]],
    constant     float& scale [[buffer(1)]],
    constant     uint&  n     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    buf[gid] *= scale;
}

// Hash-layer routing weights (recon §6): indices come from the fixed
// tid2eid table (uploaded by the caller); weights are still score-based.
//   w_i = sqrt(softplus(logits[idx_i]]));  w_i /= sum_j w_j;  w_i *= route_scale
// No correction bias on hash layers. One thread handles k <= 32.
kernel void v4c_router_weights_at_indices(
    device const float* logits       [[buffer(0)]],   // [num_experts] fp32
    device const int*   indices      [[buffer(1)]],   // [k]
    device       float* out_weights  [[buffer(2)]],   // [k]
    constant     uint&  k            [[buffer(3)]],
    constant     float& route_scale  [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (lid != 0) { return; }
    float w[32];
    float sum = 0.0f;
    const uint count = min(k, 32u);
    for (uint i = 0; i < count; ++i) {
        const float s = logits[indices[i]];
        // sqrt(softplus(s)) with the reference's fp32 math (PyTorch
        // softplus thresholds to linear above 20).
        const float sp = s > 20.0f ? s : fast::log(1.0f + fast::exp(s));
        w[i] = fast::sqrt(max(sp, 0.0f));
        sum += w[i];
    }
    for (uint i = 0; i < count; ++i) {
        out_weights[i] = w[i] / sum * route_scale;
    }
}

// CSA lightning-indexer compressed-entry flush (V4F-04 decode graph).
// 128-dim variant of v4_csa_compress_group: overlapped pooling of the
// previous group's first 128 channels and the current group's second 128
// channels (8 rows), softmax over gate scores + ape, RMSNorm with fp32
// gamma, group-start partial RoPE on the trailing 64 dims, FP16 out.
// Hadamard + FP4 QAT sim of the reference write-side is not applied
// (parity seam; the indexer cache stores FP16).
kernel void v4c_indexer_compress_group(
    device const float* prev_kv       [[buffer(0)]],   // [4, 256] fp32
    device const float* cur_kv        [[buffer(1)]],   // [4, 256] fp32
    device const float* prev_gate     [[buffer(2)]],   // [4, 256] fp32
    device const float* cur_gate      [[buffer(3)]],   // [4, 256] fp32
    device const float* ape           [[buffer(4)]],   // [4, 256] fp32
    device const float* gamma         [[buffer(5)]],   // [128] fp32
    device       half*  out           [[buffer(6)]],   // [128] fp16
    constant     uint&  rope_position [[buffer(7)]],
    constant     float& rope_theta    [[buffer(8)]],
    constant     float& yarn_factor   [[buffer(9)]],
    constant     float& orig_seq_len  [[buffer(10)]],
    constant     float& beta_fast     [[buffer(11)]],
    constant     float& beta_slow     [[buffer(12)]],
    constant     uint&  use_yarn      [[buffer(13)]],
    constant     float& norm_eps      [[buffer(14)]],
    uint lid            [[thread_position_in_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr uint HD = 128;
    threadgroup float x[HD];
    threadgroup float reduce_scratch[kV4CThreads / 32];
    threadgroup float bcast;

    const uint d = lid;

    float kv_rows[8], gate_rows[8];
    if (d < HD) {
        for (uint j = 0; j < 4; ++j) {
            kv_rows[j]     = prev_kv[j * 256 + d];
            gate_rows[j]   = prev_gate[j * 256 + d] + ape[j * 256 + d];
            kv_rows[4 + j] = cur_kv[j * 256 + HD + d];
            gate_rows[4 + j] = cur_gate[j * 256 + HD + d] + ape[j * 256 + HD + d];
        }
        float mx = gate_rows[0];
        for (uint j = 1; j < 8; ++j) { mx = max(mx, gate_rows[j]); }
        float sum = 0.0f;
        for (uint j = 0; j < 8; ++j) {
            gate_rows[j] = fast::exp(gate_rows[j] - mx);
            sum += gate_rows[j];
        }
        float acc = 0.0f;
        for (uint j = 0; j < 8; ++j) { acc = fma(gate_rows[j], kv_rows[j], acc); }
        x[d] = acc / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RMSNorm over the 128 channels (only the first HD threads hold rows).
    float sq = (d < HD) ? x[d] * x[d] : 0.0f;
    sq = simd_sum(sq);
    if (simd_lane_id == 0) { reduce_scratch[simd_group_id] = sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < kV4CThreads / 32) ? reduce_scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < HD) {
        x[d] = x[d] * rsqrt(bcast / float(HD) + norm_eps) * gamma[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Partial RoPE on the trailing 64 dims at the group-start position.
    constexpr uint rope_base = HD - 64;
    if (d >= rope_base && d < rope_base + 32) {
        const uint i = d - rope_base;
        const float freq = use_yarn != 0u
            ? v4c_yarn_freq(i, rope_theta, yarn_factor, orig_seq_len,
                            beta_fast, beta_slow, 64u)
            : pow(rope_theta, -2.0f * float(i) / 64.0f);
        const float angle = float(rope_position) * freq;
        const float cs = cos(angle);
        const float sn = sin(angle);
        const float x0 = x[d];
        const float x1 = x[d + 32];
        x[d]      = x0 * cs - x1 * sn;
        x[d + 32] = x0 * sn + x1 * cs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d < HD) { out[d] = half(x[d]); }
}
