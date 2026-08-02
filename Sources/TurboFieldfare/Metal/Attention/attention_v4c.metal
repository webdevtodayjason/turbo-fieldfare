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
//   v4c_indexer_hadamard_fp4_qat   normalized 128-point WHT followed by
//                                  block-32 e2m1/e8m0 quantize-dequantize.
//   v4c_window_fp8_qat              window-KV non-RoPE FP8 activation
//                                  quantize-dequantize, block size 64.
//   v4c_indexer_compress_group     CSA lightning-indexer compressed-entry
//                                  flush: overlapped 8-token softmax pooling
//                                  at 128 dims + RMSNorm + group-start
//                                  partial RoPE + the same WHT/FP4 QAT.
// ============================================================================

constant constexpr uint kV4CThreads = 256;
constant constexpr uint kV4CIndexDim = 128;
constant constexpr uint kV4CFP4Block = 32;

static inline float v4c_block_reduce_sum(float value,
                                         uint simd_lane_id,
                                         uint simd_group_id,
                                         uint simdgroups,
                                         threadgroup float* scratch,
                                         threadgroup float* broadcast) {
    float sum = simd_sum(value);
    if (simd_lane_id == 0) { scratch[simd_group_id] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float group = simd_lane_id < simdgroups ? scratch[simd_lane_id] : 0.0f;
        group = simd_sum(group);
        if (simd_lane_id == 0) { *broadcast = group; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *broadcast;
}

inline int v4c_log2_ceil_positive(float x) {
    const uint bits = as_type<uint>(x);
    const int exponent = int((bits >> 23) & 0xFFu) - 127;
    return exponent + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
}

inline float v4c_pow2(int exponent) {
    return as_type<float>(uint(exponent + 127) << 23);
}

inline float v4c_bf16_round(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return as_type<float>(bits & 0xFFFF0000u);
}

inline float v4c_fp4_roundtrip(float value, float scale) {
    constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float q = min(fabs(value / scale), 6.0f);
    uint best = 0;
    float best_diff = INFINITY;
    for (uint code = 0; code < 8; ++code) {
        const float diff = fabs(levels[code] - q);
        if (diff < best_diff || (diff == best_diff && (code & 1u) == 0u)) {
            best = code;
            best_diff = diff;
        }
    }
    const float rounded = levels[best] * scale;
    return value < 0.0f ? -rounded : rounded;
}

inline uchar v4c_e4m3_encode(float x) {
    const uchar sign = x < 0.0f ? 0x80u : 0x00u;
    const float ax = min(fabs(x), 448.0f);
    if (ax < 0x1p-6f) {
        const uint mantissa = uint(rint(ax * 512.0f));
        if (mantissa >= 8u) { return sign | 0x08u; }
        return sign | uchar(mantissa);
    }
    int exponent = int(floor(log2(ax)));
    const float fractional = ax * exp2(float(-exponent)) - 1.0f;
    uint mantissa = uint(rint(fractional * 8.0f));
    if (mantissa == 8u) { mantissa = 0u; exponent += 1; }
    if (exponent > 8) { return sign | 0x7Eu; }
    return sign | uchar((uint(exponent + 7) << 3) | mantissa);
}

inline float v4c_e4m3_decode(uchar bits) {
    const uint exponent = (uint(bits) >> 3) & 0xFu;
    const uint mantissa = uint(bits & 0x7u);
    const float magnitude = exponent == 0u
        ? float(mantissa) * 0x1p-9f
        : (1.0f + float(mantissa) * 0.125f) * exp2(float(int(exponent) - 7));
    return (bits & 0x80u) != 0u ? -magnitude : magnitude;
}

// Reference rotate_activation + fp4_act_quant(inplace=True) for one or more
// 128-dim indexer query rows. One threadgroup owns one row.
[[kernel, max_total_threads_per_threadgroup(kV4CIndexDim)]]
kernel void v4c_indexer_hadamard_fp4_qat(
    device half* buf [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= rows) { return; }
    threadgroup float x[kV4CIndexDim];
    threadgroup float scales[kV4CIndexDim / kV4CFP4Block];
    device half* out = buf + row * kV4CIndexDim;
    x[lid] = float(out[lid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr float inv_sqrt_2 = 0.7071067811865475244f;
    for (uint stride = 1; stride < kV4CIndexDim; stride <<= 1) {
        if (lid < kV4CIndexDim / 2) {
            const uint block = lid / stride;
            const uint k = lid - block * stride;
            const uint i0 = block * (stride << 1) + k;
            const uint i1 = i0 + stride;
            const float a = x[i0];
            const float b = x[i1];
            x[i0] = (a + b) * inv_sqrt_2;
            x[i1] = (a - b) * inv_sqrt_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < kV4CIndexDim / kV4CFP4Block) {
        float amax = 0.0f;
        for (uint i = 0; i < kV4CFP4Block; ++i) {
            amax = max(amax, fabs(x[lid * kV4CFP4Block + i]));
        }
        const float scaled_max = max(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f);
        scales[lid] = v4c_pow2(v4c_log2_ceil_positive(scaled_max));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[lid] = half(v4c_fp4_roundtrip(x[lid], scales[lid / kV4CFP4Block]));
}

// Reference Attention.forward window-KV QAT simulation. Each row is 512
// channels; only the first 448 non-RoPE channels are quantized in block-64
// e4m3 with power-of-two ue8m0-equivalent scales. The trailing 64 RoPE
// channels remain untouched for positional precision.
[[kernel, max_total_threads_per_threadgroup(kV4CThreads)]]
kernel void v4c_window_fp8_qat(
    device half* buf [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (row >= rows) { return; }
    constexpr uint width = 512;
    constexpr uint non_rope = 448;
    constexpr uint block = 64;
    constexpr uint blocks = non_rope / block;
    device half* values = buf + row * width;
    threadgroup float scales[blocks];

    if (lid < blocks) {
        float amax = 0.0f;
        for (uint i = 0; i < block; ++i) {
            amax = max(amax, fabs(float(values[lid * block + i])));
        }
        const float raw_scale = max(amax, 1e-4f) * (1.0f / 448.0f);
        scales[lid] = v4c_pow2(v4c_log2_ceil_positive(raw_scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = lid; d < non_rope; d += kV4CThreads) {
        const float scale = scales[d / block];
        const uchar code = v4c_e4m3_encode(float(values[d]) / scale);
        values[d] = half(v4c_e4m3_decode(code) * scale);
    }
}

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

// Decode-only fusion of window KV RMSNorm, trailing RoPE, and activation QAT.
// The normalized values are explicitly stored as half before the dependent
// stages, preserving the reference's fp16 round-trip. RoPE and scale discovery
// then touch disjoint ranges and run concurrently. This is byte-equivalent to
// v4b_rmsnorm followed by v4b_rope_trailing and v4c_window_fp8_qat, while
// removing two tiny dispatches for every transformer layer and generated token.
[[kernel, max_total_threads_per_threadgroup(kV4CThreads)]]
kernel void v4c_window_norm_rope_fp8_qat_decode(
    device half* buf                 [[buffer(0)]],
    device const float* gamma        [[buffer(1)]],
    constant float& eps              [[buffer(2)]],
    constant float& position         [[buffer(3)]],
    constant float& theta            [[buffer(4)]],
    constant float& yarn_factor      [[buffer(5)]],
    constant float& orig_seq_len     [[buffer(6)]],
    constant float& beta_fast        [[buffer(7)]],
    constant float& beta_slow        [[buffer(8)]],
    constant uint& use_yarn          [[buffer(9)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]])
{
    constexpr uint width = 512;
    constexpr uint non_rope = 448;
    constexpr uint rope_dim = width - non_rope;
    constexpr uint block = 64;
    constexpr uint blocks = non_rope / block;
    threadgroup float scales[blocks];
    threadgroup float reduce_scratch[32];
    threadgroup float reduce_broadcast;

    float partial = 0.0f;
    for (uint i = lid; i < width; i += lsize) {
        const float value = float(buf[i]);
        partial = fma(value, value, partial);
    }
    const float square_sum = v4c_block_reduce_sum(
        partial, simd_lane_id, simd_group_id, simdgroups,
        reduce_scratch, &reduce_broadcast);
    const float reciprocal_rms = rsqrt(square_sum / float(width) + eps);
    for (uint i = lid; i < width; i += lsize) {
        buf[i] = half(float(buf[i]) * reciprocal_rms * gamma[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < rope_dim / 2) {
        const float freq = v4c_yarn_freq(lid, theta, yarn_factor,
                                         orig_seq_len, beta_fast, beta_slow,
                                         rope_dim);
        const float angle = position * freq;
        const float cs = cos(angle);
        const float sn = sin(angle);
        const uint i0 = non_rope + 2u * lid;
        const uint i1 = i0 + 1u;
        const float x0 = float(buf[i0]);
        const float x1 = float(buf[i1]);
        buf[i0] = half(x0 * cs - x1 * sn);
        buf[i1] = half(x0 * sn + x1 * cs);
    }

    if (lid < blocks) {
        float amax = 0.0f;
        for (uint i = 0; i < block; ++i) {
            amax = max(amax, fabs(float(buf[lid * block + i])));
        }
        const float raw_scale = max(amax, 1e-4f) * (1.0f / 448.0f);
        scales[lid] = v4c_pow2(v4c_log2_ceil_positive(raw_scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = lid; d < non_rope; d += kV4CThreads) {
        const float scale = scales[d / block];
        const uchar code = v4c_e4m3_encode(float(buf[d]) / scale);
        buf[d] = half(v4c_e4m3_decode(code) * scale);
    }
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
// The final WHT + FP4 quantize-dequantize matches the reference write-side;
// the cache stores the simulated values in FP16.
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
    threadgroup float qat_scales[HD / kV4CFP4Block];

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
        x[d] = v4c_bf16_round(acc / sum);
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
        x[d] = v4c_bf16_round(
            x[d] * rsqrt(bcast / float(HD) + norm_eps) * gamma[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Partial RoPE on the trailing 64 dims at the group-start position.
    constexpr uint rope_base = HD - 64;
    if (d >= rope_base && d < rope_base + 32) {
        // Interleaved (adjacent-pair) convention; see v4b_rope_trailing.
        const uint i = d - rope_base;
        const uint i0 = rope_base + 2u * i;
        const uint i1 = i0 + 1u;
        const float freq = use_yarn != 0u
            ? v4c_yarn_freq(i, rope_theta, yarn_factor, orig_seq_len,
                            beta_fast, beta_slow, 64u)
            : pow(rope_theta, -2.0f * float(i) / 64.0f);
        const float angle = float(rope_position) * freq;
        const float cs = cos(angle);
        const float sn = sin(angle);
        const float x0 = x[i0];
        const float x1 = x[i1];
        x[i0] = v4c_bf16_round(x0 * cs - x1 * sn);
        x[i1] = v4c_bf16_round(x0 * sn + x1 * cs);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr float inv_sqrt_2 = 0.7071067811865475244f;
    for (uint stride = 1; stride < HD; stride <<= 1) {
        if (d < HD / 2) {
            const uint block = d / stride;
            const uint k = d - block * stride;
            const uint i0 = block * (stride << 1) + k;
            const uint i1 = i0 + stride;
            const float a = x[i0];
            const float b = x[i1];
            x[i0] = (a + b) * inv_sqrt_2;
            x[i1] = (a - b) * inv_sqrt_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (d < HD / kV4CFP4Block) {
        float amax = 0.0f;
        for (uint i = 0; i < kV4CFP4Block; ++i) {
            amax = max(amax, fabs(x[d * kV4CFP4Block + i]));
        }
        const float scaled_max = max(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f);
        qat_scales[d] = v4c_pow2(v4c_log2_ceil_positive(scaled_max));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < HD) {
        out[d] = half(v4c_fp4_roundtrip(x[d], qat_scales[d / kV4CFP4Block]));
    }
}
