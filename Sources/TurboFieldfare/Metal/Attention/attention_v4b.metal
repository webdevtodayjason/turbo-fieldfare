#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention_v4b — DeepSeek V4-Flash decode-side boundary kernels (V4F-03,
// second wave). Separate module from attention_v4.metal so the committed
// Wave-1 file is untouched; compiled standalone via V4ShaderLibrary.
//
// Kernel inventory:
//   v4b_rope_trailing        trailing-slice partial RoPE (NeoX pairs internal
//                            to the trailing rope_dim slice) over [rows, width]
//                            fp16, forward or conjugate (inverse) at a float
//                            position. Covers: q / window KV rotation,
//                            compressed-entry group-start rotation (caller
//                            passes the group-start position), and the
//                            negative-position output de-rotation.
//   v4b_hc_params            mHC dynamic parameter generation: RMS-normalized
//                            flattened 4x state -> 24 mixes -> sigmoid clamps
//                            (pre/post) + eps-biased Sinkhorn (comb 4x4).
//                            mix_count 4 selects the pre-only hc_head mode.
//   v4b_hc_pre               branch input gather: y = sum_j pre[j] * x[j].
//   v4b_hc_post              boundary merge: out[k] = post[k] * sublayer
//                            + sum_j comb[j,k] * residual[j] (column gather).
//   v4b_hca_compress_group   HCA 128:1 non-overlapped pooling (softmax
//                            weights + ape positional bias) + RMSNorm +
//                            group-start partial RoPE + FP8-split quantize.
//                            Mirrors v4_csa_compress_group.
//   v4b_grouped_gemv_bf16    grouped output-projection down stage:
//                            [8 groups of (1024 x 4096)] bf16 wo_a rows.
//   v4b_gemv_f32             fp32-weight GEMV (compressor wkv/wgate).
//   v4b_rmsnorm              RMSNorm, fp16 in/out, fp32 gamma, fp32 math.
//   v4b_perhead_renorm       weight-free per-head RMS renorm (V4 q path,
//                            recon note #6).
//
// All helpers are static and v4b_-prefixed so this file can later merge into
// the shared runtime library without symbol collisions.
// ============================================================================

constant constexpr uint kV4bNonRopeDim = 448;
constant constexpr uint kV4bRopeDim    = 64;
constant constexpr uint kV4bFP8Blocks  = 7;    // 448 / 64
constant constexpr uint kV4bHCGroups   = 4;    // hc_mult
constant constexpr uint kV4bHCMixes    = 24;   // (2 + hc) * hc
constant constexpr uint kV4bSinkhornRowNorms = 19;  // recon: 1 softmax + 20 col + 19 row

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

static inline float v4b_block_reduce_sum(float v,
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

/// DeepSeek YaRN-adjusted frequency for rope pair `pair` (rope dim 64),
/// matching the reference precompute_freqs_cis (mirrors v4_yarn_freq).
static inline float v4b_yarn_freq(uint pair,
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

static inline float v4b_rope_freq(uint pair,
                                  float theta,
                                  float yarn_factor,
                                  float orig_seq_len,
                                  float beta_fast,
                                  float beta_slow,
                                  uint use_yarn) {
    return use_yarn != 0u
        ? v4b_yarn_freq(pair, theta, yarn_factor, orig_seq_len, beta_fast, beta_slow)
        : pow(theta, -2.0f * float(pair) / 64.0f);
}

/// e4m3 / ue8m0 codecs (mirror of attention_v4.metal, v4b_ prefix).
static inline float v4b_e4m3_decode(uchar b) {
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

static inline uchar v4b_e4m3_encode(float x) {
    uchar sign = x < 0.0f ? 0x80u : 0x00u;
    float ax = min(fabs(x), 448.0f);
    if (ax < 0x1p-6f) {
        uint m = uint(rint(ax * 512.0f));
        if (m >= 8u) { return sign | 0x08u; }
        return sign | uchar(m);
    }
    int e = int(floor(log2(ax)));
    float mant = ax * exp2(float(-e)) - 1.0f;
    uint m = uint(rint(mant * 8.0f));
    if (m == 8u) { m = 0u; e += 1; }
    if (e > 8) { return sign | 0x7Eu; }
    return sign | uchar((uint(e + 7) << 3) | m);
}

static inline uchar v4b_ue8m0_encode(float scale) {
    int e = int(ceil(log2(scale)));
    e = clamp(e, -127, 128);
    return uchar(e + 127);
}

static inline float v4b_ue8m0_decode(uchar b) {
    return exp2(float(int(b) - 127));
}

// ============================================================================
// v4b_rope_trailing — partial RoPE on the trailing rope_dim slice of every
// row of X [rows, width] fp16, in place. Slice-local pair i (of
// rope_dim/2) rotates channels (width - rope_dim + i, width - rope_dim/2 + i)
// at `position` (float; may be negative or fractional). inverse != 0 applies
// the complex conjugate (the reference's output de-rotation at the query
// position, recon note #4). Each pair is owned by exactly one thread, so the
// in-place update is race-free.
//
// Grid: rows * (rope_dim / 2) threads.
// ============================================================================

kernel void v4b_rope_trailing(
    device       half*  X             [[buffer(0)]],   // [rows, width] fp16
    constant     uint&  rows          [[buffer(1)]],
    constant     uint&  width         [[buffer(2)]],
    constant     uint&  rope_dim      [[buffer(3)]],
    constant     float& position      [[buffer(4)]],
    constant     uint&  inverse       [[buffer(5)]],
    constant     float& theta         [[buffer(6)]],
    constant     float& yarn_factor   [[buffer(7)]],
    constant     float& orig_seq_len  [[buffer(8)]],
    constant     float& beta_fast     [[buffer(9)]],
    constant     float& beta_slow     [[buffer(10)]],
    constant     uint&  use_yarn      [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint half_rope = rope_dim / 2;
    if (gid >= rows * half_rope) { return; }
    const uint r = gid / half_rope;
    const uint i = gid % half_rope;
    const uint base = r * width + (width - rope_dim);

    const float freq = v4b_rope_freq(i, theta, yarn_factor, orig_seq_len,
                                     beta_fast, beta_slow, use_yarn);
    const float angle = position * freq;
    const float cs = cos(angle);
    const float sn = inverse != 0u ? -sin(angle) : sin(angle);
    const uint i0 = base + i;
    const uint i1 = base + half_rope + i;
    const float x0 = float(X[i0]);
    const float x1 = float(X[i1]);
    X[i0] = half(x0 * cs - x1 * sn);
    X[i1] = half(x0 * sn + x1 * cs);
}

// ============================================================================
// v4b_hc_params — mHC dynamic parameters for one token (recon §4, exact
// ordering). All math in fp32.
//
//   rsqrt   = rsqrt(mean(x^2 over 4*dim) + norm_eps)
//   mixes[j] = (hc_fn[j] . x) * rsqrt                     j in 0..<mix_count
//   pre[j]   = sigmoid(mixes[j] * scale[0] + base[j]) + hc_eps
//   post[j]  = 2 * sigmoid(mixes[4+j] * scale[1] + base[4+j])
//   comb[j,k]= mixes[8+j*4+k] * scale[2] + base[8+j*4+k]
//   Sinkhorn: row-softmax (max-subtracted, normalized) + hc_eps, then column
//   normalize (col_sum + hc_eps), then 19x { row normalize, column
//   normalize }. Totals: 1 softmax + 20 column norms + 19 row norms.
//
// mix_count == 4 selects the pre-only ParallelHead.hc_head mode (scale has
// one entry; only out[0..4) is written).
//
// Output layout (fp32): [0..4) pre, [4..8) post, [8..24) comb row-major
// (j = residual copy / row, k = output copy / column).
//
// One threadgroup of 768 threads (24 simdgroups).
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(768)]]
void v4b_hc_params(
    device const float* x             [[buffer(0)]],   // [4*dim]
    device const float* hc_fn         [[buffer(1)]],   // [mix_count, 4*dim]
    device const float* hc_base       [[buffer(2)]],   // [mix_count]
    device const float* hc_scale      [[buffer(3)]],   // [3] (head mode: [1])
    device       float* out           [[buffer(4)]],   // [28] (head mode: [4])
    constant     uint&  dim           [[buffer(5)]],
    constant     uint&  mix_count     [[buffer(6)]],   // 24 full, 4 head
    constant     float& norm_eps      [[buffer(7)]],
    constant     float& hc_eps        [[buffer(8)]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float reduce_scratch[32];
    threadgroup float bcast;
    threadgroup float mixes[kV4bHCMixes];

    const uint n = kV4bHCGroups * dim;

    // Phase A: mean square of the flattened state.
    float partial = 0.0f;
    for (uint i = lid; i < n; i += lsize) { partial = fma(x[i], x[i], partial); }
    const float sq = v4b_block_reduce_sum(partial, simd_lane_id, simd_group_id,
                                          simdgroups, reduce_scratch, &bcast);
    const float rsqrt_ms = rsqrt(sq / float(n) + norm_eps);

    // Phase B: one simdgroup per mix row.
    if (simd_group_id < mix_count) {
        device const float* w = hc_fn + simd_group_id * n;
        float d = 0.0f;
        for (uint i = simd_lane_id; i < n; i += 32u) { d = fma(w[i], x[i], d); }
        d = simd_sum(d);
        if (simd_lane_id == 0) { mixes[simd_group_id] = d * rsqrt_ms; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase C: split + clamps + Sinkhorn, single thread (4x4 math).
    if (lid != 0) { return; }

    if (mix_count == kV4bHCGroups) {
        // hc_head pre-only mode.
        for (uint j = 0; j < kV4bHCGroups; ++j) {
            const float z = mixes[j] * hc_scale[0] + hc_base[j];
            out[j] = 1.0f / (1.0f + exp(-z)) + hc_eps;
        }
        return;
    }

    for (uint j = 0; j < kV4bHCGroups; ++j) {
        const float zp = mixes[j] * hc_scale[0] + hc_base[j];
        out[j] = 1.0f / (1.0f + exp(-zp)) + hc_eps;
        const float zs = mixes[4 + j] * hc_scale[1] + hc_base[4 + j];
        out[4 + j] = 2.0f / (1.0f + exp(-zs));
    }

    float comb[kV4bHCGroups * kV4bHCGroups];
    for (uint j = 0; j < kV4bHCGroups; ++j) {
        for (uint k = 0; k < kV4bHCGroups; ++k) {
            comb[j * 4 + k] = mixes[8 + j * 4 + k] * hc_scale[2]
                            + hc_base[8 + j * 4 + k];
        }
    }
    // (1) row softmax (max-subtracted, normalized) + hc_eps.
    for (uint j = 0; j < kV4bHCGroups; ++j) {
        float mx = comb[j * 4];
        for (uint k = 1; k < kV4bHCGroups; ++k) { mx = max(mx, comb[j * 4 + k]); }
        float sum = 0.0f;
        for (uint k = 0; k < kV4bHCGroups; ++k) {
            comb[j * 4 + k] = exp(comb[j * 4 + k] - mx);
            sum += comb[j * 4 + k];
        }
        for (uint k = 0; k < kV4bHCGroups; ++k) {
            comb[j * 4 + k] = comb[j * 4 + k] / sum + hc_eps;
        }
    }
    // (2) first column normalize.
    for (uint k = 0; k < kV4bHCGroups; ++k) {
        float sum = 0.0f;
        for (uint j = 0; j < kV4bHCGroups; ++j) { sum += comb[j * 4 + k]; }
        for (uint j = 0; j < kV4bHCGroups; ++j) { comb[j * 4 + k] /= (sum + hc_eps); }
    }
    // (3) 19x { row normalize, column normalize }.
    for (uint it = 0; it < kV4bSinkhornRowNorms; ++it) {
        for (uint j = 0; j < kV4bHCGroups; ++j) {
            float sum = 0.0f;
            for (uint k = 0; k < kV4bHCGroups; ++k) { sum += comb[j * 4 + k]; }
            for (uint k = 0; k < kV4bHCGroups; ++k) { comb[j * 4 + k] /= (sum + hc_eps); }
        }
        for (uint k = 0; k < kV4bHCGroups; ++k) {
            float sum = 0.0f;
            for (uint j = 0; j < kV4bHCGroups; ++j) { sum += comb[j * 4 + k]; }
            for (uint j = 0; j < kV4bHCGroups; ++j) { comb[j * 4 + k] /= (sum + hc_eps); }
        }
    }
    for (uint i = 0; i < kV4bHCGroups * kV4bHCGroups; ++i) { out[8 + i] = comb[i]; }
}

// ============================================================================
// v4b_hc_pre — branch input: y[d] = sum_j pre[j] * x[j*dim + d]. fp32.
// Grid: dim threads.
// ============================================================================

kernel void v4b_hc_pre(
    device const float* x             [[buffer(0)]],   // [4*dim]
    device const float* params        [[buffer(1)]],   // v4b_hc_params out
    device       float* y             [[buffer(2)]],   // [dim]
    constant     uint&  dim           [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) { return; }
    float acc = 0.0f;
    for (uint j = 0; j < kV4bHCGroups; ++j) {
        acc = fma(params[j], x[j * dim + gid], acc);
    }
    y[gid] = acc;
}

// ============================================================================
// v4b_hc_post — sublayer boundary merge:
//   out[k*dim + d] = post[k] * sublayer[d] + sum_j comb[j,k] * res[j*dim + d]
// The comb gather is by COLUMN (output copy k reads comb[:, k] over residual
// copies j — recon note #10). Thread d touches only the 4 strided slots at
// [j*dim + d], so out may alias res (in-place stream update). fp32; the
// sublayer output arrives fp16. Grid: dim threads.
// ============================================================================

kernel void v4b_hc_post(
    device const float* res           [[buffer(0)]],   // [4*dim]
    device const half*  sublayer      [[buffer(1)]],   // [dim] fp16
    device const float* params        [[buffer(2)]],   // v4b_hc_params out
    device       float* out           [[buffer(3)]],   // [4*dim]
    constant     uint&  dim           [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) { return; }
    const float s = float(sublayer[gid]);
    // Preload all four residual slots BEFORE writing: out may alias res.
    float r[kV4bHCGroups];
    for (uint j = 0; j < kV4bHCGroups; ++j) { r[j] = res[j * dim + gid]; }
    for (uint k = 0; k < kV4bHCGroups; ++k) {
        float acc = params[4 + k] * s;
        for (uint j = 0; j < kV4bHCGroups; ++j) {
            acc = fma(params[8 + j * 4 + k], r[j], acc);
        }
        out[k * dim + gid] = acc;
    }
}

// ============================================================================
// v4b_hca_compress_group — HCA pooling for ONE compressed entry (128:1,
// non-overlapped; recon §3). Mirror of v4_csa_compress_group with the
// overlap channel-split removed:
//   sm[j, d] = softmax_j(gate[j, d] + ape[j, d])   (softmax over 128 rows)
//   x[d]     = sum_j sm[j, d] * kv[j, d]
// then RMSNorm(512, gamma), partial RoPE on the trailing 64 dims at the
// group-start position (compress theta + YaRN), FP8-split quantize of the
// non-rope dims, FP16 write of the rope dims.
//
// One threadgroup, 512 threads (thread d owns channel d).
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(512)]]
void v4b_hca_compress_group(
    device const float* kv            [[buffer(0)]],   // [128, 512] fp32
    device const float* gate          [[buffer(1)]],   // [128, 512] fp32
    device const float* ape           [[buffer(2)]],   // [128, 512] fp32
    device const float* gamma         [[buffer(3)]],   // [512] fp32
    device       uchar* out_values    [[buffer(4)]],   // [448] e4m3
    device       uchar* out_scales    [[buffer(5)]],   // [8] ue8m0 (7 used)
    device       half*  out_rope      [[buffer(6)]],   // [64] fp16
    constant     uint&  rope_position [[buffer(7)]],
    constant     float& rope_theta    [[buffer(8)]],
    constant     float& yarn_factor   [[buffer(9)]],
    constant     float& orig_seq_len  [[buffer(10)]],
    constant     float& beta_fast     [[buffer(11)]],
    constant     float& beta_slow     [[buffer(12)]],
    constant     uint&  use_yarn      [[buffer(13)]],
    constant     float& norm_eps      [[buffer(14)]],
    uint lid             [[thread_position_in_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    constexpr uint HD = 512;
    constexpr uint ROWS = 128;
    threadgroup float x[HD];
    threadgroup float reduce_scratch[16];    // 512 threads = 16 simdgroups
    threadgroup float bcast;
    threadgroup float block_scales[kV4bFP8Blocks];

    const uint d = lid;

    // Softmax-weighted pool over the 128 rows for this channel.
    float mx = -INFINITY;
    for (uint j = 0; j < ROWS; ++j) {
        mx = max(mx, gate[j * HD + d] + ape[j * HD + d]);
    }
    float acc = 0.0f;
    float sum = 0.0f;
    for (uint j = 0; j < ROWS; ++j) {
        const float w = exp(gate[j * HD + d] + ape[j * HD + d] - mx);
        sum += w;
        acc = fma(w, kv[j * HD + d], acc);
    }
    acc /= sum;

    // RMSNorm over the 512 channels.
    const float sq = v4b_block_reduce_sum(acc * acc, simd_lane_id,
                                          simd_group_id, simdgroups,
                                          reduce_scratch, &bcast);
    x[d] = acc * rsqrt(sq / float(HD) + norm_eps) * gamma[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Partial RoPE on the trailing 64 dims at the group-start position.
    if (d >= kV4bNonRopeDim && d < kV4bNonRopeDim + kV4bRopeDim / 2) {
        const uint i = d - kV4bNonRopeDim;
        const float freq = v4b_rope_freq(i, rope_theta, yarn_factor,
                                         orig_seq_len, beta_fast, beta_slow,
                                         use_yarn);
        const float angle = float(rope_position) * freq;
        const float cs = cos(angle);
        const float sn = sin(angle);
        const float x0 = x[d];
        const float x1 = x[d + kV4bRopeDim / 2];
        x[d]                  = x0 * cs - x1 * sn;
        x[d + kV4bRopeDim / 2] = x0 * sn + x1 * cs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FP8 quantize the non-rope dims: 7 blocks of 64 channels.
    if (d < kV4bFP8Blocks) {
        float amax = 0.0f;
        for (uint i = 0; i < 64; ++i) {
            amax = max(amax, fabs(x[d * 64 + i]));
        }
        block_scales[d] = v4b_ue8m0_decode(
            v4b_ue8m0_encode(max(amax, 1e-4f) / 448.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d < kV4bNonRopeDim) {
        const float scale = block_scales[d >> 6];
        out_values[d] = v4b_e4m3_encode(x[d] / scale);
        if ((d & 63u) == 0u) {
            out_scales[d >> 6] = v4b_ue8m0_encode(scale);
        }
    } else {
        out_rope[d - kV4bNonRopeDim] = half(x[d]);
    }
}

// ============================================================================
// v4b_grouped_gemv_bf16 — grouped output-projection down stage (recon §2,
// grouped output projection): o [8, 4096] through wo_a viewed [8, 1024, 4096]
// bf16, producing the o_lora_rank low-rank per group. Row index decomposes as
// (group = row / rows_per_group, r = row % rows_per_group); the group's input
// slice is x + group * N. One SIMD group per row, eight rows per threadgroup.
// Requires N % 64 == 0.
// ============================================================================

kernel void v4b_grouped_gemv_bf16(
    device const bfloat* W              [[buffer(0)]],   // [M, N] bf16
    device const half*   x              [[buffer(1)]],   // [groups * N] fp16
    device       half*   y              [[buffer(2)]],   // [M] fp16
    constant     uint&   M              [[buffer(3)]],
    constant     uint&   N              [[buffer(4)]],
    constant     uint&   rows_per_group [[buffer(5)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint group = row / rows_per_group;
    device const bfloat* W_row = W + uint(row) * N;
    device const half* x_group = x + uint(group) * N;

    float acc = 0.0f;
    for (uint base = 0; base < N; base += 64u) {
        const uint i0 = base + lane * 2u;
        acc = fma(float(W_row[i0]),       float(x_group[i0]),       acc);
        acc = fma(float(W_row[i0 + 1u]),  float(x_group[i0 + 1u]),  acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = half(acc); }
}

// ============================================================================
// v4b_gemv_f32 — fp32-weight GEMV for the compressor wkv/wgate projections
// (fp32 Linears per recon §2/§3; the fp16 hidden state is read exactly and
// the dot accumulates in fp32). One SIMD group per row, eight rows per
// threadgroup, float4 main loop. Requires N % 128 == 0 and a 16-byte-aligned
// weights base (wrapper-enforced).
// ============================================================================

kernel void v4b_gemv_f32(
    device const float* W      [[buffer(0)]],   // [M, N] fp32
    device const half*  x      [[buffer(1)]],   // [N] fp16
    device       float* y      [[buffer(2)]],   // [M] fp32
    constant     uint&  M      [[buffer(3)]],
    constant     uint&  N      [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    device const float* W_row = W + uint(row) * N;

    float acc = 0.0f;
    for (uint base = 0; base < N; base += 128u) {
        const uint i0 = base + lane * 4u;
        const float4 w = *((device const float4*)(W_row + i0));
        acc = fma(w.x, float(x[i0]),       acc);
        acc = fma(w.y, float(x[i0 + 1u]),  acc);
        acc = fma(w.z, float(x[i0 + 2u]),  acc);
        acc = fma(w.w, float(x[i0 + 3u]),  acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = acc; }
}

// ============================================================================
// v4b_rmsnorm — y = x * rsqrt(mean(x^2) + eps) * gamma, fp16 in/out, fp32
// gamma and math. In-place safe (elementwise after the reduction). One
// threadgroup of 256 threads.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(256)]]
void v4b_rmsnorm(
    device const half*  x        [[buffer(0)]],   // [n] fp16
    device const float* gamma    [[buffer(1)]],   // [n] fp32 (unused if !use_gamma)
    device       half*  y        [[buffer(2)]],   // [n] fp16
    constant     uint&  n        [[buffer(3)]],
    constant     float& eps      [[buffer(4)]],
    constant     uint&  use_gamma [[buffer(5)]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float reduce_scratch[32];
    threadgroup float bcast;

    float partial = 0.0f;
    for (uint i = lid; i < n; i += lsize) {
        const float v = float(x[i]);
        partial = fma(v, v, partial);
    }
    const float sq = v4b_block_reduce_sum(partial, simd_lane_id, simd_group_id,
                                          simdgroups, reduce_scratch, &bcast);
    const float rs = rsqrt(sq / float(n) + eps);
    for (uint i = lid; i < n; i += lsize) {
        const float g = use_gamma != 0u ? gamma[i] : 1.0f;
        y[i] = half(float(x[i]) * rs * g);
    }
}

// ============================================================================
// v4b_perhead_renorm — V4's weight-free per-head RMS renorm after wq_b
// (recon note #6): x[h, :] *= rsqrt(mean(x[h, :]^2) + eps), in place, fp16
// storage with fp32 math. One threadgroup per head, 256 threads.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(256)]]
void v4b_perhead_renorm(
    device       half*  x        [[buffer(0)]],   // [heads, head_dim] fp16
    constant     uint&  head_dim [[buffer(1)]],
    constant     float& eps      [[buffer(2)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float reduce_scratch[32];
    threadgroup float bcast;

    device half* row = x + uint(tg_id) * head_dim;
    float partial = 0.0f;
    for (uint i = lid; i < head_dim; i += lsize) {
        const float v = float(row[i]);
        partial = fma(v, v, partial);
    }
    const float sq = v4b_block_reduce_sum(partial, simd_lane_id, simd_group_id,
                                          simdgroups, reduce_scratch, &bcast);
    const float rs = rsqrt(sq / float(head_dim) + eps);
    for (uint i = lid; i < head_dim; i += lsize) {
        row[i] = half(float(row[i]) * rs);
    }
}
