#include <metal_stdlib>
using namespace metal;

// ============================================================================
// prefill_boundary_v4 — DeepSeek V4-Flash batched prefill boundary kernels
// (V4F-06c, work-order A1). Batched (row-parallel) variants of the decode
// boundary kernels in attention_v4b.metal: the chunked prefill runner moves
// up to 128 prompt tokens per chunk through each layer boundary in one
// dispatch instead of one dispatch per token.
//
// Kernel inventory:
//   v4pf_hc_params        batched mHC dynamic parameter generation: one
//                         threadgroup (768 threads) per row, exact decode
//                         ordering (sigmoid pre + eps, 2*sigmoid post,
//                         eps-biased Sinkhorn: 1 row-softmax + 20 column
//                         norms + 19 row norms). Output [rows, 24] fp32.
//   v4pf_hc_pre           batched branch input gather:
//                         y[r, d] = sum_j pre[r, j] * x[r, j*dim + d].
//   v4pf_hc_post          batched boundary merge:
//                         out[r, k*dim+d] = post[r, k] * sublayer[r, d]
//                         + sum_j comb[r, j, k] * residual[r, j*dim + d]
//                         (comb gather by COLUMN).
//   v4pf_rmsnorm_f32f16   batched RMSNorm, fp32 in / fp16 out, fp32 gamma,
//                         fp32 math (the attn/ffn boundary norm; the decode
//                         v4b_rmsnorm is fp16 in/out).
//   v4pf_rope_trailing    batched trailing-slice partial RoPE with a
//                         PER-ROW positions buffer (fp32). Interleaved
//                         adjacent-pair convention (elements 2i, 2i+1 of
//                         the rope slice — NOT half-split; see the design
//                         note pitfalls). inverse != 0 applies the complex
//                         conjugate (output de-rotation) with POSITIVE
//                         positions.
//
// All helpers are static and v4pf_-prefixed so this file can merge into the
// shared runtime library without symbol collisions.
// ============================================================================

constant constexpr uint kV4pfHCGroups   = 4;    // hc_mult
constant constexpr uint kV4pfHCMixes    = 24;   // (2 + hc) * hc
constant constexpr uint kV4pfSinkhornRowNorms = 19;  // recon: 1 softmax + 20 col + 19 row

// -------------------------------------------------------------------------
// Helpers (mirrors of the v4b_ decode helpers, v4pf_ prefix)
// -------------------------------------------------------------------------

static inline float v4pf_block_reduce_sum(float v,
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
/// matching the reference precompute_freqs_cis (mirror of v4b_yarn_freq).
static inline float v4pf_yarn_freq(uint pair,
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

static inline float v4pf_rope_freq(uint pair,
                                   float theta,
                                   float yarn_factor,
                                   float orig_seq_len,
                                   float beta_fast,
                                   float beta_slow,
                                   uint use_yarn) {
    return use_yarn != 0u
        ? v4pf_yarn_freq(pair, theta, yarn_factor, orig_seq_len, beta_fast, beta_slow)
        : pow(theta, -2.0f * float(pair) / 64.0f);
}

// ============================================================================
// v4pf_hc_params — batched mHC dynamic parameters (recon §4, exact decode
// ordering; mirror of v4b_hc_params with a row dimension). All math fp32.
//
//   rsqrt     = rsqrt(mean(x[r]^2 over 4*dim) + norm_eps)
//   mixes[j]  = (hc_fn[j] . x[r]) * rsqrt                    j in 0..<24
//   pre[j]    = sigmoid(mixes[j] * scale[0] + base[j]) + hc_eps
//   post[j]   = 2 * sigmoid(mixes[4+j] * scale[1] + base[4+j])
//   comb[j,k] = mixes[8+j*4+k] * scale[2] + base[8+j*4+k]
//   Sinkhorn: row-softmax (max-subtracted, normalized) + hc_eps, then column
//   normalize (col_sum + hc_eps), then 19x { row normalize, column
//   normalize }. Totals: 1 softmax + 20 column norms + 19 row norms.
//
// Output layout per row (fp32): [0..4) pre, [4..8) post, [8..24) comb
// row-major (j = residual copy / row, k = output copy / column).
//
// Grid: `rows` threadgroups of 768 threads; threadgroup r owns row r.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(768)]]
void v4pf_hc_params(
    device const float* x             [[buffer(0)]],   // [rows, 4*dim]
    device const float* hc_fn         [[buffer(1)]],   // [24, 4*dim]
    device const float* hc_base       [[buffer(2)]],   // [24]
    device const float* hc_scale      [[buffer(3)]],   // [3]
    device       float* out           [[buffer(4)]],   // [rows, 24]
    constant     uint&  dim           [[buffer(5)]],
    constant     float& norm_eps      [[buffer(6)]],
    constant     float& hc_eps        [[buffer(7)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float reduce_scratch[32];
    threadgroup float bcast;
    threadgroup float mixes[kV4pfHCMixes];

    const uint n = kV4pfHCGroups * dim;
    device const float* xr = x + uint(tg_id) * n;
    device float* outr = out + uint(tg_id) * kV4pfHCMixes;

    // Phase A: mean square of the flattened row state.
    float partial = 0.0f;
    for (uint i = lid; i < n; i += lsize) { partial = fma(xr[i], xr[i], partial); }
    const float sq = v4pf_block_reduce_sum(partial, simd_lane_id, simd_group_id,
                                           simdgroups, reduce_scratch, &bcast);
    const float rsqrt_ms = rsqrt(sq / float(n) + norm_eps);

    // Phase B: one simdgroup per mix row.
    if (simd_group_id < kV4pfHCMixes) {
        device const float* w = hc_fn + simd_group_id * n;
        float d = 0.0f;
        for (uint i = simd_lane_id; i < n; i += 32u) { d = fma(w[i], xr[i], d); }
        d = simd_sum(d);
        if (simd_lane_id == 0) { mixes[simd_group_id] = d * rsqrt_ms; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase C: split + clamps + Sinkhorn, single thread (4x4 math).
    if (lid != 0) { return; }

    for (uint j = 0; j < kV4pfHCGroups; ++j) {
        const float zp = mixes[j] * hc_scale[0] + hc_base[j];
        outr[j] = 1.0f / (1.0f + exp(-zp)) + hc_eps;
        const float zs = mixes[4 + j] * hc_scale[1] + hc_base[4 + j];
        outr[4 + j] = 2.0f / (1.0f + exp(-zs));
    }

    float comb[kV4pfHCGroups * kV4pfHCGroups];
    for (uint j = 0; j < kV4pfHCGroups; ++j) {
        for (uint k = 0; k < kV4pfHCGroups; ++k) {
            comb[j * 4 + k] = mixes[8 + j * 4 + k] * hc_scale[2]
                            + hc_base[8 + j * 4 + k];
        }
    }
    // (1) row softmax (max-subtracted, normalized) + hc_eps.
    for (uint j = 0; j < kV4pfHCGroups; ++j) {
        float mx = comb[j * 4];
        for (uint k = 1; k < kV4pfHCGroups; ++k) { mx = max(mx, comb[j * 4 + k]); }
        float sum = 0.0f;
        for (uint k = 0; k < kV4pfHCGroups; ++k) {
            comb[j * 4 + k] = exp(comb[j * 4 + k] - mx);
            sum += comb[j * 4 + k];
        }
        for (uint k = 0; k < kV4pfHCGroups; ++k) {
            comb[j * 4 + k] = comb[j * 4 + k] / sum + hc_eps;
        }
    }
    // (2) first column normalize.
    for (uint k = 0; k < kV4pfHCGroups; ++k) {
        float sum = 0.0f;
        for (uint j = 0; j < kV4pfHCGroups; ++j) { sum += comb[j * 4 + k]; }
        for (uint j = 0; j < kV4pfHCGroups; ++j) { comb[j * 4 + k] /= (sum + hc_eps); }
    }
    // (3) 19x { row normalize, column normalize }.
    for (uint it = 0; it < kV4pfSinkhornRowNorms; ++it) {
        for (uint j = 0; j < kV4pfHCGroups; ++j) {
            float sum = 0.0f;
            for (uint k = 0; k < kV4pfHCGroups; ++k) { sum += comb[j * 4 + k]; }
            for (uint k = 0; k < kV4pfHCGroups; ++k) { comb[j * 4 + k] /= (sum + hc_eps); }
        }
        for (uint k = 0; k < kV4pfHCGroups; ++k) {
            float sum = 0.0f;
            for (uint j = 0; j < kV4pfHCGroups; ++j) { sum += comb[j * 4 + k]; }
            for (uint j = 0; j < kV4pfHCGroups; ++j) { comb[j * 4 + k] /= (sum + hc_eps); }
        }
    }
    for (uint i = 0; i < kV4pfHCGroups * kV4pfHCGroups; ++i) { outr[8 + i] = comb[i]; }
}

// ============================================================================
// v4pf_hc_pre — batched branch input:
//   y[r*dim + d] = sum_j params[r*24 + j] * x[r*4*dim + j*dim + d]
// fp32. Grid: rows * dim threads.
// ============================================================================

kernel void v4pf_hc_pre(
    device const float* x             [[buffer(0)]],   // [rows, 4*dim]
    device const float* params        [[buffer(1)]],   // [rows, 24]
    device       float* y             [[buffer(2)]],   // [rows, dim]
    constant     uint&  dim           [[buffer(3)]],
    constant     uint&  rows          [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows * dim) { return; }
    const uint r = gid / dim;
    const uint d = gid % dim;
    device const float* xr = x + r * (kV4pfHCGroups * dim) + d;
    device const float* pr = params + r * kV4pfHCMixes;
    float acc = 0.0f;
    for (uint j = 0; j < kV4pfHCGroups; ++j) {
        acc = fma(pr[j], xr[j * dim], acc);
    }
    y[gid] = acc;
}

// ============================================================================
// v4pf_hc_post — batched sublayer boundary merge:
//   out[r, k*dim+d] = post[r,k] * sublayer[r,d]
//                     + sum_j comb[r,j,k] * res[r, j*dim+d]
// The comb gather is by COLUMN (output copy k reads comb[:, k] over residual
// copies j — recon note #10). Thread (r, d) touches only the 4 strided slots
// at [r, j*dim + d], so out may alias res per row (in-place stream update).
// fp32; the sublayer output arrives fp16. Grid: rows * dim threads.
// ============================================================================

kernel void v4pf_hc_post(
    device const float* res           [[buffer(0)]],   // [rows, 4*dim]
    device const half*  sublayer      [[buffer(1)]],   // [rows, dim] fp16
    device const float* params        [[buffer(2)]],   // [rows, 24]
    device       float* out           [[buffer(3)]],   // [rows, 4*dim]
    constant     uint&  dim           [[buffer(4)]],
    constant     uint&  rows          [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows * dim) { return; }
    const uint r = gid / dim;
    const uint d = gid % dim;
    const uint row_stride = kV4pfHCGroups * dim;
    const float s = float(sublayer[gid]);
    device const float* pr = params + r * kV4pfHCMixes;
    // Preload all four residual slots BEFORE writing: out may alias res.
    float rv[kV4pfHCGroups];
    for (uint j = 0; j < kV4pfHCGroups; ++j) {
        rv[j] = res[r * row_stride + j * dim + d];
    }
    for (uint k = 0; k < kV4pfHCGroups; ++k) {
        float acc = pr[4 + k] * s;
        for (uint j = 0; j < kV4pfHCGroups; ++j) {
            acc = fma(pr[8 + j * 4 + k], rv[j], acc);
        }
        out[r * row_stride + k * dim + d] = acc;
    }
}

// ============================================================================
// v4pf_rmsnorm_f32f16 — batched RMSNorm: y[r, i] = x[r, i] *
// rsqrt(mean(x[r]^2) + eps) * gamma[i]. fp32 in, fp16 out, fp32 gamma and
// math (the prefill attn/ffn boundary norm; input is the fp32 hc_pre gather
// or the fp32 residual stream, output feeds the fp16 projection path).
// One threadgroup of 256 threads per row.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(256)]]
void v4pf_rmsnorm_f32f16(
    device const float* x        [[buffer(0)]],   // [rows, n] fp32
    device const float* gamma    [[buffer(1)]],   // [n] fp32 (unused if !use_gamma)
    device       half*  y        [[buffer(2)]],   // [rows, n] fp16
    constant     uint&  n        [[buffer(3)]],
    constant     float& eps      [[buffer(4)]],
    constant     uint&  use_gamma [[buffer(5)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float reduce_scratch[32];
    threadgroup float bcast;

    device const float* xr = x + uint(tg_id) * n;
    device half* yr = y + uint(tg_id) * n;

    float partial = 0.0f;
    for (uint i = lid; i < n; i += lsize) {
        const float v = xr[i];
        partial = fma(v, v, partial);
    }
    const float sq = v4pf_block_reduce_sum(partial, simd_lane_id, simd_group_id,
                                           simdgroups, reduce_scratch, &bcast);
    const float rs = rsqrt(sq / float(n) + eps);
    for (uint i = lid; i < n; i += lsize) {
        const float g = use_gamma != 0u ? gamma[i] : 1.0f;
        yr[i] = half(xr[i] * rs * g);
    }
}

// Batched, bit-exact mirror of decode `v4c_rmsnorm_f32in`. Keep the decode
// kernel's constant 256-thread reduction geometry and expression order rather
// than routing through the generic helper above. Small compiler differences in
// the generic form can move a handful of FP16 outputs by one ULP, which then
// changes the causal window history during long prompt prefill.
[[kernel, max_total_threads_per_threadgroup(256)]]
void v4pf_rmsnorm_f32f16_serial_order(
    device const float* x     [[buffer(0)]],   // [rows, n] fp32
    device const float* gamma [[buffer(1)]],   // [n] fp32
    device half* out          [[buffer(2)]],   // [rows, n] fp16
    constant uint& n          [[buffer(3)]],
    constant float& eps       [[buffer(4)]],
    uint row                  [[threadgroup_position_in_grid]],
    uint lid                  [[thread_position_in_threadgroup]],
    uint simd_lane_id         [[thread_index_in_simdgroup]],
    uint simd_group_id        [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reduce_scratch[8];
    threadgroup float bcast;
    device const float* xr = x + uint64_t(row) * n;
    device half* yr = out + uint64_t(row) * n;

    float sq = 0.0f;
    for (uint i = lid; i < n; i += 256u) {
        const float v = xr[i];
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (simd_lane_id == 0u) { reduce_scratch[simd_group_id] = sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0u) {
        float t = (simd_lane_id < 8u) ? reduce_scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0u) { bcast = rsqrt(t / float(n) + eps); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rstd = bcast;
    for (uint i = lid; i < n; i += 256u) {
        yr[i] = half(xr[i] * rstd * gamma[i]);
    }
}

// Batched mirror of decode `v4b_rmsnorm`: fp16 input/output with identical
// 256-thread reduction geometry. This preserves the decode quantization point
// after wq_a/window_wkv and is safe in place because each row keeps its fp16
// stride (unlike packing half output into fp32 input storage).
[[kernel, max_total_threads_per_threadgroup(256)]]
void v4pf_rmsnorm_f16f16(
    device const half* x         [[buffer(0)]],   // [rows, n] fp16
    device const float* gamma    [[buffer(1)]],   // [n] fp32
    device half* y               [[buffer(2)]],   // [rows, n] fp16
    constant uint& n             [[buffer(3)]],
    constant float& eps          [[buffer(4)]],
    constant uint& use_gamma     [[buffer(5)]],
    uint tg_id                   [[threadgroup_position_in_grid]],
    uint lid                     [[thread_position_in_threadgroup]],
    uint lsize                   [[threads_per_threadgroup]],
    uint simd_lane_id            [[thread_index_in_simdgroup]],
    uint simd_group_id           [[simdgroup_index_in_threadgroup]],
    uint simdgroups              [[simdgroups_per_threadgroup]]
) {
    threadgroup float reduce_scratch[32];
    threadgroup float bcast;
    device const half* xr = x + uint64_t(tg_id) * n;
    device half* yr = y + uint64_t(tg_id) * n;

    float partial = 0.0f;
    for (uint i = lid; i < n; i += lsize) {
        const float v = float(xr[i]);
        partial = fma(v, v, partial);
    }
    const float sq = v4pf_block_reduce_sum(partial, simd_lane_id, simd_group_id,
                                           simdgroups, reduce_scratch, &bcast);
    const float rs = rsqrt(sq / float(n) + eps);
    for (uint i = lid; i < n; i += lsize) {
        const float g = use_gamma != 0u ? gamma[i] : 1.0f;
        yr[i] = half(float(xr[i]) * rs * g);
    }
}

// ============================================================================
// v4pf_rope_trailing — batched partial RoPE on the trailing rope_dim slice of
// every row of X [rows, width] fp16, in place, with a PER-ROW position
// buffer (prompt tokens in a chunk sit at distinct absolute positions).
// Slice-local pair i (of rope_dim/2) rotates the INTERLEAVED adjacent pair
// (2i, 2i+1) of the rope slice — matching the reference apply_rotary_emb
// (x.unflatten(-1, (-1, 2)) -> view_as_complex). NOT the half-split pairing
// (i, i + rope_dim/2); that convention bug garbled all distance-dependent
// attention while leaving same-position scores intact (design note
// pitfalls). inverse != 0 applies the complex conjugate for the output
// de-rotation; positions stay POSITIVE (see 8fffb1a). Each (row, pair) is
// owned by exactly one thread, so the in-place update is race-free.
//
// Grid: rows * (rope_dim / 2) threads.
// ============================================================================

kernel void v4pf_rope_trailing(
    device       half*  X             [[buffer(0)]],   // [rows, width] fp16
    device const float* positions     [[buffer(1)]],   // [rows] fp32
    constant     uint&  rows          [[buffer(2)]],
    constant     uint&  width         [[buffer(3)]],
    constant     uint&  rope_dim      [[buffer(4)]],
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

    const float freq = v4pf_rope_freq(i, theta, yarn_factor, orig_seq_len,
                                      beta_fast, beta_slow, use_yarn);
    const float angle = positions[r] * freq;
    const float cs = cos(angle);
    const float sn = inverse != 0u ? -sin(angle) : sin(angle);
    // Interleaved (adjacent-pair) convention; see the header comment.
    const uint i0 = base + 2u * i;
    const uint i1 = base + 2u * i + 1u;
    const float x0 = float(X[i0]);
    const float x1 = float(X[i1]);
    X[i0] = half(x0 * cs - x1 * sn);
    X[i1] = half(x0 * sn + x1 * cs);
}
