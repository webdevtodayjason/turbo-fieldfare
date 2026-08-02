#include <metal_stdlib>
using namespace metal;

// ============================================================================
// prefill_moe_v4 — DeepSeek V4-Flash grouped routed-MoE prefill (V4F-06b).
//
// Generalizes the decode fused MoE (moe_v4.metal) from one token to a chunk
// of up to 128 prompt tokens, following the Gemma prefill pattern
// (prefill_grouped_routed_moe_batched_phase1 / _batched_down /
// prefill_moe_reduce_token_major) with the V4 format deltas:
//
//   * Expert weights are FP4 e2m1 pairs (low nibble first along K) with ue8m0
//     power-of-two scales per 32 elements along K. No biases.
//   * Tiles hold at most 8 experts (the 16-slot per-layer cache carries the
//     in-flight tile plus the next tile being fetched).
//   * SwiGLU with swiglu_limit = 10 (V4F-reference-notes §5):
//         up   = clamp(up, -10, 10)
//         gate = min(gate, 10)          // gate has NO lower clamp
//         act  = silu(gate) * up
//   * Routing weights are F32 (normalized gathered sqrt-softplus scores x
//     route_scale 1.5) applied in the token-major reduce, not in the down
//     projection; w2 is linear so the placement is identical to the
//     reference's weight-before-w2.
//
// Dataflow per tile:
//   phase1: for each sorted (token, expert) pair p and each f in [0, F):
//       acts[p * F + f] = swiglu(gate_e[f] . x_token, up_e[f] . x_token)
//   down:   for each pair p and each d in [0, D):
//       route_partials[(token * top_k + rank) * D + d] = down_e[d] . acts[p]
//   reduce (once after ALL tiles drain): for each token t, d:
//       h2[t * D + d] = sum_r route_weights[t * top_k + r]
//                         * route_partials[(t * top_k + r) * D + d]
// Pairs for one token may span tiles; the scatter into route_partials makes
// cross-tile accumulation order irrelevant.
//
// Each (pair, row) dot product runs on one SIMD group with the same block
// structure as the decode kernel: 8 groups (256 elements, 128 bytes) per
// block, one aligned uint per lane; byte-per-lane remainder over lanes 0..15.
// Requires K % 32 == 0 and 4-byte-aligned blob sub-tensor offsets.
//
// All symbols are v4pm_-prefixed and self-contained (no dependency on
// moe_v4.metal) so this module also compiles standalone via
// `MetalContext.moduleLibrary(device:module:)`.
// ============================================================================

constant constexpr uint kV4PMGroupSize = 32;        // FP4 elements per ue8m0 scale
constant constexpr uint kV4PMMaxTileExperts = 8;    // experts per streamed tile
constant constexpr uint kV4PMRowsPerTG = 8;         // one SIMD group per row

constant float kV4PME2M1Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

struct V4PMPair {
    uint token;
    uint expert;
    uint rank;
    uint weight_bits_and_reserved;
};

struct V4PMExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint up_W_off;
    uint up_s_off;
    uint down_W_off;
    uint down_s_off;
};

struct V4PMRoutedBlobs {
    device const uint8_t* blob[kV4PMMaxTileExperts];
};

struct V4PMStreamedParams {
    uint pair_start;
    uint pair_count;
    uint D;
    uint F;
    uint top_k;
    uint hidden_stride_elements;
    uint live_expert_count;
    uint local_expert_0;
    uint local_expert_1;
    uint local_expert_2;
    uint local_expert_3;
    uint local_expert_4;
    uint local_expert_5;
    uint local_expert_6;
    uint local_expert_7;
    uint gate_W_off;
    uint gate_s_off;
    uint up_W_off;
    uint up_s_off;
    uint down_W_off;
    uint down_s_off;
};

static inline float v4pm_ue8m0(uint8_t b) {
    return exp2(float(int(b) - 127));
}

static inline float v4pm_e2m1(uint code) {
    return kV4PME2M1Lut[code & 0xFu];
}

// Reference SwiGLU (swiglu_limit = 10): silu(min(gate, 10)) * clamp(up, ±10).
// Gate has no lower clamp. Stable sigmoid keeps deeply negative (unclamped)
// gates saturating to 0 instead of overflowing exp.
static inline float v4pm_swiglu(float gate, float up) {
    const float g = min(gate, 10.0f);
    const float u = clamp(up, -10.0f, 10.0f);
    const float e = exp(-fabs(g));
    const float sig = (g >= 0.0f) ? (1.0f / (1.0f + e)) : (e / (1.0f + e));
    return g * sig * u;
}

static inline uint v4pm_local_slot(constant V4PMStreamedParams& p, uint expert) {
    if (p.live_expert_count > 0 && p.local_expert_0 == expert) return 0;
    if (p.live_expert_count > 1 && p.local_expert_1 == expert) return 1;
    if (p.live_expert_count > 2 && p.local_expert_2 == expert) return 2;
    if (p.live_expert_count > 3 && p.local_expert_3 == expert) return 3;
    if (p.live_expert_count > 4 && p.local_expert_4 == expert) return 4;
    if (p.live_expert_count > 5 && p.local_expert_5 == expert) return 5;
    if (p.live_expert_count > 6 && p.local_expert_6 == expert) return 6;
    if (p.live_expert_count > 7 && p.local_expert_7 == expert) return 7;
    return kV4PMMaxTileExperts;
}

// FP4 row dot over N elements against x, one SIMD group per row.
static inline float v4pm_fp4_row_dot(
    device const uint8_t* W,
    device const uint8_t* S,
    device const half*    x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups  = N / kV4PMGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const uint8_t* s_row = S + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 8;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * 8u + (lane >> 2);
        const float s = v4pm_ue8m0(s_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        float dot = 0.0f;
        dot = fma(v4pm_e2m1(b0 & 0xFu), float(xa.x), dot);
        dot = fma(v4pm_e2m1(b0 >> 4),   float(xa.y), dot);
        dot = fma(v4pm_e2m1(b1 & 0xFu), float(xa.z), dot);
        dot = fma(v4pm_e2m1(b1 >> 4),   float(xa.w), dot);
        dot = fma(v4pm_e2m1(b2 & 0xFu), float(xb.x), dot);
        dot = fma(v4pm_e2m1(b2 >> 4),   float(xb.y), dot);
        dot = fma(v4pm_e2m1(b3 & 0xFu), float(xb.z), dot);
        dot = fma(v4pm_e2m1(b3 >> 4),   float(xb.w), dot);
        acc = fma(s, dot, acc);
    }
    for (uint g = full_blocks * 8u; g < n_groups; ++g) {
        if (lane < 16u) {
            const float s = v4pm_ue8m0(s_row[g]);
            const uint8_t byte = W_row[g * (kV4PMGroupSize / 2) + lane];
            const float x0 = float(x[g * kV4PMGroupSize + lane * 2u]);
            const float x1 = float(x[g * kV4PMGroupSize + lane * 2u + 1u]);
            float dot = fma(v4pm_e2m1(uint(byte & 0x0Fu)), x0, 0.0f);
            dot = fma(v4pm_e2m1(uint(byte >> 4)), x1, dot);
            acc = fma(s, dot, acc);
        }
    }
    return simd_sum(acc);
}

// Gate and up rows share activation loads; x is read once per block.
static inline float2 v4pm_fp4_gate_up_dots(
    device const uint8_t* gateW,
    device const uint8_t* gateS,
    device const uint8_t* upW,
    device const uint8_t* upS,
    device const half*    x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups  = N / kV4PMGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW   + uint(row) * row_bytes;
    device const uint8_t* gS_row = gateS + uint(row) * n_groups;
    device const uint8_t* uS_row = upS   + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 8;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint gw4 = *((device const uint*)(gW_row + byte_base));
        const uint uw4 = *((device const uint*)(uW_row + byte_base));
        const uint g = blk * 8u + (lane >> 2);
        const float gs = v4pm_ue8m0(gS_row[g]);
        const float us = v4pm_ue8m0(uS_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);

        const uint gb0 =  gw4        & 0xFFu;
        const uint gb1 = (gw4 >> 8)  & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(v4pm_e2m1(gb0 & 0xFu), e0, g_dot);
        g_dot = fma(v4pm_e2m1(gb0 >> 4),   e1, g_dot);
        g_dot = fma(v4pm_e2m1(gb1 & 0xFu), e2, g_dot);
        g_dot = fma(v4pm_e2m1(gb1 >> 4),   e3, g_dot);
        g_dot = fma(v4pm_e2m1(gb2 & 0xFu), e4, g_dot);
        g_dot = fma(v4pm_e2m1(gb2 >> 4),   e5, g_dot);
        g_dot = fma(v4pm_e2m1(gb3 & 0xFu), e6, g_dot);
        g_dot = fma(v4pm_e2m1(gb3 >> 4),   e7, g_dot);

        const uint ub0 =  uw4        & 0xFFu;
        const uint ub1 = (uw4 >> 8)  & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(v4pm_e2m1(ub0 & 0xFu), e0, u_dot);
        u_dot = fma(v4pm_e2m1(ub0 >> 4),   e1, u_dot);
        u_dot = fma(v4pm_e2m1(ub1 & 0xFu), e2, u_dot);
        u_dot = fma(v4pm_e2m1(ub1 >> 4),   e3, u_dot);
        u_dot = fma(v4pm_e2m1(ub2 & 0xFu), e4, u_dot);
        u_dot = fma(v4pm_e2m1(ub2 >> 4),   e5, u_dot);
        u_dot = fma(v4pm_e2m1(ub3 & 0xFu), e6, u_dot);
        u_dot = fma(v4pm_e2m1(ub3 >> 4),   e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        u_acc = fma(us, u_dot, u_acc);
    }
    for (uint g = full_blocks * 8u; g < n_groups; ++g) {
        if (lane < 16u) {
            const float gs = v4pm_ue8m0(gS_row[g]);
            const float us = v4pm_ue8m0(uS_row[g]);
            const uint8_t gbv = gW_row[g * (kV4PMGroupSize / 2) + lane];
            const uint8_t ubv = uW_row[g * (kV4PMGroupSize / 2) + lane];
            const float x0 = float(x[g * kV4PMGroupSize + lane * 2u]);
            const float x1 = float(x[g * kV4PMGroupSize + lane * 2u + 1u]);
            float g_dot = fma(v4pm_e2m1(uint(gbv & 0x0Fu)), x0, 0.0f);
            g_dot = fma(v4pm_e2m1(uint(gbv >> 4)), x1, g_dot);
            float u_dot = fma(v4pm_e2m1(uint(ubv & 0x0Fu)), x0, 0.0f);
            u_dot = fma(v4pm_e2m1(uint(ubv >> 4)), x1, u_dot);
            g_acc = fma(gs, g_dot, g_acc);
            u_acc = fma(us, u_dot, u_acc);
        }
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

// Phase 1: grouped FP4 gate/up GEMM + clamped SwiGLU over gathered token
// rows. One SIMD group per (pair, f) row, 8 rows per threadgroup.
//   rowg = pair_local * F + f   (pair_local indexes the tile's pair slice)
//   x    = hidden + pair.token * hidden_stride_elements
//   acts[pair_local * F + f] = swiglu(gate, up)
kernel void prefill_moe_v4_grouped_phase1_gate_up_swiglu(
    device const half*             hidden      [[buffer(0)]],
    device const V4PMPair*         sorted_pairs [[buffer(1)]],
    device half*                   acts        [[buffer(2)]],
    device const V4PMRoutedBlobs&  routed      [[buffer(3)]],
    constant V4PMStreamedParams&   p           [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    const uint rowg = tg_idx * kV4PMRowsPerTG + sg_idx;
    if (rowg >= p.pair_count * p.F) return;
    const uint pair_local = rowg / p.F;
    const uint f = rowg % p.F;

    const V4PMPair pair = sorted_pairs[p.pair_start + pair_local];
    const uint slot = v4pm_local_slot(p, pair.expert);
    if (slot >= p.live_expert_count) return;

    device const uint8_t* base = routed.blob[slot];
    device const uint8_t* gW = base + p.gate_W_off;
    device const uint8_t* gS = base + p.gate_s_off;
    device const uint8_t* uW = base + p.up_W_off;
    device const uint8_t* uS = base + p.up_s_off;
    device const half* x = hidden + uint(pair.token) * p.hidden_stride_elements;

    const float2 gu = v4pm_fp4_gate_up_dots(gW, gS, uW, uS, x, f, p.D, lane);
    if (lane == 0) acts[pair_local * p.F + f] = half(v4pm_swiglu(gu.x, gu.y));
}

// Phase 2: grouped FP4 down projection, scattered into token-major route
// partials. One SIMD group per (pair, d) row, 8 rows per threadgroup.
//   rowg = pair_local * D + d
//   route_partials[(pair.token * top_k + pair.rank) * D + d] = down_d . acts_pair
// Unweighted: the F32 routing weights are applied in the token-major reduce.
kernel void prefill_moe_v4_grouped_down_scatter(
    device const V4PMPair*         sorted_pairs   [[buffer(0)]],
    device half*                   route_partials [[buffer(1)]],
    device const half*             acts           [[buffer(2)]],
    device const V4PMRoutedBlobs&  routed         [[buffer(3)]],
    constant V4PMStreamedParams&   p              [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    const uint rowg = tg_idx * kV4PMRowsPerTG + sg_idx;
    if (rowg >= p.pair_count * p.D) return;
    const uint pair_local = rowg / p.D;
    const uint d = rowg % p.D;

    const V4PMPair pair = sorted_pairs[p.pair_start + pair_local];
    const uint slot = v4pm_local_slot(p, pair.expert);
    if (slot >= p.live_expert_count) return;

    device const uint8_t* base = routed.blob[slot];
    device const uint8_t* dW = base + p.down_W_off;
    device const uint8_t* dS = base + p.down_s_off;
    device const half* act = acts + pair_local * p.F;

    const float value = v4pm_fp4_row_dot(dW, dS, act, d, p.F, lane);
    if (lane == 0) {
        route_partials[(uint(pair.token) * p.top_k + uint(pair.rank)) * p.D + d] = half(value);
    }
}

// Token-major routing-weighted reduce, run once after every tile's down
// scatter has drained. route_weights is F32 [T * top_k] straight from the
// router (normalized sqrt-softplus scores x route_scale).
//   h2[t * D + d] = sum_r route_weights[t * top_k + r]
//                     * route_partials[(t * top_k + r) * D + d]
kernel void prefill_moe_v4_reduce_token_major(
    device const half*  route_partials [[buffer(0)]],
    device const float* route_weights  [[buffer(1)]],
    device half*        h2             [[buffer(2)]],
    constant uint&      T              [[buffer(3)]],
    constant uint&      top_k          [[buffer(4)]],
    constant uint&      D              [[buffer(5)]],
    uint2               gid            [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    float acc = 0.0f;
    for (uint r = 0; r < top_k; ++r) {
        const uint partial_index = (t * top_k + r) * D + d;
        acc = fma(route_weights[t * top_k + r],
                  float(route_partials[partial_index]),
                  acc);
    }
    h2[t * D + d] = half(acc);
}
