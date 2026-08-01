#include <metal_stdlib>
using namespace metal;

// ============================================================================
// moe_v4 — DeepSeek V4-Flash fused routed MoE (decode) and router (V4F-02).
//
// Deltas vs the Gemma moe.metal stack:
//   * Expert weights are FP4 e2m1 pairs (low nibble first along K) with ue8m0
//     power-of-two scales per 32 elements along K. No biases anywhere, so the
//     b·Σx term and its bookkeeping are gone.
//   * Six streamed experts (top-6), not eight.
//   * SwiGLU with swiglu_limit = 10, transcribed from the official reference
//     (V4F-reference-notes §5):
//         up   = clamp(up, -10, 10)
//         gate = min(gate, 10)          // NOTE: gate has NO lower clamp
//         act  = silu(gate) * up
//     The routing weight is applied to the down-projection output, which is
//     identical to the reference's weight-before-w2 because w2 is linear.
//   * Router gate is BF16 [256, 4096] with an F32 static correction bias.
//     Scoring: s = sqrt(softplus(logit)); selection on s + bias; weights
//     gathered from the UNbiased s, normalized by their sum, scaled by
//     route_scale (1.5 Flash). No softmax, no per-expert scale vector.
//
// Expert blobs carry six 4-byte-aligned sub-tensors per expert:
//   gate W [F, D/2], gate ue8m0 scales [F, D/32],
//   up   W [F, D/2], up   ue8m0 scales [F, D/32],
//   down W [D, F/2], down ue8m0 scales [D, F/32].
// Offsets arrive in one V4ExpertOffsets struct shared by all blobs.
//
// All helpers are `static` and v4m_-prefixed so this file can later merge
// into the shared runtime library without symbol collisions.
// ============================================================================

constant constexpr uint kV4MoEGroupSize = 32;      // FP4 elements per ue8m0 scale
constant constexpr uint kV4MaxStreamedExperts = 6; // top-6 routing

constant float kV4MoEE2M1Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

struct V4ExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint up_W_off;
    uint up_s_off;
    uint down_W_off;
    uint down_s_off;
};

struct V4RoutedBlobs {
    device const uint8_t* blob[kV4MaxStreamedExperts];
};

static inline float v4m_ue8m0(uint8_t b) {
    return exp2(float(int(b) - 127));
}

static inline float v4m_e2m1(uint code) {
    return kV4MoEE2M1Lut[code & 0xFu];
}

// Reference SwiGLU activation (swiglu_limit = 10):
//   act = silu(min(gate, 10)) * clamp(up, -10, 10).
// Gate has no lower clamp. Stable sigmoid: silu(g) = g * sigmoid(g) computed
// as g/(1+e^-|g|) for g >= 0 and g*e^-|g|/(1+e^-|g|) for g < 0, so deeply
// negative (unclamped) gates saturate to 0 instead of overflowing exp.
static inline float v4m_swiglu(float gate, float up) {
    const float g = min(gate, 10.0f);
    const float u = clamp(up, -10.0f, 10.0f);
    const float e = exp(-fabs(g));
    const float sig = (g >= 0.0f) ? (1.0f / (1.0f + e)) : (e / (1.0f + e));
    return g * sig * u;
}

// FP4 row GEMV over N elements. Same block structure as
// dequant_fp4_e2m1_gemv_simd: 8 groups (256 elements, 128 bytes) per block,
// one aligned uint per lane; byte-per-lane remainder over lanes 0..15.
// Requires N % 32 == 0 and a 4-byte-aligned W (blob sub-tensors are padded).
static inline float v4m_fp4_gemv_row(
    device const uint8_t* W,
    device const uint8_t* S,
    device const half*    x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups  = N / kV4MoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const uint8_t* s_row = S + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 8;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * 8u + (lane >> 2);
        const float s = v4m_ue8m0(s_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        float dot = 0.0f;
        dot = fma(v4m_e2m1(b0 & 0xFu), float(xa.x), dot);
        dot = fma(v4m_e2m1(b0 >> 4),   float(xa.y), dot);
        dot = fma(v4m_e2m1(b1 & 0xFu), float(xa.z), dot);
        dot = fma(v4m_e2m1(b1 >> 4),   float(xa.w), dot);
        dot = fma(v4m_e2m1(b2 & 0xFu), float(xb.x), dot);
        dot = fma(v4m_e2m1(b2 >> 4),   float(xb.y), dot);
        dot = fma(v4m_e2m1(b3 & 0xFu), float(xb.z), dot);
        dot = fma(v4m_e2m1(b3 >> 4),   float(xb.w), dot);
        acc = fma(s, dot, acc);
    }
    for (uint g = full_blocks * 8u; g < n_groups; ++g) {
        if (lane < 16u) {
            const float s = v4m_ue8m0(s_row[g]);
            const uint8_t byte = W_row[g * (kV4MoEGroupSize / 2) + lane];
            const float x0 = float(x[g * kV4MoEGroupSize + lane * 2u]);
            const float x1 = float(x[g * kV4MoEGroupSize + lane * 2u + 1u]);
            float dot = fma(v4m_e2m1(uint(byte & 0x0Fu)), x0, 0.0f);
            dot = fma(v4m_e2m1(uint(byte >> 4)), x1, dot);
            acc = fma(s, dot, acc);
        }
    }
    return simd_sum(acc);
}

// Gate and up rows share activation loads; x is read once per block.
static inline float2 v4m_fp4_gate_up_rows(
    device const uint8_t* gateW,
    device const uint8_t* gateS,
    device const uint8_t* upW,
    device const uint8_t* upS,
    device const half*    x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups  = N / kV4MoEGroupSize;
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
        const float gs = v4m_ue8m0(gS_row[g]);
        const float us = v4m_ue8m0(uS_row[g]);
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
        g_dot = fma(v4m_e2m1(gb0 & 0xFu), e0, g_dot);
        g_dot = fma(v4m_e2m1(gb0 >> 4),   e1, g_dot);
        g_dot = fma(v4m_e2m1(gb1 & 0xFu), e2, g_dot);
        g_dot = fma(v4m_e2m1(gb1 >> 4),   e3, g_dot);
        g_dot = fma(v4m_e2m1(gb2 & 0xFu), e4, g_dot);
        g_dot = fma(v4m_e2m1(gb2 >> 4),   e5, g_dot);
        g_dot = fma(v4m_e2m1(gb3 & 0xFu), e6, g_dot);
        g_dot = fma(v4m_e2m1(gb3 >> 4),   e7, g_dot);

        const uint ub0 =  uw4        & 0xFFu;
        const uint ub1 = (uw4 >> 8)  & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(v4m_e2m1(ub0 & 0xFu), e0, u_dot);
        u_dot = fma(v4m_e2m1(ub0 >> 4),   e1, u_dot);
        u_dot = fma(v4m_e2m1(ub1 & 0xFu), e2, u_dot);
        u_dot = fma(v4m_e2m1(ub1 >> 4),   e3, u_dot);
        u_dot = fma(v4m_e2m1(ub2 & 0xFu), e4, u_dot);
        u_dot = fma(v4m_e2m1(ub2 >> 4),   e5, u_dot);
        u_dot = fma(v4m_e2m1(ub3 & 0xFu), e6, u_dot);
        u_dot = fma(v4m_e2m1(ub3 >> 4),   e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        u_acc = fma(us, u_dot, u_acc);
    }
    for (uint g = full_blocks * 8u; g < n_groups; ++g) {
        if (lane < 16u) {
            const float gs = v4m_ue8m0(gS_row[g]);
            const float us = v4m_ue8m0(uS_row[g]);
            const uint8_t gbv = gW_row[g * (kV4MoEGroupSize / 2) + lane];
            const uint8_t ubv = uW_row[g * (kV4MoEGroupSize / 2) + lane];
            const float x0 = float(x[g * kV4MoEGroupSize + lane * 2u]);
            const float x1 = float(x[g * kV4MoEGroupSize + lane * 2u + 1u]);
            float g_dot = fma(v4m_e2m1(uint(gbv & 0x0Fu)), x0, 0.0f);
            g_dot = fma(v4m_e2m1(uint(gbv >> 4)), x1, g_dot);
            float u_dot = fma(v4m_e2m1(uint(ubv & 0x0Fu)), x0, 0.0f);
            u_dot = fma(v4m_e2m1(uint(ubv >> 4)), x1, u_dot);
            g_acc = fma(gs, g_dot, g_acc);
            u_acc = fma(us, u_dot, u_acc);
        }
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

static inline void v4m_phase1_body(
    device const V4RoutedBlobs& routed,
    constant V4ExpertOffsets&   routed_offsets,
    device const half*          x,
    device half*                acts,
    uint slot,
    uint f,
    uint D,
    uint F,
    uint lane
) {
    device const uint8_t* base = routed.blob[slot];
    const V4ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* gS = base + re.gate_s_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const uint8_t* uS = base + re.up_s_off;

    const float2 gu = v4m_fp4_gate_up_rows(gW, gS, uW, uS, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(v4m_swiglu(gu.x, gu.y));
}

kernel void moe_v4_phase1_gate_up_act_swiglu(
    device const V4RoutedBlobs& routed         [[buffer(0)]],
    constant V4ExpertOffsets&   routed_offsets [[buffer(1)]],
    device const half*          x              [[buffer(2)]],
    device half*                acts           [[buffer(3)]],
    constant uint&              D              [[buffer(4)]],
    constant uint&              F              [[buffer(5)]],
    constant uint&              top_k          [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    v4m_phase1_body(routed, routed_offsets, x, acts, rowg / F, rowg % F, D, F, lane);
}

kernel void moe_v4_phase1_gate_up_act_swiglu_subset(
    device const V4RoutedBlobs& routed         [[buffer(0)]],
    constant V4ExpertOffsets&   routed_offsets [[buffer(1)]],
    device const half*          x              [[buffer(2)]],
    device half*                acts           [[buffer(3)]],
    constant uint&              D              [[buffer(4)]],
    constant uint&              F              [[buffer(5)]],
    constant uint&              top_k          [[buffer(6)]],
    device const uint*          active_slots   [[buffer(7)]],
    constant uint&              active_count   [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * F) return;
    const uint slot = active_slots[rowg / F];
    if (slot >= top_k) return;
    v4m_phase1_body(routed, routed_offsets, x, acts, slot, rowg % F, D, F, lane);
}

// One threadgroup per output element d; six SIMD groups, one per expert slot,
// each running the FP4 down-projection row. Lane 0 of each group writes the
// routing-weighted partial; thread 0 sums partials with the residual.
kernel void moe_v4_phase2_down_reduce_k6(
    device const V4RoutedBlobs& routed         [[buffer(0)]],
    constant V4ExpertOffsets&   routed_offsets [[buffer(1)]],
    device const half*          acts           [[buffer(2)]],
    device const float*         routing_w      [[buffer(3)]],
    device const half*          residual       [[buffer(4)]],
    device half*                y              [[buffer(5)]],
    constant uint&              D              [[buffer(6)]],
    constant uint&              F              [[buffer(7)]],
    uint d      [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[kV4MaxStreamedExperts];
    if (d >= D) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const V4ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const uint8_t* dS = base + re.down_s_off;
    device const half* act_slot = acts + sg_idx * F;

    const float value = v4m_fp4_gemv_row(dW, dS, act_slot, d, F, lane);
    if (lane == 0) partial[sg_idx] = routing_w[sg_idx] * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        for (uint i = 0; i < kV4MaxStreamedExperts; ++i) acc += partial[i];
        y[d] = half(acc);
    }
}

// ----------------------------------------------------------------------------
// Router: BF16 gate GEMV + sqrt-softplus/noaux_tc top-6 selection.
// ----------------------------------------------------------------------------

// logits[e] = W_bf16[e, :] · x. Four rows per threadgroup, one SIMD group per
// row, two elements per lane per step. Requires D % 64 == 0.
kernel void router_v4_gemv_bf16(
    device const bfloat* W          [[buffer(0)]],   // [num_experts, D] BF16
    device const half*   x          [[buffer(1)]],
    device float*        out_logits [[buffer(2)]],   // [num_experts] F32
    constant uint&       num_experts [[buffer(3)]],
    constant uint&       D           [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 4;
    const uint e = tg_idx * rows_per_tg + sg_idx;
    if (e >= num_experts) return;
    device const bfloat* W_row = W + uint(e) * D;

    float acc = 0.0f;
    for (uint base = 0; base < D; base += 64u) {
        const uint i0 = base + lane * 2u;
        acc = fma(float(W_row[i0]),      float(x[i0]),      acc);
        acc = fma(float(W_row[i0 + 1u]), float(x[i0 + 1u]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

// Reference Gate.forward selection:
//   s   = sqrt(softplus(logit))            (softplus via the same
//         l > 20 ? l : log(1 + exp(l)) form as the CPU reference)
//   sel = s + bias                          (static noaux_tc bias, selection only)
//   top-6 by sel, ties to the lower expert index (torch.topk order)
//   w_i = s[idx_i] / sum_j s[idx_j] * route_scale
// Bias never enters the weights.
kernel void router_v4_topk_select_k6(
    device const float* logits      [[buffer(0)]],   // [num_experts]
    device const float* bias        [[buffer(1)]],   // [num_experts]
    device uint*        out_indices [[buffer(2)]],   // [6]
    device float*       out_weights [[buffer(3)]],   // [6]
    constant uint&      num_experts [[buffer(4)]],
    constant float&     route_scale [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    uint  top_idx[6];
    float top_sel[6];
    float top_score[6];
    for (uint i = 0; i < 6; ++i) {
        top_idx[i] = 0u;
        top_sel[i] = -INFINITY;
        top_score[i] = 0.0f;
    }

    for (uint e = 0; e < num_experts; ++e) {
        const float l = logits[e];
        const float sp = (l > 20.0f) ? l : log(1.0f + exp(l));
        const float s = sqrt(max(sp, 0.0f));
        const float sel = s + bias[e];
        if (sel <= top_sel[5]) continue;
        uint pos = 6u;
        for (uint i = 0; i < 6; ++i) {
            if (sel > top_sel[i] || (sel == top_sel[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= 6u) continue;
        for (uint i = 5; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_sel[i] = top_sel[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_sel[pos] = sel;
        top_score[pos] = s;
    }

    float sum = 0.0f;
    for (uint i = 0; i < 6; ++i) sum += top_score[i];
    const float inv_sum = 1.0f / sum;
    for (uint i = 0; i < 6; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = top_score[i] * inv_sum * route_scale;
    }
}
