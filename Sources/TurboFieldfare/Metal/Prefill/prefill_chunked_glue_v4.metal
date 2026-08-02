#include <metal_stdlib>
using namespace metal;

// v4cg_* — V4 chunked-prefill batched glue kernels.
// Kept in its own module so MetalContext registration can remain centralized.

static inline float v4cg_bf16(uint16_t bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline float v4cg_sqrtsoftplus(float x) {
    // Exact arithmetic shape used by router_v4_topk_select_k6.
    const float sp = x > 20.0f ? x : log(1.0f + exp(x));
    return sqrt(max(sp, 0.0f));
}

static inline float v4cg_hash_sqrtsoftplus(float x) {
    // Exact arithmetic shape used by v4c_router_weights_at_indices.
    const float sp = x > 20.0f ? x : fast::log(1.0f + fast::exp(x));
    return fast::sqrt(max(sp, 0.0f));
}

kernel void v4cg_bf16_embedding_gather_broadcast(
    device const uint16_t* embeddings [[buffer(0)]], // [vocab, dim] bf16 bits
    device const int* tokenIDs [[buffer(1)]],         // [rows]
    device float* out [[buffer(2)]],                  // [rows, 4, dim]
    constant uint& rows [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = rows * dim;
    if (gid >= total) { return; }
    const uint row = gid / dim;
    const uint d = gid - row * dim;
    const int token = tokenIDs[row];
    const float value = v4cg_bf16(embeddings[uint64_t(token) * dim + d]);
    const uint64_t base = (uint64_t(row) * 4ull * dim) + d;
    out[base] = value;
    out[base + dim] = value;
    out[base + uint64_t(2u) * dim] = value;
    out[base + uint64_t(3u) * dim] = value;
}

// Batched form of `router_v4_gemv_bf16` with deliberately identical
// per-lane accumulation and SIMD reduction order. It is generic over output
// rows and covers routers, compressor projections, and indexer weights.
kernel void v4cg_bf16_gemm_serial_order(
    device const uint16_t* W [[buffer(0)]], // [outRows, dim] bf16 bits
    device const half* X [[buffer(1)]],     // [rows, dim]
    device float* output [[buffer(2)]],     // [rows, outRows]
    constant uint& rows [[buffer(3)]],
    constant uint& outRows [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint outputTiles = (outRows + 3u) / 4u;
    const uint row = tg / outputTiles;
    const uint outputRow = (tg % outputTiles) * 4u + sg;
    if (row >= rows || outputRow >= outRows) { return; }

    device const uint16_t* Wrow = W + uint64_t(outputRow) * dim;
    device const half* Xrow = X + uint64_t(row) * dim;
    float acc = 0.0f;
    for (uint base = 0; base < dim; base += 64u) {
        const uint i0 = base + lane * 2u;
        acc = fma(v4cg_bf16(Wrow[i0]), float(Xrow[i0]), acc);
        acc = fma(v4cg_bf16(Wrow[i0 + 1u]), float(Xrow[i0 + 1u]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) { output[uint64_t(row) * outRows + outputRow] = acc; }
}

kernel void v4cg_router_top6_sqrtsoftplus(
    device const float* logits [[buffer(0)]],     // [rows, 256]
    device const float* staticBias [[buffer(1)]], // [256], selection only
    device uint* outIDs [[buffer(2)]],            // [rows, 6]
    device float* outWeights [[buffer(3)]],       // [rows, 6]
    constant uint& rows [[buffer(4)]],
    constant float& routeScale [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (row >= rows || tid != 0u) { return; }
    device const float* rowLogits = logits + uint64_t(row) * 256ull;
    device uint* rowIDs = outIDs + uint64_t(row) * 6ull;
    device float* rowWeights = outWeights + uint64_t(row) * 6ull;

    // Deliberately mirror router_v4_topk_select_k6 statement-for-statement.
    uint top_idx[6];
    float top_sel[6];
    float top_score[6];
    for (uint i = 0; i < 6; ++i) {
        top_idx[i] = 0u;
        top_sel[i] = -INFINITY;
        top_score[i] = 0.0f;
    }
    for (uint e = 0; e < 256u; ++e) {
        const float l = rowLogits[e];
        const float sp = (l > 20.0f) ? l : log(1.0f + exp(l));
        const float s = sqrt(max(sp, 0.0f));
        const float sel = s + staticBias[e];
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
        rowIDs[i] = top_idx[i];
        rowWeights[i] = top_score[i] * inv_sum * routeScale;
    }
}

kernel void v4cg_hash_router_weights_sqrtsoftplus(
    device const float* logits [[buffer(0)]], // [rows, 256]
    device const int* tid2eid [[buffer(1)]],   // [rows, 6]
    device float* outWeights [[buffer(2)]],    // [rows, 6]
    constant uint& rows [[buffer(3)]],
    constant float& routeScale [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (row >= rows || lid != 0u) { return; }
    device const float* rowLogits = logits + uint64_t(row) * 256ull;
    device const int* rowIDs = tid2eid + uint64_t(row) * 6ull;
    device float* rowWeights = outWeights + uint64_t(row) * 6ull;

    // Deliberately mirror v4c_router_weights_at_indices statement-for-statement.
    float w[32];
    float sum = 0.0f;
    for (uint i = 0; i < 6; ++i) {
        const float s = rowLogits[rowIDs[i]];
        const float sp = s > 20.0f ? s : fast::log(1.0f + fast::exp(s));
        w[i] = fast::sqrt(max(sp, 0.0f));
        sum += w[i];
    }
    for (uint i = 0; i < 6; ++i) {
        rowWeights[i] = w[i] / sum * routeScale;
    }
}

kernel void v4cg_add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) { return; }
    out[gid] = a[gid] + b[gid];
}
