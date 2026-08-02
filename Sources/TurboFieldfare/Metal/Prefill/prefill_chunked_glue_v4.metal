#include <metal_stdlib>
using namespace metal;

// v4cg_* — V4 chunked-prefill batched glue kernels.
// Kept in its own module so MetalContext registration can remain centralized.

static inline float v4cg_bf16(uint16_t bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline float v4cg_sqrtsoftplus(float x) {
    // Stable softplus: max(x, 0) + log(1 + exp(-abs(x))).
    const float sp = max(x, 0.0f) + log(1.0f + exp(-fabs(x)));
    return sqrt(sp);
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

kernel void v4cg_router_top6_sqrtsoftplus(
    device const float* logits [[buffer(0)]],     // [rows, 256]
    device const float* staticBias [[buffer(1)]], // [256], selection only
    device int* outIDs [[buffer(2)]],             // [rows, 6]
    device float* outWeights [[buffer(3)]],       // [rows, 6]
    constant uint& rows [[buffer(4)]],
    constant float& routeScale [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) { return; }

    int bestID[6] = {0, 1, 2, 3, 4, 5};
    float bestKey[6];
    for (uint i = 0; i < 6; ++i) {
        const uint eid = i;
        bestKey[i] = v4cg_sqrtsoftplus(logits[uint64_t(row) * 256ull + eid] + staticBias[eid]);
    }
    // Sort initial six by descending key, lower ID first for ties.
    for (uint i = 1; i < 6; ++i) {
        const int id = bestID[i];
        const float key = bestKey[i];
        int j = int(i) - 1;
        while (j >= 0 && (key > bestKey[j] || (key == bestKey[j] && id < bestID[j]))) {
            bestID[j + 1] = bestID[j];
            bestKey[j + 1] = bestKey[j];
            --j;
        }
        bestID[j + 1] = id;
        bestKey[j + 1] = key;
    }

    for (uint eid = 6; eid < 256; ++eid) {
        const float key = v4cg_sqrtsoftplus(logits[uint64_t(row) * 256ull + eid] + staticBias[eid]);
        if (key > bestKey[5] || (key == bestKey[5] && int(eid) < bestID[5])) {
            int j = 4;
            while (j >= 0 && (key > bestKey[j] || (key == bestKey[j] && int(eid) < bestID[j]))) {
                bestID[j + 1] = bestID[j];
                bestKey[j + 1] = bestKey[j];
                --j;
            }
            bestID[j + 1] = int(eid);
            bestKey[j + 1] = key;
        }
    }

    float sum = 0.0f;
    float scores[6];
    for (uint i = 0; i < 6; ++i) {
        scores[i] = v4cg_sqrtsoftplus(logits[uint64_t(row) * 256ull + uint(bestID[i])]);
        sum += scores[i];
    }
    const float inv = (sum > 0.0f) ? (routeScale / sum) : 0.0f;
    for (uint i = 0; i < 6; ++i) {
        outIDs[uint64_t(row) * 6ull + i] = bestID[i];
        outWeights[uint64_t(row) * 6ull + i] = scores[i] * inv;
    }
}

kernel void v4cg_hash_router_weights_sqrtsoftplus(
    device const float* logits [[buffer(0)]], // [rows, 256]
    device const int* tid2eid [[buffer(1)]],   // [rows, 6]
    device float* outWeights [[buffer(2)]],    // [rows, 6]
    constant uint& rows [[buffer(3)]],
    constant float& routeScale [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) { return; }
    float scores[6];
    float sum = 0.0f;
    for (uint i = 0; i < 6; ++i) {
        const int eid = tid2eid[uint64_t(row) * 6ull + i];
        const float s = v4cg_sqrtsoftplus(logits[uint64_t(row) * 256ull + uint(eid)]);
        scores[i] = s;
        sum += s;
    }
    const float inv = (sum > 0.0f) ? (routeScale / sum) : 0.0f;
    for (uint i = 0; i < 6; ++i) {
        outWeights[uint64_t(row) * 6ull + i] = scores[i] * inv;
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
