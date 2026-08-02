#!/usr/bin/env python3
"""Independent from-spec dequant + GEMV golden generator for the V4F
real-tensor validation gate (option C).

Reads the real DeepSeek V4-Flash layer-5 tensors staged in
scratch/v4f-recon/real-tensors/ (byte-exact range fetches from shard 7)
and writes golden outputs to scratch/v4f-recon/golden/:

  * dequantized matrices (raw little-endian float32 .bin)
  * a fixed-seed GEMV result per matrix, computed twice:
      - float64 accumulation (primary reference)
      - float32 accumulation (kernel-parity reference)
    The GEMV input x is fixed-seed standard normal, rounded to float16
    (the GPU kernels take fp16 activations), stored as float32.
  * router goldens: BF16 GEMV logits (f64), sqrt-softplus scores,
    top-6 selection indices and normalized route_scale=1.5 weights
  * fused SwiGLU MoE goldens for expert 0 (phase1 acts, phase2 output)

Format truth source: docs/experiments/recon/V4F-reference-notes.md section 7,
verified against DeepSeek's inference/kernel.py:
  * FP4 experts: e2m1 codes packed 2/byte, LOW nibble = element 2i
    (from cast_e2m1fn_to_e4m3fn: low = x & 0x0F, high = x >> 4),
    ue8m0 scale 2^(b-127) per 32 elements along K.
  * FP8 dense: e4m3 per OCP (subnormal step 2^-9, e=15/m=7 NaN),
    ue8m0 scale per 128x128 block, grid row-major.

numpy only. No torch, no pip installs.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECON = os.path.join(ROOT, "scratch", "v4f-recon")
REAL = os.path.join(RECON, "real-tensors")
GOLDEN = os.path.join(RECON, "golden")

SEED_BASE = 0x0004F000  # fixed; per-tensor seeds derived below
ROUTE_SCALE = np.float32(1.5)
TOP_K = 6
SWIGLU_LIMIT = 10.0
# Fixed routing weights for the fused-MoE phase-2 golden (expert 0's real
# blob is replicated into all 6 slots on the Swift side).
MOE_ROUTING = np.array([0.28, 0.24, 0.20, 0.16, 0.12, 0.08], dtype=np.float32)

# e2m1fn value table, index = 4-bit code (sign in bit 3). Exact in fp32.
E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def ue8m0(bytes_: np.ndarray) -> np.ndarray:
    """ue8m0 -> 2^(b - 127), exact power of two in fp32/fp64."""
    return np.ldexp(np.ones_like(bytes_, dtype=np.float64),
                    bytes_.astype(np.int32) - 127)


def dequant_fp4(container: np.ndarray, scales: np.ndarray,
                m: int, n: int) -> np.ndarray:
    """[M, N/2] uint8 of e2m1 pairs (low nibble = element 2i) +
    [M, N/32] ue8m0 -> [M, N] float32. Exact."""
    assert container.shape == (m, n // 2), (container.shape, m, n)
    assert scales.shape == (m, n // 32), (scales.shape, m, n)
    low = E2M1_LUT[container & 0x0F]          # element 2i
    high = E2M1_LUT[container >> 4]           # element 2i+1
    w = np.empty((m, n), dtype=np.float32)
    w[:, 0::2] = low
    w[:, 1::2] = high
    w *= np.repeat(ue8m0(scales), 32, axis=1).astype(np.float32)
    return w


def e4m3_decode(q: np.ndarray) -> np.ndarray:
    """OCP e4m3 decode, vectorized. NaN code (e=15,m=7) decodes to 480
    with a warning count; real checkpoints never carry it."""
    q = q.astype(np.uint8)
    sign = np.where(q & 0x80 != 0, -1.0, 1.0)
    e = ((q >> 3) & 0x0F).astype(np.int32)
    m = (q & 0x07).astype(np.int32)
    sub = m.astype(np.float64) * 2.0 ** -9
    norm = (8 + m).astype(np.float64) * np.exp2(e - 10)
    mag = np.where(e == 0, sub, norm)
    return (sign * mag).astype(np.float32)


def dequant_fp8_block(codes: np.ndarray, scales: np.ndarray,
                      m: int, n: int) -> np.ndarray:
    """[M, N] e4m3 + [ceil(M/128), ceil(N/128)] ue8m0 -> [M, N] float32.
    Exact."""
    assert codes.shape == (m, n), (codes.shape, m, n)
    gm, gn = (m + 127) // 128, (n + 127) // 128
    assert scales.shape == (gm, gn), (scales.shape, gm, gn)
    w = e4m3_decode(codes)
    s = ue8m0(scales).astype(np.float32)
    s_full = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)[:m, :n]
    return w * s_full


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Raw BF16 uint16 array -> float32 via bit shift."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def fixed_x(seed: int, n: int) -> np.ndarray:
    """Fixed-seed standard normal, rounded to fp16 (kernel input format),
    returned as float32."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n, dtype=np.float32)
    return x.astype(np.float16).astype(np.float32)


def gemv_pair(w32: np.ndarray, x: np.ndarray):
    """(float64-accumulated, float32-accumulated) GEMV."""
    g64 = w32.astype(np.float64) @ x.astype(np.float64)
    g32 = w32 @ x  # numpy f32 matmul, f32 accumulation
    return g64, g32


def save(name: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    path = os.path.join(GOLDEN, name)
    arr.tofile(path)
    return {"file": name, "shape": list(arr.shape), "dtype": str(arr.dtype)}


def softplus(l: np.ndarray) -> np.ndarray:
    # Same branch form as the reference/kernel: l > 20 ? l : log(1 + exp(l)).
    return np.where(l > 20.0, l, np.log1p(np.exp(l)))


def silu_stable(g: np.ndarray) -> np.ndarray:
    e = np.exp(-np.abs(g))
    sig = np.where(g >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return g * sig


def load_raw(stem: str, dtype: np.dtype, shape):
    path = os.path.join(REAL, stem + ".bin")
    arr = np.fromfile(path, dtype=dtype)
    n = int(np.prod(shape))
    assert arr.size == n, f"{stem}: {arr.size} elements, expected {n} for {shape}"
    return arr.reshape(shape)


def stats(name: str, w: np.ndarray) -> dict:
    return {
        "tensor": name,
        "min": float(w.min()), "max": float(w.max()),
        "mean": float(w.mean()), "std": float(w.std()),
        "abs_max": float(np.abs(w).max()),
    }


def main():
    os.makedirs(GOLDEN, exist_ok=True)
    with open(os.path.join(RECON, "tensor-ranges.json")) as f:
        ranges = {e["name"]: e for e in json.load(f)}

    manifest = {"seed_base": SEED_BASE, "route_scale": float(ROUTE_SCALE),
                "top_k": TOP_K, "swiglu_limit": SWIGLU_LIMIT,
                "moe_routing": MOE_ROUTING.tolist(), "tensors": {}, "sanity": []}

    # name key -> (stem, kind, container shape, logical shape)
    fp4 = {
        "w1": ("layers.5.ffn.experts.0.w1", (2048, 2048), (2048, 4096)),
        "w2": ("layers.5.ffn.experts.0.w2", (4096, 1024), (4096, 2048)),
        "w3": ("layers.5.ffn.experts.0.w3", (2048, 2048), (2048, 4096)),
    }
    fp8 = {
        "wq_a": ("layers.5.attn.wq_a", (1024, 4096), (8, 32)),
        "wq_b": ("layers.5.attn.wq_b", (32768, 1024), (256, 8)),
        "shared_w1": ("layers.5.ffn.shared_experts.w1", (2048, 4096), (16, 32)),
    }

    dequants = {}

    # Cross-check container shapes against the staged metadata.
    for key, (stem, cshape, lshape) in fp4.items():
        meta = ranges[stem + ".weight"]
        assert tuple(meta["shape"]) == cshape, (key, meta["shape"], cshape)
        smeta = ranges[stem + ".scale"]
        assert tuple(smeta["shape"]) == (lshape[0], lshape[1] // 32), key
    for key, (stem, cshape, sshape) in fp8.items():
        meta = ranges[stem + ".weight"]
        assert tuple(meta["shape"]) == cshape, (key, meta["shape"], cshape)
        smeta = ranges[stem + ".scale"]
        assert tuple(smeta["shape"]) == sshape, key

    for i, (key, (stem, cshape, lshape)) in enumerate(sorted(fp4.items())):
        m, n = lshape
        container = load_raw(stem + ".weight", np.uint8, cshape)
        scales = load_raw(stem + ".scale", np.uint8, (m, n // 32))
        w = dequant_fp4(container, scales, m, n)
        dequants[key] = w
        seed = SEED_BASE + 16 * i
        x = fixed_x(seed, n)
        g64, g32 = gemv_pair(w, x)
        entry = {"kind": "fp4_e2m1_ue8m0_g32", "stem": stem, "logical_shape": [m, n],
                 "container_shape": list(cshape), "seed": seed,
                 "dequant": save(f"{key}.dequant.f32.bin", w),
                 "x": save(f"{key}.x.f32.bin", x),
                 "gemv_f64": save(f"{key}.gemv.f64.bin", g64),
                 "gemv_f32": save(f"{key}.gemv.f32.bin", g32),
                 "scale_byte_min": int(scales.min()), "scale_byte_max": int(scales.max())}
        manifest["tensors"][key] = entry
        manifest["sanity"].append(stats(key, w))
        print(f"[fp4] {key} {lshape} seed={seed} scaleBytes=[{scales.min()},{scales.max()}] "
              f"dequant std={w.std():.6f} gemv64|max|={np.abs(g64).max():.6f}")

    for i, (key, (stem, cshape, sshape)) in enumerate(sorted(fp8.items())):
        m, n = cshape
        codes = load_raw(stem + ".weight", np.uint8, cshape)
        scales = load_raw(stem + ".scale", np.uint8, sshape)
        nan_codes = int(((codes & 0x7F) == 0x7F).sum())
        w = dequant_fp8_block(codes, scales, m, n)
        dequants[key] = w
        seed = SEED_BASE + 16 * (10 + i)
        x = fixed_x(seed, n)
        g64, g32 = gemv_pair(w, x)
        entry = {"kind": "fp8_e4m3_ue8m0_b128", "stem": stem, "logical_shape": [m, n],
                 "seed": seed,
                 "dequant": save(f"{key}.dequant.f32.bin", w),
                 "x": save(f"{key}.x.f32.bin", x),
                 "gemv_f64": save(f"{key}.gemv.f64.bin", g64),
                 "gemv_f32": save(f"{key}.gemv.f32.bin", g32),
                 "scale_byte_min": int(scales.min()), "scale_byte_max": int(scales.max()),
                 "e4m3_nan_codes": nan_codes}
        manifest["tensors"][key] = entry
        manifest["sanity"].append(stats(key, w))
        print(f"[fp8] {key} {cshape} seed={seed} scaleBytes=[{scales.min()},{scales.max()}] "
              f"nanCodes={nan_codes} dequant std={w.std():.6f} gemv64|max|={np.abs(g64).max():.6f}")

    # --- Router (gate.weight BF16 [256,4096], gate.bias F32 [256]) ---
    gate_w = bf16_to_f32(load_raw("layers.5.ffn.gate.weight", np.uint16, (256, 4096)))
    gate_b = load_raw("layers.5.ffn.gate.bias", np.float32, (256,))
    seed = SEED_BASE + 16 * 20
    x = fixed_x(seed, 4096)
    logits64 = gate_w.astype(np.float64) @ x.astype(np.float64)
    scores = np.sqrt(np.maximum(softplus(logits64), 0.0))
    sel = scores + gate_b.astype(np.float64)
    order = np.argsort(-sel, kind="stable")  # stable: ties to lower index
    idx = order[:TOP_K].astype(np.uint32)
    gathered = scores[idx]
    weights = (gathered / gathered.sum() * float(ROUTE_SCALE)).astype(np.float32)
    manifest["tensors"]["router"] = {
        "kind": "bf16_gemv_sqrtsoftplus_top6",
        "stem": "layers.5.ffn.gate", "logical_shape": [256, 4096], "seed": seed,
        "dequant": save("router.dequant.f32.bin", gate_w),
        "bias": save("router.bias.f32.bin", gate_b),
        "x": save("router.x.f32.bin", x),
        "logits_f64": save("router.logits.f64.bin", logits64),
        "scores_f32": save("router.scores.f32.bin", scores.astype(np.float32)),
        "indices_u32": save("router.indices.u32.bin", idx),
        "weights_f32": save("router.weights.f32.bin", weights),
    }
    manifest["sanity"].append(stats("router_weight", gate_w))
    margin = float(sel[idx[5]] - np.partition(sel, -TOP_K - 1)[-TOP_K - 1])
    manifest["tensors"]["router"]["selection_margin_rank6_vs_rank7"] = margin
    print(f"[router] seed={seed} idx={idx.tolist()} weights={weights.tolist()} "
          f"margin6v7={margin:.6e}")

    # --- Fused SwiGLU MoE, expert 0 (phase1 acts + phase2 output) ---
    seed = SEED_BASE + 16 * 21
    xm = fixed_x(seed, 4096)
    w1, w2, w3 = dequants["w1"].astype(np.float64), dequants["w2"].astype(np.float64), dequants["w3"].astype(np.float64)
    x64 = xm.astype(np.float64)
    gate = w1 @ x64
    up = w3 @ x64
    g = np.minimum(gate, SWIGLU_LIMIT)          # no lower clamp on gate
    u = np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    act = silu_stable(g) * u
    acts16 = act.astype(np.float16).astype(np.float32)  # kernel stores fp16 acts
    rng = np.random.default_rng(SEED_BASE + 16 * 22)
    residual = rng.standard_normal(4096, dtype=np.float32).astype(np.float16).astype(np.float32)
    down64 = w2 @ acts16.astype(np.float64)
    out64 = residual.astype(np.float64) + float(MOE_ROUTING.sum()) * down64
    # float32-accumulated variant
    down32 = dequants["w2"] @ acts16
    out32 = residual + MOE_ROUTING.sum() * down32
    manifest["tensors"]["moe"] = {
        "kind": "fused_swiglu_moe_expert0", "seed_x": seed, "seed_residual": SEED_BASE + 16 * 22,
        "x": save("moe.x.f32.bin", xm),
        "acts_f32": save("moe.acts.f32.bin", acts16),
        "residual_f32": save("moe.residual.f32.bin", residual),
        "routing_f32": save("moe.routing.f32.bin", MOE_ROUTING),
        "out_f64": save("moe.out.f64.bin", out64),
        "out_f32": save("moe.out.f32.bin", out32),
        "gate_preact_min": float(gate.min()), "gate_preact_max": float(gate.max()),
        "up_preact_min": float(up.min()), "up_preact_max": float(up.max()),
    }
    print(f"[moe] seed={seed} gate preact [{gate.min():.3f},{gate.max():.3f}] "
          f"up preact [{up.min():.3f},{up.max():.3f}] "
          f"clamps: gate>10 {int((gate > 10).sum())}, gate<-10 {int((gate < -10).sum())}, "
          f"|up|>10 {int((np.abs(up) > 10).sum())}")

    with open(os.path.join(GOLDEN, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote goldens + manifest to {GOLDEN}")


if __name__ == "__main__":
    main()
