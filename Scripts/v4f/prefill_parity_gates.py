#!/usr/bin/env python3
"""V4F-06c-B acceptance gates: chunked prefill parity + speedup.

Compares token-exact greedy outputs and prefill wall-time between the
current serial-prefill path (prefillConfig .off) and the chunked V4
prefill runner. Usage:
  python3 Scripts/v4f/prefill_parity_gates.py [--tag NAME]

The script expects the CLI to support a flag or env var to select the
chunked prefill path once V4F-06c-B lands (the B worker is told to
wire it; the exact toggle is recorded here when it exists).
"""
import re, subprocess, sys, time

CLI = ".build/release/TurboFieldfareCLI"
MODEL = "scratch/deepseek-v4-flash.gturbo"

FRANCE = "The capital of France is"
STORY57 = ("The city of Marlowe sat at the bend of a slow river, and its bell tower marked the hours. "
           "Every autumn the townspeople held the Festival of Lanterns on the old stone bridge. "
           "Question: In which season is the Festival of Lanterns held? Answer:")
LONG231 = open("scratch/v4f-recon/long-prompt.txt").read() if __import__("os").path.exists("scratch/v4f-recon/long-prompt.txt") else None
HCA296 = open("scratch/v4f-recon/long-prompt-500.txt").read() if __import__("os").path.exists("scratch/v4f-recon/long-prompt-500.txt") else None

CASES = [("france32", FRANCE, 32), ("story57", STORY57, 24)]
if LONG231: CASES.append(("long231", LONG231, 48))
if HCA296: CASES.append(("hca296", HCA296, 48))

def run(prompt, max_new, extra_env=None):
    import os
    env = dict(os.environ)
    if extra_env: env.update(extra_env)
    t0 = time.time()
    r = subprocess.run([CLI, "--model", MODEL, "--prompt", prompt,
                        "--max-new", str(max_new), "--temperature", "0"],
                       capture_output=True, text=True, timeout=7200, env=env)
    footer = r.stderr[r.stderr.find("[family"):] if "[family" in r.stderr else ""
    match = re.search(r"\bprefill_s=([0-9]+(?:\.[0-9]+)?)", footer)
    prefill_seconds = float(match.group(1)) if match else None
    return r.returncode, r.stdout, footer, time.time() - t0, prefill_seconds

def main():
    tag = "gate"
    if "--tag" in sys.argv: tag = sys.argv[sys.argv.index("--tag") + 1]
    chunk_tokens = "128"
    if "--chunk-tokens" in sys.argv:
        chunk_tokens = sys.argv[sys.argv.index("--chunk-tokens") + 1]
    if chunk_tokens not in {"32", "64", "128"}:
        raise SystemExit("--chunk-tokens must be 32, 64, or 128")
    min_speedup = 5.0
    if "--min-speedup" in sys.argv:
        min_speedup = float(sys.argv[sys.argv.index("--min-speedup") + 1])
    failures = 0
    for name, prompt, n in CASES:
        rc_serial, out_serial, foot_serial, wall_serial, prefill_serial = run(prompt, n)
        rc_chunk, out_chunk, foot_chunk, wall_chunk, prefill_chunk = run(
            prompt, n, extra_env={
                "TURBO_V4_CHUNKED_PREFILL": "1",
                "TURBO_V4_PREFILL_CHUNK_TOKENS": chunk_tokens,
            })
        same = rc_serial == 0 and rc_chunk == 0 and out_serial == out_chunk
        timed = prefill_serial is not None and prefill_chunk is not None
        speedup = prefill_serial / max(prefill_chunk, 1e-9) if timed else 0.0
        fast_enough = timed and speedup >= min_speedup
        failures += 0 if same and fast_enough else 1
        print(f"{name}: parity={'PASS' if same else 'FAIL'} "
              f"prefill serial={prefill_serial!s}s chunk={prefill_chunk!s}s "
              f"speedup={speedup:.2f}x wall={wall_serial:.1f}/{wall_chunk:.1f}s", flush=True)
        if rc_serial or rc_chunk:
            print(f"  exit serial={rc_serial} chunk={rc_chunk}")
        if not timed:
            print("  missing prefill_s timing in one or both V4 footers")
        elif not fast_enough:
            print(f"  speedup below required {min_speedup:.2f}x")
        if not same:
            print(f"  serial: {out_serial[-120:]!r}")
            print(f"  chunk : {out_chunk[-120:]!r}")
    print(f"{tag}: {failures} parity failures")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
