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
import subprocess, sys, time

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
    return r.stdout, footer, time.time() - t0

def main():
    tag = "gate"
    if "--tag" in sys.argv: tag = sys.argv[sys.argv.index("--tag") + 1]
    failures = 0
    for name, prompt, n in CASES:
        out_serial, foot_serial, t_serial = run(prompt, n)
        out_chunk, foot_chunk, t_chunk = run(prompt, n, extra_env={"TURBO_V4_CHUNKED_PREFILL": "1"})
        same = out_serial == out_chunk
        failures += 0 if same else 1
        speedup = t_serial / max(t_chunk, 1e-9)
        print(f"{name}: parity={'PASS' if same else 'FAIL'} "
              f"serial {t_serial:.1f}s chunk {t_chunk:.1f}s speedup {speedup:.2f}x", flush=True)
        if not same:
            print(f"  serial: {out_serial[-120:]!r}")
            print(f"  chunk : {out_chunk[-120:]!r}")
    print(f"{tag}: {failures} parity failures")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
