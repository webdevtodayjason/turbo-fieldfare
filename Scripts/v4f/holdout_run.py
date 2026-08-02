#!/usr/bin/env python3
"""V4F-05 holdout runner for DeepSeek V4-Flash (see
docs/experiments/V4F-05-holdout-plan.md).

Runs the three community real-generation cases against the V4 install
with declared protocol deviations: DSML chat framing via --messages-file,
V4 manifest SHA, HCA-active 4K context. One warmup + one measured run
per case; fixed seeds; results under scratch/v4f-holdout/.

Usage: python3 Scripts/v4f/holdout_run.py
"""
import json, os, subprocess, sys, time

CLI = ".build/release/TurboFieldfareCLI"
MODEL = "scratch/deepseek-v4-flash.gturbo"
OUT = "scratch/v4f-holdout"
CASES = [
    ("short-explanation", "docs/benchmark-prompts/real-generation-v1/short-explanation.json", 20260721),
    ("medium-review", "docs/benchmark-prompts/real-generation-v1/medium-review.json", 20260722),
    ("long-synthesis", "docs/benchmark-prompts/real-generation-v1/long-synthesis.json", 20260723),
]
MAX_NEW = 1024
CTX = 4096

def case_messages(path):
    body = json.load(open(path))
    # Community prompts are chat-framed JSON; reuse as user messages.
    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict) and "messages" in body:
        rows = body["messages"]
    else:
        rows = [{"role": "user", "content": body if isinstance(body, str) else json.dumps(body)}]
    tmp = os.path.join(OUT, "tmp-messages.json")
    json.dump(rows, open(tmp, "w"))
    return tmp

def run(tag, messages_file, seed):
    t0 = time.time()
    r = subprocess.run([CLI, "--model", MODEL, "--messages-file", messages_file,
                        "--max-new", str(MAX_NEW), "--max-context", str(CTX),
                        "--temperature", "0.2", "--top-k", "64", "--top-p", "0.95",
                        "--seed", str(seed)],
                       capture_output=True, text=True, timeout=28800)
    dt = time.time() - t0
    open(os.path.join(OUT, f"{tag}.txt"), "w").write(r.stdout + "\n---STDERR---\n" + r.stderr)
    return r.returncode, dt

def main():
    os.makedirs(OUT, exist_ok=True)
    # System record (declared deviations: V4 manifest SHA).
    with open(os.path.join(OUT, "system.txt"), "w") as f:
        for cmd in [["git", "rev-parse", "HEAD"], ["sw_vers"], ["swift", "--version"],
                    ["shasum", "-a", "256", os.path.join(MODEL, "manifest.json")]]:
            f.write("$ " + " ".join(cmd) + "\n")
            f.write(subprocess.run(cmd, capture_output=True, text=True).stdout + "\n")
    for name, path, seed in CASES:
        mf = case_messages(path)
        code, _ = run(f"{name}-warmup", mf, seed)      # discarded warmup
        code, dt = run(f"{name}-measured", mf, seed)   # measured run
        print(f"{name}: exit {code} wall {dt:.1f}s", flush=True)
        if code != 0:
            sys.exit(f"case {name} failed with exit {code}")
    print("done; results in", OUT)

if __name__ == "__main__":
    main()
