#!/usr/bin/env python3
"""Long-context validation ladder for the V4-Flash port (V4F-05/06c).

Runs a series of increasingly long story-QA prompts through the release CLI
and checks that the answer stays correct as CSA then HCA compressed paths
activate. Serial prefill makes >1K prompts slow (~0.7 s/token) until the
prefill kernels land; the ladder is sized for that.

Usage: python3 Scripts/v4f/longctx_ladder.py [--max-level N]
"""
import subprocess, sys, time

CLI = ".build/release/TurboFieldfareCLI"
MODEL = "scratch/deepseek-v4-flash.gturbo"

SENTENCES = [
    "The city of Marlowe sat at the bend of a slow river, and for three hundred years its bell tower had marked the hours with a sound like bronze rain.",
    "Every autumn the townspeople held the Festival of Lanterns, when children carried paper lights across the old stone bridge and the bakers sold saffron buns still warm from the oven.",
    "In the year of the great flood, the river rose past the second arch and the miller lost his wheel, yet the tower kept time above the water.",
    "Old Ines, who had seen seventy festivals, said the lanterns looked even better reflected in the floodwater, doubled by the dark mirror of the drowned streets.",
    "The town rebuilt the mill that winter with timber from the hill forests, and when spring came the river returned to its banks as if nothing had happened.",
    "Generations later, visitors still climb the tower to see the bend of the river and hear the bronze rain.",
    "Years later a historian asked Old Ines why the festival survived the flood, and she said the lanterns were never about the streets but about the people carrying them.",
    "The museum on the square keeps one paper lantern from the flood year, repaired with rice paste and patience.",
    "Musicians played in the taverns until the candles burned low, and the songs named every bridge and every baker.",
    "On quiet nights the miller swore he could still hear the flood in the stones of the new wheel.",
]
QUESTION = " Question: In which season is the Festival of Lanterns held? Answer:"
# ~44 tokens per sentence pair; levels chosen to bracket: window-only,
# CSA active (132+), HCA active (256+), and deeper.
LEVELS = [2, 4, 8, 12, 20, 30, 50]   # sentence counts
ANSWERS = ("autumn", "fall")

def run(prompt, max_new=24):
    t0 = time.time()
    r = subprocess.run([CLI, "--model", MODEL, "--prompt", prompt,
                        "--max-new", str(max_new), "--temperature", "0"],
                       capture_output=True, text=True, timeout=7200)
    out = r.stdout
    footer = ""
    if "[family" in r.stderr:
        footer = r.stderr[r.stderr.find("[family"):]
    elif "[family" in out:
        footer = out[out.find("[family"):]
        out = out[:out.find("[family")]
    return r.returncode, out.strip(), footer.strip(), time.time() - t0

def main():
    max_level = int(sys.argv[sys.argv.index("--max-level") + 1]) if "--max-level" in sys.argv else len(LEVELS)
    print(f"{'sentences':>9} {'exit':>4} {'ok':>3}  answer excerpt / footer")
    failures = 0
    for ns in LEVELS[:max_level]:
        body = " ".join((SENTENCES * (ns // len(SENTENCES) + 1))[:ns])
        code, out, footer, dt = run(body + QUESTION)
        ok = any(a in out.lower() for a in ANSWERS)
        failures += 0 if ok else 1
        print(f"{ns:>9} {code:>4} {'YES' if ok else 'NO':>3}  {out[:80]!r}  {footer}  {dt:.0f}s", flush=True)
    print(f"done: {failures} failures")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
