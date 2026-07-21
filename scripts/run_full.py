#!/usr/bin/env python3
"""Run full LLM experiment on data/sampled_100.csv (N=100) — task 8.2 LR."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "sampled_100.csv"

if not DATASET.exists():
    print(f"Missing dataset: {DATASET}")
    sys.exit(1)

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "run_experiment.py"),
    "--dataset",
    str(DATASET),
    "--artifact-prefix",
    "full",
    "--prompt-profile",
    "pilot",
    "--checkpoint-interval",
    "50",
]
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd))
