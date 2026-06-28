#!/usr/bin/env python3
"""Run pilot experiment on data/pilot_sample.csv (N=20)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "pilot_sample.csv"

if not DATASET.exists():
    print(f"Missing pilot dataset: {DATASET}")
    print("Run first: python scripts/create_pilot.py")
    sys.exit(1)

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "run_experiment.py"),
    "--dataset",
    str(DATASET),
    "--artifact-prefix",
    "pilot",
    "--prompt-profile",
    "pilot",
    "--checkpoint-interval",
    "5",
]
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd))
