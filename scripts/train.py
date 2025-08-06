#!/usr/bin/env python3
"""
CLI entry to train one or all models with optional hyper-param overrides.
"""
import argparse, subprocess, sys
from pathlib import Path

# Add parent directory to path for src imports
sys.path.append(str(Path(__file__).parent.parent))

import src.config as cfg
AVAILABLE = ["lgb", "enet", "cat", "xgb"]

def run(model: str, trials: int, folds: int, seed: int):
    cmd = [
        sys.executable, "-m", "src.train_models",
        "--model", model,
        "--trials", str(trials),
        "--folds",  str(folds),
        "--seed",   str(seed),
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=AVAILABLE + ["all"], required=True)
    ap.add_argument("--trials", type=int, default=cfg.N_TRIALS)
    ap.add_argument("--folds",  type=int, default=cfg.N_SPLITS)
    ap.add_argument("--seed",   type=int, default=cfg.RANDOM_STATE)
    args = ap.parse_args()

    targets = AVAILABLE if args.model == "all" else [args.model]
    for m in targets:
        run(m, args.trials, args.folds, args.seed)