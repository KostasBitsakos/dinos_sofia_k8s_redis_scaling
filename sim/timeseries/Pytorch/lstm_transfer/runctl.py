"""Run-control helpers shared by the lstm_transfer scripts: output location, seeding, telemetry (PLAN section 7)."""
import json
import os
import random

import numpy as np


def results_folder(default):
    """$QRL_RUN_DIR/results if QRL_RUN_DIR is set, else the legacy folder name."""
    run_dir = os.environ.get("QRL_RUN_DIR")
    return os.path.join(run_dir, "results") if run_dir else default


def seed_everything(seed):
    """seed=None keeps the old unseeded behaviour. Environment and agent use the global np.random / random."""
    if seed is None:
        return
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_metrics(**metrics):
    run_dir = os.environ.get("QRL_RUN_DIR")
    if not run_dir:
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(metrics) + "\n")


def write_progress(step, total):
    run_dir = os.environ.get("QRL_RUN_DIR")
    if not run_dir:
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "progress.json"), "w") as f:
        json.dump({"step": step, "total": total}, f)


def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--legacy", action="store_true")
    env_seed = os.environ.get("QRL_RUN_SEED")  # exported by tools/launch_run.py from the frozen spec
    p.add_argument("--seed", type=int, default=int(env_seed) if env_seed not in (None, "") else None)
    p.add_argument("--model-steps", default="1,2,5,10,20")
    p.add_argument("--eval-steps", type=int, default=5000)
    p.add_argument("--num-evaluations", type=int, default=5)
    p.add_argument("--fine-tune-steps", type=int, default=1000)
    a = p.parse_args(argv)
    a.model_steps = tuple(int(x) for x in a.model_steps.split(","))
    return a
