#!/usr/bin/env python3
"""Source-task pretraining of the DInos+T LSTM trunk on Sim v2 features (cycle 1).

SOURCE TASK (documented, differs from the evaluation task):
  - simulator: sim v2, reward_mode "dinos", reward variant v2 (default) or d_sec (--reward-variant), same 6 base features
    and 3 actions as the target, so the trunk input is feature-compatible (no affine map to the old DInos features);
  - workload family W1 (daily two-peak profile + MMPP bursts, Poisson arrivals) instead of the evaluation family W0
    (sine + spike); the target task is W0;
  - simulator trace seeds 5000-5999 (the target uses training seeds 0-999 and evaluates on 1000-1004 / 1100-1109), episodes
    cycle through the block in order.
  - agent: lstm_scratch with the cycle-1 tuned options (see DEFAULT_CFG below), features "base"; saved object is the
    online LSTM state_dict (trunk + Q head). quick_test config "pretrained_path" loads it into dinos_t (the head is
    re-initialised when agent_options.reset_head is true).
The CPU cost of this script is charged to the budget. Output: <out>.pth and <out>.json (task description, settings,
per-episode source returns, CPU seconds).

  pretrain_source.py --steps 100000 --seed 0 --out tuning/checkpoints/c1_source_w1_v2 [--reward-variant v2]
Telemetry: metrics.jsonl / progress.json under $QRL_RUN_DIR when set.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from tuning import quick_test  # noqa: E402

SOURCE_WORKLOAD = "W1"
SOURCE_SEEDS = tuple(range(5000, 6000))
DEFAULT_CFG = {"agent": "lstm_scratch", "learn_scale": 0.001, "features": "base",
               "agent_options": {"eps_schedule": "linear", "eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 37500,
                                 "huber_beta": 1.0, "grad_clip": 10.0, "n_step": 3, "explore_repeat_max": 8,
                                 "gamma": 0.99, "lr": 1.5e-05}}


def pretrain(steps, seed, out, reward_variant="v2", tel_dir=None, workload=SOURCE_WORKLOAD, seeds=SOURCE_SEEDS):
    import torch
    cfg = dict(DEFAULT_CFG, reward_variant=reward_variant)
    cfg["agent_options"] = dict(cfg["agent_options"], eps_decay_steps=max(int(0.375 * steps), 1))
    holder = []
    t0 = time.process_time()
    res = quick_test.run(cfg, seed, steps, eval_seeds=[], tel_dir=tel_dir, agent_out=holder, workload=workload,
                         train_sim_seeds=list(seeds))
    cpu = time.process_time() - t0
    torch.save(holder[0].online_model.state_dict(), out + ".pth")
    meta = dict(source_task=dict(simulator="sim_v2 reward_mode=dinos", workload=workload,
                                 sim_seed_block=[seeds[0], seeds[-1]], reward_variant=reward_variant, features="base"),
                config=cfg, seed=seed, steps=steps, cpu_s_total=cpu, train_cpu_s=res["train_cpu_s"],
                grad_updates=res["grad_updates"],
                episode_returns=[e["ret"] for e in res["train_episodes"]],
                episode_sim_seeds=[e["sim_seed"] for e in res["train_episodes"]])
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED", 0)))
    p.add_argument("--out", required=True)
    p.add_argument("--reward-variant", default="v2")
    a = p.parse_args(argv)
    m = pretrain(a.steps, a.seed, a.out, a.reward_variant, os.environ.get("QRL_RUN_DIR"))
    print(json.dumps({"out": a.out + ".pth", "cpu_s_total": m["cpu_s_total"], "steps": a.steps}))


if __name__ == "__main__":
    main()
