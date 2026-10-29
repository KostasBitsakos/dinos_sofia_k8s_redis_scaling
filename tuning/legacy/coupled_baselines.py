#!/usr/bin/env python3
"""Cycle 4 baselines on the coupled legacy env (tuning/legacy/coupled_env.py), selected on val, reported on val + test.

  python3 tuning/legacy/coupled_baselines.py --out tuning/results/c4/baselines.json [--config cfg.json] [--seed 0]

Policies (all greedy, deterministic; nothing is trained):
  always_down       action 0 every tick
  hpa_rule_50_50    the original threshold rule P/HPA/simpleagent.py (cpu > 50 or mem > 50 -> +1, else -1), verbatim
  hpa_rule_tuned    the same rule with (T_c, T_m) in grid x grid (default {10,20,...,90}^2), best mean return on val
  fixed_ramped_N    move +-1 per tick from the start (6 pods) to N, then hold; N in 1..20, best on val
  fixed_prepro_N    pods set to N right after reset (preprovisioned), then hold; N in 1..20, best on val
Splits = run_legacy.SPLITS: val 1100-1104, test 2000-2019, test2 3000-3019, test4 5000-5019 (numpy/random/torch reseeded per episode,
env reset, 5000 ticks). Selection uses ONLY val; each --report-splits split (default test; cycle 6: test2 or test,test2)
is evaluated for the selected policies (and the two fixed ones) after it, stored under its split name.
Env: --env (default dense, as the rule agents in run_legacy) wrapped by --env-variant (default coupled).
Single thread, one process. --seed is recorded and seeds the process (no policy here draws random numbers).
Telemetry (PLAN 7): $QRL_RUN_DIR/metrics.jsonl (step = episodes evaluated) and progress.json.
--config: optional JSON with keys "grid" (threshold values) and "fixed" (pod counts); defaults below.
"""
import argparse
import contextlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tuning.legacy import coupled_env, simpleagent  # noqa: E402
from tuning.legacy import run_legacy as RL  # noqa: E402

DEFAULT = {"grid": list(range(10, 100, 10)), "fixed": list(range(1, 21))}


def make_env(env_name, variant):
    base = RL.ENVS[env_name]
    return coupled_env.make(base)() if variant == "coupled" else base()


def evaluate(env, policy, seeds, prepro=None):
    """Greedy eval as run_legacy.eval_reset (reseed, reset, 5000 ticks) with stats; prepro=N sets pods to N after
    reset (the policy then holds N)."""
    eps = []
    for s in seeds:
        RL.seed_all(s)
        s0 = env.reset()
        if prepro is not None:
            env.num_vms = int(prepro)
            s0 = np.array(s0, dtype=float)
            s0[5] = prepro
        eps.append(RL.run_episode(env, policy, False, s0, RL.EP_TICKS, False, True))
    return RL.summarize(eps)


def brief(r, episodes=False):
    if episodes:
        return {k: r[k] for k in ("mean_ret", "sd_ret", "mean_pods", "p95_latency_mean", "components_mean",
                                  "n_episodes", "episodes")}
    return {k: r[k] for k in ("mean_ret", "sd_ret", "mean_pods", "p95_latency_mean", "components_mean", "n_episodes")}


def hold(n):
    return lambda s: 2 if s[5] < n else (0 if s[5] > n else 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env", choices=tuple(RL.ENVS), default="dense")
    p.add_argument("--env-variant", choices=RL.VARIANTS, default="coupled")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED", 0)))
    p.add_argument("--report-splits", default="test", help="comma list of held-out splits (test, test2), not val/fair")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    reps = RL.parse_splits(a.report_splits)
    if any(sp in ("val", "fair") for sp in reps):
        p.error("--report-splits takes held-out splits only (selection is on val)")
    torch.set_num_threads(1)
    cfg = dict(DEFAULT)
    if a.config:
        with open(a.config) as f:
            cfg.update(json.load(f))
    RL.check_seeds(a.seed, "fair", ("val",) + reps)
    RL.seed_all(a.seed)
    val = RL.SPLITS["val"]
    n_val = (2 + len(cfg["grid"]) ** 2 + 2 * len(cfg["fixed"])) * len(val)
    tel = RL.Telemetry(os.environ.get("QRL_RUN_DIR"), n_val + 5 * sum(len(RL.SPLITS[sp]) for sp in reps))
    env = make_env(a.env, a.env_variant)
    t0, c0 = time.time(), time.process_time()
    done = [0]

    def ev(policy, seeds, prepro=None, name=""):
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            r = evaluate(env, policy, seeds, prepro)
        done[0] += len(seeds)
        tel.log(done[0], mean_ret=r["mean_ret"], mean_pods=r["mean_pods"], p95_latency=r["p95_latency_mean"])
        tel.progress(done[0])
        return r

    res = dict(env=a.env, env_variant=a.env_variant, seed=a.seed, config=cfg, val_seeds=list(val),
               selection="max mean episode return on val", val={}, report_splits=list(reps))
    for sp in reps:
        res[f"{sp}_seeds" if sp != "test" else "test_seeds"] = list(RL.SPLITS[sp])
        res[sp] = {}
    # ---- validation (selection)
    res["val"]["always_down"] = brief(ev(lambda s: 0, val))
    res["val"]["hpa_rule_50_50"] = brief(ev(simpleagent.SimpleAgent(50, 50).act, val))
    grid = {}
    for tc in cfg["grid"]:
        for tm in cfg["grid"]:
            grid[f"{tc}_{tm}"] = brief(ev(simpleagent.SimpleAgent(tc, tm).act, val))
    ramp = {n: brief(ev(hold(n), val)) for n in cfg["fixed"]}
    pre = {n: brief(ev(hold(n), val, prepro=n)) for n in cfg["fixed"]}
    best_hpa = max(grid, key=lambda k: grid[k]["mean_ret"])
    best_ramp = max(ramp, key=lambda k: ramp[k]["mean_ret"])
    best_pre = max(pre, key=lambda k: pre[k]["mean_ret"])
    res["val_grid_hpa"] = {k: {"mean_ret": v["mean_ret"], "mean_pods": v["mean_pods"],
                               "p95_latency_mean": v["p95_latency_mean"]} for k, v in grid.items()}
    res["val_fixed_ramped"] = {str(k): {"mean_ret": v["mean_ret"], "mean_pods": v["mean_pods"]} for k, v in ramp.items()}
    res["val_fixed_prepro"] = {str(k): {"mean_ret": v["mean_ret"], "mean_pods": v["mean_pods"]} for k, v in pre.items()}
    tc, tm = (int(x) for x in best_hpa.split("_"))
    sel = {"hpa_rule_tuned": {"T_c": tc, "T_m": tm}, "fixed_ramped": {"N": int(best_ramp)},
           "fixed_prepro": {"N": int(best_pre)}}
    res["selected"] = sel
    res["val"]["hpa_rule_tuned"] = grid[best_hpa]
    res["val"]["fixed_ramped"] = ramp[best_ramp]
    res["val"]["fixed_prepro"] = pre[best_pre]
    res["val_cpu_s"] = time.process_time() - c0
    # ---- held-out splits (selected policies only, frozen above)
    for sp in reps:
        test, r = RL.SPLITS[sp], res[sp]
        r["always_down"] = brief(ev(lambda s: 0, test), episodes=True)
        r["hpa_rule_50_50"] = brief(ev(simpleagent.SimpleAgent(50, 50).act, test), episodes=True)
        r["hpa_rule_tuned"] = brief(ev(simpleagent.SimpleAgent(tc, tm).act, test), episodes=True)
        r["fixed_ramped"] = brief(ev(hold(int(best_ramp)), test), episodes=True)
        r["fixed_prepro"] = brief(ev(hold(int(best_pre)), test, prepro=int(best_pre)), episodes=True)
    res["cpu_s"], res["wall_s"] = time.process_time() - c0, time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"selected": sel, "cpu_s": round(res["cpu_s"], 1),
                      **{f"{sp}_mean_ret": {k: round(v["mean_ret"]) for k, v in res[sp].items()} for sp in reps}}))
    return res


if __name__ == "__main__":
    main()
