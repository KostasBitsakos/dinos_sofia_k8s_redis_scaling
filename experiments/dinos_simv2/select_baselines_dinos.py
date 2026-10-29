#!/usr/bin/env python3
"""Dev-only reselection of the classical baselines under a given reward mode (default 'dinos'; R-20261017-017).

Families (all selected on DEVELOPMENT seeds 1000-1004 only, default config, no training, no test seed):
  k8s_hpa       target utilisation %  over policies.HPA_TARGET_GRID (10..95 step 5)
  threshold     (up, down, cooldown)  over train_eval.THRESHOLD_GRID_V21 (down < up)
  fixed         static c in [min_pods, max_pods]   (preprovisioned at c)
  static_ramp   static c in [min_pods, max_pods]   (starts at init_pods, ramps to c)
Selection = highest mean return over the dev seeds; ties -> lexicographically smallest candidate. Edge/interior
status of every selected value is recorded. `--budget N` caps the number of candidates per family (deterministic
random subset, seed 0) so that families can be compared at an equal tuning budget; default = full grids (the most
generous setting for the baselines). Always-min / always-max are parameter-free and listed for reference. The cost-
optimal oracle is NOT a bound under the DInos reward and is not run.
Output: selection_baselines_<mode>.json (next to this file unless --out). Takes a few CPU-minutes (processes: --jobs).
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapter  # noqa: E402,F401  (puts sim_v2 on sys.path)
import freeze  # noqa: E402
from params import SimParams  # noqa: E402

DEV_SEEDS = tuple(range(1000, 1005))
os.environ.setdefault("OMP_NUM_THREADS", "1")


def candidates(P, budget=None):
    from policies import HPA_TARGET_GRID
    from train_eval import THRESHOLD_GRID_V21 as G
    import numpy as np
    fam = {
        "k8s_hpa": [("k8s_hpa", int(t)) for t in HPA_TARGET_GRID],
        "threshold": [("threshold", (up, dn, cd)) for up, dn, cd in itertools.product(G["up"], G["down"], G["cooldown"])
                      if dn < up],
        "fixed": [("fixed", c) for c in range(P.min_pods, P.max_pods + 1)],
        "static_ramp": [("static_ramp", c) for c in range(P.min_pods, P.max_pods + 1)],
    }
    if budget:
        rng = np.random.default_rng(0)
        for k, v in fam.items():
            if len(v) > budget:
                idx = sorted(rng.choice(len(v), size=budget, replace=False))
                fam[k] = [v[i] for i in idx]
    return fam


def _make(kind, arg):
    from policies import Fixed, K8sHPA, StaticRamp, Threshold
    return {"k8s_hpa": lambda a: K8sHPA(a), "threshold": lambda a: Threshold(*a), "fixed": lambda a: Fixed(a),
            "static_ramp": lambda a: StaticRamp(a)}[kind](arg)


def _eval(job):
    kind, arg, reward_mode, seeds = job
    import numpy as np
    from env import SimV2
    from policies import K8sHPA, run_episode
    P = SimParams().with_(reward_mode=reward_mode)
    kw = K8sHPA.env_kwargs if kind == "k8s_hpa" else {}
    env = SimV2(P, workload="W0", **kw)
    pol = _make(kind, arg)
    rets = [run_episode(env, pol, s)["ret"] for s in seeds]
    return kind, arg, float(np.mean(rets)), [float(r) for r in rets]


def reference_returns(P, seeds):
    import numpy as np
    from env import SimV2
    from policies import AlwaysMax, AlwaysMin, run_episode
    out = {}
    for name, pol in (("always_min", AlwaysMin()), ("always_max", AlwaysMax())):
        env = SimV2(P, workload="W0")
        out[name] = float(np.mean([run_episode(env, pol, s)["ret"] for s in seeds]))
    return out


def select(rows):
    """rows: [(arg, mean)] -> (best arg, ties). Highest mean, then smallest candidate."""
    best = max(m for _, m in rows)
    tied = sorted(a for a, m in rows if m == best)
    return tied[0], tied


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-mode", default="dinos", choices=("cost", "dinos"))
    ap.add_argument("--budget", type=int, default=None, help="max candidates per family (default: full grids)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from workloads import check_seed
    for s in DEV_SEEDS:
        check_seed("tune", s)
    P = SimParams().with_(reward_mode=a.reward_mode)
    fam = candidates(P, a.budget)
    jobs = [(k, arg, a.reward_mode, DEV_SEEDS) for k, v in fam.items() for _, arg in v]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs) as pool:
        res = pool.map(_eval, jobs, chunksize=1)
    out = dict(reward_mode=a.reward_mode, workload="W0", config="default", seeds=list(DEV_SEEDS),
               budget_per_family=a.budget, n_candidates={k: len(v) for k, v in fam.items()},
               episodes_per_candidate=len(DEV_SEEDS), selected={}, tables={}, interior={}, ties={})
    from train_eval import THRESHOLD_GRID_V21 as G
    from policies import HPA_TARGET_GRID
    for kind in fam:
        rows = [(arg, m) for k, arg, m, _ in res if k == kind]
        best, tied = select(rows)
        out["selected"][kind] = list(best) if isinstance(best, tuple) else best
        out["ties"][kind] = [list(t) if isinstance(t, tuple) else t for t in tied]
        out["tables"][kind] = [[list(arg) if isinstance(arg, tuple) else arg, m] for arg, m in rows]
        out.setdefault("mean_return_selected", {})[kind] = max(m for _, m in rows)
        if kind == "k8s_hpa":
            out["interior"][kind] = bool(min(HPA_TARGET_GRID) < best < max(HPA_TARGET_GRID))
        elif kind == "threshold":
            out["interior"][kind] = {"up": bool(min(G["up"]) < best[0] < max(G["up"])),
                                     "down": bool(min(G["down"]) < best[1] < max(G["down"])),
                                     "cooldown": bool(best[2] < max(G["cooldown"]))}   # 0 is the natural floor
        else:
            out["interior"][kind] = bool(P.min_pods < best < P.max_pods)
    out["reference_always"] = reference_returns(P, DEV_SEEDS)
    if a.reward_mode == "dinos":
        import env as _env
        out["reward_dinos_sha256"] = _env.load_reward_dinos()[1]
    out.update(git_head=freeze.git_head(HERE), wall_s=time.time() - t0,
               note="dev seeds 1000-1004 only; in-sample by construction; no test seed, no training")
    path = a.out or os.path.join(HERE, f"selection_baselines_{a.reward_mode}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({"out": path, "selected": out["selected"], "mean_return_selected": out["mean_return_selected"],
                      "interior": out["interior"], "reference_always": out["reference_always"],
                      "wall_s": out["wall_s"]}))
    return out


if __name__ == "__main__":
    main()
