#!/usr/bin/env python3
"""Cycle 2 trivial baselines: fixed-pod policies (k pods held for the whole episode, action 1 = no-op, reset(init_pods=k))
evaluated on the dev eval seeds 1000-1004 under every reward variant {v2, d_sec} x load_def {offered, served}.
No training, no randomness besides the sim seed (deterministic). Writes results/c2/trivial.json:
{variant_loaddef: {k: {sum (5-episode sum of return), mean_*, components, failed, failed_over_arrived, sla, lat, rw_lat, churn, eval: [...]}}}
  python3 tuning/trivial_baselines.py [--out tuning/results/c2/trivial.json]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments", "dinos_simv2"))
sys.path.insert(0, os.path.dirname(HERE))
from adapter import SimParams  # noqa: E402
from tuning import diagnostics, features, reward_variants  # noqa: E402

KS = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30)
DEV_SEEDS = tuple(range(1000, 1005))
NOOP = 1
COMBOS = [(v, l) for v in ("v2", "d_sec") for l in ("offered", "served")]


def run_fixed(k, variant, load_def, seeds=DEV_SEEDS, workload="W0"):
    P = SimParams().with_(reward_mode="dinos")
    sim = reward_variants.wrap(features.make_sim("base", workload, P), variant, load_def)
    evals = []
    for s in seeds:
        st = diagnostics.EpisodeStats()
        sim.reset(int(s), init_pods=int(k))
        done = False
        while not done:
            _, r, done, info = sim.step(NOOP)
            st.add(NOOP, r, info)
        evals.append(dict(seed=int(s), **st.summary()))
    return _aggregate(int(k), evals, "k")


def _aggregate(k, evals, keyname):
    m = lambda key: float(np.mean([e[key] for e in evals]))
    return dict(**{keyname: int(k)}, sum=float(np.sum([e["ret"] for e in evals])), pods=m("mean_pods"), sla=m("sla_violation_rate"),
                c3L=m("sum_3L"), c5D=m("sum_neg5D"), c10N=m("sum_neg10N"), c05U=m("sum_neg05U"), c05M=m("sum_neg05M"),
                failed=m("failed_requests"), failed_over_arrived=m("failed_over_arrived"), lat_s=m("mean_latency_s"),
                rw_lat_s=m("req_weighted_latency_s"), churn=m("churn_pods"), eval=evals)


HPA_DEV_JSON = os.path.join(os.path.dirname(HERE), "experiments", "dinos_simv2", "hpa_selection_dev.json")
HPA_SENS = (20, 30, 50, 60, 70)


def run_hpa(target, variant, load_def, seeds=DEV_SEEDS, workload="W0"):
    """K8sHPA(target) driven as in reward_check/select_hpa_dev: env built with K8sHPA.env_kwargs
    (absolute replica actions, cpu_util_pct appended to the RAW observation), pol.act(raw obs), pol.reset() after env reset."""
    from adapter import DinosSimV2
    from policies import K8sHPA
    P = SimParams().with_(reward_mode="dinos")
    sim = reward_variants.wrap(DinosSimV2(workload, P, **K8sHPA.env_kwargs), variant, load_def)
    evals = []
    for s in seeds:
        pol = K8sHPA(int(target))
        st = diagnostics.EpisodeStats()
        sim.reset(int(s), init_pods=getattr(pol, "init_pods", None))
        pol.reset()
        done = False
        while not done:
            a = pol.act(sim.raw)
            _, r, done, info = sim.step(a)
            st.add(1, r, info)   # action-count slot unused for replica actions
        evals.append(dict(seed=int(s), **st.summary()))
    return _aggregate(int(target), evals, "target")


def run_hpa_all(targets, combos=COMBOS, seeds=DEV_SEEDS):
    return {f"{v}_{l}": {str(t): run_hpa(t, v, l, seeds) for t in targets} for v, l in combos}


def run_all(ks=KS, combos=COMBOS, seeds=DEV_SEEDS):
    return {f"{v}_{l}": {str(k): run_fixed(k, v, l, seeds) for k in ks} for v, l in combos}


def main_hpa(out):
    sel = json.load(open(HPA_DEV_JSON))["selected_target_pct"]
    targets = sorted({int(sel), *HPA_SENS})
    res = run_hpa_all(targets)
    best = {name: max(tab.items(), key=lambda kv: (kv[1]["sum"], -int(kv[0])))[0] for name, tab in res.items()}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(eval_seeds=list(DEV_SEEDS), dev_selected_target_pct=int(sel), targets=targets, best_target=best, results=res),
                  f, indent=1)
    for name, tab in res.items():
        print(name, "best target (of those run):", best[name])
        for t, r in tab.items():
            print(f"  hpa={t:>2s} sum={r['sum']:12.0f} pods={r['pods']:5.1f} failed={r['failed']:10.3e} f/a={r['failed_over_arrived']:.4f} "
                  f"lat={r['lat_s']:.3f} rwlat={r['rw_lat_s']:.3f} churn={r['churn']:.1f}")
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "results", "c2", "trivial.json"))
    p.add_argument("--hpa", action="store_true", help="evaluate K8sHPA (dev target + sensitivity) -> results/c2/hpa.json instead")
    a = p.parse_args(argv)
    if a.hpa:
        return main_hpa(a.out if a.out != os.path.join(HERE, "results", "c2", "trivial.json")
                        else os.path.join(HERE, "results", "c2", "hpa.json"))
    res = run_all()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(eval_seeds=list(DEV_SEEDS), ks=list(KS), results=res), f, indent=1)
    for name, tab in res.items():
        print(name)
        for k, r in tab.items():
            print(f"  k={k:>2s} sum={r['sum']:12.0f} pods={r['pods']:5.1f} failed={r['failed']:10.3e} f/a={r['failed_over_arrived']:.4f} "
                  f"sla={r['sla']:.3f} lat={r['lat_s']:.3f} rwlat={r['rw_lat_s']:.3f} 3L={r['c3L']:.0f} 5D={r['c5D']:.0f} 10N={r['c10N']:.0f}")
    return res


if __name__ == "__main__":
    main()
