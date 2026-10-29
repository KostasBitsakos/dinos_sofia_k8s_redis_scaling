#!/usr/bin/env python3
"""Exploratory W0 analysis (PREREG_explore_w0.md, explore_w0.json). Post-processing only, no simulation.

  analyze_explore_w0.py --out DIR RUN_DIR [RUN_DIR ...]

Reads `explore_eval.json` of every run directory (9 agent entries + 1 comparator entry), checks that they belong to the
registered protocol (same evaluation seeds, configs, frozen runs), and writes `explore_w0_report.json` and
`explore_w0_table.txt`:
  * agent return per evaluation episode = mean over its training seeds (per-seed contrasts reported too);
  * paired differences (method - static_ramp_best) and (method - fixed_best) for each agent, threshold and k8s_hpa,
    with a percentile bootstrap CI over evaluation episodes (explore_w0.json `bootstrap`);
  * default config is primary, every other config descriptive. EXPLORATORY: nothing is accepted or rejected.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_eval  # noqa: E402

LABEL = "EXPLORATORY (PLAN 8.1 amendment R-20261017-014): gates remain FAILED; uncalibrated provisional simulator"
ORACLE_LABEL = "oracle = clairvoyant +-1 reference (not an upper bound on multi-pod HPA performance)"
CI_LABEL = ("95% CIs resample evaluation episodes: conditional on the three trained models per agent, "
            "training-seed population uncertainty omitted")


def load_runs(dirs):
    runs = []
    for d in dirs:
        p = os.path.join(d, "explore_eval.json")
        with open(p) as f:
            r = json.load(f)
        r["_dir"] = os.path.abspath(d)
        runs.append(r)
    return runs


def _returns(rows, seeds):
    by = {r["seed"]: r["ret"] for r in rows}
    if sorted(by) != sorted(seeds):
        raise ValueError("evaluation episodes do not match the registered seeds")
    return np.array([by[s] for s in seeds], dtype=float)


def contrast(a, b, boot):
    d = np.asarray(a) - np.asarray(b)
    lo, hi = train_eval.bootstrap_ci(d, boot["resamples"], boot["seed"], boot["level"])
    return dict(mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi, wins=int((d > 0).sum()), n=int(d.size))


def analyze(runs, cfg=None, allow_incomplete=False, allow_unfrozen=False):
    cfg = cfg or train_eval.load_explore()
    seeds, cfgs, boot = list(cfg["eval_seeds"]), list(cfg["configs"]), cfg["bootstrap"]
    agents = cfg["training"]["agents"]
    tseeds = cfg["training"]["train_seeds"]
    for r in runs:
        if r["protocol"] != cfg["protocol"] or r["configs"] != cfgs or r["workload"] != cfg["workload"]:
            raise ValueError(f"{r['_dir']}: not a run of the registered protocol/configs")
        if r["eval_seeds"] != seeds:
            raise ValueError(f"{r['_dir']}: evaluation seeds differ from the registration")
        if not r["frozen"] and not allow_unfrozen:
            raise ValueError(f"{r['_dir']}: unfrozen run (smoke) cannot enter the analysis")
    base = [r for r in runs if r["agent"] == "explore_baselines"]
    if len(base) != 1:
        raise ValueError(f"need exactly one explore_baselines run, got {len(base)}")
    base = base[0]
    have = {(r["agent"], r["seed"]): r for r in runs if r["agent"] in agents}
    missing = [(a, s) for a in agents for s in tseeds if (a, s) not in have]
    if missing and not allow_incomplete:
        raise ValueError(f"missing registered entries: {missing}")
    if len(have) != sum(1 for r in runs if r["agent"] in agents):
        raise ValueError("duplicate agent entries")
    rep = dict(label=LABEL, oracle_label=ORACLE_LABEL, ci_label=CI_LABEL, complete=not missing, missing=missing, eval_seeds=seeds, configs=cfgs,
               primary_config=cfg["primary_config"], bootstrap=boot, runs=sorted(r["_dir"] for r in runs),
               hpa_target_pct=base.get("hpa_target_pct"), per_config={})
    refs = cfg["comparators"]["primary"]
    for cname in cfgs:
        rets = {n: _returns(base["results"][n][cname], seeds) for n in base["results"]}
        per_seed = {}
        for a in agents:
            arrs = [_returns(have[(a, s)]["results"][a][cname], seeds) for s in tseeds if (a, s) in have]
            if arrs:
                per_seed[a] = arrs
                rets[a] = np.mean(arrs, axis=0)
        block = dict(role="primary" if cname == cfg["primary_config"] else "descriptive",
                     mean_return={n: float(v.mean()) for n, v in rets.items()}, contrasts={}, per_seed_contrasts={})
        for m in list(per_seed) + cfg["comparators"]["disclosed"]:
            block["contrasts"][m] = {f"vs_{ref}": contrast(rets[m], rets[ref], boot) for ref in refs}
        for a, arrs in per_seed.items():
            block["per_seed_contrasts"][a] = {
                str(s): {f"vs_{ref}": contrast(x, rets[ref], boot) for ref in refs}
                for s, x in zip([s for s in tseeds if (a, s) in have], arrs)}
        rep["per_config"][cname] = block
    return rep


def table(rep):
    L = [LABEL, ORACLE_LABEL, CI_LABEL, f"complete={rep['complete']} n_episodes={len(rep['eval_seeds'])} HPA target={rep['hpa_target_pct']} %"]
    for cname, b in rep["per_config"].items():
        L.append(f"\n[{cname}] ({b['role']})")
        L.append(f"  {'policy':<34}{'mean return':>14}")
        for n, v in sorted(b["mean_return"].items(), key=lambda x: -x[1]):
            n = "oracle (clairvoyant +-1 ref.)" if n == "oracle" else n
            L.append(f"  {n:<34}{v:>14.1f}")
        for m, cs in b["contrasts"].items():
            L.append("  " + m + ": " + "; ".join(
                f"{k} {c['mean_diff']:+.1f} cond. CI [{c['ci_lo']:+.1f}, {c['ci_hi']:+.1f}] wins {c['wins']}/{c['n']}"
                for k, c in cs.items()))
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-incomplete", action="store_true", help="report is labelled INCOMPLETE")
    a = p.parse_args(argv)
    rep = analyze(load_runs(a.runs), allow_incomplete=a.allow_incomplete)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "explore_w0_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    txt = table(rep)
    with open(os.path.join(a.out, "explore_w0_table.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    return rep


if __name__ == "__main__":
    main()
