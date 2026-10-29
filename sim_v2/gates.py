"""Frozen Sim v2 gate suite (council C-20261016-01, revised after Codex review of dcee4cc). Logic only;
run_gates.py is the entry point.

Policies: always_min, always_max, threshold (tuned), two static families:
  fixed(c)        episode starts with c ready pods, never scales (true fixed allocation);
  static_ramp(c)  starts at params.init_pods and ramps to c (pays the ramp).
Selections (threshold, fixed_best, static_ramp_best) use tune seeds only.

Gates (held-out seeds):
  G1 conservation   every episode: |arrived - served - failed - outstanding| <= tol * arrived at every step,
                    and nothing outstanding after settlement.
  G2 superiority    base config, per varying-load workload: threshold minus each fixed(c) and each static_ramp(c),
                    paired one-sided t-test over paired held-out seeds; Holm over all comparisons; pass iff every
                    Holm-adjusted p < alpha_one_sided (0.025). Reported lower bounds are UNADJUSTED.
  G3 oracle bound   for every episode, the DP bound for that episode's start state >= return - tol.
  G4 ranking        over theta x delta-factor configs, for the ranked set {always_min, always_max, fixed_best,
                    static_ramp_best, threshold}: no pair is significantly ordered one way in one config and the
                    other way in another (paired two-sided 95% CI of the difference excludes 0 with opposite signs).
                    Two versions, both gated: 'retuned' (selections redone per config) and 'frozen' (base-config
                    selections evaluated unchanged under each config).
  G5 F1 regression  always_min is not the best policy on any varying-load workload (base config).
Reported, not gated: oracle margins, exact mean orders.
"""
import itertools

import numpy as np

from env import SimParams, SimV2
from oracle import oracle_solve
from policies import (AlwaysMax, AlwaysMin, Fixed, OracleReplay, StaticRamp, Threshold, run_episode,
                      select_static, tune_threshold)
from stats import holm, paired_ci, paired_one_sided
from workloads import check_seed, make_workload

RANKED = ("always_min", "always_max", "fixed_best", "static_ramp_best", "threshold")


def heldout_seeds(cfg, seed):
    s0 = cfg["heldout_seeds_start"] + seed * cfg["n_seeds"]
    return [check_seed("heldout", s) for s in range(s0, s0 + cfg["n_seeds"])]


def units(cfg):
    """Work units (label, param overrides, workload, full); full=True is the base config (all statics)."""
    out = []
    sens = cfg["sensitivity"]
    base = SimParams(**cfg.get("param_overrides", {}))
    for th, df in itertools.product(sens["theta"], sens["delta_factor"]):
        ov = dict(cfg.get("param_overrides", {}), theta=th, rebalance_loss=base.rebalance_loss * df)
        full = th == base.theta and df == 1.0
        for w in cfg["workloads"]:
            out.append((f"theta={th:g},delta_x{df:g}", ov, w, full))
    out.sort(key=lambda u: not u[3])  # base config first
    return out


def run_unit(args):
    """args = (label, overrides, workload, full, cfg, seeds, frozen); frozen = base selections or None."""
    label, overrides, workload, full, cfg, seeds, frozen = args
    P = SimParams(**overrides)
    grid = {k: tuple(v) for k, v in cfg["threshold_grid"].items()}
    thr, _ = tune_threshold(P, workload, cfg["tune_seeds"], grid)
    fb, _ = select_static(P, workload, cfg["tune_seeds"], "fixed")
    rb, _ = select_static(P, workload, cfg["tune_seeds"], "ramp")
    pols = {"always_min": AlwaysMin(), "always_max": AlwaysMax(), "fixed_best": fb, "static_ramp_best": rb,
            "threshold": thr}
    if frozen:
        pols["threshold_frozen"] = Threshold(*frozen["threshold"])
        pols["fixed_best_frozen"] = Fixed(frozen["fixed_best"])
        pols["static_ramp_best_frozen"] = StaticRamp(frozen["static_ramp_best"])
    if full:
        for c in range(P.min_pods, P.max_pods + 1):
            pols[f"fixed_{c}"] = Fixed(c)
            pols[f"static_ramp_{c}"] = StaticRamp(c)
    res = {k: [] for k in pols}
    res["oracle_replay"] = []
    bound_for = {k: [] for k in res}
    bounds = []
    wl = make_workload(workload)
    for s in seeds:
        trace = wl.trace(s, P.episode_steps)
        b, acts, by_start = oracle_solve(P, trace)
        bounds.append(b)
        env = SimV2(P, trace=trace)
        for k, pol in list(pols.items()) + [("oracle_replay", OracleReplay(acts))]:
            res[k].append(run_episode(env, pol, s))
            start = getattr(pol, "init_pods", None)
            bound_for[k].append(b if start is None else by_start[start])
    return dict(label=label, workload=workload, full=full, overrides=overrides, bounds=bounds, bound_for=bound_for,
                selections={"threshold": [thr.up, thr.down, thr.cooldown], "fixed_best": fb.c,
                            "static_ramp_best": rb.c},
                names={"threshold": thr.name, "fixed_best": fb.name, "static_ramp_best": rb.name},
                returns={k: [e["ret"] for e in v] for k, v in res.items()},
                components={k: {c: float(np.mean([e[c] for e in v])) for c in ("cost", "sla", "churn", "bad",
                                                                                  "failed", "changes", "pod_steps")}
                            for k, v in res.items()},
                cons=max(e["max_resid"] / max(e["arrived"], 1.0) for v in res.values() for e in v),
                outstanding=max(e["outstanding_end"] for v in res.values() for e in v))


def ranking(results, frozen):
    """Pairwise paired 95% CIs per (workload, config) for the ranked set; stable iff no significant reversal."""
    suffix = "_frozen"
    pairs = {}
    for r in results:
        rets = {}
        for k in RANKED:
            key = k + suffix if (frozen and not r["full"] and k + suffix in r["returns"]) else k
            rets[k] = np.asarray(r["returns"][key])
        for a, b in itertools.combinations(RANKED, 2):
            mean, lo, hi = paired_ci(rets[a] - rets[b])
            sign = 1 if lo > 0 else (-1 if hi < 0 else 0)
            pairs.setdefault((r["workload"], a, b), []).append(dict(config=r["label"], mean=mean, lo=lo, hi=hi,
                                                                   sign=sign))
    stable = all(not ({1, -1} <= {x["sign"] for x in v}) for v in pairs.values())
    return stable, pairs


def evaluate(cfg, results):
    """Apply G1-G5. Returns (verdict, G2 rows, ranking tables, oracle violations)."""
    tol = cfg["tolerances"]
    g = {}
    g["G1_conservation"] = all(r["cons"] <= tol["conservation_rel"] and r["outstanding"] == 0.0 for r in results)
    viol = []
    for r in results:
        for k, rets in r["returns"].items():
            for b, ret in zip(r["bound_for"][k], rets):
                if ret > b + tol["oracle_abs"] + tol["oracle_rel"] * abs(b):
                    viol.append((r["label"], r["workload"], k, b, ret))
    g["G3_oracle_bound"] = not viol
    base = [r for r in results if r["full"]]
    rows = []
    for r in base:
        thr = np.asarray(r["returns"]["threshold"])
        for k, rets in r["returns"].items():
            if (k.startswith("fixed_") or k.startswith("static_ramp_")) and "best" not in k:
                st = paired_one_sided(thr - np.asarray(rets), cfg["alpha_one_sided"])
                rows.append(dict(workload=r["workload"], static=k, mean=st["mean"], lb_unadjusted=st["lb"],
                                 p=st["p"]))
    adj = holm([x["p"] for x in rows]) if rows else []
    for x, a in zip(rows, adj):
        x["p_holm"] = float(a)
    g["G2_threshold_beats_static"] = bool(rows) and all(x["p_holm"] < cfg["alpha_one_sided"] for x in rows)
    stable_r, pairs_r = ranking(results, frozen=False)
    stable_f, pairs_f = ranking(results, frozen=True)
    g["G4_ranking_stable_retuned"] = stable_r
    g["G4_ranking_stable_frozen"] = stable_f
    f1 = []
    for r in base:
        means = {k: float(np.mean(v)) for k, v in r["returns"].items() if k != "oracle_replay"}
        f1.append(max(means, key=means.get) != "always_min")
    g["G5_always_min_not_optimal"] = bool(f1) and all(f1)
    g["ALL"] = all(g.values())
    return g, rows, {"retuned": pairs_r, "frozen": pairs_f}, viol


def frozen_selections(base_results):
    return {r["workload"]: r["selections"] for r in base_results}
