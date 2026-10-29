#!/usr/bin/env python3
"""Reward SLO candidate (researcher decision R-20261017-018): slo_s = 2 x p95 latency of ONE pod (c=1, production 200m limit) at HALF of its
measured saturation throughput. Offline, read-only; never invents: any missing piece gives slo_s = null + reason.

  slo_candidate.py --sweeps RUN [RUN...] --slo-runs RUN [RUN...] [--out slo_candidate.json] [--pod-cpu-limit 0.2]
Interpretation (ours, to be confirmed by the researcher): "50 % load" = OFFERED load equal to half the bracketed saturation throughput C1
(fit.capacity_per_c; >= 2 sweeps within 15 %, median). p95 is interpolated (log-linear) against the OFFERED rate between the two valid levels of
a repetition that bracket 0.5 x C1.
Preregistered acceptance (all must hold, else slo_s = null):
  * >= 3 DISTINCT valid repetitions (distinct run dirs, none of them a --sweeps dir); fewer -> null, never a single-run value;
  * both bracketing levels valid by fit.py (no generator lag / CPU-bound worker / host interference) with drops, timeouts and errors each
    <= 0.1 % of offered (fit_extra.FAIL_MAX);
  * the C1 estimate exists.
Uncertainty: t-interval (95 %) across repetitions of the mean p95, repeated at C1_min / C1_median / C1_max of the c=1 sweeps (capacity
uncertainty), widened by +-2.5 % (half of the 5 % histogram bucket) on both ends; `slo_s_ci95_total` is the union. 
`deployment_specific` = true when > 10 % of CFS periods were throttled at the levels used (or throttling data is missing): then p95 mostly
measures the 200m CPU quota, not Redis queueing, and the SLO must be labelled as a property of this pod limit."""
import argparse
import json
import math
import os
import statistics

import fit
import fit_extra

T95 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}   # two-sided 95 %, df = n - 1
MIN_RUNS, BUCKET_HALF = 3, 0.025


def interp_p95(levels, target):
    """levels: valid level dicts (offered, p95 in ms). Returns (p95 ms at offered == target, the two bracketing levels) or (None, None)."""
    pts = sorted((x for x in levels if x["p95"] == x["p95"] and math.isfinite(x["p95"]) and x["p95"] > 0), key=lambda x: x["offered"])
    for a, b in zip(pts, pts[1:]):
        if a["offered"] <= target <= b["offered"] and b["offered"] > a["offered"]:
            w = (target - a["offered"]) / (b["offered"] - a["offered"])
            return math.exp(math.log(a["p95"]) + w * (math.log(b["p95"]) - math.log(a["p95"]))), (a, b)
    return None, None


def _ci(vals):
    m = statistics.mean(vals)
    h = T95[len(vals) - 1] * statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) - 1 in T95 else float("nan")
    return m, m - h, m + h


def compute(sweep_dirs, slo_dirs, load=fit.load_run):
    res = {"definition": "slo_s = 2 x p95(c=1, offered load = 0.5 x bracketed saturation throughput)", "slo_s": None, "reason": None,
           "runs_used": [], "runs_rejected": [], "deployment_specific": None,
           "interpretation": "'50 % load' read by us as offered load at half of the measured saturation throughput (researcher to confirm)"}
    sweep_dirs = [os.path.realpath(d) if os.path.exists(d) else d for d in sweep_dirs]
    slo_dirs = list(dict.fromkeys(os.path.realpath(d) if os.path.exists(d) else d for d in slo_dirs))   # distinct
    if set(slo_dirs) & set(sweep_dirs):
        res["reason"] = "a run dir is both a sweep and an SLO repetition"
        return res
    rows, tele = [], []
    for d in sweep_dirs:
        r, _, t = load(d)
        rows += r
        tele += t
    cap, per_c, rejected, _ = fit.capacity_per_c(fit.level_table(rows, tele))
    if 1 not in cap:
        res["reason"] = f"no bracketed c=1 capacity from the sweeps (needs >= 2 sweeps within 15 %): {rejected}"
        return res
    caps = sorted(per_c[1])
    res.update(c1_capacity_ops_s=cap[1], target_offered_ops_s=0.5 * cap[1], c1_sweep_values=caps)
    per_run = []
    for d in slo_dirs:
        r, _, t = load(d)
        lv = [x for (run, c, ph, k), x in fit.level_table(r, t).items() if ph == 1 and c == 1 and not x["invalid"] and x["has_hist"]
              and fit_extra.clean(x)]
        p, pair = interp_p95(lv, 0.5 * cap[1])
        if p is None:
            res["runs_rejected"].append({"run": d, "reason": "no two valid, failure-free levels bracket 0.5 x C1 (offered)",
                                         "offered": sorted(round(x["offered"]) for x in lv)})
            continue
        per_run.append({"run": d, "levels": pair, "lv": lv})
        res["runs_used"].append({"run": d, "p95_ms": p, "offered_bracket": [pair[0]["offered"], pair[1]["offered"]],
                                 "throttled_frac": [x.get("throttled_frac") for x in pair]})
    res["n_runs"] = len(per_run)
    if len(per_run) < MIN_RUNS:
        res["reason"] = f"{len(per_run)} valid repetition(s); {MIN_RUNS} distinct valid runs are required (no single-run SLO)"
        return res
    # capacity uncertainty: redo the interpolation at the extreme capacities of the c=1 sweeps
    cis = {}
    for name, cc in (("c1_min", caps[0]), ("c1_median", cap[1]), ("c1_max", caps[-1])):
        vals = []
        for pr in per_run:
            p, _ = interp_p95(pr["lv"], 0.5 * cc)
            if p is not None:
                vals.append(p)
        if len(vals) >= MIN_RUNS:
            cis[name] = _ci(vals)
    if "c1_median" not in cis:
        res["reason"] = "fewer than 3 repetitions bracket the target at the median capacity"
        return res
    m = cis["c1_median"][0]
    lo = min(v[1] for v in cis.values()) * (1 - BUCKET_HALF)
    hi = max(v[2] for v in cis.values()) * (1 + BUCKET_HALF)
    thr = [f for u in res["runs_used"] for f in u["throttled_frac"]]
    res["deployment_specific"] = True if (not thr or any(f is None or f > fit_extra.THROTTLE_DOMINATED for f in thr)) else False
    res.update(p95_ms_mean=m, p95_ms_ci95_total=[lo, hi], ci_components={k: list(v) for k, v in cis.items()},
               ci_includes="across-repetition t-interval, capacity spread of the c=1 sweeps, +-2.5 % histogram bucket",
               slo_s=2 * m / 1e3, slo_s_ci95_total=[2 * lo / 1e3, 2 * hi / 1e3])
    res["deployment_specific_note"] = ("p95 at the used levels was throttling-dominated or throttling data is missing: report the SLO as a property of "
                                       "the 200m pod limit, not of Redis") if res["deployment_specific"] else None
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", nargs="+", required=True)
    ap.add_argument("--slo-runs", nargs="+", required=True)
    ap.add_argument("--out", default="slo_candidate.json")
    ap.add_argument("--pod-cpu-limit", type=float, default=0.2)
    a = ap.parse_args()
    fit.POD_CPU_LIMIT = a.pod_cpu_limit
    res = compute(a.sweeps, a.slo_runs)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
