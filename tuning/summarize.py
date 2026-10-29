#!/usr/bin/env python3
"""Summarise quick_test result JSONs: per config, mean/min/max of eval_mean_return over training seeds, pods, churn, SLA,
plus (cycle 1) the SUM over the eval episodes ("sum", = n_eval * mean, 5 for the dev block) and the mean per-episode reward
components (3L, -5D, -10N, -0.5U, -0.5M), failed requests and mean latency (s). Old result files without components
give nan in those columns.
  summarize.py results/w1 [--out summary.json]"""
import glob, json, os, re, sys
import numpy as np

d = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    if f.endswith("summary.json"):
        continue
    r = json.load(open(f))
    c = re.sub(r"-s\d+\.json$", "", os.path.basename(f))
    rows.setdefault(c, []).append(r)


def comp(rs, k):
    v = [e[k] for r in rs for e in r["eval"] if k in e]
    return float(np.mean(v)) if v else float("nan")


out = []
for c, rs in rows.items():
    ret = [r["eval_mean_return"] for r in rs]
    sums = [r.get("eval_sum_return", float(np.sum([e["ret"] for e in r["eval"]]))) for r in rs]
    sla = [np.mean([e["sla_violation_rate"] for e in r["eval"]]) for r in rs]
    out.append(dict(config=c, n=len(rs), mean=float(np.mean(ret)), min=float(min(ret)), max=float(max(ret)),
                    sum=float(np.mean(sums)), pods=float(np.mean([r["eval_mean_pods"] for r in rs])),
                    changes=float(np.mean([r["eval_mean_pod_changes"] for r in rs])), sla=float(np.mean(sla)),
                    c3L=comp(rs, "sum_3L"), c5D=comp(rs, "sum_neg5D"), c10N=comp(rs, "sum_neg10N"),
                    c05U=comp(rs, "sum_neg05U"), c05M=comp(rs, "sum_neg05M"),
                    failed=comp(rs, "failed_requests"), lat_s=comp(rs, "mean_latency_s"),
                    rw_lat_s=comp(rs, "req_weighted_latency_s"), f_over_a=comp(rs, "failed_over_arrived"),
                    churn=comp(rs, "churn_pods")))
out.sort(key=lambda x: -x["mean"])
print(f"{'config':26s} {'n':>2s} {'mean':>11s} {'min':>11s} {'max':>11s} {'sum':>11s} {'pods':>6s} {'chg':>7s} {'sla':>6s}"
      f" {'+3L':>9s} {'-5D':>9s} {'-10N':>9s} {'-.5U':>9s} {'-.5M':>9s} {'failed':>9s} {'lat_s':>6s} {'rwlat_s':>7s} {'f/a':>7s} {'churn':>7s}")
for o in out:
    print(f"{o['config']:26s} {o['n']:2d} {o['mean']:11.0f} {o['min']:11.0f} {o['max']:11.0f} {o['sum']:11.0f} {o['pods']:6.2f} "
          f"{o['changes']:7.1f} {o['sla']:6.3f} {o['c3L']:9.0f} {o['c5D']:9.0f} {o['c10N']:9.0f} {o['c05U']:9.0f} "
          f"{o['c05M']:9.0f} {o['failed']:9.2e} {o['lat_s']:6.3f} {o['rw_lat_s']:7.3f} {o['f_over_a']:7.4f} {o['churn']:7.1f}")
json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
