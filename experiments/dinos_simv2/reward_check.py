#!/usr/bin/env python3
"""Falsification replay of the DInos reward adapter (R-20261017-017). Dev seeds 1000-1004 only, W0, full episodes.

Replays always_min, always_max, fixed(22), static_ramp(22), threshold, k8s_hpa (dev-selected target) and a fixed(c)
sweep with reward_mode='dinos'; prints per-policy mean return, mean component sums (L, D, N, U, M terms), drops
(failed requests) and mean pods. Verdict on "gains reward by dropping": for fixed(c) with the same pod count the
only thing that differs between two runs is the trace, so we test the same-pods marginal instead:
  (a) per-step regression inside each fixed(c): corr(step reward, step failed) must not be positive after removing
      the offered-load term (r + 3L is the load-independent part); and
  (b) across the fixed sweep, any pair c1 < c2 where fixed(c1) has MORE drops and HIGHER return is listed; this is
      expected only through the N term (10 per pod) and is reported with the reward decomposition.
Usage: python3 reward_check.py [--json out.json]. Deterministic, ~1 CPU-minute.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapter  # noqa: E402,F401  (puts sim_v2 on sys.path)
import numpy as np  # noqa: E402
from env import SimV2  # noqa: E402
from params import SimParams  # noqa: E402
from policies import AlwaysMax, AlwaysMin, Fixed, K8sHPA, StaticRamp, Threshold  # noqa: E402

SEEDS = tuple(range(1000, 1005))
assert all(1000 <= s <= 1004 for s in SEEDS)
SWEEP = (1, 5, 10, 15, 20, 22, 25, 30)


def episode(P, pol, seed):
    kw = getattr(pol, "env_kwargs", {})
    env = SimV2(P, workload="W0", **kw)
    obs, _ = env.reset(seed, init_pods=getattr(pol, "init_pods", None))
    pol.reset()
    s = dict(ret=0.0, r_L=0.0, r_D=0.0, r_N=0.0, r_U=0.0, r_M=0.0, failed=0.0, bad=0.0, pods=0.0, steps=0)
    rs, fs = [], []
    while True:
        obs, r, term, _, info = env.step(pol.act(obs))
        c = info["dinos"]
        s["ret"] += r
        for k in ("r_L", "r_D", "r_N", "r_U", "r_M"):
            s[k] += c[k]
        s["failed"] += info["failed"]
        s["bad"] += info["bad_requests"]
        s["pods"] += c["N"]
        s["steps"] += 1
        rs.append(r - c["r_L"])      # load-independent part of the step reward
        fs.append(info["failed"])
        if term:
            break
    s["pods"] /= s["steps"]
    return s, np.array(rs), np.array(fs)


def main():
    P = SimParams().with_(reward_mode="dinos")
    hpa_t = json.load(open(os.path.join(HERE, "hpa_selection_dev.json")))["selected_target_pct"]
    pols = [AlwaysMin(), AlwaysMax(), Fixed(22), StaticRamp(22), Threshold(), K8sHPA(hpa_t)]
    pols += [Fixed(c) for c in SWEEP if c != 22]
    rows, corr = {}, {}
    for pol in pols:
        eps = [episode(P, pol, sd) for sd in SEEDS]
        rows[pol.name] = {k: float(np.mean([e[0][k] for e in eps])) for k in eps[0][0] if k != "steps"}
        rr = np.concatenate([e[1] for e in eps])
        ff = np.concatenate([e[2] for e in eps])
        corr[pol.name] = float(np.corrcoef(rr, ff)[0, 1]) if ff.std() > 0 and rr.std() > 0 else None
    print(f"{'policy':20s} {'return':>12s} {'3L':>10s} {'-5D':>12s} {'-10N':>10s} {'-.5U':>9s} {'-.5M':>9s} "
          f"{'failed':>11s} {'pods':>6s}")
    for n, r in rows.items():
        print(f"{n:20s} {r['ret']:12.0f} {r['r_L']:10.0f} {r['r_D']:12.0f} {r['r_N']:10.0f} {r['r_U']:9.0f} "
              f"{r['r_M']:9.0f} {r['failed']:11.0f} {r['pods']:6.2f}")
    sw = sorted((int(n.split("_")[1]), r) for n, r in rows.items() if n.startswith("fixed_"))
    pairs = [(c1, c2) for i, (c1, a) in enumerate(sw) for c2, b in sw[i + 1:]
             if a["failed"] > b["failed"] and a["ret"] > b["ret"]]
    print("\nfixed sweep pairs c1<c2 with MORE drops and HIGHER return at c1:", pairs or "none")
    print("corr(step reward minus load term, step failed) per policy:", {k: (None if v is None else round(v, 3)) for k, v in corr.items()})
    exploit = bool(pairs) or any(v is not None and v > 0 for v in corr.values())
    print("VERDICT:", "DROPPING IMPROVES REWARD in some comparison -> adapter REJECTED (see pairs/corr)"
          if exploit else "no evidence that dropping requests raises reward (adapter not rejected)")
    if "--json" in sys.argv:
        json.dump(dict(rows=rows, corr=corr, pairs=pairs, exploit=exploit, seeds=list(SEEDS)),
                  open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
