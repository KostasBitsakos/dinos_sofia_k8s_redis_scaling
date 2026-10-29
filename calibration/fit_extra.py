"""Stage-B analyses added to fit.py (imported by fit.fit_all). All read-only, never invent: a missing piece gives None + a reason.
  * SLO-feasible capacity per c (C1/C3): largest served ops/s of a valid level whose p95 <= threshold AND whose failures are negligible.
  * M/M/c (shared queue, Erlang C) vs per-pod M/M/1 p95 against the measured p95 (C2); service variability cv^2 from the Pollaczek-Khinchine
    slope of the MEAN latency (C2).
  * CPU vs served per c (C7 table) and CFS throttling per level.
  * Unsaturated transition dip (C4, loads < 1): served/offered and p95 during the rebalance vs before.
  * Substitution check of the 50m namespace against 200m (capacity scaling only).
Generator-limited levels are never used: every function takes only levels with t["invalid"] == [] (fit.level_table rules)."""
import math
import statistics

import numpy as np

SLO_THRESHOLDS_MS = (10, 50, 100, 250)
FAIL_MAX = 0.001            # preregistered: drops, timeouts and errors each <= 0.1 % of offered at a level used for SLO-feasible capacity
THROTTLE_DOMINATED = 0.10   # preregistered: > 10 % of CFS periods throttled marks a level as throttling-affected


def _valid(table, c, sweep_only=True):
    return [t for (run, cc, ph, lv), t in table.items() if cc == c and (ph == 1 or not sweep_only) and not t["invalid"] and t["offered"] > 0]


def clean(t):
    o = t["offered"]
    return t["drop"] / o <= FAIL_MAX and t["timeout"] / o <= FAIL_MAX and t["err"] / o <= FAIL_MAX


def slo_feasible(table, cs, thresholds=SLO_THRESHOLDS_MS):
    out = {}
    for c in cs:
        lv = [t for t in _valid(table, c) if clean(t) and math.isfinite(t["p95"])]
        out[str(c)] = {}
        for thr in thresholds:
            ok = [t["served"] for t in lv if t["p95"] <= thr]
            out[str(c)][str(thr)] = {"served_ops_s": max(ok) if ok else None, "n_levels_ok": len(ok), "n_levels_clean": len(lv),
                                     "note": None if ok else "no valid clean level meets the threshold"}
    return out


def erlang_c(c, a):
    """P(wait) of M/M/c with offered load a = lambda/mu Erlangs (a < c)."""
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) * c / (c - a)
    return top / (s + top)


def sojourn_quantile(q, mu, theta, pw):
    """Quantile of T = S + W, S ~ Exp(mu), W = 0 w.p. 1-pw else Exp(theta). Bisection on the exact tail."""
    def tail(t):
        if abs(theta - mu) < 1e-9 * mu:
            hyp = (1 + mu * t) * math.exp(-mu * t)
        else:
            hyp = (theta * math.exp(-mu * t) - mu * math.exp(-theta * t)) / (theta - mu)
        return (1 - pw) * math.exp(-mu * t) + pw * hyp
    lo, hi = 0.0, 1.0
    while tail(hi) > 1 - q:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) > 1 - q else (lo, mid)
    return (lo + hi) / 2


def queue_models(table, cap):
    """For c >= 2: measured p95 vs per-pod M/M/1 (load split evenly, mu = C/c) and shared-queue M/M/c, on valid clean sub-saturation levels.
    Reports mean log-ratio measured/model for each (0 = perfect) and the better model. Network / other latency is not modelled here, so
    a constant offset shows up as a ratio > 1 at low load: compare the two models' SHAPE via `rmse_log`, not the absolute level."""
    out = {}
    for c in sorted(cap):
        if c < 2:
            continue
        mu = cap[c] / c
        pts = []
        for t in _valid(table, c):
            lam = t["offered"]
            rho = lam / cap[c]
            if not (0.05 <= rho <= 0.9) or not clean(t) or not math.isfinite(t["p95"]):
                continue
            m1 = math.log(20) / (mu - lam / c) * 1e3
            mc = sojourn_quantile(0.95, mu, c * mu - lam, erlang_c(c, lam / mu)) * 1e3
            pts.append((rho, t["p95"], m1, mc))
        if len(pts) < 3:
            out[str(c)] = {"n_points": len(pts), "note": "fewer than 3 usable levels"}
            continue
        r1 = np.array([math.log(p / m) for _, p, m, _ in pts])
        rc = np.array([math.log(p / m) for _, p, _, m in pts])
        e1, ec = float(np.sqrt(np.mean((r1 - r1.mean()) ** 2))), float(np.sqrt(np.mean((rc - rc.mean()) ** 2)))
        out[str(c)] = {"n_points": len(pts), "mean_log_ratio_mm1": float(r1.mean()), "mean_log_ratio_mmc": float(rc.mean()),
                       "rmse_log_mm1": e1, "rmse_log_mmc": ec, "better_shape": "M/M/c" if ec < e1 else "per-pod M/M/1",
                       "note": "service rate mu = C(c)/c from saturation capacity; no offset term"}
    return out


def mean_latency_ms(hist_rows):
    """Mean of ok-latencies from the 5 % log histogram (bucket midpoints). Timeout-censored mass is not in it."""
    from loadgen import BASE, LAT0
    n = tot = 0
    for r in hist_rows:
        for b, cnt in (r.get("hist") or {}).items():
            tot += cnt * LAT0 * BASE ** (int(b) + 0.5)
            n += cnt
    return 1e3 * tot / n if n else float("nan")


def cv2_estimate(table, rows, cap):
    """M/G/1 per pod: W(rho) = a + b * rho/(1-rho) with b = (1+cv2)/(2 mu)  =>  cv2 = 2 mu b - 1 (mu in 1/ms). OLS over valid clean levels
    with 0.05 <= rho <= 0.9 (>= 4 needed). The offset a absorbs network + client latency. Heavy-tail / throttling stalls inflate cv2: it is
    the effective service variability of the whole chain, not Redis's command-time variance."""
    out = {}
    by = {}
    for r in rows:
        if r.get("warm") == 0 and r.get("phase") == 1:
            by.setdefault((r["_run"], r["pods"], r["level"]), []).append(r)
    for c in sorted(cap):
        xs, ys = [], []
        for (run, cc, ph, lv), t in table.items():
            if cc != c or ph != 1 or t["invalid"] or not clean(t):
                continue
            rho = t["offered"] / cap[c]
            if 0.05 <= rho <= 0.9:
                w = mean_latency_ms(by.get((run, cc, lv), []))
                if math.isfinite(w):
                    xs.append(rho / (1 - rho))
                    ys.append(w)
        if len(xs) < 4:
            out[str(c)] = {"n_points": len(xs), "note": "fewer than 4 usable levels"}
            continue
        A = np.vstack([np.ones(len(xs)), xs]).T
        (a, b), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
        mu_ms = cap[c] / c / 1e3
        out[str(c)] = {"n_points": len(xs), "offset_ms": float(a), "slope_ms": float(b), "cv2": float(2 * mu_ms * b - 1),
                       "note": "negative or tiny slope = latency not queueing-dominated; do not use"}
    return out


def cpu_vs_served(table, cs):
    return {str(c): sorted(({"served": round(t["served"], 1), "offered": round(t["offered"], 1), "pods_busy": t.get("pods_busy"),
                             "throttled_frac": t.get("throttled_frac")} for t in _valid(table, c)), key=lambda x: x["offered"]) for c in cs}


def unsaturated_dip(rows, ep, tele, pre_s=25.0):
    """Loads < 1: served/offered and p95 during [first_slot, rebalance_end] relative to the pre window. Sub-saturation, so this is a latency /
    shortfall effect, not a capacity loss (fit.episode_loss censors those)."""
    from fit import quantile_censored, NB
    rs = [r for r in rows if r.get("phase") == 2 and r.get("warm") == 0]
    if ep.get("dir") != "up" or not all(k in ep for k in ("t_first_slot", "t_reb_end", "t_cmd")) or not rs:
        return None

    def agg(lo, hi):
        sel = [r for r in rs if lo <= r["t"] < hi]
        if len(sel) < 3:
            return None
        hist = [0] * NB
        for r in sel:
            for b, cnt in (r.get("hist") or {}).items():
                hist[int(b)] += cnt
        cens = int(sum(r["timeout_ops"] for r in sel))
        off = sum(r["offered_ops"] for r in sel)
        return {"n": len(sel), "ratio": sum(r.get("served_ops", r["completed_ops"]) for r in sel) / off if off else None,
                "p95_ms": 1e3 * quantile_censored(hist, cens, 0.95) if sum(hist) + cens else None}
    pre, dur = agg(ep["t_cmd"] - pre_s, ep["t_cmd"] - 1), agg(ep["t_first_slot"], ep["t_reb_end"])
    if not pre or not dur:
        return None
    return {"t_cmd": ep["t_cmd"], "pre": pre, "during": dur, "duration_s": ep["t_reb_end"] - ep["t_first_slot"],
            "d_ratio": (dur["ratio"] - pre["ratio"]) if dur["ratio"] is not None and pre["ratio"] is not None else None}


def substitution_check(cap200, cap50, tol=0.15):
    """PREREGISTERED: r_c = C200(c)/C50(c) for the c measured in both namespaces; accepted iff every r_c is within +-tol of the mean ratio.
    Validates capacity SCALING with the pod CPU limit only (not latency tails, start-up, migration, or the CPU mapping)."""
    cs = sorted(set(cap200) & set(cap50))
    if len(cs) < 2:
        return {"accepted": None, "reason": f"need >= 2 common pod counts with bracketed capacity in both namespaces, have {cs}"}
    r = {c: cap200[c] / cap50[c] for c in cs}
    m = statistics.mean(r.values())
    dev = {c: abs(v / m - 1) for c, v in r.items()}
    return {"common_c": cs, "ratio": {str(c): v for c, v in r.items()}, "mean_ratio": m, "max_dev": max(dev.values()), "tol": tol,
            "accepted": max(dev.values()) <= tol,
            "scope": "capacity scaling vs pod CPU limit only; latency tails, start-up, migration and CPU mapping are NOT validated"}


def analyse(table, cap, rows, eps, tele):
    cs = sorted({k[1] for k in table if k[2] == 1})
    res = {"slo_feasible_capacity": slo_feasible(table, cs), "queue_models": queue_models(table, cap),
           "service_cv2": cv2_estimate(table, rows, cap), "cpu_vs_served": cpu_vs_served(table, cs)}
    dips = [d for d in (unsaturated_dip(rows, ep, tele) for ep in eps) if d]
    res["unsaturated_transition_dip"] = dips
    return res
