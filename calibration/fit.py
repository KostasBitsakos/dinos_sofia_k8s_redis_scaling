#!/usr/bin/env python3
"""Fit Sim v2 parameters from calibration run outputs; write a provisional -> calibrated diff. Never applies it.

  fit.py RUN_DIR [RUN_DIR ...] [--out DIR]

Reads metrics.jsonl, events.jsonl and telemetry.jsonl of each run dir (loadgen.py / scale_to.py / HostWatch output;
phase 0 probe, 1 sweep, 2 transition). Writes DIR/fit_result.json, DIR/params_diff.json, DIR/params_diff.txt.

Rules (Codex review of 6b21de0):
  * Latency quantiles of a level come from the MERGED log-histograms of its measured windows (never averages of
    per-window quantiles). Timeouts are right-censored observations at the client timeout: a quantile that falls into the
    censored mass is reported as +inf and such a level takes no part in the latency fit. Errors/drops are failures.
  * A level is INVALID (excluded, with the reason listed) if: generator lag > 50 ms; error share > 1 %; a generator worker
    is CPU-bound (gen_cores/workers > 0.85); the shared host is busy (> 60 % of cpus: co-location interference); or there
    is no CPU telemetry (fail closed).
  * Capacity of one sweep needs a demonstrated saturation bracket: >= 1 valid level with completed/offered >= 0.97 AND
    >= 2 valid levels with completed/offered < 0.95 whose completions agree within 10 % (plateau). Capacity = their mean.
  * C(c) needs >= 2 bracketed sweeps from different run dirs that agree within 15 % (median is used).
  * mu, theta: OLS of log C on log c over accepted c (theta needs >= 3 pod counts).
  * d: median over scale-ups of (first_slot - scale_cmd), in 15 s steps.
  * delta, k and recovery (capacity-relative, saturating load, server-side served_ops, only windows with valid telemetry; an interval with
    < 80 % valid windows, unsaturated windows, or non-busy reference pods is CENSORED and excluded, never read as zero loss):
      up   : delta = 1 - mean(completed in [first_slot, rebalance_end]) / C_after(hi);  k = ceil(duration / 15 s)
      down : delta = 1 - mean(completed in [rebalance_start, pod_removed]) / C_after(lo)   (can be < 0: clipped at 0)
    plus loss_pre = 1 - mean(completed in [scale_cmd, first_slot)) / C_before(lo) to choose rebalance_trigger.
  * timeout: client policy, not calibrated; observed tail only.
"""
import argparse
import json
import math
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loadgen import BASE, LAT0, NB  # noqa: E402  (histogram geometry shared with the generator)

STEP_S = 15.0
LN20 = math.log(20.0)
LAG_LIMIT_MS, ERR_SHARE_MAX, GEN_CPU_PER_WORKER, HOST_BUSY_FRAC = 50.0, 0.01, 0.85, 0.60
SUB_DROP_MAX, SUB_TIMEOUT_MAX, POD_BUSY_MIN = 0.001, 0.01, 0.80
# cores per pod: 0.2 for the production limit (200m); 0.05 for the stage-B 50m namespace (env or --pod-cpu-limit). Never mix limits in one fit.
POD_CPU_LIMIT = float(os.environ.get("QRL_POD_CPU_LIMIT", 0.2))
SUB, SAT, PLATEAU, REPEAT_SPREAD = 0.97, 0.95, 0.10, 0.15


def load_run(d):
    def rd(name):
        p = os.path.join(d, name)
        return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []
    rows = rd("metrics.jsonl")
    for r in rows:
        r["_run"] = d
    return rows, rd("events.jsonl"), rd("telemetry.jsonl")


def quantile_censored(hist, n_cens, q):
    """Quantile of ok-latencies (bucket midpoints, seconds) with n_cens right-censored observations (>= timeout).
    Returns +inf when the quantile lies in the censored mass, nan with no data."""
    n_ok = sum(hist)
    n = n_ok + n_cens
    if n == 0:
        return float("nan")
    target, acc = q * n, 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= target:
            return LAT0 * BASE ** (i + 0.5)
    return float("inf")


def level_table(rows, tele):
    groups = {}
    for r in rows:
        if r.get("warm") != 0 or r.get("level", -1) < 0 or r.get("phase") not in (0, 1):
            continue
        groups.setdefault((r["_run"], r["pods"], r["phase"], r["level"]), []).append(r)
    out = {}
    for k, rs in groups.items():
        n = len(rs)
        hist = [0] * NB
        for r in rs:
            for b, cnt in (r.get("hist") or {}).items():
                hist[int(b)] += cnt
        mean = lambda f: sum(x[f] for x in rs) / n
        off = mean("offered_ops")
        served = sum(x.get("served_ops", x["completed_ops"]) for x in rs) / n
        t = {"n": n, "offered": off, "completed": mean("completed_ops"), "served": served, "err": mean("error_ops"),
             "timeout": mean("timeout_ops"), "drop": mean("drop_ops"), "lag_ms": max(x["generator_lag_ms"] for x in rs),
             "target": rs[0]["target_rate"], "workers": rs[0].get("workers", 1)}
        cens = int(round(sum(x["timeout_ops"] for x in rs)))
        for name, q in (("p50", .5), ("p95", .95), ("p99", .99)):
            t[name] = 1e3 * quantile_censored(hist, cens, q) if sum(hist) + cens else float("nan")
        t["has_hist"] = sum(hist) > 0
        t0, t1 = min(x["t"] for x in rs), max(x["t"] for x in rs) + 1
        tl = [x for x in tele if t0 <= x["t"] <= t1]
        reasons = []
        if t["lag_ms"] > LAG_LIMIT_MS:
            reasons.append("generator-lag")
        if off > 0 and t["err"] / off > ERR_SHARE_MAX:
            reasons.append("errors")
        if not tl:
            reasons.append("no-cpu-telemetry")
        else:
            host = statistics.mean(x["host_busy_cores"] / x["ncpu"] for x in tl)
            if any("gen_worker_max_cores" not in x for x in tl):
                reasons.append("no-cpu-telemetry")
                gen_worker = float("nan")
            else:
                gen_worker = max(x["gen_worker_max_cores"] for x in tl)  # busiest single worker, not an average
            busy = [x["pods_cores"] / (POD_CPU_LIMIT * x["n_pods"]) for x in tl if x.get("pods_cores") is not None and x.get("n_pods")]
            t.update({"gen_worker_max_cores": gen_worker, "host_busy_frac": host,
                      "pods_busy": statistics.mean(busy) if busy else None})
            if gen_worker > GEN_CPU_PER_WORKER:
                reasons.append("generator-cpu-bound")
            if host > HOST_BUSY_FRAC:
                reasons.append("host-interference")
        thr = [x for x in tl if x.get("d_nr_periods") is not None]
        nper = sum(x["d_nr_periods"] for x in thr)
        t["throttled_frac"] = sum(x["d_nr_throttled"] for x in thr) / nper if nper > 0 else None   # share of CFS periods with throttling
        t["throttled_s_per_s"] = sum(x["d_throttled_s"] for x in thr) / max(1e-9, 2.0 * len(thr)) if thr else None
        t["invalid"] = reasons
        out[k] = t
    return out


def sweep_capacity(levels):
    """levels: list of valid level dicts of ONE sweep -> (capacity or None, reason)."""
    ok = lambda x: x["offered"] > 0
    sub = [x for x in levels if ok(x) and x["served"] / x["offered"] >= SUB and x["drop"] / x["offered"] <= SUB_DROP_MAX
           and x["timeout"] / x["offered"] <= SUB_TIMEOUT_MAX]
    sat = sorted((x for x in levels if ok(x) and x["served"] / x["offered"] < SAT and x.get("pods_busy") is not None
                  and x["pods_busy"] >= POD_BUSY_MIN), key=lambda x: x["offered"])
    if not sub:
        return None, "no valid sub-saturation level with near-zero drops and timeouts"
    if len(sat) < 2:
        return None, "fewer than 2 valid saturated levels with busy pods (a plateau from client drops does not count)"
    top = sat[-2:]
    if abs(top[0]["served"] - top[1]["served"]) > PLATEAU * max(top[0]["served"], top[1]["served"]):
        return None, "no plateau in the two highest saturated levels"
    return statistics.mean(x["served"] for x in top), None


def capacity_per_c(table):
    sweeps = {}
    for (run, c, ph, lv), t in table.items():
        if ph == 1:
            sweeps.setdefault((run, c), []).append(t)
    per_c, rejected, invalid = {}, [], []
    for (run, c), lv in sorted(sweeps.items()):
        for t in lv:
            if t["invalid"]:
                invalid.append({"run": run, "c": c, "target": t["target"], "reasons": t["invalid"]})
        cap, why = sweep_capacity([t for t in lv if not t["invalid"]])
        if cap is None:
            rejected.append({"run": run, "c": c, "reason": why})
        else:
            per_c.setdefault(c, []).append(cap)
    cap, unstable = {}, []
    for c, v in per_c.items():
        if len(v) < 2:
            rejected.append({"c": c, "reason": f"only {len(v)} bracketed sweep(s); >= 2 required"})
            continue
        med = statistics.median(v)
        spread = (max(v) - min(v)) / med
        if spread > REPEAT_SPREAD:
            rejected.append({"c": c, "reason": f"sweeps disagree by {spread:.0%} (> {REPEAT_SPREAD:.0%})", "values": v})
            continue
        cap[c] = med
    return cap, per_c, rejected, invalid


def fit_power(cap):
    cs = sorted(c for c, v in cap.items() if v > 0)
    if len(cs) < 2:
        return None
    x = np.log(np.array(cs, float))
    y = np.log(np.array([cap[c] for c in cs], float))
    A = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    n = len(cs)
    se = float("nan")
    if n >= 3:
        se = math.sqrt(float(res @ res) / (n - 2) / float(((x - x.mean()) ** 2).sum()))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"mu": float(math.exp(beta[0])), "theta": float(beta[1]), "theta_se": se,
            "r2": 1 - float(res @ res) / ss_tot if ss_tot > 0 else float("nan"), "n_points": n, "cs": cs,
            "caveat": None if n >= 3 else "fewer than 3 pod counts: theta is not identifiable"}


def fit_latency(table, cap):
    out = {}
    for c in sorted(cap):
        pts = [(t["offered"] / cap[c], t["p95"]) for (run, cc, ph, lv), t in table.items()
               if cc == c and ph == 1 and not t["invalid"] and t["has_hist"] and t["offered"] / cap[c] < 0.97
               and math.isfinite(t["p95"]) and t["timeout"] == 0 and t["drop"] == 0]
        if len(pts) < 3:
            out[c] = {"n_points": len(pts), "note": "too few uncensored sub-saturation levels"}
            continue
        x = np.array([1.0 / (1.0 - r) for r, _ in pts])
        y = np.array([p for _, p in pts])
        A = np.vstack([np.ones_like(x), x]).T
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        b_sim = 1e3 * LN20 * c / cap[c]
        ss = float(((y - y.mean()) ** 2).sum())
        out[c] = {"n_points": len(pts), "a_ms": float(a), "b_ms": float(b), "b_sim_ms": b_sim, "b_ratio": float(b / b_sim),
                  "r2": 1 - float(((y - A @ np.array([a, b])) ** 2).sum()) / ss if ss > 0 else float("nan")}
    return out


# ------------------------------------------------------------------ transitions
UP_KEYS = {"pod_ready": "t_ready", "first_served": "t_first_served", "add_node": "t_add", "first_slot": "t_first_slot", "rebalance_start": "t_reb_start",
           "rebalance_end": "t_reb_end", "healthy": "t_healthy"}
DOWN_KEYS = {"pod_removed": "t_removed", "fix_start": "t_fix", "covered": "t_covered", "rebalance_start": "t_reb_start",
             "rebalance_end": "t_reb_end", "healthy": "t_healthy"}


def episodes(events):
    eps, cur = [], None
    for e in events:
        n = e["event"]
        if n == "scale_cmd":
            cur = {"dir": e.get("direction"), "t_cmd": e["t"]}
            eps.append(cur)
        elif n in ("scale_failed", "wrong_pod_removed"):
            cur = None
        elif cur is not None:
            key = (UP_KEYS if cur["dir"] == "up" else DOWN_KEYS).get(n)
            if key and key not in cur:
                cur[key] = e["t"]
            if n == "pod_removed":  # plain replica decrement: record which pod Kubernetes took and what it held
                cur["removed"] = cur.get("removed", []) + [e.get("pod")]
                cur["slots_lost"] = cur.get("slots_lost", 0) + (e.get("slots_lost") or 0)
                cur["keys_lost"] = cur.get("keys_lost", 0) + (e.get("keys_lost") or 0)
                cur["was_newest"] = e.get("was_newest")
    for i, ep in enumerate(eps):
        ep["t_next"] = eps[i + 1]["t_cmd"] if i + 1 < len(eps) else None
    return eps


def tele_at(tele, t, tol=2.5):
    sel = [x for x in tele if abs(x["t"] - t) <= tol]
    return sel


def window_ok(tele, r, need_busy):
    """A 1 s window is usable only with valid telemetry: host not saturated, no CPU-bound generator worker, and for
    reference windows pods demonstrably busy."""
    sel = tele_at(tele, r["t"] + 0.5)
    if not sel or any("gen_worker_max_cores" not in x for x in sel):
        return False
    if max(x["gen_worker_max_cores"] for x in sel) > GEN_CPU_PER_WORKER:
        return False
    if statistics.mean(x["host_busy_cores"] / x["ncpu"] for x in sel) > HOST_BUSY_FRAC:
        return False
    if need_busy:
        busy = [x["pods_cores"] / (POD_CPU_LIMIT * x["n_pods"]) for x in sel if x.get("pods_cores") is not None and x.get("n_pods")]
        if not busy or statistics.mean(busy) < POD_BUSY_MIN:
            return False
    return True


def _win(rows, tele, lo, hi, need_busy, min_valid=0.8):
    """(mean served, mean offered, n valid, reason) over windows in [lo, hi); None reason = fine. Windows without valid
    telemetry are dropped; fewer than 80 % valid means the interval is censored."""
    sel = [r for r in rows if lo <= r["t"] < hi]
    if not sel:
        return None, "no windows"
    good = [r for r in sel if window_ok(tele, r, need_busy)]
    if len(good) < min_valid * len(sel):
        return None, f"only {len(good)}/{len(sel)} windows have valid telemetry"
    g = lambda r: r.get("served_ops", r["completed_ops"])
    return (statistics.mean(g(r) for r in good), statistics.mean(r["offered_ops"] for r in good), len(good)), None


def episode_loss(rows, ep, tele, pre_s=25.0):
    """Capacity-relative loss for one scale episode, or (None, reason). Throughput = server-side served_ops."""
    rs = [r for r in rows if r.get("phase") == 2]
    if not rs:
        return None, "no phase-2 metrics"
    t_end = ep.get("t_healthy")
    if t_end is None:
        return None, "episode did not reach healthy"
    before, why = _win(rs, tele, ep["t_cmd"] - pre_s, ep["t_cmd"] - 1, True)
    if before is None:
        return None, "before window: " + why
    after, why = _win(rs, tele, t_end + 5, (ep["t_next"] or max(r["t"] for r in rs)) - 1, True)
    if after is None:
        return None, "after window: " + why
    if before[2] < 5 or after[2] < 10:
        return None, "pre/post windows too short"
    for name, w in (("before", before), ("after", after)):
        if w[1] < 1.05 * w[0]:
            return None, f"{name} window not saturated (offered {w[1]:.0f} vs served {w[0]:.0f}): loss would be censored"
    if ep["dir"] == "up":
        if not all(k in ep for k in ("t_first_slot", "t_reb_end", "t_cmd")):
            return None, "missing events"
        during, why = _win(rs, tele, ep["t_first_slot"], ep["t_reb_end"], False)
        if during is None:
            return None, "rebalance window: " + why
        if during[1] < 1.05 * during[0]:
            return None, "rebalance window not saturated: served may equal offered, loss censored"
        pre_serve, _ = _win(rs, tele, ep["t_cmd"], ep["t_first_slot"], False)
        return {"dir": "up", "C_lo": before[0], "C_hi": after[0],
                "delta_raw": 1 - during[0] / after[0], "delta": max(0.0, min(1.0, 1 - during[0] / after[0])),
                "loss_pre": (1 - pre_serve[0] / before[0]) if pre_serve else None,
                "k_steps": math.ceil((ep["t_reb_end"] - ep["t_first_slot"]) / STEP_S),
                "d_serve_s": ep["t_first_slot"] - ep["t_cmd"],
                "d_first_served_s": (ep["t_first_served"] - ep["t_cmd"]) if "t_first_served" in ep else None, "d_ready_s": ep.get("t_ready", float("nan")) - ep["t_cmd"],
                "d_full_s": ep["t_reb_end"] - ep["t_cmd"]}, None
    if not all(k in ep for k in ("t_removed", "t_covered", "t_cmd")):
        return None, "missing events"
    # Plain replica decrement (2025 behaviour): capacity drops when Kubernetes kills a pod, slots stay uncovered until
    # `cluster fix`. Window = scale_cmd .. first time served >= 90 % of the new nominal capacity for 3 windows in a row.
    c_lo = after[0]
    limit = min(ep["t_next"] or float("inf"), t_end + 30.0)   # search only inside THIS episode (+30 s after healthy)
    seq = sorted((r for r in rs if ep["t_cmd"] <= r["t"] < limit), key=lambda r: r["t"])
    g = lambda r: r.get("served_ops", r["completed_ops"])
    t_rest = next((seq[i]["t"] for i in range(len(seq) - 2) if all(g(x) >= 0.9 * c_lo for x in seq[i:i + 3])
                   and seq[i]["t"] > ep["t_removed"]), None)
    if t_rest is None:
        return None, "throughput not restored before the episode ended (censored)"
    during, why = _win(rs, tele, ep["t_cmd"], t_rest, False, min_valid=0.8)
    if during is None:
        return None, "recovery window: " + why
    if during[1] < 1.05 * during[0]:
        return None, "recovery window not saturated: loss censored"
    return {"dir": "down", "C_hi": before[0], "C_lo": c_lo, "delta_raw": 1 - during[0] / c_lo,
            "delta": max(0.0, min(1.0, 1 - during[0] / c_lo)),
            "k_steps": max(1, math.ceil((t_rest - ep["t_cmd"]) / STEP_S)),
            "covered_s": ep["t_covered"] - ep["t_cmd"], "restored_s": t_rest - ep["t_cmd"],
            "removed": ep.get("removed"), "was_newest": ep.get("was_newest"),
            "slots_lost": ep.get("slots_lost"), "keys_lost": ep.get("keys_lost")}, None


def fit_all(run_dirs):
    rows, events, tele = [], [], []
    for d in run_dirs:
        r, e, t = load_run(d)
        rows += r
        events += e
        tele += t
    res = {"runs": list(run_dirs)}
    table = level_table(rows, tele)
    cap, per_sweep, rejected, invalid = capacity_per_c(table)
    res["capacity_per_c"] = {str(c): v for c, v in sorted(cap.items())}
    res["capacity_sweeps"] = {str(c): v for c, v in sorted(per_sweep.items())}
    res["rejected_capacity"] = rejected
    res["invalid_levels"] = invalid
    res["power"] = fit_power(cap)
    res["latency"] = {str(c): v for c, v in fit_latency(table, cap).items()}
    tl = [r for r in rows if r.get("phase") in (0, 1) and r.get("warm") == 0]
    if tl:
        p99 = [r["p99_ms"] for r in tl if r["p99_ms"] == r["p99_ms"]]
        res["tail"] = {"max_window_p99_ms": max(p99, default=None),
                       "timeout_share": sum(r["timeout_ops"] for r in tl) / max(1e-9, sum(r["offered_ops"] for r in tl)),
                       "note": "timeout is a client policy, not calibrated; observed tail only"}
    eps = episodes(events)
    ups = [e for e in eps if e["dir"] == "up" and "t_first_slot" in e]
    d_s = [u["t_first_slot"] - u["t_cmd"] for u in ups]
    res["d"] = {"n": len(d_s), "seconds_median": statistics.median(d_s) if d_s else None, "seconds_all": d_s,
                "steps_median": statistics.median(d_s) / STEP_S if d_s else None,
                "pod_ready_s_median": statistics.median([u["t_ready"] - u["t_cmd"] for u in ups if "t_ready" in u])
                if any("t_ready" in u for u in ups) else None}
    fs = [u["t_first_served"] - u["t_cmd"] for u in ups if "t_first_served" in u]
    res["d"]["first_served"] = {"n": len(fs), "seconds_all": fs, "seconds_median": statistics.median(fs) if fs else None,
                                "note": "scale_cmd -> first client GET/SET answered by the new pod (INFO commandstats); needs load; "
                                        "episodes without it are `first_served_none` events, not zeros"}
    import fit_extra
    res.update(fit_extra.analyse(table, cap, rows, eps, tele))
    good, bad = [], []
    for ep in eps:
        r, why = episode_loss(rows, ep, tele)
        (good if r else bad).append(r or {"dir": ep["dir"], "t_cmd": ep["t_cmd"], "excluded": why})
    ups_g = [g for g in good if g["dir"] == "up"]
    downs_g = [g for g in good if g["dir"] == "down"]
    med = lambda v: statistics.median(v) if v else None
    pooled = [g["delta"] for g in good]
    res["delta_k"] = {"n_up": len(ups_g), "n_down": len(downs_g), "episodes": good, "excluded": bad,
                      "delta_up_median": med([g["delta"] for g in ups_g]), "delta_down_median": med([g["delta"] for g in downs_g]),
                      "down_recovery": {"covered_s_median": med([g["covered_s"] for g in downs_g]),
                                        "restored_s_median": med([g["restored_s"] for g in downs_g]),
                                        "slots_lost_median": med([g["slots_lost"] for g in downs_g if g.get("slots_lost") is not None]),
                                        "keys_lost_median": med([g["keys_lost"] for g in downs_g if g.get("keys_lost") is not None]),
                                        "share_newest_removed": (sum(1 for g in downs_g if g.get("was_newest")) / len(downs_g)) if downs_g else None},
                      "delta_median": med(pooled), "k_steps_median": med([g["k_steps"] for g in good]),
                      "loss_pre_median": med([g["loss_pre"] for g in ups_g if g.get("loss_pre") is not None]),
                      "caveat": "capacity-relative under saturating load; ramp-up of the new pod's share is lumped into "
                                "delta (sim has one lump); up and down windows differ in semantics, see PLAN.md"}
    return res


def make_diff(res):
    sys.path.insert(0, os.path.join(HERE, "..", "sim_v2"))
    from params import SimParams
    P = SimParams()
    rows = []

    def add(name, prov, cal, why):
        rows.append({"param": name, "provisional": prov, "calibrated": cal, "evidence": why})
    pw = res.get("power")
    if pw:
        add("mu_ops_per_pod", P.mu_ops_per_pod, round(pw["mu"], 1), f"OLS on {pw['n_points']} pod counts, R2={pw['r2']:.3f}")
        if pw["n_points"] >= 3:
            add("theta", P.theta, round(pw["theta"], 3), f"se={pw['theta_se']:.3f}")
    d = res.get("d", {})
    if d.get("steps_median") is not None:
        add("startup_delay_steps", P.startup_delay_steps, max(0, round(d["steps_median"])),
            f"median {d['seconds_median']:.1f} s over {d['n']} scale-ups (cmd -> first slot)")
    dk = res.get("delta_k", {})
    if dk.get("delta_median") is not None:
        add("rebalance_loss", P.rebalance_loss, round(dk["delta_median"], 3),
            f"median of {dk['n_up']} up + {dk['n_down']} down episodes, capacity-relative; down includes full recovery; " + dk["caveat"])
        add("rebalance_steps", P.rebalance_steps, max(1, int(dk["k_steps_median"])), "ceil(window / 15 s), median")
        if dk.get("loss_pre_median") is not None:
            trig = "topology" if dk["loss_pre_median"] < 0.05 else "action"
            add("rebalance_trigger", P.rebalance_trigger, trig,
                f"capacity loss before the new pod serves = {dk['loss_pre_median']:.3f} (< 0.05 -> topology)")
    add("timeout_s", P.timeout_s, P.timeout_s, "unchanged: client policy, not measurable as a cluster parameter")
    txt = ["# Sim v2 parameter diff (NOT applied). provisional -> calibrated"]
    for r in rows:
        mark = "  " if r["provisional"] == r["calibrated"] else "->"
        txt.append(f"{r['param']:22s} {r['provisional']!s:>10} {mark} {r['calibrated']!s:<10} # {r['evidence']}")
    if res.get("rejected_capacity"):
        txt.append(f"# {len(res['rejected_capacity'])} capacity sweep(s)/pod counts rejected: see fit_result.json")
    lat = res.get("latency", {})
    if lat:
        txt.append("# latency curve (sim assumes a=0, b=b_sim): " + "; ".join(
            f"c={c}: a={v['a_ms']:.2f} ms, b/b_sim={v['b_ratio']:.2f}" for c, v in lat.items() if "a_ms" in v))
        txt.append("# SLO note: sim slo_s = 4 ln20 / mu changes with mu; the researcher must confirm the SLO after calibration.")
    return rows, "\n".join(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--substitution", nargs=2, metavar=("FIT_200M_JSON", "FIT_50M_JSON"),
                    help="compare two fit_result.json files (one per namespace) with the preregistered tolerance and exit")
    ap.add_argument("--pod-cpu-limit", type=float, help="cores per pod of the namespace these runs used (default 0.2, or $QRL_POD_CPU_LIMIT)")
    a = ap.parse_args()
    if a.substitution:
        import fit_extra
        c = [{int(k): v for k, v in json.load(open(f))["capacity_per_c"].items()} for f in a.substitution]
        print(json.dumps(fit_extra.substitution_check(c[0], c[1]), indent=2))
        return
    if not a.runs:
        ap.error("RUN dirs are required")
    if a.pod_cpu_limit:
        globals()["POD_CPU_LIMIT"] = a.pod_cpu_limit
    out = a.out or a.runs[0]
    os.makedirs(out, exist_ok=True)
    res = fit_all(a.runs)
    rows, txt = make_diff(res)
    json.dump(res, open(os.path.join(out, "fit_result.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(out, "params_diff.json"), "w"), indent=2)
    open(os.path.join(out, "params_diff.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
