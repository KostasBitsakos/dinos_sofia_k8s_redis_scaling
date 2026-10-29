#!/usr/bin/env python3
"""Calibration entry for one pod count c: scale to c, prefill, find saturation (probe), run the open-loop rate sweep.

RUNS ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC. Mutates the cluster (scale_to.py) and writes keys 'cal:*'.
Phases written to metrics.jsonl: phase 0 = saturation probe (geometric staircase), phase 1 = sweep at fixed
fractions of the probed capacity. Fixed mix: 95 % GET / 5 % SET, 16 B keys, 64 B values, 2000 uniform keys.
"""
import argparse
import json
import os
import subprocess
import sys

import capstate
import common
from common import event

HERE = os.path.dirname(os.path.abspath(__file__))
FRACTIONS = [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5, 2.0]  # >= 2 levels well above 1.0 to bracket saturation (2.0 = overload)


CHILDREN = common.Children()
HOST = None


def lg(schedule, a, phase, level_base, prefill=False, mem=None):
    cmd = [sys.executable, os.path.join(HERE, "loadgen.py"), "--schedule", schedule, "--seed", str(a.seed),
           "--workers", str(a.workers), "--pods", str(a.c), "--phase", str(phase), "--level-base", str(level_base),
           "--timeout", str(a.timeout), "--keys", str(a.keys), "--value-size", str(a.value_size),
           "--max-outstanding", str(a.max_outstanding)]
    if prefill:
        cmd.append("--prefill")
    p = CHILDREN.popen(cmd)
    if mem:
        mem.proc = p
    if HOST:
        HOST.gen_pid = p.pid
    rc = p.wait()
    if HOST:
        HOST.gen_pid = None
    if rc != 0 and not (mem and mem.tripped):   # a memory trip is handled (recovery) by the caller; any other failure fails the level
        event("loadgen_failed", rc=rc, schedule=schedule)
        raise SystemExit(f"load generator exited with {rc} (schedule {schedule}): level and entry failed")
    return rc


def read_rows(phase):
    path = os.path.join(common.run_dir(), "metrics.jsonl")
    return [r for r in map(json.loads, open(path)) if r.get("phase") == phase and r.get("warm") == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, required=True)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED") or 0))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--start-rate-per-pod", type=float, default=500)
    ap.add_argument("--probe-step-s", type=float, default=12)
    ap.add_argument("--level-s", type=float, default=40)
    ap.add_argument("--max-rate", type=float, default=300000)
    ap.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS), help="comma list of sweep fractions of the capacity")
    ap.add_argument("--keys", type=int, default=2000, help="dataset size in keys (prefilled)")
    ap.add_argument("--value-size", type=int, default=64)
    ap.add_argument("--max-outstanding", type=int, default=5000, help="client cap per worker; beyond it requests are dropped")
    ap.add_argument("--tag", default="default", help="label for the capacity state file (e.g. the pod CPU limit of this namespace)")
    ap.add_argument("--capacity-auto", action="store_true",
                    help="skip the probe: use the estimate recorded by an earlier entry for (--tag, --c)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    fractions = [float(x) for x in a.fractions.split(",") if x.strip()]
    if not fractions or any(f <= 0 for f in fractions):
        raise SystemExit("--fractions must be positive numbers")
    common.check_context()
    if a.dry_run:
        how = "capacity from state" if a.capacity_auto else (f"probe from {a.start_rate_per_pod * a.c:.0f} ops/s x1.6 per "
                                                              f"{a.probe_step_s}s up to {a.max_rate:.0f}")
        print(f"c={a.c}: scale_to {a.c}; prefill {a.keys} keys x {a.value_size} B; {how}; sweep {fractions} x capacity, "
              f"{a.level_s}s each, {a.workers} workers")
        return
    common.install_sigterm_as_exception()
    common.require_healthy()  # an earlier failed entry must have been rolled back before anything starts
    subprocess.run([sys.executable, os.path.join(HERE, "scale_to.py"), "--target", str(a.c)], check=True)
    common.require_safe_to_load()
    common.write_progress(0, 1 + len(fractions))
    global HOST
    mem = common.MemWatch()
    mem.expect_pods = True  # pods must exist while loading or scaling
    HOST = common.HostWatch()
    try:
        rate, level, best, stop_reason = a.start_rate_per_pod * a.c, 0, 0.0, None
        first = True
        if a.capacity_auto:   # no probe: prefill alone (rate 1 op/s for 1 s) then the sweep from the recorded estimate
            best, how = capstate.estimate(a.tag, a.c)
            event("capacity_auto", tag=a.tag, c=a.c, capacity_est=best, method=how)
            lg("1:1", a, 0, 0, prefill=True, mem=mem)
            rate = a.max_rate + 1
            stop_reason = "capacity-auto"
        while rate <= a.max_rate:
            lg(f"{rate:.0f}:{a.probe_step_s}", a, 0, level, prefill=first, mem=mem)
            first = False
            if mem.tripped:
                stop_reason = "memory"
                break
            rows = [r for r in read_rows(0) if r["level"] == level]
            if not rows:
                stop_reason = "no data"
                break
            comp = sum(r["completed_ops"] for r in rows) / len(rows)
            off = sum(r["offered_ops"] for r in rows) / len(rows)
            p99 = max(r["p99_ms"] for r in rows if r["p99_ms"] == r["p99_ms"]) if any(r["p99_ms"] == r["p99_ms"] for r in rows) else 0
            lag = max(r["generator_lag_ms"] for r in rows)
            best = max(best, comp)
            event("probe_level", level=level, offered=off, completed=comp, p99_ms=p99, lag_ms=lag)
            level += 1
            if lag > 50:
                stop_reason = "generator-limited"
                break
            if comp < 0.9 * off or p99 > 1000:
                stop_reason = "saturated"
                break
            rate *= 1.6
        event("probe_done", capacity_est=best, reason=stop_reason or "max-rate", c=a.c)
        if not a.capacity_auto and best > 0:   # any reason: the validity of the sweep is judged offline by fit.py
            capstate.record(a.tag, a.c, best, run=os.environ.get("QRL_RUN_ID"))
        if best <= 0 or mem.tripped:
            sys.exit("probe failed; not sweeping")
        budget = common.gen_budget(a.workers)
        run_f = [f for f in fractions if best * f <= budget]
        for f in fractions:
            if best * f > budget:   # a level the generator cannot drive validly is not run: it could only be mistaken for server capacity
                event("fraction_skipped_gen_budget", fraction=f, rate=best * f, budget=budget, workers=a.workers)
        if not run_f:
            sys.exit("no sweep fraction is within the generator budget; not sweeping")
        if best * max(run_f) < 1.2 * best:
            event("gen_budget_no_overload", capacity_est=best, budget=budget, note="no level >= 1.2 x capacity: the sweep cannot bracket saturation")
        sched = ",".join(f"{best * f:.0f}:{a.level_s}" for f in run_f)
        common.write_progress(1, 1 + len(fractions))
        lg(sched, a, 1, 0, mem=mem)
        if mem.tripped:
            sys.exit("memory abort during sweep")
        event("sweep_done", c=a.c, capacity_est=best, rss_peak=mem.peak)
    except KeyboardInterrupt:
        event("interrupted")
        raise SystemExit("interrupted: run rollback.sh")
    finally:
        CHILDREN.reap()
        mem.close()
        HOST.close()
        if mem.tripped:  # stop all scaling by this experiment, bring the cluster back to the 3-pod state
            common.recover_after_trip()
    h = common.health(a.c)
    event("health_after", **h)
    common.write_progress(1 + len(fractions), 1 + len(fractions))
    if not h["ok"]:
        sys.exit(f"cluster unhealthy after sweep: {h['reasons']}")


if __name__ == "__main__":
    main()
