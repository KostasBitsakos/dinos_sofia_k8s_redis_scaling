#!/usr/bin/env python3
"""Transition measurement under SATURATING open-loop load, tied to the scaling events (not to a fixed duration).

RUNS ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC. Mutates the cluster (scale_to.py) and writes keys 'cal:*'.
The load is --rate-frac x C_est(hi) with rate-frac > 1, so the cluster is saturated at BOTH sizes and the completion
rate equals the capacity at every instant (fit.py then reports capacity-relative loss, 1 - C_during / C_nominal).
Per repetition (phase 2), one load generator runs for the whole sequence:
  pre-window (--pre-s) -> scale up lo -> hi (blocks until healthy) -> settle -> scale down hi -> lo (blocks until healthy)
  -> settle -> SIGTERM to the generator (it stops arrivals, settles requests, flushes, exits 0).
The generator's schedule length is only a safety cap. scale_to.py writes the event timestamps to events.jsonl.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

import capstate
import common
from common import event

HERE = os.path.dirname(os.path.abspath(__file__))
CHILDREN = common.Children()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--capacity", type=float, help="C_est(hi) in ops/s from the stage-A sweep at c=hi")
    ap.add_argument("--capacity-auto", action="store_true",
                    help="take C_est(hi) from the probe estimates recorded by earlier sweep entries of this spec (--tag); "
                         "interpolated / extrapolated linearly in c when hi was not swept (method is logged)")
    ap.add_argument("--tag", default="default", help="capacity-state label (the pod CPU limit of the namespace)")
    ap.add_argument("--keys", type=int, default=2000, help="dataset size in keys")
    ap.add_argument("--value-size", type=int, default=64, help="value bytes; data volume = keys x (value + ~80 B overhead)")
    ap.add_argument("--rate-frac", type=float, default=1.15)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pre-s", type=float, default=30)
    ap.add_argument("--settle-s", type=float, default=45)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED") or 0))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    common.check_context()
    if a.capacity_auto == (a.capacity is not None):
        raise SystemExit("give exactly one of --capacity and --capacity-auto")
    method = "given"
    if a.capacity_auto:
        if a.dry_run:
            a.capacity, method = 1.0, "dry-run placeholder"
        else:
            a.capacity, method = capstate.estimate(a.tag, a.hi)
    rate = a.capacity * a.rate_frac
    budget = common.gen_budget(a.workers)
    if rate > budget:   # clamp: a generator-saturated load must not pose as a server-saturated one; fit.py censors an unsaturated window
        print(f"rate {rate:.0f} ops/s exceeds the generator budget {budget:.0f} with {a.workers} workers: clamped")
        rate = budget
        method += "; CLAMPED to generator budget"
    cap_s = a.pre_s + 2 * (900 + a.settle_s) + 60
    if a.dry_run:
        print(f"{a.lo}<->{a.hi} at {rate:.0f} ops/s ({a.rate_frac} x C_est, {method}), {a.reps} reps, {a.keys} keys x "
              f"{a.value_size} B; load runs until the scaling events end")
        return
    common.install_sigterm_as_exception()
    common.require_healthy()
    event("transition_plan", lo=a.lo, hi=a.hi, capacity_est=a.capacity, capacity_method=method, rate=rate,
          keys=a.keys, value_size=a.value_size)
    py = sys.executable
    mem = common.MemWatch()
    mem.expect_pods = True  # pods must exist while loading or scaling
    host = common.HostWatch()
    lgp = None
    try:
        subprocess.run([py, os.path.join(HERE, "scale_to.py"), "--target", str(a.lo)], check=True)
        common.require_safe_to_load()
        for rep in range(a.reps):
            cmd = [py, os.path.join(HERE, "loadgen.py"), "--schedule", f"{rate:.0f}:{cap_s:.0f}", "--seed", str(a.seed + rep),
                   "--workers", str(a.workers), "--pods", str(a.lo), "--phase", "2", "--level-base", str(rep),
                   "--timeout", "10", "--warmup", "0", "--keys", str(a.keys), "--value-size", str(a.value_size)] + ["--prefill"]
            lgp = CHILDREN.popen(cmd)
            mem.proc, host.gen_pid = lgp, lgp.pid
            time.sleep(4 + a.pre_s)  # loadgen start-up + pre-window (baseline capacity at lo)
            if lgp.poll() is not None or mem.tripped:
                raise RuntimeError(f"load generator ended during the pre-window (exit {lgp.poll()})")
            common.require_no_trip()
            event("rep_start", rep=rep, direction="up", rate=rate)
            subprocess.run([py, os.path.join(HERE, "scale_to.py"), "--target", str(a.hi)], check=True)
            time.sleep(a.settle_s)  # post-up window (capacity at hi)
            common.require_no_trip()
            event("rep_start", rep=rep, direction="down", rate=rate)
            subprocess.run([py, os.path.join(HERE, "scale_to.py"), "--target", str(a.lo)], check=True)
            time.sleep(a.settle_s)
            lgp.send_signal(signal.SIGTERM)  # lifecycle ends with the scaling events
            rc = lgp.wait(timeout=60)
            if rc != 0 and not mem.tripped:
                raise RuntimeError(f"load generator exited with {rc}")
            if mem.tripped:
                raise RuntimeError("memory abort")
            common.write_progress(rep + 1, a.reps)
        event("transitions_done", health=common.health(a.lo))
    except (RuntimeError, KeyboardInterrupt, subprocess.SubprocessError) as e:  # incl. "scaling is stopped" after a trip
        event("transition_failed", error=str(e))
        sys.exit(f"FAILED: {e}. Run rollback.sh before the next entry.")
    finally:
        CHILDREN.reap()
        mem.close()
        host.close()
        if mem.tripped:  # no further scaling by this experiment; reconcile back to the 3-pod state
            common.recover_after_trip()


if __name__ == "__main__":
    main()
