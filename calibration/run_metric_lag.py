#!/usr/bin/env python3
"""Observation lag of the cluster's resource metrics (C6): step load on c pods, read-only `kubectl top pod` polled every
--poll-s, against the cgroup CPU that HostWatch writes to telemetry.jsonl (ground truth, 2 s) and the generator's own windows.

RUNS ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC. Mutates only by SETs of 'cal:*' keys and (via scale_to) the
replica count. Per repetition (phase 3): idle --idle-s -> step to --frac x C_est for --high-s -> idle --low-s. Each `top` sample
is an event `top_sample` {cpu_m, mem_mi, pods}; event `lag_rep` marks the repetition. The lag itself (step time -> the time
`top` first reports >= 90 % of its plateau, same for the fall) is computed offline from these files, never here.
This does NOT create an HPA and does not read other namespaces: the HPA sync period / tolerance / stabilisation are documented
Kubernetes defaults (PREREG_stageB.md says what is and is not measured).
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time

import capstate
import common
from common import event

HERE = os.path.dirname(os.path.abspath(__file__))
CHILDREN = common.Children()
_TOP = re.compile(r"^\S+\s+(\d+)m\s+(\d+)Mi\s*$")


def parse_top(text):
    """kubectl top pod --no-headers output -> (sum cpu millicores, sum memory MiB, pods listed). Unparsable lines are skipped
    and counted out; an empty parse returns None (metrics-server has not served the pod yet)."""
    cpu = mem = n = 0
    for line in text.strip().splitlines():
        m = _TOP.match(line.strip())
        if m:
            cpu, mem, n = cpu + int(m.group(1)), mem + int(m.group(2)), n + 1
    return (cpu, mem, n) if n else None


def poller(stop, every):
    while not stop.wait(every):
        t = time.time()
        try:
            rc, out, _ = common.sh([*common.KUBECTL, "top", "pod", "--no-headers"], timeout=20)
        except Exception as e:  # noqa: BLE001 - the poll must not kill the run; the gap is data
            event("top_sample", ok=0, error=str(e)[:100])
            continue
        r = parse_top(out) if rc == 0 else None
        if r:
            event("top_sample", ok=1, t_poll=t, cpu_m=r[0], mem_mi=r[1], pods=r[2])
        else:
            event("top_sample", ok=0, t_poll=t, rc=rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, required=True)
    ap.add_argument("--capacity", type=float)
    ap.add_argument("--capacity-auto", action="store_true")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--idle-s", type=float, default=60)
    ap.add_argument("--high-s", type=float, default=120)
    ap.add_argument("--low-s", type=float, default=90)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--poll-s", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED") or 0))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    common.check_context()
    if a.capacity_auto == (a.capacity is not None):
        raise SystemExit("give exactly one of --capacity and --capacity-auto")
    if a.dry_run:
        print(f"c={a.c}: {a.reps} x (idle {a.idle_s}s, {a.frac} x C_est for {a.high_s}s, idle {a.low_s}s); kubectl top every {a.poll_s}s")
        return
    cap, how = (capstate.estimate(a.tag, a.c) if a.capacity_auto else (a.capacity, "given"))
    common.install_sigterm_as_exception()
    common.require_healthy()
    subprocess.run([sys.executable, os.path.join(HERE, "scale_to.py"), "--target", str(a.c)], check=True)
    common.require_safe_to_load()
    mem, host, stop = common.MemWatch(), common.HostWatch(), threading.Event()
    mem.expect_pods = True
    th = threading.Thread(target=poller, args=(stop, a.poll_s), daemon=True)
    try:
        th.start()
        rate = cap * a.frac
        for rep in range(a.reps):
            sched = f"1:{a.idle_s:.0f},{rate:.0f}:{a.high_s:.0f},1:{a.low_s:.0f}"
            event("lag_rep", rep=rep, rate=rate, capacity_est=cap, method=how, idle_s=a.idle_s, high_s=a.high_s, low_s=a.low_s)
            p = CHILDREN.popen([sys.executable, os.path.join(HERE, "loadgen.py"), "--schedule", sched, "--seed", str(a.seed + rep),
                                "--workers", str(a.workers), "--pods", str(a.c), "--phase", "3", "--level-base", str(10 * rep),
                                "--timeout", "10", "--warmup", "0"] + (["--prefill"] if rep == 0 else []))
            mem.proc, host.gen_pid = p, p.pid
            rc = p.wait()
            if rc != 0 and not mem.tripped:
                raise RuntimeError(f"load generator exited with {rc}")
            if mem.tripped:
                raise RuntimeError("memory abort")
            common.write_progress(rep + 1, a.reps)
        event("metric_lag_done", health=common.health(a.c))
    except (RuntimeError, KeyboardInterrupt, subprocess.SubprocessError) as e:
        event("metric_lag_failed", error=str(e))
        sys.exit(f"FAILED: {e}. Run rollback.sh before the next entry.")
    finally:
        stop.set()
        CHILDREN.reap()
        mem.close()
        host.close()
        if mem.tripped:
            common.recover_after_trip()


if __name__ == "__main__":
    main()
