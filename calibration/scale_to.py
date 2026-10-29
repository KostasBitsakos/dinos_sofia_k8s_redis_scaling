#!/usr/bin/env python3
"""Scale the calibration Redis Deployment (namespace qrl-calib) the way the 2025 agent did, and measure what happens.

RUNS ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC.
This script MUTATES the cluster (kubectl scale; redis-cli --cluster create/add-node/fix/rebalance, cluster forget).
Context kubernetes-admin@kubernetes, namespace qrl-calib (isolated copy of the researcher's Deployment; stateless, data
loss accepted: R-20261016-018).

Scale up, per pod:   scale +1 -> wait Ready + PING -> add-node -> rebalance --cluster-use-empty-masters -> ensure covered
Scale down, per pod: scale -1 by changing replicas ONLY. Kubernetes removes whichever pod it likes; that is the behaviour
                     under study, so we record WHICH pod went and how many slots/keys it held (event pod_removed), forget the
                     dead node on the survivors, run `redis-cli --cluster fix` until all 16384 slots are covered
                     (event covered), then rebalance. Nothing aborts on a "wrong victim".
Events (timestamps) go to $QRL_RUN_DIR/events.jsonl; fit.py derives d, delta, k and the recovery time from them.

  scale_to.py --target N [--max 30] [--timeout 900] [--dry-run]
  scale_to.py --bootstrap N     form the cluster (CLUSTER CREATE, no replicas) on pods that have no slots yet
"""
import argparse
import os
import sys
import tempfile
import time

import common
from common import event, kubectl, rcli, redis_pods, sh

MAX_PODS_HARD = 30


class Deadline:
    def __init__(self, seconds):
        self.t_end = time.time() + seconds

    def left(self):
        return self.t_end - time.time()

    def check(self, what):
        if self.left() <= 0:
            raise TimeoutError(f"timeout while {what}")


def wait_for(pred, dl, what, every=1.0):
    while True:
        v = pred()
        if v:
            return v
        dl.check(what)
        time.sleep(every)


def ping_ok(ip):
    rc, out, _ = rcli(ip, "ping", timeout=5)
    return rc == 0 and out.strip() == "PONG"


def wait_ready(n, dl):
    def pred():
        pods = redis_pods()
        return pods if len(pods) == n and all(p["ready"] and p["ip"] and ping_ok(p["ip"]) for p in pods) else None
    return wait_for(pred, dl, f"waiting for {n} Ready pods")


def wait_healthy(n, dl, what="health"):
    wait_for(lambda: common.health(n)["ok"], dl, f"waiting for {what} with {n} pods", every=2.0)


def my_slots(ip, node_id):
    for nd in common.cluster_nodes(ip):
        if nd["id"] == node_id:
            return sum(hi - lo + 1 for lo, hi in nd["slots"])
    return 0


def node_id_of(ip, target_ip):
    for nd in common.cluster_nodes(ip):
        if nd["addr"].startswith(target_ip + ":"):
            return nd["id"]
    return None


def dbsize(ip):
    rc, out, _ = rcli(ip, "dbsize", timeout=10)
    return int(out.strip()) if rc == 0 and out.strip().isdigit() else None


CHILDREN = common.Children()


def sh_popen(cmd):
    """Child process tracked so main()'s finally can terminate and reap it (deadline, exception, SIGTERM)."""
    import subprocess
    common.check_endpoints(cmd)
    common.require_no_trip()
    return CHILDREN.popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tracked(cmd, dl, timeout=120):
    """Run a (mutating) command as a tracked child: killed on a memory trip, on the deadline, on SIGTERM. Checked for trip
    BEFORE it starts. Returns (rc, stdout, stderr)."""
    common.check_endpoints(cmd)
    common.require_no_trip()
    with tempfile.TemporaryFile("w+") as o, tempfile.TemporaryFile("w+") as e:
        p = CHILDREN.popen(cmd, stdout=o, stderr=e, text=True)
        t_end = time.time() + timeout
        try:
            while p.poll() is None:
                if time.time() > t_end:
                    raise TimeoutError(f"{' '.join(cmd[:3])} timed out after {timeout}s")
                dl.check(f"running {' '.join(cmd[:3])}")
                if os.path.exists(common.trip_marker()):
                    raise RuntimeError("memory trip: command killed, scaling stopped")
                time.sleep(0.2)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
        o.seek(0)
        e.seek(0)
        return p.returncode, o.read(), e.read()


CONVERGENCE_ERRORS = ("don't agree", "dont agree", "not agree", "Nodes don't agree", "CLUSTERDOWN", "Cluster is not consistent",
                      "Please fix your cluster problems", "Hash slot not served", "is not empty", "Not all 16384 slots")
REBALANCE_ATTEMPTS = 3
REBALANCE_RETRY_WAIT = 3.0
CONVERGE_MAX_WAIT = 60.0


def log_cli(name, cmd, rc, out="", err=""):
    """Append a redis-cli invocation with rc/stdout/stderr to $QRL_RUN_DIR/redis-cli-<name>.log (diagnosable failures)."""
    try:
        with open(os.path.join(common.run_dir(), f"redis-cli-{name}.log"), "a") as f:
            f.write(f"### {time.strftime('%H:%M:%S')} {' '.join(cmd)} -> rc={rc}\n{out}\n--- stderr ---\n{err}\n")
    except OSError as ex:
        print(f"[warn] could not write redis-cli log: {ex}", flush=True)


def tracked_logged(name, cmd, dl, timeout=120):
    rc, out, err = tracked(cmd, dl, timeout)
    log_cli(name, cmd, rc, out, err)
    return rc, out, err


def is_convergence_error(text):
    return any(m.lower() in text.lower() for m in CONVERGENCE_ERRORS)


def wait_converged(dl, n_nodes, max_wait=CONVERGE_MAX_WAIT, every=1.0):
    """After add-node, wait (bounded) until every Ready pod lists n_nodes known nodes with the same id set and
    `redis-cli --cluster check` says all nodes agree. Emits converge_wait {seconds, ok, n_nodes}. Never raises on timeout
    (the rebalance retry handles the rest) but the event records ok=false."""
    t0 = time.time()
    ok = False
    while True:
        dl.check("waiting for cluster convergence")
        common.require_no_trip()
        pods = [p for p in redis_pods() if p["ip"] and p["ready"]]
        views = [frozenset(nd["id"] for nd in common.cluster_nodes(p["ip"])) for p in pods]
        if pods and all(len(v) == n_nodes for v in views) and len(set(views)) == 1:
            anchor = anchor_ip()
            if anchor:
                rc, out, err = tracked_logged("check", ["redis-cli", "--cluster", "check", f"{anchor}:{common.PORT}"], dl, 30)
                if rc == 0 and "All nodes agree about slots configuration" in out:
                    ok = True
                    break
        if time.time() - t0 >= max_wait:
            break
        time.sleep(every)
    event("converge_wait", seconds=round(time.time() - t0, 3), ok=ok, n_nodes=n_nodes)
    return ok


def rebalance_cmd(anchor, extra=()):
    return ["redis-cli", "--cluster", "rebalance", f"{anchor}:{common.PORT}", "--cluster-use-empty-masters",
            "--cluster-threshold", "1", "--cluster-yes", *extra]


def rebalance_logged(dl, what="rebalance", extra=(), watch=None, n_nodes=None):
    """Run redis-cli rebalance as a tracked child with stdout/stderr in redis-cli-rebalance.log. Up to REBALANCE_ATTEMPTS
    tries; a retry happens only for convergence-type errors (waits for convergence first); anything else fails at once."""
    import subprocess
    for attempt in range(1, REBALANCE_ATTEMPTS + 1):
        common.require_no_trip()
        anchor = anchor_ip()
        verify_topology()
        cmd = rebalance_cmd(anchor, extra)
        common.check_endpoints(cmd)
        with tempfile.TemporaryFile("w+") as o:
            p = CHILDREN.popen(cmd, stdout=o, stderr=subprocess.STDOUT, text=True)
            while p.poll() is None:
                dl.check(what)
                common.require_no_trip()
                if watch:
                    watch()
                time.sleep(0.5)
            o.seek(0)
            out = o.read()
        log_cli("rebalance", cmd, p.returncode, out)
        if p.returncode == 0:
            return
        if attempt < REBALANCE_ATTEMPTS and is_convergence_error(out):
            event("rebalance_retry", attempt=attempt, error=out.strip()[-200:])
            time.sleep(REBALANCE_RETRY_WAIT)
            if n_nodes:
                wait_converged(dl, n_nodes)
            continue
        raise RuntimeError(f"{what} failed (rc={p.returncode}): {out.strip()[-200:]}")


def scale_workload(n, dl=None):
    """Replica change as a tracked, cancellable child (trip checked before it starts)."""
    dl = dl or Deadline(120)
    rc, out, err = tracked([*common.KUBECTL, "scale", common.WORKLOAD, f"--replicas={n}"], dl, 60)
    if rc != 0:
        raise RuntimeError(f"kubectl scale to {n} failed: {err.strip()[:200]}")


def verify_topology(expect_empty=False):
    """CLUSTER NODES on EVERY allowlisted pod (not one anchor), before and after every cluster-mutating redis-cli command.
    A node IP outside the qrl-calib pod IPs, a pod whose view cannot be read, or an empty read records a TRIP (all scaling stops)
    and aborts. expect_empty=True (before CREATE): every participating pod must see only itself and own no slots."""
    pods = [p for p in redis_pods() if p["ip"]]
    if not pods:
        common.write_trip("topology-unreadable", note="no allowlisted pod to inspect")
        raise RuntimeError("no allowlisted pod to inspect the cluster topology")
    allowed = set(common.allowed_ips(force=True))
    for p in pods:
        try:
            nodes = common.cluster_nodes_strict(p["ip"])
        except RuntimeError as e:
            common.write_trip("topology-unreadable", pod=p["name"], error=str(e)[:150])
            raise
        bad = sorted({nd["addr"].rsplit(":", 1)[0] for nd in nodes} - allowed)
        if bad:
            common.write_trip("foreign-node", pod=p["name"], nodes=bad)
            raise RuntimeError(f"{p['name']} sees nodes outside namespace qrl-calib: {bad}")
        if expect_empty and (len(nodes) != 1 or any(nd["slots"] for nd in nodes)):
            common.write_trip("pre-create-view", pod=p["name"], nodes=len(nodes))
            raise RuntimeError(f"{p['name']} is not a fresh single-node view before CREATE ({len(nodes)} nodes known)")


def anchor_ip():
    for p in redis_pods():
        if p["ready"] and p["ip"] and ping_ok(p["ip"]):
            return p["ip"]
    return None


def covered_now():
    ip = anchor_ip()
    if not ip:
        return False
    st = rcli(ip, "cluster", "info", timeout=5)[1]
    return "cluster_state:ok" in st and common.slots_covered(common.cluster_nodes(ip)) == 16384


def run_rebalance(dl, extra=(), what="rebalance"):
    rebalance_logged(dl, what, extra)
    verify_topology()


def ensure_covered(dl, n_pods):
    """After ANY scale event: forget dead nodes, `--cluster fix` until all 16384 slots are covered, then wait for health.
    Emits fix_start / covered (first time coverage and cluster_state ok are seen)."""
    t_fix = False
    emitted = False
    while True:
        dl.check("covering slots")
        common.require_no_trip()   # cancellation point before every iteration (forget / fix are mutating)
        pods = [p for p in redis_pods() if p["ip"] and p["ready"]]
        anchor = anchor_ip()
        if anchor is None:
            time.sleep(1)
            continue
        ips = {p["ip"] for p in pods}
        for nd in common.cluster_nodes(anchor):  # dead members must be forgotten on every survivor (gossip re-adds them)
            if nd["addr"].rsplit(":", 1)[0] not in ips or any(f in nd["flags"] for f in ("fail", "noaddr")):
                for p in pods:
                    tracked(["redis-cli", "-h", p["ip"], "-p", str(common.PORT), "cluster", "forget", nd["id"]], dl, 10)
        if covered_now():
            if not emitted:
                event("covered")
                emitted = True
            if common.health(n_pods)["ok"]:
                return
        else:
            if not t_fix:
                event("fix_start")
                t_fix = True
            verify_topology()
            rc, out, err = tracked_logged("fix", ["redis-cli", "--cluster", "fix", f"{anchor}:{common.PORT}", "--cluster-yes"], dl, 120)
            print(out[-400:], err[-200:], flush=True)
            verify_topology()
        time.sleep(1)


def bootstrap(n, dl):
    pods = wait_ready(n, dl)
    for p in pods:
        cov = common.slots_covered(common.cluster_nodes(p["ip"]))
        if cov != 0:
            raise SystemExit(f"{p['name']} already owns slots; bootstrap only forms a new cluster")
    verify_topology(expect_empty=True)   # every participating node: empty view, allowlisted, before CREATE
    event("bootstrap_start", pods=n)
    addrs = [f"{p['ip']}:{common.PORT}" for p in pods]
    rc, out, err = tracked(["redis-cli", "--cluster", "create", *addrs, "--cluster-replicas", "0", "--cluster-yes"], dl, 120)
    print(out, err, flush=True)
    if rc != 0:
        raise SystemExit("cluster create failed")
    verify_topology()
    wait_healthy(n, dl, "bootstrap health")
    event("bootstrap_done", pods=n)


def scale_up_one(dl):
    common.require_no_trip()
    pods = redis_pods()
    cur = len(pods)
    existing_ips = {p["ip"] for p in pods}
    event("scale_cmd", direction="up", from_pods=cur, to_pods=cur + 1)
    scale_workload(cur + 1, dl)
    new = wait_for(lambda: [p for p in redis_pods() if p["ip"] not in existing_ips and p["ready"] and ping_ok(p["ip"])]
                   if len(redis_pods()) == cur + 1 else None, dl, "waiting for the new pod")[0]
    event("pod_ready", pod=new["name"], ip=new["ip"])
    served = {"base": common.served_calls(new["ip"]), "done": False}   # C5: first CLIENT request served (not slot ownership)
    anchor = anchor_ip()
    verify_topology()
    rc, out, err = tracked_logged("add-node", ["redis-cli", "--cluster", "add-node", f"{new['ip']}:{common.PORT}", f"{anchor}:{common.PORT}"], dl, 60)
    print(out, err, flush=True)
    if rc != 0:
        raise RuntimeError("add-node failed")
    event("add_node", pod=new["name"])
    verify_topology()
    nid = wait_for(lambda: node_id_of(anchor, new["ip"]), dl, "new node visible")
    event("rebalance_start", pod=new["name"])
    first = []

    def watch():
        if not first and my_slots(anchor, nid) > 0:
            first.append(1)
            event("first_slot", pod=new["name"])
        watch_served()

    def watch_served():
        if served["done"] or served["base"] is None:
            return
        n = common.served_calls(new["ip"])
        if n is not None and n > served["base"]:
            served["done"] = True
            event("first_served", pod=new["name"], calls=n - served["base"])
    wait_converged(dl, cur + 1)
    rebalance_logged(dl, "rebalance", watch=watch, n_nodes=cur + 1)
    if not first:
        event("first_slot", pod=new["name"], note="seen only after rebalance end")
    t_wait = time.time() + 5.0   # under load the first GET/SET arrives within a second of slot ownership; at idle there is none
    while not served["done"] and served["base"] is not None and time.time() < t_wait:
        watch_served()
        time.sleep(0.25)
    if not served["done"]:
        event("first_served_none", pod=new["name"], note="no client GET/SET reached the new pod within 5 s of rebalance end (idle?)")
    verify_topology()
    event("rebalance_end", pod=new["name"], slots=my_slots(anchor, nid))
    ensure_covered(dl, cur + 1)
    event("healthy", pods=cur + 1)


def scale_down_one(dl):
    """Plain replica decrement, exactly like the 2025 agent. Whichever pod Kubernetes removes is recorded, not judged."""
    common.require_no_trip()
    pods = redis_pods()
    cur = len(pods)
    if cur <= 1:
        raise SystemExit("refusing to go below 1 pod")
    anchor = anchor_ip()
    view = {nd["addr"].rsplit(":", 1)[0]: nd for nd in common.cluster_nodes(anchor)} if anchor else {}
    keys = {p["ip"]: dbsize(p["ip"]) for p in pods if p["ip"]}
    event("scale_cmd", direction="down", from_pods=cur, to_pods=cur - 1)
    scale_workload(cur - 1, dl)
    wait_for(lambda: len(redis_pods()) == cur - 1, dl, "pod removal")
    left = {p["name"] for p in redis_pods()}
    gone = [p for p in pods if p["name"] not in left]
    for g in gone:
        nd = view.get(g["ip"])
        lost = sum(hi - lo + 1 for lo, hi in nd["slots"]) if nd else 0
        event("pod_removed", pod=g["name"], slots_lost=lost, keys_lost=keys.get(g["ip"]),
              was_newest=(g["name"] == pods[-1]["name"]))
    ensure_covered(dl, cur - 1)
    event("rebalance_start", pod="all")
    run_rebalance(dl)
    event("rebalance_end", pod="all")
    ensure_covered(dl, cur - 1)
    event("healthy", pods=cur - 1)


def scale_to(target, max_pods, timeout, dry):
    if not 1 <= target <= min(max_pods, MAX_PODS_HARD):
        raise SystemExit(f"target {target} outside [1, {min(max_pods, MAX_PODS_HARD)}]")
    common.check_context()
    dl = Deadline(timeout)
    h = common.health()
    cur = h["pods"]
    print("start state", h, flush=True)
    if not h["ok"]:
        raise SystemExit(f"cluster not healthy before scaling: {h['reasons']}; nothing changed")
    print(f"plan: {cur} -> {target}: {abs(target - cur)} single-pod {'up' if target > cur else 'down'} steps", flush=True)
    if dry:
        return
    event("scale_to_start", from_pods=cur, to_pods=target)
    while len(redis_pods()) < target:
        scale_up_one(dl)
    while len(redis_pods()) > target:
        scale_down_one(dl)
    event("scale_to_done", pods=target)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target", type=int)
    g.add_argument("--bootstrap", type=int)
    ap.add_argument("--max", type=int, default=MAX_PODS_HARD)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    common.check_context()
    common.install_sigterm_as_exception()
    mem = common.MemWatch(on_trip=CHILDREN.reap) if not a.dry_run else None  # rebalance/migration also uses memory
    if mem:
        mem.expect_pods = True   # pods must exist for the whole scaling run
    try:
        common.require_no_trip()
        if a.bootstrap:
            if a.dry_run:
                print(f"would form a {a.bootstrap}-master cluster (no replicas) on the current pods")
            else:
                bootstrap(a.bootstrap, Deadline(a.timeout))
        else:
            scale_to(a.target, a.max, a.timeout, a.dry_run)
    except (TimeoutError, RuntimeError, KeyboardInterrupt) as e:
        event("scale_failed", error=str(e))
        sys.exit(f"FAILED: {e}. Cluster state is unknown; run rollback.sh.")
    finally:
        CHILDREN.reap()
        if mem:
            mem.close()


if __name__ == "__main__":
    main()
