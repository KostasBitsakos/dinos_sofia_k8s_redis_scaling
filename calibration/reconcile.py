#!/usr/bin/env python3
"""Idempotent reconciliation to the starting state: TARGET pods, all 16384 slots covered, no open (migrating/importing)
slots, no stale or failed cluster members, every pod a master, no calibration keys. Works from any interrupted state.

RUNS ONLY VIA tools/launch_run.py WITH AN APPROVED real SPEC (called by common.recover_after_trip after a memory trip).
Mutates the cluster in namespace qrl-calib only. The spec's rollback is `kubectl delete namespace qrl-calib` (rollback.sh).

`next_action(state, target)` is a pure function of an observed state, so it is tested against a fake cluster
(tests/test_reconcile.py). The loop observes, applies ONE action, observes again, and ends only when `done`;
it fails loudly (non-zero exit) if it does not converge, if an action fails, or if calibration keys remain.
"""
import argparse
import sys

import common
from common import event


def next_action(st, target):
    """st = {pods:[{name,ip,ready,dbsize,cal_keys,view:{covered,known}}], nodes:[{id,addr,flags,slots,open}], covered:int,
    keys_left:int (cal:* keys), foreign_keys:int (other keys), formed:bool}. Returns (action, arg).
    Order matters: repair before growth, growth before shrink, shrink before rebalance. Test data (cal:*) may be lost
    (decision R-20261016-018); keys that are not ours abort everything."""
    pods, nodes = st["pods"], st["nodes"]
    if any(not p["ready"] or not p["ip"] for p in pods):
        return ("wait", None)
    if st.get("foreign_keys", 0) > 0:
        return ("abort", f"{st['foreign_keys']} keys that are not cal:* exist; refusing to touch the cluster")
    ips = {p["ip"] for p in pods}
    addr_ip = lambda n: n["addr"].rsplit(":", 1)[0]
    if st["covered"] == 0 and not any(n["slots"] for n in nodes):
        # no slots anywhere: nothing to lose that matters (cal:* data loss is accepted, R-20261016-018)
        if len(pods) != target:
            return ("plain_scale", target)
        return ("bootstrap", target)
    views = [p["view"] for p in pods if p.get("view")]
    if any(v["covered"] != 16384 or v["known"] != len(pods) for v in views) and len(views) == len(pods) and \
            not any(n["open"] for n in nodes) and st["covered"] == 16384:
        # every pod must see the same, complete topology; a divergent view is repaired by fix before anything else
        return ("fix", None)
    if any(n["open"] for n in nodes) or st["covered"] < 16384:
        return ("fix", None)
    bad = [n for n in nodes if addr_ip(n) not in ips or any(f in n["flags"] for f in ("fail", "fail?", "noaddr", "handshake"))]
    if bad:
        return ("forget", bad[0]["id"])
    known = {addr_ip(n) for n in nodes}
    missing = [p for p in pods if p["ip"] not in known]
    if missing:
        return ("add_node", missing[0]["ip"])
    if len(pods) > target:
        return ("scale_down_one", None)  # plain replica decrement + fix, as in 2025; lost cal:* keys are acceptable
    if len(pods) < target:
        return ("scale_up_one", None)
    if any(not n["slots"] for n in nodes if "master" in n["flags"]):
        return ("rebalance", None)
    if st["keys_left"] > 0:
        return ("cleanup", None)
    return ("done", None)


def reconcile(ops, target, max_iters=80):
    for i in range(max_iters):
        st = ops.observe()
        act, arg = next_action(st, target)
        if act == "abort":
            raise RuntimeError(str(arg))
        event("reconcile_step", i=i, action=act, arg=arg, pods=len(st["pods"]), covered=st["covered"],
              keys_left=st["keys_left"])
        if act == "done":
            return True
        getattr(ops, act)(arg)
        if act == "cleanup":
            left = ops.observe()["keys_left"]
            if left > 0:
                raise RuntimeError(f"cleanup left {left} calibration keys")
    raise RuntimeError(f"reconcile did not converge in {max_iters} steps")


class RealOps:
    """Every mutation runs through scale_to.tracked (cancellable, killed on a trip); only reads use common.sh/rcli."""

    def __init__(self, timeout):
        import scale_to
        self.st, self.dl = scale_to, scale_to.Deadline(timeout)

    def _t(self, cmd, timeout=120):
        return self.st.tracked(cmd, self.dl, timeout)

    def _anchor(self, pods):
        for p in pods:
            if p["ip"] and self.st.ping_ok(p["ip"]):
                return p["ip"]
        return None

    def observe(self):
        pods = common.redis_pods()
        anchor = self._anchor(pods)
        nodes = common.cluster_nodes(anchor) if anchor else []
        keys = foreign = 0
        for p in pods:
            if not p["ip"]:
                continue
            rc, out, _ = common.rcli(p["ip"], "--scan", "--pattern", "cal:*", timeout=60)
            rc2, dbs, _e = common.rcli(p["ip"], "dbsize", timeout=10)
            if rc != 0 or rc2 != 0 or not dbs.strip().isdigit():
                raise RuntimeError(f"cannot read keys on {p['name']}")
            p["cal_keys"], p["dbsize"] = len(out.split()), int(dbs.strip())
            keys += p["cal_keys"]
            foreign += p["dbsize"] - p["cal_keys"]
            vn = common.cluster_nodes(p["ip"])  # this pod's own view of the topology
            p["view"] = {"covered": common.slots_covered(vn), "known": len(vn)}
        return {"pods": pods, "nodes": nodes, "covered": common.slots_covered(nodes), "keys_left": keys,
                "foreign_keys": foreign, "formed": len(nodes) > 1}

    def wait(self, _):
        import time
        self.dl.check("waiting for pods")
        time.sleep(3)

    def plain_scale(self, n):
        self.st.scale_workload(n, self.dl)
        self.st.wait_for(lambda: len(common.redis_pods()) == n, self.dl, "plain scale")

    def bootstrap(self, n):
        # data are disposable test keys (R-20261016-018) and foreign keys were already ruled out by observe: flush, then reset
        for p in common.redis_pods():
            self._t(["redis-cli", "-h", p["ip"], "-p", str(common.PORT), "flushall"], 30)
        for p in common.redis_pods():
            rc, out, err = self._t(["redis-cli", "-h", p["ip"], "-p", str(common.PORT), "cluster", "reset", "hard"], 30)
            if rc != 0 or "OK" not in out:
                raise RuntimeError(f"cluster reset failed on {p['name']}: {out} {err}")
        self.st.bootstrap(n, self.dl)

    def fix(self, _):
        self.st.verify_topology()
        rc, out, err = self._t(["redis-cli", "--cluster", "fix", f"{self._anchor(common.redis_pods())}:{common.PORT}", "--cluster-yes"], 300)
        if rc != 0:
            raise RuntimeError(f"cluster fix failed: {out[-300:]} {err[-300:]}")
        self.st.verify_topology()

    def forget(self, node_id):
        for p in common.redis_pods():
            if p["ip"]:
                self._t(["redis-cli", "-h", p["ip"], "-p", str(common.PORT), "cluster", "forget", node_id], 10)

    def add_node(self, ip):
        anchor = self._anchor([p for p in common.redis_pods() if p["ip"] != ip])
        self.st.verify_topology()
        rc, out, err = self._t(["redis-cli", "--cluster", "add-node", f"{ip}:{common.PORT}", f"{anchor}:{common.PORT}"], 60)
        if rc != 0:
            raise RuntimeError(f"add-node failed: {err[-200:]} {out[-200:]}")
        self.st.verify_topology()

    def scale_up_one(self, _):
        self.st.scale_up_one(self.dl)

    def scale_down_one(self, _):
        self.st.scale_down_one(self.dl)

    def rebalance(self, _):
        self.st.run_rebalance(self.dl)

    def cleanup(self, _):
        for p in common.redis_pods():
            rc, out, err = self._t(["redis-cli", "-h", p["ip"], "-p", str(common.PORT), "flushall"], 60)
            if rc != 0:
                raise RuntimeError(f"flushall failed on {p['name']}: {err[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    common.check_context()
    common.install_sigterm_as_exception()
    import scale_to
    try:
        reconcile(RealOps(a.timeout), a.target)
        h = common.health(a.target)
        event("reconcile_done", **h)
        if not h["ok"]:
            raise RuntimeError(f"final health failed: {h['reasons']}")
    except (RuntimeError, TimeoutError, KeyboardInterrupt) as e:
        event("reconcile_failed", error=str(e))
        sys.exit(f"ROLLBACK FAILED: {e}")
    finally:
        scale_to.CHILDREN.reap()


if __name__ == "__main__":
    main()
