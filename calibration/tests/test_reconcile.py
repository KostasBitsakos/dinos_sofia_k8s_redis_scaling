"""reconcile.next_action / reconcile() against a fake cluster, from several interrupted states."""
import random

import pytest

import reconcile


class FakeCluster:
    """Minimal model: pods with ips, cluster membership with slot ownership, open slots, keys."""

    def __init__(self, pods, owners, extra_nodes=(), open_slots=False, keys=0, formed=True):
        self.pods = {f"p{i}": {"ip": f"10.0.0.{i}", "ready": True} for i in range(pods)}
        self.members = {}  # node id -> {ip, slots, flags}
        self.open_slots = open_slots
        self.keys = keys  # cal:* keys spread over pods round-robin
        self.pod_keys = {n: 0 for n in self.pods}
        for i in range(keys):
            self.pod_keys[sorted(self.pods)[i % len(self.pods)]] += 1
        self.foreign = {}  # pod name -> non-cal keys
        self.removed_with_keys = []
        for i, n in enumerate(owners):
            self.members[f"n{i}"] = {"ip": f"10.0.0.{i}", "slots": n, "flags": ["master"]}
        for j, (ip, n, fl) in enumerate(extra_nodes):
            self.members[f"x{j}"] = {"ip": ip, "slots": n, "flags": fl}
        self.log = []

    def observe(self):
        pods = [{"name": k, "ip": v["ip"], "ready": v["ready"], "cal_keys": self.pod_keys.get(k, 0),
                 "dbsize": self.pod_keys.get(k, 0) + self.foreign.get(k, 0)} for k, v in sorted(self.pods.items())]
        nodes = [{"id": k, "addr": v["ip"] + ":6379", "flags": v["flags"], "slots": [(0, v["slots"] - 1)] if v["slots"] else [],
                  "open": ["[1->-x]"] if (self.open_slots and i == 0) else []} for i, (k, v) in enumerate(self.members.items())]
        cov = sum(v["slots"] for v in self.members.values() if "master" in v["flags"]) if not self.open_slots else sum(
            v["slots"] for v in self.members.values())
        return {"pods": pods, "nodes": nodes, "covered": min(cov, 16384), "keys_left": sum(self.pod_keys.values()),
                "foreign_keys": sum(self.foreign.values()), "formed": len(nodes) > 1}

    def _rebal(self):
        ms = [v for v in self.members.values() if "master" in v["flags"] and v["ip"] in {p["ip"] for p in self.pods.values()}]
        for v in ms:
            v["slots"] = 16384 // len(ms)
        ms[0]["slots"] += 16384 - sum(v["slots"] for v in ms)

    def wait(self, _): pass

    def plain_scale(self, n):
        while len(self.pods) < n:
            i = max(int(k[1:]) for k in self.pods) + 1 if self.pods else 0
            self.pods[f"p{i}"] = {"ip": f"10.0.0.{i}", "ready": True}
        while len(self.pods) > n:
            self._drop(sorted(self.pods)[-1])

    def _drop(self, name):
        if self.pod_keys.get(name, 0) + self.foreign.get(name, 0) > 0:
            self.removed_with_keys.append(name)  # data loss: the tests assert this never happens
        self.pods.pop(name)

    def bootstrap(self, n):
        self.members = {f"n{i}": {"ip": p["ip"], "slots": 0, "flags": ["master"]} for i, p in enumerate(self.pods.values())}
        self._rebal()

    def fix(self, _):
        self.open_slots = False
        total = sum(v["slots"] for v in self.members.values())
        if total < 16384:
            next(iter(self.members.values()))["slots"] += 16384 - total

    def forget(self, nid):
        self.members.pop(nid)

    def add_node(self, ip):
        self.members[f"n{ip}"] = {"ip": ip, "slots": 0, "flags": ["master"]}

    def scale_up_one(self, _):
        self.plain_scale(len(self.pods) + 1)
        self.add_node(self.pods[sorted(self.pods)[-1]]["ip"])
        self._rebal()

    def scale_down_one(self, _):
        name = sorted(self.pods)[-1]
        ip = self.pods[name]["ip"]
        for k in [k for k, v in self.members.items() if v["ip"] == ip]:
            self.members.pop(k)
        self._drop(name)
        self._rebal()

    def rebalance(self, _):
        self._rebal()

    def cleanup(self, _):
        self.pod_keys = {n: 0 for n in self.pods}


def converge(c, target=3):
    assert reconcile.reconcile(c, target)
    st = c.observe()
    assert len(st["pods"]) == target and st["covered"] == 16384 and st["keys_left"] == 0
    assert all(n["slots"] for n in st["nodes"]) and not any(n["open"] for n in st["nodes"])
    assert {n["addr"].split(":")[0] for n in st["nodes"]} == {p["ip"] for p in st["pods"]}
    # idempotent: a second pass takes no action
    assert reconcile.next_action(c.observe(), target)[0] == "done"


def test_unformed_cluster_is_bootstrapped():
    converge(FakeCluster(3, [0, 0, 0]))


def test_extra_pods_open_slots_and_keys():
    converge(FakeCluster(6, [3000, 3000, 3000, 3000, 2000, 2384], open_slots=True, keys=40))


def test_missing_pod_in_cluster_and_stale_member():
    c = FakeCluster(3, [8192, 8192], extra_nodes=[("10.0.9.9", 0, ["master", "fail"])])
    converge(c)


def test_uncovered_slots_are_fixed():
    converge(FakeCluster(3, [5000, 5000, 5000]))


def test_too_few_pods():
    converge(FakeCluster(1, [16384]))


@pytest.mark.parametrize("seed", range(5))
def test_random_interruptions(seed):
    r = random.Random(seed)
    pods = r.randint(1, 8)
    owners = [r.randint(0, 5000) for _ in range(pods)]
    converge(FakeCluster(pods, owners, open_slots=r.random() < 0.5, keys=r.randint(0, 9)))


def test_cleanup_failure_is_loud():
    class Bad(FakeCluster):
        def cleanup(self, _):
            pass  # keys remain
    with pytest.raises(RuntimeError):
        reconcile.reconcile(Bad(3, [5462, 5461, 5461], keys=3), 3)


def test_shrink_with_keys_is_allowed_data_loss_is_accepted():
    """R-20261016-018: stateless calibration namespace; lost test keys are fine, the cluster must still end covered."""
    c = FakeCluster(4, [5462, 5461, 5461, 0], keys=0)
    c.pod_keys["p3"] = 7
    assert reconcile.reconcile(c, 3)
    assert len(c.observe()["pods"]) == 3 and c.observe()["covered"] == 16384


def test_foreign_keys_abort_everything():
    c = FakeCluster(5, [4000, 4000, 4000, 4384, 0])
    c.foreign["p4"] = 2
    with pytest.raises(RuntimeError, match="not cal"):
        reconcile.reconcile(c, 3)
    assert len(c.pods) == 5  # nothing was touched
