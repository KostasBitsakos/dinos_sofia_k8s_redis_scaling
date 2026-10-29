"""Safety paths: deterministic pod removal, child reaping, cgroup memory reader. All mocked; no cluster."""
import os
import subprocess
import time

import pytest

import common
import scale_to


class FakeCluster2:
    """Mock of kubectl + redis-cli for scale_down_one: Kubernetes removes the pod named `removes`."""

    def __init__(self, removes="p0", slots_of_removed=5461):
        self.pods = [{"name": f"p{i}", "ip": f"10.0.0.{i}", "ready": True, "uid": f"u{i}", "created": str(i)} for i in range(4)]
        self.removes, self.lost = removes, slots_of_removed
        self.calls, self.events, self.covered = [], [], False
        self.fixes = 0

    def install(self, mp):
        def tracked(cmd, dl, timeout=120):
            a = tuple(cmd[5:]) if cmd[0] == "kubectl" else tuple(cmd)   # drop the kubectl --context/--namespace prefix
            self.calls.append(a)
            if a[0] == "scale":
                self.pods = [p for p in self.pods if p["name"] != self.removes]
            return 0, "", ""
        mp.setattr(scale_to, "redis_pods", lambda: list(self.pods))
        mp.setattr(scale_to, "tracked", tracked)
        mp.setattr(scale_to, "anchor_ip", lambda: self.pods[0]["ip"])
        mp.setattr(scale_to, "dbsize", lambda ip: 11)
        mp.setattr(common, "require_no_trip", lambda: None)
        mp.setattr(common, "cluster_nodes", lambda ip: [{"id": f"id{i}", "addr": f"10.0.0.{i}:6379", "flags": ["master"],
                                                           "slots": [(0, self.lost - 1)] if f"10.0.0.{i}" == next(
                                                               p["ip"] for p in [{"ip": "10.0.0." + self.removes[1:]}]) else []}
                                                          for i in range(4)])
        mp.setattr(scale_to, "ensure_covered", lambda dl, n: self.events.append(("ensure_covered", n)))
        mp.setattr(scale_to, "run_rebalance", lambda dl, **k: self.events.append(("rebalance",)))
        mp.setattr(scale_to, "event", lambda name, **kw: self.events.append((name, kw)))
        mp.setattr(scale_to.time, "sleep", lambda s: None)


def test_scale_down_is_a_plain_replica_decrement_and_records_the_victim(monkeypatch):
    k = FakeCluster2(removes="p0", slots_of_removed=5461)   # Kubernetes removes the OLDEST pod: not judged, only recorded
    k.install(monkeypatch)
    scale_to.scale_down_one(scale_to.Deadline(30))
    scales = [c for c in k.calls if c[0] == "scale"]
    assert len(scales) == 1 and scales[0][-1] == "--replicas=3" and scales[0][1] == "deployment/redis-autoscale-cluster"
    assert not any(c[0] == "annotate" for c in k.calls)           # no deletion-cost games
    rem = [e for e in k.events if e[0] == "pod_removed"][0][1]
    assert rem["pod"] == "p0" and rem["slots_lost"] == 5461 and rem["keys_lost"] == 11 and rem["was_newest"] is False
    names = [e[0] for e in k.events]
    assert names.index("pod_removed") < names.index("ensure_covered") < names.index("rebalance") < names.index("healthy")


def test_scale_down_with_newest_pod_removed_also_works(monkeypatch):
    k = FakeCluster2(removes="p3", slots_of_removed=100)
    k.install(monkeypatch)
    scale_to.scale_down_one(scale_to.Deadline(30))
    rem = [e for e in k.events if e[0] == "pod_removed"][0][1]
    assert rem["pod"] == "p3" and rem["was_newest"] is True and rem["slots_lost"] == 100


def test_ensure_covered_forgets_dead_nodes_and_runs_fix_until_covered(monkeypatch):
    state = {"covered": False, "fix": 0, "forget": []}
    pods = [{"name": "p1", "ip": "10.0.0.1", "ready": True}, {"name": "p2", "ip": "10.0.0.2", "ready": True}]
    monkeypatch.setattr(scale_to, "redis_pods", lambda: pods)
    monkeypatch.setattr(scale_to, "anchor_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(common, "cluster_nodes", lambda ip: [
        {"id": "a", "addr": "10.0.0.1:6379", "flags": ["master"], "slots": []},
        {"id": "dead", "addr": "10.0.0.9:6379", "flags": ["master", "fail"], "slots": []}])
    monkeypatch.setattr(scale_to, "covered_now", lambda: state["covered"])
    monkeypatch.setattr(scale_to, "verify_topology", lambda: None)

    def fake_tracked(cmd, dl, timeout=120):
        if "forget" in cmd:
            state["forget"].append((cmd[2], tuple(cmd[5:7])))
            return 0, "OK", ""
        state["fix"] += 1
        state["covered"] = state["fix"] >= 2          # coverage needs two fix passes
        return 0, "", ""
    monkeypatch.setattr(scale_to, "tracked", fake_tracked)
    monkeypatch.setattr(common, "require_no_trip", lambda: None)
    monkeypatch.setattr(common, "health", lambda n=None: {"ok": True})
    ev = []
    monkeypatch.setattr(scale_to, "event", lambda name, **kw: ev.append(name))
    monkeypatch.setattr(scale_to.time, "sleep", lambda s: None)
    scale_to.ensure_covered(scale_to.Deadline(30), 2)
    assert state["fix"] == 2 and ev == ["fix_start", "covered"]
    assert {ip for ip, a in state["forget"]} == {"10.0.0.1", "10.0.0.2"} and all(a == ("cluster", "forget") for _, a in state["forget"])


def test_children_reap_terminates_and_waits():
    ch = common.Children()
    p = ch.popen(["sleep", "60"])
    t = time.time()
    ch.reap(grace=5)
    assert p.poll() is not None and time.time() - t < 6 and ch.procs == []


def test_pod_memory_reads_working_set_from_cgroup(tmp_path):
    d = tmp_path / "kubepods-burstable-podabc_def.slice:cri-containerd:1"
    sandbox = tmp_path / "kubepods-burstable-podabc_def.slice:cri-containerd:0"
    for x, lim in ((d, 16777216), (sandbox, 9223372036854771712)):
        x.mkdir()
        (x / "memory.limit_in_bytes").write_text(str(lim))
    (d / "memory.usage_in_bytes").write_text("16000000")
    (d / "memory.stat").write_text("cache 8000000\ninactive_file 5000000\nrss 3000000\n")
    usage, lim, src = common.pod_memory({"name": "x", "uid": "abc-def"}, cg_root=str(tmp_path))
    assert (usage, lim, src) == (11000000, 16777216, "cgroup")


def test_unreadable_memory_fails_closed(monkeypatch):
    monkeypatch.setattr(common, "sh", lambda *a, **k: (1, "", "no metrics"))
    with pytest.raises(RuntimeError):
        common.pod_memory({"name": "x", "uid": "nope"}, cg_root="/nonexistent")


def res(frac, aof=0, rewrite=0, bgsave=0, pod="p"):
    return {"pod": pod, "frac": frac, "aof_size": aof, "aof_rewrite": rewrite, "bgsave": bgsave}


def test_evaluate_trips_on_memory_persistence_flags_and_aof():
    assert common.evaluate([res(0.5), res(0.7)], 0)[0] is None
    assert common.evaluate([res(0.5), res(0.96)], 0)[0] == "memory"
    assert common.evaluate([res(0.5), res(0.9, bgsave=1)], 0)[0] == "persist-spike"   # fork near the limit
    assert common.evaluate([res(0.5, bgsave=1)], 0)[0] is None                          # fork with headroom: only logged
    assert common.evaluate([res(0.5, aof=30_000_000)], 0)[0] == "aof-growth"


def test_check_all_is_concurrent_and_fails_closed(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(common, "check_pod", lambda p: (time.sleep(0.3), {"pod": p["name"]})[1])
    pods = [{"name": f"p{i}"} for i in range(12)]
    t = time.time()
    with ThreadPoolExecutor(16) as pool:
        out = common.check_all(pods, pool)
    assert len(out) == 12 and time.time() - t < 1.0   # sequential would take 3.6 s

    def boom(p):
        raise RuntimeError("unreadable")
    monkeypatch.setattr(common, "check_pod", boom)
    with ThreadPoolExecutor(4) as pool, pytest.raises(RuntimeError):
        common.check_all(pods[:2], pool)


def test_persistence_response_is_checked(monkeypatch):
    monkeypatch.setattr(common, "redis_info", lambda ip, section, timeout=2.0: "aof_enabled:1\r\n")  # fields missing
    with pytest.raises(RuntimeError, match="no rdb_bgsave_in_progress|no aof_rewrite_in_progress"):
        common.persistence_state("1.2.3.4")
    monkeypatch.setattr(common, "redis_info", lambda ip, section, timeout=2.0:
                        "aof_enabled:1\r\naof_rewrite_in_progress:0\r\nrdb_bgsave_in_progress:0\r\n")
    with pytest.raises(RuntimeError, match="AOF is on"):
        common.persistence_state("1.2.3.4")
    monkeypatch.setattr(common, "redis_info", lambda ip, section, timeout=2.0:
                        "aof_enabled:0\r\naof_rewrite_in_progress:0\r\nrdb_bgsave_in_progress:1\r\n")
    assert common.persistence_state("1.2.3.4")["bgsave"] == 1
    monkeypatch.setattr(common, "redis_info", lambda ip, section, timeout=2.0:
                        "aof_enabled:1\r\naof_current_size:5\r\naof_rewrite_in_progress:1\r\nrdb_bgsave_in_progress:0\r\n")
    assert common.persistence_state("1.2.3.4") == {"aof_enabled": 1, "aof_size": 5, "aof_rewrite": 1, "bgsave": 0}


def test_cpu_lookup_uses_cpuacct_tree_by_container_name(tmp_path):
    mem, cpu = tmp_path / "mem", tmp_path / "cpu"
    name = "kubepods-burstable-podabc_def.slice:cri-containerd:9"
    (mem / name).mkdir(parents=True)
    (mem / name / "memory.limit_in_bytes").write_text("16777216")
    (cpu / name).mkdir(parents=True)
    (cpu / name / "cpuacct.usage").write_text("12345")
    assert common.pod_cpu_ns({"uid": "abc-def"}, mem_root=str(mem), cpu_root=str(cpu)) == 12345


def test_trip_blocks_scaling_until_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    common.require_no_trip()
    open(common.trip_marker(), "w").write("memory")
    with pytest.raises(RuntimeError, match="scaling is stopped"):
        common.require_no_trip()
    calls = []
    monkeypatch.setattr(common.subprocess, "run", lambda cmd, timeout=None: calls.append(cmd) or type("R", (), {"returncode": 0})())
    assert common.recover_after_trip() == 0 and "reconcile.py" in calls[0][1]
    common.require_no_trip()  # marker removed after a successful recovery
