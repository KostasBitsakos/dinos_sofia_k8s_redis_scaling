"""Isolation and fail-closed behaviour: namespace lock, explicit kubectl context, endpoint allowlist, rollback proof of absence,
tracked mutating commands, memory-watch pod refresh, image pin, bounded recovery search."""
import json
import os
import re
import stat
import subprocess
import sys
import threading
import time

import pytest

import common
import fit
import scale_to

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_namespace_is_hard_coded():
    assert common.NAMESPACE == "qrl-calib" and common.KUBECTL[:5] == ["kubectl", "--context", common.CONTEXT, "--namespace", "qrl-calib"]
    for mod in ("common", "loadgen"):
        r = subprocess.run([sys.executable, "-c", f"import {mod}"], cwd=HERE, capture_output=True, text=True,
                           env={**os.environ, "QRL_NAMESPACE": "default"})
        assert r.returncode != 0 and "qrl-calib" in (r.stderr + r.stdout)


def test_every_kubectl_call_names_context_and_namespace():
    for f in os.listdir(HERE):
        if f.endswith(".py"):
            for n, line in enumerate(open(os.path.join(HERE, f)), 1):
                if '"kubectl"' in line and not line.lstrip().startswith("#"):
                    ok = (f == "common.py" and ('KUBECTL = ["kubectl", "--context"' in line or '"config"' in line or 'exe == "kubectl"' in line)) or \
                         (f in ("loadgen.py", "make_spec_stageB.py") and '"--context"' in line)
                    assert ok, f"{f}:{n}: kubectl call without explicit context/namespace: {line.strip()}"
    for spec in ("calibration_spec.json", "transitions_spec.template.json", "spec_stageB.json", "spec_stageB_optional16.json"):
        sp = json.load(open(os.path.join(HERE, spec)))
        assert sp["real"]["namespace"] == "qrl-calib"
        for e in sp["entries"]:
            cmd = e["cmd"][e["cmd"].index("--") + 1:] if "--" in e["cmd"] else e["cmd"]
            if cmd[0] == "kubectl":
                assert cmd[1:5] == ["--context", "kubernetes-admin@kubernetes", "--namespace", "qrl-calib"], e
    assert "kubectl --context $CTX" in open(os.path.join(HERE, "rollback.sh")).read()


def test_endpoints_outside_the_namespace_are_refused(monkeypatch):
    monkeypatch.setattr(common, "allowed_ips", lambda force=False: frozenset({"10.244.1.1"}))
    common.check_endpoints(["redis-cli", "-h", "10.244.1.1", "ping"])
    common.check_endpoints(["kubectl", "get", "pods"])          # not redis-cli: ignored
    for bad in (["redis-cli", "-h", "10.244.37.144", "ping"], ["redis-cli", "--cluster", "fix", "10.0.0.5:6379"]):
        with pytest.raises(RuntimeError, match="not a pod of namespace"):
            common.check_endpoints(bad)
    with pytest.raises(RuntimeError):
        common.sh(["redis-cli", "-h", "10.244.37.144", "ping"])


def test_image_is_pinned_by_digest():
    y = open(os.path.join(HERE, "k8s", "qrl-calib.yaml")).read()
    assert re.search(r"image: capitanodocker/redis-autoscale-cluster@sha256:[0-9a-f]{64}", y)
    assert "imagePullPolicy: IfNotPresent" in y
    assert set(re.findall(r"^\s+namespace:\s*(\S+)", y, re.M)) == {"qrl-calib"}


# ---------------------------------------------------------------- rollback.sh with a fake kubectl
def run_rollback(tmp_path, delete_rc=0, get_rc=1, get_msg='Error from server (NotFound): namespaces "qrl-calib" not found'):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    k = fake / "kubectl"
    k.write_text(f"""#!/usr/bin/env bash
args="$*"
case "$args" in
  *"config current-context"*) echo kubernetes-admin@kubernetes; exit 0;;
  *"delete namespace"*) exit {delete_rc};;
  *"get namespace"*) echo '{get_msg}' >&2; exit {get_rc};;
esac
exit 9
""")
    k.chmod(k.stat().st_mode | stat.S_IEXEC)
    return subprocess.run(["bash", os.path.join(HERE, "rollback.sh")], capture_output=True, text=True,
                          env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}, timeout=60)


def test_rollback_succeeds_only_on_notfound(tmp_path):
    r = run_rollback(tmp_path)
    assert r.returncode == 0 and "NotFound" in r.stdout


def test_rollback_fails_loudly_on_delete_error(tmp_path):
    r = run_rollback(tmp_path, delete_rc=1)
    assert r.returncode != 0 and "ROLLBACK FAILED" in r.stdout


def test_rollback_fails_when_namespace_still_exists(tmp_path):
    r = run_rollback(tmp_path, get_rc=0, get_msg="qrl-calib Terminating")
    assert r.returncode != 0 and "still exists" in r.stdout


def test_rollback_does_not_treat_other_errors_as_absence(tmp_path):
    r = run_rollback(tmp_path, get_rc=1, get_msg="The connection to the server was refused")
    assert r.returncode != 0 and "cannot prove absence" in r.stdout


# ---------------------------------------------------------------- tracked mutating commands
def test_tracked_command_is_killed_on_a_memory_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(common, "check_endpoints", lambda cmd: None)
    threading.Timer(0.6, lambda: open(common.trip_marker(), "w").write("memory")).start()
    t = time.time()
    with pytest.raises(RuntimeError, match="memory trip"):
        scale_to.tracked(["sleep", "30"], scale_to.Deadline(60), timeout=60)
    assert time.time() - t < 5
    scale_to.CHILDREN.reap(grace=2)
    assert all(p.poll() is not None for p in scale_to.CHILDREN.procs) or scale_to.CHILDREN.procs == []


def test_tracked_command_refuses_to_start_after_a_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    open(common.trip_marker(), "w").write("memory")
    started = []
    monkeypatch.setattr(scale_to.CHILDREN, "popen", lambda *a, **k: started.append(a))
    with pytest.raises(RuntimeError, match="scaling is stopped"):
        scale_to.tracked(["sleep", "1"], scale_to.Deadline(5))
    assert not started


def test_scale_workload_and_ensure_covered_check_the_trip_first(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    open(common.trip_marker(), "w").write("memory")
    calls = []
    monkeypatch.setattr(scale_to.CHILDREN, "popen", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError):
        scale_to.scale_workload(4)
    with pytest.raises(RuntimeError):
        scale_to.ensure_covered(scale_to.Deadline(5), 3)
    assert calls == []


# ---------------------------------------------------------------- memory watch refresh
def test_removed_pod_does_not_trip_the_memory_watch(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    w = common.MemWatch.__new__(common.MemWatch)
    w.pool = ThreadPoolExecutor(4)
    pods_now = [{"name": "p1", "ip": "1", "uid": "u1", "ready": True, "ready_since": 1.0}]
    stale = [{"name": "p1", "ip": "1", "uid": "u1", "ready": True, "ready_since": 1.0}, {"name": "gone", "ip": "2", "uid": "u2", "ready": True, "ready_since": 1.0}]

    def check_pod(p):
        if p["name"] == "gone":
            raise RuntimeError("cgroup vanished")
        return {"pod": p["name"], "frac": 0.5, "aof_size": 0, "aof_rewrite": 0, "bgsave": 0, "source": "cgroup"}
    monkeypatch.setattr(common, "check_pod", check_pod)
    monkeypatch.setattr(common, "redis_pods", lambda: pods_now)
    state = {"pods": stale, "t_pods": time.time()}
    res = w._step(state)
    assert [r["pod"] for r in res] == ["p1"] and state["pods"] == pods_now
    # a pod that is really unreadable (fresh list) still raises -> trip
    pods_now.append({"name": "gone", "ip": "2", "uid": "u2", "ready": True, "ready_since": 1.0})
    with pytest.raises(Exception):
        w._step({"pods": stale, "t_pods": time.time()})


# ---------------------------------------------------------------- fit: recovery search bounded to the episode
def test_recovery_search_does_not_leave_the_episode():
    C_HI, C_LO = 12000.0, 8000.0
    rows, tele, t = [], [], 1000.0

    def seg(n, comp):
        nonlocal t
        for _ in range(n):
            rows.append({"t": t, "phase": 2, "offered_ops": 1.15 * C_HI, "completed_ops": comp, "served_ops": comp,
                         "p99_ms": 1})
            tele.append({"t": t + 0.5, "ncpu": 40, "host_busy_cores": 2.0, "gen_cores": 1.0, "gen_worker_max_cores": 0.2,
                         "n_pods": 3, "pods_cores": 0.57})
            t += 1
    seg(30, C_HI)                       # before
    t_cmd = t
    seg(60, 0.2 * C_LO)                 # degraded
    t_healthy = t
    seg(35, 0.2 * C_LO)                 # still degraded 35 s after healthy (search limit is healthy + 30 s)
    seg(40, C_LO)                       # throughput only returns after the limit
    ep = {"dir": "down", "t_cmd": t_cmd, "t_removed": t_cmd + 1, "t_covered": t_cmd + 20, "t_healthy": t_healthy,
          "t_next": t, "slots_lost": 1, "keys_lost": 1}
    r, why = fit.episode_loss(rows, ep, tele)
    assert r is None and "not restored" in why
    ep["t_next"] = None
    ep2 = dict(ep)
    ep2["t_healthy"] = t_healthy + 40   # healthy later -> limit later -> the same data now restore inside the episode
    r2, why2 = fit.episode_loss(rows, ep2, tele)
    assert r2 is not None or "not restored" not in (why2 or "")


# ---------------------------------------------------------------- no untracked mutation anywhere
import ast
import signal


def test_no_untracked_mutating_subprocess_in_the_source():
    """Every subprocess/os.system call that involves kubectl or redis-cli (or a variable command) is in an allowlisted function:
    the read-only runner `sh` (which refuses mutating verbs), pure reads, or the tracked runner. Mutations go through
    scale_to.tracked / CHILDREN.popen."""
    allowed = {("common.py", "sh"), ("common.py", "allowed_ips"), ("common.py", "popen"), ("loadgen.py", "allowed_ips"),
               ("loadgen.py", "fetch_slotmap"), ("loadgen.py", "main"),
               ("gate.py", "main")}   # loadgen main: python workers only; gate main: runs the approved spec entry itself as a child
    pat = re.compile(r"kubectl|redis-cli|KUBECTL|\bcmd\b")
    for f in os.listdir(HERE):
        if not f.endswith(".py"):
            continue
        src = open(os.path.join(HERE, f)).read()
        tree = ast.parse(src)
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for c in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                name = ast.unparse(c.func)
                if re.match(r"(subprocess\.(run|check_call|check_output|call|Popen)|os\.system|os\.popen)$", name):
                    seg = ast.get_source_segment(src, c) or ""
                    if pat.search(seg):
                        assert (f, fn.name) in allowed, f"{f}:{fn.name}: untracked subprocess on kubectl/redis-cli: {seg[:80]}"


def test_cluster_meet_never_issued_by_us():
    for f in os.listdir(HERE):
        if f.endswith(".py") and not f.startswith("test"):
            assert not re.search(r'"meet"|cluster meet', open(os.path.join(HERE, f)).read(), re.I), f


def test_readonly_guard_refuses_mutations_in_sh():
    ro = [["kubectl", "--context", "c", "--namespace", "n", "get", "pods"], ["redis-cli", "-h", "1.1.1.1", "cluster", "nodes"],
          ["redis-cli", "-h", "1.1.1.1", "dbsize"], ["redis-cli", "-h", "1.1.1.1", "--scan", "--pattern", "cal:*"],
          ["redis-cli", "-h", "1.1.1.1", "info", "memory"], ["python3", "x.py"]]
    rw = [[*common.KUBECTL, "scale", "deployment/x", "--replicas=2"], [*common.KUBECTL, "delete", "namespace", "qrl-calib"],
          ["kubectl", "apply", "-f", "x"], ["redis-cli", "--cluster", "fix", "1.1.1.1:6379"], ["redis-cli", "-h", "1.1.1.1", "flushall"],
          ["redis-cli", "-h", "1.1.1.1", "cluster", "forget", "abc"], ["redis-cli", "-h", "1.1.1.1", "cluster", "reset", "hard"],
          ["redis-cli", "-h", "1.1.1.1", "cluster", "meet", "9.9.9.9", "6379"], ["redis-cli", "-h", "1.1.1.1", "unlink", "k"]]
    assert all(common.is_readonly(c) for c in ro) and not any(common.is_readonly(c) for c in rw)
    with pytest.raises(RuntimeError, match="mutating command refused"):
        common.sh([*common.KUBECTL, "scale", "deployment/x", "--replicas=2"])


# ---------------------------------------------------------------- topology allowlist
def topo(monkeypatch, tmp_path, views, pods=("10.0.0.1", "10.0.0.2", "10.0.0.3")):
    """views: ip -> list of node dicts | Exception. Returns nothing; sets up mocks."""
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(scale_to, "redis_pods", lambda: [{"name": f"p{i}", "ip": ip, "ready": True} for i, ip in enumerate(pods)])
    monkeypatch.setattr(common, "allowed_ips", lambda force=False: frozenset(pods))

    def strict(ip):
        v = views[ip]
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr(common, "cluster_nodes_strict", strict)


def node(ip, slots=()):
    return {"id": ip, "addr": f"{ip}:6379", "flags": ["master"], "slots": list(slots)}


GOOD = {ip: [node("10.0.0.1"), node("10.0.0.2"), node("10.0.0.3")] for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3")}


def trip_reason():
    return open(common.trip_marker()).read() if os.path.exists(common.trip_marker()) else None


def test_topology_ok_on_every_pod(monkeypatch, tmp_path):
    topo(monkeypatch, tmp_path, GOOD)
    scale_to.verify_topology()
    assert trip_reason() is None


def test_foreign_node_seen_by_a_non_anchor_pod_records_a_trip(monkeypatch, tmp_path):
    views = dict(GOOD)
    views["10.0.0.3"] = [node("10.0.0.3"), node("10.244.37.144")]     # only the LAST pod sees the outsider
    topo(monkeypatch, tmp_path, views)
    with pytest.raises(RuntimeError, match="outside namespace"):
        scale_to.verify_topology()
    assert trip_reason() == "foreign-node"
    with pytest.raises(RuntimeError):
        common.require_no_trip()


@pytest.mark.parametrize("bad", [RuntimeError("CLUSTER NODES on 10.0.0.2 failed or was empty (rc=1)"), ])
def test_failed_or_empty_read_is_a_trip_not_clean(monkeypatch, tmp_path, bad):
    views = dict(GOOD)
    views["10.0.0.2"] = bad
    topo(monkeypatch, tmp_path, views)
    with pytest.raises(RuntimeError):
        scale_to.verify_topology()
    assert trip_reason() == "topology-unreadable"


def test_no_pods_to_inspect_is_a_trip(monkeypatch, tmp_path):
    topo(monkeypatch, tmp_path, {}, pods=())
    with pytest.raises(RuntimeError, match="no allowlisted pod"):
        scale_to.verify_topology()
    assert trip_reason() == "topology-unreadable"


def test_strict_reader_rejects_failed_and_empty_output(monkeypatch):
    monkeypatch.setattr(common, "rcli", lambda *a, **k: (1, "", "conn refused"))
    with pytest.raises(RuntimeError):
        common.cluster_nodes_strict("10.0.0.1")
    monkeypatch.setattr(common, "rcli", lambda *a, **k: (0, "\n", ""))
    with pytest.raises(RuntimeError, match="empty"):
        common.cluster_nodes_strict("10.0.0.1")


def test_pre_create_every_node_must_have_an_empty_allowlisted_view(monkeypatch, tmp_path):
    fresh = {ip: [node(ip)] for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3")}
    topo(monkeypatch, tmp_path, fresh)
    scale_to.verify_topology(expect_empty=True)
    stale = dict(fresh)
    stale["10.0.0.2"] = [node("10.0.0.2"), node("10.0.0.1")]          # already knows a peer: not a fresh node
    topo(monkeypatch, tmp_path, stale)
    with pytest.raises(RuntimeError, match="not a fresh"):
        scale_to.verify_topology(expect_empty=True)
    assert trip_reason() == "pre-create-view"
    os.remove(common.trip_marker())
    owner = dict(fresh)
    owner["10.0.0.1"] = [node("10.0.0.1", [(0, 100)])]
    topo(monkeypatch, tmp_path, owner)
    with pytest.raises(RuntimeError):
        scale_to.verify_topology(expect_empty=True)


def test_bootstrap_checks_the_empty_views_before_create(monkeypatch):
    order = []
    pods = [{"name": f"p{i}", "ip": f"10.0.0.{i}", "ready": True} for i in range(3)]
    monkeypatch.setattr(scale_to, "wait_ready", lambda n, dl: pods)
    monkeypatch.setattr(common, "cluster_nodes", lambda ip: [])
    monkeypatch.setattr(scale_to, "verify_topology", lambda expect_empty=False: order.append(("verify", expect_empty)))
    monkeypatch.setattr(scale_to, "tracked", lambda cmd, dl, timeout=120: (order.append(cmd[2]), (0, "", ""))[1])
    monkeypatch.setattr(scale_to, "wait_healthy", lambda *a, **k: None)
    monkeypatch.setattr(scale_to, "event", lambda *a, **k: None)
    scale_to.bootstrap(3, scale_to.Deadline(30))
    assert order[0] == ("verify", True) and order[1] == "create" and order[2] == ("verify", False)


def test_verify_topology_surrounds_every_cluster_mutation(monkeypatch):
    order = []
    monkeypatch.setattr(scale_to, "verify_topology", lambda expect_empty=False: order.append("verify"))
    monkeypatch.setattr(scale_to, "tracked", lambda cmd, dl, timeout=120: (order.append(cmd[2] if cmd[1] == "--cluster" else "x"), (0, "", ""))[1])
    monkeypatch.setattr(scale_to, "covered_now", lambda: len(order) > 6)
    monkeypatch.setattr(scale_to, "anchor_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(scale_to, "redis_pods", lambda: [{"name": "p", "ip": "10.0.0.1", "ready": True}])
    monkeypatch.setattr(common, "cluster_nodes", lambda ip: [])
    monkeypatch.setattr(common, "health", lambda n=None: {"ok": True})
    monkeypatch.setattr(common, "require_no_trip", lambda: None)
    monkeypatch.setattr(scale_to, "event", lambda *a, **k: None)
    monkeypatch.setattr(scale_to.time, "sleep", lambda s: None)
    scale_to.ensure_covered(scale_to.Deadline(30), 1)
    i = order.index("fix")
    assert order[i - 1] == "verify" and order[i + 1] == "verify"


# ---------------------------------------------------------------- memory watch never dies silently
def make_watch(monkeypatch, pods, expect, evaluate_raises=False):
    monkeypatch.setattr(common, "redis_pods", lambda: pods)
    monkeypatch.setattr(common, "check_pod", lambda p: {"pod": p["name"], "frac": 0.5, "aof_size": 0, "aof_rewrite": 0,
                                                       "bgsave": 0, "source": "cgroup"})
    if evaluate_raises:
        monkeypatch.setattr(common, "evaluate", lambda r, a: (_ for _ in ()).throw(ValueError("boom")))
    w = common.MemWatch(every=0.05)
    w.expect_pods = expect
    time.sleep(0.5)
    w.close()
    return w


def test_empty_pod_list_is_not_a_crash_and_trips_only_when_pods_are_expected(monkeypatch):
    assert make_watch(monkeypatch, [], expect=False).tripped is False
    w = make_watch(monkeypatch, [], expect=True)
    assert w.tripped and open(common.trip_marker()).read() == "pods-vanished"


def test_any_exception_in_the_watcher_records_a_trip(monkeypatch):
    w = make_watch(monkeypatch, [{"name": "p", "ip": "1.1.1.1", "uid": "u", "ready": True, "ready_since": 1.0}], expect=True, evaluate_raises=True)
    assert w.tripped and open(common.trip_marker()).read() == "watcher-error"


# ---------------------------------------------------------------- gate.py
import gate


def test_gate_decisions():
    assert gate.decide(None, False)[0]
    assert gate.decide(("done", 0), False)[0]
    assert not gate.decide(("failed", 1), False)[0]
    assert not gate.decide(("stopped", -15), False)[0]
    assert not gate.decide(("running", None), True)[0] and not gate.decide(("missing", None), False)[0]
    assert gate.decide(("failed", 1), True)[0]            # the rollback entry runs after a failure


def test_gate_reads_the_previous_entry_from_the_run_dirs(tmp_path):
    runs = tmp_path
    rid = "20261016-210857-R-20261016-018-e%d-a1"
    assert gate.previous_status(rid % 0, str(runs)) is None
    assert gate.previous_status(rid % 1, str(runs)) == ("missing", None)
    (runs / (rid % 0)).mkdir()
    assert gate.previous_status(rid % 1, str(runs)) == ("running", None)
    (runs / (rid % 0) / "launcher.json").write_text(json.dumps({"status": "failed", "exit_code": 2}))
    assert gate.previous_status(rid % 1, str(runs)) == ("failed", 2)


def test_gate_refuses_runs_rollback_and_never_starts_the_command(tmp_path):
    runs = tmp_path / "runs"
    prev = runs / "20261016-210857-R-20261016-018-e0-a1"
    prev.mkdir(parents=True)
    (prev / "launcher.json").write_text(json.dumps({"status": "failed", "exit_code": 1}))
    cur = runs / "20261016-211000-R-20261016-018-e1-a1"
    cur.mkdir()
    marker = tmp_path / "started"
    fake = tmp_path / "bin"
    fake.mkdir()
    k = fake / "kubectl"
    k.write_text('#!/usr/bin/env bash\ncase "$*" in *current-context*) echo kubernetes-admin@kubernetes;; '
                 '*"get namespace"*) echo "Error from server (NotFound)" >&2; exit 1;; esac\nexit 0\n')
    k.chmod(0o755)
    env = {**os.environ, "QRL_RUN_ID": cur.name, "QRL_RUN_DIR": str(cur), "PATH": f"{fake}:{os.environ['PATH']}"}
    r = subprocess.run([sys.executable, os.path.join(HERE, "gate.py"), "--", "touch", str(marker)], env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode != 0 and "running rollback.sh" in r.stdout
    assert not marker.exists()
    assert "namespace qrl-calib is gone" in (cur / "rollback_status.json").read_text()
    (prev / "launcher.json").write_text(json.dumps({"status": "done", "exit_code": 0}))
    r = subprocess.run([sys.executable, os.path.join(HERE, "gate.py"), "--", "touch", str(marker)], env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and marker.exists()


def test_every_spec_entry_goes_through_the_gate():
    for spec in ("calibration_spec.json", "transitions_spec.template.json", "spec_stageB.json", "spec_stageB_optional16.json"):
        sp = json.load(open(os.path.join(HERE, spec)))
        es = sp["entries"]
        for i, e in enumerate(es):
            assert e["cmd"][1] == "gate.py" and "--" in e["cmd"], e
        assert "--allow-failed" in es[-1]["cmd"] and "rollback.sh" in es[-1]["cmd"]
        assert all("--allow-failed" not in e["cmd"] for e in es[:-1])
        assert all("--health" in e["cmd"] for e in es if any(x in e["cmd"] for x in ("run_c.py", "run_transition.py")))


# ---------------------------------------------------------------- gate runs the entry as a child
def gate_env(tmp_path, prev_status="done", prev_code=0, kubectl_log=None):
    runs = tmp_path / "runs"
    prev = runs / "20261016-210857-R-20261016-018-e0-a1"
    prev.mkdir(parents=True, exist_ok=True)
    (prev / "launcher.json").write_text(json.dumps({"status": prev_status, "exit_code": prev_code}))
    cur = runs / "20261016-211000-R-20261016-018-e1-a1"
    cur.mkdir(exist_ok=True)
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    log = tmp_path / "rollback_called"
    k = fake / "kubectl"
    k.write_text(f'#!/usr/bin/env bash\ncase "$*" in *current-context*) echo kubernetes-admin@kubernetes;; '
                 f'*"delete namespace"*) touch {log};; *"get namespace"*) echo "Error from server (NotFound)" >&2; exit 1;; esac\nexit 0\n')
    k.chmod(0o755)
    return {**os.environ, "QRL_RUN_ID": cur.name, "QRL_RUN_DIR": str(cur), "PATH": f"{fake}:{os.environ['PATH']}"}, log


def run_gate(env, *args, timeout=60):
    return subprocess.run([sys.executable, os.path.join(HERE, "gate.py"), *args], env=env, capture_output=True, text=True, timeout=timeout)


def test_gate_failing_entry_triggers_rollback_and_propagates_the_exit_code(tmp_path):
    env, log = gate_env(tmp_path)
    env["QRL_LAST_ROLLBACK"] = str(tmp_path / "last.json")
    env["QRL_ROLLBACK_LOCK"] = str(tmp_path / "lock")
    r = run_gate(env, "--", "bash", "-c", "exit 7")
    assert r.returncode == 7 and log.exists() and "running rollback.sh" in r.stdout


def test_gate_successful_entry_does_not_roll_back(tmp_path):
    env, log = gate_env(tmp_path)
    r = run_gate(env, "--", "true")
    assert r.returncode == 0 and not log.exists()


def test_gate_health_exception_rolls_back_and_refuses(tmp_path):
    env, log = gate_env(tmp_path)
    env["PATH"] = str(tmp_path / "bin") + ":/usr/bin:/bin"        # fake kubectl answers `config`, but `get pods` json is garbage
    (tmp_path / "bin" / "kubectl").write_text(
        '#!/usr/bin/env bash\ncase "$*" in *current-context*) echo kubernetes-admin@kubernetes;; '
        f'*"delete namespace"*) touch {log};; *"get namespace"*) echo "NotFound" >&2; exit 1;; *"get pods"*) echo "not json";; esac\nexit 0\n')
    marker = tmp_path / "started"
    r = run_gate(env, "--health", "--", "touch", str(marker))
    assert r.returncode != 0 and not marker.exists() and log.exists()


def slow_kubectl(tmp_path, log, delay):
    """Fake kubectl whose `delete namespace` takes `delay` s before it succeeds (a rollback that is still running when the
    experiment's process group gets SIGKILL)."""
    k = tmp_path / "bin" / "kubectl"
    k.write_text('#!/usr/bin/env bash\ncase "$*" in *current-context*) echo kubernetes-admin@kubernetes;; '
                 f'*"delete namespace"*) sleep {delay}; touch {log};; *"get namespace"*) echo "Error from server (NotFound)" >&2; exit 1;; esac\nexit 0\n')
    k.chmod(0o755)


def test_detached_rollback_survives_sigterm_then_sigkill_of_the_process_group(tmp_path):
    env, log = gate_env(tmp_path)
    slow_kubectl(tmp_path, log, 3)
    env["QRL_LAST_ROLLBACK"] = str(tmp_path / "last.json")
    env["QRL_ROLLBACK_LOCK"] = str(tmp_path / "lock")
    # the gate (and its child) in their own session = "the experiment's process group", as launch_run.py creates it
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "gate.py"), "--", "sleep", "60"], env=env, stdout=subprocess.PIPE,
                         text=True, start_new_session=True)
    time.sleep(1.5)
    t = time.time()
    os.killpg(p.pid, 15)                       # launch_run: SIGTERM to the group on stop / wall-clock
    p.wait(timeout=15)
    assert time.time() - t < 8 and p.returncode != 0     # the gate exits quickly, it does not wait for the 3 s rollback
    time.sleep(1.0)
    try:
        os.killpg(p.pid, 9)                    # ... and SIGKILL of the group 1 s later
    except ProcessLookupError:
        pass
    assert not log.exists()                    # the rollback is still in flight at this point
    for _ in range(60):                        # the detached runner must still finish
        if log.exists() and os.path.exists(tmp_path / "last.json") and json.load(open(tmp_path / "last.json"))["status"] != "running":
            break
        time.sleep(0.25)
    st = json.load(open(tmp_path / "last.json"))
    assert log.exists() and st["status"] == "done" and st["exit_code"] == 0 and st["namespace_absent_proof"] is True
    assert json.load(open(os.path.join(env["QRL_RUN_DIR"], "rollback_status.json")))["status"] == "done"
    rep = subprocess.run([sys.executable, os.path.join(HERE, "rollback_status.py"), "--file", str(tmp_path / "last.json")],
                         capture_output=True, text=True)
    assert rep.returncode == 0 and "rollback done" in rep.stdout


def test_rollback_entry_waits_for_an_inflight_rollback_on_the_lock(tmp_path):
    env, log = gate_env(tmp_path)
    slow_kubectl(tmp_path, log, 3)
    env["QRL_ROLLBACK_LOCK"] = str(tmp_path / "lock")
    first = subprocess.Popen(["bash", os.path.join(HERE, "rollback.sh")], env=env, stdout=subprocess.DEVNULL)
    time.sleep(0.8)
    t = time.time()
    second = subprocess.run(["bash", os.path.join(HERE, "rollback.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert second.returncode == 0 and time.time() - t > 1.5      # it queued behind the first one
    assert first.wait(timeout=30) == 0


def test_rollback_status_reports_failure_and_absence(tmp_path):
    f = tmp_path / "x.json"
    assert subprocess.run([sys.executable, os.path.join(HERE, "rollback_status.py"), "--file", str(f)], capture_output=True).returncode == 1
    f.write_text(json.dumps({"status": "failed", "pid": 1, "start": 1, "end": 2, "exit_code": 1, "namespace_absent_proof": False,
                             "reason": "t", "output_tail": "boom"}))
    r = subprocess.run([sys.executable, os.path.join(HERE, "rollback_status.py"), "--file", str(f)], capture_output=True, text=True)
    assert r.returncode == 1 and "failed" in r.stdout


def test_gate_rollback_entry_runs_the_rollback_detached_and_reports_its_status(tmp_path):
    env, log = gate_env(tmp_path, prev_status="failed", prev_code=1)
    env["QRL_LAST_ROLLBACK"] = str(tmp_path / "last.json")
    env["QRL_ROLLBACK_LOCK"] = str(tmp_path / "lock")
    r = run_gate(env, "--allow-failed", "--", "bash", "rollback.sh")
    assert r.returncode == 0 and log.exists()
    st = json.load(open(tmp_path / "last.json"))
    assert st["status"] == "done" and st["exit_code"] == 0 and st["namespace_absent_proof"] is True
    assert json.load(open(os.path.join(env["QRL_RUN_DIR"], "rollback_status.json")))["status"] == "done"


# ---------------------------------------------------------------- run_c checks the generator exit code
def test_run_c_fails_the_level_on_nonzero_generator_exit(monkeypatch):
    import argparse
    import run_c

    class P:
        pid = 1
        def __init__(self, rc): self.rc = rc
        def wait(self): return self.rc
        def poll(self): return self.rc
    a = argparse.Namespace(seed=1, workers=2, c=1, timeout=10, keys=2000, value_size=64, max_outstanding=5000)
    monkeypatch.setattr(run_c.CHILDREN, "popen", lambda cmd: P(3))
    monkeypatch.setattr(run_c, "event", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="exited with 3"):
        run_c.lg("100:1", a, 0, 0)
    monkeypatch.setattr(run_c.CHILDREN, "popen", lambda cmd: P(0))
    assert run_c.lg("100:1", a, 0, 0) == 0
    tripped = type("M", (), {"tripped": True, "proc": None})()
    monkeypatch.setattr(run_c.CHILDREN, "popen", lambda cmd: P(-15))
    assert run_c.lg("100:1", a, 0, 0, mem=tripped) == -15      # a memory trip is the caller's recovery path, not a level failure


def test_sigterm_during_the_health_query_starts_the_detached_rollback(tmp_path):
    env, log = gate_env(tmp_path)
    k = tmp_path / "bin" / "kubectl"
    k.write_text('#!/usr/bin/env bash\ncase "$*" in *current-context*) echo kubernetes-admin@kubernetes;; '
                 f'*"delete namespace"*) touch {log};; *"get namespace"*) echo "NotFound" >&2; exit 1;; *"get pods"*) sleep 30;; esac\nexit 0\n')
    k.chmod(0o755)
    marker = tmp_path / "started"
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "gate.py"), "--health", "--", "touch", str(marker)], env=env,
                         start_new_session=True, stdout=subprocess.PIPE, text=True)
    time.sleep(2.0)                      # the gate is inside the (hanging) health query
    t = time.time()
    os.killpg(p.pid, 15)
    p.wait(timeout=15)
    assert time.time() - t < 5 and p.returncode != 0 and not marker.exists()
    for _ in range(60):
        if log.exists():
            break
        time.sleep(0.25)
    assert log.exists()                  # the detached rollback ran


def test_unresponsive_child_is_killed_as_a_group_and_reaped_before_gate_exits(tmp_path):
    env, log = gate_env(tmp_path)
    pidfile = tmp_path / "pids"
    # child ignores SIGTERM and has a grandchild that does too
    script = ("import os,signal,subprocess,sys,time\n"
              "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
              f"g=subprocess.Popen(['bash','-c','trap \"\" TERM; sleep 120'])\n"
              f"open({str(pidfile)!r},'w').write(str(os.getpid())+' '+str(g.pid))\n"
              "time.sleep(120)\n")
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "gate.py"), "--", sys.executable, "-c", script], env=env,
                         start_new_session=True, stdout=subprocess.PIPE, text=True)
    for _ in range(40):
        if pidfile.exists() and pidfile.read_text().count(" ") == 1:
            break
        time.sleep(0.25)
    child_pid, grand_pid = map(int, pidfile.read_text().split())
    t = time.time()
    os.kill(p.pid, 15)                   # SIGTERM to the gate only
    p.wait(timeout=40)
    took = time.time() - t
    assert 9 < took < 25                 # waited ~10 s, then killed the group
    for pid in (child_pid, grand_pid):   # gate never exits while the experiment child (or its group) is alive
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            raise AssertionError(f"process {pid} survived the gate")
    for _ in range(60):
        if log.exists():
            break
        time.sleep(0.25)
    assert log.exists() and p.returncode != 0    # and the detached rollback completed


def test_grandchild_is_killed_even_though_the_leader_exited_on_sigterm(tmp_path):
    env, log = gate_env(tmp_path)
    pidfile = tmp_path / "pids2"
    script = ("import os,subprocess,sys,time\n"
              "g=subprocess.Popen(['bash','-c','trap \"\" TERM; sleep 120'])\n"
              f"open({str(pidfile)!r},'w').write(str(g.pid))\n"
              "time.sleep(120)\n")          # leader has the default SIGTERM action: it dies at once, the grandchild lives on
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "gate.py"), "--", sys.executable, "-c", script], env=env,
                         start_new_session=True, stdout=subprocess.PIPE, text=True)
    for _ in range(40):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.25)
    grand = int(pidfile.read_text())
    os.kill(p.pid, 15)                       # SIGTERM to the gate only
    p.wait(timeout=40)
    try:
        os.kill(grand, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    if alive:                                # a zombie awaiting init's reaping does not count as running
        alive = "Z" not in open(f"/proc/{grand}/stat").read().rsplit(")", 1)[1].split()[0]
    assert not alive                         # killed BEFORE the gate exited
    for _ in range(60):
        if log.exists():
            break
        time.sleep(0.25)
    assert log.exists() and p.returncode != 0


def test_normal_exit_also_cleans_leftover_group_members(tmp_path):
    env, log = gate_env(tmp_path)
    pidfile = tmp_path / "pids3"
    script = ("import subprocess\n"
              "g=subprocess.Popen(['bash','-c','trap \"\" TERM; sleep 120'])\n"
              f"open({str(pidfile)!r},'w').write(str(g.pid))\n")      # leader exits 0 at once, grandchild stays
    r = subprocess.run([sys.executable, os.path.join(HERE, "gate.py"), "--", sys.executable, "-c", script], env=env,
                       capture_output=True, text=True, timeout=60)
    grand = int(pidfile.read_text())
    assert r.returncode == 0
    try:
        os.kill(grand, 0)
        state = open(f"/proc/{grand}/stat").read().rsplit(")", 1)[1].split()[0]
    except (ProcessLookupError, FileNotFoundError):
        state = "Z"
    assert state == "Z"


# ---------------------------------------------------------------- one cleanup path (reviews 8)
def counting_kubectl(tmp_path, log):
    k = tmp_path / "bin" / "kubectl"
    k.write_text('#!/usr/bin/env bash\ncase "$*" in *current-context*) echo kubernetes-admin@kubernetes;; '
                 f'*"delete namespace"*) echo x >> {log};; *"get namespace"*) echo "Error from server (NotFound)" >&2; exit 1;; esac\nexit 0\n')
    k.chmod(0o755)


def alive(pid):
    try:
        os.kill(pid, 0)
        return "Z" not in open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except (ProcessLookupError, FileNotFoundError):
        return False


def test_sigterm_after_the_leader_exited_does_not_skip_cleanup_of_a_surviving_descendant(tmp_path):
    env, log = gate_env(tmp_path)
    counting_kubectl(tmp_path, log)
    pidfile = tmp_path / "pids4"
    # the leader exits 0 at once; the descendant ignores SIGTERM and stays. The gate is then inside reap_group (10 s grace).
    script = ("import subprocess\ng=subprocess.Popen(['bash','-c','trap \"\" TERM; sleep 120'])\n"
              f"open({str(pidfile)!r},'w').write(str(g.pid))\n")
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "gate.py"), "--", sys.executable, "-c", script], env=env,
                         start_new_session=True, stdout=subprocess.PIPE, text=True)
    for _ in range(40):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.1)
    grand = int(pidfile.read_text())
    time.sleep(2.0)
    t = time.time()
    os.kill(p.pid, 15)                      # SIGTERM arrives AFTER the leader exited, descendant alive
    assert alive(grand)
    time.sleep(1.0)
    os.kill(p.pid, 15)                      # a second SIGTERM during reap_group
    os.kill(p.pid, 2)                       # and a SIGINT
    assert p.poll() is None                 # the gate did not leave while the descendant lives (old fast path would have)
    p.wait(timeout=40)
    assert not alive(grand)                 # killed before the gate exited
    assert p.returncode != 0
    for _ in range(40):
        if log.exists():
            break
        time.sleep(0.25)
    assert log.read_text().count("x") == 1  # the detached rollback was started exactly once


def test_reap_group_is_idempotent(monkeypatch):
    import gate
    calls = []
    monkeypatch.setattr(gate, "_reap_group", lambda pgid, leader, grace: calls.append(pgid) or True)
    gate._REAP.pop(12345, None)
    assert gate.reap_group(12345, None) is True and gate.reap_group(12345, None) is True
    assert calls == [12345]


def test_unkillable_member_fails_the_gate_writes_the_marker_and_blocks_the_next_gate(tmp_path, monkeypatch):
    import gate
    runs = tmp_path / "runs"
    cur = runs / "20261016-211000-R-20261016-018-e1-a1"
    prev = runs / "20261016-210857-R-20261016-018-e0-a1"
    for d in (cur, prev):
        d.mkdir(parents=True)
    (prev / "launcher.json").write_text(json.dumps({"status": "done", "exit_code": 0}))
    monkeypatch.setenv("QRL_RUN_ID", cur.name)
    monkeypatch.setenv("QRL_RUN_DIR", str(cur))
    rollbacks = []
    monkeypatch.setattr(gate, "reap_group", lambda pgid, leader, grace=10.0: False)      # members survive even SIGKILL
    monkeypatch.setattr(gate, "start_rollback", lambda reason, wait=True, timeout=450: rollbacks.append(reason))
    monkeypatch.setattr(sys, "argv", ["gate.py", "--", "true"])
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(SystemExit) as e:
            gate.main()
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    assert e.value.code not in (0, None)                  # non-zero even though the entry itself exited 0
    assert (cur / "cleanup_failed").exists() and rollbacks == ["cleanup_failed"]
    # the next entry's gate refuses (and would roll back)
    assert gate.previous_status("20261016-211500-R-20261016-018-e2-a1", str(runs)) == ("cleanup_failed", None)
    assert not gate.decide(("cleanup_failed", None), False)[0]
    assert gate.decide(("cleanup_failed", None), True)[0]          # only the rollback entry may follow


def test_reap_group_reports_false_when_a_member_cannot_be_killed(monkeypatch):
    import gate
    monkeypatch.setattr(gate.os, "killpg", lambda pgid, sig: None)           # nothing ever dies
    monkeypatch.setattr(gate, "group_alive", lambda pgid: True)
    monkeypatch.setattr(gate.time, "sleep", lambda s: None)
    t = [0.0]
    monkeypatch.setattr(gate.time, "time", lambda: t.__setitem__(0, t[0] + 1.0) or t[0])
    leader = type("L", (), {"poll": lambda s: 0, "wait": lambda s: 0})()
    gate._REAP.pop(777, None)
    assert gate.reap_group(777, leader, grace=3.0) is False


# ---------------------------------------------------------------- fresh node: `myself` with an empty IP (first real run)
FRESH = "f8b6777bd496e259bd5b661392ca20d4dfa962cd :6379@16379 myself,master - 0 0 0 connected\n"   # real output of a fresh node


def test_fresh_node_myself_line_with_empty_ip_is_the_queried_pod(monkeypatch):
    nodes = common.parse_cluster_nodes(FRESH)
    assert nodes[0]["addr"] == ":6379" and nodes[0]["flags"] == ["myself", "master"] and nodes[0]["slots"] == []
    monkeypatch.setattr(common, "rcli", lambda ip, *a, **k: (0, FRESH, ""))
    n = common.cluster_nodes_strict("10.244.5.7")
    assert n[0]["addr"] == "10.244.5.7:6379" and common.cluster_nodes("10.244.5.7")[0]["addr"] == "10.244.5.7:6379"


def test_pre_create_check_accepts_fresh_nodes_and_still_rejects_other_empty_addresses(monkeypatch, tmp_path):
    ips = ("10.0.0.1", "10.0.0.2", "10.0.0.3")
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(scale_to, "redis_pods", lambda: [{"name": f"p{i}", "ip": ip, "ready": True} for i, ip in enumerate(ips)])
    monkeypatch.setattr(common, "allowed_ips", lambda force=False: frozenset(ips))
    monkeypatch.setattr(common, "rcli", lambda ip, *a, **k: (0, FRESH, ""))
    scale_to.verify_topology(expect_empty=True)                 # three fresh nodes: passes
    assert not os.path.exists(common.trip_marker())
    # a NON-myself line with an empty address, or garbage, is still a trip
    for bad in ("aaa :6379@16379 master - 0 0 1 connected\n", "aaa garbage master - 0 0 1 connected\n"):
        monkeypatch.setattr(common, "rcli", lambda ip, *a, bad=bad, **k: (0, FRESH + bad, ""))
        with pytest.raises(RuntimeError):
            scale_to.verify_topology()
        assert open(common.trip_marker()).read() == "foreign-node"
        os.remove(common.trip_marker())


def test_after_create_and_add_node_views_have_real_ips(monkeypatch, tmp_path):
    """Post-create / add-node checks: a formed node reports real IPs on its myself line; a joining node that is still blank is the
    queried pod itself and must not be seen as foreign or forgotten."""
    ips = ("10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4")
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(scale_to, "redis_pods", lambda: [{"name": f"p{i}", "ip": ip, "ready": True} for i, ip in enumerate(ips)])
    monkeypatch.setattr(common, "allowed_ips", lambda force=False: frozenset(ips))
    formed = "".join(f"id{i} {ip}:6379@16379 {'myself,' if ip == '10.0.0.1' else ''}master - 0 0 {i} connected {i*100}-{i*100+99}\n"
                     for i, ip in enumerate(ips[:3]))
    fresh_new = "idnew :6379@16379 myself,master - 0 0 0 connected\n"      # 4th pod before it joined
    monkeypatch.setattr(common, "rcli", lambda ip, *a, **k: (0, fresh_new if ip == "10.0.0.4" else formed, ""))
    scale_to.verify_topology()                                    # formed cluster + one blank joining pod: no trip
    # ensure_covered must not forget the blank myself node: it maps to the queried (allowlisted) pod IP
    nodes = common.cluster_nodes("10.0.0.4")
    assert nodes[0]["addr"].rsplit(":", 1)[0] in ips


# ---------------------------------------------------------------- memory watch: only Ready pods, with a grace period
def test_watchable_skips_not_ready_and_grace_pods():
    st = {}
    pods = [{"name": "old", "ready": True, "ready_since": 1000.0}, {"name": "new", "ready": True, "ready_since": 1990.0},
            {"name": "starting", "ready": False, "ready_since": None}, {"name": "noinfo", "ready": True, "ready_since": None}]
    got = [p["name"] for p in common.watchable(pods, st, now=2000.0)]
    assert got == ["old"]                       # new: Ready 10 s < 15 s grace; noinfo: first seen now; starting: not Ready
    got = [p["name"] for p in common.watchable(pods, st, now=2000.0 + common.READY_GRACE_S)]
    assert got == ["old", "new", "noinfo"]      # after grace, Ready pods are required; the not-Ready one never is


def test_not_ready_pod_does_not_trip_but_unreadable_ready_pod_after_grace_does(monkeypatch):
    def check_pod(p):
        if p["name"] in ("starting", "bad"):
            raise RuntimeError("cannot read memory (no cgroup, no kubectl top)")
        return {"pod": p["name"], "frac": 0.5, "aof_size": 0, "aof_rewrite": 0, "bgsave": 0, "source": "cgroup"}
    now = time.time()
    ok = {"name": "ok", "ip": "1", "uid": "u1", "ready": True, "ready_since": now - 100}
    starting = {"name": "starting", "ip": "2", "uid": "u2", "ready": False, "ready_since": None}
    bad = {"name": "bad", "ip": "3", "uid": "u3", "ready": True, "ready_since": now - 100}
    monkeypatch.setattr(common, "check_pod", check_pod)
    w = common.MemWatch.__new__(common.MemWatch)
    from concurrent.futures import ThreadPoolExecutor
    w.pool = ThreadPoolExecutor(4)
    monkeypatch.setattr(common, "redis_pods", lambda: [ok, starting])
    assert [r["pod"] for r in w._step({"pods": [ok, starting], "t_pods": time.time()})] == ["ok"]
    monkeypatch.setattr(common, "redis_pods", lambda: [ok, bad])
    with pytest.raises(Exception):
        w._step({"pods": [ok, bad], "t_pods": time.time()})
    # whole watch: only a not-Ready pod exists while pods are expected -> no trip (nothing to read yet)
    pods = [starting]
    assert make_watch(monkeypatch, pods, expect=True).tripped is False
