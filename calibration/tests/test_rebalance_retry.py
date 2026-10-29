"""Rebalance diagnostics, convergence wait and retry (mocked subprocess/cluster; nothing touches a cluster)."""
import os

import pytest

import scale_to


class FakeProc:
    def __init__(self, rc, text, out_file):
        self.returncode = rc
        out_file.write(text)
        out_file.flush()

    def poll(self):
        return self.returncode


@pytest.fixture
def env(monkeypatch):
    ev = []
    monkeypatch.setattr(scale_to, "event", lambda name, **kw: ev.append((name, kw)))
    monkeypatch.setattr(scale_to.time, "sleep", lambda s: None)
    monkeypatch.setattr(scale_to, "anchor_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(scale_to, "verify_topology", lambda **k: None)
    monkeypatch.setattr(scale_to.common, "check_endpoints", lambda c: None)
    monkeypatch.setattr(scale_to.common, "require_no_trip", lambda: None)
    return ev


def script_popen(monkeypatch, results):
    calls = []

    def popen(cmd, stdout=None, stderr=None, text=None):
        calls.append(cmd)
        rc, txt = results[len(calls) - 1]
        return FakeProc(rc, txt, stdout)
    monkeypatch.setattr(scale_to.CHILDREN, "popen", popen)
    return calls


def test_failure_output_is_logged_and_reported(env, monkeypatch):
    calls = script_popen(monkeypatch, [(1, "[ERR] Something else broke\n")])
    with pytest.raises(RuntimeError, match="Something else broke"):
        scale_to.rebalance_logged(scale_to.Deadline(30))
    assert len(calls) == 1   # not a convergence error: no retry
    log = open(os.path.join(os.environ["QRL_RUN_DIR"], "redis-cli-rebalance.log")).read()
    assert "Something else broke" in log and "rc=1" in log


def test_convergence_error_is_retried_then_succeeds(env, monkeypatch):
    calls = script_popen(monkeypatch, [(1, "[ERR] Nodes don't agree about configuration!\n"), (0, "ok\n")])
    monkeypatch.setattr(scale_to, "wait_converged", lambda dl, n: env.append(("wc", n)) or True)
    scale_to.rebalance_logged(scale_to.Deadline(30), n_nodes=2)
    assert len(calls) == 2
    names = [e[0] for e in env]
    assert names == ["rebalance_retry", "wc"]


def test_gives_up_after_three_attempts(env, monkeypatch):
    calls = script_popen(monkeypatch, [(1, "Nodes don't agree about configuration")] * 3)
    monkeypatch.setattr(scale_to, "wait_converged", lambda dl, n: True)
    with pytest.raises(RuntimeError, match="rebalance failed"):
        scale_to.rebalance_logged(scale_to.Deadline(30), n_nodes=2)
    assert len(calls) == 3
    assert [e[0] for e in env].count("rebalance_retry") == 2


def test_wait_converged_waits_until_views_agree(env, monkeypatch):
    pods = [{"ip": "10.0.0.1", "ready": True}, {"ip": "10.0.0.2", "ready": True}]
    monkeypatch.setattr(scale_to, "redis_pods", lambda: pods)
    state = {"n": 0}

    def nodes(ip):
        state["n"] += 1
        if state["n"] <= 2:   # first poll: second pod still knows only itself
            return [{"id": "a"}, {"id": "b"}] if ip == "10.0.0.1" else [{"id": "b"}]
        return [{"id": "a"}, {"id": "b"}]
    monkeypatch.setattr(scale_to.common, "cluster_nodes", nodes)
    monkeypatch.setattr(scale_to, "tracked", lambda cmd, dl, t=120: (0, "[OK] All nodes agree about slots configuration.", ""))
    assert scale_to.wait_converged(scale_to.Deadline(30), 2) is True
    name, kw = env[-1]
    assert name == "converge_wait" and kw["ok"] is True


def test_wait_converged_is_bounded(env, monkeypatch):
    monkeypatch.setattr(scale_to, "redis_pods", lambda: [{"ip": "10.0.0.1", "ready": True}])
    monkeypatch.setattr(scale_to.common, "cluster_nodes", lambda ip: [{"id": "a"}])
    t = {"v": 1000.0}
    monkeypatch.setattr(scale_to.time, "time", lambda: t.__setitem__("v", t["v"] + 5) or t["v"])
    assert scale_to.wait_converged(scale_to.Deadline(10 ** 6), 2, max_wait=20) is False
    assert env[-1][1]["ok"] is False
