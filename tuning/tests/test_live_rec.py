"""Opt-in dashboard observers (--live-file, --record-traj; tuning/legacy/live_rec.py): ring buffer, atomic write,
CLI plumbing, and bit-identity of the result with the observers on vs off (tiny run)."""
import contextlib
import json
import os

import numpy as np
import pytest

from tuning.legacy import c5_protocol as C5
from tuning.legacy import live_rec as LR
from tuning.legacy import run_legacy as RL


def test_ring_buffer_and_atomic_write(tmp_path):
    clock = [0.0]
    p = str(tmp_path / "live" / "x.json")
    r = LR.LiveRecorder(p, {"arm": "derp"}, every=25, keep=600, min_interval=0.5, clock=lambda: clock[0])
    for i in range(1000):
        clock[0] += 0.1
        r.add("train", [50.0 + i, 0, 0, 0, 0, 3.0, 2.0], i % 3, 1.5)
    d = json.load(open(p))
    assert d["arm"] == "derp" and d["cols"] == list(LR.COLS)
    assert len(d["rows"]) == 600 and d["rows"][-1][0] == d["t"] and d["t"] % 25 == 0
    assert d["rows"][-1][4] == 2 and d["rows"][-1][1] == "train" and not d["done"]
    assert not [f for f in os.listdir(tmp_path / "live") if ".tmp" in f]
    r.flush(done=True)
    d = json.load(open(p))
    assert d["done"] and d["t"] == 1000 and d["rows"][0][0] == 401


def test_throttle_limits_writes(tmp_path):
    n = [0]
    r = LR.LiveRecorder(str(tmp_path / "y.json"), every=1, min_interval=10.0, clock=lambda: 5.0)
    orig = LR.atomic_write_json
    try:
        LR.atomic_write_json = lambda path, obj: n.__setitem__(0, n[0] + 1)
        for _ in range(100):
            r.add("eval", np.zeros(6), 1, 0.0)
    finally:
        LR.atomic_write_json = orig
    assert n[0] == 0      # 5.0 - last_write 0.0 < min_interval 10: no write


def test_traj_recorder():
    t = LR.TrajRecorder()
    t.add([10.0, 1, 40.0, 50.0, 0.7, 2.0, 1.0], 2, -3.25)
    t.add([11.0, 1, 41.0, 51.0, 0.8, 3.0], 1, 4.0)
    assert t.d["load"] == [10.0, 11.0] and t.d["pending"] == [1, 0] and t.d["action"] == [2, 1]
    assert set(t.d) == set(LR.TRAJ_KEYS)


def _tiny(tmp_path, tag, **kw):
    out = str(tmp_path / tag / "r.json")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        C5.run_c5("derp", 7, total_ticks=40, checkpoints=[20, 40], eval_splits="val", out=out, eval_ticks=30,
                  reward_div=10.0, obs_scale="on", eps_schedule="linear10k", env_family="orig",
                  orig_variant="spike300_d5", **kw)
    return json.load(open(out))


def _strip(o, new=False):
    if isinstance(o, dict):
        return {k: _strip(v, new) for k, v in o.items() if not ("cpu_s" in k or "wall_s" in k or "per_1k" in k)
                and not (new and k in ("live_file", "record_traj", "trajectory"))}
    if isinstance(o, list):
        return [_strip(v, new) for v in o]
    return o


def test_observers_do_not_change_numbers(tmp_path):
    off = _tiny(tmp_path, "off")
    on = _tiny(tmp_path, "on", live_file=str(tmp_path / "live" / "l.json"), record_traj=2)
    assert "live_file" not in off and "record_traj" not in off
    assert json.dumps(_strip(off), sort_keys=True) == json.dumps(_strip(on, True), sort_keys=True)
    ck = on["checkpoint_results"]["40"]["val"]
    tr = ck["trajectory"]["episodes"]
    assert len(tr) == 2 and len(tr[0]["load"]) == 30 and len(on["checkpoint_results"]["20"]["val"]["trajectory"]["episodes"]) == 2
    assert abs(sum(tr[0]["reward"]) - ck["episodes"][0]["ret"]) < 0.05 * 30
    assert np.isclose(np.mean(tr[1]["pods"]), ck["episodes"][1]["mean_pods"], atol=0.01)
    live = json.load(open(tmp_path / "live" / "l.json"))
    assert live["done"] and live["t"] == 40 + 2 * 5 * 30 and live["rows"][-1][1] == "eval"


def test_cli_plumbing(monkeypatch):
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        return {"arm": "derp", "cpu_s": 0.0, "train_cpu_s_per_1k_ticks": 0.0, "checkpoint_results": {}, "eval_splits": []}
    monkeypatch.setattr(C5, "run_c5", fake)
    base = ["--agent", "derp", "--total-ticks", "100", "--seed", "7", "--env-family", "orig", "--orig-variant", "orig"]
    RL.main(base)
    assert "live_file" not in seen and "record_traj" not in seen
    seen.clear()
    RL.main(base + ["--live-file", "/tmp/x.json", "--record-traj", "3"])
    assert seen["live_file"] == "/tmp/x.json" and seen["record_traj"] == 3
    with pytest.raises(SystemExit):
        RL.main(["--agent", "derp", "--train-ticks", "1000", "--live-file", "/tmp/x.json"])
