import json
import os

import check_sweep


def _w(path, name, recs):
    with open(os.path.join(path, name), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_check_flags_unbracketed_sweep_and_reports_counts(tmp_path):
    rows, tele = [], []
    for lv, (off, srv) in enumerate([(100, 100), (200, 200), (300, 290)]):
        for i in range(2):
            t = 1000.0 + lv * 10 + i
            rows.append({"t": t, "phase": 1, "level": lv, "warm": 0, "pods": 1, "offered_ops": off, "completed_ops": srv,
                         "served_ops": srv, "error_ops": 0, "timeout_ops": 0, "drop_ops": 0, "generator_lag_ms": 1,
                         "target_rate": off, "p99_ms": 1.0, "hist": {"70": 5}, "workers": 1})
            tele.append({"t": t + 0.5, "ncpu": 40, "host_busy_cores": 2.0, "gen_worker_max_cores": 0.1, "n_pods": 1,
                         "pods_cores": 0.19})
    _w(tmp_path, "metrics.jsonl", rows)
    _w(tmp_path, "telemetry.jsonl", tele)
    _w(tmp_path, "events.jsonl", [{"t": 1.0, "event": "sweep_done"}])
    r = check_sweep.check(str(tmp_path))
    assert r["metric_rows"] == 6 and r["valid_sweep_levels"] == "3/3" and len(r["sweep"]) == 3
    assert r["capacity"] is None and any("saturated" in p for p in r["problems"])   # no saturated level -> not usable


def test_check_reports_abort_events(tmp_path):
    _w(tmp_path, "events.jsonl", [{"t": 1.0, "event": "memory_abort", "why": "unreadable"}])
    r = check_sweep.check(str(tmp_path))
    assert "event memory_abort" in r["problems"] and "no phase-1 sweep levels" in r["problems"]
