"""fit.py on synthetic data with known truth. No cluster, no real runs."""
import json
import math
import random

import fit
import loadgen

MU, THETA = 4000.0, 0.9
FRACS = [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5]


def cap(c):
    return MU * c ** THETA


def hist_at(ms, n):
    return {str(loadgen.bucket(ms / 1e3)): n}


def write_sweep(d, cs, rng, noise=0.01, telemetry=True, fracs=FRACS, gen_cores=2.0, host_busy=4.0, lag=2.0, scale=1.0,
                busy=0.95, worker_max=None, drop_share=0.0):
    d.mkdir(parents=True, exist_ok=True)
    rows, tele, t = [], [], 1.0e9
    for c in cs:
        C = cap(c) * scale
        for lv, f in enumerate(fracs):
            off = C * f
            comp = min(off, C)
            p95 = 0.4 + 1e3 * math.log(20) * c / C / max(1 - f, 0.02) if f < 1 else 800.0
            for w in range(40):
                n_ok = int(comp * (1 + rng.gauss(0, noise)))
                to = max(0.0, off - comp) * (1 - drop_share if f >= 1 else 1)
                dr = max(0.0, off - comp) * drop_share if f >= 1 else 0.0
                rows.append({"step": len(rows), "t": t, "phase": 1, "level": lv, "target_rate": off, "pods": c,
                             "warm": 1 if w < 10 else 0, "workers": 8, "offered_ops": off * (1 + rng.gauss(0, noise)),
                             "completed_ops": n_ok, "served_ops": n_ok, "error_ops": 0.0, "timeout_ops": to, "drop_ops": dr,
                             "failed_ops": to, "p50_ms": p95 / 3, "p95_ms": p95, "p99_ms": p95 * 1.5,
                             "generator_lag_ms": lag, "hist": hist_at(p95 * 0.97, n_ok) if f < 1 else hist_at(50, n_ok)})
                if telemetry:
                    tele.append({"t": t + 0.5, "ncpu": 40, "host_busy_cores": host_busy, "gen_cores": gen_cores,
                                 "gen_worker_max_cores": worker_max if worker_max is not None else gen_cores / 8,
                                 "n_pods": c, "pods_cores": 0.2 * c * busy})
                t += 1
    with open(d / "metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(d / "telemetry.jsonl", "w") as f:
        for r in tele:
            f.write(json.dumps(r) + "\n")


def two_sweeps(tmp_path, cs, **kw):
    ds = []
    for i, scale in enumerate((1.0, 1.01)):
        d = tmp_path / f"s{i}"
        write_sweep(d, cs, random.Random(i), scale=scale, **kw)
        ds.append(str(d))
    return ds


def test_recovers_mu_theta_and_latency_from_repeated_sweeps(tmp_path):
    res = fit.fit_all(two_sweeps(tmp_path, [1, 2, 4, 8, 16, 30]))
    p = res["power"]
    assert abs(p["mu"] - MU * 1.005) / MU < 0.06 and abs(p["theta"] - THETA) < 0.02 and p["n_points"] == 6
    lat = res["latency"]["4"]
    assert abs(lat["b_ratio"] - 1.0) < 0.15 and abs(lat["a_ms"] - 0.4) < 0.7
    assert not res["rejected_capacity"]


def test_single_sweep_is_not_enough(tmp_path):
    write_sweep(tmp_path / "a", [1, 2, 4], random.Random(0))
    res = fit.fit_all([str(tmp_path / "a")])
    assert res["capacity_per_c"] == {} and res["power"] is None
    assert any(">= 2 required" in r["reason"] for r in res["rejected_capacity"])


def test_disagreeing_repeats_are_rejected(tmp_path):
    ds = []
    for i, scale in enumerate((1.0, 1.5)):
        write_sweep(tmp_path / f"s{i}", [1, 2, 4], random.Random(i), scale=scale)
        ds.append(str(tmp_path / f"s{i}"))
    res = fit.fit_all(ds)
    assert res["capacity_per_c"] == {} and any("disagree" in r["reason"] for r in res["rejected_capacity"])


def test_no_saturation_bracket_means_no_capacity(tmp_path):
    ds = two_sweeps(tmp_path, [1, 2, 4], fracs=[0.3, 0.5, 0.7, 0.85, 0.95])
    res = fit.fit_all(ds)
    assert res["capacity_per_c"] == {}
    assert any("bracket" in r["reason"] or "saturated" in r["reason"] for r in res["rejected_capacity"])


def test_generator_cpu_host_interference_and_missing_telemetry_invalidate(tmp_path):
    for kw, why in (({"gen_cores": 7.5}, "generator-cpu-bound"), ({"host_busy": 30.0}, "host-interference"),
                    ({"telemetry": False}, "no-cpu-telemetry"), ({"lag": 500.0}, "generator-lag")):
        ds = two_sweeps(tmp_path / why, [1, 2, 4], **kw)
        res = fit.fit_all(ds)
        assert res["capacity_per_c"] == {}, why
        assert any(why in x["reasons"] for x in res["invalid_levels"]), why


def test_plateau_from_client_drops_with_idle_pods_is_not_capacity(tmp_path):
    ds = two_sweeps(tmp_path, [1, 2, 4], busy=0.3, drop_share=0.9)   # pods mostly idle, unserved load is client drops
    res = fit.fit_all(ds)
    assert res["capacity_per_c"] == {}
    assert any("busy pods" in r["reason"] for r in res["rejected_capacity"])


def test_no_clean_subsaturation_level_means_no_capacity(tmp_path):
    ds = two_sweeps(tmp_path, [1, 2, 4], fracs=[1.05, 1.2, 1.5])   # no sub-knee level at all
    assert fit.fit_all(ds)["capacity_per_c"] == {}


def test_one_cpu_bound_worker_invalidates_even_if_average_is_low(tmp_path):
    ds = two_sweeps(tmp_path, [1, 2, 4], gen_cores=2.0, worker_max=0.95)   # total 2 cores over 8 workers, one worker pegged
    res = fit.fit_all(ds)
    assert res["capacity_per_c"] == {} and any("generator-cpu-bound" in x["reasons"] for x in res["invalid_levels"])


def test_histograms_are_merged_not_averaged():
    h = [0] * loadgen.NB
    h[loadgen.bucket(0.001)] += 50
    h[loadgen.bucket(0.100)] += 50
    p95 = fit.quantile_censored(h, 0, 0.95)
    assert abs(p95 - 0.100) / 0.100 < 0.06          # mean of per-window p95 would be ~0.0505
    assert fit.quantile_censored(h, 0, 0.25) < 0.0015


def test_timeouts_are_right_censored():
    h = [0] * loadgen.NB
    h[loadgen.bucket(0.002)] = 90
    assert math.isfinite(fit.quantile_censored(h, 10, 0.5))
    assert fit.quantile_censored(h, 10, 0.95) == float("inf")   # p95 lies in the censored 10 %
    assert fit.quantile_censored(h, 0, 0.95) < 0.003


# ---------------------------------------------------------------- transitions
C_LO, C_HI, DELTA_UP = 8000.0, 12000.0, 0.25
DOWN_PROFILE = [(10, 0.0), (20, 0.6)]   # 10 s with slots uncovered (cluster down), then 20 s at 60 % while rebalancing


def write_transitions(d, saturated=True, reps=2, telemetry=True, worker_max=0.3):
    d.mkdir(parents=True, exist_ok=True)
    rows, ev, tele, t = [], [], [], 2.0e9

    def seg(n, comp, pods=3):
        nonlocal t
        for _ in range(n):
            off = 1.15 * C_HI if saturated else 0.9 * comp
            rows.append({"step": len(rows), "t": t, "phase": 2, "level": 0, "target_rate": off, "pods": 2, "warm": 0,
                         "offered_ops": off, "completed_ops": comp, "served_ops": comp, "error_ops": 0, "timeout_ops": 0,
                         "drop_ops": 0, "failed_ops": 0, "p50_ms": 1, "p95_ms": 2, "p99_ms": 3, "generator_lag_ms": 1,
                         "workers": 8})
            if telemetry:
                tele.append({"t": t + 0.5, "ncpu": 40, "host_busy_cores": 4.0, "gen_cores": 2.0, "gen_worker_max_cores": worker_max,
                             "n_pods": pods, "pods_cores": 0.2 * pods * 0.95})
            t += 1

    for rep in range(reps):
        seg(30, C_LO)                                    # pre-window at lo
        t_cmd = t
        seg(20, C_LO)                                    # pod start-up + add-node: nominal lo capacity
        t_first = t
        ev += [{"t": t_cmd, "event": "scale_cmd", "direction": "up"}, {"t": t_cmd + 15, "event": "pod_ready"},
               {"t": t_cmd + 18, "event": "add_node"}, {"t": t_first, "event": "first_slot"},
               {"t": t_first, "event": "rebalance_start"}]
        seg(30, C_HI * (1 - DELTA_UP))                   # rebalance: 30 s -> k = 2
        ev += [{"t": t, "event": "rebalance_end"}, {"t": t + 2, "event": "healthy"}]
        seg(45, C_HI)                                    # settle at hi
        t_cmd2 = t
        ev += [{"t": t_cmd2, "event": "scale_cmd", "direction": "down"},
               {"t": t_cmd2 + 1, "event": "pod_removed", "pod": "p0", "slots_lost": 5461, "keys_lost": 5, "was_newest": False},
               {"t": t_cmd2 + 2, "event": "fix_start"}, {"t": t_cmd2 + 10, "event": "covered"}]
        for n, frac in DOWN_PROFILE:
            seg(n, C_LO * frac)
        ev += [{"t": t, "event": "rebalance_end"}, {"t": t + 2, "event": "healthy"}]
        seg(45, C_LO)                                    # settle at lo
    for name, rs in (("metrics.jsonl", rows), ("telemetry.jsonl", tele), ("events.jsonl", ev)):
        with open(d / name, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")


def test_capacity_relative_delta_k_d_up_and_down(tmp_path):
    write_transitions(tmp_path)
    res = fit.fit_all([str(tmp_path)])
    dk = res["delta_k"]
    assert dk["n_up"] == 2 and dk["n_down"] == 2
    assert abs(dk["delta_up_median"] - DELTA_UP) < 0.02
    assert abs(dk["delta_down_median"] - 0.6) < 0.03                       # 1 - (0*10 + 0.6*20)/30 = 0.6
    rec = dk["down_recovery"]
    assert abs(rec["covered_s_median"] - 10) < 1e-6 and abs(rec["restored_s_median"] - 30) < 1.5
    assert rec["slots_lost_median"] == 5461 and rec["keys_lost_median"] == 5 and rec["share_newest_removed"] == 0
    assert dk["k_steps_median"] == 2 and dk["loss_pre_median"] < 0.01
    assert res["d"]["n"] == 2 and abs(res["d"]["seconds_median"] - 20) < 1e-6
    rows, txt = fit.make_diff(res)
    names = {r["param"]: r for r in rows}
    assert names["rebalance_trigger"]["calibrated"] == "topology" and names["startup_delay_steps"]["calibrated"] == 1


def test_unsaturated_transition_is_excluded_not_reported_as_zero_loss(tmp_path):
    write_transitions(tmp_path, saturated=False)
    res = fit.fit_all([str(tmp_path)])
    assert res["delta_k"]["n_up"] == 0 and res["delta_k"]["delta_median"] is None
    assert all("not saturated" in x["excluded"] for x in res["delta_k"]["excluded"])


def test_episodes_without_valid_telemetry_are_censored(tmp_path):
    for kw in ({"telemetry": False}, {"worker_max": 0.95}):
        d = tmp_path / str(len(kw)) / str(kw.get("worker_max"))
        write_transitions(d, **kw)
        res = fit.fit_all([str(d)])
        assert res["delta_k"]["delta_median"] is None
        assert all("telemetry" in x["excluded"] for x in res["delta_k"]["excluded"])


def test_diff_written_not_applied(tmp_path, capsys):
    ds = two_sweeps(tmp_path, [1, 2, 4, 8])
    write_transitions(tmp_path / "tr")
    before = open(fit.os.path.join(fit.HERE, "..", "sim_v2", "params.py")).read()
    fit.sys.argv = ["fit.py", *ds, str(tmp_path / "tr"), "--out", str(tmp_path / "out")]
    fit.main()
    assert (tmp_path / "out" / "params_diff.txt").exists() and (tmp_path / "out" / "params_diff.json").exists()
    out = capsys.readouterr().out
    assert "mu_ops_per_pod" in out and "startup_delay_steps" in out and "NOT applied" in out
    assert open(fit.os.path.join(fit.HERE, "..", "sim_v2", "params.py")).read() == before
