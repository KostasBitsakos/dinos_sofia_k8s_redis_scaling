import json
import math
import os
import re
import subprocess
import sys

import pytest

import capstate
import common
import fit
import make_spec_stageB as mk
import run_c
import run_metric_lag
import run_transition

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_CAPACITY_STATE", str(tmp_path / "cap.json"))
    monkeypatch.setattr(common, "check_context", lambda: None)


def test_capstate_exact_interpolated_extrapolated_missing():
    with pytest.raises(LookupError):
        capstate.estimate("t", 2)
    capstate.record("t", 1, 100.0)
    capstate.record("t", 4, 400.0)
    capstate.record("other", 2, 5.0)
    assert capstate.estimate("t", 4) == (400.0, "exact")
    v, how = capstate.estimate("t", 2)
    assert v == pytest.approx(200.0) and how.startswith("interpolated")
    v, how = capstate.estimate("t", 9)
    assert v == pytest.approx(900.0) and how.startswith("linear extrapolation")
    assert capstate.estimate("other", 2)[0] == 5.0


def test_overload_fraction_2_in_default_sweep():
    assert 2.0 in run_c.FRACTIONS and max(run_c.FRACTIONS) == 2.0
    assert sum(f > 1.0 for f in run_c.FRACTIONS) >= 3


def _argv(monkeypatch, *a):
    monkeypatch.setattr(sys, "argv", ["x", *a])


def test_run_c_dry_run_options(monkeypatch, capsys):
    _argv(monkeypatch, "--c", "2", "--fractions", "0.5,2.0", "--keys", "6000", "--value-size", "1024", "--capacity-auto", "--dry-run")
    run_c.main()
    out = capsys.readouterr().out
    assert "6000 keys x 1024 B" in out and "capacity from state" in out and "[0.5, 2.0]" in out
    _argv(monkeypatch, "--c", "2", "--fractions", "0.5,-1", "--dry-run")
    with pytest.raises(SystemExit):
        run_c.main()


def test_run_c_passes_dataset_and_outstanding_to_loadgen(monkeypatch):
    seen = []

    class P:
        pid, returncode = 1, 0

        def wait(self):
            return 0

    monkeypatch.setattr(run_c.CHILDREN, "popen", lambda cmd: (seen.append(cmd), P())[1])
    ns = type("A", (), dict(seed=1, workers=3, c=2, timeout=2.0, keys=6000, value_size=1024, max_outstanding=20000))()
    run_c.lg("10:1", ns, 1, 0)
    cmd = " ".join(seen[0])
    assert "--keys 6000" in cmd and "--value-size 1024" in cmd and "--max-outstanding 20000" in cmd and "--timeout 2.0" in cmd


def test_run_transition_capacity_options(monkeypatch, capsys):
    capstate.record("50m", 8, 200000.0)
    _argv(monkeypatch, "--lo", "8", "--hi", "9", "--capacity-auto", "--tag", "50m", "--rate-frac", "1.2", "--keys", "6000",
          "--value-size", "1024", "--dry-run")
    # dry-run uses a placeholder; the real path reads the state:
    run_transition.main()
    assert "6000 keys x 1024 B" in capsys.readouterr().out
    _argv(monkeypatch, "--lo", "8", "--hi", "9", "--dry-run")
    with pytest.raises(SystemExit):
        run_transition.main()       # neither --capacity nor --capacity-auto
    _argv(monkeypatch, "--lo", "8", "--hi", "9", "--capacity", "1", "--capacity-auto", "--dry-run")
    with pytest.raises(SystemExit):
        run_transition.main()       # both


def test_parse_top():
    t = "redis-a-1   3m    9Mi\nredis-b-2   120m   10Mi\n"
    assert run_metric_lag.parse_top(t) == (123, 19, 2)
    assert run_metric_lag.parse_top("error: Metrics not available for pod qrl-calib/x") is None
    assert run_metric_lag.parse_top("") is None


def test_run_metric_lag_dry_run(monkeypatch, capsys):
    _argv(monkeypatch, "--c", "1", "--capacity-auto", "--dry-run")
    run_metric_lag.main()
    assert "kubectl top every 2.0s" in capsys.readouterr().out


def test_fit_pod_cpu_limit_override(monkeypatch):
    assert fit.POD_CPU_LIMIT == 0.2
    out = subprocess.run([sys.executable, "-c", "import fit; print(fit.POD_CPU_LIMIT)"], cwd=HERE, capture_output=True, text=True,
                         env=dict(os.environ, QRL_POD_CPU_LIMIT="0.05"))
    assert out.stdout.strip() == "0.05"


# ---------------------------------------------------------------- the frozen specs
def _spec(name):
    return json.load(open(os.path.join(HERE, name)))


@pytest.mark.parametrize("fn,gen", [("spec_stageB.json", mk.main_spec), ("spec_stageB_optional16.json", mk.opt16_spec)])
def test_spec_matches_generator_and_is_safe(fn, gen):
    spec = _spec(fn)
    assert spec == gen(), "committed spec differs from make_spec_stageB.py output: regenerate"
    assert spec["scope"] == "real" and spec["real"]["namespace"] == "qrl-calib" and spec["max_parallel"] == 1
    names = [e["name"] for e in spec["entries"]]
    assert len(set(names)) == len(names)
    for e in spec["entries"]:
        cmd = e["cmd"]
        assert cmd[1] == "gate.py", e["name"]                      # every entry gated
        if "kubectl" in cmd:
            i = cmd.index("kubectl")
            assert cmd[i + 1:i + 5] == ["--context", "kubernetes-admin@kubernetes", "--namespace", "qrl-calib"], e["name"]
            assert cmd[i + 5] in ("apply", "rollout"), e["name"]
            assert "scale" not in cmd and "patch" not in cmd and "edit" not in cmd
            if cmd[i + 5] == "apply":
                assert os.path.exists(os.path.join(HERE, cmd[cmd.index("-f") + 1]))
        if any(x in cmd for x in ("run_c.py", "run_transition.py", "run_metric_lag.py")):
            assert "--health" in cmd, e["name"]
    assert spec["entries"][-1]["cmd"][:4] == [mk.PY, "gate.py", "--allow-failed", "--"] and spec["entries"][-1]["cmd"][-1] == "rollback.sh"


def test_main_spec_capacity_auto_has_earlier_sweep_of_same_tag_and_workers_bounded():
    es = _spec("spec_stageB.json")["entries"]
    swept = set()
    ns_tag = None
    for e in es:
        s = " ".join(e["cmd"])
        m = re.search(r"qrl-calib(-50m)?\.yaml", s)
        if "apply" in e["cmd"]:
            ns_tag = "50m" if "50m" in s else "200m"
            swept = set()
        tag = re.search(r"--tag (\S+)", s)
        if tag:
            assert tag.group(1) == ns_tag, e["name"]                # the tag always names the namespace's CPU limit
            if "--capacity-auto" in s:
                assert any(True for _ in swept), e["name"]            # a probing sweep of this namespace came first
            elif "run_c.py" in s:
                swept.add(int(re.search(r"--c (\d+)", s).group(1)))
        w = re.search(r"--workers (\d+)", s)
        if w:
            assert int(w.group(1)) <= 16
        if "--capacity-auto" in s and "run_transition" in s:
            hi = int(re.search(r"--hi (\d+)", s).group(1))
            assert swept and hi <= 2 * max(swept) + 1                # extrapolation at most ~1 pod beyond the largest sweep at 8
        k, v = re.search(r"--keys (\d+)", s), re.search(r"--value-size (\d+)", s)
        if k and v:
            pods = int(re.search(r"--c (\d+)", s).group(1)) if "run_c.py" in s else int(re.search(r"--lo (\d+)", s).group(1))
            assert int(k.group(1)) * (int(v.group(1)) + 80) / pods <= 1.2e6, e["name"]   # <= 1.2 MB per pod: working set stays < ~0.75 of 16Mi
    assert ns_tag == "50m"


def test_no_200m_c4_and_generator_budget_respected():
    es = _spec("spec_stageB.json")["entries"]
    names = [e["name"] for e in es]
    assert not any(n.startswith("sweep-200m-c4") for n in names)
    for e in es:
        s = " ".join(e["cmd"])
        w = re.search(r"--workers (\d+)", s)
        if w:
            assert int(w.group(1)) <= 16
    import common
    assert common.gen_budget(16) == pytest.approx(16 * 0.85 / 38e-6)
    assert 350e3 < common.gen_budget(16) < 360e3


def test_three_loads_and_three_volumes_in_transitions():
    names = [e["name"] for e in _spec("spec_stageB.json")["entries"]]
    assert {n for n in names if n.startswith("transitions-load")} == {f"transitions-load{f}-50m-2-3-k2000x512B" for f in (0.5, 0.9, 1.2)}
    vols = {re.search(r"k(\d+x\d+)B", n).group(1) for n in names if n.startswith("transitions-")}
    assert len(vols) >= 3


def test_pod_cpu_stat_and_throttle_fraction(tmp_path):
    uid = "aaaa-bbbb"
    mem = tmp_path / "mem"
    cpu = tmp_path / "cpu"
    d = f"kubepods-burstable-pod{uid.replace('-', '_')}.slice:cri-containerd:abc"
    (mem / d).mkdir(parents=True)
    (cpu / d).mkdir(parents=True)
    (mem / d / "memory.limit_in_bytes").write_text("16777216\n")
    (cpu / d / "cpu.stat").write_text("nr_periods 100\nnr_throttled 40\nthrottled_time 2000000000\n")
    (cpu / d / "cpu.cfs_quota_us").write_text("20000\n")
    (cpu / d / "cpu.cfs_period_us").write_text("100000\n")
    st = common.pod_cpu_stat({"uid": uid}, str(mem), str(cpu))
    assert st == {"quota_us": 20000, "period_us": 100000, "nr_periods": 100, "nr_throttled": 40, "throttled_ns": 2_000_000_000}
    assert common.pod_cpu_stat({"uid": "zzz"}, str(mem), str(cpu)) is None      # unreadable: None, never zero


def _lvl(rows_off, rows_srv, p95, extra=None, hist=40):
    return dict(offered=rows_off, served=rows_srv, p95=p95, drop=0.0, timeout=0.0, err=0.0, invalid=[], has_hist=True, **(extra or {}))


def test_level_table_carries_throttling():
    rows = [{"t": 1000.0 + i, "phase": 1, "level": 0, "warm": 0, "pods": 1, "offered_ops": 100, "completed_ops": 100, "served_ops": 100,
             "error_ops": 0, "timeout_ops": 0, "drop_ops": 0, "generator_lag_ms": 1, "target_rate": 100, "p99_ms": 1.0, "hist": {"40": 5},
             "workers": 1, "_run": "r"} for i in range(4)]
    tele = [{"t": 1000.0 + 2 * k, "ncpu": 40, "host_busy_cores": 1.0, "gen_cores": .1, "gen_worker_max_cores": .1, "n_pods": 1, "pods_cores": .1,
             "d_nr_periods": 20, "d_nr_throttled": 5, "d_throttled_s": 0.1} for k in range(3)]
    t = list(fit.level_table(rows, tele).values())[0]
    assert t["throttled_frac"] == pytest.approx(0.25)
    t2 = list(fit.level_table(rows, [{k: v for k, v in x.items() if not k.startswith("d_")} for x in tele]).values())[0]
    assert t2["throttled_frac"] is None


def test_slo_feasible_ignores_invalid_and_failing_levels():
    import fit_extra as fx
    good = {(f"r{i}", 1, 1, i): _lvl(o, o, p) for i, (o, p) in enumerate([(100, 5), (200, 20), (300, 90), (400, 400)])}
    bad = _lvl(250, 250, 1.0)
    bad["drop"] = 5.0          # 2 % drops
    good[("r9", 1, 1, 9)] = bad
    gen = _lvl(260, 260, 1.0)
    gen["invalid"] = ["generator-cpu-bound"]
    good[("r10", 1, 1, 10)] = gen
    out = fx.slo_feasible(good, [1])["1"]
    assert out["10"]["served_ops_s"] == 100 and out["50"]["served_ops_s"] == 200 and out["100"]["served_ops_s"] == 300
    assert out["250"]["served_ops_s"] == 300


def test_erlang_and_mm1_agree_for_c1_and_mmc_is_faster_than_mm1():
    import fit_extra as fx
    mu, lam = 1000.0, 600.0
    q = fx.sojourn_quantile(0.95, mu, mu - lam, lam / mu)
    assert q == pytest.approx(math.log(20) / (mu - lam), rel=1e-6)
    c, lam = 4, 3200.0
    mm1 = math.log(20) / (mu - lam / c)
    mmc = fx.sojourn_quantile(0.95, mu, c * mu - lam, fx.erlang_c(c, lam / mu))
    assert mmc < mm1


def test_cv2_recovers_mm1_and_mg1():
    import fit_extra as fx
    from loadgen import BASE, LAT0
    mu = 1000.0                           # per pod ops/s
    cap = {1: mu}
    rows, table = [], {}
    for lv, rho in enumerate([0.1, 0.3, 0.5, 0.7, 0.85]):
        w_ms = 1e3 * (1 / mu + rho / (2 * mu * (1 - rho)) * (1 + 1.0)) + 0.2         # M/M/1 (cv2 = 1) + 0.2 ms network offset
        b = round(math.log(w_ms / 1e3 / LAT0) / math.log(BASE) - 0.5)
        rows.append({"warm": 0, "phase": 1, "_run": "r", "pods": 1, "level": lv, "hist": {str(b): 100}})
        table[("r", 1, 1, lv)] = _lvl(rho * mu, rho * mu, 1.0)
    out = fx.cv2_estimate(table, rows, cap)["1"]
    assert out["n_points"] == 5 and 0.0 < out["cv2"] < 2.5        # coarse: 5 % buckets; the point is a finite, sane estimate near 1


def test_substitution_tolerance():
    import fit_extra as fx
    assert fx.substitution_check({1: 100, 2: 200}, {1: 25, 2: 50})["accepted"] is True
    r = fx.substitution_check({1: 100, 2: 200}, {1: 25, 2: 70})
    assert r["accepted"] is False and "NOT validated" in r["scope"]
    assert fx.substitution_check({1: 100}, {1: 25})["accepted"] is None


def test_first_served_in_episodes_and_d():
    ev = [{"t": 10.0, "event": "scale_cmd", "direction": "up"}, {"t": 12.0, "event": "pod_ready"}, {"t": 20.0, "event": "first_slot"},
          {"t": 21.0, "event": "first_served"}]
    ep = fit.episodes(ev)[0]
    assert ep["t_first_served"] == 21.0 and ep["t_first_slot"] == 20.0


def _slo_runs(n_slo, fail_first=False, throttled=0.0):
    sweep = [(100, 100, 40), (500, 500, 60), (900, 900, 70), (1000, 600, 90), (1500, 600, 95)]
    ladder = [(250, 250, 50), (300, 300, 54), (350, 350, 58)]
    runs = {k: _fake_run(v, 3, k) for k, v in (("s1", sweep), ("s2", sweep))}
    for i in range(n_slo):
        runs[f"slo{i}"] = _fake_run(ladder, 3, f"slo{i}")
    if fail_first:
        for r in runs["slo0"][0]:
            r["drop_ops"] = r["offered_ops"] * 0.02
    for rows, _, tele in runs.values():
        for x in tele:
            x.update(d_nr_periods=10, d_nr_throttled=int(10 * throttled), d_throttled_s=0.0)
    return runs


def test_slo_requires_three_distinct_valid_runs_and_uses_offered():
    import slo_candidate as sc
    runs = _slo_runs(3)
    res = sc.compute(["s1", "s2"], ["slo0", "slo1", "slo2"], load=lambda d: runs[d])
    assert res["slo_s"] is not None and res["n_runs"] == 3 and res["deployment_specific"] is False
    lo, hi = res["slo_s_ci95_total"]
    assert lo <= res["slo_s"] <= hi and res["target_offered_ops_s"] == pytest.approx(300)
    one = sc.compute(["s1", "s2"], ["slo0"], load=lambda d: runs[d])
    assert one["slo_s"] is None and "3 distinct" in one["reason"]
    dup = sc.compute(["s1", "s2"], ["slo0", "slo0", "slo1"], load=lambda d: runs[d])     # duplicates collapse
    assert dup["slo_s"] is None
    two_ok = _slo_runs(3, fail_first=True)
    res3 = sc.compute(["s1", "s2"], ["slo0", "slo1", "slo2"], load=lambda d: two_ok[d])    # one run with 2 % drops is rejected -> only 2 valid
    assert res3["slo_s"] is None and len(res3["runs_rejected"]) == 1
    thr = _slo_runs(3, throttled=0.5)
    assert sc.compute(["s1", "s2"], ["slo0", "slo1", "slo2"], load=lambda d: thr[d])["deployment_specific"] is True
    assert sc.compute(["s1", "s2"], ["s1", "slo1", "slo2"], load=lambda d: runs.get(d))["reason"].startswith("a run dir is both")


def test_generator_saturated_levels_never_give_capacity():
    rows = []
    for lv, (off, srv) in enumerate([(100, 100), (200, 200), (300, 190), (400, 190)]):
        for i in range(3):
            rows.append({"t": 1000.0 + lv * 10 + i, "phase": 1, "level": lv, "warm": 0, "pods": 1, "offered_ops": off, "completed_ops": srv,
                         "served_ops": srv, "error_ops": 0, "timeout_ops": 0, "drop_ops": 0, "generator_lag_ms": 1, "target_rate": off,
                         "p99_ms": 1.0, "hist": {"70": 5}, "workers": 1, "_run": "r"})
    tele = [{"t": 1000.0 + k, "ncpu": 40, "host_busy_cores": 1.0, "gen_cores": 1.0, "gen_worker_max_cores": 0.95, "n_pods": 1, "pods_cores": 0.2}
            for k in range(100)]            # the plateau coincides with a CPU-bound generator worker
    cap, _, rej, inv = fit.capacity_per_c(fit.level_table(rows, tele))
    assert cap == {} and inv and all("generator-cpu-bound" in i["reasons"] for i in inv)


def test_cpu_limit_variant_yaml_differs_only_in_cpu():
    a = open(os.path.join(HERE, "k8s", "qrl-calib.yaml")).read()
    b = open(os.path.join(HERE, "k8s", "qrl-calib-50m.yaml")).read()
    b = "\n".join(l for l in b.splitlines() if not l.startswith("# STAGE B") and not l.startswith("# can saturate") and not l.startswith("# test scaling")).replace("50m", "TMP")
    a = a.replace("150m", "TMP").replace("200m", "TMP")
    assert b.replace("\n\n", "\n").strip() == a.strip()
    assert set(re.findall(r"^\s+namespace:\s*(\S+)", b, re.M)) == {"qrl-calib"}


# ---------------------------------------------------------------- SLO candidate (R-20261017-018)
def _fake_run(c1_pts, nlev, name="r"):
    rows = []
    for lv, (off, srv, h) in enumerate(c1_pts):
        for i in range(nlev):
            rows.append({"t": 1000.0 + lv * 100 + i, "phase": 1, "level": lv, "warm": 0, "pods": 1, "offered_ops": off, "completed_ops": srv,
                         "served_ops": srv, "error_ops": 0, "timeout_ops": 0, "drop_ops": 0, "generator_lag_ms": 1, "target_rate": off,
                         "p99_ms": 1.0, "hist": {str(h): 10}, "workers": 1, "_run": name})
    tele = [{"t": 1000.0 + k, "ncpu": 40, "host_busy_cores": 1.0, "gen_cores": 0.5, "gen_worker_max_cores": 0.1, "n_pods": 1,
             "pods_cores": 0.2} for k in range(1000)]
    return rows, [], tele




def test_slo_interpolation_is_against_offered():
    import slo_candidate as sc
    lv = [{"offered": 100.0, "p95": 2.0}, {"offered": 200.0, "p95": 8.0}]
    p, pair = sc.interp_p95(lv, 150.0)
    assert p == pytest.approx(4.0) and len(pair) == 2
    assert sc.interp_p95(lv, 50.0) == (None, None) and sc.interp_p95(lv, 250.0) == (None, None)


def test_served_calls_parses_get_and_set_only(monkeypatch):
    info = "# Commandstats\r\ncmdstat_get:calls=10,usec=5\r\ncmdstat_set:calls=4,usec=2\r\ncmdstat_restore-asking:calls=99,usec=1\r\ncmdstat_cluster|nodes:calls=7\r\n"
    monkeypatch.setattr(common, "redis_info", lambda ip, section, timeout=2.0: info)
    assert common.served_calls("10.0.0.1") == 14

    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(common, "redis_info", boom)
    assert common.served_calls("10.0.0.1") is None
