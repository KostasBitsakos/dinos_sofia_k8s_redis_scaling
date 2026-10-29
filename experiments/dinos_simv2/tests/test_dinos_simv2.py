import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import adapter  # noqa: E402
import train_eval  # noqa: E402
from adapter import DinosSimV2, SimParams, normalise  # noqa: E402


def test_window_shape_and_range():
    sim = DinosSimV2("W0")
    w = sim.reset(0)
    assert w.shape == (10, 6)
    for _ in range(30):
        w, r, done, info = sim.step(2)
    assert w.shape == (10, 6) and np.isfinite(w).all()
    assert w.min() >= 0.0 and w.max() <= 1.5


def test_normalise_fixed_constants():
    P = SimParams()
    n = normalise([30000.0, 100.0, 9999.0, 0.0, 30.0, 4.0], P)
    assert np.allclose(n, [1.0, 1.0, 1.0, 0.0, 1.0, 1.0], atol=1e-3)
    assert np.allclose(normalise([0, 0, 0, 0, 0, 0], P), 0.0)
    # fixed: the same raw observation always maps to the same vector (no running statistics)
    assert np.array_equal(normalise([5000, 50, 20, 1, 6, 1], P), normalise([5000, 50, 20, 1, 6, 1], P))


def test_actions_and_window_roll():
    sim = DinosSimV2("W0")
    w0 = sim.reset(1)
    assert np.allclose(w0, w0[0])  # filled with the first observation
    w1, *_ = sim.step(1)
    assert np.allclose(w1[:-1], w0[1:])


def _args(tmp, agent="lstm_scratch", **kw):
    a = ["--agent", agent, "--train-steps", "40", "--eval-episodes", "0", "--log-every", "20", "--batch-size", "8",
         "--checkpoint-every", "0", "--amendment-id", "R-TEST-000"]
    for k, v in kw.items():
        a += [f"--{k}", str(v)]
    return a


@pytest.mark.parametrize("agent", ["lstm_scratch", "dinos_t", "derp"])
def test_one_training_step_runs(tmp_path, monkeypatch, agent):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    extra = {"pretrained": 5} if agent == "dinos_t" else {}
    train_eval.main(_args(tmp_path, agent, **extra))
    rows = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    train_rows = [r for r in rows if "train_mean_reward" in r]
    assert [r["step"] for r in train_rows] == [20, 40]
    assert np.isfinite(train_rows[-1]["loss"])  # at least one replay happened
    assert json.load(open(tmp_path / "progress.json"))["step"] == 40
    assert (tmp_path / "checkpoints" / "model.pth").exists()


def test_eval_metrics_content_and_determinism(tmp_path, monkeypatch):
    outs = []
    for i in range(2):
        d = tmp_path / str(i)
        monkeypatch.setenv("QRL_RUN_DIR", str(d))
        # short episodes would need params changes; evaluate a baseline-free agent on one real 5000-step episode
        train_eval.main(["--agent", "lstm_scratch", "--train-steps", "60", "--eval-episodes", "1", "--log-every", "30",
                         "--batch-size", "8", "--checkpoint-every", "0", "--seed", "3", "--amendment-id", "R-TEST-000"])
        rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
        outs.append([{k: v for k, v in r.items() if "wall" not in k} for r in rows])
    ev = [r for r in outs[0] if "eval_return" in r][0]
    for k in ("eval_cost", "eval_sla", "eval_churn", "eval_mean_pods", "eval_sla_violation_rate",
              "eval_mean_p95_ms", "eval_seed", "eval_episode"):
        assert k in ev
    assert ev["eval_seed"] == 20000
    assert ev["eval_return"] == pytest.approx(ev["eval_cost"] + ev["eval_sla"] + ev["eval_churn"], rel=1e-9)
    assert outs[0] == outs[1]  # same seed, same config -> identical metrics


# ---------- Sim v2.1: threshold grid edge check, sensitivity evaluation ----------
def test_threshold_edge_check():
    from policies import Threshold
    ok = Threshold(40, 25, 0)  # interior; cooldown 0 is the natural floor
    assert train_eval.threshold_edge_axes(ok) == []
    train_eval.assert_threshold_interior(ok)
    for thr, axis in ((Threshold(25, 10, 1), "up"), (Threshold(95, 10, 1), "up"), (Threshold(40, 5, 1), "down"),
                      (Threshold(70, 60, 1), "down"), (Threshold(40, 25, 8), "cooldown")):
        assert axis in train_eval.threshold_edge_axes(thr)
        with pytest.raises(RuntimeError):
            train_eval.assert_threshold_interior(thr)
    g = train_eval.THRESHOLD_GRID_V21
    assert min(g["up"]) < 40 < max(g["up"])  # contains the old Sim v2 grid and its edge point (40) in the interior


def test_baselines_edge_failure_and_sensitivity_rows(tmp_path, monkeypatch):
    """Tiny grid so the smoke is fast: a grid whose optimum is on the edge must fail after logging."""
    import policies
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(train_eval, "TUNE_SEEDS", (1000,))
    monkeypatch.setattr(train_eval, "THRESHOLD_GRID_V21", {"up": (60, 90), "down": (20, 30), "cooldown": (0, 1)})
    monkeypatch.setattr(policies, "select_static", lambda *a, **k: (policies.Fixed(10), []))
    with pytest.raises(RuntimeError, match="grid edge"):
        train_eval.main(["--agent", "baselines", "--eval-episodes", "1", "--sensitivity", "default"])
    rows = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    assert rows[0]["threshold_on_grid_edge"] >= 1


def test_agent_eval_under_sensitivity_configs(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    train_eval.main(["--agent", "lstm_scratch", "--train-steps", "40", "--eval-episodes", "1", "--log-every", "20",
                     "--batch-size", "8", "--checkpoint-every", "0", "--amendment-id", "R-TEST-000",
                     "--sensitivity", "default,tail_scale_1.5,cpu_offset_10"])
    rows = [json.loads(l) for l in open(tmp_path / "metrics.jsonl") if "eval_return" in l]
    assert [r["config_index"] for r in rows] == [0, 1, 2]
    names = json.load(open(tmp_path / "configs.json"))
    assert names == {"0": "default", "1": "tail_scale_1.5", "2": "cpu_offset_10"}
    assert rows[1]["eval_sla"] <= rows[0]["eval_sla"] + 1e-9  # tighter SLO can only add bad requests
    assert json.load(open(tmp_path / "progress.json")) == {"step": 43, "total": 43}


def _strip_commit(spec):
    return {k: v for k, v in spec.items() if k != "frozen_git_commit"}


def _strip_freeze(spec):
    """HISTORICAL specs (make_spec.HISTORICAL): their manifests record what was approved/run and are not regenerated
    after later code changes; the generator must still produce the same entries and limits."""
    out = {k: v for k, v in spec.items() if k not in ("frozen_git_commit", "frozen_sha256")}
    return out


def test_spec_v21_valid_and_matches_generator():
    import make_spec
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "tools")))
    from labdb import validate_spec
    spec = make_spec.build(allow_dirty=True, legacy=True)
    assert validate_spec(spec)
    assert spec["limits"]["wall_clock_hours"] == 1.5 and spec["max_parallel"] == 5 and len(spec["entries"]) == 10
    assert all("--sensitivity" in e["cmd"] and "all" in e["cmd"] for e in spec["entries"])
    assert sum("100000" in e["cmd"] for e in spec["entries"]) == 9
    for e in spec["entries"]:   # freeze enforced in every entry, never bypassed
        assert e["cmd"][e["cmd"].index("--freeze") + 1] == "spec_w0_v21.json" and "--unfrozen" not in e["cmd"]
    with open(os.path.join(HERE, "..", "spec_w0_v21.json")) as f:
        assert _strip_freeze(json.load(f)) == _strip_freeze(spec)  # on disk is current: rerun make_spec.py


def test_preflight_spec_valid():
    import make_spec
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "tools")))
    from labdb import validate_spec
    spec = make_spec.build_preflight(allow_dirty=True, legacy=True)
    assert validate_spec(spec)
    assert len(spec["entries"]) == 1 and spec["limits"]["wall_clock_hours"] == 1.0
    c = spec["entries"][0]["cmd"]
    assert "preflight" in c and c[c.index("--freeze") + 1] == "spec_preflight_v21.json"
    with open(os.path.join(HERE, "..", "spec_preflight_v21.json")) as f:
        assert _strip_freeze(json.load(f)) == _strip_freeze(spec)


# ---------- freeze enforcement ----------
def _git_repo(tmp_path):
    import subprocess
    r = tmp_path / "repo"
    r.mkdir()
    (r / "a.py").write_text("x = 1\n")
    (r / "b.json").write_text("{}\n")
    for c in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(r), *c], check=True)
    return r


def _spec(repo, **over):
    import freeze
    files = ["a.py", "b.json"]
    spec = {"frozen_sha256": {f: freeze.sha256_file(str(repo / f)) for f in files},
            "frozen_git_commit": freeze.git_head(str(repo))}
    spec.update(over)
    (repo / "spec.json").write_text(json.dumps(spec))
    return str(repo / "spec.json"), files


def test_freeze_accepts_clean_and_refuses_tampering(tmp_path):
    import freeze
    repo = _git_repo(tmp_path)
    sp, files = _spec(repo)
    import subprocess
    assert freeze.verify_freeze(sp, required=files)["git_head"]
    # 1) file edited (hash mismatch)
    (repo / "a.py").write_text("x = 2\n")
    with pytest.raises(freeze.FreezeError, match="sha256 mismatch"):
        freeze.verify_freeze(sp, required=files)
    # 2) hash re-recorded but change not committed -> git dirty
    sp, _ = _spec(repo)
    with pytest.raises(freeze.FreezeError, match="not clean"):
        freeze.verify_freeze(sp, required=files)
    # 3) committed after the frozen commit -> frozen files changed since the recorded commit
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "edit"], check=True)
    old = freeze.git_head(str(repo))
    sp2, _ = _spec(repo, frozen_git_commit=subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD~1"], capture_output=True, text=True).stdout.strip())
    with pytest.raises(freeze.FreezeError, match="changed between"):
        freeze.verify_freeze(sp2, required=files)
    assert old
    # 4) missing file / manifest lacking a required file / unrelated commit
    sp3, _ = _spec(repo)
    with pytest.raises(freeze.FreezeError, match="lacks required"):
        freeze.verify_freeze(sp3, required=files + ["c.py"])
    (repo / "b.json").unlink()
    with pytest.raises(freeze.FreezeError, match="missing"):
        freeze.verify_freeze(sp3, required=files)


def test_train_eval_refuses_tampered_spec_before_training(tmp_path, monkeypatch):
    import freeze
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    spec = json.load(open(os.path.join(HERE, "..", "spec_w0_v21.json")))
    base = os.path.join(HERE, "..")
    spec["frozen_sha256"] = {os.path.normpath(os.path.join(base, k)): v for k, v in spec["frozen_sha256"].items()}
    spec["frozen_sha256"][os.path.normpath(os.path.join(base, "adapter.py"))] = "0" * 64   # tampered
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    monkeypatch.setattr(freeze, "FROZEN", list(spec["frozen_sha256"]))
    with pytest.raises(freeze.FreezeError, match="sha256 mismatch"):
        train_eval.main(["--agent", "lstm_scratch", "--train-steps", "40", "--eval-episodes", "0", "--freeze", str(sp)])
    assert not (tmp_path / "run" / "metrics.jsonl").exists()


def test_no_freeze_flag_refuses(monkeypatch):
    monkeypatch.delenv("QRL_ALLOW_UNFROZEN", raising=False)
    with pytest.raises(SystemExit, match="refusing"):
        train_eval.main(["--agent", "lstm_scratch", "--train-steps", "40"])


def test_frozen_manifest_covers_all_dependencies():
    """Real spec + real imports: every source file the agents load is in the manifest, and the manifest lists the
    sim_v2 modules, the sensitivity JSON and the pretrained checkpoint."""
    import agents
    import freeze
    # generators / gate tools / selection scripts imported by other test modules in the same pytest process are not
    # dependencies of a training run (pre-existing failure when sim_v2/tests ran first: gates.py, stats.py)
    for m in ("make_spec", "gates", "stats", "run_gates", "select_baselines_dinos", "reward_check", "determinism_check"):
        sys.modules.pop(m, None)
    agents.make_lstm(5)
    agents.make_derp()
    import policies, workloads, sensitivity, env, oracle, params  # noqa: E401,F401
    base = os.path.dirname(HERE)
    man = freeze.make_manifest(base)
    freeze.check_loaded_modules(man, base)
    keys = " ".join(man)
    for needle in ("env.py", "params.py", "oracle.py", "policies.py", "workloads.py", "sensitivity.py",
                   "sensitivity_v21.json", "modelCPU_5.pth", "agents.py", "adapter.py", "train_eval.py", "agentHuber.py"):
        assert needle in keys
    with pytest.raises(freeze.FreezeError, match="missing from the frozen manifest"):
        freeze.check_loaded_modules({k: v for k, v in man.items() if not k.endswith("policies.py")}, base)


# ---------- seed hygiene ----------
def test_seed_sets_disjoint_and_fresh():
    from workloads import SEED_RANGES
    sets = train_eval.seed_sets(5)
    assert sets["test"] == {20000 + i for i in range(5)}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not sets[a] & sets[b], (a, b)
    full = train_eval.seed_sets(1000)
    assert not full["test"] & set(train_eval.BURNED)                      # burned 10000-10999 never evaluated on
    assert not set(SEED_RANGES["test"]) & set(SEED_RANGES["heldout"])
    assert not set(SEED_RANGES["test"]) & (set(SEED_RANGES["train"]) | set(SEED_RANGES["tune"]))
    # the baselines tune seeds and the gate config's tune seeds stay validation, the gate held-out block stays burned
    assert set(train_eval.TUNE_SEEDS) <= set(SEED_RANGES["tune"])
    with open(os.path.join(HERE, "..", "..", "..", "sim_v2", "gates_config.json")) as f:
        g = json.load(f)
    assert g["heldout_seeds_start"] in train_eval.BURNED


def test_train_seeds_never_touch_validation_or_test():
    sd = train_eval.train_seeds(0, 20)
    assert not set(sd) & (set(train_eval.TUNE_SEEDS) | set(train_eval.eval_seeds(1000)))
    with pytest.raises(ValueError):
        from workloads import check_seed
        check_seed("test", 10000)


# ---------- PLAN 8.1 gate ----------
def test_gate_check_reports_and_fails_loudly():
    good = dict(threshold=[-10, -11], fixed_best=[-12, -13], always_min=[-90, -80], always_max=[-50, -60],
                oracle=[-8, -9])
    g = train_eval.gate_check(good)
    assert g["passed"] and g["oracle_sane"] and g["controls"]["fixed_best"]["wins"] == 2
    bad = dict(good, fixed_best=[-9, -9.5])      # a static control beats the tuned threshold
    g = train_eval.gate_check(bad)
    assert not g["passed"] and not g["controls"]["fixed_best"]["beats"]
    g = train_eval.gate_check(dict(good, oracle=[-20, -20]))
    assert g["passed"] and not g["oracle_sane"]


def test_baselines_gate_failure_raises(tmp_path, monkeypatch):
    import policies
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(train_eval, "TUNE_SEEDS", (1000,))
    monkeypatch.setattr(train_eval, "THRESHOLD_GRID_V21", {"up": (40, 60, 90), "down": (5, 20, 60), "cooldown": (0, 1, 8)})
    monkeypatch.setattr(policies, "select_static", lambda *a, **k: (policies.Fixed(10), []))
    monkeypatch.setattr(train_eval, "threshold_edge_axes", lambda *a, **k: [])
    monkeypatch.setattr(train_eval, "gate_check", lambda r: dict(passed=False, oracle_sane=True, controls={}))
    with pytest.raises(RuntimeError, match="sanity gate FAILED"):
        train_eval.main(["--agent", "baselines", "--eval-episodes", "1", "--sensitivity", "default"])
    assert (tmp_path / "gate_failed.json").exists()


def test_preflight_smoke(tmp_path, monkeypatch):
    """Tiny grid, one validation seed, one fixed(c) candidate range: structure of the report, not the values."""
    import policies
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(train_eval, "TUNE_SEEDS", (1000,))
    monkeypatch.setattr(train_eval, "THRESHOLD_GRID_V21", {"up": (40, 60, 90), "down": (5, 20, 60), "cooldown": (0, 1, 8)})
    monkeypatch.setattr(train_eval, "threshold_edge_axes", lambda *a, **k: [])
    orig = policies.select_static
    monkeypatch.setattr(policies, "select_static", lambda P, w, s, fam, candidates=None: orig(P, w, s, fam, range(5, 25, 5)))
    try:
        train_eval.main(["--agent", "preflight"])
    except RuntimeError as e:   # the tiny grid may legitimately fail the gate; the report is written first
        assert "preflight sanity FAILED" in str(e)
    rep = json.load(open(tmp_path / "preflight.json"))
    assert rep["policy_order"] == ["threshold", "fixed_best", "always_min", "always_max", "oracle"]
    for m in rep["summary"].values():
        assert m["ret"] == pytest.approx(m["cost"] + m["sla"] + m["churn"], rel=1e-9)
        assert len(m["per_episode_return"]) == 1
    assert {"threshold_tuning", "fixed_selection", "oracle"} <= set(rep["timings_s"])
    assert rep["seeds"] == [1000] and (tmp_path / "preflight_table.txt").exists()
    assert json.load(open(tmp_path / "freeze_report.json"))["unfrozen"] is True


def test_diagnose_spec_valid():
    import make_spec
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "tools")))
    from labdb import validate_spec
    spec = make_spec.build_diagnose(allow_dirty=True, legacy=True)
    assert validate_spec(spec)
    assert len(spec["entries"]) == 1 and spec["limits"]["wall_clock_hours"] == 1.0
    c = spec["entries"][0]["cmd"]
    assert "diagnose" in c and c[c.index("--freeze") + 1] == "spec_diagnose_v21.json" and "--unfrozen" not in c
    assert c[c.index("--preflight-json") + 1] == make_spec.PREFLIGHT_JSON
    with open(os.path.join(HERE, "..", "spec_diagnose_v21.json")) as f:
        assert _strip_freeze(json.load(f)) == _strip_freeze(spec)


def test_diagnose_smoke_no_retuning(tmp_path, monkeypatch):
    """200-step episodes, 2 configs, preflight.json fixture: selection is loaded (tune functions never called),
    traces reproduce the table, outputs exist, sha256 recorded."""
    import hashlib
    import policies
    import sensitivity
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    pre = tmp_path / "preflight.json"
    pre.write_text(json.dumps(dict(workload="W0", seeds=list(train_eval.TUNE_SEEDS), threshold=[40.0, 25.0, 0],
                                   fixed_best_c=22)))
    def small(base=None):
        return (base or SimParams()).with_(episode_steps=200)
    monkeypatch.setattr(train_eval, "SimParams", small)
    monkeypatch.setattr(sensitivity, "names", lambda: ["default", "rebalance_k14"])
    orig = sensitivity.params
    monkeypatch.setattr(sensitivity, "params", lambda name, base=None: orig(name, base or small()))
    def boom(*a, **k):
        raise AssertionError("diagnose must not re-tune")
    monkeypatch.setattr(policies, "tune_threshold", boom)
    monkeypatch.setattr(policies, "select_static", boom)
    out = train_eval.main(["--agent", "diagnose", "--preflight-json", str(pre), "--unfrozen"])
    run = tmp_path / "run"
    rep = json.load(open(run / "diagnose.json"))
    assert rep["preflight_sha256"] == hashlib.sha256(pre.read_bytes()).hexdigest() == out["preflight_sha256"]
    assert rep["threshold"] == [40.0, 25.0, 0] and rep["fixed_best_c"] == 22 and rep["no_reselection"]
    assert set(rep["sensitivity"]) == {"default", "rebalance_k14"}
    assert set(rep["sensitivity"]["default"]["mean_return"]) == {"threshold", "fixed_best", "always_min", "always_max", "oracle"}
    assert len(rep["trace_files"]) == 15 and (run / "diagnose_table.txt").exists()
    assert len(list((run / "plots").glob("diagnose_seed*.png"))) == 5
    for pn in ("threshold", "fixed_best", "oracle"):
        a = rep["accounting_per_seed"][pn]["1000"]
        assert a["all"]["ret"] == pytest.approx(a["startup"]["ret"] + a["rest"]["ret"])
        assert a["startup"]["steps"] == 150 and a["rest"]["steps"] == 50
    import csv
    rows = list(csv.DictReader(open(run / "traces" / "trace_seed1000_threshold.csv")))
    assert len(rows) == 200 and "rebalancing" in rows[0] and "backlog" in rows[0]
    assert rep["accounting_per_seed"]["fixed_best"]["1000"]["all"]["rebalancing_fraction"] == 0.0


# ---------- Sim v2.1 protocol rev 1 ----------
def test_rev1_seeds_disjoint_and_never_used_in_code():
    from workloads import SEED_RANGES
    cfg = train_eval.load_gate_rev1()
    ev, sel = set(cfg["eval_seeds"]), set(cfg["selection_seeds"])
    assert ev == set(range(1200, 1210)) == set(train_eval.REV1_SEEDS) and sel == set(train_eval.TUNE_SEEDS)
    assert not ev & sel and ev <= set(SEED_RANGES["tune"])
    assert not ev & set(train_eval.REV1_BURNED) and set(train_eval.REV1_BURNED) <= set(SEED_RANGES["tune"])
    assert not ev & (set(SEED_RANGES["train"]) | set(SEED_RANGES["test"]) | set(train_eval.BURNED))
    assert not ev & set(train_eval.train_seeds(0, 1000))
    with open(os.path.join(HERE, "..", "..", "..", "sim_v2", "gates_config.json")) as f:
        assert not ev & set(json.load(f)["tune_seeds"])
    # no source file/test other than the registration itself names a rev-1 seed (1200-1209) as a literal
    import re
    root = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
    pat = re.compile(r"\b120[0-9]\b")
    # make_report_figs.py only prints seed ranges as prose / figure labels of stored results (no simulator import)
    assert not re.search(r"^\s*(import|from)\s+(env|adapter|train_eval|sim_v2|policies|workloads)\b",
                         open(os.path.join(root, "experiments", "dinos_simv2", "make_report_figs.py")).read(), re.M)
    for d in ("sim_v2", "experiments/dinos_simv2"):
        for dp, _, fs in os.walk(os.path.join(root, d)):
            for fn in fs:
                if fn.endswith((".py", ".json", ".md")) and fn not in ("test_dinos_simv2.py", "test_explore_w0.py", "conftest.py", "train_eval.py", "gate_rev1.json", "PREREG_rev1.md", "make_report_figs.py") \
                        and "spec_" not in fn and "README" not in fn:
                    for i, line in enumerate(open(os.path.join(dp, fn), errors="ignore")):
                        if pat.search(line) and True:
                            raise AssertionError(f"{fn}:{i+1} mentions a rev-1 seed: {line.strip()}")


def test_static_ramp_starts_at_c0_ramps_at_max_rate_then_holds():
    from policies import StaticRamp, Fixed
    from env import SimV2
    P = SimParams().with_(episode_steps=60)
    sim = SimV2(P, workload="W0")
    pol = StaticRamp(10)
    assert getattr(pol, "init_pods", None) is None      # starts at the env default c0, like threshold
    obs, _ = sim.reset(1000, init_pods=None)       # selection block; tests never use the rev-1 seeds
    assert int(round(obs[4])) == P.init_pods == 6
    pol.reset()
    totals, ups = [], 0
    for _ in range(60):
        a = pol.act(obs)
        ups += a == 2
        obs, _, term, trunc, info = sim.step(a)
        totals.append(info["ready"] + info["pending"])
        if term or trunc:
            break
    assert totals[:4] == [7, 8, 9, 10] and set(totals[4:]) == {10}   # +1/step (max_delta), then hold
    assert ups == 4
    assert getattr(Fixed(22), "init_pods") == 22                     # preprovisioned comparator is separate


def test_gate_rev1_ci_logic_synthetic():
    cfg = train_eval.load_gate_rev1()
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 10)
    good = dict(threshold=base + 10, static_ramp_best=base, fixed_best=base + 20, always_min=base - 5,
                always_max=base - 5, oracle=base + 30)
    g = train_eval.gate_rev1_check(good, cfg)
    assert g["passed"] and g["controls"]["static_ramp_best"]["ci_lo"] > 0
    assert g["controls"]["static_ramp_best"]["wins"] == 10
    assert g["comparators"]["fixed_best_preprovisioned"]["mean_diff"] < 0
    # deterministic bootstrap
    assert train_eval.gate_rev1_check(good, cfg)["controls"]["static_ramp_best"]["ci_lo"] == g["controls"]["static_ramp_best"]["ci_lo"]
    # mean > 0 but CI includes 0 -> fail
    noisy = dict(good, threshold=base + np.array([30, -25, 28, -22, 1, 0.5, -1, 2, -3, 3.0]))
    g = train_eval.gate_rev1_check(noisy, cfg)
    r = g["controls"]["static_ramp_best"]
    assert r["mean_diff"] > 0 and r["ci_lo"] <= 0 and not g["passed"]
    # each other criterion fails alone
    assert not train_eval.gate_rev1_check(dict(good, always_min=base + 11), cfg)["passed"]
    assert not train_eval.gate_rev1_check(dict(good, always_max=base + 11), cfg)["passed"]
    assert not train_eval.gate_rev1_check(dict(good, oracle=base + 5), cfg)["passed"]
    assert not train_eval.gate_rev1_check(dict(good, threshold=base - 10), cfg)["passed"]
    # fixed_best comparator never gates
    assert train_eval.gate_rev1_check(dict(good, fixed_best=base + 1000), cfg)["passed"]
    # startup split adds up
    st = {k: v * 0.1 for k, v in good.items()}
    g = train_eval.gate_rev1_check(good, cfg, st)["controls"]["static_ramp_best"]
    assert g["startup_mean_diff"] + g["rest_mean_diff"] == pytest.approx(g["mean_diff"])


def test_rev1_spec_valid_and_freeze_covers_registration():
    import freeze
    import make_spec
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "tools")))
    from labdb import validate_spec
    spec = make_spec.build_rev1(allow_dirty=True, legacy=True)
    assert validate_spec(spec)
    assert len(spec["entries"]) == 1 and spec["limits"]["wall_clock_hours"] == 1.0
    c = spec["entries"][0]["cmd"]
    assert "preflight_rev1" in c and c[c.index("--freeze") + 1] == "spec_preflight_rev1.json" and "--unfrozen" not in c
    assert "gate_rev1.json" in spec["frozen_sha256"] and "PREREG_rev1.md" in spec["frozen_sha256"]
    with open(os.path.join(HERE, "..", "spec_preflight_rev1.json")) as f:
        assert _strip_freeze(json.load(f)) == _strip_freeze(spec)


def _small(monkeypatch):
    import policies
    monkeypatch.setattr(train_eval, "SimParams", lambda base=None: (base or SimParams()).with_(episode_steps=200))
    orig = policies.select_static
    monkeypatch.setattr(policies, "select_static", lambda P, w, seeds, fam, **k: orig(P, w, seeds, fam, candidates=[4, 8, 12]))

    def boom(*a, **k):
        raise AssertionError("must not re-tune the threshold")
    monkeypatch.setattr(policies, "tune_threshold", boom)


def _fixture_pre(tmp_path):
    pre = tmp_path / "preflight.json"
    pre.write_text(json.dumps(dict(workload="W0", seeds=list(train_eval.TUNE_SEEDS), threshold=[40.0, 25.0, 0],
                                   fixed_best_c=22)))
    return pre


def test_preflight_rev1_smoke_on_development_seeds(tmp_path, monkeypatch):
    """200-step episodes, DEVELOPMENT seeds 1000-1004 via the unfrozen override, selection loaded not tuned."""
    import hashlib
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    pre = _fixture_pre(tmp_path)
    _small(monkeypatch)
    try:
        out = train_eval.main(["--agent", "preflight_rev1", "--preflight-json", str(pre), "--unfrozen",
                               "--rev1-eval-seeds", "1000,1001,1002,1003,1004"])
    except RuntimeError as e:   # a gate failure on 200-step episodes is legitimate; the report must exist
        assert "gate rev 1 FAILED" in str(e)
        out = None
    run = tmp_path / "run"
    rep = json.load(open(run / "preflight_rev1.json"))
    assert rep["eval_seeds"] == [1000, 1001, 1002, 1003, 1004] and rep["no_reselection"] and rep["c0"] == 6
    assert rep["static_ramp_best_c"] in (4, 8, 12) and rep["threshold"] == [40.0, 25.0, 0]
    assert rep["preflight_sha256"] == hashlib.sha256(pre.read_bytes()).hexdigest()
    assert all(len(v) == 5 for v in rep["per_episode_return"].values())
    assert (run / "preflight_rev1_table.txt").exists()
    assert out is None or out["gate_passed"] is True


def test_gate_block_cannot_be_evaluated_by_tests_or_smokes(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    pre = _fixture_pre(tmp_path)
    for seeds in ("1200,1201", "1100"):   # gate block and burned first choice
        with pytest.raises(RuntimeError, match="gate seed block|burned"):
            train_eval.main(["--agent", "preflight_rev1", "--preflight-json", str(pre), "--unfrozen",
                             "--rev1-eval-seeds", seeds])
    # no override: the unfrozen run would evaluate the gate block -> refused
    with pytest.raises(RuntimeError, match="gate seed block"):
        train_eval.main(["--agent", "preflight_rev1", "--preflight-json", str(pre), "--unfrozen"])
    # the conftest guard stops any direct simulator use on the block
    with pytest.raises(AssertionError, match="never evaluate"):
        DinosSimV2("W0", SimParams()).reset(1203)
    with pytest.raises(AssertionError, match="never evaluate"):
        DinosSimV2("W0", SimParams()).reset(1105)


ORIG_PRE = "/home/ubuntu/qrl-lab/.lab/runs/20261017-140714-R-20261017-004-e0-a1/preflight.json"


def test_original_preflight_pinned_and_recorded_fail(tmp_path):
    cfg = train_eval.load_gate_rev1()
    assert os.path.exists(ORIG_PRE), "original preflight.json missing: the pinned FAIL cannot be verified"
    pre, sha = train_eval.verify_original_preflight(ORIG_PRE, cfg)
    assert sha == cfg["original_preflight"]["sha256"] and pre["gate"]["passed"] is False and pre["fixed_best_c"] == 22
    bad = tmp_path / "preflight.json"
    d = json.load(open(ORIG_PRE))
    d["fixed_best_c"] = 21
    bad.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="sha256"):
        train_eval.verify_original_preflight(str(bad), cfg)


def test_static_ramp_tie_break_is_smallest_c(monkeypatch):
    import policies
    monkeypatch.setattr(policies, "run_episode", lambda env, pol, s: {"ret": -1.0})
    best, scores = policies.select_static(SimParams(), "W0", train_eval.TUNE_SEEDS, "ramp")   # default: ascending 1..30
    assert [c for c, _ in scores] == list(range(1, 31)) and best.c == 1   # all tied -> smallest c


def _passing_artifact(tmp_path, **over):
    import hashlib
    cfg = train_eval.load_gate_rev1()
    rep = dict(protocol=train_eval.REV1_PROTOCOL, gate=dict(passed=True), eval_seeds=list(train_eval.REV1_SEEDS),
               no_reselection=True, preflight_sha256=cfg["original_preflight"]["sha256"])
    rep.update(over)
    p = tmp_path / "preflight_rev1.json"
    p.write_text(json.dumps(rep))
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def _agent_cmd(tmp_path, *extra):
    return ["--agent", "lstm_scratch", "--train-steps", "20", "--eval-episodes", "0", "--log-every", "20",
            "--batch-size", "8", "--checkpoint-every", "0", *extra]


def test_agents_refuse_without_rev1_pass_or_amendment(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    with pytest.raises(SystemExit, match="refusing to train agents"):
        train_eval.main(_agent_cmd(tmp_path))
    art, sha = _passing_artifact(tmp_path)
    with pytest.raises(SystemExit, match="sha256 does not match"):
        train_eval.main(_agent_cmd(tmp_path, "--rev1-artifact", art, "--rev1-sha256", "0" * 64))
    (tmp_path / "f").mkdir()
    failing, fsha = _passing_artifact(tmp_path / "f", gate=dict(passed=False))
    with pytest.raises(SystemExit, match="not a passing"):
        train_eval.main(_agent_cmd(tmp_path, "--rev1-artifact", failing, "--rev1-sha256", fsha))
    with pytest.raises(SystemExit, match="amendment-id"):
        train_eval.main(_agent_cmd(tmp_path, "--amendment-id", "nonsense"))
    train_eval.main(_agent_cmd(tmp_path, "--amendment-id", "R-20261017-099"))
    rec = json.load(open(tmp_path / "run" / "rev1_authorization.json"))
    assert rec["kind"] == "approved_amendment" and rec["id"] == "R-20261017-099"
    assert rec["lab_db_record"]["state"] == "approved"      # checked in the (test) lab database, not syntax only
    art, sha = _passing_artifact(tmp_path)
    train_eval.main(_agent_cmd(tmp_path, "--rev1-artifact", art, "--rev1-sha256", sha))
    assert json.load(open(tmp_path / "run" / "rev1_authorization.json"))["sha256"] == sha
