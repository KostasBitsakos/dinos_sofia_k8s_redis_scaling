"""Exploratory W0 (PLAN 8.1 amendment R-20261017-014): registration, seed guards, amendment check, smokes, analysis.
Smokes use development seeds 1000-1004 only; the exploratory block is refused by the guards and by conftest."""
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import train_eval  # noqa: E402
from adapter import DinosSimV2, SimParams  # noqa: E402

AMEND = "R-20261017-014"


def _tools():
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "tools")))


# ---------- registration
def test_registration_consistent():
    from workloads import SEED_RANGES
    import sensitivity
    from policies import HPA_TARGET_GRID
    cfg = train_eval.load_explore()
    assert cfg["amendment_id"] == AMEND and cfg["protocol"] == train_eval.EXPLORE_PROTOCOL
    ev = set(cfg["eval_seeds"])
    assert ev == set(train_eval.EXPLORE_SEEDS) == set(range(1300, 1320)) and len(cfg["eval_seeds"]) == 20
    assert ev <= set(SEED_RANGES["tune"]) and not ev & set(train_eval.TUNE_SEEDS)
    assert not ev & (set(train_eval.REV1_SEEDS) | set(train_eval.REV1_BURNED) | set(train_eval.BURNED)
                     | set(SEED_RANGES["test"]) | set(SEED_RANGES["train"]))
    assert cfg["configs"] == sensitivity.names() and cfg["sensitivity"] == "all"
    assert cfg["hpa"]["target_grid"] == list(HPA_TARGET_GRID)
    sel = json.load(open(os.path.join(HERE, "..", "hpa_selection_dev.json")))
    assert sel["selected_target_pct"] == cfg["hpa"]["target_pct"] == 40 and sel["interior"]
    assert sel["seeds"] == list(train_eval.TUNE_SEEDS) and [t for t, _ in sel["table"]] == list(HPA_TARGET_GRID)
    best = max(sel["table"], key=lambda x: (x[1], -x[0]))[0]
    assert best == 40
    assert cfg["comparators"]["primary"] == ["static_ramp_best", "fixed_best"]
    assert set(sum(cfg["comparators"].values(), [])) == set(train_eval.EXPLORE_POLICIES)
    prereg = open(os.path.join(HERE, "..", "PREREG_explore_w0.md")).read()
    assert cfg["plan_amendment_quote"] in prereg.replace("\n> ", " ")
    assert "asymmetry" in prereg and "1300-1319" in prereg and "LAST" in prereg
    assert "clairvoyant +-1 reference" in prereg and "CONDITIONAL on the three trained models" in prereg
    assert "Request = limit\nis an ASSUMPTION" in prereg or "Request = limit is an ASSUMPTION" in prereg.replace("\n  ", " ")
    assert cfg["limits"]["wall_clock_hours_per_entry"] == 2.0


def test_plan_quote_matches_plan():
    plan = open("/home/ubuntu/qrl-lab/docs/PLAN.md").read().replace("\n", " ")
    q = train_eval.load_explore()["plan_amendment_quote"]
    assert q in plan


def test_explore_seeds_never_named_elsewhere():
    root = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
    pat = re.compile(r"\b13[01][0-9]\b")
    # make_report_figs.py only prints seed ranges as prose / labels of stored-result figures (checked below: it
    # imports neither the simulator nor the adapter, so it cannot evaluate any seed)
    allowed = {"train_eval.py", "explore_w0.json", "PREREG_explore_w0.md", "conftest.py", "test_explore_w0.py",
               "make_report_figs.py"}
    src = open(os.path.join(root, "experiments", "dinos_simv2", "make_report_figs.py")).read()
    assert not re.search(r"^\s*(import|from)\s+(env|adapter|train_eval|sim_v2|policies|workloads)\b", src, re.M)
    for d in ("sim_v2", "experiments/dinos_simv2"):
        for dp, _, fs in os.walk(os.path.join(root, d)):
            for fn in fs:
                if fn.endswith((".py", ".json", ".md")) and fn not in allowed and "spec_" not in fn:
                    for i, line in enumerate(open(os.path.join(dp, fn), errors="ignore")):
                        if pat.search(line):
                            raise AssertionError(f"{fn}:{i+1} mentions an exploratory seed: {line.strip()}")


def test_spec_explore_valid_and_matches_generator():
    import make_spec
    _tools()
    from labdb import validate_spec
    spec = make_spec.build_explore(allow_dirty=True, legacy=True)
    assert validate_spec(spec)
    assert spec["limits"]["wall_clock_hours"] == 2.0 and spec["max_parallel"] == 5 and len(spec["entries"]) == 10
    for e in spec["entries"]:
        c = e["cmd"]
        assert c[c.index("--freeze") + 1] == "spec_explore_w0.json" and "--unfrozen" not in c
        assert c[c.index("--amendment-id") + 1] == AMEND and c[c.index("--protocol") + 1] == "explore_w0"
        assert c[c.index("--sensitivity") + 1] == "all" and "--explore-eval-seeds" not in c
    agents = [e for e in spec["entries"] if "explore_baselines" not in e["cmd"]]
    assert len(agents) == 9 and all(e["cmd"][e["cmd"].index("--train-steps") + 1] == "100000" for e in agents)
    assert sorted(e["seed"] for e in agents) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    for f in ("explore_w0.json", "PREREG_explore_w0.md", "analyze_explore_w0.py", "hpa_selection_dev.json",
              "../../sim_v2/policies.py", "../../sim_v2/env.py"):
        assert os.path.normpath(f) in spec["frozen_sha256"]
    # spec_explore_w0.json is HISTORICAL (approved as R-20261017-015, never regenerated): entries / limits must equal
    # the generator's, and its manifest must be the content of the frozen files AT THE COMMIT it was frozen at.
    with open(os.path.join(HERE, "..", "spec_explore_w0.json")) as f:
        on_disk = json.load(f)
    drop = ("frozen_git_commit", "frozen_sha256")
    assert {k: v for k, v in on_disk.items() if k not in drop} == {k: v for k, v in spec.items() if k not in drop}
    import hashlib
    import subprocess
    commit = on_disk["frozen_git_commit"]
    assert set(on_disk["frozen_sha256"]) <= set(spec["frozen_sha256"]) | {"../../sim_v2/reward_dinos.json"}
    for rel, want in on_disk["frozen_sha256"].items():
        path = os.path.relpath(os.path.normpath(os.path.join(HERE, "..", rel)), subprocess.run(
            ["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
        blob = subprocess.run(["git", "-C", HERE, "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
        assert hashlib.sha256(blob).hexdigest() == want, rel


def test_new_specs_require_explicit_reward_mode():
    import make_spec
    _tools()
    from labdb import validate_spec
    with pytest.raises(SystemExit):
        make_spec.build_explore(allow_dirty=True)
    with pytest.raises(SystemExit):
        make_spec.build_explore(allow_dirty=True, reward_mode="reward")
    with pytest.raises(SystemExit):
        make_spec.build_explore(allow_dirty=True, reward_mode="dinos", legacy=True)
    for rm in ("cost", "dinos"):
        spec = make_spec.build_explore(allow_dirty=True, reward_mode=rm)
        assert validate_spec(spec) and spec["reward_mode"] == rm and spec["learn_scale"] == 0.02
        for e in spec["entries"]:
            c = e["cmd"]
            assert c[c.index("--reward-mode") + 1] == rm and float(c[c.index("--learn-scale") + 1]) == 0.02
    legacy = make_spec.build_explore(allow_dirty=True, legacy=True)
    assert "reward_mode" not in legacy and all("--reward-mode" not in e["cmd"] for e in legacy["entries"])


# ---------- seed guards
def test_explore_block_guard():
    for bad in ((1300,), (1319, 1000)):
        with pytest.raises(RuntimeError, match="frozen run"):
            train_eval.guard_explore_block(bad, frozen=False)
    assert train_eval.guard_explore_block(train_eval.EXPLORE_SEEDS, frozen=True) == train_eval.EXPLORE_SEEDS
    for bad in ((1200,), (1105,)):
        for frozen in (False, True):
            with pytest.raises(RuntimeError, match="gate seed block|burned"):
                train_eval.guard_explore_block(bad, frozen)
    for bad in ((20000,), (10001,), (5,), (1005,)):
        for frozen in (False, True):
            with pytest.raises(RuntimeError, match="neither"):
                train_eval.guard_explore_block(bad, frozen)
    assert train_eval.guard_explore_block((1000, 1004), frozen=False) == (1000, 1004)
    with pytest.raises(AssertionError, match="never evaluate"):
        DinosSimV2("W0", SimParams()).reset(1310)


def _explore_cmd(*extra, agent="lstm_scratch"):
    return ["--agent", agent, "--train-steps", "20", "--log-every", "20", "--batch-size", "8",
            "--checkpoint-every", "0", "--protocol", "explore_w0", "--sensitivity", "default", *extra]


def test_explore_refuses_block_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    with pytest.raises(RuntimeError, match="frozen run"):      # unfrozen run without override would use the block
        train_eval.main(_explore_cmd("--amendment-id", AMEND))
    with pytest.raises(RuntimeError, match="frozen run"):
        train_eval.main(_explore_cmd("--amendment-id", AMEND, "--explore-eval-seeds", "1000,1301"))
    with pytest.raises(RuntimeError, match="gate seed block"):
        train_eval.main(_explore_cmd("--amendment-id", AMEND, "--explore-eval-seeds", "1203"))
    with pytest.raises(RuntimeError, match="neither"):
        train_eval.main(_explore_cmd("--amendment-id", AMEND, "--explore-eval-seeds", "20001"))


# ---------- amendment check (lab database, read-only)
def test_amendment_must_be_approved_in_lab_db(tmp_path, monkeypatch, _lab_db):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    base = ["--agent", "lstm_scratch", "--train-steps", "20", "--eval-episodes", "0", "--log-every", "20",
            "--batch-size", "8", "--checkpoint-every", "0"]
    with pytest.raises(SystemExit, match="not a request"):
        train_eval.main(base + ["--amendment-id", "R-20990101-001"])
    with pytest.raises(SystemExit, match="state 'open'"):
        train_eval.main(base + ["--amendment-id", "R-TEST-OPEN"])
    before = _lab_db.read_bytes()
    train_eval.main(base + ["--amendment-id", "R-TEST-000"])
    assert _lab_db.read_bytes() == before                          # opened read-only, nothing written
    rec = json.load(open(tmp_path / "run" / "rev1_authorization.json"))
    assert rec["lab_db_record"]["state"] == "approved" and rec["lab_db_record"]["db"] == str(_lab_db)
    # the exploratory protocol needs exactly the registered amendment
    with pytest.raises(SystemExit, match=AMEND):
        train_eval.main(_explore_cmd("--amendment-id", "R-TEST-000", "--explore-eval-seeds", "1000"))
    monkeypatch.setenv("QRL_LAB_DB", str(tmp_path / "nope.db"))
    with pytest.raises(SystemExit, match="not found"):
        train_eval.main(base + ["--amendment-id", "R-TEST-000"])


def test_frozen_runs_ignore_db_override(monkeypatch, tmp_path):
    monkeypatch.setenv("QRL_LAB_DB", str(tmp_path / "fake.db"))
    assert train_eval.lab_db_path(frozen=True) == "/home/ubuntu/qrl-lab/.lab/lab.db"
    assert train_eval.lab_db_path(frozen=False) == str(tmp_path / "fake.db")


def test_real_amendment_is_approved_read_only():
    """The real lab database (read-only) records R-20261017-014 as approved."""
    path = train_eval.lab_db_path(frozen=True)
    if not os.path.isfile(path):
        pytest.skip("lab database not present on this machine")
    row = train_eval.verify_amendment_approved(AMEND, frozen=True)
    assert row["state"] == "approved" and row["id"] == AMEND


# ---------- smokes on development seeds
def test_explore_agent_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(train_eval, "DinosSimV2",
                        lambda w="W0", p=None, **k: DinosSimV2(w, (p or SimParams()).with_(episode_steps=100), **k))
    train_eval.main(_explore_cmd("--amendment-id", AMEND, "--explore-eval-seeds", "1000,1001"))
    out = json.load(open(tmp_path / "run" / "explore_eval.json"))
    assert out["frozen"] is False and out["eval_seeds"] == [1000, 1001] and out["checkpoint"] == "last"
    rows = out["results"]["lstm_scratch"]["default"]
    assert [r["seed"] for r in rows] == [1000, 1001] and all(np.isfinite(r["ret"]) for r in rows)


def test_explore_comparators_smoke(tmp_path, monkeypatch):
    import policies
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(train_eval, "SimParams", lambda base=None: (base or SimParams()).with_(episode_steps=150))
    monkeypatch.setattr(train_eval, "DinosSimV2",
                        lambda w="W0", p=None, **k: DinosSimV2(w, (p or SimParams()).with_(episode_steps=150), **k))
    orig = policies.tune_hpa
    monkeypatch.setattr(policies, "tune_hpa", lambda P, w, s, grid: orig(P, w, s[:1], grid=(35, 40, 45)))
    out = train_eval.main(["--agent", "explore_baselines", "--protocol", "explore_w0", "--amendment-id", AMEND,
                           "--explore-eval-seeds", "1000,1001", "--sensitivity", "default,cpu_offset_10"])
    rep = json.load(open(tmp_path / "run" / "explore_eval.json"))
    assert rep["policy_index"] == list(train_eval.EXPLORE_POLICIES)
    assert rep["threshold"] == [40.0, 25.0, 0] and rep["fixed_best_c"] == 22 and rep["static_ramp_best_c"] == 22
    assert rep["hpa_target_pct"] == 40 and len(rep["hpa_selection_table"]) == 3
    for n in train_eval.EXPLORE_POLICIES:
        for c in ("default", "cpu_offset_10"):
            assert [r["seed"] for r in rep["results"][n][c]] == [1000, 1001]
    assert set(out["mean_return_default"]) == set(train_eval.EXPLORE_POLICIES)
    # the HPA really scales by more than one pod per step somewhere, the +-1 policies never do
    assert rep["results"]["k8s_hpa"]["default"][0]["mean_pods"] > 0
    with pytest.raises(SystemExit, match="needs --protocol"):
        train_eval.main(["--agent", "explore_baselines", "--amendment-id", AMEND])


def test_explore_comparators_need_amendment(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    with pytest.raises(SystemExit, match="refusing"):
        train_eval.main(["--agent", "explore_baselines", "--protocol", "explore_w0", "--explore-eval-seeds", "1000"])


# ---------- analysis
def _fake_runs(seeds, cfgs, frozen=True, drop=None):
    rng = np.random.default_rng(0)
    cfg = train_eval.load_explore()

    def rows(mu):
        return [dict(seed=s, ret=float(mu + rng.normal(0, 1))) for s in seeds]
    base = dict(protocol=cfg["protocol"], agent="explore_baselines", seed=0, workload="W0", configs=cfgs,
                eval_seeds=seeds, frozen=frozen, hpa_target_pct=40, _dir="b",
                results={n: {c: rows(-100 + 10 * i) for c in cfgs} for i, n in enumerate(train_eval.EXPLORE_POLICIES)})
    runs = [base]
    for a in cfg["training"]["agents"]:
        for s in cfg["training"]["train_seeds"]:
            if drop == (a, s):
                continue
            runs.append(dict(protocol=cfg["protocol"], agent=a, seed=s, workload="W0", configs=cfgs, eval_seeds=seeds,
                             frozen=frozen, _dir=f"{a}{s}", results={a: {c: rows(-90) for c in cfgs}}))
    return runs, cfg


def test_analysis_contrasts_and_refusals():
    import analyze_explore_w0 as an
    cfg0 = train_eval.load_explore()
    runs, cfg = _fake_runs(cfg0["eval_seeds"], cfg0["configs"])
    rep = an.analyze(runs, cfg)
    assert rep["complete"] and rep["per_config"]["default"]["role"] == "primary"
    assert all(b["role"] == "descriptive" for c, b in rep["per_config"].items() if c != "default")
    c = rep["per_config"]["default"]["contrasts"]
    assert set(c) == {"dinos_t", "lstm_scratch", "derp", "threshold", "k8s_hpa"}
    # agents at about -90, static_ramp_best at -100, fixed_best at -90: positive vs ramp, about 0 vs fixed
    assert c["derp"]["vs_static_ramp_best"]["ci_lo"] > 0 and c["derp"]["vs_static_ramp_best"]["n"] == 20
    assert c["derp"]["vs_fixed_best"]["ci_lo"] < 0 < c["derp"]["vs_fixed_best"]["ci_hi"]
    assert len(rep["per_config"]["default"]["per_seed_contrasts"]["derp"]) == 3
    txt = an.table(rep)
    assert "EXPLORATORY" in txt and "clairvoyant +-1 reference" in txt and "conditional on the three trained models" in txt
    assert "upper bound" not in txt.replace("not an upper bound", "")
    with pytest.raises(ValueError, match="unfrozen"):
        an.analyze(_fake_runs(cfg0["eval_seeds"], cfg0["configs"], frozen=False)[0], cfg)
    with pytest.raises(ValueError, match="missing"):
        an.analyze(_fake_runs(cfg0["eval_seeds"], cfg0["configs"], drop=("derp", 1))[0], cfg)
    rep = an.analyze(_fake_runs(cfg0["eval_seeds"], cfg0["configs"], drop=("derp", 1))[0], cfg, allow_incomplete=True)
    assert not rep["complete"] and rep["missing"] == [("derp", 1)]
    with pytest.raises(ValueError, match="seeds differ"):
        an.analyze(_fake_runs(list(train_eval.TUNE_SEEDS), cfg0["configs"])[0], cfg)
