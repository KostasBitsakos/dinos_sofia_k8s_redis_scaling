"""Cycle 22: second held-out confirmation split test11 (12000-12019) in confirm_guard / run_legacy / orig_baselines.
Per-split single use; test10 stays closed; legacy ledger records without 'split' count as test10. Nothing is trained:
run_c5 and the baseline policies are stubbed. Manifests are synthetic (no dependency on the lab make_c18.py)."""
import json
import os

import pytest

from tuning.legacy import c5_protocol as C5
from tuning.legacy import confirm_guard as CG
from tuning.legacy import orig_baselines as OB
from tuning.legacy import run_legacy as RL

SEEDS = [65, 66, 67, 68, 69]
BASE = ["--agent", "lstm_scratch", "--agent-variant", "a1", "--total-ticks", "100", "--checkpoints", "100",
        "--env-family", "orig", "--orig-variant", "spike300_d5"]


@pytest.fixture
def lab(tmp_path, monkeypatch):
    d = tmp_path / "lab"
    d.mkdir()
    monkeypatch.setenv("QRL_CONFIRM_LAB_DIR", str(d))
    monkeypatch.setenv("QRL_CONFIRM_LEDGER", str(d / "confirm_ledger.json"))
    monkeypatch.delenv(CG.ENV, raising=False)
    return d


@pytest.fixture
def stub_c5(monkeypatch):
    calls = []
    monkeypatch.setattr(C5, "run_c5", lambda *x, **kw: calls.append(x) or {
        "arm": "x", "cpu_s": 0.0, "train_cpu_s_per_1k_ticks": 0.0, "checkpoint_results": {}, "eval_splits": []})
    return calls


def make_manifest(lab, split, tag, seeds=SEEDS):
    """Entries + manifest for `split` as make_c18.py --confirm writes them; returns (manifest path, agent argvs,
    baseline argv)."""
    man = str(lab / f"{tag}_confirm_manifest.json")
    ents = [dict(name=f"{tag}-s{s}", cmd=["py", "tuning/legacy/run_legacy.py"] + BASE +
                 ["--seed", str(s), "--eval-split", f"val,{split}", "--confirm-manifest", man]) for s in seeds]
    ents.append(dict(name=f"{tag}-baselines", cmd=["py", "tuning/legacy/orig_baselines.py", "--orig-variant",
                                                   "spike300_d5", "--report-splits", split, "--seed", "0", "--out",
                                                   str(lab / f"{tag}_b.json"), "--confirm-manifest", man]))
    ef = lab / f"{tag}_entries.json"
    json.dump(ents, open(ef, "w"), indent=1)
    json.dump(dict(kind=CG.KIND, tag=tag, report=split, eval_splits=f"val,{split}", entries=str(ef),
                   entries_sha256=CG.sha256_file(ef), training_seeds=list(seeds),
                   locked=dict(arms=["dinos_a1"], architectures={"dinos_a1": "a1"}, checkpoints=[100],
                               checkpoint_rule=CG.LOCKED_RULE)), open(man, "w"), indent=1)
    return man, [e["cmd"][2:] for e in ents[:-1]], ents[-1]["cmd"][2:]


def test_splits_disjoint():
    assert RL.SPLITS["test11"] == tuple(range(12000, 12020)) == RL.TEST11_SEEDS
    assert max(RL.TEST11_SEEDS) < 20000
    items = list(RL.SPLITS.items()) + [("test3", tuple(range(4000, 4020)))]      # test3 lives in c7_oracles.py
    for i, (k, v) in enumerate(items):
        assert len(set(v)) == len(v), k
        for k2, v2 in items[i + 1:]:
            assert not set(v) & set(v2), (k, k2)
    assert RL.parse_splits("val,test11") == ("val", "test11")
    with pytest.raises(SystemExit):
        RL.check_seeds(12007, "fair", ("val", "test11"))
    assert CG.CONFIRM_SPLITS == ("test10", "test11")


def test_test11_refused_without_manifest(lab, stub_c5):
    b = BASE + ["--seed", "65"]
    for sp in ("val,test11", "test11"):
        with pytest.raises(SystemExit, match="refusing test11"):
            RL.main(b + ["--eval-split", sp])
    with pytest.raises(SystemExit, match="refusing test11"):
        OB.main(["--orig-variant", "spike300_d5", "--report-splits", "test11", "--out", "/tmp/never.json"])
    with pytest.raises(SystemExit, match="one confirmation split per process"):
        RL.main(b + ["--eval-split", "val,test10,test11"])
    RL.main(b + ["--eval-split", "val,test9"])
    assert len(stub_c5) == 1


def test_test11_manifest_runs_once_and_is_split_bound(lab, stub_c5):
    m11, agents11, bl11 = make_manifest(lab, "test11", "c22c")
    sha11 = CG.freeze(m11)
    led = json.load(open(lab / "confirm_ledger.json"))
    assert led["manifests"][sha11]["split"] == "test11" and led["manifests"][sha11]["status"] == "frozen"
    RL.main(agents11[0])
    assert len(stub_c5) == 1
    with pytest.raises(SystemExit, match="already started"):
        RL.main(agents11[0])
    RL.main(agents11[1])                                    # same manifest, another entry
    # the test11 manifest cannot authorise a test10 run (same argv with the split swapped)
    a10 = [x if x != "val,test11" else "val,test10" for x in agents11[2]]
    with pytest.raises(SystemExit, match="refusing test10: manifest is for test11"):
        RL.main(a10)
    # no new test11 manifest once test11 is in use
    m11b, _, _ = make_manifest(lab, "test11", "c22d", seeds=[75, 76, 77, 78, 79])
    with pytest.raises(SystemExit, match="refusing test11: test11 was already used"):
        CG.freeze(m11b)
    assert len(stub_c5) == 2
    seen = []
    OB_pol = OB.policies
    try:
        OB.policies = lambda v: seen.append(v) or (_ for _ in ()).throw(RuntimeError("stub"))
        with pytest.raises(RuntimeError, match="stub"):
            OB.main(bl11)
    finally:
        OB.policies = OB_pol
    assert seen == ["spike300_d5"]


def test_test10_manifest_cannot_authorise_test11(lab, stub_c5):
    m10, agents10, _ = make_manifest(lab, "test10", "c22t")
    CG.freeze(m10)
    a11 = [x if x != "val,test10" else "val,test11" for x in agents10[0]]
    with pytest.raises(SystemExit, match="refusing test11: manifest is for test10"):
        RL.main(a11)
    # a ledger record bound to test10 is refused for test11 even if the manifest file claims test11
    led = json.load(open(lab / "confirm_ledger.json"))
    m11, agents11, _ = make_manifest(lab, "test11", "c22u", seeds=[85, 86, 87, 88, 89])
    sha = CG.sha256_file(m11)
    led["manifests"][sha] = dict(path=m11, status="frozen", started={})      # legacy record: no split -> test10
    json.dump(led, open(lab / "confirm_ledger.json", "w"))
    with pytest.raises(SystemExit, match="ledger records manifest .* for test10, not test11"):
        RL.main(agents11[0])
    assert not stub_c5


def test_legacy_ledger_closes_test10_only(lab, stub_c5):
    """The real ledger shape: 7f9face1-like record without 'split', status in_use -> test10 closed, test11 open."""
    old = "7f9face1" + "0" * 56
    json.dump({"manifests": {old: dict(path="/x.json", tag="c18c", training_seeds=[60, 61, 62, 63, 64],
                                       status="in_use", started={"c18c-x": "t"}),
                             "ecd47c01" + "0" * 56: dict(path="/y.json", status="superseded")}},
              open(lab / "confirm_ledger.json", "w"))
    m10, agents10, _ = make_manifest(lab, "test10", "c22v")
    with pytest.raises(SystemExit, match="refusing test10: test10 was already used by manifest 7f9face1"):
        CG.freeze(m10)
    m11, agents11, _ = make_manifest(lab, "test11", "c22w", seeds=[95, 96, 97, 98, 99])
    sha11 = CG.freeze(m11)
    RL.main(agents11[0])
    assert len(stub_c5) == 1
    led = json.load(open(lab / "confirm_ledger.json"))
    assert led["manifests"][old]["status"] == "in_use"            # untouched
    assert "split" not in led["manifests"][old]
    assert led["manifests"][sha11]["status"] == "in_use"


def test_freeze_supersedes_only_same_split(lab):
    m10, _, _ = make_manifest(lab, "test10", "c22x")
    m11, _, _ = make_manifest(lab, "test11", "c22y", seeds=[105, 106, 107, 108, 109])
    s10 = CG.freeze(m10)
    s11 = CG.freeze(m11)
    m11b, _, _ = make_manifest(lab, "test11", "c22z", seeds=[115, 116, 117, 118, 119])
    s11b = CG.freeze(m11b)
    man = json.load(open(lab / "confirm_ledger.json"))["manifests"]
    assert man[s10]["status"] == "frozen" and man[s11]["status"] == "superseded" and man[s11b]["status"] == "frozen"
    assert os.path.exists(m10)


def test_manifest_split_must_be_confirmation_split(lab):
    m, _, _ = make_manifest(lab, "test9", "c22q")
    with pytest.raises(SystemExit, match="not one of"):
        CG.freeze(m)


# ------------------------------------------------ evaluator boundaries (Codex review of a191edb: direct calls)
C5_KW = dict(total_ticks=100, checkpoints=[100], agent_variant="a1", env_family="orig", orig_variant="spike300_d5")


@pytest.mark.parametrize("split", ["test10", "test11"])
def test_direct_run_c5_and_run_refused(lab, split):
    with pytest.raises(SystemExit, match=f"refusing {split}: c5_protocol.run_c5: evaluating {split} needs a grant"):
        C5.run_c5("lstm_scratch", 65, eval_splits=("val", split), **C5_KW)
    with pytest.raises(SystemExit, match=f"refusing {split}: run_legacy.run: evaluating {split} needs a grant"):
        RL.run("derp", 1000, 65, "fair", eval_splits=("val", split))
    with pytest.raises(SystemExit, match=f"refusing {split}: {split} .* one-time"):          # CLI guard
        OB.main(["--orig-variant", "spike300_d5", "--report-splits", split, "--out", "/tmp/never.json"])
    real = CG.require
    try:                                       # even with the CLI guard bypassed, the evaluator boundary refuses
        CG.require = lambda *a, **k: None
        with pytest.raises(SystemExit, match=f"refusing {split}: orig_baselines.main"):
            OB.main(["--orig-variant", "spike300_d5", "--report-splits", split, "--out", "/tmp/never.json"])
    finally:
        CG.require = real
    from tuning.legacy import coupled_baselines as CB
    with pytest.raises(SystemExit, match=f"refusing {split}: coupled_baselines.main"):
        CB.main(["--report-splits", split, "--out", "/tmp/never.json"])


def test_direct_both_splits_refused(lab):
    with pytest.raises(SystemExit, match="one confirmation split per process"):
        C5.run_c5("lstm_scratch", 65, eval_splits=("val", "test10", "test11"), **C5_KW)
    with pytest.raises(SystemExit, match="one confirmation split per process"):
        RL.run("derp", 1000, 65, "fair", eval_splits=("val", "test10", "test11"))
    assert CG.authorise(("val", "test9"), "x") is None and CG.authorise(("fair",), "x") is None   # unchanged


def test_cli_grant_reaches_run_c5_once(lab, monkeypatch):
    """Real run_c5 (no stub): the CLI grant passes the boundary once; Telemetry is stubbed to stop before training.
    A later direct call in the same process (grant consumed) and a forged grant (ledger mismatch) are refused."""
    m11, agents11, _ = make_manifest(lab, "test11", "c22g")
    CG.freeze(m11)
    monkeypatch.setattr(RL, "Telemetry", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        RL.main(agents11[0])
    assert not CG._GRANTS
    with pytest.raises(SystemExit, match="needs a grant"):
        C5.run_c5("lstm_scratch", 66, eval_splits=("val", "test11"), **C5_KW)
    CG._GRANTS["test11"] = dict(sha256="0" * 64, entry="forged", split="test11")
    with pytest.raises(SystemExit, match="ledger does not record entry forged"):
        C5.run_c5("lstm_scratch", 66, eval_splits=("val", "test11"), **C5_KW)
    # a test11 grant does not open test10
    RL_note = CG.require(("val", "test11"), "run_legacy.py", agents11[1], m11, 66)
    assert RL_note["split"] == "test11"
    with pytest.raises(SystemExit, match="refusing test10: c5_protocol.run_c5"):
        C5.run_c5("lstm_scratch", 66, eval_splits=("val", "test10"), **C5_KW)
