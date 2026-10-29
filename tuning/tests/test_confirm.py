"""test10 execution guard (tuning/legacy/confirm_guard.py) and locked-checkpoint export (sum_c10.py --export-locked);
base review of ef0e770, findings 1 and 4. Nothing is trained: run_c5 / the baseline evaluator are replaced by stubs."""
import json
import os

import pytest
import torch

from tuning.legacy import c5_protocol as C5
from tuning.legacy import confirm_guard as CG
from tuning.legacy import orig_baselines as OB
from tuning.legacy import run_legacy as RL

LAB = os.path.expanduser("~/qrl-lab/.lab/tuning")
MAKE_C18 = os.path.join(LAB, "make_c18.py")
lab = pytest.mark.skipif(not os.path.exists(MAKE_C18), reason="lab generator not present")


def fresh_seeds(n=5, skip=(), start=500):
    """n training seeds not used by any *_entries.json of the live lab (the lab grows every cycle: 60-74 are taken
    since c18/c22), below the eval blocks (>= 1000)."""
    used = CG.tuning_seeds(LAB) | set(skip)
    out = [s for s in range(start, 1000) if s not in used][:n]
    assert len(out) == n
    return out


FRESH = fresh_seeds()
SEEDS = ",".join(map(str, FRESH))


def _load(name):
    import importlib.util
    import sys
    if LAB not in sys.path:
        sys.path.insert(0, LAB)
    spec = importlib.util.spec_from_file_location(name, os.path.join(LAB, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "confirm_ledger.json"
    monkeypatch.setenv("QRL_CONFIRM_LEDGER", str(p))
    monkeypatch.delenv(CG.ENV, raising=False)
    return p


@pytest.fixture
def stub_c5(monkeypatch):
    calls = []
    monkeypatch.setattr(C5, "run_c5", lambda *x, **kw: calls.append(x) or {
        "arm": "x", "cpu_s": 0.0, "train_cpu_s_per_1k_ticks": 0.0, "checkpoint_results": {}, "eval_splits": []})
    return calls


def _confirm(tmp_path, tag="c18zc", seeds=SEEDS, extra=()):
    m = _load("make_c18")
    out = tmp_path / f"{tag}_entries.json"
    ents, info = m.main(["--arch-search", "--arches", "a2", "--transfer-v2", "a1h", "--confirm", "--seeds", seeds,
                         "--tag", tag, "--no-live", "--out", str(out)] + list(extra))
    return ents, info


def test_test10_refused_without_manifest(ledger, stub_c5):
    base = ["--agent", "lstm_scratch", "--agent-variant", "a1", "--total-ticks", "100", "--checkpoints", "100",
            "--env-family", "orig", "--orig-variant", "spike300_d5", "--seed", "60"]
    with pytest.raises(SystemExit, match="refusing test10"):
        RL.main(base + ["--eval-split", "val,test10"])
    with pytest.raises(SystemExit, match="refusing test10"):
        RL.main(base + ["--eval-split", "test10"])
    with pytest.raises(SystemExit, match="refusing test10"):            # flag without test10
        RL.main(base + ["--eval-split", "val,test9", "--confirm-manifest", "/nonexistent.json"])
    RL.main(base + ["--eval-split", "val,test9"])                       # other splits unaffected
    assert len(stub_c5) == 1
    with pytest.raises(SystemExit, match="refusing test10"):
        OB.main(["--orig-variant", "spike300_d5", "--report-splits", "test10", "--out", "/tmp/never.json"])


@lab
def test_confirm_manifest_single_use(tmp_path, ledger, stub_c5, monkeypatch):
    s10 = _load("sum_c10")
    ents, info = _confirm(tmp_path)
    man = json.load(open(info["confirm_manifest"]))
    assert info["confirm_manifest_sha256"] == CG.sha256_file(info["confirm_manifest"])
    assert man["training_seeds"] == FRESH and man["locked"]["checkpoint_rule"] == CG.LOCKED_RULE
    assert man["locked"]["checkpoint_rule"] == s10.LOCKED_RULE
    arms = {s10.entry_arm(s10.cmd_args(e["cmd"])) for e in ents if "--agent" in e["cmd"]}
    assert {"dinos_a2", "mlp_a2", "transfer_a1h", "transfer_mlp_a1h", "lstm_scratch", "mlp_hist10", "derp",
            "derp_hist10", "transfer_orig_orig_d0"} == arms == set(man["locked"]["arms"])
    assert man["locked"]["architectures"]["mlp_a2"] == "m156"
    assert sum("baselines" in e["name"] for e in ents) == 1
    assert all(e["cmd"][-2:] == ["--confirm-manifest", info["confirm_manifest"]] for e in ents)
    led = json.load(open(ledger))
    assert led["manifests"][info["confirm_manifest_sha256"]]["status"] == "frozen"
    agent = [e for e in ents if "--agent" in e["cmd"]]
    RL.main(agent[0]["cmd"][2:])                               # first use runs
    assert len(stub_c5) == 1
    with pytest.raises(SystemExit, match="already started"):   # same entry again: refused
        RL.main(agent[0]["cmd"][2:])
    RL.main(agent[1]["cmd"][2:])                               # same batch, another entry: allowed
    led = json.load(open(ledger))
    rec = led["manifests"][info["confirm_manifest_sha256"]]
    assert rec["status"] == "in_use" and set(rec["started"]) == {agent[0]["name"], agent[1]["name"]}
    c = list(agent[2]["cmd"][2:])                              # a command that is not a frozen entry
    c[c.index("--seed") + 1] = "999" if FRESH[-1] != 999 else "998"
    with pytest.raises(SystemExit, match="not exactly one entry"):
        RL.main(c)
    monkeypatch.setenv(CG.ENV, "/elsewhere.json")              # env and flag disagree
    with pytest.raises(SystemExit, match="different files"):
        RL.main(agent[2]["cmd"][2:])
    monkeypatch.delenv(CG.ENV)
    with pytest.raises(SystemExit, match="already used"):      # no second confirmation manifest
        _confirm(tmp_path, tag="c18zd", seeds=",".join(map(str, fresh_seeds(skip=FRESH))))
    assert len(stub_c5) == 2


@lab
def test_confirm_env_var_and_tamper(tmp_path, ledger, stub_c5, monkeypatch):
    ents, info = _confirm(tmp_path)
    agent = [e for e in ents if "--agent" in e["cmd"]]
    # the env var alone (entry command without the flag is not a frozen entry -> refused)
    monkeypatch.setenv(CG.ENV, info["confirm_manifest"])
    with pytest.raises(SystemExit, match="not exactly one entry"):
        RL.main(agent[0]["cmd"][2:-2])
    RL.main(agent[0]["cmd"][2:])                               # env + identical flag: fine
    monkeypatch.delenv(CG.ENV)
    # editing the entries file after freezing breaks the manifest
    ef = info["out"]
    js = json.load(open(ef))
    js.append(dict(js[-1], name="extra"))
    json.dump(js, open(ef, "w"))
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        RL.main(agent[1]["cmd"][2:])
    # a manifest not recorded in the ledger is refused
    os.remove(ledger)
    json.dump(js[:-1], open(ef, "w"), indent=1)
    with pytest.raises(SystemExit):
        RL.main(agent[1]["cmd"][2:])
    assert len(stub_c5) == 1


@lab
def test_confirm_refuses_tuning_seeds(tmp_path, ledger):
    with pytest.raises(SystemExit):
        _confirm(tmp_path, seeds="53," + ",".join(map(str, FRESH[:4])))   # 53 = c18 tuning seed
    with pytest.raises(SystemExit):
        _confirm(tmp_path, seeds="0," + ",".join(map(str, FRESH[:4])))    # earlier cycles too
    assert {53, 54, 0} <= CG.tuning_seeds(LAB)
    assert not os.path.exists(ledger)


@lab
def test_confirm_baselines_guard(tmp_path, ledger, monkeypatch):
    ents, info = _confirm(tmp_path)
    b = [e for e in ents if "--agent" not in e["cmd"]][0]
    seen = []
    monkeypatch.setattr(OB, "policies", lambda v: seen.append(v) or (_ for _ in ()).throw(RuntimeError("stub")))
    with pytest.raises(RuntimeError, match="stub"):            # guard passed, then the stub stops the work
        OB.main(b["cmd"][2:])
    with pytest.raises(SystemExit, match="already started"):
        OB.main(b["cmd"][2:])
    assert seen == ["spike300_d5"]


@lab
def test_export_locked(tmp_path):
    """Across-seed checkpoint exported for every seed; per-seed .selected.pt kept and labelled."""
    s10 = _load("sum_c10")
    m = _load("make_c18")
    ef = tmp_path / "c18zl_entries.json"
    ents, _ = m.main(["--arch-search", "--arches", "a1", "--existing", "none", "--seeds", "53,54", "--tag", "c18zl",
                      "--no-live", "--out", str(ef)])
    ents = [e for e in ents if "--agent" in e["cmd"] and "a1" == e["cmd"][e["cmd"].index("--agent-variant") + 1]]
    json.dump(ents, open(ef, "w"))
    d = tmp_path / "r"
    d.mkdir()
    vals = {53: {1000: 10, 2000: 20, 5000: 0, 10000: 0, 20000: 0},    # per-seed best 2000
            54: {1000: 30, 2000: 0, 5000: 0, 10000: 0, 20000: 0}}     # per-seed best 1000; across-seed 1000 (20 > 10)
    for e in ents:
        a = s10.cmd_args(e["cmd"])
        s = int(a["--seed"])
        cr = {str(c): {sp: {"mean_ret": v, "mean_pods": 1.0,
                            "episodes": [{"seed": x, "ret": v, "frac_const_pods": 0.0} for x in s10.SPLIT_SEEDS[sp]]}
                       for sp in ("val", "test9")} for c, v in vals[s].items()}
        sel = max(sorted(vals[s]), key=lambda c: (vals[s][c], -c))
        r = dict(protocol="c5", arm="dinos_a1", seed=s, env_family="orig", orig_variant="spike300_d5",
                 agent="lstm_scratch", agent_variant="a1", obs_scale="on", eps_schedule="linear10k", reward_div=10.0,
                 checkpoints=sorted(vals[s]), eval_splits=["val", "test9"], startup_delay=5, state_size=7,
                 total_ticks=20000, eval_seeds={sp: s10.SPLIT_SEEDS[sp] for sp in ("val", "test9")},
                 checkpoint_results=cr, saved_weights=dict(selected_checkpoint=sel, selection_scope="per_seed"))
        root = d / os.path.splitext(os.path.basename(a["--out"]))[0]
        json.dump(r, open(f"{root}.json", "w"))
        for c in vals[s]:
            torch.save({"w": torch.tensor([float(c + s)])}, f"{root}.ck{c}.pt")
        torch.save({"w": torch.tensor([float(sel + s)])}, f"{root}.selected.pt")
    out = tmp_path / "sum.json"
    args = ["--results", str(d), "--entries", str(ef), "--split", "test9", "--seeds", "53,54", "--out", str(out)]
    s10.main(args + ["--export-locked"])
    summ = json.load(open(out))["spike300_d5"]
    arm = summ["arms"]["dinos_a1"]
    assert arm["checkpoint"] == 1000 and arm["selection_scope"] == "across_seed"
    assert arm["per_seed_selected_checkpoint"] == {"53": 2000, "54": 1000}
    assert summ["selection"]["target_checkpoint_scope"] == "across_seed"
    for s in (53, 54):
        root = str(d / f"spike300_d5-dinos_a1-t20000-grid-s{s}")
        assert torch.load(f"{root}.locked.pt")["w"].item() == 1000 + s       # the across-seed checkpoint
        note = json.load(open(f"{root}.locked.json"))
        assert note["locked_checkpoint"] == 1000 and note["selection_scope"] == "across_seed"
        assert note["per_seed_equals_locked"] == (s == 54)
    assert torch.load(str(d / "spike300_d5-dinos_a1-t20000-grid-s53.selected.pt"))["w"].item() == 2053   # kept
    s10.main(args + ["--export-locked"])                       # idempotent; .locked.json files are skipped on reload
    torch.save({"w": torch.tensor([0.0])}, str(d / "spike300_d5-dinos_a1-t20000-grid-s53.locked.pt"))
    with pytest.raises(SystemExit, match="different bytes"):
        s10.main(args + ["--export-locked"])
