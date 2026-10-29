"""Cycle 10b: matched-mechanics dense control (agent_dense_fixed.py, arm mlp_hist10) and the confirmation tooling
(.lab/tuning/make_c10b.py, sum_c10.py on test7)."""
import importlib
import inspect
import json
import os
import sys

import numpy as np
import pytest
import torch

from tuning.legacy import agent_dense_fixed as DFX
from tuning.legacy import agent_lstm_fixed as FX
from tuning.legacy import c5_protocol as C5
from tuning.legacy import run_legacy as RL

LAB = "/home/ubuntu/qrl-lab/.lab/tuning"
NETS = ("online_model", "target_model", "optimizer", "dense_hidden")
COND = dict(reward_div=10, obs_scale="on", eps_schedule="linear10k", env_family="orig")


def _pair(F=7, sync=250):
    return FX.make_scratch(F, 3, RL.SEQ, sync), DFX.make_scratch(F, 3, RL.SEQ, sync)


# ---------------------------------------------------------------- matched mechanics
@pytest.mark.parametrize("F,sync", [(6, 250), (7, 250), (7, 7)])
def test_every_hyperparameter_shared(F, sync):
    lstm, mlp = _pair(F, sync)
    a = {k: v for k, v in vars(lstm).items() if k not in NETS}
    b = {k: v for k, v in vars(mlp).items() if k not in NETS}
    assert set(a) == set(b)
    for k in a:
        if k == "memory":
            assert type(a[k]) is type(b[k]) and a[k].maxlen == b[k].maxlen == 5000 and len(b[k]) == 0
        elif k == "criterion":
            assert type(a[k]) is type(b[k]) is torch.nn.SmoothL1Loss and a[k].reduction == b[k].reduction
        else:
            assert a[k] == b[k], k
    assert (mlp.gamma, mlp.learning_rate, mlp.epsilon, mlp.epsilon_min, mlp.epsilon_decay, mlp.target_sync) == \
        (0.98, 1e-4, 1.0, 0.05, 0.995, sync)
    assert type(lstm.optimizer) is type(mlp.optimizer) is torch.optim.Adam
    ga, gb = lstm.optimizer.defaults, mlp.optimizer.defaults
    assert ga == gb
    # the optimizer drives the MLP's own parameters, the LSTM nets are gone
    assert {id(p) for p in mlp.optimizer.param_groups[0]["params"]} == {id(p) for p in mlp.online_model.parameters()}
    assert isinstance(mlp.online_model, DFX.FlatMLP) and isinstance(mlp.target_model, DFX.FlatMLP)
    # every training/acting method is inherited, not redefined
    own = set(vars(DFX.DenseFixedAgent)) - {"__init__", "__module__", "__doc__", "__qualname__"}
    assert not own
    for m in ("act", "replay", "ddqn_target", "remember", "_x", "update_target_model", "load_weights"):
        assert getattr(DFX.DenseFixedAgent, m) is getattr(FX.FixedLSTMAgent, m)
    assert inspect.signature(DFX.make_scratch).parameters.keys() == inspect.signature(FX.make_scratch).parameters.keys()


@pytest.mark.parametrize("F", (6, 7))
def test_param_counts_within_20_percent(F):
    lstm, mlp = _pair(F)
    nl, nm = C5.n_params(lstm.online_model), C5.n_params(mlp.online_model)
    assert abs(nm / nl - 1) <= 0.20
    assert (nl, nm) == {6: (334211, 326703), 7: (334723, 332103)}[F]


def test_mlp_init_uses_only_its_own_torch_draws():
    torch.manual_seed(5)
    a = DFX.make_scratch(7, 3, RL.SEQ)
    torch.manual_seed(5)
    ref = DFX.FlatMLP(7, RL.SEQ, DFX.DENSE_HIDDEN, 3)
    for (k1, v1), (k2, v2) in zip(a.online_model.state_dict().items(), ref.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
    assert all(torch.equal(v, a.target_model.state_dict()[k]) for k, v in a.online_model.state_dict().items())


def test_flatten_is_row_major_window():
    m = DFX.FlatMLP(7, RL.SEQ, 16, 3)
    w = np.random.RandomState(0).rand(RL.SEQ, 7)
    x = torch.tensor(w[None], dtype=torch.float32)
    ref = m.net(torch.tensor(w.reshape(1, -1), dtype=torch.float32))
    assert torch.equal(m(x), ref)
    assert m.init_hidden() is None


# ---------------------------------------------------------------- identical window feeding through the protocol
def _small(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test7", (8000,))


def _record(monkeypatch, variant):
    """Run 60 ticks with a deterministic (window-ignoring) policy; return the states seen by act and stored."""
    seen, agents = [], []
    k = [0]

    def act(self, state, verbose=False):
        seen.append(np.array(state, copy=True))
        k[0] += 1
        return (k[0] * 7) % 3

    monkeypatch.setattr(FX.FixedLSTMAgent, "act", act)
    for mod in (FX, DFX):
        orig = mod.make_scratch

        def wrap(*a, _o=orig, **kw):
            ag = _o(*a, **kw)
            agents.append(ag)
            return ag
        monkeypatch.setattr(mod, "make_scratch", wrap)
    r = C5.run_c5("lstm_scratch", 3, total_ticks=60, checkpoints=[60], eval_splits="val", eval_ticks=10,
                  orig_variant="spike300_d5", agent_variant=variant, target_sync=7, **COND)
    monkeypatch.undo()
    _small(monkeypatch)
    return r, seen, list(agents[-1].memory)


def test_window_feeding_identical(monkeypatch):
    _small(monkeypatch)
    torch.set_num_threads(1)
    r1, s1, m1 = _record(monkeypatch, "fixed")
    r2, s2, m2 = _record(monkeypatch, "dense_fixed")
    assert len(s1) == len(s2) >= 60 and all(x.shape == (RL.SEQ, 7) for x in s2)
    assert all(np.array_equal(a, b) for a, b in zip(s1, s2))
    assert len(m1) == len(m2) == 60
    for (a1, b1, c1, d1, e1), (a2, b2, c2, d2, e2) in zip(m1, m2):
        assert np.array_equal(a1, a2) and b1 == b2 and c1 == c2 and np.array_equal(d1, d2) and e1 == e2
    assert r1["arm"] == "lstm_scratch" and r2["arm"] == "mlp_hist10"
    for k in ("epsilon_start", "epsilon_min", "epsilon_decay", "learning_rate", "target_sync_replays",
              "double_dqn_target", "reward_div", "obs_scale", "eps_schedule", "state_size", "replays", "target_syncs",
              "obs_scale_divisors", "target_stream_seed"):
        assert r1[k] == r2[k], k
    assert r2["network"] == "mlp_flat_window" and r2["input_size"] == 70 and r2["dense_hidden"] == 540
    assert "network" not in r1


def test_c5_dense_fixed_smoke(monkeypatch):
    _small(monkeypatch)
    torch.set_num_threads(1)
    r = C5.run_c5("lstm_scratch", 11, total_ticks=40, checkpoints=[20, 40], eval_splits="val,test7", eval_ticks=30,
                  orig_variant="orig_d5", agent_variant="dense_fixed", target_sync=7, **COND)
    assert r["arm"] == "mlp_hist10" and r["n_params"] == 332103 and r["agent_variant"] == "dense_fixed"
    ep = r["checkpoint_results"]["40"]["test7"]["episodes"][0]
    assert abs(ep["served"] + ep["vm_cost"] - ep["ret"]) < 1e-6 and "mean_pending" in ep
    r6 = C5.run_c5("lstm_scratch", 11, total_ticks=20, checkpoints=[20], eval_splits="val", eval_ticks=10,
                   orig_variant="spike300", agent_variant="dense_fixed", **COND)
    assert r6["n_params"] == 326703 and r6["target_sync_replays"] == FX.DEFAULT_TARGET_SYNC


def test_dense_fixed_validation(monkeypatch):
    _small(monkeypatch)
    with pytest.raises(SystemExit):
        C5.run_c5("derp", 3, total_ticks=10, orig_variant="orig_d5", agent_variant="dense_fixed", **COND)
    with pytest.raises(SystemExit):
        C5.run_c5("dinos_t", 3, pretrain_ticks=10, finetune_ticks=10, agent_variant="dense_fixed",
                  orig_variant="orig_d5", pretrain_env_family="orig", pretrain_variant="orig_d0", **COND)
    with pytest.raises(SystemExit):     # coupled path
        C5.run_c5("lstm_scratch", 3, total_ticks=10, agent_variant="dense_fixed")


def test_cli_dense_fixed(monkeypatch, tmp_path):
    _small(monkeypatch)
    out = tmp_path / "r.json"
    res = RL.main(["--agent", "lstm_scratch", "--agent-variant", "dense_fixed", "--env-family", "orig",
                   "--orig-variant", "orig_d5", "--total-ticks", "20", "--eval-split", "val", "--target-sync", "5",
                   "--reward-div", "10", "--obs-scale", "on", "--eps-schedule", "linear10k", "--out", str(out)])
    assert res["arm"] == "mlp_hist10" and json.load(open(out))["target_sync_replays"] == 5


# ---------------------------------------------------------------- splits / tooling
def test_test7_disjoint_from_every_other_split_and_training_seeds():
    t7 = set(RL.SPLITS["test7"])
    assert t7 == set(range(8000, 8020))
    for k, v in RL.SPLITS.items():
        if k != "test7":
            assert not t7 & set(v), k
    for s in range(10, 15):     # c10b training seeds
        RL.check_seeds(s, "fair", ("val", "test7"))


def _lab(name):
    if not os.path.exists(os.path.join(LAB, name + ".py")):
        pytest.skip(f"{LAB}/{name}.py not present")
    if LAB not in sys.path:
        sys.path.insert(0, LAB)
    return importlib.import_module(name)


def test_make_c10b_dry_run(tmp_path, capsys):
    M = _lab("make_c10b")
    out = tmp_path / "e.json"
    M.main(["--out", str(out)])
    info = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    ents = json.load(open(out))
    assert info["targets"] == ["orig_d5", "spike300_d5"] and info["seeds"] == [10, 11, 12, 13, 14]
    assert len(ents) == 2 * 5 * 5 + 2 and info["n_entries"] == len(ents)
    S = _lab("sum_c10")
    arms = {}
    for e in ents:
        a = S.cmd_args(e["cmd"])
        if "orig_baselines.py" in e["cmd"][1]:
            assert a["--report-splits"] == "test7"
            continue
        assert a["--eval-split"] == "val,test7" and int(a["--seed"]) in range(10, 15)
        arm = S.entry_arm(a)
        arms.setdefault(arm, set()).add(a["--orig-variant"])
        assert a["--checkpoints"] == ("1000,10000,20000" if arm.startswith("transfer") else "10000,20000")
    assert set(arms) == {"lstm_scratch", "mlp_hist10", "derp", "derp_hist10", "transfer_orig_orig_d0"}
    names = [os.path.basename(S.cmd_args(e["cmd"])["--out"]) for e in ents]
    assert len(set(names)) == len(names)
    with pytest.raises(SystemExit):
        M.main(["--out", str(out), "--seeds", "8000"])
    with pytest.raises(SystemExit):
        M.main(["--out", str(out), "--arms", "lstm_scratch,bogus"])


def _fake_result(a, rets):
    """A minimal run JSON consistent with entry args a; test7 episode returns rets[ck] (20 values per checkpoint)."""
    sp = a["--eval-split"].split(",")
    seeds = {"val": list(range(1100, 1105)), "test7": list(range(8000, 8020))}
    cks = [int(x) for x in a["--checkpoints"].split(",")]
    cr = {}
    for c in cks:
        cr[str(c)] = {}
        for s_ in sp:
            r = rets[c] if s_ == "test7" else [float(c)] * 5
            cr[str(c)][s_] = dict(mean_ret=float(np.mean(r)), mean_pods=6.0, episodes=[
                dict(seed=t, ret=float(x), frac_const_pods=0.5, mean_pending=0.1) for t, x in zip(seeds[s_], r)])
    arm = "mlp_hist10" if a.get("--agent-variant") == "dense_fixed" else a["--agent"]
    return dict(env_family="orig", orig_variant=a["--orig-variant"], agent=a["--agent"], obs_scale="on",
                eps_schedule="linear10k", arm=arm, agent_variant=a["--agent-variant"], seed=int(a["--seed"]),
                checkpoints=cks, eval_splits=sp, reward_div=10.0, startup_delay=5, state_size=7,
                total_ticks=int(a["--total-ticks"]), eval_seeds={s_: seeds[s_] for s_ in sp}, checkpoint_results=cr)


def test_sum_c10_lstm_minus_mlp_contrast(tmp_path, capsys):
    M, S = _lab("make_c10b"), _lab("sum_c10")
    ent = tmp_path / "e.json"
    M.main(["--out", str(ent), "--targets", "orig_d5", "--arms", "lstm_scratch,mlp_hist10", "--seeds", "10,11"])
    capsys.readouterr()
    res = tmp_path / "res"
    res.mkdir()
    for e in json.load(open(ent)):
        a = S.cmd_args(e["cmd"])
        if "--agent" not in a:
            continue
        off = 100.0 if a["--agent-variant"] == "fixed" else 0.0     # LSTM better by exactly 100 at ck 20000
        rets = {10000: [0.0] * 20, 20000: [off + int(a["--seed"]) + t for t in range(20)]}
        json.dump(_fake_result(a, rets), open(res / os.path.basename(a["--out"]), "w"))
    out = tmp_path / "s.json"
    S.main(["--results", str(res), "--entries", str(ent), "--split", "test7", "--seeds", "10,11", "--out", str(out)])
    r = json.load(open(out))["orig_d5"]
    assert r["arms"]["mlp_hist10"]["status"] == "complete" and r["arms"]["mlp_hist10"]["checkpoint"] == 20000
    c = r["contrasts"]["lstm_scratch - mlp_hist10"]
    assert c["mean_diff"] == pytest.approx(100.0) and c["traces_positive"] == 20 and c["n_seeds"] == 2
    assert "lstm_scratch - derp_hist10" not in r["contrasts"]
    assert "CONTRAST lstm_scratch - mlp_hist10: +100" in capsys.readouterr().out
