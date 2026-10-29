"""Cycle 6: fixed LSTM agent (hidden reset, detached Double-DQN target, periodic target sync), --diag observers,
test2 split, and the unchanged verbatim c5 path (reference JSON produced by the pre-cycle-6 code, commit fcb691b)."""
import json
import os
import random

import numpy as np
import pytest
import torch

from tuning.legacy import agent_lstm_fixed as FX
from tuning.legacy import c5_protocol as C5
from tuning.legacy import c6_diag
from tuning.legacy import coupled_baselines as CB
from tuning.legacy import networks_lstm
from tuning.legacy import run_legacy as RL

REF = os.path.join(os.path.dirname(__file__), "data", "c5_verbatim_ref_s3.json")
C6_KEYS = {"agent_variant", "target_sync_replays", "double_dqn_target", "hidden_reset_per_window"}
C8_KEYS = {"reward_div", "obs_scale", "obs_scale_divisors", "eps_schedule", "eps_linear_ticks"}   # cycle 8, defaults


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if "cpu_s" not in k and "wall_s" not in k}
    if isinstance(o, list):
        return [_strip(x) for x in o]
    return o


def _small(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test", (2000,))
    monkeypatch.setitem(RL.SPLITS, "test2", (3000,))


def _fill(agent, n=40, seed=0):
    rng = np.random.RandomState(seed)
    for _ in range(n):
        agent.remember(rng.rand(RL.SEQ, RL.STATE) * 10, int(rng.randint(3)), float(rng.randn()),
                       rng.rand(RL.SEQ, RL.STATE) * 10, bool(rng.rand() < 0.1))


# ---------------------------------------------------------------- fixed agent
def test_hidden_reset_output_independent_of_prior_calls():
    torch.manual_seed(0)
    m = FX.ResetLSTM(RL.STATE, 128, 3, RL.ACTIONS)
    x1, x2 = torch.randn(1, RL.SEQ, RL.STATE), torch.randn(1, RL.SEQ, RL.STATE)
    with torch.no_grad():
        a = m(x1)
        m(x2)
        m(x2)
        b = m(x1)
        ref = networks_lstm.myLSTM(RL.STATE, 128, 3, RL.ACTIONS)
        ref.load_state_dict(m.state_dict())             # same state_dict layout as the verbatim network
        r1 = ref(x1)                                    # verbatim from fresh zero hidden == reset forward
        r2 = ref(x1)                                    # verbatim carries the hidden state -> differs
        both = m(torch.cat([x2, x1]))                   # batch rows are independent windows
    assert torch.equal(a, b) and torch.allclose(a, r1, atol=1e-6)
    assert not torch.allclose(r1, r2, atol=1e-6)
    assert torch.allclose(both[1], a[0], atol=1e-6)


def test_target_sync_period():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    ag = FX.make_scratch(RL.STATE, RL.ACTIONS, RL.SEQ, target_sync=3)
    _fill(ag)
    init = {k: v.clone() for k, v in ag.target_model.state_dict().items()}

    def same(a, b):
        return all(torch.equal(a[k], b[k]) for k in a)
    for i in range(1, 8):
        ag.replay(RL.BATCH)
        tgt, onl = ag.target_model.state_dict(), ag.online_model.state_dict()
        if i % 3 == 0:
            assert same(tgt, onl)
        else:
            assert not same(tgt, onl)
        if i < 3:
            assert same(tgt, init)
    assert ag.n_replays == 7 and ag.n_target_syncs == 2
    assert ag.epsilon == pytest.approx(0.995 ** 7)
    assert FX.make_scratch(RL.STATE, RL.ACTIONS, RL.SEQ).target_sync == 250


def test_ddqn_target_detached_online_argmax_target_value():
    torch.manual_seed(1)
    ag = FX.make_scratch(RL.STATE, RL.ACTIONS, RL.SEQ)
    with torch.no_grad():
        ag.online_model.fc.bias.copy_(torch.tensor([0.0, 100.0, 0.0]))    # online argmax = 1
        ag.target_model.fc.bias.copy_(torch.tensor([0.0, -50.0, 200.0]))  # target's own max = action 2
    s = np.random.RandomState(0).rand(RL.SEQ, RL.STATE)
    y = ag.ddqn_target(1.5, s, False)
    with torch.no_grad():
        qt = ag.target_model(ag._x(s))[0]
    assert isinstance(y, torch.Tensor) and not y.requires_grad and y.grad_fn is None
    assert float(y) == pytest.approx(1.5 + 0.98 * float(qt[1]))
    assert float(y) < 1.5 + 0.98 * float(qt.max()) - 1.0                   # not the plain-DQN max
    assert ag.ddqn_target(2.0, s, True) == 2.0


def test_fixed_update_moves_only_replayed_action_error():
    """With zero hidden per window the eval and train forwards agree: the loss only sees the replayed action."""
    random.seed(0)
    torch.manual_seed(0)
    ag = FX.make_scratch(RL.STATE, RL.ACTIONS, RL.SEQ)
    _fill(ag, seed=3)
    dg = c6_diag.Diag(1, np.random.RandomState(0).rand(4, RL.SEQ, RL.STATE), 0)
    dg.attach(ag, "t", lambda: 0)
    ag.replay(RL.BATCH)
    assert dg.records[-1]["spurious_nonaction_abs"]["p95"] < 1e-6        # float rounding only
    assert dg.records[-1]["optimizer_steps"] == RL.BATCH and dg.records[-1]["param_delta_norm"] > 0
    assert dg.order_probes[-1]["reset_max_abs"] < 1e-5


def test_transfer_fixed_hyperparameters_and_weights():
    torch.manual_seed(0)
    w = networks_lstm.myLSTM(RL.STATE, 128, 3, RL.ACTIONS).state_dict()
    a = FX.make_transfer(RL.STATE, RL.ACTIONS, RL.SEQ, w, 250)
    v = RL.make_transfer(w)
    for k in ("epsilon", "epsilon_min", "epsilon_decay", "learning_rate", "gamma"):
        assert getattr(a, k) == getattr(v, k), k
    assert a.memory.maxlen == v.memory.maxlen
    assert all(torch.equal(a.online_model.state_dict()[k], w[k]) for k in w)
    s = FX.make_scratch(RL.STATE, RL.ACTIONS, RL.SEQ)
    vs = RL.make_lstm_scratch()
    for k in ("epsilon", "epsilon_min", "epsilon_decay", "learning_rate", "gamma"):
        assert getattr(s, k) == getattr(vs, k), k


# ---------------------------------------------------------------- protocol integration
def test_verbatim_path_unchanged_vs_reference(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test", (2000,))
    torch.set_num_threads(1)
    kw = dict(target_env="lstm_transfer", eval_splits="val,test", eval_ticks=30)
    got = {"lstm_scratch": C5.run_c5("lstm_scratch", 3, total_ticks=40, checkpoints=[20, 40], **kw),
           "dinos_t_legacy": C5.run_c5("dinos_t", 3, source_variant="legacy", pretrain_ticks=20, finetune_ticks=30,
                                       checkpoints=[10, 30], **kw),
           "derp": C5.run_c5("derp", 3, total_ticks=40, checkpoints=[40], **kw)}
    with open(REF) as f:
        ref = _strip(json.load(f))
    got = _strip(json.loads(json.dumps(got)))
    for arm in ref:
        extra = set(got[arm]) - set(ref[arm])
        assert extra == (C8_KEYS if arm == "derp" else C6_KEYS | C8_KEYS), (arm, extra)
        assert {k: got[arm][k] for k in ref[arm]} == ref[arm], arm
        if arm != "derp":
            assert got[arm]["agent_variant"] == "verbatim" and got[arm]["target_sync_replays"] == 1


@pytest.mark.parametrize("variant", ("verbatim", "fixed"))
def test_diag_does_not_change_training(monkeypatch, tmp_path, variant):
    _small(monkeypatch)
    torch.set_num_threads(1)
    kw = dict(source_variant="legacy", pretrain_ticks=30, finetune_ticks=40, checkpoints=[40], target_env="lstm_transfer",
              eval_splits="val,test2", eval_ticks=20, agent_variant=variant)
    a = C5.run_c5("dinos_t", 4, **kw)
    out = tmp_path / "d.json"
    b = C5.run_c5("dinos_t", 4, diag=5, out=str(out), **kw)
    assert _strip(a["checkpoint_results"]) == _strip(b["checkpoint_results"])
    assert a["target_train"] == {**b["target_train"], "eval_cpu_s": a["target_train"]["eval_cpu_s"]}
    d = json.loads((tmp_path / "d.diag.json").read_text())
    assert b["diag"] == str(tmp_path / "d.diag.json") and d["agent_variant"] == variant
    ph = [r["phase"] for r in d["records"]]
    assert ph[0] == "source" and "target" in ph and d["order_probes"]
    tr = [r for r in d["records"] if r["phase"] == "target" and not r["initial"]]
    assert sum(sum(r["act_hist"]) for r in tr) == 40 and tr[-1]["param_delta_norm"] > 0
    assert sum(tr[-1]["probe_greedy_hist"]) == c6_diag.PROBE_N
    spur = max(r["spurious_nonaction_abs"]["p95"] for r in tr if r["spurious_nonaction_abs"])
    if variant == "fixed":
        assert spur < 1e-6 and b["target_sync_replays"] == 250 and b["double_dqn_target"]
        assert all(p["reset_max_abs"] < 1e-5 for p in d["order_probes"])
    else:
        assert spur > 1e-4 and b["target_sync_replays"] == 1


def test_fixed_variant_used_for_source_and_target(monkeypatch):
    _small(monkeypatch)
    made = []
    for name in ("make_scratch", "make_transfer"):
        orig = getattr(FX, name)

        def rec(*a, _o=orig, _n=name, **k):
            ag = _o(*a, **k)
            made.append((_n, ag.target_sync))
            return ag
        monkeypatch.setattr(FX, name, rec)
    r = C5.run_c5("dinos_t", 2, source_variant="legacy", pretrain_ticks=20, finetune_ticks=20, target_env="lstm_transfer",
                  eval_splits="val", eval_ticks=5, agent_variant="fixed", target_sync=7)
    assert made == [("make_scratch", 7), ("make_transfer", 7)]
    assert r["target_sync_replays"] == 7 and r["replays"] == 20 - RL.BATCH and r["target_syncs"] == 0
    made.clear()
    C5.run_c5("lstm_scratch", 2, total_ticks=20, target_env="lstm_transfer", eval_splits="val", eval_ticks=5,
              agent_variant="fixed")
    assert made == [("make_scratch", 250)]


def test_c6_guards_and_test2_split():
    assert RL.SPLITS["test2"] == tuple(range(3000, 3020))
    assert RL.parse_splits("val,test2") == ("val", "test2")
    with pytest.raises(SystemExit):
        RL.check_seeds(3005, "fair", ("val", "test2"))
    for args in (("derp", "fixed", None, None), ("derp", "verbatim", None, 10), ("lstm_scratch", "verbatim", 250, None),
                 ("dinos_t", "fixed", 0, None), ("dinos_t", "fixed", None, 0), ("lstm_scratch", "other", None, None)):
        with pytest.raises(SystemExit):
            C5.validate_c6(*args)
    assert C5.validate_c6("dinos_t", "fixed", None, None) == 250
    assert C5.validate_c6("lstm_scratch", "verbatim", None, 100) is None
    with pytest.raises(SystemExit):
        RL.main(["--agent", "lstm_scratch", "--train-ticks", "1000", "--agent-variant", "fixed"])   # c5 flags needed


def test_baselines_report_test2(tmp_path, monkeypatch):
    monkeypatch.setattr(RL, "EP_TICKS", 100)
    monkeypatch.setitem(RL.SPLITS, "val", (1100,))
    monkeypatch.setitem(RL.SPLITS, "test", (2000,))
    monkeypatch.setitem(RL.SPLITS, "test2", (3000, 3001))
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"grid": [50], "fixed": [6]}))
    r = CB.main(["--env", "lstm_transfer", "--config", str(cfg), "--out", str(tmp_path / "b.json"),
                 "--report-splits", "test,test2"])
    assert r["report_splits"] == ["test", "test2"] and r["test2_seeds"] == [3000, 3001]
    assert r["test2"]["always_down"]["n_episodes"] == 2 and r["test"]["always_down"]["n_episodes"] == 1
    with pytest.raises(SystemExit):
        CB.main(["--config", str(cfg), "--out", str(tmp_path / "x.json"), "--report-splits", "val"])
