"""Cycle 9b: transfer onto the orig family (--pretrain-env-family / --pretrain-variant / --finetune-eps), DERP with a
flattened observation history (--derp-history), parameter counts, and the test6 split."""
import json

import numpy as np
import pytest
import torch

from tuning.legacy import agent_lstm_fixed as FX
from tuning.legacy import c5_protocol as C5
from tuning.legacy import run_legacy as RL


def _small(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test6", (7000,))


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if "cpu_s" not in k and "wall_s" not in k}
    if isinstance(o, list):
        return [_strip(x) for x in o]
    return o


def _sd(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _transfer(seed=3, **kw):
    torch.set_num_threads(1)
    args = dict(pretrain_ticks=40, finetune_ticks=40, checkpoints=[20, 40], eval_splits="val,test6", eval_ticks=30,
                agent_variant="fixed", target_sync=7, reward_div=10, obs_scale="on", eps_schedule="linear10k",
                env_family="orig", orig_variant="spike300", pretrain_env_family="orig",
                pretrain_variant="noise15_multispike")
    args.update(kw)
    return C5.run_c5("dinos_t", seed, **args)


# ---------------------------------------------------------------- splits
def test_test6_split_is_fresh_and_disjoint():
    assert RL.SPLITS["test6"] == tuple(range(7000, 7020))
    others = set().union(*(set(v) for k, v in RL.SPLITS.items() if k != "test6"))
    assert not set(RL.SPLITS["test6"]) & others
    for s in range(10, 15):                      # c9b training seeds lie in no eval split
        assert all(s not in v for v in RL.SPLITS.values())
        RL.check_seeds(s, "fair", ("val", "test6"))
    with pytest.raises(SystemExit):
        RL.check_seeds(7005, "fair", ("val", "test6"))


def test_pretrain_stream_disjoint_from_eval_traces():
    """The source stream RandomState([seed, 5, 2]) yields a trace different from every val/test6 eval trace."""
    def trace(seed, n=50):
        e = C5.make_env("orig:noise15_multispike", "coupled", seed)
        e.reset()
        return np.array([e.step(1)[0][0] for _ in range(n)])
    for seed in range(10, 15):
        src = trace(C5.stream_seed(seed, C5.SOURCE_STREAM))
        tgt = trace(C5.stream_seed(seed, C5.TARGET_STREAM))
        assert not np.allclose(src, tgt)
        for s in RL.SPLITS["val"] + RL.SPLITS["test6"]:
            assert not np.allclose(src, trace(s))


# ---------------------------------------------------------------- transfer
def test_transfer_loads_pretrained_weights(monkeypatch):
    _small(monkeypatch)
    inits, phases = [], {}
    real_make, real_train = FX.make_scratch, C5.train_episodic

    def make(*a, **k):
        ag = real_make(*a, **k)
        inits.append(_sd(ag.online_model))
        return ag

    def train(agent, *a, **k):
        phase = a[6] if len(a) > 6 else k.get("phase")   # (act, env, ticks, lstm, tel, tick0, phase)
        start = (_sd(agent.online_model), _sd(agent.target_model), float(agent.epsilon), len(agent.memory))
        out = real_train(agent, *a, **k)
        phases[phase] = dict(start=start, end=_sd(agent.online_model), agent=agent)
        return out
    monkeypatch.setattr(FX, "make_scratch", make)
    monkeypatch.setattr(C5, "train_episodic", train)
    r = _transfer()
    assert len(inits) == 2 and set(phases) == {"source", "target"}
    src_end = phases["source"]["end"]
    on0, tg0, eps0, mem0 = phases["target"]["start"]
    assert _same(on0, src_end) and _same(tg0, src_end)          # fine-tune starts from the pretrained weights
    assert not _same(on0, inits[1])                              # ... not from its own random init
    assert not _same(src_end, inits[0])                          # pretraining changed the weights
    assert mem0 == 0 and phases["target"]["agent"] is not phases["source"]["agent"]   # fresh memory / agent
    assert eps0 == 1.0 and r["epsilon_start"] == 1.0 and r["learning_rate"] == 1e-4
    assert r["arm"] == "transfer_orig_noise15_multispike" and r["source_env"] == "orig:noise15_multispike"
    assert r["pretrain_ticks"] == 40 and r["finetune_ticks"] == 40 and r["target_ticks"] == 40
    assert r["total_ticks"] == 80 and r["checkpoints"] == [20, 40]
    assert set(r["checkpoint_results"]) == {"20", "40"}
    assert r["source_stream_seed"] == [3, 5, 2] and r["target_stream_seed"] == [3, 5, 1]
    assert r["n_params"] == r["pretrain"]["n_params"] == C5.n_params(FX.make_scratch(6, 3, 10).online_model)
    assert r["checkpoint_results"]["40"]["test6"]["n_episodes"] == 1


def test_transfer_deterministic_and_finetune_eps(monkeypatch):
    _small(monkeypatch)
    r1, r2 = _transfer(), _transfer()
    assert _strip(r1) == _strip(r2)
    lo = _transfer(finetune_eps="low")
    assert lo["epsilon_start"] == pytest.approx(0.2) and lo["finetune_eps"] == "low"
    # linear10k from 0.2 to the floor 0.05: after 20 ticks epsilon = 0.2 - 0.15 * 19 / 10000
    assert lo["checkpoint_results"]["20"]["epsilon_at_checkpoint"] == pytest.approx(0.2 - 0.15 * 19 / 1e4)
    assert r1["checkpoint_results"]["20"]["epsilon_at_checkpoint"] == pytest.approx(1.0 - 0.95 * 19 / 1e4)
    assert lo["pretrain"] == r1["pretrain"]                       # pretraining does not depend on finetune_eps


def test_transfer_from_coupled_source(monkeypatch):
    _small(monkeypatch)
    r = _transfer(pretrain_env_family="coupled", pretrain_variant="legacy", pretrain_ticks=30, finetune_ticks=20,
                  checkpoints=[20])
    assert r["arm"] == "transfer_coupled_legacy" and r["source_env"] == "dense"
    assert r["source_obs_scale_divisors"] == dict(zip(C5.OBS_FEATURES, C5.OBS_SCALE.tolist()))
    assert r["obs_scale_divisors"]["pods"] == 20.0


def test_c9b_validation():
    kw = dict(pretrain_ticks=10, finetune_ticks=10, env_family="orig", orig_variant="orig", agent_variant="fixed")
    with pytest.raises(SystemExit):     # verbatim agent
        C5.run_c5("dinos_t", 3, **dict(kw, agent_variant="verbatim"), pretrain_env_family="orig", pretrain_variant="orig")
    with pytest.raises(SystemExit):     # --source-variant on the orig family
        C5.run_c5("dinos_t", 3, source_variant="legacy", **kw, pretrain_env_family="orig", pretrain_variant="orig")
    with pytest.raises(SystemExit):     # bad orig variant
        C5.run_c5("dinos_t", 3, **kw, pretrain_env_family="orig", pretrain_variant="noise")
    with pytest.raises(SystemExit):     # coupled source needs legacy|coupled
        C5.run_c5("dinos_t", 3, **kw, pretrain_env_family="coupled", pretrain_variant="orig")
    with pytest.raises(SystemExit):     # no pretraining ticks
        C5.run_c5("dinos_t", 3, **dict(kw, pretrain_ticks=0), pretrain_env_family="orig", pretrain_variant="orig")
    with pytest.raises(SystemExit):     # dinos_t on orig without a pretrain family
        C5.run_c5("dinos_t", 3, **kw)
    with pytest.raises(SystemExit):     # coupled target
        C5.run_c5("dinos_t", 3, source_variant="legacy", pretrain_ticks=10, finetune_ticks=10, agent_variant="fixed",
                  pretrain_env_family="orig", pretrain_variant="orig")
    with pytest.raises(SystemExit):
        C5.run_c5("lstm_scratch", 3, total_ticks=10, env_family="orig", orig_variant="orig", finetune_eps="low")
    with pytest.raises(SystemExit):
        C5.run_c5("lstm_scratch", 3, total_ticks=10, env_family="orig", orig_variant="orig", derp_history=10)
    with pytest.raises(SystemExit):
        C5.run_c5("derp", 3, total_ticks=10, derp_history=10)                  # coupled family
    with pytest.raises(SystemExit):
        C5.run_c5("derp", 3, total_ticks=10, env_family="orig", orig_variant="orig", derp_history=0)


# ---------------------------------------------------------------- DERP history
def test_dense_history_window():
    s0, s1 = np.arange(6.0), np.arange(6.0) + 10
    w = C5.window0(s0, False, 10)
    assert w.shape == (60,) and np.all(w[:54] == 0) and np.all(w[54:] == s0)
    w2 = C5.push(w, s1, False, 10)
    assert np.all(w2[:48] == 0) and np.all(w2[48:54] == s0) and np.all(w2[54:] == s1)
    lw = C5.push(C5.window0(s0, True), s1, True)                # same content as the LSTM window, flattened
    assert np.array_equal(lw.reshape(-1), w2)
    assert np.array_equal(C5.window0(s0, False), s0) and C5.push(w, s1, False) is s1       # hist 1 unchanged


def test_derp_history_run(monkeypatch):
    _small(monkeypatch)
    torch.set_num_threads(1)
    run = lambda h: C5.run_c5("derp", 3, total_ticks=40, checkpoints=[40], eval_splits="val", eval_ticks=30,  # noqa: E731
                              reward_div=10, obs_scale="on", eps_schedule="linear10k", env_family="orig",
                              orig_variant="spike300", derp_history=h)
    r1, r10 = run(1), run(10)
    assert r1["arm"] == "derp" and r10["arm"] == "derp_hist10"
    assert r1["derp_history"] == 1 and r10["derp_history"] == 10 and r10["input_size"] == 60
    assert r1["n_params"] == 6 * 128 + 128 + 128 * 128 + 128 + 128 * 3 + 3
    assert r10["n_params"] == 60 * 128 + 128 + 128 * 128 + 128 + 128 * 3 + 3
    assert _strip(run(10)) == _strip(r10)


# ---------------------------------------------------------------- CLI
def test_cli_c9b(monkeypatch, tmp_path):
    _small(monkeypatch)
    out = tmp_path / "t.json"
    res = RL.main(["--agent", "dinos_t", "--agent-variant", "fixed", "--target-sync", "7", "--env-family", "orig",
                   "--orig-variant", "spike300", "--pretrain-env-family", "orig", "--pretrain-variant", "orig",
                   "--pretrain-ticks", "20", "--finetune-ticks", "20", "--checkpoints", "20", "--eval-split", "val",
                   "--finetune-eps", "low", "--reward-div", "10", "--obs-scale", "on", "--eps-schedule", "linear10k",
                   "--seed", "10", "--out", str(out)])
    j = json.load(open(out))
    assert res["arm"] == j["arm"] == "transfer_orig_orig" and j["finetune_eps"] == "low" and j["seed"] == 10
    for bad in (["--pretrain-env-family", "orig"], ["--finetune-eps", "low"], ["--derp-history", "10"]):
        with pytest.raises(SystemExit):     # cycle-9b flags without --env-family orig
            RL.main(["--agent", "derp", "--env-variant", "coupled", "--total-ticks", "30"] + bad)
