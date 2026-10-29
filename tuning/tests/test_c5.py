"""Cycle 5 protocol (tuning/legacy/c5_protocol.py via run_legacy.py): private env RNG streams / matched target traces,
reset-on-done with the reset observation, checkpoint evaluation on deep copies, CLI guards."""
import json

import numpy as np
import pytest
import torch

from tuning.legacy import c5_protocol as C5
from tuning.legacy import coupled_baselines as CB
from tuning.legacy import env_transfer, run_legacy as RL


def _trace(env, actions, noise=False):
    out = [env.reset()]
    for a in actions:
        if noise:
            np.random.rand(3)                       # agent-side global draws must not move the env stream
        out.append(env.step(a)[0])
    return np.array(out)


def test_stream_env_trace_independent_of_actions_and_global_rng():
    rng = np.random.RandomState(0)
    a1, a2 = rng.randint(0, 3, 600), np.zeros(600, dtype=int)
    raw1 = _trace(C5.StreamEnv(env_transfer.Environment(), [3, 5, 1]), a1)
    np.random.seed(123)
    raw2 = _trace(C5.StreamEnv(env_transfer.Environment(), [3, 5, 1]), a2, noise=True)
    # exogenous draws (load, cpu, mem, latency) identical; only capacity (10n) and pods depend on the actions
    for col in (0, 2, 3, 4):
        assert np.array_equal(raw1[:, col], raw2[:, col])
    assert not np.array_equal(raw1[:, 5], raw2[:, 5])
    c1 = _trace(C5.make_env("lstm_transfer", "coupled", [3, 5, 1]), a1)
    assert np.array_equal(c1[:, 0], raw1[:, 0]) and np.array_equal(c1[0], _trace(
        C5.make_env("lstm_transfer", "coupled", [3, 5, 1]), a2)[0])       # identical initial observation
    other = _trace(C5.make_env("lstm_transfer", "coupled", [4, 5, 1]), a1)
    assert not np.array_equal(other[:, 0], c1[:, 0])


def test_eval_trace_equals_baselines_trace():
    """Eval episode with seed s on a private stream == coupled_baselines' np.random.seed(s) path (same target)."""
    RL.seed_all(1100)
    env = CB.make_env("lstm_transfer", "coupled")
    s0 = env.reset()
    ref = RL.run_episode(env, lambda s: 0, False, s0, 400, False, True)
    np.random.seed(7)
    e2 = C5.make_env("lstm_transfer", "coupled", 1100)
    got = C5.eval_episode(e2, lambda s: 0, False, e2.reset(), 400)
    assert got["ret"] == ref["ret"] and got["p95_latency"] == ref["p95_latency"]


class _Agent:
    def __init__(self):
        self.memory, self.epsilon = [], 1.0

    def remember(self, *t):
        self.memory.append(t)

    def replay(self, b):
        pass


class _ShortEnv:
    """done every 7 ticks; reset observation all 9s, step observation all 1s."""
    def __init__(self):
        self.t, self.resets = 0, 0

    def reset(self):
        self.t, self.resets = 0, self.resets + 1
        return np.full(RL.STATE, 9.0)

    def step(self, a):
        self.t += 1
        return (np.ones(RL.STATE), 1.0, 0, 0, 0, 0, 0, self.t >= 7)


@pytest.mark.parametrize("lstm", (True, False))
def test_reset_on_done_uses_reset_observation(lstm):
    ag, env = _Agent(), _ShortEnv()
    r = C5.train_episodic(ag, lambda s: 1, env, 20, lstm)
    assert env.resets == 1 + 2                              # start + after ticks 7 and 14
    assert r["episode_returns"] == [7.0, 7.0] and r["partial_episode_return"] == 6.0
    first, after_done = ag.memory[0][0], ag.memory[7][0]   # states at tick 1 and tick 8
    for s in (first, after_done):
        if lstm:
            assert np.all(s[-1] == 9.0) and np.all(s[:-1] == 0.0)
        else:
            assert np.all(s == 9.0)
    assert sum(1 for m in ag.memory if m[4]) == 2


def test_eval_window_starts_with_reset_observation():
    w = C5.window0(np.arange(6.0), True)
    assert w.shape == (RL.SEQ, RL.STATE) and np.array_equal(w[-1], np.arange(6.0)) and not w[:-1].any()


def _small(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test", (2000,))


def test_checkpoint_eval_does_not_alter_training(monkeypatch, tmp_path):
    _small(monkeypatch)
    torch.set_num_threads(1)
    kw = dict(total_ticks=40, target_env="lstm_transfer", eval_splits="val,test", eval_ticks=30)
    a = C5.run_c5("lstm_scratch", 3, checkpoints=[40], **kw)
    b = C5.run_c5("lstm_scratch", 3, checkpoints=[10, 25, 40], out=str(tmp_path / "r.json"), tel_dir=str(tmp_path), **kw)
    assert a["checkpoint_results"]["40"]["val"]["mean_ret"] == b["checkpoint_results"]["40"]["val"]["mean_ret"]
    assert a["checkpoint_results"]["40"]["test"]["episodes"] == b["checkpoint_results"]["40"]["test"]["episodes"]
    assert a["target_train"]["epsilon_end"] == b["target_train"]["epsilon_end"]
    assert a["target_train"]["partial_episode_return"] == b["target_train"]["partial_episode_return"]
    for t in (10, 25, 40):
        ck = json.loads((tmp_path / f"r.ck{t}.json").read_text())
        assert ck["checkpoint"]["target_tick"] == t and ck["arm"] == "lstm_scratch"
    assert json.loads((tmp_path / "progress.json").read_text()) == {"step": 40 + 3 * 3 * 30, "total": 40 + 3 * 3 * 30}


def test_target_traces_matched_across_arms(monkeypatch):
    _small(monkeypatch)
    seen = {}
    orig = C5.make_env

    def rec(name, variant, seed):
        e = orig(name, variant, seed)
        step = e.step
        key = (name, variant, tuple(seed) if isinstance(seed, list) else seed)
        seen.setdefault(key, [])
        log = []
        seen[key].append(log)

        def s(a):
            out = step(a)
            log.append(float(out[0][0]))
            return out
        e.__dict__["step"] = s
        return e
    monkeypatch.setattr(C5, "make_env", rec)
    kw = dict(target_env="lstm_transfer", eval_splits="val", eval_ticks=5, checkpoints=[30])
    C5.run_c5("derp", 2, total_ticks=30, **kw)
    C5.run_c5("lstm_scratch", 2, total_ticks=30, **kw)
    C5.run_c5("dinos_t", 2, source_variant="legacy", pretrain_ticks=20, finetune_ticks=30, **kw)
    C5.run_c5("dinos_t", 2, source_variant="none", finetune_ticks=30, **kw)
    tgt = seen[("lstm_transfer", "coupled", (2, 5, 1))]
    assert len(tgt) == 4
    loads = tgt
    assert all(len(x) == 30 for x in loads) and all(x == loads[0] for x in loads)
    assert ("dense", "legacy", (2, 5, 2)) in seen                      # source on its own stream
    ev = seen[("lstm_transfer", "coupled", 1100)]
    assert len(ev) == 4 and all(x == ev[0] for x in ev)       # identical eval traces across arms


def test_validation_and_cli_guards():
    assert C5.validate("lstm_scratch", None, None, None, 40000, [20000, 40000]) == (0, 40000, [20000, 40000])
    assert C5.validate("dinos_t", "coupled", 10000, 30000, None, [1000, 10000, 30000])[:2] == (10000, 30000)
    assert C5.validate("dinos_t", "none", None, 30000, None, None) == (0, 30000, [30000])
    for args in (("dinos_t", "none", 10, 30, None, None), ("dinos_t", "legacy", None, 30, None, None),
                 ("derp", "legacy", None, None, 40, None), ("derp", None, None, None, 40, [50]),
                 ("hpa_rule", None, None, None, 40, None)):
        with pytest.raises(SystemExit):
            C5.validate(*args)
    with pytest.raises(SystemExit):
        RL.main(["--agent", "derp", "--total-ticks", "40"])                # c5 needs coupled
    with pytest.raises(SystemExit):
        RL.main(["--agent", "derp", "--env-variant", "coupled", "--total-ticks", "40", "--train-ticks", "1000"])
    with pytest.raises(SystemExit):
        RL.main(["--agent", "derp"])                                       # legacy path still needs --train-ticks


def test_evaluate_works_on_a_copy(monkeypatch):
    _small(monkeypatch)
    ag = RL.make_lstm_scratch()
    x = torch.randn(1, RL.SEQ, RL.STATE)
    ag.online_model.train()
    ag.online_model(x).sum().backward()                    # hidden is now a non-leaf of a graph, as in training
    h = ag.online_model.hidden
    w = {k: v.clone() for k, v in ag.online_model.state_dict().items()}
    st = np.random.get_state()[1].copy()
    C5.evaluate(ag.online_model, True, "lstm_transfer", (1100,), "val", 20)
    assert ag.online_model.hidden is h and ag.online_model.training
    assert all(torch.equal(w[k], v) for k, v in ag.online_model.state_dict().items())
    assert np.array_equal(np.random.get_state()[1], st)
