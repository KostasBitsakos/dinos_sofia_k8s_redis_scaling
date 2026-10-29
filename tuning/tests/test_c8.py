"""Cycle 8: --reward-div, --obs-scale, --eps-schedule on the c5 protocol (all arms) and the test4 split.
The defaults are checked against tuning/tests/data/c6_fixed_ref_s3.json, produced by the pre-cycle-8 code (commit
fd42de8) with the same tiny budget (fixed lstm_scratch, fixed dinos_t legacy, derp; seed 3)."""
import json
import os

import numpy as np
import pytest
import torch

from tuning.legacy import c5_protocol as C5
from tuning.legacy import c7_oracles as C7
from tuning.legacy import coupled_baselines as CB
from tuning.legacy import coupled_env as CE
from tuning.legacy import env_dense, env_transfer
from tuning.legacy import run_legacy as RL

REF = os.path.join(os.path.dirname(__file__), "data", "c6_fixed_ref_s3.json")
C8_KEYS = {"reward_div", "obs_scale", "obs_scale_divisors", "eps_schedule", "eps_linear_ticks"}


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if "cpu_s" not in k and "wall_s" not in k}
    if isinstance(o, list):
        return [_strip(x) for x in o]
    return o


def _ref_runs(**c8):
    torch.set_num_threads(1)
    kw = dict(target_env="lstm_transfer", eval_splits="val,test", eval_ticks=30, agent_variant="fixed", target_sync=7,
              **c8)
    return {"lstm_scratch_fixed": C5.run_c5("lstm_scratch", 3, total_ticks=60, checkpoints=[30, 60], **kw),
            "dinos_t_legacy_fixed": C5.run_c5("dinos_t", 3, source_variant="legacy", pretrain_ticks=40,
                                              finetune_ticks=40, checkpoints=[20, 40], **kw),
            "derp": C5.run_c5("derp", 3, total_ticks=60, checkpoints=[60], target_env="lstm_transfer",
                              eval_splits="val,test", eval_ticks=30, **c8)}


def _small(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test", (2000,))


# ---------------------------------------------------------------- defaults reproduce cycle 6
def test_defaults_reproduce_c6_fixed_and_derp(monkeypatch):
    _small(monkeypatch)
    with open(REF) as f:
        ref = json.load(f)
    for c8 in ({}, dict(reward_div=1.0, obs_scale="off", eps_schedule="original")):
        got = _strip(json.loads(json.dumps(_ref_runs(**c8))))
        for arm in ref:
            assert set(got[arm]) - set(ref[arm]) == C8_KEYS, arm
            assert {k: got[arm][k] for k in ref[arm]} == ref[arm], arm
            assert got[arm]["reward_div"] == 1.0 and got[arm]["obs_scale"] == "off"
            assert got[arm]["eps_schedule"] == "original" and got[arm]["obs_scale_divisors"] is None


def test_each_option_changes_training(monkeypatch):
    """Sanity: every non-default option changes the training of the same tiny run (it is actually wired in).
    Compared on the agent's loss history (eval returns of a barely trained net can coincide)."""
    _small(monkeypatch)
    monkeypatch.setattr(C5, "LINEAR_EPS_TICKS", 30)
    from tuning.legacy import agent_lstm_fixed as FX
    made = []
    orig = FX.make_scratch

    def rec(*a, **k):
        made.append(orig(*a, **k))
        return made[-1]
    monkeypatch.setattr(FX, "make_scratch", rec)
    kw = dict(total_ticks=60, checkpoints=[60], target_env="lstm_transfer", eval_splits="val", eval_ticks=5,
              agent_variant="fixed")
    C5.run_c5("lstm_scratch", 3, **kw)
    base = np.array(made[-1].loss_history)
    for c8 in (dict(reward_div=210.0), dict(obs_scale="on"), dict(eps_schedule="linear10k")):
        C5.run_c5("lstm_scratch", 3, **kw, **c8)
        h = np.array(made[-1].loss_history)
        assert len(h) == len(base) == 60 - 16 and not np.array_equal(h, base), c8
        if "reward_div" in c8:
            assert h.mean() < base.mean() / 20          # Huber on targets in units of 210


# ---------------------------------------------------------------- reward scaling and observation scaling
class _Rec:
    """Stub agent: records what train_episodic stores; never learns."""

    def __init__(self):
        self.memory, self.epsilon, self.seen_eps = [], 0.5, []

    def remember(self, *tr):
        self.memory.append(tr)

    def replay(self, n):
        pass


def _record(lstm, ticks=60, **kw):
    ag = _Rec()
    env = C5.make_env("lstm_transfer", "coupled", [9, 5, 1])
    acts = iter(np.random.RandomState(1).randint(3, size=ticks))
    tr = C5.train_episodic(ag, lambda s: int(next(acts)), env, ticks, lstm, **kw)
    return ag, tr


@pytest.mark.parametrize("lstm", (True, False))
def test_reward_div_scales_stored_reward_only(lstm):
    a1, t1 = _record(lstm)
    a2, t2 = _record(lstm, reward_div=210.0)
    r1 = np.array([m[2] for m in a1.memory])
    r2 = np.array([m[2] for m in a2.memory])
    assert np.array_equal(r2, r1 / 210.0) and np.abs(r1).max() > 1.0
    assert t1 == t2                                                  # reported returns unscaled
    for m1, m2 in zip(a1.memory, a2.memory):
        assert np.array_equal(m1[0], m2[0]) and m1[1] == m2[1] and np.array_equal(m1[3], m2[3]) and m1[4] == m2[4]


def test_scaled_target_is_target_over_d():
    """y_D = r/D + gamma*Q_D' with Q_D = Q/D is exactly y/D: the scaled problem is the same problem in units of D."""
    rng = np.random.RandomState(0)
    r, q, g, D = rng.randn(100) * 300, rng.randn(100) * 5000, 0.98, 210.0
    assert np.allclose(r / D + g * (q / D), (r + g * q) / D)


@pytest.mark.parametrize("lstm", (True, False))
def test_obs_scale_divides_agent_observations(lstm):
    a1, t1 = _record(lstm)
    a2, t2 = _record(lstm, obs=C5.obs_fn("on"))
    assert t1 == t2 and C5.obs_fn("off") is None
    for m1, m2 in zip(a1.memory, a2.memory):
        assert np.allclose(m2[0], np.asarray(m1[0]) / C5.OBS_SCALE) and np.allclose(m2[3], np.asarray(m1[3]) / C5.OBS_SCALE)
    if lstm:
        assert np.all(a2.memory[0][0][:-1] == 0)       # zero padding rows stay zero


def test_obs_order_matches_env_layout():
    """OBS_SCALE = [mean load, reference capacity, %, %, latency amplitude, initial pods] in the order of the
    lstm_transfer (and dense source) observation [load, capacity, cpu, mem, latency, num_vms]."""
    assert C5.OBS_FEATURES == ("load", "capacity", "cpu", "mem", "latency", "pods")
    for cls in (env_transfer.Environment, env_dense.Environment):
        e = cls()
        assert C5.OBS_SCALE[0] == e.load_amplitude + e.load_offset == CE.REF_LOAD
        assert C5.OBS_SCALE[1] == 10 * e.num_vms and C5.OBS_SCALE[5] == e.num_vms == CE.REF_PODS
        assert C5.OBS_SCALE[2] == C5.OBS_SCALE[3] == CE.PCT_MAX
    assert C5.OBS_SCALE[4] == env_transfer.Environment().latency_amplitude
    L, c0, m0, l0 = C7.draws(5000, 3)                               # raw draws of the eval trace 5000
    env = C5.make_env("lstm_transfer", "coupled", 5000)
    s0 = env.reset()
    s = env.step(2)[0]                                              # +1 pod at t = 1 (no spike)
    g, c, m, lat = CE.coupling(L[0], 7.0, c0[0], m0[0], l0[0])
    assert s0[5] == 6 and s0[1] == 60.0
    assert np.allclose(s, [L[0], 70.0, c, m, lat, 7.0])             # [load, capacity, cpu, mem, latency, pods]
    sc = C5.obs_fn("on")(s)
    assert np.isclose(sc[5], 7 / 6) and np.isclose(sc[1], 70 / 60) and np.isclose(sc[0], s[0] / 70)


# ---------------------------------------------------------------- epsilon schedule
def test_linear_eps_values():
    f = C5.linear_eps(1.0, 0.05, 10000)
    assert f(0) == 1.0 and np.isclose(f(5000), 0.525) and f(10000) == 0.05 and f(40000) == 0.05
    assert np.isclose(f(9999), 1.0 - 0.95 * 0.9999)
    g = C5.linear_eps(0.2, 0.1, 10000)
    assert g(0) == 0.2 and np.isclose(g(2500), 0.175) and g(10000) == 0.1


@pytest.mark.parametrize("agent", ("lstm_scratch", "derp"))
def test_linear_eps_drives_real_agent(monkeypatch, agent):
    _small(monkeypatch)
    monkeypatch.setattr(C5, "LINEAR_EPS_TICKS", 40)
    seen = []
    orig = C5.train_episodic

    def spy(ag, act, *a, **k):
        def f(s):
            seen.append(ag.epsilon)
            return act(s)
        return orig(ag, f, *a, **k)
    monkeypatch.setattr(C5, "train_episodic", spy)
    kw = dict(agent_variant="fixed") if agent != "derp" else {}
    r = C5.run_c5(agent, 3, total_ticks=60, checkpoints=[60], target_env="lstm_transfer", eval_splits="val",
                  eval_ticks=5, eps_schedule="linear10k", **kw)
    e0, fl = (1.0, 0.05) if agent != "derp" else (1.0, 0.01)
    exp = [e0 + (fl - e0) * t / 40 if t < 40 else fl for t in range(60)]
    assert np.allclose(seen, exp) and r["epsilon_decay"] == 1.0 and r["eps_linear_ticks"] == 40
    assert r["target_train"]["epsilon_end"] == pytest.approx(fl)


def test_linear_eps_per_phase_transfer(monkeypatch):
    """dinos_t: the source agent goes 1 -> 0.05 and the transfer agent 0.2 -> 0.1, each over its own phase ticks."""
    _small(monkeypatch)
    monkeypatch.setattr(C5, "LINEAR_EPS_TICKS", 20)
    seen = {}
    orig = C5.train_episodic

    def spy(ag, act, env, ticks, lstm, tel=None, tick0=0, phase="target", *a, **k):
        def f(s):
            seen.setdefault(phase, []).append(ag.epsilon)
            return act(s)
        return orig(ag, f, env, ticks, lstm, tel, tick0, phase, *a, **k)
    monkeypatch.setattr(C5, "train_episodic", spy)
    r = C5.run_c5("dinos_t", 3, source_variant="legacy", pretrain_ticks=30, finetune_ticks=30, target_env="lstm_transfer",
                  eval_splits="val", eval_ticks=5, agent_variant="fixed", eps_schedule="linear10k", reward_div=210.0,
                  obs_scale="on")
    assert seen["source"][0] == 1.0 and np.isclose(seen["source"][10], 0.525) and seen["source"][-1] == 0.05
    assert seen["target"][0] == 0.2 and np.isclose(seen["target"][10], 0.15) and seen["target"][-1] == 0.1
    assert r["epsilon_start"] == 0.2 and r["reward_div"] == 210.0 and r["obs_scale"] == "on"


# ---------------------------------------------------------------- guards, CLI, test4
def test_guards_and_cli_forwarding(monkeypatch):
    for bad in ((0, "off", "original"), (-1, "off", "original"), (float("nan"), "off", "original"),
                (1, "yes", "original"), (1, "off", "linear")):
        with pytest.raises(SystemExit):
            C5.validate_c8(*bad)
    with pytest.raises(SystemExit):
        RL.main(["--agent", "derp", "--train-ticks", "1000", "--reward-div", "210"])     # needs c5 flags
    got = {}

    def fake(*a, **k):
        got.update(k, args=a)
        return dict(arm="derp", cpu_s=0.0, train_cpu_s_per_1k_ticks=0.0, checkpoint_results={}, eval_splits=[])
    monkeypatch.setattr(C5, "run_c5", fake)
    RL.main(["--agent", "derp", "--total-ticks", "100", "--env-variant", "coupled", "--target-env", "lstm_transfer",
             "--eval-split", "val,test4", "--reward-div", "210", "--obs-scale", "on", "--eps-schedule", "linear10k"])
    assert got["reward_div"] == 210.0 and got["obs_scale"] == "on" and got["eps_schedule"] == "linear10k"
    assert got["args"][9] == "val,test4" and got["args"][5] == 100


def test_test4_split_everywhere(tmp_path, monkeypatch):
    assert RL.SPLITS["test4"] == tuple(range(5000, 5020)) and C7.SPLITS["test4"] == RL.SPLITS["test4"]
    assert not set(RL.SPLITS["test4"]) & set().union(*(set(v) for k, v in RL.SPLITS.items() if k != "test4"))
    assert not set(RL.SPLITS["test4"]) & set(C7.SPLITS["test3"])
    assert RL.parse_splits("val,test4") == ("val", "test4")
    with pytest.raises(SystemExit):
        RL.check_seeds(5003, "fair", ("val", "test4"))
    # baselines report test4
    monkeypatch.setattr(RL, "EP_TICKS", 50)
    monkeypatch.setitem(RL.SPLITS, "val", (1100,))
    monkeypatch.setitem(RL.SPLITS, "test4", (5000, 5001))
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"grid": [50], "fixed": [5]}))
    r = CB.main(["--env", "lstm_transfer", "--config", str(cfg), "--out", str(tmp_path / "b.json"),
                 "--report-splits", "test4"])
    assert r["test4_seeds"] == [5000, 5001] and r["test4"]["hpa_rule_tuned"]["n_episodes"] == 2


def test_oracles_test4_trace_is_eval_trace():
    """c7_oracles on test4 replays the same trace the agents' test4 evaluation sees (fresh env, RandomState(s))."""
    env = C5.make_env("lstm_transfer", "coupled", 5000)
    env.reset()
    L = [env.step(1)[0][0] for _ in range(20)]
    d = C7.draws(5000)
    assert np.allclose(np.asarray(d[0])[:20], L)
