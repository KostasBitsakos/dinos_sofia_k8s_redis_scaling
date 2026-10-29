import json
import random

import numpy as np
import pytest
import torch

from tuning import diagnostics, quick_test, transfer_map, tuned_agents


def _fill(agent, n, dim, seq=None, rng=None):
    rng = rng or np.random.default_rng(0)
    for _ in range(n):
        s = rng.random((seq, dim)) if seq else rng.random(dim)
        s2 = rng.random((seq, dim)) if seq else rng.random(dim)
        agent.remember(s, int(rng.integers(3)), float(rng.normal()), s2, bool(rng.random() < 0.05))


def test_eps_schedule_values():
    f = tuned_agents.eps_at
    assert f("linear", 0, 1.0, 0.1, 100) == 1.0
    assert f("linear", 50, 1.0, 0.1, 100) == pytest.approx(0.55)
    assert f("linear", 500, 1.0, 0.1, 100) == pytest.approx(0.1)
    assert f("exp", 100, 1.0, 0.1, 100) == pytest.approx(0.1)
    assert f("exp", 50, 1.0, 0.01, 100) == pytest.approx(0.1)


def test_check_cfg_rejects_bad():
    with pytest.raises(ValueError):
        tuned_agents.check_cfg({"nope": 1})
    with pytest.raises(ValueError):
        tuned_agents.check_cfg({"eps_schedule": "cosine"})
    with pytest.raises(ValueError):
        tuned_agents.check_cfg({"memory_size": 0})


def test_defaults_keep_frozen_behaviour():
    import agents
    torch.manual_seed(0)
    ref = agents.make_derp()
    torch.manual_seed(0)
    tuned = tuned_agents.make_tuned_derp({})
    assert tuned.memory.maxlen == 500 and ref.epsilon == tuned.epsilon == 1.0
    assert all(torch.equal(a, b) for a, b in zip(ref.online_model.parameters(), tuned.online_model.parameters()))
    # frozen replay path: same seeds -> same weights after replays
    for ag in (ref, tuned):
        random.seed(1)
        _fill(ag, 40, 6)
        random.seed(2)
        ag.replay(8)
    assert all(torch.equal(a, b) for a, b in zip(ref.online_model.parameters(), tuned.online_model.parameters()))
    assert tuned.epsilon == ref.epsilon == pytest.approx(0.995)
    lref = agents.make_lstm(5)
    ltuned = tuned_agents.make_tuned_lstm(5, {})
    assert ltuned.epsilon == lref.epsilon == 0.2 and ltuned.memory.maxlen == 5000 and ltuned.target_update_every == 100


def test_derp_batched_double_dqn_target_lag_and_memory():
    opts = dict(batched_double_dqn=True, target_update_every=3, memory_size=5000)
    a = tuned_agents.make_tuned_derp(opts)
    assert a.memory.maxlen == 5000
    _fill(a, 64, 6)
    t0 = [p.clone() for p in a.target_model.parameters()]
    for _ in range(2):
        a.replay(16)
    assert a.grad_updates == 2
    assert all(torch.equal(x, y) for x, y in zip(t0, a.target_model.parameters()))      # not synced yet
    assert not all(torch.equal(x, y) for x, y in zip(a.online_model.parameters(), a.target_model.parameters()))
    a.replay(16)                                                                          # 3rd update -> sync
    assert all(torch.equal(x, y) for x, y in zip(a.online_model.parameters(), a.target_model.parameters()))
    assert len(a.loss_history) == 3


def test_double_dqn_target_formula():
    a = tuned_agents.make_tuned_derp(dict(batched_double_dqn=True))
    nx = torch.rand(5, 6)
    r, d = torch.rand(5), torch.tensor([0., 0., 1., 0., 0.])
    y = a.compute_targets(r, nx, d)
    with torch.no_grad():
        astar = a.online_model(nx).argmax(1)
        exp = r + a.gamma * (1 - d) * a.target_model(nx)[torch.arange(5), astar]
    assert torch.allclose(y, exp) and y[2] == r[2]


def test_scheduled_epsilon_identical_for_all_agents():
    cfg = dict(eps_schedule="linear", eps_start=1.0, eps_end=0.05, eps_decay_steps=1000)
    ags = [tuned_agents.make_tuned_derp(cfg), tuned_agents.make_tuned_lstm(None, cfg), tuned_agents.make_tuned_lstm(5, cfg)]
    assert ags[2].epsilon == 1.0                       # dinos_t starts at 1.0 like scratch
    for step in (0, 250, 1000, 5000):
        eps = []
        for ag in ags:
            tuned_agents.set_epsilon(ag, step)
            eps.append(ag.epsilon)
        assert eps[0] == eps[1] == eps[2] == pytest.approx(tuned_agents.eps_at("linear", step, 1.0, 0.05, 1000))
    # replay no longer decays eps
    _fill(ags[1], 40, 6, seq=10)
    e = ags[1].epsilon
    ags[1].replay(8)
    assert ags[1].epsilon == e


def test_reset_head_keeps_trunk_only():
    import agents
    ref = agents.make_lstm(5)
    torch.manual_seed(3)
    t = tuned_agents.make_tuned_lstm(5, {"reset_head": True})
    assert all(torch.equal(a, b) for a, b in zip(ref.online_model.lstm.parameters(), t.online_model.lstm.parameters()))
    assert not torch.equal(ref.online_model.fc.weight, t.online_model.fc.weight)
    assert torch.equal(t.online_model.fc.weight, t.target_model.fc.weight)
    with pytest.raises(ValueError):
        tuned_agents.make_tuned_lstm(None, {"reset_head": True})


def test_transfer_map():
    w = np.random.default_rng(0).random((10, 6))
    assert transfer_map.map_window(w, "none") is w
    m = transfer_map.map_window(w, "pretrain")
    assert m.shape == w.shape
    np.testing.assert_allclose(m[:, 0], w[:, 0] * 100)
    np.testing.assert_allclose(m[:, 5], w[:, 4] * 20)
    np.testing.assert_allclose(m[:, 1], 10 * m[:, 5])
    np.testing.assert_allclose(m[:, 2], w[:, 1] * 100)
    assert np.all(m[:, 3] == 52.5)
    np.testing.assert_allclose(m[:, 4], w[:, 2] * 2.4)
    with pytest.raises(ValueError):
        transfer_map.map_window(w, "x")


def test_episode_stats():
    st = diagnostics.EpisodeStats()
    for a, d in [(2, 1), (1, 0), (0, -1), (2, 1), (2, 1)]:
        st.add(a, -1.0, dict(accepted_change=d, ready=4, pending=1, bad_requests=1.0, arrived=10.0))
    s = st.summary()
    assert s["pod_changes"] == 4 and s["reversals"] == 2 and s["scale_up"] == 3 and s["scale_down"] == 1
    assert s["mean_pods"] == 5 and s["sla_violation_rate"] == pytest.approx(0.1) and s["action_counts"] == [1, 1, 3]


def test_quick_test_seed_guard():
    cfg = {"agent": "derp"}
    for tr in (-1, 1000, 20000):
        with pytest.raises(SystemExit):
            quick_test.check_seeds(tr, [1000])
    for ev in ([999], [1005], [20000], [1000, 1999]):
        with pytest.raises(SystemExit):
            quick_test.check_seeds(0, ev)
    quick_test.check_seeds(999, [1000, 1004])
    with pytest.raises(SystemExit):
        quick_test.run(cfg, 5000, 10, out=None)
    with pytest.raises(SystemExit):
        quick_test.run(cfg, 0, 10, eval_seeds=[20000])


@pytest.mark.parametrize("name", ["derp_tuned", "dinos_t_tuned", "lstm_scratch_tuned"])
def test_quick_test_runs(tmp_path, monkeypatch, name):
    import os
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    cfg = json.load(open(os.path.join(os.path.dirname(quick_test.__file__), "configs", name + ".json")))
    out = tmp_path / "r.json"
    # short episode via a tiny steps count; one dev seed to keep the unit test fast
    res = quick_test.run(cfg, 0, 60, eval_seeds=[1000], out=str(out), tel_dir=str(tmp_path))
    r = json.load(open(out))
    assert r["reward_mode"] == "dinos" and r["eval_seeds"] == [1000]
    ep = r["train_episodes"][0]
    for k in ("pod_changes", "reversals", "mean_pods", "sla_violation_rate", "td_loss", "mean_abs_q"):
        assert k in ep
    assert ep["complete"] is False and ep["steps"] == 60
    assert (tmp_path / "progress.json").exists() and (tmp_path / "metrics.jsonl").exists()


@pytest.mark.parametrize("which", ["derp", "derp_batched", "lstm_scratch", "dinos_t"])
def test_agents_really_train(which):
    """Weights change and grad_updates > 0 after a short replay run (regression for the grad_updates == 0 LSTM report)."""
    torch.manual_seed(0)
    random.seed(0)
    if which.startswith("derp"):
        agent = tuned_agents.make_tuned_derp({"batched_double_dqn": which == "derp_batched"})
        dim, seq = 6, None
    else:
        agent = tuned_agents.make_tuned_lstm(5 if which == "dinos_t" else None, {})
        dim, seq = 6, agent.sequence_length
    _fill(agent, 40, dim, seq)
    before = [p.detach().clone() for p in agent.online_model.parameters()]
    for _ in range(5):
        agent.replay(8)
    assert agent.grad_updates >= 5
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.online_model.parameters()))
    assert len(agent.loss_history) == 5


def test_learner_side_reward_options_and_grad_clip():
    a = tuned_agents.make_tuned_derp({"reward_transform": "symlog", "reward_clip": 3.0, "grad_clip": 0.5,
                                      "huber_beta": 2.0})
    s = np.zeros(6)
    for r in (-500.0, 0.5, 50.0, -1e6):
        a.remember(s, 0, r, s, False)
    got = [m[2] for m in a.memory]
    assert got == pytest.approx([-3.0, np.log1p(0.5), 3.0, -3.0])
    assert a.criterion.beta == 2.0
    _fill(a, 40, 6)
    for g in a.online_model.parameters():
        g.grad = torch.full_like(g, 100.0)
    a.optimizer.step()
    # the clip is applied before the step: the gradient norm is now <= 0.5
    n = torch.sqrt(sum((g.grad ** 2).sum() for g in a.online_model.parameters()))
    assert float(n) <= 0.5 + 1e-4
    with pytest.raises(ValueError):
        tuned_agents.check_cfg({"reward_transform": "tanh"})
    with pytest.raises(ValueError):
        tuned_agents.check_cfg({"grad_clip": 0})
    d = tuned_agents.make_tuned_derp({})
    d.remember(s, 0, -500.0, s, False)
    assert d.memory[0][2] == -500.0          # defaults off


def test_w2_configs_valid():
    import glob, os
    fs = glob.glob(os.path.join(os.path.dirname(quick_test.__file__), "configs", "w2", "*.json"))
    assert len(fs) == 8
    for f in fs:
        c = json.load(open(f))
        assert set(c) <= quick_test.CFG_KEYS
        tuned_agents.check_cfg(c["agent_options"])


def test_n_step_returns_and_gamma():
    a = tuned_agents.make_tuned_derp({"n_step": 3})
    assert a.gamma == pytest.approx(0.95 ** 3)
    g = 0.95
    for i in range(5):
        a.remember(np.full(6, i), 0, float(i + 1), np.full(6, i + 1), False)
    assert len(a.memory) == 3
    r0 = a.memory[0]
    assert r0[0][0] == 0 and r0[3][0] == 3 and r0[2] == pytest.approx(1 + 2 * g + 3 * g * g) and not r0[4]
    a.remember(np.full(6, 5), 0, 6.0, np.full(6, 6), True)        # terminal: flush the tail
    assert len(a.memory) == 6
    tail = list(a.memory)[3:]
    assert [m[0][0] for m in tail] == [3, 4, 5] and all(m[4] and m[3][0] == 6 for m in tail)
    assert tail[2][2] == pytest.approx(6.0)
    assert tail[1][2] == pytest.approx(5 + 6 * g)
    b = tuned_agents.make_tuned_derp({})
    assert b.gamma == 0.95 and b.n_step == 1
    for i in range(3):
        b.remember(np.zeros(6), 0, 1.0, np.zeros(6), False)
    assert len(b.memory) == 3


def test_n_step_lstm_target_uses_gamma_n():
    a = tuned_agents.make_tuned_lstm(None, {"n_step": 5})
    assert a.gamma == pytest.approx(0.98 ** 5)


def test_explore_repeat_training_only():
    np.random.seed(0)
    random.seed(0)
    a = tuned_agents.make_tuned_derp({"explore_repeat_max": 8})
    a.epsilon = 1.0
    s = np.zeros(6)
    acts = [a.act(s) for _ in range(4000)]
    runs = sum(1 for i in range(1, len(acts)) if acts[i] != acts[i - 1])
    base = tuned_agents.make_tuned_derp({})
    base.epsilon = 1.0
    ba = [base.act(s) for _ in range(4000)]
    bruns = sum(1 for i in range(1, len(ba)) if ba[i] != ba[i - 1])
    assert runs < 0.6 * bruns                      # persistent: far fewer action switches than iid random
    # long-run explore fraction stays ~eps: with eps=1 every step is random/held, never greedy
    a.reset_exploration()
    a.epsilon = 0.0                               # greedy evaluation: identical to the untouched agent
    g = [a.act(s) for _ in range(20)]
    a.online_model.eval()
    with torch.no_grad():
        want = int(torch.argmax(a.online_model(torch.tensor(s, dtype=torch.float32))))
    assert set(g) == {want}                        # greedy = argmax Q
    with pytest.raises(ValueError):
        tuned_agents.check_cfg({"explore_repeat_max": 0})
    with pytest.raises(ValueError):
        tuned_agents.check_cfg({"n_step": 0})


def test_input_adapter_matches_pretrain_map_at_init():
    ag = tuned_agents.make_tuned_lstm(5, {"reset_head": True}, input_adapter=True)
    x = np.random.default_rng(0).random((4, 10, 6))
    W, b = transfer_map.affine_pretrain()
    assert np.allclose(x @ W.T + b, transfer_map.map_window(x, "pretrain"))
    with torch.no_grad():
        got = ag.online_model.adapter(torch.tensor(x, dtype=torch.float32)).numpy()
    assert np.allclose(got, transfer_map.map_window(x, "pretrain"), rtol=1e-4, atol=1e-4)
    assert transfer_map.map_window(x, "adapter") is x
    assert any(p is ag.online_model.adapter.weight for g in ag.optimizer.param_groups for p in g["params"])
    assert len(ag.memory) == 0
    ag.update_target_model()
    _fill(ag, 40, 6, seq=10)
    before = ag.online_model.adapter.weight.detach().clone()
    ag.replay(16)
    assert not torch.equal(before, ag.online_model.adapter.weight)


def test_eval_pod_trajectory_and_w3_configs():
    import glob, os
    st = diagnostics.EpisodeStats()
    for i in range(700):
        st.add(1, 0.0, dict(accepted_change=0, ready=i % 5 + 1, pending=1, bad_requests=0, arrived=1))
    s = st.summary()
    assert (s["pods_t0"], s["pods_t100"], s["pods_t500"], s["pods_end"]) == (2, 2, 2, 6)
    fs = glob.glob(os.path.join(os.path.dirname(quick_test.__file__), "configs", "w3", "*.json"))
    assert len(fs) == 3
    for f in fs:
        c = json.load(open(f))
        assert set(c) <= quick_test.CFG_KEYS
        tuned_agents.check_cfg(c["agent_options"])
        assert c["learn_scale"] == 1e-3 and c["agent_options"]["n_step"] == 3


class _FakeSim:
    """Scripted sim: reward 1.0 per step, episode ends after `n` steps; records the actions it receives."""

    def __init__(self, n=100):
        self.n, self.t, self.actions = n, 0, []

    def step(self, a):
        self.actions.append(a)
        self.t += 1
        return np.full((10, 6), float(self.t)), 1.0, self.t >= self.n, {}


def _agent_with(cfg, proposals):
    a = tuned_agents.make_tuned_derp(cfg)
    it = iter(proposals)
    a.act = lambda s, *x, **k: next(it)
    return a


def test_decision_every_holds_and_stores_k_step_transition():
    a = _agent_with({"decision_every": 4}, [2, 0])
    assert a.gamma == pytest.approx(0.95 ** 4)
    sim, w = _FakeSim(), np.zeros((10, 6))
    st = lambda w: w[-1]
    w, done, used = tuned_agents.macro_step(a, sim, w, st, 0.5, 1000)
    assert used == 4 and sim.actions == [2, 1, 1, 1]
    assert len(a.memory) == 1
    s, act, r, s2, d = a.memory[0]
    assert act == 2 and r == pytest.approx(2.0) and s2[0] == 4 and not d
    # budget cut mid-macro: nothing stored
    tuned_agents.macro_step(a, sim, w, st, 0.5, 2)
    assert len(a.memory) == 1 and sim.actions[4:] == [0, 1]


def test_decision_every_composes_with_n_step_and_terminal():
    a = _agent_with({"decision_every": 2, "n_step": 2}, [2] * 10)
    assert a.gamma == pytest.approx(0.95 ** 4)          # gamma^(k n)
    sim, w = _FakeSim(n=6), np.zeros((10, 6))
    st = lambda w: w[-1]
    done = False
    while not done:
        w, done, _ = tuned_agents.macro_step(a, sim, w, st, 1.0, 1000)
    g2 = 0.95 ** 2
    assert len(a.memory) == 3
    m0, m1, m2 = a.memory
    assert m0[2] == pytest.approx(2 + 2 * g2) and m0[3][0] == 4 and not m0[4]
    assert m1[2] == pytest.approx(2 + 2 * g2) and m1[3][0] == 6 and m1[4]
    assert m2[2] == pytest.approx(2.0) and m2[4]


def test_scale_down_cooldown_converts_and_records_executed():
    a = _agent_with({"scale_down_cooldown": 3}, [2, 0, 0, 0, 0, 0, 1, 0])
    sim, w = _FakeSim(n=100), np.zeros((10, 6))
    st = lambda w: w[-1]
    for _ in range(8):
        w, _, _ = tuned_agents.macro_step(a, sim, w, st, 1.0, 1000)
    # up at t0; downs at t1-3 blocked; t4 passes (cooldown over), which restarts it: t5 blocked, t6 hold, t7 blocked
    assert sim.actions == [2, 1, 1, 1, 0, 1, 1, 1]
    assert [m[1] for m in a.memory] == sim.actions        # executed actions in replay


def test_control_defaults_off_and_validation():
    a = tuned_agents.make_tuned_derp({})
    assert a.ctl.k == 1 and a.ctl.m == 0 and a.gamma == 0.95
    sim, w = _FakeSim(), np.zeros((10, 6))
    a.act = lambda s: 0
    for _ in range(3):
        w, _, _ = tuned_agents.macro_step(a, sim, w, lambda x: x[-1], 1.0, 1000)
    assert sim.actions == [0, 0, 0]
    for bad in ({"decision_every": 0}, {"scale_down_cooldown": -1}, {"decision_every": 1.5}):
        with pytest.raises(ValueError):
            tuned_agents.check_cfg(bad)


def test_quick_test_with_control_runs(tmp_path):
    cfg = json.load(open(quick_test.os.path.join(quick_test.HERE, "configs", "w3", "lstm_scratch_w3.json")))
    cfg["agent_options"].update(decision_every=4, scale_down_cooldown=20)
    res = quick_test.run(cfg, 0, 60, eval_seeds=[1000], tel_dir=str(tmp_path))
    ac = res["train_episodes"][0]["action_counts"]
    assert sum(ac) == 60 and ac[1] >= 45                  # >= 3 of every 4 steps are hold
    assert res["eval"][0]["action_counts"][1] >= 0.75 * res["eval"][0]["steps"]


# ---- features: "demand" ----
def test_demand_features_shape_and_range():
    from tuning import features
    sim = features.make_sim("demand")
    w = sim.reset(1)
    assert w.shape == (10, 9) and np.all(w[:, 6:] == [0.0, 0.5, 0.0])
    for a in [2, 2, 1, 0, 1] * 20:
        w, _, _, info = sim.step(a)
    assert w.shape == (10, 9) and w[:, 6:].min() >= 0.0 and w[:, 6:].max() <= 1.0
    big = sim.P.mu_ops_per_pod * sim.P.max_pods
    assert w[-1, 6] == pytest.approx(info["lambda"] / big)
    assert w[-1, 8] == pytest.approx(min(info["lambda"] / (info["ready"] * sim.P.mu_ops_per_pod), 2.0) / 2.0)
    assert features.make_sim("base").reset(1).shape == (10, 6)


def test_demand_features_no_future_leakage():
    from tuning import features
    lam = np.random.default_rng(0).uniform(1000, 20000, 60)
    lam2 = lam.copy()
    lam2[30:] *= 3.0                                    # change only the future from step 30 on
    rows = []
    for tr in (lam, lam2):
        sim = features.DemandSim("W0", None, trace=tr)
        sim.reset(1)
        rows.append([sim.step(1)[0][-1].copy() for _ in range(30)])
    np.testing.assert_array_equal(np.array(rows[0]), np.array(rows[1]))   # rows 0..29 only see lam[0..29]
    sim = features.DemandSim("W0", None, trace=lam)
    sim.reset(1)
    for _ in range(9):
        w = sim.step(1)[0]
    big = sim.P.mu_ops_per_pod * sim.P.max_pods
    d = np.clip((lam[8] - lam[4]) / (0.25 * big), -1, 1)                  # K = 4 steps back, both past
    assert w[-1, 7] == pytest.approx(0.5 + 0.5 * d)


@pytest.mark.parametrize("agent,imap,pre", [("derp", "none", None), ("lstm_scratch", "none", None),
                                              ("lstm_scratch", "adapter", None), ("dinos_t", "adapter", 5)])
def test_demand_quick_test_runs(tmp_path, agent, imap, pre):
    cfg = dict(agent=agent, input_map=imap, features="demand", learn_scale=0.001, batch_size=16, agent_options={})
    if pre:
        cfg["pretrained"] = pre
    res = quick_test.run(cfg, 0, 60, eval_seeds=[1000], tel_dir=str(tmp_path))
    assert res["eval"][0]["steps"] > 0


def test_demand_adapter_is_n_to_6_and_dinos_t_needs_adapter():
    cfg = dict(agent="dinos_t", pretrained=5, input_map="adapter", features="demand", agent_options={})
    a = quick_test.build(cfg)
    assert tuple(a.online_model.adapter.weight.shape) == (6, 9) and a.state_size == 9
    with pytest.raises(SystemExit):
        quick_test.build(dict(cfg, input_map="none"))


# ---- cycle 1: reward variant d_sec, confirmation seeds, source pretraining ----
def _sim(variant, workload="W0"):
    from tuning import features as featmod, reward_variants
    from adapter import SimParams
    P = SimParams().with_(reward_mode="dinos")
    return reward_variants.wrap(featmod.make_sim("base", workload, P), variant)


def test_d_sec_replaces_only_the_D_term():
    a, b = _sim("v2"), _sim("d_sec")
    a.reset(0)
    b.reset(0)
    for act in [2, 2, 1, 0, 1, 1, 2, 0] * 5:
        _, r1, _, i1 = a.step(act)
        _, r2, _, i2 = b.step(act)
        c1, c2 = i1["dinos"], i2["dinos"]
        assert c2["D"] == pytest.approx(c1["p95_all_s"]) and c2["D_v2"] == pytest.approx(c1["D"])
        assert r2 - r1 == pytest.approx(-5.0 * (c1["p95_all_s"] - c1["D"]))
        assert r2 == pytest.approx(c2["r_L"] + c2["r_D"] + c2["r_N"] + c2["r_U"] + c2["r_M"])
        assert c2["r_N"] == c1["r_N"] and c2["r_U"] == c1["r_U"]


def test_d_sec_unit_matches_when_slo_is_1s_and_rejects_unknown():
    from tuning import reward_variants
    with pytest.raises(ValueError):
        reward_variants.wrap(_sim("v2"), "nope")
    assert _sim("v2").P.slo_s > 0


def test_eval_seed_blocks():
    quick_test.check_seeds(0, [1000, 1004, 1100, 1109])
    for bad in ([1005], [1110], [20000], [999], [10000]):
        with pytest.raises(SystemExit):
            quick_test.check_seeds(0, bad)
    with pytest.raises(SystemExit):
        quick_test.check_seeds(1000, [1000])


def test_quick_test_dsec_records_components_and_sum(tmp_path):
    cfg = {"agent": "derp", "reward_variant": "d_sec", "learn_scale": 0.001, "agent_options": {}}
    res = quick_test.run(cfg, 0, 60, eval_seeds=[1100, 1101], tel_dir=str(tmp_path))
    assert res["reward_variant"] == "d_sec" and res["eval_n_episodes"] == 2
    assert res["eval_sum_return"] == pytest.approx(2 * res["eval_mean_return"])
    for e in res["eval"]:
        tot = e["sum_3L"] + e["sum_neg5D"] + e["sum_neg10N"] + e["sum_neg05U"] + e["sum_neg05M"]
        assert tot == pytest.approx(e["ret"], rel=1e-9)
        assert e["failed_requests"] >= 0 and e["mean_latency_s"] >= 0 and "mean_pods" in e


def test_pretrain_source_checkpoint_loads_into_dinos_t(tmp_path):
    from tuning import pretrain_source
    out = str(tmp_path / "ck")
    m = pretrain_source.pretrain(40, 0, out, "d_sec", workload="W1", seeds=(5000, 5001))
    assert m["source_task"]["workload"] == "W1" and set(m["episode_sim_seeds"]) <= {5000, 5001}
    cfg = {"agent": "dinos_t", "pretrained_path": out + ".pth", "features": "demand", "input_map": "adapter",
           "agent_options": {"reset_head": True}}
    ag = quick_test.build(cfg)
    sd = torch.load(out + ".pth")
    assert torch.equal(ag.online_model.net.lstm.weight_ih_l0, sd["lstm.weight_ih_l0"])
    assert not torch.equal(ag.online_model.net.fc.weight, sd["fc.weight"])      # head reset
    w = ag.online_model.adapter.weight.detach()
    assert torch.equal(w[:, :6], torch.eye(6)) and float(w[:, 6:].abs().sum()) == 0.0
    res = quick_test.run(cfg, 0, 40, eval_seeds=[1000], tel_dir=str(tmp_path))
    assert res["eval_n_episodes"] == 1


# ---- cycle 2: load_def, extra diagnostics, trivial baselines ----
def test_served_load_changes_only_L_term():
    from tuning import features as featmod, reward_variants
    from adapter import SimParams
    P = SimParams().with_(reward_mode="dinos")
    for variant in ("v2", "d_sec"):
        a = reward_variants.wrap(featmod.make_sim("base", "W0", P), variant, "offered")
        b = reward_variants.wrap(featmod.make_sim("base", "W0", P), variant, "served")
        a.reset(3, init_pods=2)   # under-provisioned so served < offered
        b.reset(3, init_pods=2)
        differs = False
        for act in [1, 1, 1, 0, 1, 1, 1, 0] * 8:
            _, r1, _, i1 = a.step(act)
            _, r2, _, i2 = b.step(act)
            c1, c2 = i1["dinos"], i2["dinos"]
            Pm, K = b.env.P, b.env.dinos
            assert c2["L"] == pytest.approx(10.0 * i2["served"] / (15.0 * Pm.mu_ops_per_pod))
            assert r2 - r1 == pytest.approx(c2["r_L"] - c1["r_L"])
            assert r2 == pytest.approx(c2["r_L"] + c2["r_D"] + c2["r_N"] + c2["r_U"] + c2["r_M"])
            for k in ("r_D", "r_N", "r_U", "r_M", "D", "N", "U"):
                assert c2[k] == pytest.approx(c1[k])
            assert c2["L"] <= c1["L"] * (1 + 1e-9)
            differs |= c2["L"] < c1["L"] - 1e-9
        assert differs


def test_load_def_default_unchanged_and_validated(tmp_path):
    from tuning import features as featmod, reward_variants
    from adapter import SimParams
    P = SimParams().with_(reward_mode="dinos")
    s = featmod.make_sim("base", "W0", P)
    assert reward_variants.wrap(s, "v2") is s and reward_variants.wrap(s, "v2", "offered") is s
    with pytest.raises(ValueError):
        reward_variants.wrap(s, "v2", "nope")
    with pytest.raises(SystemExit):
        quick_test.build({"agent": "derp", "load_def": "nope"})
    cfg = {"agent": "derp", "reward_variant": "v2", "load_def": "served", "learn_scale": 0.001, "agent_options": {}}
    res = quick_test.run(cfg, 0, 60, eval_seeds=[1000], tel_dir=str(tmp_path))
    assert res["load_def"] == "served"
    e = res["eval"][0]
    assert e["failed_over_arrived"] >= 0 and e["req_weighted_latency_s"] >= 0 and e["churn_pods"] >= 0
    assert quick_test.run({**cfg, "load_def": "offered"}, 0, 60, eval_seeds=[1000], tel_dir=str(tmp_path))["load_def"] == "offered"


def test_c2_configs_valid():
    import glob
    fs = sorted(glob.glob(str(__import__("pathlib").Path(quick_test.HERE) / "configs" / "c2" / "*_served.json")))
    assert len(fs) == 6
    for f in fs:
        c = json.load(open(f))
        assert c["load_def"] == "served" and set(c) <= quick_test.CFG_KEYS
        base = json.load(open(f.replace("/c2/", "/c1/").replace("_served", "")))
        assert {k: v for k, v in c.items() if k != "load_def"} == base


def test_trivial_baselines_deterministic():
    from tuning import trivial_baselines as tb
    a = tb.run_fixed(6, "v2", "served", seeds=(1000,))
    b = tb.run_fixed(6, "v2", "served", seeds=(1000,))
    assert a["sum"] == b["sum"] and a["failed"] == b["failed"]
    assert tb.run_fixed(6, "v2", "offered", seeds=(1000,))["churn"] == 0.0
    assert a["pods"] == pytest.approx(6.0)


def test_hpa_baseline_deterministic():
    from tuning import trivial_baselines as tb
    a = tb.run_hpa(40, "d_sec", "served", seeds=(1000,))
    b = tb.run_hpa(40, "d_sec", "served", seeds=(1000,))
    assert a["target"] == 40 and a["sum"] == b["sum"] and a["failed"] == b["failed"]
    assert 1.0 <= a["pods"] <= 30.0 and a["churn"] > 0.0
