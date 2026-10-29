"""Cycle 7 oracles/controllers (tuning/legacy/c7_oracles.py)."""
import itertools

import numpy as np

from tuning.legacy import c5_protocol as C5
from tuning.legacy import c7_oracles as O
from tuning.legacy import coupled_env as CE
from tuning.legacy import run_legacy as RL


def test_reward_matrix_bit_identical_to_env():
    ticks = 400
    L, c0, m0, l0 = O.draws(1100, ticks)
    R = O.reward_matrix(L, c0, m0, l0)
    env = C5.make_env("lstm_transfer", "coupled", 1100)
    env.reset()
    rng = np.random.RandomState(7)
    for t in range(ticks):
        out = env.step(int(rng.randint(3)))
        n = int(out[0][5])
        assert out[1] == R[t, n - 1]                  # exact, not approximate


def test_replay_equals_computed_for_every_policy():
    r = O.trace_policies(1101, 5, ticks=300)          # raises on any mismatch
    for p in O.POLICIES:
        assert r[p]["ret"] == r[p]["computed_ret"]
        assert r[p]["ticks"] == 300


def test_draws_match_c6_baseline_trace():
    # fixed N=5 ramped on val: the same value coupled_baselines.py wrote in results/c6 (bit-identical traces)
    N, means = O.val_select()
    assert N == 5
    assert means["5"] == 478225.56210234325
    assert means["6"] == 475126.8785292641


def test_dp_is_optimal_brute_force():
    L, c0, m0, l0 = O.draws(1102, 6)
    R = O.reward_matrix(L, c0, m0, l0)
    best = max(O.path_return(R, a)[0] for a in itertools.product((0, 1, 2), repeat=6))
    acts, v = O.dp_actions(R)
    assert abs(O.path_return(R, acts)[0] - best) < 1e-9
    assert abs(v - best) < 1e-9


def test_dp_ge_greedy_ge_fixed_short_trace():
    L, c0, m0, l0 = O.draws(1100, 300)
    R = O.reward_matrix(L, c0, m0, l0)
    dp = O.path_return(R, O.dp_actions(R)[0])[0]
    gr = O.path_return(R, O.greedy_actions(R))[0]
    fixed = {N: O.path_return(R, O.fixed_actions(N, 300))[0] for N in range(1, 21)}
    assert dp >= gr - 1e-9
    assert all(dp >= f - 1e-9 for f in fixed.values())       # DP bounds every policy (theorem)
    assert gr >= fixed[5]          # empirical on this trace only: greedy >= fixed is NOT a theorem in general
    ER = O.expected_reward_matrix(L)
    assert dp >= O.path_return(R, O.dp_actions(ER)[0])[0] - 1e-9
    assert dp >= O.path_return(R, O.greedy_actions(ER))[0] - 1e-9


def test_bounds_and_moves():
    R = np.zeros((50, 20))
    R[:, 0] = 1.0                                     # pushes down to 1
    acts = O.dp_actions(R)[0]
    _, pods = O.path_return(R, acts)
    assert min(pods) == 1 and all(abs(a - b) <= 1 for a, b in zip([6] + pods, pods))
    R2 = np.zeros((50, 20))
    R2[:, 19] = 1.0
    _, pods = O.path_return(R2, O.dp_actions(R2)[0])
    assert pods[13] == 20 and max(pods) == 20
    _, pods = O.path_return(R2, O.greedy_actions(R2))      # myopic: no local gain, holds at 6
    assert set(pods) == {6}
    R3 = np.tile(np.arange(1.0, 21.0), (50, 1))            # reward increasing in n: greedy climbs to 20
    _, pods = O.path_return(R3, O.greedy_actions(R3))
    assert pods[13] == 20 and max(pods) == 20


def test_expected_reward_monte_carlo():
    rng = np.random.RandomState(0)
    E = O._E
    for t, L in ((10, 60.0), (1550, 150.0), (300, 115.0)):
        sp = E.spike_position <= t < E.spike_position + E.spike_length
        bump = E.spike_amplitude if sp else 0.0
        k = 200000
        c0 = rng.uniform(*E.cpu_range, size=k) + bump
        m0 = rng.uniform(*E.memory_range, size=k) + bump
        mu = E.latency_amplitude * np.sin(2 * np.pi * 2 * t / E.load_period) + E.latency_amplitude
        if sp:
            mu += E.spike_amplitude * 0.8 * abs(np.sin(2 * np.pi * 20 * (t - E.spike_position) / E.spike_length))
        l0 = np.maximum(mu + rng.normal(0, 0.1, size=k), 0)
        ER = O.expected_reward_matrix(np.array([L]), ticks=np.array([t]))[0]
        for n in (1, 3, 5, 12):
            g = 6 * L / (70 * n)
            mc = np.mean(3 * L - 5 * g * l0 - 10 * n - 0.5 * np.minimum(100, g * c0) - 0.5 * np.minimum(100, g * m0))
            assert abs(ER[n - 1] - mc) < 0.15, (t, L, n, ER[n - 1], mc)


def test_causal_uses_window_only():
    assert O.causal_inputs() == ("window",)
    ctl = O.CausalController(trend=True)
    assert set(vars(ctl)) == {"trend", "c_rng", "m_rng"}       # no env, trace or clock handle
    seen = []
    env = C5.make_env("lstm_transfer", "coupled", 1103)
    s0 = env.reset()
    w = C5.window0(s0, True)
    for _ in range(60):
        seen.append(w.copy())
        a = ctl.act(w)
        out = env.step(a)
        w = C5.push(w, out[0], True)
    for x in seen:
        assert x.shape == (RL.SEQ, RL.STATE)
    # decisions are a pure function of the window: replaying the recorded windows gives the same actions, and a
    # window is fully determined before the next draws are made (prefix of a longer replay is unchanged)
    a60 = O.replay(1103, O.CausalController(trend=True), 60)["actions"]
    a30 = O.replay(1103, O.CausalController(trend=True), 30)["actions"]
    assert a60[:30] == a30
    assert [ctl.act(x) for x in seen] == a60


def test_causal_decouples_latency_exactly():
    env = C5.make_env("lstm_transfer", "coupled", 1104)
    env.reset()
    out = env.step(2)                                   # 7 pods at tick 1
    L, n, lat = out[0][0], out[0][5], out[0][4]
    g = CE.REF_PODS * L / (CE.REF_LOAD * n)
    raw = O.draws(1104, 1)
    assert np.isclose(lat / g, raw[3][0], rtol=1e-12)    # de-coupled latency = the env's raw latency draw
