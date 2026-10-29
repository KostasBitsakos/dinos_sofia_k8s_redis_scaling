"""Reward adapter 'dinos' (R-20261017-017): hand-computed step, drop accounting, cost mode bit-identical."""
import hashlib
import json
import math
import os

import numpy as np
import pytest

from env import SimV2, SimParams, REWARD_DINOS_PATH

LN20 = math.log(20.0)
# sha256 of a 300-step trajectory (reward, obs, bad_requests, p95) in cost mode, computed with the code BEFORE
# the adapter was added (git HEAD 678c8a5 era) -- must never change.
GOLDEN_COST_300 = "90ccc4719d32e6b1111dcc68f11409c057db7b53309df47f06d4a4d6010149a7"


def _traj_hash(P, steps=300, seed=1000):
    env = SimV2(P.with_(episode_steps=steps), workload="W0")
    env.reset(seed)
    h = hashlib.sha256()
    for t in range(steps):
        a = (0, 1, 2, 2, 1, 0, 0)[t % 7] if t > 100 else 0
        o, r, term, _, info = env.step(a)
        h.update(np.float64([r]).tobytes())
        h.update(o.tobytes())
        h.update(np.float64([info["bad_requests"], info["p95_ms"]]).tobytes())
        if term:
            break
    return h.hexdigest()


def test_cost_mode_bit_identical():
    assert SimParams().reward_mode == "cost"
    assert _traj_hash(SimParams()) == GOLDEN_COST_300


def test_dinos_mode_does_not_change_dynamics_or_obs():
    # same obs/bad/p95 stream; only the reward differs
    a = _traj_hash(SimParams())
    b = _traj_hash(SimParams().with_(reward_mode="dinos"))
    assert a != b   # reward differs, hence hash differs
    env = SimV2(SimParams().with_(reward_mode="dinos", episode_steps=5), trace=[5000.0] * 5)
    env2 = SimV2(SimParams().with_(episode_steps=5), trace=[5000.0] * 5)
    env.reset(3), env2.reset(3)
    for _ in range(5):
        o1, _, _, _, i1 = env.step(1)
        o2, r2, _, _, i2 = env2.step(1)
        assert np.array_equal(o1, o2) and i1["reward_cost"] == i2["reward_cost"]


def test_hand_computed_step():
    # legacy p95/served-U definition (variant 'legacy_p95'); the default is V2, see test_default_is_v2
    P = SimParams().with_(reward_mode="dinos", init_pods=10, episode_steps=3, dinos_variant="legacy_p95")
    env = SimV2(P, trace=[5000.0] * 3)
    env.reset(0)
    _, r, _, _, info = env.step(1)
    # lam 5000, 10 pods, mu 1000: C = 10000, rho = .5, p95 = ln20/(1000*.5) = 5.99 ms, SLO = 2 ln20/500 -> D = .5
    # L = 10*5000/1000 = 50, N = 10, U = 50, M = 52.5
    c = info["dinos"]
    assert (c["L"], c["N"], c["M"]) == (50.0, 10.0, 52.5)
    assert c["D"] == pytest.approx(0.5, rel=1e-12) and c["U"] == pytest.approx(50.0)
    assert r == pytest.approx(3 * 50 - 5 * 0.5 - 10 * 10 - 0.5 * 50 - 0.5 * 52.5) == pytest.approx(-3.75)
    assert info["reward_dinos"] == r


def test_failed_requests_count_at_timeout():
    # 1 pod, lambda far above capacity for long enough that backlog times out: p95 over all requests = 60 s
    P = SimParams().with_(reward_mode="dinos", init_pods=1, min_pods=1, episode_steps=12, max_delta=1, dinos_variant="legacy_p95")
    env = SimV2(P, trace=[5000.0] * 12)
    env.reset(0)
    seen = []
    for _ in range(12):
        _, r, _, _, info = env.step(1)
        seen.append((info["failed"], info["dinos"]["p95_all_s"], info["dinos"]["D"]))
    assert any(f > 0 for f, _, _ in seen)
    big = [(f, p, d) for f, p, d in seen if f > 0.05 * (f + 1000 * 15)]
    assert big and all(p == 60.0 and d == pytest.approx(60.0 / P.slo_s) for _, p, d in big)


def test_constants_file_matches_old_sim_coefficients():
    k = json.load(open(REWARD_DINOS_PATH))
    assert (k["c_load"], k["c_latency"], k["c_pods"], k["c_cpu"], k["c_mem"]) == (3.0, 5.0, 10.0, 0.5, 0.5)
    assert k["memory_pct"] == 52.5 == (40 + 65) / 2


# ---------------------------------------------------------------- Codex review follow-ups (R-20261017-017)
def _fresh_env(variant="legacy_p95"):
    # the tests below pin properties of the legacy p95 definition unless a variant is requested
    P = SimParams().with_(reward_mode="dinos", init_pods=10, episode_steps=3, dinos_variant=variant)
    env = SimV2(P, trace=[5000.0] * 3)
    env.reset(0)
    return env, P


def _reward(env, lam, served, failed, C=10000.0, billed=10.0):
    env._mix = None   # fast-path state: no queue, single M/M/1 component at rho = lam / C
    r, c = env._dinos_reward(lam, served, failed, C, billed, env.P.control_step_s)
    return r, c


def test_terminal_backlog_is_charged_as_failures_in_the_last_step():
    # 1 pod, load 5x capacity: the queue never drains; the final step settles the backlog as failures
    P = SimParams().with_(reward_mode="dinos", init_pods=1, min_pods=1, episode_steps=6, max_delta=1)
    env = SimV2(P, trace=[5000.0] * 6)
    env.reset(0)
    served = failed = arrived = 0.0
    last = None
    for t in range(6):
        _, r, term, _, info = env.step(1)
        served, failed, arrived = served + info["served"], failed + info["failed"], arrived + info["arrived"]
        if t < 5:
            assert info["settled"] == 0.0 and info["outstanding"] > 0.0
            # outstanding arrivals are in neither count of this step (they are counted when they resolve)
            assert info["served"] + info["failed"] < info["arrived"] + 1e-9 or info["failed"] > 0.0
        last = info
        assert term == (t == 5)
    assert last["settled"] > 0.0 and last["outstanding"] == 0.0
    assert last["failed"] >= last["settled"]                    # settled backlog is inside the last step's failed
    assert last["dinos"]["p95_all_s"] == P.timeout_s            # settled failures >> 5% of the step -> D at timeout
    assert last["dinos"]["D"] == pytest.approx(P.timeout_s / P.slo_s)
    assert served + failed == pytest.approx(arrived, rel=1e-9)  # nothing is lost or counted twice
    assert env.conservation_residual() == pytest.approx(0.0, abs=1e-6)


def test_no_terminal_charge_when_queue_is_empty():
    env, _ = _fresh_env()
    for _ in range(3):
        _, _, term, _, info = env.step(1)
    assert term and info["settled"] == 0.0 and info["failed"] == 0.0


def test_same_state_substitution_high_utilisation_failed_is_worse():
    # identical state / capacity / billed pods / offered load; only the split of the SAME requests changes
    env, P = _fresh_env()
    lam, T = 9000.0, P.control_step_s
    arr = lam * T
    base, _ = _reward(env, lam, arr, 0.0)
    prev = base
    for f in (1e-4, 1e-3, 0.01, 0.03, 0.049):
        r, _ = _reward(env, lam, arr * (1 - f), arr * f)
        assert r < prev, f          # strictly decreasing in the failed share (below the threshold) at this load
        prev = r
    assert prev < base - 10.0


def test_p95_failure_threshold_boundary():
    env, P = _fresh_env()
    lam, T = 5000.0, P.control_step_s
    n = 100.0
    just_below, c1 = _reward(env, lam, 95.001, 4.999)    # 4.999 % failed: p95 is a served latency
    exactly, c2 = _reward(env, lam, 95.0, 5.0)           # exactly 5 %: the p95 request is a failure -> timeout
    just_above, c3 = _reward(env, lam, 94.999, 5.001)
    assert c1["p95_all_s"] < P.timeout_s and c1["D"] < 5.0
    assert c2["p95_all_s"] == P.timeout_s == c3["p95_all_s"]
    assert c2["D"] == pytest.approx(P.timeout_s / P.slo_s)
    assert just_below > just_above + 1e4 and just_below > exactly + 1e4   # a cliff of D = timeout/SLO ~ 5000
    # the other side of the cliff is flat in D: more failures only lower U (documented, see next test)
    assert c3["D"] == c2["D"]


def test_known_non_monotone_regions_of_the_adapter_are_pinned():
    """Properties of the FROZEN adapter formula, pinned so a change is noticed (not a claim that they are desirable):
    (a) at low utilisation, failing 1 % of the requests RAISES the reward (U falls by 0.5 * 100 f / ... faster than
        D rises), i.e. the failure-substitution ordering does not hold at rho ~ 0.5 for small failed shares;
    (b) above the 5 % cliff D is constant, so a larger failed share raises the reward again (U falls)."""
    env, P = _fresh_env()
    lam, T = 5000.0, P.control_step_s
    arr = lam * T
    r0, _ = _reward(env, lam, arr, 0.0)
    r1, _ = _reward(env, lam, arr * 0.99, arr * 0.01)
    assert r1 > r0                                       # (a)
    r6, _ = _reward(env, lam, arr * 0.94, arr * 0.06)
    r30, _ = _reward(env, lam, arr * 0.70, arr * 0.30)
    assert r30 > r6 and r30 < r0 - 1e4                   # (b), but still catastrophically below no-failure


# ---------------------------------------------------------------- reward variants (prototype, default unchanged)
def test_variants_default_unchanged_and_mean_variant_monotone():
    k = json.load(open(REWARD_DINOS_PATH))
    assert (k["d_def"], k["u_def"]) == ("mean", "offered") and k["variants"]["default"] == k["variants"]["V2"]
    assert k["variants"]["legacy_p95"] == {"d_def": "p95", "u_def": "served"}
    env, P = _fresh_env()
    lam, arr = 5000.0, 5000.0 * P.control_step_s
    r0, _ = _reward(env, lam, arr, 0.0)
    env.dinos = {**env.dinos, **k["variants"]["V2"]}
    prev = None
    for f in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5):
        r, c = _reward(env, lam, arr * (1 - f), arr * f)
        assert prev is None or r < prev
        prev = r
    assert c["U"] == pytest.approx(50.0)     # offered-demand U does not fall when requests fail
    r_v, _ = _reward(env, lam, arr, 0.0)
    D = 0.002 / P.slo_s    # mean M/M/1 sojourn 1/(mu(1-rho)) = 2 ms
    assert r_v == pytest.approx(3 * 50 - 5 * D - 10 * 10 - 0.5 * 50 - 0.5 * 52.5)


def test_default_is_v2_and_legacy_selectable():
    """R-20261017-023: reward_mode='dinos' defaults to V2; dinos_variant='legacy_p95' reproduces the old reward."""
    assert SimParams().dinos_variant == "default"
    env, P = _fresh_env("default")
    assert (env.dinos["d_def"], env.dinos["u_def"]) == ("mean", "offered")
    env.step(1)
    lam, arr = 5000.0, 5000.0 * P.control_step_s
    r0, c0 = _reward(env, lam, arr, 0.0)
    assert c0["D"] == pytest.approx(0.002 / P.slo_s) and c0["U"] == pytest.approx(50.0)   # mean M/M/1 sojourn 2 ms
    r1, c1 = _reward(env, lam, arr * 0.5, arr * 0.5)
    assert c1["U"] == pytest.approx(50.0) and r1 < r0 - 1e3                                # failures never help
    leg, _ = _fresh_env("legacy_p95")
    assert (leg.dinos["d_def"], leg.dinos["u_def"]) == ("p95", "served")
    with pytest.raises(ValueError):
        _fresh_env("nope")
