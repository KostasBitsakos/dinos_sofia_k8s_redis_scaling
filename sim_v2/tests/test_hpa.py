"""K8sHPA (docs/hpa_spec.md, kubernetes v1.34.0) and the HPA env interface (replica_actions, util_obs)."""
import numpy as np
import pytest

from env import OBS_NAMES, SimParams, SimV2
from policies import HPA_TARGET_GRID, K8sHPA, hpa_replica_calc, run_episode
import policies

U = len(OBS_NAMES)   # index of cpu_util_pct


def obs(ready, pending, util):
    o = np.zeros(U + 1)
    o[4], o[5], o[U] = ready, pending, util
    return o


def fresh(target=50, **kw):
    h = K8sHPA(target, **kw)
    h.reset()
    return h


# ---------- replica calculator (spec §1, §2)
def test_tolerance_band_inclusive_and_truncation():
    # target 50: ratio in [0.9, 1.1] -> keep current
    for util in (45.0, 50.0, 55.0, 55.99):           # 55.99 truncates to 55 -> 1.1, inside
        assert hpa_replica_calc(6, 6, 0, 0, util, 50) == 6
    assert hpa_replica_calc(6, 6, 0, 0, 56.0, 50) == 7      # 1.12 -> ceil(6.72)
    assert hpa_replica_calc(6, 6, 0, 0, 44.9, 50) == 6      # truncates to 44 -> 0.88 -> ceil(5.28) = 6
    assert hpa_replica_calc(6, 6, 0, 0, 40.0, 50) == 5      # 0.8 -> ceil(4.8)
    assert hpa_replica_calc(6, 6, 0, 0, -3.0, 50) == 0      # usage cannot be negative: 0 -> raw 0 (min applied later)


def test_no_metrics_is_error_path():
    assert hpa_replica_calc(6, 0, 6, 0, 80.0, 50) is None
    assert hpa_replica_calc(6, 6, 0, 0, float("nan"), 50) is None


def test_unready_pods_zero_on_scale_up_and_direction_flip():
    # ready 4 at 100 %, 2 unready, target 50: ratio 2 -> unready count 0 %: int(400/6) = 66 -> 1.32 -> ceil(7.92) = 8
    assert hpa_replica_calc(6, 4, 2, 0, 100.0, 50) == 8
    # ready 4 at 60 %: ratio 1.2 -> with 2 unready at 0 %: int(240/6) = 40 -> 0.8 < 1: direction flip -> current
    assert hpa_replica_calc(6, 4, 2, 0, 60.0, 50) == 6
    # without unready pods the same 1.2 would scale to ceil(1.2 * 4) = 5 on 4 ready pods (plain path)
    assert hpa_replica_calc(4, 4, 0, 0, 60.0, 50) == 5
    # scale-down: unready pods are ignored, ratio applies to ready pods only: ceil(0.4 * 4) = 2
    assert hpa_replica_calc(6, 4, 2, 0, 20.0, 50) == 2


def test_missing_metrics_fallback():
    # ratio 0.4 < 1: 2 missing pods get max(100, 50) = 100 %: int((80 + 200) / 6) = 46 -> 0.92 within -> current
    assert hpa_replica_calc(6, 4, 0, 2, 20.0, 50) == 6
    # ratio 2 > 1: missing pods get 0 %: int(400 / 6) = 66 -> 1.32 -> ceil(7.92) = 8
    assert hpa_replica_calc(6, 4, 0, 2, 100.0, 50) == 8
    # ratio 0.2: missing at 100 % -> int((40 + 200)/6) = 40 -> 0.8 -> ceil(4.8) = 5 (< current, same direction)
    assert hpa_replica_calc(6, 4, 0, 2, 10.0, 50) == 5


# ---------- behavior (spec §3)
def test_scale_up_cap_max_of_100pct_and_4_pods():
    h = fresh(50)
    assert h.act(obs(2, 0, 400.0)) == 6       # raw ceil(8 * 2) = 16; cap max(ceil(2 * 2), 2 + 4) = 6
    h = fresh(50)
    assert h.act(obs(10, 0, 400.0)) == 20     # raw 80; cap max(20, 14) = 20
    h = fresh(50)
    assert h.act(obs(25, 0, 400.0)) == 30     # cap 50 bounded by maxReplicas 30
    h = fresh(50)
    assert h.act(obs(4, 0, 60.0)) == 5        # small changes are not capped


def test_scale_up_has_no_stabilization():
    h = fresh(50)
    seq = [h.act(obs(6, 0, 30.0)) for _ in range(3)]      # wants 4, held by the initial recommendation
    assert seq == [6, 6, 6]
    assert h.act(obs(6, 0, 100.0)) == 12                   # immediate: raw ceil(2 * 6) = 12, cap 12


def test_scale_down_stabilization_300s_is_20_syncs():
    h = fresh(50)
    out = [h.act(obs(10, 0, 50.0)) for _ in range(5)]      # within tolerance: raw 10, stays 10
    assert out == [10] * 5
    # load drops: raw recommendation ceil(0.2 * 10) = 2, but the max over the last 300 s (20 syncs) is 10
    low = []
    for _ in range(25):
        low.append(h.act(obs(10, 0, 10.0)))
    # syncs 5..19 still see a raw 10 (syncs 0..4 or the initial record) in the window (s > t - 20)
    # last raw 10 stored at sync 4 -> excluded from sync 24 on (4 > 24 - 20 is false)
    assert low[:19] == [10] * 19
    assert low[19] == 2                                     # sync 24, 100 %/15 s policy allows the whole drop
    assert h.down_events[0] == (24, 8)                      # (the test keeps reporting 10 pods afterwards)


def test_initial_recommendation_blocks_scale_down_for_window():
    h = fresh(50)
    out = [h.act(obs(6, 0, 5.0)) for _ in range(22)]
    assert out[:20] == [6] * 20 and out[20] == 1           # raw ceil(0.1 * 6) = 1; initial 6 expires after 20 syncs


def test_stabilization_uses_stored_raw_not_applied():
    # The applied value is capped (2 -> 6) but the stored raw recommendation is 16; on the next sync the scale-down
    # window max is 16, so a lower raw value cannot shrink the deployment below current.
    h = fresh(50)
    assert h.act(obs(2, 0, 400.0)) == 6
    assert h.recs[-1] == (0, 16)
    assert h.act(obs(6, 0, 10.0)) == 6


def test_first_sync_without_metrics_does_not_scale():
    env = SimV2(SimParams().with_(episode_steps=30), workload="W0", **K8sHPA.env_kwargs)
    o, _ = env.reset(1000)
    assert np.isnan(o[U])
    h = fresh(40)
    assert h.act(o) == 6 and h.recs == [(0, 6)]


def test_custom_policies_and_periods():
    # Pods 1 / 3 syncs, select Max: added pods in the period reduce the allowance (periodStart = cur - added)
    h = fresh(50, up_policies=(("Pods", 1, 3),))
    assert [h.act(obs(c, 0, 400.0)) for c in (4, 5, 5, 5)] == [5, 5, 5, 6]


# ---------- env interface
def test_env_options_off_by_default_and_stream_unchanged():
    P = SimParams().with_(episode_steps=300)
    a = SimV2(P, workload="W0")
    b = SimV2(P, workload="W0", util_obs=True)
    c = SimV2(P, workload="W0", replica_actions=True)
    assert a.obs_names == OBS_NAMES and b.obs_names == OBS_NAMES + ("cpu_util_pct",)
    oa, _ = a.reset(1001)
    ob, _ = b.reset(1001)
    oc, _ = c.reset(1001)
    assert np.array_equal(oa, ob[:U]) and np.array_equal(oa, oc)
    rng = np.random.default_rng(5)
    for _ in range(300):
        act = int(rng.integers(0, 3))
        total = int(oc[4] + oc[5])
        oa, ra, ta, _, ia = a.step(act)
        ob, rb, *_ = b.step(act)
        desired = min(max(total + {0: -1, 1: 0, 2: 1}[act], 1), 30)
        oc, rc, *_ = c.step(desired)
        assert np.array_equal(oa, ob[:U]) and ra == rb == rc and np.array_equal(oa, oc)
        cpu_raw = ob[U]
        assert ob[1] == min(max(cpu_raw, 0.0), 100.0)        # cpu_pct is exactly the clipped util reading
        if ta:
            break


def test_replica_actions_multi_pod_and_range():
    env = SimV2(SimParams().with_(episode_steps=20), workload="W0", replica_actions=True)
    env.reset(1002)
    _, _, _, _, info = env.step(15)                           # +9 pods in one step (pending, d = 2)
    assert info["ready"] + info["pending"] == 15 and info["accepted_change"] == 9
    _, _, _, _, info = env.step(3)                            # cancels pending first
    assert info["ready"] + info["pending"] == 3
    for bad in (0, 31):
        with pytest.raises(ValueError):
            env.step(bad)


def test_hpa_episode_runs_and_respects_caps():
    P = SimParams().with_(episode_steps=600)
    env = SimV2(P, workload="W0", **K8sHPA.env_kwargs)
    h = K8sHPA(40)
    o, _ = env.reset(1003)
    h.reset()
    prev = 6
    big = 0
    for _ in range(600):
        d = h.act(o)
        assert 1 <= d <= 30 and d <= max(2 * prev, prev + 4)
        big += abs(d - prev) > 1
        o, r, term, _, info = env.step(d)
        prev = info["ready"] + info["pending"]
        if term:
            break
    assert big > 0                                            # multi-pod changes actually happen
    r = run_episode(SimV2(P, workload="W0", **K8sHPA.env_kwargs), K8sHPA(40), 1003)
    assert r["steps"] == 600 and np.isfinite(r["ret"])


def test_tune_hpa_tie_break_smallest_and_grid(monkeypatch):
    monkeypatch.setattr(policies, "run_episode", lambda env, pol, s: {"ret": -1.0})
    best, table = policies.tune_hpa(SimParams(), "W0", range(1000, 1005))
    assert [t for t, _ in table] == list(HPA_TARGET_GRID) and best.target == min(HPA_TARGET_GRID)
    with pytest.raises(ValueError):
        policies.tune_hpa(SimParams(), "W0", [20000])
