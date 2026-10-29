"""Sim v2 unit tests (R-20261016-013). Seconds each; no experiment batches."""
import math
import time

import numpy as np
import pytest

from env import OBS_NAMES, SimParams, SimV2, ObsWindow, pod_transition, resolution_convergence, run_actions
from oracle import bad_rate_lower_bound, oracle_bound, oracle_solve
from policies import AlwaysMax, AlwaysMin, Fixed, OracleReplay, StaticRamp, Threshold, run_episode
from stats import holm, paired_ci, paired_one_sided, t_cdf, t_ppf
from workloads import SEED_RANGES, W2, check_seed, make_workload, pods_needed

P0 = SimParams()


# ---------- parameters ----------
def test_every_param_provisional_with_source():
    for name, default, unit, prov, src in SimParams.provenance():
        assert prov is True and src, name
    assert abs(P0.slo_s - 4 * math.log(20) / 1000) < 1e-12


# ---------- conservation ----------
@pytest.mark.parametrize("wl", ["W0", "W1", "W3"])
@pytest.mark.parametrize("pol", [AlwaysMin(), StaticRamp(15), Fixed(15), Threshold(50, 30, 1)])
def test_conservation(wl, pol):
    env = SimV2(P0.with_(episode_steps=400), workload=wl, debug=True)  # debug asserts every step
    out = run_episode(env, pol, 3)
    assert out["outstanding_end"] == 0.0
    assert abs(out["final_resid"]) <= 1e-9 * out["arrived"]
    assert out["max_resid"] <= 1e-9 * out["arrived"]


def test_failure_counted_once_and_settled():
    # 1 pod at 1000 ops/s, 3000 ops/s for 20 steps: backlog grows, times out, rest settled at the end
    P = P0.with_(init_pods=1, episode_steps=20)
    env = SimV2(P, trace=np.full(20, 3000.0), debug=True)
    out = run_episode(env, AlwaysMin(), 0)
    assert out["settled"] > 0 and out["failed"] > out["settled"]
    assert abs(out["served"] + out["failed"] - out["arrived"]) <= 1e-9 * out["arrived"]
    # every request is bad at most once: bad <= arrived
    assert out["bad"] <= out["arrived"] * (1 + 1e-12)


# ---------- pod lifecycle ----------
def test_startup_delay():
    P = P0.with_(init_pods=3, startup_delay_steps=2)
    env = SimV2(P, trace=np.full(10, 100.0))
    env.reset(0)
    seq = [env.step(a)[4] for a in (2, 1, 1, 1)]
    assert [(i["ready"], i["pending"]) for i in seq] == [(3, 1), (3, 1), (4, 0), (4, 0)]
    assert [i["billed_pod_s"] for i in seq] == [60.0, 60.0, 60.0, 60.0]


def test_removal_immediate_pending_cancelled_first():
    P = P0.with_(init_pods=3)
    r, p, _, dp, bill, _ = pod_transition(P, 3, (0, 0), 0, 2)
    assert (r, p, bill) == (3, (0, 1), 4)
    r, p, _, dp, bill, _ = pod_transition(P, r, p, 0, 0)  # down cancels the pending pod
    assert (r, p, bill, dp) == (3, (0, 0), 3, 1)
    r, p, _, dp, bill, _ = pod_transition(P, 3, (0, 0), 0, 0)  # down removes a ready pod now
    assert (r, bill) == (2, 2)
    r, p, _, dp, bill, _ = pod_transition(P, 1, (0, 0), 0, 0)  # floor at min_pods
    assert (r, dp) == (1, 0)


def test_plus_minus_two_actions():
    P = P0.with_(max_delta=2, init_pods=29)
    env = SimV2(P, trace=np.full(5, 100.0))
    env.reset(0)
    assert env.n_actions == 5
    i = env.step(4)[4]
    assert i["accepted_change"] == 1 and i["pending"] == 1  # clipped at 30
    with pytest.raises(ValueError):
        pod_transition(P0, 3, (0, 0), 0, 4)


# ---------- rebalance ----------
def test_rebalance_loss():
    P = P0.with_(init_pods=10, rebalance_loss=0.3, rebalance_steps=1)
    env = SimV2(P, trace=np.full(5, 1000.0))
    env.reset(0)
    caps = [env.step(a)[4]["capacity"] for a in (1, 0, 1, 1)]
    assert caps == pytest.approx([10000.0, 9000 * 0.7, 9000.0, 9000.0])
    P2 = P.with_(rebalance_steps=2)
    env = SimV2(P2, trace=np.full(5, 1000.0))
    env.reset(0)
    caps = [env.step(a)[4]["capacity"] for a in (0, 1, 1)]
    assert caps == pytest.approx([9000 * 0.7, 9000 * 0.7, 9000.0])
    # topology trigger: scale-up costs capacity when the pod becomes ready, not when requested
    P3 = P.with_(rebalance_trigger="topology")
    env = SimV2(P3, trace=np.full(5, 1000.0))
    env.reset(0)
    caps = [env.step(a)[4]["capacity"] for a in (2, 1, 1, 1)]
    assert caps == pytest.approx([10000.0, 10000.0, 11000 * 0.7, 11000.0])


# ---------- F1 regression ----------
@pytest.mark.parametrize("wl", ["W0", "W1"])
def test_f1_always_min_not_optimal(wl):
    P = P0.with_(episode_steps=600)
    env = SimV2(P, workload=wl)
    rets = {p.name: run_episode(env, p, 1000)["ret"] for p in (AlwaysMin(), StaticRamp(20), Fixed(20), AlwaysMax(),
                                                                Threshold(50, 30, 1))}
    assert max(rets, key=rets.get) != "always_min"
    assert rets["always_min"] < min(v for k, v in rets.items() if k != "always_min")


def test_reward_depends_on_pods_via_latency():
    # under the same load more pods lower the SLA term; fewer pods lower the cost term
    tr = np.full(50, 8000.0)
    lo = run_episode(SimV2(P0, trace=tr), Fixed(9), 0)
    hi = run_episode(SimV2(P0, trace=tr), Fixed(16), 0)
    assert hi["sla"] > lo["sla"] and hi["cost"] < lo["cost"]


# ---------- resolution convergence ----------
def test_resolution_convergence_helper():
    tr = make_workload("W0").trace(1000, 300)
    acts = [1] * 300  # 6 pods, overloaded most of the time: the queue path is exercised
    out, rel = resolution_convergence(P0.with_(episode_steps=300), tr, acts, dts=(1.0, 3.0, 15.0))
    assert set(out) == {1.0, 3.0, 15.0}
    print("resolution rel diff vs dt=1 s:", rel, {k: round(v["return"]) for k, v in out.items()})
    # no backlog -> identical at every resolution (fast path is resolution-free)
    o2, rel2 = resolution_convergence(P0.with_(episode_steps=50), np.full(50, 3000.0), [1] * 50)
    assert rel2["return"] < 1e-12
    assert all(math.isfinite(v["return"]) for v in out.values())


# ---------- seeding ----------
def test_seeding_determinism_and_splits():
    for wl in ("W0", "W1", "W3"):
        a = make_workload(wl).trace(10000, 500)
        b = make_workload(wl).trace(10000, 500)
        c = make_workload(wl).trace(10001, 500)
        assert np.array_equal(a, b) and not np.array_equal(a, c)
    e1, e2 = SimV2(workload="W1"), SimV2(workload="W1")
    o1 = run_episode(e1, Threshold(50, 30, 1), 7)
    o2 = run_episode(e2, Threshold(50, 30, 1), 7)
    assert o1 == o2
    assert not set(SEED_RANGES["train"]) & set(SEED_RANGES["tune"])
    assert not set(SEED_RANGES["tune"]) & set(SEED_RANGES["heldout"])
    with pytest.raises(ValueError):
        check_seed("tune", 5)


def test_w0_peak_about_25_pods():
    tr = make_workload("W0").trace(1000, 5000)
    assert 23 <= pods_needed(tr.max()) <= 26


def test_w2_chronological_split_train_scaling(tmp_path):
    # 10 rows: train = rows 0-5 (max 6), tune = rows 6-7, held-out = rows 8-9 (values 100, 200)
    vals = [1, 2, 3, 4, 5, 6, 50, 60, 100, 200]
    p = tmp_path / "t.csv"
    p.write_text("ts,rate\n" + "\n".join(f"{i},{v}" for i, v in enumerate(vals)) + "\n")
    with pytest.raises(ValueError):
        W2(str(p)).segments()  # two numeric columns: must be explicit
    w = W2(str(p), column="rate", peak_ops=600.0)
    seg = w.segments()
    assert seg["train"].max() == 600.0                      # scaled by the train max only
    assert sorted(seg["heldout"]) == [10000.0, 20000.0]     # held-out never rescaled to its own max
    assert set(w.trace(10003, 7)) <= {10000.0, 20000.0}     # held-out seed reads only held-out rows
    assert set(w.trace(1003, 5)) <= {5000.0, 6000.0}
    assert set(w.trace(3, 20)) <= set(seg["train"])
    with pytest.raises(ValueError):
        w.trace(5000, 5)


def test_fixed_starts_at_c_without_ramp():
    env = SimV2(P0.with_(episode_steps=30), workload="W1")
    out = run_episode(env, Fixed(14), 1000)
    assert out["changes"] == 0 and out["pod_steps"] == 14 * 30 and out["churn"] == 0.0
    ramp = run_episode(env, StaticRamp(14), 1000)
    assert ramp["changes"] == 8
    assert env.P.init_pods == 6  # params untouched by the per-episode override


def test_obs_window():
    env = SimV2(P0.with_(episode_steps=20), workload="W0")
    obs, _ = env.reset(0)
    w = ObsWindow(10, len(env.obs_names))
    x = w.reset(obs)
    assert x.shape == (10, 6) and np.allclose(x, obs)
    o2 = env.step(1)[0]
    x = w.push(o2)
    assert np.allclose(x[-1], o2) and np.allclose(x[-2], obs)


def test_default_obs_is_deployable_oracle_obs_flagged():
    assert OBS_NAMES == ("completed_ops", "cpu_pct", "p95_ms", "failed_per_s", "ready", "pending")
    tr = np.full(30, 9000.0)  # 6 pods -> overload, backlog > 0
    env = SimV2(P0, trace=tr)
    env.reset(0)
    for _ in range(10):
        obs, _, _, _, info = env.step(1)
    assert obs.shape == (6,) and info["outstanding"] > 0
    assert obs[0] == pytest.approx(info["served"] / 15.0)       # completed ops/s, not offered lambda
    assert obs[0] < 9000.0
    assert obs[3] == pytest.approx(info["failed"] / 15.0)
    envo = SimV2(P0, trace=tr, oracle_obs=True)
    envo.reset(0)
    for _ in range(10):
        obs, _, _, _, info = envo.step(1)
    assert envo.obs_names[-2:] == ("offered_lambda", "backlog")
    assert obs[6] == 9000.0 and obs[7] == pytest.approx(info["outstanding"])


# ---------- oracle bound ----------
def test_bad_rate_lower_bound_is_lower():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lam, C, m = rng.uniform(0, 3000), rng.uniform(100, 3000), rng.uniform(0.5, 20)
        h = bad_rate_lower_bound(lam, C, m)
        for f in np.linspace(0, min(lam, C), 50):
            assert h <= (lam - f) + f * math.exp(-m * (1 - f / C)) + 1e-9


def test_oracle_bound_tiny_instance():
    """Bound >= return of every policy, and of every one of 3**6 action sequences (brute force)."""
    import itertools
    P = P0.with_(max_pods=4, init_pods=2, episode_steps=6, startup_delay_steps=1, mu_ops_per_pod=1000.0)
    tr = np.array([500.0, 2500.0, 3800.0, 1500.0, 4200.0, 800.0])
    bound, acts = oracle_bound(P, tr)
    env = SimV2(P, trace=tr, debug=True)
    best = -np.inf
    for seq in itertools.product(range(3), repeat=6):
        ret, _, _ = run_actions(env, seq)
        assert ret <= bound + 1e-9
        best = max(best, ret)
    replay, _, _ = run_actions(env, acts)
    assert replay <= best + 1e-9
    print(f"tiny: bound={bound:.3f} brute-force best={best:.3f} oracle replay={replay:.3f}")


@pytest.mark.parametrize("wl", ["W0", "W1", "W3"])
def test_oracle_bound_dominates_policies(wl):
    P = P0.with_(episode_steps=800)
    tr = make_workload(wl).trace(10000, 800)
    bound, acts = oracle_bound(P, tr)
    env = SimV2(P, trace=tr)
    for pol in (AlwaysMin(), AlwaysMax(), StaticRamp(12), StaticRamp(22), Threshold(50, 30, 1), OracleReplay(acts)):
        assert run_episode(env, pol, 10000)["ret"] <= bound + 1e-6


def test_oracle_bound_per_start_state_covers_fixed():
    P = P0.with_(episode_steps=300)
    tr = make_workload("W0").trace(10000, 300)
    b, acts, by_start = oracle_solve(P, tr)
    assert by_start[P.init_pods] == pytest.approx(b)
    assert oracle_bound(P, tr, init_pods=20)[0] == pytest.approx(by_start[20])
    env = SimV2(P, trace=tr)
    for c in (5, 12, 20, 30):
        assert run_episode(env, Fixed(c), 10000)["ret"] <= by_start[c] + 1e-6


# ---------- stats ----------
def test_stats_helpers():
    assert abs(t_ppf(0.975, 9) - 2.262157) < 1e-5
    assert abs(t_cdf(1.833113, 9) - 0.95) < 1e-6
    r = paired_one_sided([1.0, 2.0, 3.0, 4.0], 0.025)
    assert r["p"] < 0.05 and r["lb"] > 0
    assert list(holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    m, lo, hi = paired_ci([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and lo == pytest.approx(2.5 - 2.5 ** 0.0 * 3.182446 * (1.6666667 ** 0.5) / 2, rel=1e-5)


def test_ranking_gate_detects_reversal_only():
    from gates import ranking
    base = {k: [0.0] * 5 for k in ("always_min", "always_max", "fixed_best", "static_ramp_best", "threshold")}
    def unit(label, thr):
        rets = {k: list(v) for k, v in base.items()}
        rets["threshold"] = thr
        return dict(workload="W0", label=label, full=label == "a", returns=rets)
    up, down, noisy = [1.0, 1.1, 0.9, 1.2, 1.0], [-1.0, -1.1, -0.9, -1.2, -1.0], [1.0, -1.0, 0.5, -0.5, 0.1]
    assert ranking([unit("a", up), unit("b", noisy)], frozen=False)[0]       # unclear, not a reversal
    assert not ranking([unit("a", up), unit("b", down)], frozen=False)[0]    # significant reversal


# ---------- runtime ----------
def test_runtime_5000_steps():
    env = SimV2(P0, workload="W1")
    t = time.perf_counter()
    out = run_episode(env, Threshold(50, 30, 1), 10000)
    dt = time.perf_counter() - t
    assert out["steps"] == 5000
    print(f"runtime 5000-step W1 threshold episode: {dt:.3f} s")
    env = SimV2(P0, workload="W0")
    t = time.perf_counter()
    run_episode(env, AlwaysMin(), 10000)
    print(f"runtime 5000-step W0 always_min (worst case, always overloaded): {time.perf_counter() - t:.3f} s")
    assert dt <= 2.0


# ---------- Sim v2.1 sensitivity knobs ----------
def _digest(P=None, wl="W0", seed=3, n=5000, tag=0):
    import hashlib
    env = SimV2(P or SimParams(), workload=wl)
    obs, _ = env.reset(seed)
    rng = np.random.default_rng(100 + tag)
    h = hashlib.sha256()
    h.update(obs.tobytes())
    for _ in range(n):
        a = int(rng.integers(0, 3)) if rng.random() < 0.3 else 1
        obs, r, term, _, info = env.step(a)
        h.update(obs.tobytes())
        for v in (r, info["p95_ms"], info["bad_requests"]):
            h.update(np.float64(v).tobytes())
        if term:
            break
    return h.hexdigest()


def test_v21_defaults_bit_identical_to_sim_v2():
    """Digests recorded from the Sim v2 code (commit 6cd0fc8) before the v2.1 knobs were added."""
    assert _digest() == "cf501e920519372a5f1e6a1d10e06e25122e7755c9a9d34966bf04edc97637a8"
    assert _digest(wl="W1", seed=10001, tag=1) == "94938cb1c0df3ccf0fdbb923ecd9bd5d78802837c55acae6d7ecc26fae0cc7be"
    assert _digest(SimParams(theta=0.9, rebalance_trigger="topology"), wl="W3", seed=10002, tag=2) == \
        "d4f60252e63e742c0dee327e1a5a699ab1cd931961e2ddc9559a1defd3158c15"


def test_v21_frozen_configs_valid():
    import sensitivity
    assert sensitivity.names()[0] == "default" and sensitivity.overrides("default") == {}
    assert sensitivity.params("default") == SimParams()
    fields_ = {f for f, *_ in SimParams.provenance()}
    for n in sensitivity.names():
        ov = sensitivity.overrides(n)
        assert set(ov) <= fields_
        P = sensitivity.params(n)
        assert P.slo_eff_s > 0
        if n != "default":
            assert ov and sensitivity._load()[n]["source"]


def test_v21_knob_effects():
    tr = np.full(40, 3000.0)
    base = SimV2(P0, trace=tr)
    o0, _ = base.reset(1)
    o1 = base.step(1)[0]
    # latency: p95 shifted/scaled, SLO tightened; capacity untouched
    Pl = P0.with_(latency_add_s=0.001, tail_scale=1.5)
    env = SimV2(Pl, trace=tr)
    env.reset(1)
    ol, r, _, _, il = env.step(1)
    b = SimV2(P0, trace=tr)
    b.reset(1)
    ob, rb, _, _, ib = b.step(1)
    assert il["p95_ms"] == pytest.approx(1.0 + 1.5 * ib["p95_ms"])
    assert il["capacity"] == ib["capacity"] and il["slo_violations"] > ib["slo_violations"]
    # cpu offset: observation only
    Pc = P0.with_(cpu_offset_pct=10.0)
    env = SimV2(Pc, trace=tr)
    env.reset(1)
    oc, rc, _, _, ic = env.step(1)
    assert rc == rb and ic["capacity"] == ib["capacity"] and ic["slo_violations"] == ib["slo_violations"]
    assert oc[1] == pytest.approx(ob[1] + 10.0)
    with pytest.raises(ValueError):
        SimParams(latency_add_s=1.0).slo_eff_s


@pytest.mark.parametrize("cfg", ["default", "rebalance_k14", "rebalance_delta_0.45", "latency_add_0.2ms",
                                 "tail_scale_1.5", "theta_0.8"])
def test_v21_oracle_matches_env_objective(cfg):
    """No-backlog regime (arrivals below capacity): the DP value equals the env return of its own actions
    up to the small slack of the fresh-work refusal relaxation; and the bound dominates other policies."""
    import sensitivity
    P = sensitivity.params(cfg).with_(episode_steps=300)
    tr = make_workload("W0").trace(10000, 300)
    bound, acts = oracle_bound(P, tr)
    env = SimV2(P, trace=tr)
    for pol in (AlwaysMax(), StaticRamp(14), Threshold(50, 30, 1), OracleReplay(acts)):
        assert run_episode(env, pol, 10000)["ret"] <= bound + 1e-6
    # exact check on a tiny instance with no overload (backlog-free): brute force best equals the bound
    import itertools
    Pt = P.with_(max_pods=4, init_pods=2, episode_steps=5, startup_delay_steps=1)
    t5 = np.array([300.0, 900.0, 1500.0, 1100.0, 400.0])
    bt, at = oracle_bound(Pt, t5)
    envt = SimV2(Pt, trace=t5, debug=True)
    best = max(run_actions(envt, s)[0] for s in itertools.product(range(3), repeat=5))
    replay = run_actions(envt, at)[0]
    assert best <= bt + 1e-9 and replay <= bt + 1e-9
    # relaxation allows refusing fresh work, so the bound may exceed the optimum only by a small relative gap
    assert bt - best <= 0.05 * abs(best) + 1e-9, (bt, best)


def test_sandbox_seed_guard(monkeypatch):
    """QRL_SANDBOX_SEEDS: reset() raises for a seed outside the ranges; unset = no check."""
    import pytest
    from env import SimParams, SimV2
    env = SimV2(SimParams().with_(episode_steps=3), trace=[100.0] * 3)
    monkeypatch.delenv("QRL_SANDBOX_SEEDS", raising=False)
    env.reset(seed=2000)
    monkeypatch.setenv("QRL_SANDBOX_SEEDS", "0-999,1000-1004")
    env.reset(seed=5)
    env.reset(seed=1004)
    for bad in (1005, 10000, -1):
        with pytest.raises(PermissionError):
            env.reset(seed=bad)
    import env as E
    with pytest.raises(PermissionError):
        E._sandbox_seed_guard(20000)
