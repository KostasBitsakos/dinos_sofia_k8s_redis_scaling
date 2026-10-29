"""Cycle 10: pod startup delay variants '<variant>_d<D>' (orig_variants.py), delay-aware baselines, 7-feature runs on
the c5 protocol, test7 split. D = 0 must reproduce the cycle-9 variant exactly."""
import json

import numpy as np
import pytest
import torch

from tuning.legacy import c5_protocol as C5
from tuning.legacy import orig_baselines as OB
from tuning.legacy import orig_variants as OV
from tuning.legacy import run_legacy as RL

BASES = ("orig", "spike300", "multispike", "noise15_multispike")


def _run(variant, seed, acts):
    e = OV.OrigEnv(variant, seed)
    out = [(e.reset(), None, None, None)]
    for a in acts:
        out.append(e.step(a))
    return out


def _seqs():
    """Several action sequences: uniform random, up-heavy, down-heavy, bursts."""
    rs = np.random.RandomState(11)
    return [rs.randint(0, 3, 1500), rs.choice(3, 1500, p=[0.1, 0.2, 0.7]), rs.choice(3, 1500, p=[0.7, 0.2, 0.1]),
            np.array(([2] * 30 + [1] * 20 + [0] * 40) * 16)]


# ---------------------------------------------------------------- parsing
def test_variant_names():
    assert OV.split_delay("orig_d5") == ("orig", 5) and OV.split_delay("spike300") == ("spike300", None)
    assert OV.split_delay("noise15_multispike_d10") == ("noise15_multispike", 10)
    assert OV.startup_delay("orig") is None and OV.startup_delay("orig_d0") == 0
    assert OV.state_size("orig") == 6 and OV.state_size("spike300_d5") == 7
    assert list(OV.obs_scale("orig_d5")) == [100.0, 100.0, 60.0, 70.0, 2.4, 20.0, 20.0]
    assert OV.obs_scale("orig") is OV.OBS_SCALE
    for bad in ("orig_d", "orig_dx", "foo_d5", "orig_d101", "orig_d-1"):
        with pytest.raises(SystemExit):
            OV.parse(bad)
    assert OV.parse("spike300_d5") == OV.parse("spike300")


# ---------------------------------------------------------------- parity D = 0
@pytest.mark.parametrize("base", BASES)
def test_d0_reproduces_cycle9_variant_exactly(base):
    for seed in (0, 1, 1100, 6007, [3, 5, 1]):
        for acts in _seqs():
            ref, d0 = _run(base, seed, acts), _run(base + "_d0", seed, acts)
            for (s1, r1, x1, i1), (s2, r2, x2, i2) in zip(ref, d0):
                assert len(s1) == 6 and len(s2) == 7 and s2[6] == 0.0
                assert np.array_equal(s1, s2[:6])
                assert r1 == r2 and x1 == x2 and i1 == i2


def test_d0_clone_and_reset_parity():
    a, b = OV.OrigEnv("orig", 4), OV.OrigEnv("orig_d0", 4)
    for _ in range(2):        # two episodes in a row (reset mid-stream)
        assert np.array_equal(a.reset(), b.reset()[:6])
        for act in _seqs()[0][:300]:
            pa, pb = a.clone().step(act), b.clone().step(act)
            ra, rb = a.step(act), b.step(act)
            assert np.array_equal(pa[0], ra[0]) and np.array_equal(pb[0], rb[0])
            assert np.array_equal(ra[0], rb[0][:6]) and ra[1] == rb[1]


# ---------------------------------------------------------------- delay semantics
@pytest.mark.parametrize("D", (1, 3, 7))
def test_pending_becomes_active_exactly_after_D_ticks(D):
    e = OV.OrigEnv(f"orig_d{D}", 2)
    e.reset()
    s, *_ = e.step(2)                              # request in step 1
    assert s[5] == 6 and s[6] == 1
    for k in range(2, D + 1):                      # steps 2..D: still pending
        s, *_ = e.step(1)
        assert s[5] == 6 and s[6] == 1, k
    s, *_ = e.step(1)                              # step D + 1 = request step + D: active
    assert s[5] == 7 and s[6] == 0


@pytest.mark.parametrize("base", ("orig", "spike300"))
def test_delayed_env_equals_base_env_with_shifted_requests(base):
    """Exogenous trace, capacity (ACTIVE VMs) and served are those of the base env in which each +1 happens D ticks
    later; the reward additionally pays 3 per pending pod."""
    D = 4
    acts = [2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1] + [1] * 20
    shifted = [1] * len(acts)
    for i, a in enumerate(acts):
        if a == 2:
            shifted[i + D] = 2
    dl, ref = _run(f"{base}_d{D}", 6001, acts), _run(base, 6001, shifted)
    for (s1, r1, _, i1), (s2, r2, _, i2) in list(zip(dl, ref))[1:]:
        assert np.array_equal(s1[:6], s2)
        p = s1[6]
        assert i1["served"] == i2["served"]
        assert i1["vm_cost"] == -3 * (s1[5] + p)
        assert r1 == pytest.approx(r2 - 3 * p, abs=1e-9)
        assert i1["served"] + i1["vm_cost"] == r1


def test_scale_down_cancels_pending_first_newest_first():
    D = 5
    e = OV.OrigEnv(f"orig_d{D}", 3)
    e.reset()
    e.step(2)                                      # step 1: pending [5]
    s, *_ = e.step(2)                              # step 2: pending [4, 5]
    assert s[5] == 6 and s[6] == 2
    s, *_ = e.step(0)                              # step 3: cancels the newest; active untouched
    assert s[5] == 6 and s[6] == 1
    for _ in (4, 5):                               # steps 4, 5: the first request is still pending
        s, *_ = e.step(1)
        assert s[5] == 6 and s[6] == 1
    s, *_ = e.step(1)                              # step 6 = request step 1 + D
    assert s[5] == 7 and s[6] == 0
    s, *_ = e.step(0)                              # nothing pending: an active pod goes
    assert s[5] == 6 and s[6] == 0


def test_bounds_on_active_plus_pending_and_cost():
    e = OV.OrigEnv("orig_d10", 5)
    e.reset()
    for _ in range(40):
        s, r, _, i = e.step(2)
        assert 1 <= s[5] and s[5] + s[6] <= 20
        assert i["vm_cost"] == -3 * (s[5] + s[6])
    assert s[5] + s[6] == 20 and s[6] == 0         # 14 requests, all active after 10 ticks
    for _ in range(40):
        s, r, _, i = e.step(0)
        assert s[5] >= 1 and s[6] == 0
    assert s[5] == 1


def test_delay_clone_is_independent():
    e = OV.OrigEnv("spike300_d3", 8)
    e.reset()
    e.step(2)
    c = e.clone()
    c.step(2)
    c.step(0)
    c.step(0)
    assert e.n_pending == 1 and e.num_vms == 6
    peek, real = e.clone().step(1), e.step(1)
    assert np.array_equal(peek[0], real[0]) and peek[1] == real[1]


def test_reset_clears_pending():
    e = OV.OrigEnv("orig_d5", 1)
    e.reset()
    e.step(2)
    s0 = e.reset()
    assert len(s0) == 7 and s0[6] == 0 and e.n_pending == 0 and s0[5] == 6


# ---------------------------------------------------------------- baselines
def test_rules_count_pending_pods():
    # load 50, capacity 30, 6 active -> read 0.5 -> target 10; with 4 pending the total is 10: hold
    s = [50, 30, 0, 0, 0, 6, 4]
    assert OB.causal(False)(None)(s) == 1 and OB.causal(True)(None)(s) == 1
    assert OB.causal(False)(None)(s[:6]) == 2
    assert OB.hold(10)(None)(s) == 1 and OB.hold(9)(None)(s) == 0
    # hpa_demand: capacity once pending are up = 100 * 9 / 6 = 150; load 100 < 0.8 * 150 -> hold, not up
    act = OB.hpa_demand(0.8)(None)
    assert act([100, 100, 0, 0, 0, 6]) == 2 and act([100, 100, 0, 0, 0, 6, 0]) == 2
    assert act([100, 100, 0, 0, 0, 6, 3]) == 1


def test_causal_trend_d_extrapolates_D_plus_1_ticks():
    env = OV.OrigEnv("orig_d10", 0)
    tr = OB.causal_trend_d(5)(env)
    s = lambda load: [load, 30, 0, 0, 0, 6, 0]     # read_hat 0.5 -> target ceil(load_hat / 5)  # noqa: E731
    for load in (10, 12, 14, 16, 18):              # < k past loads: no trend; 18 -> target 4 -> down
        assert tr(s(load)) == 0
    # load_hat = 20 + 11 * (20 - 10) / 5 = 42 -> target 9 > 6 -> up (plain causal: 20 -> 4 -> down)
    assert tr(s(20)) == 2 and OB.causal(False)(None)(s(20)) == 0
    # Codex example: D=5, loads 20..25 -> load_hat = 25 + 6 * 5 / 5 = 31 -> target 7 > 6 -> up (D-tick: 30 -> 6, hold)
    tr5 = OB.causal_trend_d(5)(OV.OrigEnv("orig_d5", 0))
    acts = [tr5(s(load)) for load in (20, 21, 22, 23, 24, 25)]
    assert acts[-1] == 2


def test_oracle_dstep_uses_true_future():
    D = 3
    e = OV.OrigEnv(f"orig_d{D}", 1101)
    s = e.reset()
    act = OB.oracle_dstep(e)
    for _ in range(200):
        c = e.clone()
        for _ in range(D + 1):
            f = c.step(1)[0]
        want = OB.toward(OB.pods(s), OB.causal_target([f[0], f[1], 0, 0, 0, f[5]], break_even=True))
        a = act(s)
        assert a == want
        s = e.step(a)[0]


def test_plain_variants_keep_the_cycle9_policy_set():
    assert set(OB.policies("orig")) == set(OB.POLICIES)
    assert set(OB.policies("orig_d0")) == set(OB.POLICIES) | {"causal_trend_d", "oracle_Dstep"}


def test_baselines_short_trace_delay(monkeypatch, tmp_path):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test5", (6000, 6001))
    out = tmp_path / "b.json"
    res = OB.main(["--orig-variant", "spike300_d5", "--ticks", "80", "--out", str(out)])
    assert json.load(open(out))["startup_delay"] == 5
    for sp in ("val", "test5"):
        assert set(res[sp]) == {"fixed_best", "hpa_demand", "hpa_rule", "always_down", "causal", "causal_be",
                                "causal_trend", "oracle_1step", "causal_trend_d", "oracle_Dstep"}
        for v in res[sp].values():
            assert all(abs(e["served"] + e["vm_cost"] - e["ret"]) < 1e-6 for e in v["episodes"])
            assert "mean_pending" in v["episodes"][0]
    with pytest.raises(SystemExit):
        OB.main(["--orig-variant", "orig_dx", "--ticks", "10", "--out", str(out)])


def test_baselines_d0_equal_plain_cycle9_rules():
    """Without pending pods every rule is the cycle-9 rule: same returns on the same trace."""
    for name in ("causal", "causal_be", "causal_trend", "hpa_rule", "oracle_1step"):
        r0 = OB.evaluate("orig", OB.POLICIES[name], (1100,), 150)["per_trace"]["1100"]
        r1 = OB.evaluate("orig_d0", OB.POLICIES[name], (1100,), 150)["per_trace"]["1100"]
        assert r0 == r1, name
    for mk in (OB.hold(9), OB.hpa_demand(0.8)):
        assert OB.evaluate("orig", mk, (1101,), 150)["per_trace"] == OB.evaluate("orig_d0", mk, (1101,), 150)["per_trace"]


# ---------------------------------------------------------------- c5 protocol with 7 features
def _small(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test5", (6000,))


COND = dict(reward_div=10, obs_scale="on", eps_schedule="linear10k", env_family="orig")


@pytest.mark.parametrize("agent,kw,inp", [("lstm_scratch", dict(agent_variant="fixed", target_sync=7), 7),
                                          ("derp", {}, 7), ("derp", dict(derp_history=10), 70)])
def test_c5_delay_variant_smoke(monkeypatch, agent, kw, inp):
    _small(monkeypatch)
    torch.set_num_threads(1)
    r = C5.run_c5(agent, 3, total_ticks=40, checkpoints=[20, 40], eval_splits="val,test5", eval_ticks=30,
                  orig_variant="spike300_d5", **COND, **kw)
    assert r["startup_delay"] == 5 and r["state_size"] == 7 and r["obs_features"][-1] == "pending"
    assert r["obs_scale_divisors"]["pending"] == 20.0 and len(r["obs_scale_divisors"]) == 7
    if agent == "derp":
        assert r["input_size"] == inp
    ep = r["checkpoint_results"]["40"]["val"]["episodes"][0]
    assert "mean_pending" in ep and abs(ep["served"] + ep["vm_cost"] - ep["ret"]) < 1e-6


def test_c5_d0_delay_run_matches_plain_returns_for_a_fixed_policy():
    """eval_episode on orig_d0 vs orig with the same (window-ignoring) policy: identical returns."""
    for lstm, hist in ((True, 1), (False, 1), (False, 3)):
        env0 = C5.make_env("orig:orig", "coupled", 1100)
        e1 = C5.eval_episode(env0, lambda s: 2, lstm, env0.reset(), 60, hist=hist)
        env = C5.make_env("orig:orig_d0", "coupled", 1100)
        e2 = C5.eval_episode(env, lambda s: 2, lstm, env.reset(), 60, hist=hist)
        assert e1["ret"] == e2["ret"] and e2["mean_pending"] == 0.0


def test_window_widths():
    s0 = np.arange(7.0)
    w = C5.window0(s0, True)
    assert w.shape == (RL.SEQ, 7) and np.array_equal(w[-1], s0)
    w2 = C5.push(w, s0 + 1, True)
    assert w2.shape == (RL.SEQ, 7) and np.array_equal(w2[-2], s0)
    h = C5.window0(s0, False, 3)
    assert h.shape == (21,) and np.array_equal(C5.push(h, s0 + 1, False, 3)[-14:-7], s0)


def test_c5_transfer_d0_to_delay(monkeypatch):
    _small(monkeypatch)
    torch.set_num_threads(1)
    r = C5.run_c5("dinos_t", 3, pretrain_ticks=30, finetune_ticks=30, checkpoints=[30], eval_splits="val",
                  eval_ticks=20, agent_variant="fixed", target_sync=7, orig_variant="orig_d5",
                  pretrain_env_family="orig", pretrain_variant="orig_d0", **COND)
    assert r["arm"] == "transfer_orig_orig_d0" and r["state_size"] == 7
    assert r["source_obs_scale_divisors"]["pending"] == 20.0
    assert r["pretrain"]["n_params"] == r["n_params"]


def test_c5_delay_validation(monkeypatch):
    _small(monkeypatch)
    with pytest.raises(SystemExit):        # verbatim LSTM hard-codes 6 inputs
        C5.run_c5("lstm_scratch", 3, total_ticks=10, orig_variant="orig_d5", **COND)
    with pytest.raises(SystemExit):        # 6-feature source -> 7-feature target
        C5.run_c5("dinos_t", 3, pretrain_ticks=10, finetune_ticks=10, agent_variant="fixed", orig_variant="orig_d5",
                  pretrain_env_family="orig", pretrain_variant="orig", **COND)
    with pytest.raises(SystemExit):        # 7-feature source -> 6-feature target
        C5.run_c5("dinos_t", 3, pretrain_ticks=10, finetune_ticks=10, agent_variant="fixed", orig_variant="spike300",
                  pretrain_env_family="orig", pretrain_variant="orig_d0", **COND)
    with pytest.raises(SystemExit):        # coupled source -> delay target
        C5.run_c5("dinos_t", 3, pretrain_ticks=10, finetune_ticks=10, agent_variant="fixed", orig_variant="orig_d5",
                  pretrain_env_family="coupled", pretrain_variant="legacy", **COND)


def test_cli_delay_variant(monkeypatch, tmp_path):
    _small(monkeypatch)
    out = tmp_path / "r.json"
    res = RL.main(["--agent", "derp", "--env-family", "orig", "--orig-variant", "orig_d10", "--total-ticks", "20",
                   "--eval-split", "val", "--reward-div", "10", "--obs-scale", "on", "--out", str(out)])
    assert res["orig_variant"] == "orig_d10" and json.load(open(out))["startup_delay"] == 10


# ---------------------------------------------------------------- splits
def test_test7_split_fresh():
    assert RL.SPLITS["test7"] == tuple(range(8000, 8020))
    others = set().union(*(set(v) for k, v in RL.SPLITS.items() if k != "test7"))
    assert not set(RL.SPLITS["test7"]) & others
