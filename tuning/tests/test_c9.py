"""Cycle 9: original env P/environment.py (env_orig.py), its adapter orig_variants.py, the c5 orig family, test5 and
orig_baselines.py (design .lab/tuning/cycles/c9_design.md)."""
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from tuning.legacy import c5_protocol as C5
from tuning.legacy import env_orig, orig_baselines as OB, orig_variants as OV, port_sources
from tuning.legacy import run_legacy as RL

ORIG_SRC = os.path.join(port_sources.LEGACY_ROOT, "environment.py")
need_legacy = pytest.mark.skipif(not os.path.isfile(ORIG_SRC), reason="original legacy code not present")


def _actions(n=5000, seed=7):
    return np.random.RandomState(seed).randint(0, 3, n)


def _verbatim(seed, acts):
    np.random.seed(seed)
    e = env_orig.Environment()
    states, rewards, dones = [e.reset()], [], []
    for a in acts:
        s, r, d = e.step(a)
        states.append(s)
        rewards.append(r)
        dones.append(d)
    return np.array(states), rewards, dones


def _adapter(variant, seed, acts):
    e = OV.OrigEnv(variant, seed)
    states, rewards, dones, infos = [e.reset()], [], [], []
    for a in acts:
        s, r, d, i = e.step(a)
        states.append(s)
        rewards.append(r)
        dones.append(d)
        infos.append(i)
    return np.array(states), rewards, dones, infos


# ---------------------------------------------------------------- port + parity
@need_legacy
def test_orig_port_is_byte_identical():
    with open(ORIG_SRC, "rb") as f:
        raw = f.read()
    assert hashlib.md5(raw).hexdigest().startswith("2096a045")
    with open(env_orig.__file__, "rb") as f:
        port = f.read()
    assert port.split(b"\n", 4)[4] == raw            # 4 header lines, then the original byte for byte
    assert port.decode() == port_sources.render("env_orig.py")


@pytest.mark.parametrize("seed", [0, 1100, 6000])
def test_adapter_orig_equals_verbatim(seed):
    acts = _actions()
    vs, vr, vd = _verbatim(seed, acts)
    np.random.seed(12345)
    g0 = np.random.get_state()
    s, r, d, info = _adapter("orig", seed, acts)
    g1 = np.random.get_state()
    assert np.array_equal(vs, s) and vr == r and vd == d         # exact, bit for bit
    assert d[-1] and not any(d[:-1])
    assert np.array_equal(g0[1], g1[1]) and g0[2:] == g1[2:]         # the global numpy stream is untouched
    assert all(x == i["served"] + i["vm_cost"] for x, i in zip(r, info))
    assert set(s[:, 5]) <= set(range(1, 21))


def test_adapters_do_not_share_streams():
    acts = _actions(300)
    a = OV.OrigEnv("orig", 5)
    b = OV.OrigEnv("orig", 5)
    a.reset()
    b.reset()
    sa, sb = [], []
    for x in acts:              # interleaved calls: each adapter keeps its own stream
        sa.append(a.step(x)[0])
        sb.append(b.step(x)[0])
    assert np.array_equal(np.array(sa), np.array(sb))
    assert np.array_equal(np.array(sa), _adapter("orig", 5, acts)[0][1:])


# ---------------------------------------------------------------- variants
@pytest.mark.parametrize("variant", OV.VARIANTS)
def test_variant_deterministic_per_seed(variant):
    acts = _actions(5000)
    a, ra, _, _ = _adapter(variant, 1101, acts)
    b, rb, _, _ = _adapter(variant, 1101, acts)
    c, _, _, _ = _adapter(variant, 1102, acts)
    assert np.array_equal(a, b) and ra == rb
    assert not np.array_equal(a[:, 0], c[:, 0])


def test_noise_variant_scales_only_load_noise():
    acts = np.ones(5000, dtype=int)
    o = _adapter("orig", 3, acts)[0]
    n = _adapter("noise15", 3, acts)[0]
    assert np.array_equal(o[:, 1:], n[:, 1:])                 # same draws for every other metric
    t = np.arange(5001)
    base = 50 * np.sin(2 * np.pi * t / (1000 / (2 * np.pi))) + 50
    out = (t >= 1500) & (t < 1600)
    ok = ~out & (o[:, 0] > 0) & (n[:, 0] > 0)                  # away from the spike and the clip at 0
    assert np.allclose((n[ok, 0] - base[ok]), (o[ok, 0] - base[ok]) * 15 / 0.2, rtol=1e-6, atol=1e-6)
    assert 13 < np.std(n[ok, 0] - base[ok]) < 17


def test_spike300_changes_only_the_load_spike():
    acts = np.ones(5000, dtype=int)
    o = _adapter("orig", 4, acts)[0]
    s = _adapter("spike300", 4, acts)[0]
    t = np.arange(5001)
    inside = (t >= 1500) & (t < 1600)
    assert np.array_equal(o[~inside], s[~inside]) and np.array_equal(o[:, 1:], s[:, 1:])
    assert s[inside, 0].max() > o[inside, 0].max() + 100


def test_multispike_positions():
    e = OV.OrigEnv("multispike", 1100)
    e.reset()
    first = list(e.spikes)
    assert len(first) == 4
    for k, p in enumerate(first):
        assert OV.SEG * k + 50 <= p <= OV.SEG * (k + 1) - 150
    o = _adapter("orig", 1100, np.ones(5000, dtype=int))[0]
    m = _adapter("multispike", 1100, np.ones(5000, dtype=int))[0]
    t = np.arange(5001)
    inside = np.zeros(5001, bool)
    for p in first:
        inside |= (t >= p) & (t < p + 100)
    quiet = ~inside & ~((t >= 1500) & (t < 1600))              # outside every spike of either env
    assert np.array_equal(o[quiet], m[quiet])                  # the noise stream is that of orig
    only_m = inside & ~((t >= 1500) & (t < 1600))
    assert only_m.sum() >= 300 and np.allclose(m[only_m, 2], o[only_m, 2] + 3)   # cpu +3 inside the new spikes
    if not any(1500 <= p < 1600 for p in first):
        assert not np.array_equal(o[1500:1600], m[1500:1600])   # the original spike is not there
    e.reset()
    assert e.spikes != first                                   # redrawn per episode (training resets)
    f = OV.OrigEnv("multispike", 1101)
    f.reset()
    assert f.spikes != first                                   # differs per episode seed


def test_clone_peek_equals_real_step():
    for v in ("orig", "noise15_multispike"):
        e = OV.OrigEnv(v, 9)
        e.reset()
        for a in _actions(400, 3):
            peek = e.clone().step(a)
            real = e.step(a)
            assert np.array_equal(peek[0], real[0]) and peek[1] == real[1]


def test_bad_variant():
    with pytest.raises(SystemExit):
        OV.OrigEnv("noise", 0)


# ---------------------------------------------------------------- baselines
def test_causal_target():
    # load 50, 6 VMs, capacity 30 -> read 0.5 -> ceil(50 / 5) = 10
    assert OB.causal_target([50, 30, 0, 0, 0, 6]) == 10
    assert OB.causal_target([50, 30, 0, 0, 0, 6], True) == 10
    assert OB.causal_target([50, 12, 0, 0, 0, 6], True) == 1      # read 0.2 < 0.3
    assert OB.causal_target([500, 30, 0, 0, 0, 6]) == 20
    assert OB.causal_target([0, 30, 0, 0, 0, 6]) == 1
    assert OB.causal_target([5, 0, 0, 0, 0, 6]) == 20


def test_baselines_short_trace(monkeypatch, tmp_path):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test5", (6000, 6001, 6002))
    out = tmp_path / "b.json"
    res = OB.main(["--orig-variant", "multispike", "--ticks", "120", "--out", str(out)])
    assert json.load(open(out))["selected"] == res["selected"]
    assert set(res["val_fixed"]) == {str(n) for n in range(1, 21)}
    best = res["selected"]["fixed_best"]["N"]
    assert res["val"]["fixed_best"]["mean_ret"] == max(v["mean_ret"] for v in res["val_fixed"].values())
    assert set(res["val_hpa_demand"]) == {"0.6", "0.7", "0.8", "0.9", "1.0"}
    assert res["val"]["hpa_demand"]["mean_ret"] == max(v["mean_ret"] for v in res["val_hpa_demand"].values())
    for sp, seeds in (("val", (1100, 1101)), ("test5", (6000, 6001, 6002))):
        assert set(res[sp]) == {"fixed_best", "hpa_demand", "hpa_rule", "always_down", "causal", "causal_be",
                                "causal_trend", "oracle_1step"}
        for v in res[sp].values():
            assert list(v["per_trace"]) == [str(s) for s in seeds] and 0 <= v["frac_const_pods"] <= 1
        assert res[sp]["always_down"]["frac_const_pods"] > 0.9
    # fixed_best on test5 equals the same policy evaluated directly (paired traces)
    d = OB.evaluate("multispike", OB.hold(best), (6000,), 120)
    assert d["per_trace"]["6000"] == res["test5"]["fixed_best"]["per_trace"]["6000"]


def test_hpa_demand_and_trend_rules():
    act = OB.hpa_demand(0.8)(None)
    assert act([90, 100, 0, 0, 0, 6]) == 2 and act([70, 100, 0, 0, 0, 6]) == 1 and act([50, 100, 0, 0, 0, 6]) == 0
    tr = OB.causal_trend(5)(None)
    s = lambda load: [load, 30, 0, 0, 0, 6]      # read_hat 0.5 -> target ceil(load_hat / 5)  # noqa: E731
    for load in (10, 12, 14, 16, 18):             # < k past loads: no trend; load 18 -> target 4 -> down
        assert tr(s(load)) == 0
    # 6th call: load_hat = 30 + 5 * (30 - 10) = 130 -> target 26 -> clipped 20 -> up (plain causal: 30 -> 6 -> hold)
    assert tr(s(30)) == 2 and OB.causal(False)(None)(s(30)) == 1


def test_oracle_takes_best_immediate_reward():
    e = OV.OrigEnv("orig", 1102)
    e.reset()
    act = OB.oracle_1step(e)
    s = None
    for _ in range(300):
        rs = [e.clone().step(a)[1] for a in (0, 1, 2)]
        a = act(s)
        r = e.step(a)[1]
        assert r == max(rs)


# ---------------------------------------------------------------- c5 protocol, orig family
def _small(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test5", (6000,))


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if "cpu_s" not in k and "wall_s" not in k}
    if isinstance(o, list):
        return [_strip(x) for x in o]
    return o


def test_test5_split():
    assert RL.SPLITS["test5"] == tuple(range(6000, 6020))
    assert not set(RL.SPLITS["test5"]) & set().union(*(set(v) for k, v in RL.SPLITS.items() if k != "test5"))


@pytest.mark.parametrize("agent,kw", [("lstm_scratch", dict(agent_variant="fixed", target_sync=7)), ("derp", {})])
def test_c5_orig_family_smoke(monkeypatch, agent, kw):
    _small(monkeypatch)
    torch.set_num_threads(1)
    run = lambda: C5.run_c5(agent, 3, total_ticks=60, checkpoints=[30, 60], eval_splits="val,test5", eval_ticks=40,  # noqa: E731
                            reward_div=10, obs_scale="on", eps_schedule="linear10k", env_family="orig",
                            orig_variant="noise15_multispike", **kw)
    r1, r2 = run(), run()
    assert _strip(r1) == _strip(r2)                                # deterministic per seed
    assert r1["env_family"] == "orig" and r1["target_env"] == "orig:noise15_multispike"
    assert r1["obs_scale_divisors"] == dict(zip(C5.OBS_FEATURES, [100.0, 100.0, 60.0, 70.0, 2.4, 20.0]))
    ck = r1["checkpoint_results"]["60"]
    assert set(ck) >= {"val", "test5"} and ck["test5"]["n_episodes"] == 1
    ep = ck["val"]["episodes"][0]
    assert set(ep) >= {"served", "vm_cost", "frac_const_pods", "pods_changes"} and ep["ticks"] == 40
    assert abs(ep["served"] + ep["vm_cost"] - ep["ret"]) < 1e-6
    assert set(ck["val"]["components_mean"]) == set(OV.COMP)


def test_c5_orig_eval_uses_the_baseline_traces(monkeypatch):
    """A greedy agent evaluated by c5 and the same policy in orig_baselines see the same trace per seed."""
    _small(monkeypatch)
    env = C5.make_env("orig:spike300", "coupled", 1100)
    e1 = C5.eval_episode(env, lambda s: 1, False, env.reset(), 50)
    e2 = OB.evaluate("spike300", lambda env: (lambda s: 1), (1100,), 50)["episodes"][0]
    assert e1["ret"] == e2["ret"] and e1["frac_const_pods"] == 1.0


def test_c5_orig_validation():
    with pytest.raises(SystemExit):
        C5.run_c5("dinos_t", 3, source_variant="none", finetune_ticks=10, env_family="orig", orig_variant="orig")
    with pytest.raises(SystemExit):
        C5.run_c5("derp", 3, total_ticks=10, env_family="coupled", orig_variant="orig")
    with pytest.raises(SystemExit):
        C5.run_c5("derp", 3, total_ticks=10, env_family="orig", orig_variant="noise")
    with pytest.raises(SystemExit):
        C5.run_c5("lstm_scratch", 3, total_ticks=10, env_family="orig", orig_variant="orig", diag=5)


def test_cli_orig_family(monkeypatch, tmp_path):
    _small(monkeypatch)
    out = tmp_path / "r.json"
    res = RL.main(["--agent", "derp", "--env-family", "orig", "--orig-variant", "spike300", "--total-ticks", "30",
                   "--eval-split", "val", "--reward-div", "10", "--obs-scale", "on", "--out", str(out)])
    assert res["orig_variant"] == "spike300" and json.load(open(out))["env_family"] == "orig"
    with pytest.raises(SystemExit):
        RL.main(["--agent", "derp", "--env-family", "orig", "--env-variant", "coupled", "--total-ticks", "30"])
    with pytest.raises(SystemExit):
        RL.main(["--agent", "derp", "--orig-variant", "orig", "--train-ticks", "1000"])


def test_causal_be_zero_capacity_targets_min():
    from tuning.legacy import orig_baselines as OB
    s = [50.0, 0.0, 40.0, 50.0, 1.0, 6]
    assert OB.causal_target(s, break_even=True) == OB.NMIN
    assert OB.causal_target(s) == OB.NMAX
