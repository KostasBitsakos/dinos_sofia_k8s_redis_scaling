"""Cycle 23 (council C-20261023-01): opt-in --lr / --grad-clip (fixed-agent family, both phases) and
--pending-maturity on (5 features: pending pods ready in 1..5 ticks). Defaults are covered by the fingerprint suites
(test_v2 / test_c19 / test_c20 / test_c7); here: plumbing, clipping math, maturity counts, scaling, re-matched MLPs,
smokes with every option on, and identity of the option code paths at neutral settings."""
import contextlib
import json
import os
import random

import numpy as np
import pytest
import torch

from tuning.legacy import agent_dense_fixed as DFX
from tuning.legacy import agent_lstm_fixed as FX
from tuning.legacy import agent_lstm_v2 as V2
from tuning.legacy import c5_protocol as C5
from tuning.legacy import orig_variants as OV
from tuning.legacy import run_legacy as RL

SEQ, A = 10, 3
COMMON = dict(eval_splits="val", eval_ticks=30, reward_div=10.0, obs_scale="on", eps_schedule="linear10k",
              env_family="orig", orig_variant="spike300_d5")
ON = dict(lr=5e-5, grad_clip=10.0, pending_maturity=True)
TRANSFER = dict(agent="dinos_t", pretrain_env_family="orig", pretrain_variant="orig_d0")


@pytest.fixture
def small_splits(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100,))


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _run(tmp_path, tag, agent, **kw):
    out = str(tmp_path / tag / "r.json")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        C5.run_c5(agent, 7, out=out, **dict(COMMON, **kw))
    return json.load(open(out))


# ---------------------------------------------------------------- lr / grad clip
def _agents(optim):
    return [FX.make_scratch(7, A, SEQ, optim=optim), DFX.make_scratch(7, A, SEQ, optim=optim),
            V2.make_scratch("a1h", 7, A, SEQ, optim=optim), V2.make_scratch("mlp_a1h", 7, A, SEQ, optim=optim),
            FX.make_transfer(7, A, SEQ, FX.make_scratch(7, A, SEQ).online_model.state_dict(), optim=optim),
            V2.make_scratch("a1h", 7, A, SEQ, ddqn=dict(n_step=3, dueling=True), optim=optim)]


def test_lr_override_every_fixed_agent():
    for a in _agents(None):     # current defaults: 1e-4 scratch-type agents, 5e-5 make_transfer
        assert a.lr_override is None and a.grad_clip is None
    assert [a.optimizer.param_groups[0]["lr"] for a in _agents(None)] == [1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 1e-4]
    for a in _agents(dict(lr=3e-5, grad_clip=2.0)):
        assert a.learning_rate == 3e-5 and a.optimizer.param_groups[0]["lr"] == 3e-5 and a.grad_clip == 2.0
        ids = {id(p) for p in a.online_model.parameters()}       # the new Adam covers the (dueling) online net
        assert {id(p) for g in a.optimizer.param_groups for p in g["params"]} == ids
    with pytest.raises(ValueError):
        FX.make_scratch(7, A, SEQ, optim=dict(lr=-1.0))
    with pytest.raises(ValueError):
        FX.make_scratch(7, A, SEQ, optim=dict(grad_clip=0.0))


def _filled(agent, n=40, seed=0):
    rs = np.random.RandomState(seed)
    for _ in range(n):
        agent.remember(rs.randn(SEQ, 7), int(rs.randint(3)), float(rs.randn() * 50), rs.randn(SEQ, 7), False)
    return agent


def test_clip_math():
    torch.manual_seed(0)
    a = _filled(FX.make_scratch(7, A, SEQ, optim=dict(grad_clip=0.05)))
    post = []
    step = a.optimizer.step

    def spy(*args, **kw):
        g = torch.cat([p.grad.reshape(-1) for p in a.online_model.parameters() if p.grad is not None])
        post.append(float(g.norm()))
        return step(*args, **kw)
    a.optimizer.step = spy
    random.seed(1)
    a.replay(16)
    pre = a.grad_norms
    assert len(pre) == len(post) == 16
    for g0, g1 in zip(pre, post):
        want = g0 * min(1.0, 0.05 / (g0 + 1e-6))
        assert g1 == pytest.approx(want, rel=1e-4)
    assert a.n_clipped == sum(g > 0.05 for g in pre) > 0
    st = a.grad_norm_stats()
    assert st["updates"] == 16 and st["clipped"] == a.n_clipped and st["clip_frac"] == a.n_clipped / 16
    assert st["max"] == max(pre) and st["min"] == min(pre) and st["mean"] == pytest.approx(np.mean(pre))
    assert FX.make_scratch(7, A, SEQ).grad_norm_stats() is None


def test_neutral_options_identical_weights():
    """lr = the default and a clip that never binds leave training bit-identical (g * 1.0)."""
    out = []
    for optim in (None, dict(lr=1e-4, grad_clip=1e12)):
        torch.manual_seed(3)
        a = _filled(FX.make_scratch(7, A, SEQ, optim=optim), seed=3)
        random.seed(5)
        np.random.seed(5)
        for _ in range(3):
            a.replay(16)
        out.append([p.detach().clone() for p in a.online_model.parameters()])
    assert all(torch.equal(x, y) for x, y in zip(*out))


# ---------------------------------------------------------------- pending maturity
def test_maturity_scripted_d5():
    env = OV.OrigEnv("orig_d5", 0, maturity=True)
    ref = OV.OrigEnv("orig_d5", 0)
    s, r0 = env.reset(), ref.reset()
    assert len(s) == 12 and np.array_equal(s[:7], r0) and not s[7:].any()
    #          action, pods, pending, ready_in_1..5
    script = [(2, 6, 1, [0, 0, 0, 0, 1]),
              (2, 6, 2, [0, 0, 0, 1, 1]),
              (1, 6, 2, [0, 0, 1, 1, 0]),
              (0, 6, 1, [0, 1, 0, 0, 0]),       # cancels the most recent pod (counter 3), not the older one
              (2, 6, 2, [1, 0, 0, 0, 1]),
              (1, 7, 1, [0, 0, 0, 1, 0]),       # the first pod is active (requested at tick 1, D = 5 -> tick 6)
              (1, 7, 1, [0, 0, 1, 0, 0]),
              (1, 7, 1, [0, 1, 0, 0, 0]),
              (1, 7, 1, [1, 0, 0, 0, 0]),
              (1, 8, 0, [0, 0, 0, 0, 0]),       # requested at tick 5 -> active at tick 10
              (0, 7, 0, [0, 0, 0, 0, 0])]       # nothing pending: the verbatim scale-down
    for a, pods, pend, mat in script:
        s, r, done, info = env.step(a)
        s0, r_, _, info0 = ref.step(a)
        assert np.array_equal(s[:7], s0) and r == r_ and info == info0      # features 0-6 / reward unchanged
        assert s[5] == pods and s[6] == pend and list(s[7:]) == mat and s[7:].sum() <= pend


def _ledger_run(variant, actions, seed=4):
    """Rebuild the maturity features from the controller's own history: its actions, the known delay D and the
    observed active pod count (state[5]); no env internals."""
    D = OV.startup_delay(variant)
    env = OV.OrigEnv(variant, seed, maturity=True)
    s = env.reset()
    led = []
    for a in actions:
        led = [c - 1 for c in led]
        promoted = sum(c <= 0 for c in led)
        led = [c for c in led if c > 0]
        active = s[5] + promoted
        down = a == 0 and not led and active > 1          # nothing to cancel: the verbatim scale-down
        if a == 2 and active + len(led) < 20:
            led.append(D)
        elif a == 0 and led:
            led.pop()
        s, _, _, _ = env.step(a)
        want = [sum(c == k for c in led) for k in range(1, 6)]
        assert list(s[7:]) == want and s[6] == len(led) and s[5] == active - down
        assert set(s[7:]) <= {0.0, 1.0}


def test_maturity_from_history_random_and_cap():
    rs = np.random.RandomState(0)
    _ledger_run("spike300_d5", list(rs.randint(0, 3, 3000)))
    _ledger_run("orig_d5", [2] * 40 + [0, 1, 2] * 30)          # hits the 20-pod cap (refused requests)
    _ledger_run("orig_d3", list(rs.choice([0, 1, 2, 2], 500)))   # D < 5: buckets 4, 5 stay empty
    _ledger_run("orig_d8", list(rs.choice([0, 2, 2], 500)))       # D > 5: counters 6..8 only in the pending total


def test_maturity_zero_at_d0_and_parity():
    env, ref = OV.OrigEnv("orig_d0", 2, maturity=True), OV.OrigEnv("orig_d0", 2)
    assert np.array_equal(env.reset()[:7], ref.reset())
    for a in np.random.RandomState(1).randint(0, 3, 400):
        s, r, d, i = env.step(int(a))
        s0, r0, _, _ = ref.step(int(a))
        assert np.array_equal(s[:7], s0) and r == r0 and not s[7:].any()
    with pytest.raises(SystemExit):
        OV.OrigEnv("spike300", 0, maturity=True)            # no startup delay -> no pending pods


def test_maturity_scaling_and_features():
    d = C5.obs_divisors("orig", "spike300_d5", True)
    assert d.tolist() == [100.0, 100.0, 60.0, 70.0, 2.4, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert list(C5.obs_features("orig", "spike300_d5", True))[7:] == [f"ready_in_{k}" for k in range(1, 6)]
    assert OV.state_size("spike300_d5", True) == 12 and OV.state_size("spike300_d5") == 7
    s = np.arange(12, dtype=float) + 1
    assert np.array_equal(C5.obs_fn("on", "orig", "spike300_d5", True)(s), s / d)
    assert C5.obs_divisors("orig", "spike300_d5").tolist() == OV.OBS_SCALE_DELAY.tolist()   # off: unchanged
    env = C5.make_env("orig:spike300_d5" + C5.PM_SUFFIX, "coupled", 3)
    assert env.maturity and len(env.reset()) == 12 and not C5.make_env("orig:spike300_d5", "coupled", 3).maturity


def test_rematched_param_counts():
    a1h7, a1h12 = V2.n_params(V2.build_net("a1h", 7, SEQ, A)), V2.n_params(V2.build_net("a1h", 12, SEQ, A))
    m288_12, m271_12 = (V2.n_params(V2.build_net(m, 12, SEQ, A)) for m in ("mlp_a1h", "mlp_a1h_pm"))
    assert (a1h7, a1h12, m288_12, m271_12) == (104451, 107651, 118947, 107319)
    assert abs(m271_12 / a1h12 - 1) < 0.005 and V2.MATCH_PM == {"mlp_a1h_pm": ("a1h",)}
    assert all(abs(w * w + 125 * w + 3 - a1h12) > abs(m271_12 - a1h12) for w in (270, 272))
    lstm12, d540, d522 = (V2.n_params(FX.ResetLSTM(12, 128, 3, A)), V2.n_params(DFX.FlatMLP(12, SEQ, 540, A)),
                          V2.n_params(DFX.FlatMLP(12, SEQ, DFX.DENSE_HIDDEN_PM, A)))
    assert (lstm12, d540, d522) == (337283, 359103, 337737) and abs(d522 / lstm12 - 1) < 0.002


def test_bad_options(small_splits):
    bad = [dict(agent="derp", total_ticks=20, lr=1e-4),
           dict(agent="lstm_scratch", agent_variant="fixed", total_ticks=20, lr=0.0),
           dict(agent="lstm_scratch", agent_variant="fixed", total_ticks=20, grad_clip=-1.0),
           dict(agent="lstm_scratch", agent_variant="fixed", total_ticks=20, pending_maturity=True,
                orig_variant="spike300"),
           dict(agent="lstm_scratch", agent_variant="mlp_a1h", total_ticks=20, pending_maturity=True),
           dict(agent="lstm_scratch", agent_variant="mlp_a1h_pm", total_ticks=20)]
    for kw in bad:
        kw = dict(COMMON, **kw)
        with pytest.raises(SystemExit):
            C5.run_c5(kw.pop("agent"), 7, **kw)
    with pytest.raises(SystemExit):     # coupled family: no maturity
        C5.run_c5("lstm_scratch", 7, total_ticks=20, agent_variant="fixed", eval_splits="val", pending_maturity=True)
    with pytest.raises(SystemExit):
        C5.run_c5("lstm_scratch", 7, total_ticks=20, agent_variant="verbatim", eval_splits="val", lr=1e-4)


def test_cli(monkeypatch):
    seen = {}

    def fake(*a, **kw):
        seen.clear()
        seen.update(kw)
        return {"arm": "x", "cpu_s": 0.0, "train_cpu_s_per_1k_ticks": 0.0, "checkpoint_results": {}, "eval_splits": []}
    monkeypatch.setattr(C5, "run_c5", fake)
    base = ["--seed", "7", "--env-family", "orig", "--orig-variant", "spike300_d5", "--agent", "lstm_scratch",
            "--agent-variant", "a1h", "--total-ticks", "100"]
    RL.main(base)
    assert not {"lr", "grad_clip", "pending_maturity"} & set(seen)
    RL.main(base + ["--lr", "5e-5", "--grad-clip", "10", "--pending-maturity", "on"])
    assert seen["lr"] == 5e-5 and seen["grad_clip"] == 10.0 and seen["pending_maturity"] is True
    with pytest.raises(SystemExit):
        RL.main(["--agent", "lstm_scratch", "--train-ticks", "1000", "--lr", "1e-4"])


# ---------------------------------------------------------------- smokes (every option on)
SMOKES = {"lstm_scratch": dict(agent="lstm_scratch", agent_variant="fixed", total_ticks=300, checkpoints=[300]),
          "dinos_a1h": dict(agent="lstm_scratch", agent_variant="a1h", total_ticks=300, checkpoints=[300]),
          "mlp_hist10": dict(agent="lstm_scratch", agent_variant="dense_fixed", total_ticks=300, checkpoints=[300]),
          "mlp_a1h_pm": dict(agent="lstm_scratch", agent_variant="mlp_a1h_pm", total_ticks=300, checkpoints=[300]),
          "transfer_fixed": dict(TRANSFER, agent_variant="fixed", pretrain_ticks=150, finetune_ticks=150,
                                 checkpoints=[150]),
          "transfer_a1h": dict(TRANSFER, agent_variant="a1h", pretrain_ticks=120, finetune_ticks=120,
                               checkpoints=[120])}
N_PARAMS = {"lstm_scratch": 337283, "dinos_a1h": 107651, "mlp_hist10": 337737, "mlp_a1h_pm": 107319,
            "transfer_fixed": 337283, "transfer_a1h": 107651}


@pytest.mark.parametrize("name", sorted(SMOKES))
def test_smoke_all_options_on(tmp_path, small_splits, monkeypatch, name):
    monkeypatch.setattr(C5, "LOG_EVERY", 100)
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path / "run"))
    kw = dict(SMOKES[name], **ON)
    r = _run(tmp_path, name, kw.pop("agent"), tel_dir=str(tmp_path / "run"), **kw)
    assert r["state_size"] == 12 and r["n_params"] == N_PARAMS[name] and r["learning_rate"] == 5e-5
    assert r["obs_features"][7:] == list(OV.FEATURES_MATURITY) and len(r["obs_scale_divisors"]) == 12
    assert r["optim_options"]["lr"] == 5e-5 and r["optim_options"]["grad_clip"] == 10.0
    assert r["pending_maturity"]["k"] == 5 and r["pending_maturity"]["state_size"] == 12
    assert r["target_env"] == "orig:spike300_d5"
    ck = r["checkpoint_results"][str(kw["checkpoints"][-1])]
    assert np.isfinite(ck["val"]["mean_ret"])
    phases = ("pretrain", "target") if name.startswith("transfer") else ("target",)
    assert set(r["grad_norm"]) == set(phases)
    for ph in phases:
        g = r["grad_norm"][ph]
        assert g["updates"] > 0 and g["max_norm"] == 10.0 and 0.0 <= g["clip_frac"] <= 1.0 and g["max"] >= g["p99"] >= g["p95"] >= g["p90"] >= g["p50"]
    if name.startswith("transfer"):
        assert r["pretrain"]["n_params"] == r["n_params"] and r["pretrain"]["input_size"] == 12
        assert r["pretrain"]["learning_rate"] == 5e-5
        assert r["source_env"] == "orig:orig_d0" and len(r["source_obs_scale_divisors"]) == 12
        assert r["optim_options"]["applies_to"] == "pretrain and target phases"
    if name == "mlp_hist10":
        assert r["dense_hidden"] == DFX.DENSE_HIDDEN_PM
    if name in ("dinos_a1h", "transfer_a1h", "mlp_a1h_pm"):
        assert r["hyperparameters"]["learning_rate"] == 5e-5 and r["hyperparameters"]["grad_clip_max_norm"] == 10.0
    if name == "mlp_a1h_pm":
        assert r["matched_to"] == ["a1h"]
    rows = [json.loads(x) for x in open(tmp_path / "run" / "metrics.jsonl")]
    assert any("target_clip_frac" in x and "target_grad_norm_mean" in x for x in rows)


def test_off_records_nothing_new(tmp_path, small_splits):
    r = _run(tmp_path, "off", "lstm_scratch", agent_variant="a1h", total_ticks=40, checkpoints=[40])
    assert not {"optim_options", "pending_maturity", "grad_norm"} & set(r)
    assert r["state_size"] == 7 and "grad_clip_max_norm" not in r["hyperparameters"]
    assert "grad_norm" not in r["target_train"]


def test_neutral_options_identical_run(tmp_path, small_splits):
    """Whole run: lr = default and a never-binding clip give the same returns/checkpoints as the default run."""
    kw = dict(agent_variant="a1h", total_ticks=60, checkpoints=[30, 60], ddqn=dict(n_step=3))
    a = _run(tmp_path, "a", "lstm_scratch", **kw)
    b = _run(tmp_path, "b", "lstm_scratch", lr=1e-4, grad_clip=1e12, **kw)
    assert b["grad_norm"]["target"]["clipped"] == 0
    for k in ("checkpoint_results", "target_train"):
        strip = lambda d: {x: y for x, y in d.items() if "cpu_s" not in x and x != "grad_norm"}  # noqa: E731
        ca, cb = a[k], b[k]
        if k == "checkpoint_results":
            ca = {t: strip(v) for t, v in ca.items()}
            cb = {t: strip(v) for t, v in cb.items()}
        else:
            ca, cb = strip(ca), strip(cb)
        assert ca == cb


def test_coupled_dinos_t_fixed_lr_both_phases(tmp_path, small_splits):
    """The coupled-family dinos_t fixed arm (FX.make_scratch source, FX.make_transfer fine-tune, default 5e-5)."""
    out = str(tmp_path / "c" / "r.json")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        r = C5.run_c5("dinos_t", 7, source_variant="coupled", pretrain_ticks=40, finetune_ticks=30, checkpoints=[30],
                      agent_variant="fixed", eval_splits="val", eval_ticks=20, out=out, lr=2e-5, grad_clip=5.0)
    assert r["learning_rate"] == 2e-5 and r["pretrain"]["learning_rate"] == 2e-5
    assert set(r["grad_norm"]) == {"pretrain", "target"} and r["grad_norm"]["target"]["updates"] > 0
    assert "pending_maturity" not in r
