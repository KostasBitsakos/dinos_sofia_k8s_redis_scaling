"""Cycle 19: opt-in Double-DQN add-ons of the fixed-agent family (agent_lstm_fixed.FixedLSTMAgent.set_ddqn):
n-step returns, dueling head, proportional prioritized replay; wired through c5_protocol.run_c5(ddqn=...) and the
run_legacy.py CLI (--n-step / --dueling / --per). Defaults are covered by the cycle-18 fingerprint tests (test_v2.py)."""
import contextlib
import json
import os

import numpy as np
import pytest
import torch

from tuning.legacy import agent_dense_fixed as DFX
from tuning.legacy import agent_lstm_fixed as FX
from tuning.legacy import agent_lstm_v2 as V2
from tuning.legacy import c5_protocol as C5
from tuning.legacy import run_legacy as RL

F, SEQ, A = 7, 10, 3
COMMON = dict(eval_splits="val", eval_ticks=30, reward_div=10.0, obs_scale="on", eps_schedule="linear10k",
              env_family="orig", orig_variant="spike300_d5")


@pytest.fixture
def small_splits(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100,))


def _s(i):
    return np.full((SEQ, F), float(i), dtype=np.float32)


# ---------------------------------------------------------------- n-step
def test_nstep_returns_and_flush():
    a = FX.make_scratch(F, A, SEQ, ddqn=dict(n_step=3))
    g = a.gamma
    rs = [1.0, 2.0, 3.0, 4.0, 5.0]
    for t, r in enumerate(rs):
        a.remember(_s(t), t % A, r, _s(t + 1), t == len(rs) - 1)
    m = list(a.memory)
    assert len(m) == 5 and not a._nbuf
    # two full 3-step returns, bootstrapped with gamma^3 from s_{t+3}
    for t in (0, 1):
        s, act, R, ns, done, disc = m[t]
        assert float(s[0, 0]) == t and act == t % A and not done
        assert R == pytest.approx(rs[t] + g * rs[t + 1] + g * g * rs[t + 2]) and disc == pytest.approx(g ** 3)
        assert float(ns[0, 0]) == t + 3
    # episode end: transitions 2, 3, 4 flushed as terminal with shorter returns, all pointing at the final state
    for t, exp in ((2, 3 + g * 4 + g * g * 5), (3, 4 + g * 5), (4, 5.0)):
        s, _, R, ns, done, _ = m[t]
        assert float(s[0, 0]) == t and done and R == pytest.approx(exp) and float(ns[0, 0]) == 5


def test_nstep_target_uses_gamma_n_and_double_dqn():
    torch.manual_seed(0)
    a = FX.make_scratch(F, A, SEQ, ddqn=dict(n_step=3))
    with torch.no_grad():
        for p in a.target_model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    ns = _s(1)
    x = a._x(ns)
    with torch.no_grad():
        a_star = int(torch.argmax(a.online_model(x)[0]))
        exp = 2.0 + a.gamma ** 3 * float(a.target_model(x)[0][a_star])
    assert float(a.ddqn_target(2.0, ns, False, a.gamma ** 3)) == pytest.approx(exp, rel=1e-6)
    assert a.ddqn_target(2.0, ns, True, a.gamma ** 3) == 2.0


def test_nstep_one_is_plain_tuple():
    a = FX.make_scratch(F, A, SEQ)
    a.remember(_s(0), 1, 0.5, _s(1), False)
    assert len(a.memory[0]) == 5 and a.n_step == 1 and not a.dueling and not a.per


# ---------------------------------------------------------------- dueling
@pytest.mark.parametrize("arch", ["a1h", "a4", "mlp_a1h", "mlp_a34"])
def test_dueling_shape_and_identity(arch):
    torch.manual_seed(1)
    a = V2.make_scratch(arch, F, A, SEQ, ddqn=dict(dueling=True))
    plain = V2.make_scratch(arch, F, A, SEQ)
    net = a.online_model
    parent, name = (net, "q") if not V2.is_mlp(arch) else (net.net, "fc3")
    head = getattr(parent, name)
    assert isinstance(head, FX.DuelingHead)
    w = head.value.in_features
    assert V2.n_params(net) == V2.n_params(plain.online_model) + w + 1
    feats = {}
    head.register_forward_hook(lambda m, i, o: feats.update(h=i[0]))
    x = torch.randn(4, SEQ, F)
    with torch.no_grad():
        q = net(x)
        v, adv = head.value(feats["h"]), head.advantage(feats["h"])
    assert q.shape == (4, A)
    assert torch.allclose(q.mean(dim=1, keepdim=True), v, atol=1e-6)            # mean_a Q = V
    assert torch.allclose(q - v, adv - adv.mean(dim=1, keepdim=True), atol=1e-6)
    # online and target identical; optimizer covers the new head
    assert all(torch.equal(a.target_model.state_dict()[k], t) for k, t in net.state_dict().items())
    opt_ids = {id(p) for g in a.optimizer.param_groups for p in g["params"]}
    assert {id(p) for p in net.parameters()} == opt_ids


def test_dueling_fixed_lstm_and_dense():
    for ag in (FX.make_scratch(F, A, SEQ, ddqn=dict(dueling=True)), DFX.make_scratch(F, A, SEQ, ddqn=dict(dueling=True))):
        assert ag.online_model(torch.zeros(2, SEQ, F)).shape == (2, A)
    w = FX.make_scratch(F, A, SEQ, ddqn=dict(dueling=True)).online_model.state_dict()
    t = FX.make_transfer(F, A, SEQ, w, ddqn=dict(dueling=True))
    assert all(torch.equal(t.online_model.state_dict()[k], v) for k, v in w.items())


# ---------------------------------------------------------------- PER
def test_per_favours_high_priority_and_weights():
    m = FX.PrioritizedReplay(100, seed=5)
    for i in range(100):
        m.append(i)
    m.update(np.arange(100), np.r_[np.full(99, 0.01), 10.0])     # item 99 dominant
    idx, items, w = m.sample(16, 0.4)
    counts = np.zeros(100)
    for _ in range(200):
        idx, _, w = m.sample(16, 0.4)
        counts += np.bincount(idx, minlength=100)
        assert np.all(w <= 1.0 + 1e-12) and np.all(w > 0)
    p = m.probabilities()
    assert counts[99] / counts.sum() == pytest.approx(p[99], abs=0.05) and counts[99] > 20 * counts[:99].mean()
    # new transitions get the max priority
    m.append("new")
    assert m.prio[0] == pytest.approx(m.max_p) and m.max_p == pytest.approx(10.0 + 1e-6)
    assert list(m)[-1] == "new" and len(m) == 100


def test_per_deterministic_under_seed_and_beta_anneal():
    def run():
        torch.manual_seed(2)
        a = V2.make_scratch("a4", F, A, SEQ, ddqn=dict(per=True, per_anneal_replays=10, per_seed=[7, 19, 2]))
        rng = np.random.RandomState(0)
        for t in range(40):
            a.remember(rng.rand(SEQ, F).astype(np.float32), t % A, float(rng.randn()), rng.rand(SEQ, F).astype(np.float32),
                       t % 13 == 12)
        for _ in range(12):
            a.replay(16)
        return a
    a, b = run(), run()
    assert np.array_equal(a.memory.prio, b.memory.prio) and a.loss_history == b.loss_history
    assert a.per_beta() == 1.0 and len(set(np.round(a.memory.prio[:40], 9))) > 2
    c = V2.make_scratch("a4", F, A, SEQ, ddqn=dict(per=True, per_anneal_replays=10))
    assert c.per_beta() == pytest.approx(0.4)


def test_set_ddqn_guards():
    a = FX.make_scratch(F, A, SEQ)
    with pytest.raises(ValueError):
        a.set_ddqn(n_step=0)
    a.remember(_s(0), 0, 1.0, _s(1), False)
    with pytest.raises(RuntimeError):
        a.set_ddqn(per=True)
    with pytest.raises(SystemExit):
        C5.validate_c19(dict(per=True), "derp", "verbatim", None)
    with pytest.raises(SystemExit):
        C5.validate_c19(dict(n_step=3), "lstm_scratch", "fixed", 5)
    assert C5.validate_c19(dict(n_step=1, dueling=False), "lstm_scratch", "a4", None) is None


# ---------------------------------------------------------------- smoke runs
FLAGS = {"nstep": dict(n_step=3), "dueling": dict(dueling=True), "per": dict(per=True),
         "all": dict(n_step=3, dueling=True, per=True)}


def _run(tmp_path, tag, agent, **kw):
    out = str(tmp_path / tag / "r.json")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        C5.run_c5(agent, 53, out=out, **dict(COMMON, **kw))
    return json.load(open(out))


@pytest.mark.parametrize("arch", ["a1h", "a4", "mlp_a34"])
@pytest.mark.parametrize("flag", sorted(FLAGS))
def test_smoke_scratch(tmp_path, small_splits, arch, flag):
    r = _run(tmp_path, f"{arch}_{flag}", "lstm_scratch", agent_variant=arch, total_ticks=300, checkpoints=[300],
             ddqn=FLAGS[flag])
    o = r["ddqn_options"]
    assert o["n_step"] == FLAGS[flag].get("n_step", 1) and o["dueling"] == FLAGS[flag].get("dueling", False)
    assert o["per"] == FLAGS[flag].get("per", False) and r["hyperparameters"]["ddqn_addons"]["n_step"] == o["n_step"]
    if o["per"]:
        assert r["target_train"]["per_beta_final"] == 1.0          # exact endpoint, also with the n-step delay
    base = V2.n_params(V2.make_scratch(arch, F, A, SEQ).online_model)
    assert r["n_params"] > base if o["dueling"] else r["n_params"] == base
    assert ("dueling[" in r["architecture"]) == o["dueling"]
    assert np.isfinite(r["checkpoint_results"]["300"]["val"]["mean_ret"]) and r["replays"] > 0


def test_smoke_transfer_nstep_dueling(tmp_path, small_splits):
    r = _run(tmp_path, "t", "dinos_t", agent_variant="a4", pretrain_env_family="orig", pretrain_variant="orig_d0",
             pretrain_ticks=150, finetune_ticks=150, checkpoints=[150], ddqn=dict(n_step=3, dueling=True))
    assert r["ddqn_options"]["n_step"] == 3 and r["ddqn_options"]["dueling"]
    assert r["ddqn_options"]["applies_to"] == "pretrain and target phases"
    assert r["pretrain"]["n_params"] == r["n_params"] == 32131 + 32 + 1
    assert np.isfinite(r["checkpoint_results"]["150"]["val"]["mean_ret"])


def test_cli_passes_options(tmp_path, small_splits, monkeypatch):
    seen = {}
    monkeypatch.setattr(C5, "run_c5", lambda *a, **kw: seen.update(kw) or dict(
        arm="x", cpu_s=0.0, train_cpu_s_per_1k_ticks=0.0, checkpoint_results={}, eval_splits=[]))
    base = ["--agent", "lstm_scratch", "--agent-variant", "a4", "--env-family", "orig", "--orig-variant", "spike300_d5",
            "--total-ticks", "100", "--checkpoints", "100", "--seed", "53", "--eval-split", "val"]
    RL.main(base)
    assert "ddqn" not in seen
    RL.main(base + ["--n-step", "3", "--dueling", "on", "--per", "on"])
    assert seen["ddqn"] == dict(n_step=3, dueling=True, per=True)


# ---------------------------------------------------------------- review fixes (Codex NO-GO on 4d3fd0e)
def test_per_batch_order_shuffled_and_aligned():
    m = FX.PrioritizedReplay(64, seed=11)
    for i in range(64):
        m.append(("t", i))
    orders = []
    for _ in range(50):                                   # equal priorities: stratified = one per segment
        idx, items, w = m.sample(16, 0.5)
        assert [it[1] for it in items] == list(idx) and np.allclose(w, 1.0)
        assert sorted(idx // 4) == list(range(16))
        orders.append(tuple(idx))
    assert sum(list(o) != sorted(o) for o in orders) >= 45 and len(set(orders)) == 50
    m.update(np.arange(64), np.arange(64) + 1.0)        # unequal: weight i belongs to transition i
    idx, items, w = m.sample(16, 0.7)
    pa = m.prio[:64] ** m.alpha
    exp = (64 * pa[idx] / pa.sum()) ** -0.7 / (64 * pa.min() / pa.sum()) ** -0.7
    assert np.allclose(w, exp) and [it[1] for it in items] == list(idx)


def test_nstep_phase_end_flush_bootstrapped():
    a = FX.make_scratch(F, A, SEQ, ddqn=dict(n_step=3))
    g = a.gamma
    for t, r in enumerate([1.0, 2.0]):
        a.remember(_s(t), 0, r, _s(t + 1), False)
    assert len(a.memory) == 0
    a.flush_nstep(_s(2))
    m = list(a.memory)
    assert [(R, done, disc) for _, _, R, _, done, disc in m] == [(pytest.approx(1 + g * 2), False, pytest.approx(g ** 2)),
                                                                (2.0, False, pytest.approx(g))]
    assert all(float(ns[0, 0]) == 2 for _, _, _, ns, _, _ in m) and not a._nbuf


def test_train_episodic_flush_precedes_last_replay_and_checkpoint(monkeypatch):
    """Phase end: pending n-step transitions are flushed (bootstrapped) BEFORE the last tick's replay and checkpoint,
    are in memory and eligible for that replay; the number of replays is unchanged."""
    T = 40
    a = FX.make_scratch(F, A, SEQ, ddqn=dict(n_step=3))
    events, real_flush, real_replay = [], a.flush_nstep, a.replay

    def flush(s):
        events.append(("flush", len(a._nbuf)))
        real_flush(s)

    def replay(b):
        m = list(a.memory)
        events.append(("replay", len(m), sum(1 for tr in m if not tr[4] and tr[5] != a.gamma ** 3)))
        real_replay(b)
    monkeypatch.setattr(a, "flush_nstep", flush)
    monkeypatch.setattr(a, "replay", replay)

    class Env:
        def reset(self):
            return np.zeros(F, dtype=np.float32)
    rng = np.random.RandomState(0)
    monkeypatch.setattr(C5, "env_step", lambda env, action: (rng.rand(F).astype(np.float32), 1.0, None, False))
    C5.train_episodic(a, lambda s: 0, Env(), T, True, checkpoints=[T], on_ck=lambda t: events.append(("ck", t)))
    assert events[-3:] == [("flush", 2), ("replay", T, 2), ("ck", T)]       # 2 shorter bootstrapped returns sampled-able
    assert sum(1 for e in events if e[0] == "replay") == T - 3 - RL.BATCH + 1   # same budget: memory > 16 from tick 19
    assert sum(1 for e in events if e[0] == "flush") == 1 and not a._nbuf


@pytest.mark.parametrize("arch", ["mlp_a34", "mlp_a1h"])
def test_cli_dueling_reaches_matched_mlp(tmp_path, small_splits, monkeypatch, arch):
    real = C5.run_c5
    monkeypatch.setattr(C5, "run_c5", lambda *a, **kw: real(*a, **dict(kw, eval_ticks=30)))
    out = str(tmp_path / "r.json")
    RL.main(["--agent", "lstm_scratch", "--agent-variant", arch, "--env-family", "orig", "--orig-variant",
             "spike300_d5", "--total-ticks", "30", "--checkpoints", "30", "--seed", "53", "--eval-split", "val",
             "--dueling", "on", "--out", out])
    r = json.load(open(out))
    assert r["ddqn_options"]["dueling"] and r["n_params"] == {"mlp_a34": 32416, "mlp_a1h": 104836}[arch]
    assert "dueling[V: Dense(" in r["architecture"]


def test_per_beta_endpoint_replay_index():
    a = FX.make_scratch(F, A, SEQ, ddqn=dict(per=True, per_anneal_replays=5, n_step=2))
    rng = np.random.RandomState(0)
    for t in range(30):
        a.remember(rng.rand(SEQ, F).astype(np.float32), 0, 1.0, rng.rand(SEQ, F).astype(np.float32), False)
    betas = []
    for _ in range(5):
        a.replay(16)
        betas.append(a.per_beta_last)
    assert betas[0] == pytest.approx(0.4) and betas[-1] == 1.0


@pytest.mark.parametrize("flag", ["nstep", "dueling", "per", "all"])
def test_smoke_fixed_lstm(tmp_path, small_splits, flag):
    r = _run(tmp_path, f"fx_{flag}", "lstm_scratch", agent_variant="fixed", total_ticks=200, checkpoints=[200],
             ddqn=FLAGS[flag])
    assert r["ddqn_options"]["n_step"] == FLAGS[flag].get("n_step", 1)
    base = V2.n_params(FX.make_scratch(F, A, SEQ).online_model)
    assert r["n_params"] == base + (129 if FLAGS[flag].get("dueling") else 0)
    if FLAGS[flag].get("per"):
        assert r["target_train"]["per_beta_final"] == 1.0
    assert np.isfinite(r["checkpoint_results"]["200"]["val"]["mean_ret"])


def test_smoke_fixed_transfer_all(tmp_path, small_splits):
    r = _run(tmp_path, "fxt", "dinos_t", agent_variant="fixed", pretrain_env_family="orig", pretrain_variant="orig_d0",
             pretrain_ticks=120, finetune_ticks=120, checkpoints=[120], ddqn=FLAGS["all"])
    assert r["ddqn_options"]["dueling"] and r["ddqn_options"]["per"] and r["ddqn_options"]["n_step"] == 3
    assert r["pretrain"]["per_beta_final"] == 1.0 and r["target_train"]["per_beta_final"] == 1.0
    assert r["pretrain"]["n_params"] == r["n_params"]
    assert np.isfinite(r["checkpoint_results"]["120"]["val"]["mean_ret"])
