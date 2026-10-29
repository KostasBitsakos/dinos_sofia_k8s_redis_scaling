"""Cycle 4: coupled load-sharing wrapper (tuning/legacy/coupled_env.py), run_legacy --env-variant/--eval-split,
coupled_baselines.py."""
import json

import numpy as np
import pytest

from tuning.legacy import coupled_baselines as CB
from tuning.legacy import coupled_env as CE
from tuning.legacy import env_dense, env_transfer, run_legacy as RL

BASES = (env_dense.Environment, env_transfer.Environment)


@pytest.mark.parametrize("base", BASES)
def test_reference_point_reproduces_legacy(base, monkeypatch):
    """L = 70 and n = 6 -> g = 1: state, reward and every component equal the legacy env's, bit for bit."""
    leg, cop = base(), CE.CoupledEnv(base)
    for e in (leg, cop.base):
        monkeypatch.setattr(e, "load", lambda t: 70.0)
    RL.seed_all(11)
    a = [leg.reset()] + [leg.step(1) for _ in range(300)]
    RL.seed_all(11)
    b = [cop.reset()] + [cop.step(1) for _ in range(300)]
    assert cop.last_g == 1.0
    assert np.array_equal(a[0], b[0])
    for x, y in zip(a[1:], b[1:]):
        assert np.array_equal(x[0], y[0])
        assert tuple(x[1:]) == tuple(y[1:])


@pytest.mark.parametrize("base", BASES)
def test_same_draws_and_formula(base):
    """The wrapper consumes the legacy random draws unchanged and applies g = 6L/(70n) to cpu/mem/latency."""
    leg, cop = base(), CE.CoupledEnv(base)
    rng = np.random.RandomState(5)
    acts = rng.randint(0, 3, 400)
    RL.seed_all(3)
    leg.reset()
    la = [leg.step(a) for a in acts]
    RL.seed_all(3)
    cop.reset()
    ca = [cop.step(a) for a in acts]
    for x, y in zip(la, ca):
        s0, s = x[0], y[0]
        assert s[0] == s0[0] and s[1] == s0[1] and s[5] == s0[5]       # load, capacity, pods unchanged
        g = 6 * s0[0] / (70 * s0[5])
        assert s[2] == pytest.approx(min(100, g * s0[2])) and s[3] == pytest.approx(min(100, g * s0[3]))
        assert s[4] == pytest.approx(g * s0[4])
        assert y[2:7] == pytest.approx((3 * s[0], -5 * s[4], -10 * s[5], -0.5 * s[2], -0.5 * s[3]))
        assert y[1] == pytest.approx(sum(y[2:7]))
        assert y[7] == x[7]


def test_clipping():
    g, c, m, l = CE.coupling(L=200.0, n=1, c0=60.0, m0=50.0, l0=2.0)
    assert g == pytest.approx(1200 / 70)
    assert c == 100.0 and m == 100.0                     # cpu/mem clipped at 100
    assert l == pytest.approx(g * 2.0)                   # latency is not clipped
    g, c, m, l = CE.coupling(L=10.0, n=20, c0=60.0, m0=50.0, l0=2.0)
    assert c == pytest.approx(g * 60) and c < 60          # under-load scales down, no lower clip
    assert CE.coupling(70.0, 6, 1, 1, 1)[0] == 1.0


def test_wrapper_delegates_and_keeps_bounds():
    e = CE.make(env_dense.Environment)()
    e.reset()
    assert e.num_vms == 6 and e.time == 0
    for _ in range(30):
        e.step(0)
    assert e.num_vms == 1
    for _ in range(40):
        s = e.step(2)[0]
    assert e.num_vms == 20 and s[5] == 20
    e.num_vms = 4                                          # setattr reaches the wrapped env
    assert e.base.num_vms == 4


def test_eval_splits_disjoint_and_guarded():
    sets = {k: set(v) for k, v in RL.SPLITS.items()}
    assert sets["fair"] == set(range(1000, 1005)) and sets["val"] == set(range(1100, 1105))
    assert sets["test"] == set(range(2000, 2020))
    for a in sets:
        for b in sets:
            if a < b:
                assert not sets[a] & sets[b]
    for s in (1000, 1102, 2007):
        with pytest.raises(SystemExit):
            RL.check_seeds(s, "fair", ("val", "test"))
    with pytest.raises(SystemExit):
        RL.parse_splits("val,bogus")
    assert RL.parse_splits("val,test") == ("val", "test")


def test_coupled_refuses_orig_protocol():
    with pytest.raises(SystemExit):
        RL.run("always_down", 1000, 7, "both", env_variant="coupled")
    with pytest.raises(SystemExit):
        RL.run("dinos_t", 1000, 7, "fair", env_variant="coupled")   # pretrain would be 0 ticks


def test_legacy_default_output_unchanged_keys():
    r = RL.run("always_down", 1000, 7, "fair")
    assert "env_variant" not in r and set(r["protocols"]) == {"fair"}
    assert "p95_latency" not in r["protocols"]["fair"]["episodes"][0]


def test_coupled_val_test_one_invocation(tmp_path, monkeypatch):
    monkeypatch.setattr(RL, "EP_TICKS", 300)
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    r = RL.run("always_down", 1000, 7, "fair", env_variant="coupled", eval_splits="val,test", tel_dir=str(tmp_path))
    p = r["protocols"]
    assert set(p) == {"val", "test"} and p["val"]["n_episodes"] == 5 and p["test"]["n_episodes"] == 20
    assert r["env"] == "dense" and r["env_variant"] == "coupled"
    e = p["test"]["episodes"][0]
    assert {"p95_latency", "mean_pods", "3L", "neg5lat", "neg10N", "neg05cpu", "neg05mem"} <= set(e)
    assert p["val"]["mean_pods"] < 2                       # always_down reaches 1 pod after 5 ticks
    assert json.loads((tmp_path / "progress.json").read_text())["step"] == 25 * 300
    # val alone == val within val,test (same seeds, independent of the other split)
    r2 = RL.run("always_down", 1000, 9, "fair", env_variant="coupled", eval_splits="val")
    assert r2["protocols"]["val"]["mean_ret"] == p["val"]["mean_ret"]


class _StubAgent:
    def __init__(self):
        self.memory, self.epsilon = [], 1.0

    def act(self, state, verbose=0):
        return int(np.random.randint(3))

    def remember(self, *t):
        self.memory.append(t)

    def replay(self, b):
        pass


def test_training_resets_terminated_episodes():
    e = CE.make(env_dense.Environment)()
    ag = _StubAgent()
    RL.train_chunks(ag, e, 6000, False, reset_on_done=True)
    assert e.time == 1000                                  # reset at t = 5000, then 1000 more ticks
    assert sum(1 for t in ag.memory if t[4]) == 1          # exactly one terminal transition
    e2, ag2 = env_dense.Environment(), _StubAgent()
    RL.train_chunks(ag2, e2, 6000, False)                  # legacy: never reset
    assert e2.time == 6000


def test_baselines_small(tmp_path, monkeypatch):
    monkeypatch.setattr(RL, "EP_TICKS", 200)
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test", (2000, 2001, 2002))
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"grid": [30, 70], "fixed": [1, 6, 12]}))
    out = tmp_path / "b.json"
    r = CB.main(["--config", str(cfg), "--out", str(out), "--seed", "0"])
    assert set(r["test"]) == {"always_down", "hpa_rule_50_50", "hpa_rule_tuned", "fixed_ramped", "fixed_prepro"}
    assert len(r["val_grid_hpa"]) == 4 and r["test"]["fixed_prepro"]["n_episodes"] == 3
    best = max(r["val_grid_hpa"], key=lambda k: r["val_grid_hpa"][k]["mean_ret"])
    assert r["selected"]["hpa_rule_tuned"] == dict(zip(("T_c", "T_m"), map(int, best.split("_"))))
    # holding N = 6 from the start of a reset is the same trajectory ramped or preprovisioned
    assert r["val_fixed_ramped"]["6"]["mean_ret"] == r["val_fixed_prepro"]["6"]["mean_ret"]
    assert r["val_fixed_prepro"]["12"]["mean_pods"] == 12.0
    assert json.loads(out.read_text())["selected"] == r["selected"]
