"""Cycle 18: Dinos V.2 architecture search (agent_lstm_v2.py), matched MLP controls, researcher checkpoint protocol
(--save-weights, --source-checkpoints), test10 split, and bit identity of the existing arms (tiny runs vs fingerprints
taken with one torch thread on the dinos-tuning code before cycle 18, HEAD 3fe5aeb)."""
import contextlib
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from tuning.legacy import agent_lstm_fixed as FX
from tuning.legacy import agent_lstm_v2 as V2
from tuning.legacy import c5_protocol as C5
from tuning.legacy import run_legacy as RL

HERE = os.path.dirname(os.path.abspath(__file__))
F, SEQ, A = 7, 10, 3
PARAMS = {"a1": 25475, "a1h": 104451, "a2": 35971, "a3": 32579, "a4": 32131, "mlp_a1": 25657, "mlp_a1h": 104547,
          "mlp_a234": 34129, "mlp_a2": 36039, "mlp_a34": 32269,
          "a5": 32579, "a6": 32195, "a8": 32259, "a9": 51459, "mlp_a8": 32193, "mlp_a9": 51267}
COMMON = dict(eval_splits="val,test9", eval_ticks=30, reward_div=10.0, obs_scale="on", eps_schedule="linear10k",
              env_family="orig", orig_variant="spike300_d5")


@pytest.fixture
def small_splits(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100, 1101))
    monkeypatch.setitem(RL.SPLITS, "test9", (10000,))


def _run(tmp_path, tag, agent, **kw):
    out = str(tmp_path / tag / "r.json")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        C5.run_c5(agent, 7, out=out, **dict(COMMON, **kw))
    return json.load(open(out)), out


def _strip(o, drop=()):
    if isinstance(o, dict):
        return {k: _strip(v, drop) for k, v in o.items()
                if not ("cpu_s" in k or "wall_s" in k or "per_1k" in k) and k != "out" and k not in drop}
    if isinstance(o, list):
        return [_strip(v, drop) for v in o]
    return o


# ---------------------------------------------------------------- networks
@pytest.mark.parametrize("arch", sorted(PARAMS))
def test_param_counts_and_shapes(arch):
    a = V2.make_scratch(arch, F, A, SEQ)
    assert V2.n_params(a.online_model) == PARAMS[arch]
    assert a.online_model(torch.zeros(5, SEQ, F)).shape == (5, A)
    assert V2.architecture(arch, F).endswith(f"{A})") or V2.architecture(arch, F).endswith(f"-> {A}")


def test_matched_mlps():
    for mlp, archs in V2.MATCH.items():
        for arch in archs:
            rel = PARAMS[mlp] / PARAMS[arch] - 1
            assert abs(rel) < (0.07 if mlp == "mlp_a234" else 0.01), (mlp, arch, rel)
    for W in (127, 288, 151, 156, 146, 192):     # closed form W^2 + (10F + 5) W + 3
        assert V2.n_params(V2.DFX.FlatMLP(F, SEQ, W, A)) == W * W + (10 * F + 5) * W + 3
    assert V2.n_params(V2.FlatMLPTrend(F, SEQ, 145, A, 5)) == 145 * 145 + (10 * F + 7) * 145 + 3
    # branch 2: every new arch has an exact (< 1%) control; a8's control sees the same trend features
    assert {x: m for m, xs in V2.MATCH.items() for x in xs if x in ("a5", "a6", "a8", "a9")} == \
        {"a5": "mlp_a34", "a6": "mlp_a34", "a8": "mlp_a8", "a9": "mlp_a9"}
    assert V2.ARCHS["mlp_a8"] == "m145t5" and V2.ARCHS["a8"].split("-")[2] == "trendcat5"


def test_spec_parser():
    for bad in ("", "x5", "d64-l64", "cat-cat", "l0", "d64-cat"):
        with pytest.raises(ValueError):
            V2.parse(bad)
    assert V2.parse("c32-l64-cat-d64-d32")[0] == ("c", 32)
    assert V2.parse("c32k3r2-l64-cat")[0] == ("cc", 32, 3, 2) and V2.parse("c8k2-l4")[0] == ("cc", 8, 2, 1)
    assert V2.parse("e32-l64-attcat-d64")[2] == ("att", 0) and V2.parse("l8-trendcat5-d4")[1] == ("trend", 5)
    assert V2.parse("m145t5") == [("mt", 145, 5)] and V2.parse("m146") == [("m", 146)]
    for bad in ("d4-attcat", "cat-trendcat5", "l8-cat-c8k3r2", "c8k0", "m0", "m10t0", "trendcat0", "c8r2"):
        with pytest.raises(ValueError):
            V2.parse(bad)
    net = V2.SpecQ("e16-l8-l8-d4", F, A)          # generic: new specs build without registry changes
    assert net(torch.zeros(2, SEQ, F)).shape == (2, A)


@pytest.mark.parametrize("arch", ["a1", "a2", "a3", "a4", "a5", "a6", "a8", "a9", "mlp_a8"])
def test_zero_hidden_per_window(arch):
    """No state across calls; each batch row is independent (zero (h, c) per window)."""
    torch.manual_seed(0)
    net = V2.build_net(arch, F, SEQ, A)
    x = torch.randn(4, SEQ, F)
    with torch.no_grad():
        y1, y2 = net(x), net(x)
        rows = torch.cat([net(x[i:i + 1]) for i in range(4)])
    assert torch.equal(y1, y2) and torch.allclose(y1, rows, atol=1e-6)
    assert net.init_hidden() is None


def test_a5_causal_dilated_conv():
    """Conv output at step t depends only on frames t, t-2, t-4 (left padding 4, dilation 2), never on later frames."""
    torch.manual_seed(0)
    net = V2.build_net("a5", F, SEQ, A)
    conv = net.layers[0]
    assert (conv.kernel_size, conv.dilation, conv.padding) == ((3,), (2,), (0,))
    seen = {}
    net.layers[1].register_forward_hook(lambda m, i, o: seen.update(z=i[0].clone()))
    x = torch.randn(1, SEQ, F)
    with torch.no_grad():
        net(x)
        z0 = seen["z"]
        assert z0.shape == (1, SEQ, 32)
        for j in range(SEQ):
            x2 = x.clone()
            x2[0, j] += 1.0
            net(x2)
            changed = {t for t in range(SEQ) if not torch.equal(seen["z"][0, t], z0[0, t])}
            assert changed <= {j, j + 2, j + 4} and all(t >= j for t in changed), (j, changed)


def _padded(n_real, seed=0):
    """A window as c5_protocol builds it n_real - 1 ticks after a reset: zero rows, then n_real real frames."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(1, SEQ, F)
    x[0, SEQ - n_real:] = torch.rand(n_real, F, generator=g) + 0.1
    return x


def test_a6_attention_masking():
    torch.manual_seed(0)
    net = V2.build_net("a6", F, SEQ, A)
    att = net.layers[2]
    assert att.bias is None and att.weight.shape == (1, 64)
    with torch.no_grad():
        for n_real in (1, 3, SEQ):
            net(_padded(n_real))
            a = net.last_attention[0]
            assert torch.allclose(a.sum(), torch.tensor(1.0)) and torch.all(a[:SEQ - n_real] == 0)
            assert torch.all(a[SEQ - n_real:] > 0)
        y = net(torch.zeros(2, SEQ, F))                    # all-zero (terminal) window: newest frame kept, finite
        assert torch.isfinite(y).all() and torch.equal(net.last_attention[:, -1], torch.ones(2))
        # pooled readout = sum_t a_t h_t (recomputed by hand) || newest obs
        seen = {}
        net.layers[3].register_forward_hook(lambda m, i, o: seen.update(z=i[0]))
        x = _padded(6, 1)
        net(x)
        h = net.layers[1](net.relu(net.layers[0](x)), (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64)))[0]
        s = att(h).squeeze(-1)
        s[0, :SEQ - 6] = float("-inf")
        pooled = (torch.softmax(s, 1).unsqueeze(-1) * h).sum(1)
        assert torch.allclose(seen["z"][0, :64], pooled[0], atol=1e-6) and torch.equal(seen["z"][0, 64:], x[0, -1])


def test_trend_features_causal():
    x = _padded(SEQ, 2)
    L = x[0, :, 0]
    t = V2.trend_features(x, 5)[0]
    assert torch.allclose(t, torch.stack([L[-1] - L[-2], (L[-1] - L[-6]) / 5]))
    x2 = x.clone()                                          # only frames t, t-1, t-5 and only the load column matter
    for j in set(range(SEQ)) - {SEQ - 1, SEQ - 2, SEQ - 6}:
        x2[0, j] += 3.0
    x2[0, :, 1:] += 3.0
    assert torch.equal(V2.trend_features(x2, 5), V2.trend_features(x, 5))
    assert torch.equal(V2.trend_features(_padded(1), 5)[0], torch.zeros(2))      # right after a reset: both 0
    t3 = V2.trend_features(_padded(3, 4), 5)[0]                                # t-1 available, t-5 is padding
    L3 = _padded(3, 4)[0, :, 0]
    assert torch.allclose(t3, torch.stack([L3[-1] - L3[-2], torch.tensor(0.0)]))
    assert torch.allclose(V2.trend_features(_padded(6, 4), 5)[0, 1], (_padded(6, 4)[0, -1, 0] - _padded(6, 4)[0, -6, 0]) / 5)
    with pytest.raises(ValueError):
        V2.trend_features(torch.zeros(1, 5, F), 5)


def test_a8_and_control_see_the_same_trends():
    torch.manual_seed(0)
    net, mlp = V2.build_net("a8", F, SEQ, A), V2.build_net("mlp_a8", F, SEQ, A)
    seen = {}
    net.layers[3].register_forward_hook(lambda m, i, o: seen.update(z=i[0]))
    mlp.net.fc1.register_forward_hook(lambda m, i, o: seen.update(m=i[0]))
    x = torch.cat([_padded(SEQ, 5), _padded(4, 6)])
    with torch.no_grad():
        net(x), mlp(x)
    tr = V2.trend_features(x, 5)
    assert seen["z"].shape == (2, 64 + F + 2) and torch.equal(seen["z"][:, 64:64 + F], x[:, -1])
    assert torch.equal(seen["z"][:, -2:], tr)
    assert seen["m"].shape == (2, SEQ * F + 2) and torch.equal(seen["m"][:, :-2], x.reshape(2, -1))
    assert torch.equal(seen["m"][:, -2:], tr)


def test_new_mlp_control_mechanics():
    a, b = V2.make_scratch("mlp_a8", F, A, SEQ), V2.make_scratch("mlp_a34", F, A, SEQ)
    assert isinstance(a.online_model, V2.FlatMLPTrend) and a.dense_hidden == 145
    for k in ("gamma", "epsilon", "epsilon_min", "epsilon_decay", "learning_rate", "target_sync"):
        assert getattr(a, k) == getattr(b, k), k
    assert a.memory.maxlen == 5000 and type(a).replay is type(b).replay is FX.FixedLSTMAgent.replay
    torch.manual_seed(3)
    V2.make_scratch("mlp_a8", F, A, SEQ)
    s1 = torch.get_rng_state()
    torch.manual_seed(3)
    V2.build_net("mlp_a8", F, SEQ, A), V2.build_net("mlp_a8", F, SEQ, A)
    assert torch.equal(s1, torch.get_rng_state())
    assert V2.input_size("mlp_a8", F) == 72 and V2.input_size("mlp_a34", F) == 70 and V2.input_size("a8", F) == F


def test_concat_uses_newest_obs():
    torch.manual_seed(0)
    net = V2.build_net("a1", F, SEQ, A)
    x = torch.randn(1, SEQ, F)
    seen = {}
    net.layers[2].register_forward_hook(lambda m, i, o: seen.update(z=i[0]))
    with torch.no_grad():
        net(x)
    assert seen["z"].shape == (1, 64 + F) and torch.equal(seen["z"][0, 64:], x[0, -1])


def test_agent_mechanics_inherited():
    a = V2.make_scratch("a1", F, A, SEQ)
    b = FX.make_scratch(F, A, SEQ)
    for k in ("gamma", "epsilon", "epsilon_min", "epsilon_decay", "learning_rate", "target_sync"):
        assert getattr(a, k) == getattr(b, k), k
    assert a.memory.maxlen == 5000 and type(a).replay is FX.FixedLSTMAgent.replay
    # torch stream consumed only by the new nets (the discarded myLSTMs are built in fork_rng)
    torch.manual_seed(3)
    V2.make_scratch("a1", F, A, SEQ)
    s1 = torch.get_rng_state()
    torch.manual_seed(3)
    V2.build_net("a1", F, SEQ, A), V2.build_net("a1", F, SEQ, A)
    assert torch.equal(s1, torch.get_rng_state())


def test_transfer_weight_copy():
    src = V2.make_scratch("a3", F, A, SEQ)
    w = {k: v.clone() for k, v in src.online_model.state_dict().items()}
    dst = V2.make_scratch("a3", F, A, SEQ)
    dst.load_weights(w)
    for m in (dst.online_model, dst.target_model):
        assert all(torch.equal(m.state_dict()[k], w[k]) for k in w)


# ---------------------------------------------------------------- runs
SCRATCH = {"a1": "dinos_a1", "a1h": "dinos_a1h", "a2": "dinos_a2", "a3": "dinos_a3", "a4": "dinos_a4",
           "mlp_a1": "mlp_a1", "mlp_a1h": "mlp_a1h", "mlp_a234": "mlp_a234",
           "a5": "dinos_a5", "a6": "dinos_a6", "a8": "dinos_a8", "a9": "dinos_a9", "mlp_a8": "mlp_a8", "mlp_a9": "mlp_a9"}


@pytest.mark.parametrize("arch", sorted(SCRATCH))
def test_scratch_arm_records(tmp_path, small_splits, arch):
    r, _ = _run(tmp_path, arch, "lstm_scratch", agent_variant=arch, total_ticks=40, checkpoints=[20, 40])
    assert r["arm"] == SCRATCH[arch] and r["n_params"] == PARAMS[arch] and r["net_spec"] == V2.ARCHS[arch]
    assert r["hyperparameters"]["target_sync_replays"] == 250 and r["learning_rate"] == 1e-4
    assert r["architecture"] == V2.architecture(arch, F) and r["state_size"] == F
    assert set(r["checkpoint_results"]) == {"20", "40"}
    assert r["input_size"] == V2.input_size(arch, F) and np.isfinite(r["checkpoint_results"]["40"]["val"]["mean_ret"])
    if V2.is_mlp(arch):
        assert r["dense_hidden"] == V2.parse(V2.ARCHS[arch])[0][1]


def test_transfer_source_checkpoints_and_saved_weights(tmp_path, small_splits):
    kw = dict(agent_variant="a1", pretrain_env_family="orig", pretrain_variant="orig_d0", pretrain_ticks=40,
              finetune_ticks=40, checkpoints=[20, 40])
    r, out = _run(tmp_path, "t", "dinos_t", source_checkpoints=[20, 40], save_weights=True, **kw)
    assert r["arm"] == "transfer_a1" and "transfer" in r["variant_label"]
    sel = r["source_selection"]
    vb = {int(k): v for k, v in sel["val_by_checkpoint"].items()}
    assert sel["selected"] == max(sorted(vb), key=lambda c: (vb[c], -c))
    root = os.path.splitext(out)[0]
    for t in (20, 40):
        assert os.path.exists(f"{root}.src.ck{t}.pt") and os.path.exists(f"{root}.ck{t}.pt")
    sw = r["saved_weights"]
    tv = {int(c): x["val"]["mean_ret"] for c, x in r["checkpoint_results"].items()}
    assert sw["selected_checkpoint"] == max(sorted(tv), key=lambda c: (tv[c], -c))
    a, b = torch.load(f"{root}.selected.pt"), torch.load(f"{root}.ck{sw['selected_checkpoint']}.pt")
    assert all(torch.equal(a[k], b[k]) for k in a)
    # source checkpoint == final pretrain tick only: identical run to the default (source eval changes nothing)
    r1, _ = _run(tmp_path, "t1", "dinos_t", source_checkpoints=[40], **kw)
    r0, _ = _run(tmp_path, "t0", "dinos_t", **kw)
    drop = ("source_selection", "eval_cpu_s")
    assert json.dumps(_strip(r1, drop), sort_keys=True) == json.dumps(_strip(r0, drop), sort_keys=True)


def test_transfer_mlp_arm(tmp_path, small_splits):
    r, _ = _run(tmp_path, "m", "dinos_t", agent_variant="mlp_a1", pretrain_env_family="orig", pretrain_variant="orig_d0",
                pretrain_ticks=30, finetune_ticks=30, checkpoints=[30])
    assert r["arm"] == "transfer_mlp_a1" and r["n_params"] == PARAMS["mlp_a1"] and r["pretrain"]["n_params"] == 25657


def test_bad_options(tmp_path, small_splits):
    with pytest.raises(SystemExit):     # source checkpoints need orig transfer
        _run(tmp_path, "x", "lstm_scratch", agent_variant="a1", total_ticks=20, source_checkpoints=[20])
    with pytest.raises(SystemExit):     # largest source checkpoint must be the pretrain length
        _run(tmp_path, "y", "dinos_t", agent_variant="a1", pretrain_env_family="orig", pretrain_variant="orig_d0",
             pretrain_ticks=40, finetune_ticks=20, source_checkpoints=[20])
    with pytest.raises(SystemExit):     # V.2 archs only on the orig family
        C5.run_c5("lstm_scratch", 7, total_ticks=20, agent_variant="a1", eval_splits="val")
    with pytest.raises(SystemExit):
        C5.run_c5("derp", 7, total_ticks=20, agent_variant="a1", eval_splits="val", env_family="orig",
                  orig_variant="spike300_d5")


def test_cli_arms(monkeypatch):
    seen = {}

    def fake(*a, **kw):
        seen.clear()
        seen.update(kw, agent=a[0])
        return {"arm": "x", "cpu_s": 0.0, "train_cpu_s_per_1k_ticks": 0.0, "checkpoint_results": {}, "eval_splits": []}
    monkeypatch.setattr(C5, "run_c5", fake)
    base = ["--seed", "7", "--env-family", "orig", "--orig-variant", "spike300_d5"]
    for arch in V2.ARCHS:
        RL.main(base + ["--agent", "lstm_scratch", "--agent-variant", arch, "--total-ticks", "100"])
        assert seen["agent_variant"] == arch and "save_weights" not in seen and "source_checkpoints" not in seen
        RL.main(base + ["--agent", "dinos_t", "--agent-variant", arch, "--pretrain-env-family", "orig",
                        "--pretrain-variant", "orig_d0", "--pretrain-ticks", "20000", "--finetune-ticks", "100",
                        "--source-checkpoints", "1000,2000,5000,10000,20000", "--save-weights", "on", "--out", "/tmp/x.json"])
        assert seen["source_checkpoints"] == [1000, 2000, 5000, 10000, 20000] and seen["save_weights"] is True
    with pytest.raises(SystemExit):
        RL.main(["--agent", "lstm_scratch", "--agent-variant", "a99", "--total-ticks", "100"] + base)
    assert C5.v2_arm("lstm_scratch", "a2") == "dinos_a2" and C5.v2_arm("lstm_scratch", "mlp_a234") == "mlp_a234"
    assert C5.v2_arm("dinos_t", "a1h") == "transfer_a1h" and C5.v2_arm("dinos_t", "mlp_a1") == "transfer_mlp_a1"


# ---------------------------------------------------------------- splits / identity
def test_test10_split():
    assert RL.SPLITS["test10"] == tuple(range(11000, 11020)) == RL.TEST10_SEEDS
    for k, v in RL.SPLITS.items():
        if k != "test10":
            assert not set(v) & set(RL.TEST10_SEEDS), k
    assert not set(RL.TEST10_SEEDS) & set(range(4000, 4020))
    assert RL.parse_splits("val,test10") == ("val", "test10")
    with pytest.raises(SystemExit):
        RL.check_seeds(11005, "fair", ("val", "test10"))


EXISTING = {"lstm_scratch": dict(agent="lstm_scratch", agent_variant="fixed", total_ticks=60, checkpoints=[30, 60]),
            "mlp_hist10": dict(agent="lstm_scratch", agent_variant="dense_fixed", total_ticks=60, checkpoints=[30, 60]),
            "derp": dict(agent="derp", total_ticks=60, checkpoints=[30, 60]),
            "derp_hist10": dict(agent="derp", derp_history=10, total_ticks=60, checkpoints=[30, 60]),
            "transfer": dict(agent="dinos_t", agent_variant="fixed", pretrain_env_family="orig",
                             pretrain_variant="orig_d0", pretrain_ticks=40, finetune_ticks=60, checkpoints=[20, 60])}


@pytest.mark.parametrize("name", sorted(EXISTING))
def test_existing_arms_bit_identical(tmp_path, small_splits, name):
    """Fingerprints (sha256 of the result JSON minus timing/out) recorded on the pre-cycle-18 code."""
    ref = json.load(open(os.path.join(HERE, "data", "c18_existing_arm_fingerprints.json")))
    kw = dict(EXISTING[name])
    n = torch.get_num_threads()
    torch.set_num_threads(1)            # the fingerprints and the real runs (--threads 1) use one thread
    try:
        r, _ = _run(tmp_path, name, kw.pop("agent"), **kw)
    finally:
        torch.set_num_threads(n)
    assert hashlib.sha256(json.dumps(_strip(r), sort_keys=True).encode()).hexdigest() == ref[name]
    assert not {"net_spec", "source_selection", "saved_weights", "architecture"} & set(r)
    assert np.isfinite(r["checkpoint_results"][str(kw["checkpoints"][-1])]["val"]["mean_ret"])


# ---------------------------------------------------------------- lab generator / promotion / summary (lab side)
LAB = os.path.expanduser("~/qrl-lab/.lab/tuning")
MAKE_C18 = os.path.join(LAB, "make_c18.py")


def _load(name):
    import importlib.util
    import sys
    if LAB not in sys.path:
        sys.path.insert(0, LAB)
    spec = importlib.util.spec_from_file_location(name, os.path.join(LAB, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not os.path.exists(MAKE_C18), reason="lab generator not present")
def test_make_c18_modes(tmp_path, monkeypatch):
    m = _load("make_c18")
    s10 = _load("sum_c10")
    every = set().union(*(set(v) for v in RL.SPLITS.values())) | set(range(4000, 4020))
    assert m.EVAL_SEEDS == every
    ents, info = m.main(["--arch-search", "--tag", "c18zz", "--out", str(tmp_path / "c18zz_entries.json")])
    assert info["n_entries"] == 3 * (5 + 4 + 5) + 1 and info["estimated_cpu_h"] > 0     # rung 2: exact MLPs
    arms = set()
    for e in ents:
        c = e["cmd"]
        if "--agent" not in c:
            assert c[c.index("--report-splits") + 1] == "test9"
            continue
        a = s10.cmd_args(c)                                  # every flag takes one value (sum_c10 parsing)
        assert a["--checkpoints"] == "1000,2000,5000,10000,20000" and a["--eval-split"] == "val,test9"
        assert a["--record-traj"] == "3" and int(a["--seed"]) in (53, 54, 55)
        arms.add(s10.entry_arm(a))
        seen = {}
        monkeypatch.setattr(C5, "run_c5", lambda *x, **kw: seen.update(kw, args=x) or {
            "arm": "x", "cpu_s": 0.0, "train_cpu_s_per_1k_ticks": 0.0, "checkpoint_results": {}, "eval_splits": []})
        RL.main(c[2:])                                        # the CLI accepts every generated command
        assert seen["args"][9] == "val,test9" and seen["args"][0] == a["--agent"]   # positional split arg
        assert seen.get("save_weights", False) == (a.get("--save-weights") == "on")
    assert arms == {"dinos_a1", "dinos_a1h", "dinos_a2", "dinos_a3", "dinos_a4", "mlp_a1", "mlp_a1h", "mlp_a2", "mlp_a34",
                    "lstm_scratch", "mlp_hist10", "derp", "derp_hist10", "transfer_orig_orig_d0"}
    # rung 1: 5k ticks, grid 1k/2k/5k, val only, no baselines / existing arms; --arches + --cpu-affinity
    ents, info = m.main(["--arch-search", "--rung", "1", "--arches", "a1h,a2", "--cpu-affinity", "8-19", "--seeds",
                         "53,54", "--tag", "c18zy", "--out", str(tmp_path / "c18zy_entries.json")])
    assert info["n_entries"] == 2 * 4 and all(e["cpu_affinity"] == "8-19" for e in ents)
    assert {s10.entry_arm(s10.cmd_args(e["cmd"])) for e in ents} == {"dinos_a1h", "dinos_a2", "mlp_a1h", "mlp_a234"}
    for e in ents:
        a = s10.cmd_args(e["cmd"])
        assert a["--total-ticks"] == "5000" and a["--checkpoints"] == "1000,2000,5000" and a["--eval-split"] == "val"
    # transfer V.2: 20k source with the source grid, matched MLP transfer
    ents, info = m.main(["--transfer-v2", "a2", "--tag", "c18zx", "--out", str(tmp_path / "c18zx_entries.json")])
    got = {s10.entry_arm(s10.cmd_args(e["cmd"])) for e in ents}
    assert got == {"transfer_a2", "transfer_mlp_a2"}
    a = s10.cmd_args(ents[0]["cmd"])
    assert a["--source-checkpoints"] == "1000,2000,5000,10000,20000" and a["--pretrain-ticks"] == "20000"
    # test10 only with --confirm
    for bad in (["--arch-search", "--splits", "val,test10", "--report", "test10"], ["--arch-search", "--rung", "1",
                                                                                       "--confirm"],
                ["--arch-search", "--seeds", "11003"], ["--arch-search", "--arches", "a9"]):
        with pytest.raises(SystemExit):
            m.main(bad + ["--tag", "c18zw", "--out", str(tmp_path / "w.json")])
    # branch 2 arches: only with --code code-tuning-astra (branch 1, the default, does not register them)
    astra = os.path.dirname(os.path.dirname(HERE))
    with pytest.raises(SystemExit):
        m.main(["--arch-search", "--arches", "a5", "--tag", "c18zu", "--out", str(tmp_path / "u.json")])
    ents, info = m.main(["--arch-search", "--arches", "a5,a6,a8,a9", "--existing", "none", "--code", astra,
                         "--seeds", "53", "--tag", "c18zu", "--out", str(tmp_path / "c18zu_entries.json")])
    got = {s10.entry_arm(s10.cmd_args(e["cmd"])) for e in ents if "--agent" in e["cmd"]}
    assert got == {"dinos_a5", "dinos_a6", "dinos_a8", "dinos_a9", "mlp_a34", "mlp_a8", "mlp_a9"}
    assert info["matched_mlps"] == ["mlp_a34", "mlp_a8", "mlp_a9"] and info["code"] == astra
    for e in ents:
        if "--agent" in e["cmd"]:
            seen = {}
            monkeypatch.setattr(C5, "run_c5", lambda *x, **kw: seen.update(kw) or {
                "arm": "x", "cpu_s": 0.0, "train_cpu_s_per_1k_ticks": 0.0, "checkpoint_results": {}, "eval_splits": []})
            RL.main(e["cmd"][2:])
    ents, _ = m.main(["--arch-search", "--rung", "1", "--arches", "a8", "--code", astra, "--seeds", "53",
                      "--tag", "c18zt", "--out", str(tmp_path / "c18zt_entries.json")])
    assert {s10.entry_arm(s10.cmd_args(e["cmd"])) for e in ents} == {"dinos_a8", "mlp_a8"}
    monkeypatch.setenv("QRL_CONFIRM_LEDGER", str(tmp_path / "confirm_ledger.json"))
    with pytest.raises(SystemExit):           # fewer than 5 seeds
        m.main(["--arch-search", "--confirm", "--seeds", "60,61", "--tag", "c18zv", "--out", str(tmp_path / "v.json")])
    ents, info = m.main(["--arch-search", "--confirm", "--seeds", "60,61,62,63,64", "--tag", "c18zv",
                         "--out", str(tmp_path / "c18zv_entries.json")])
    assert info["report"] == "test10" and info["eval_splits"] == "val,test10"


def _fake_result(s10, entry, vals):
    """A result JSON that passes sum_c10.validate_agent for a frozen rung-1 entry; vals = {checkpoint: val mean}."""
    a = s10.cmd_args(entry["cmd"])
    es = {"val": s10.SPLIT_SEEDS["val"]}
    ck = {str(c): {"val": {"mean_ret": v, "episodes": [{"seed": x, "ret": v} for x in es["val"]]}}
          for c, v in vals.items()}
    return dict(protocol="c5", arm=s10.entry_arm(a), seed=int(a["--seed"]), env_family="orig",
                orig_variant=a["--orig-variant"], agent=a["--agent"], agent_variant=a["--agent-variant"],
                obs_scale=a["--obs-scale"], eps_schedule=a["--eps-schedule"], reward_div=float(a["--reward-div"]),
                checkpoints=sorted(int(x) for x in a["--checkpoints"].split(",")), eval_splits=["val"],
                startup_delay=5, state_size=7, total_ticks=int(a["--total-ticks"]), eval_seeds=es,
                checkpoint_results=ck)


@pytest.mark.skipif(not os.path.exists(MAKE_C18), reason="lab generator not present")
def test_promote_ranks_on_val_mean(tmp_path):
    m, s10, pr = _load("make_c18"), _load("sum_c10"), _load("promote")
    ef = tmp_path / "c18zp_entries.json"
    ents, _ = m.main(["--arch-search", "--rung", "1", "--arches", "a1,a2,a3,a4", "--seeds", "53,54", "--tag", "c18zp",
                      "--no-live", "--out", str(ef)])
    d = tmp_path / "r"
    d.mkdir()
    data = {"dinos_a1": {53: {1000: 10, 2000: 0, 5000: 30}, 54: {1000: 12, 2000: 0, 5000: 10}},  # 11/0/20 -> 20 @5000
            "dinos_a2": {53: {1000: 50, 2000: 0, 5000: 0}, 54: {1000: 0, 2000: 0, 5000: 40}},    # 25 @1000
            "dinos_a3": {53: {1000: 5, 2000: 5, 5000: 6}, 54: {1000: 5, 2000: 5, 5000: 6}},
            "dinos_a4": {53: {1000: 1, 2000: 1, 5000: 2}},                       # seed 54 missing
            "mlp_a1": {53: {1000: 99, 2000: 99, 5000: 99}, 54: {1000: 99, 2000: 99, 5000: 99}},  # control
            "mlp_a234": {53: {1000: 1, 2000: 1, 5000: 1}}}                       # incomplete control: not blocking
    by = {(s10.entry_arm(s10.cmd_args(e["cmd"])), int(e["seed"])): e for e in ents}
    for arm, sv in data.items():
        for s, v in sv.items():
            e = by[(arm, s)]
            out = os.path.basename(e["cmd"][e["cmd"].index("--out") + 1])
            json.dump(_fake_result(s10, e, v), open(d / out, "w"))
    a4_54 = by[("dinos_a4", 54)]["name"]
    with pytest.raises(SystemExit, match="neither complete nor --failed"):          # finding 3
        pr.main(["--tag", "c18zp", "--entries", str(ef), "--results", str(d)])
    with pytest.raises(SystemExit, match="required"):
        pr.load("c18zp", None, str(d))
    res = pr.main(["--tag", "c18zp", "--entries", str(ef), "--results", str(d), "--failed", a4_54])
    out = res["promote"]["spike300_d5"]
    assert [r["arm"] for r in out["ranking"]] == ["dinos_a2", "dinos_a1", "dinos_a3"]
    assert out["ranking"][0]["checkpoint"] == 1000 and out["ranking"][0]["score"] == 25
    assert out["ranking"][0]["matched_control"]["arm"] == "mlp_a234"            # rung-1 control reported next to it
    assert out["promoted"] == ["dinos_a2"] and out["promoted_controls"] == ["mlp_a2"]   # exact rung-2 control
    assert out["failed"] == ["dinos_a4"] and "mlp_a1" in out["controls"] and "mlp_a234" in out["incomplete"]
    # finding 2: MLP arms (incl. transfer_mlp_*) are never candidates
    assert not pr.is_candidate("transfer_mlp_a1h") and not pr.is_candidate("mlp_a1") and pr.is_candidate("transfer_a1h")
    assert not pr.is_candidate("transfer_orig_orig_d0") and not pr.is_candidate("dinos_x9")
    assert pr.matched_control("transfer_a3") == "transfer_mlp_a34" and pr.matched_control("transfer_a3", 1) == \
        "transfer_mlp_a234"
    # config validation against the frozen entry command
    f = d / os.path.basename(by[("dinos_a3", 53)]["cmd"][-1])
    r = json.load(open(f))
    r["total_ticks"] = 4000
    json.dump(r, open(f, "w"))
    with pytest.raises(SystemExit, match="total_ticks"):
        pr.main(["--tag", "c18zp", "--entries", str(ef), "--results", str(d), "--failed", a4_54])
    with pytest.raises(SystemExit, match="unknown"):
        pr.check_complete({}, pr.frozen(str(ef)), {"nope"})


@pytest.mark.skipif(not os.path.exists(MAKE_C18), reason="lab generator not present")
def test_sum_c10_v2_identities():
    s = _load("sum_c10")
    a = lambda **kw: dict({"--agent": "lstm_scratch"}, **kw)  # noqa: E731
    assert s.entry_arm(a(**{"--agent-variant": "a3"})) == "dinos_a3"
    assert s.entry_arm(a(**{"--agent-variant": "mlp_a234"})) == "mlp_a234"
    assert s.entry_arm(a(**{"--agent-variant": "dense_fixed"})) == "mlp_hist10"
    assert s.entry_arm(a(**{"--agent-variant": "fixed"})) == "lstm_scratch"
    t = {"--agent": "dinos_t", "--pretrain-env-family": "orig", "--pretrain-variant": "orig_d0"}
    assert s.entry_arm(dict(t, **{"--agent-variant": "fixed"})) == "transfer_orig_orig_d0"
    assert s.entry_arm(dict(t, **{"--agent-variant": "a1h"})) == "transfer_a1h"
    assert s.entry_arm(dict(t, **{"--agent-variant": "mlp_a1h"})) == "transfer_mlp_a1h"
    for x, arch in (("dinos_a1", "a1"), ("mlp_a1", "mlp_a1")):     # sum_c10 arm names == c5_protocol arm names
        assert C5.v2_arm("lstm_scratch", arch) == x
    arms = {"transfer_a1h", "derp", "transfer_orig_orig_d0", "dinos_a1h", "mlp_a1h", "transfer_mlp_a1h"}
    con = s.v2_contrasts(arms)
    for p in (("transfer_a1h", "derp"), ("transfer_a1h", "transfer_orig_orig_d0"), ("dinos_a1h", "mlp_a1h"),
              ("transfer_a1h", "transfer_mlp_a1h"), ("transfer_a1h", "dinos_a1h")):
        assert p in con
    assert s.SPLIT_SEEDS["test10"] == list(RL.TEST10_SEEDS)
