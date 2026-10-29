"""Cycle 20: two deeper Dinos V.2 architectures (stacked LSTM128 trunks + the locked a1h dense head) with exact
parameter-matched MLP controls; registry, CLI (lstm_scratch / dinos_t with a source-checkpoint grid) and the cycle-19
DDQN add-ons (--n-step / --dueling / --per). Existing arms are covered by the fingerprint suites (test_v2/test_c19/test_c7)."""
import contextlib
import json
import os

import numpy as np
import pytest
import torch

from tuning.legacy import agent_lstm_fixed as FX
from tuning.legacy import agent_lstm_v2 as V2
from tuning.legacy import c5_protocol as C5
from tuning.legacy import run_legacy as RL

F, SEQ, A = 7, 10, 3
NEW = {"a11": ("l128-l128-cat-d128-d128", 236547, "mlp_a11", "m450", 236253),
       "a12": ("l128-l128-l128-cat-d128-d128", 368643, "mlp_a12", "m571", 368869)}
COMMON = dict(eval_splits="val", eval_ticks=30, reward_div=10.0, obs_scale="on", eps_schedule="linear10k",
              env_family="orig", orig_variant="spike300_d5")


@pytest.fixture
def small_splits(monkeypatch):
    monkeypatch.setitem(RL.SPLITS, "val", (1100,))


@pytest.fixture(autouse=True)
def one_thread():
    """Real runs use --threads 1; multi-threaded torch on the shared test cores oversubscribes badly."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _lstm(h_in, h):          # one single-layer torch LSTM: 4h(in + h) weights + 2 x 4h biases
    return 4 * h * (h_in + h) + 8 * h


@pytest.mark.parametrize("arch", sorted(NEW))
def test_registry_build_and_counts(arch):
    spec, n, mlp, mspec, mn = NEW[arch]
    assert V2.ARCHS[arch] == spec and V2.ARCHS[mlp] == mspec and V2.MATCH[mlp] == (arch,) and arch in V2.LABEL
    assert spec.endswith(V2.ARCHS["a1h"].partition("-")[2])          # same cat-d128-d128 head as a1h
    k = spec.count("l128")
    assert n == _lstm(F, 128) + (k - 1) * _lstm(128, 128) + (135 * 128 + 128) + (128 * 128 + 128) + (128 * A + A)
    torch.manual_seed(0)
    net = V2.build_net(arch, F, SEQ, A)
    assert [type(m).__name__ for m in net.layers] == ["LSTM"] * k + ["Identity", "Linear", "Linear"]
    assert all(m.hidden_size == 128 and m.num_layers == 1 for m in net.layers[:k])
    assert V2.n_params(net) == n and V2.n_params(V2.build_net(mlp, F, SEQ, A)) == mn
    assert abs(mn / n - 1) < 0.01
    W = int(mspec[1:])                      # closest mW: neighbours are farther from the arch count
    assert all(abs(w * w + 75 * w + 3 - n) > abs(mn - n) for w in (W - 1, W + 1))
    x = torch.randn(5, SEQ, F)
    with torch.no_grad():
        y = net(x)
        rows = torch.cat([net(x[i:i + 1]) for i in range(5)])
    assert y.shape == (5, A) and torch.allclose(y, rows, atol=1e-6)
    assert V2.build_net(mlp, F, SEQ, A)(x).shape == (5, A)


def _run(tmp_path, tag, agent, **kw):
    out = str(tmp_path / tag / "r.json")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        C5.run_c5(agent, 53, out=out, **dict(COMMON, **kw))
    return json.load(open(out))


@pytest.mark.parametrize("arch", ["a11", "a12", "mlp_a11", "mlp_a12"])
def test_smoke_scratch_300(tmp_path, small_splits, arch):
    r = _run(tmp_path, arch, "lstm_scratch", agent_variant=arch, total_ticks=300, checkpoints=[300])
    assert r["arm"] == C5.v2_arm("lstm_scratch", arch) and r["net_spec"] == V2.ARCHS[arch]
    assert r["n_params"] == V2.n_params(V2.build_net(arch, F, SEQ, A))
    assert r["matched_to"] == (list(V2.MATCH[arch]) if arch in V2.MATCH else None)
    assert np.isfinite(r["checkpoint_results"]["300"]["val"]["mean_ret"]) and r["replays"] > 0
    assert "ddqn_options" not in r or not (r["ddqn_options"]["dueling"] or r["ddqn_options"]["per"])


def test_smoke_transfer_a12_nstep_cli(tmp_path, small_splits, monkeypatch):
    real = C5.run_c5
    monkeypatch.setattr(C5, "run_c5", lambda *a, **kw: real(*a, **dict(kw, eval_ticks=30)))
    out = str(tmp_path / "t.json")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        RL.main(["--agent", "dinos_t", "--agent-variant", "a12", "--env-family", "orig", "--orig-variant", "spike300_d5",
                 "--pretrain-env-family", "orig", "--pretrain-variant", "orig_d0", "--pretrain-ticks", "120",
                 "--finetune-ticks", "120", "--source-checkpoints", "60,120", "--checkpoints", "60,120",
                 "--seed", "53", "--eval-split", "val", "--n-step", "3", "--out", out])
    r = json.load(open(out))
    assert r["arm"] == "transfer_a12" and r["ddqn_options"]["n_step"] == 3
    assert r["pretrain"]["n_params"] == r["n_params"] == 368643
    assert set(r["source_selection"]["val_by_checkpoint"]) == {"60", "120"}
    assert all(np.isfinite(r["checkpoint_results"][c]["val"]["mean_ret"]) for c in ("60", "120"))


@pytest.mark.parametrize("arch", ["a11", "a12", "mlp_a11", "mlp_a12"])
def test_cli_addons_all_on(tmp_path, small_splits, monkeypatch, arch):
    real = C5.run_c5
    monkeypatch.setattr(C5, "run_c5", lambda *a, **kw: real(*a, **dict(kw, eval_ticks=30)))
    out = str(tmp_path / "r.json")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        RL.main(["--agent", "lstm_scratch", "--agent-variant", arch, "--env-family", "orig", "--orig-variant",
                 "spike300_d5", "--total-ticks", "60", "--checkpoints", "60", "--seed", "53", "--eval-split", "val",
                 "--n-step", "3", "--dueling", "on", "--per", "on", "--out", out])
    r = json.load(open(out))
    o = r["ddqn_options"]
    assert o["n_step"] == 3 and o["dueling"] and o["per"]
    w = 128 if not V2.is_mlp(arch) else int(V2.ARCHS[arch][1:])
    assert r["n_params"] == V2.n_params(V2.build_net(arch, F, SEQ, A)) + w + 1      # dueling adds V: Dense(w, 1)
    assert "dueling[V: Dense(" in r["architecture"] and r["replays"] > 0
    head = V2.make_scratch(arch, F, A, SEQ, ddqn=dict(dueling=True)).online_model
    assert isinstance(head.q if not V2.is_mlp(arch) else head.net.fc3, FX.DuelingHead)
