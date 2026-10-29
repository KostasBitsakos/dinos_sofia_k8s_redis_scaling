"""reward_mode / learn_scale propagation, TD diagnostics, and the sensitivity configs not touching the reward."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import train_eval  # noqa: E402

ARGS = ["--train-steps", "40", "--eval-episodes", "1", "--log-every", "20", "--batch-size", "8",
        "--checkpoint-every", "0", "--seed", "1", "--amendment-id", "R-TEST-000"]


def _run(tmp_path, monkeypatch, *extra, agent="lstm_scratch"):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    train_eval.main(["--agent", agent, *ARGS, *extra])
    rows = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    return rows, json.load(open(tmp_path / "run_config.json"))


def test_sensitivity_configs_do_not_set_reward_mode():
    import sensitivity
    from params import SimParams
    for n in sensitivity.names():
        assert "reward_mode" not in sensitivity.overrides(n)
        assert sensitivity.params(n, SimParams().with_(reward_mode="dinos")).reward_mode == "dinos"


def test_default_run_is_cost_and_recorded(tmp_path, monkeypatch):
    rows, cfg = _run(tmp_path, monkeypatch)
    assert cfg["reward_mode"] == "cost" and cfg["learn_scale"] == 0.02
    tr = [r for r in rows if "train_mean_reward" in r]
    assert all(r["reward_mode_dinos"] == 0 and r["learn_scale"] == 0.02 for r in tr)


def test_dinos_mode_propagates_to_training_and_evaluation(tmp_path, monkeypatch):
    rows, cfg = _run(tmp_path, monkeypatch, "--reward-mode", "dinos", "--sensitivity", "default")
    assert cfg["reward_mode"] == "dinos" and len(cfg["reward_dinos_sha256"]) == 64
    tr = [r for r in rows if "train_mean_reward" in r]
    assert all(r["reward_mode_dinos"] == 1 for r in tr)
    ev = [r for r in rows if "eval_return" in r][0]
    # the cost reward per step is >= -(a few); the DInos reward of the same steps is of a very different magnitude
    assert ev["eval_return"] != pytest.approx(ev["eval_cost"] + ev["eval_sla"] + ev["eval_churn"], rel=1e-3)
    assert tr[0]["train_mean_reward"] != 0.0


def test_gate_agents_refuse_dinos():
    with pytest.raises(SystemExit):
        train_eval.main(["--agent", "baselines", "--unfrozen", "--reward-mode", "dinos"])


def test_bad_learn_scale_refused():
    for bad in ("0", "-0.02", "nan"):
        with pytest.raises(SystemExit):
            train_eval.main(["--agent", "lstm_scratch", "--unfrozen", "--learn-scale", bad])


def test_learn_scale_is_applied_to_stored_reward_for_every_agent(tmp_path, monkeypatch):
    import agents
    stored = {}
    for agent in ("lstm_scratch", "derp"):
        for scale in ("0.02", "0.5"):
            cls = agents.make_lstm if agent == "lstm_scratch" else agents.make_derp
            made = []
            orig = cls

            def spy(*a, _orig=orig, _made=made, **k):
                ag = _orig(*a, **k)
                _made.append(ag)
                return ag
            monkeypatch.setattr(agents, "make_lstm" if agent == "lstm_scratch" else "make_derp", spy)
            d = tmp_path / f"{agent}{scale}"
            d.mkdir()
            monkeypatch.setenv("QRL_RUN_DIR", str(d))
            train_eval.main(["--agent", agent, *ARGS, "--learn-scale", scale, "--log-every", "40"])
            monkeypatch.setattr(agents, "make_lstm" if agent == "lstm_scratch" else "make_derp", orig)
            stored[(agent, scale)] = [m[2] for m in made[0].memory]
    for agent in ("lstm_scratch", "derp"):
        a, b = np.array(stored[(agent, "0.02")]), np.array(stored[(agent, "0.5")])
        # before the first replay (memory <= batch size 8) both runs act identically, so the raw rewards are equal and
        # the stored ones differ exactly by the ratio of the scales (later actions depend on the learned weights)
        assert len(a) == 40 and np.allclose(b[:8], a[:8] * 25.0, rtol=1e-6)


@pytest.mark.parametrize("agent", ["lstm_scratch", "derp"])
def test_td_diagnostics_logged_and_do_not_perturb_training(tmp_path, monkeypatch, agent):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(), b.mkdir()
    out = []
    for d, extra in ((a, []), (b, ["--diag-every", "0"])):
        monkeypatch.setenv("QRL_RUN_DIR", str(d))
        train_eval.main(["--agent", agent, *ARGS, "--train-steps", "60", "--log-every", "20", *extra])
        out.append([json.loads(l) for l in open(d / "metrics.jsonl")])
    with_diag = [r for r in out[0] if "train_mean_reward" in r]
    no_diag = [r for r in out[1] if "train_mean_reward" in r]
    assert not any("diag_td_abs_mean" in r for r in no_diag)
    d = with_diag[-1]
    for k in ("diag_td_abs_mean", "diag_td_abs_p50", "diag_td_abs_p90", "diag_td_abs_p99", "diag_td_abs_max",
              "diag_q_abs_mean", "diag_q_abs_max", "diag_target_abs_mean", "diag_target_abs_max",
              "diag_huber_saturation"):
        assert np.isfinite(d[k]) and d[k] >= 0.0
    assert 0.0 <= d["diag_huber_saturation"] <= 1.0 and d["diag_td_abs_p50"] <= d["diag_td_abs_p99"] <= d["diag_td_abs_max"]
    # instrumentation must not change training: identical losses and rewards with and without it
    assert [(r["step"], r["loss"], r["train_mean_reward"]) for r in with_diag] == \
        [(r["step"], r["loss"], r["train_mean_reward"]) for r in no_diag]


def test_dinos_variant_validated_propagated_and_recorded(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        train_eval.main(["--agent", "lstm_scratch", "--unfrozen", "--dinos-variant", "nonsense"])
    import argparse
    ns = argparse.Namespace(reward_mode="dinos", dinos_variant="legacy_p95")
    assert train_eval.base_params(ns).dinos_variant == "legacy_p95"
    assert train_eval.base_params(argparse.Namespace(reward_mode="dinos")).dinos_variant == "default"
    _, cfg = _run(tmp_path, monkeypatch, "--reward-mode", "dinos", "--dinos-variant", "legacy_p95",
                  "--sensitivity", "default")
    assert cfg["dinos_variant"] == "legacy_p95"
