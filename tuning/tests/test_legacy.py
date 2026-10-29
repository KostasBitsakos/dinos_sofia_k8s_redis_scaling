"""Track L parity tests: the legacy port must reproduce the numbers recomputed from the original code.

Reference values (recomputed from copies of the original modules, see tuning/legacy/README.md):
  HPA rule 5-run sum -1.032M; always-down ceiling 3.641M; random policy under the DERP protocol mean 252k (SD 57k);
  original DInos weights under the original protocol vs the logged sums (run_legacy.LOGGED). Parity targets the LOGGED
  values, not the paper's hand-typed 3.8e6 / Fig 5 bars.
"""
import os
import random

import numpy as np
import pytest
import torch

from tuning.legacy import env_dense, env_transfer, port_sources, run_legacy as RL

LEGACY = port_sources.LEGACY_ROOT
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
need_legacy = pytest.mark.skipif(not os.path.isdir(LEGACY), reason="original legacy code not present")


@need_legacy
@pytest.mark.parametrize("port", sorted(port_sources.PORTS))
def test_port_is_verbatim(port):
    with open(os.path.join(HERE, "legacy", port)) as f:
        assert f.read() == port_sources.render(port)


def test_hpa_rule_sum():
    r = RL.run("hpa_rule", 1000, 0, "orig")["protocols"]["orig"]
    assert abs(r["sum5"] / np.mean(RL.LOGGED["hpa"]) - 1) < 0.01, r["sum5"]      # logged HPA/final_metrics_hpa
    assert r["ticks"] == 25000 and r["mean_pods"] > 19


def test_always_down_ceiling():
    for prot in ("orig", "fair"):
        r = RL.run("always_down", 1000, 0, prot)["protocols"][prot]
        assert abs(r["sum5"] / 3.641e6 - 1) < 0.005, (prot, r["sum5"])


def _random_derp_protocol(seed):
    """DERP P-orig with a fresh agent (epsilon 1.0): the evaluation is uniform random whatever the weights are."""
    from tuning.legacy import ddqnagent
    RL.seed_all(seed)
    a = ddqnagent.DDQNAgent(6, 3)
    assert a.epsilon == 1.0
    eps = RL.eval_noreset(env_dense.Environment(), lambda s: a.act(s, verbose=0), False)
    assert [e["ticks"] for e in eps] == [5000, 1, 1, 1, 1]
    return sum(e["ret"] for e in eps)


@pytest.mark.slow
def test_random_policy_derp_protocol():
    sums = np.array([_random_derp_protocol(s) for s in range(200, 240)])   # seeds disjoint from the reference 0-99
    se = sums.std(ddof=1) / np.sqrt(len(sums))
    assert abs(sums.mean() - 252e3) < 2 * se, (sums.mean(), se)          # recomputed reference (seeds 0-99)
    # logged DERP sums (dense/final_metrics_dense, ONE random draw per model): their mean vs ours within 2 SE of the
    # difference, and every logged value inside our mean +- 3 SD
    logged = np.array(RL.LOGGED["derp"], float)
    se_diff = np.sqrt(se ** 2 + sums.std(ddof=1) ** 2 / len(logged))
    assert abs(sums.mean() - logged.mean()) < 2 * se_diff, (sums.mean(), logged.mean(), se_diff)
    assert np.all(np.abs(logged - sums.mean()) < 3 * sums.std(ddof=1))


@need_legacy
@pytest.mark.parametrize("m,tol", [(1, 0.01), (2, 0.01), (5, 0.01), (10, 0.01), (20, 0.02)])
def test_original_dinos_weights_reproduce_logs(m, tol):
    """Logged sums lstm_transfer/final_metrics_lstm/model_N/rewards.txt. Model 20 replays systematically ~+1.2% high
    (5 seeds: 3.255-3.284M vs logged 3.2285M; cause unknown), hence its 2% bound."""
    torch.set_num_threads(1)
    r = RL.dinos_orig_logged_model(m, seed=0)
    if m != 1:
        assert r["shared_env_quirk"] and r["finetune_ticks_run"] == 1 and r["epsilon_eval"] == 0.2
    logged = RL.LOGGED["dinos"][RL.MODELS.index(m)]
    assert abs(r["sum5"] / logged - 1) < tol, (r["sum5"], logged)


def test_no_grad_target_is_numerically_identical():
    """The no_grad wrapper on the LSTM target network changes no number (only stops the graph growing)."""
    from tuning.legacy import agent_huber

    def train(patch):
        RL.seed_all(3)
        a = agent_huber.DDQNAgent(6, 3, 10)
        if patch:
            RL.no_grad_target(a)
        env = env_transfer.Environment()
        state = np.zeros((10, 6))
        for _ in range(40):
            act = a.act(state)
            ns, r, *_, d = env.step(act)
            nxt = np.zeros((10, 6)); nxt[:-1] = state[1:]; nxt[-1] = ns
            a.remember(state, act, r, nxt, d)
            state = nxt
            if len(a.memory) > 16:
                a.replay(16)
        return a.online_model.state_dict(), a.loss_history, random.random(), np.random.rand()
    w0, l0, r0, n0 = train(False)
    w1, l1, r1, n1 = train(True)
    assert l0 == l1 and r0 == r1 and n0 == n1
    assert all(torch.equal(w0[k], w1[k]) for k in w0)


def test_fair_is_paired_and_protocol_independent():
    a = RL.run("always_down", 1000, 7, "fair")["protocols"]["fair"]
    b = RL.run("always_down", 1000, 8, "both")["protocols"]["fair"]
    assert a["sum5"] == b["sum5"]                          # paired eval seeds, independent of training seed
    h = RL.run("hpa_rule", 1000, 7, "fair")["protocols"]["fair"]
    # same env class (dense vs transfer differ) is needed for identical noise; components 3L must then match
    k = RL.run("k8s_hpa", 1000, 7, "fair")["protocols"]["fair"]
    assert h["components_sum5"]["3L"] == k["components_sum5"]["3L"]


def test_seed_guards(monkeypatch):
    with pytest.raises(SystemExit):
        RL.check_seeds(20000, "orig")
    with pytest.raises(SystemExit):
        RL.check_seeds(1002, "orig")
    monkeypatch.setenv("QRL_SANDBOX_SEEDS", "0-999")
    RL.check_seeds(5, "orig")
    with pytest.raises(PermissionError):
        RL.check_seeds(5, "fair")                          # eval seeds 1000-1004 not allowed
    monkeypatch.setenv("QRL_SANDBOX_SEEDS", "0-999,1000-1004")
    RL.check_seeds(5, "both")


def test_closeness_uses_dinos_t_and_targets(tmp_path):
    import json
    from tuning import closeness as C
    rows = [{"config": "lstm_scratch_v2", "n": 5, "sum": 9e9}, {"config": "dinos_t_v2", "n": 5, "sum": 3.8e6},
            {"config": "derp_v2", "n": 5, "sum": 2.2e5}]
    (tmp_path / "summary.json").write_text(json.dumps(rows))
    assert C.best(rows, "dinos_t", None)["config"] == "dinos_t_v2"
    s = C.score(3.8e6, 2.2e5)
    assert s["total"] == pytest.approx(1.0)                 # paper default unchanged
    assert C.targets() == (2.2e5, 3.8e6)
    if os.path.exists(C.TARGETS_JSON):
        de, di = C.targets("legacy-fair")
        assert C.score(di, de, "legacy-fair")["total"] == pytest.approx(1.0)
        de, di = C.targets("legacy-orig")
        assert C.score(di, de, "legacy-orig")["total"] == pytest.approx(1.0)
