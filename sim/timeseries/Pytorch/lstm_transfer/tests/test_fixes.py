import os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
torch.set_num_threads(1)  # avoids heavy oversubscription on many-core hosts
from agentHuber import DDQNAgentLSTM
from environment import Environment
from networks import myLSTM

T, F, A = 4, 6, 3


def make_agent(**kw):
    torch.manual_seed(0); np.random.seed(0)
    return DDQNAgentLSTM(F, A, T, **kw)


def test_double_dqn_target_hand_computed():
    ag = make_agent()
    # tiny deterministic networks: Q(s) = fc bias only (zero fc weight), so Q is constant in s
    for m, q in ((ag.online_model, [0.0, 5.0, 1.0]), (ag.target_model, [10.0, 2.0, 7.0])):
        torch.nn.init.zeros_(m.fc.weight)
        m.fc.bias.data = torch.tensor(q)
    r = torch.tensor([1.0, 1.0]); d = torch.tensor([0.0, 1.0])
    ns = torch.zeros(2, T, F)
    y = ag.compute_targets(r, ns, d)
    # a* = argmax online = 1 -> Q_target = 2 (plain DQN would use max = 10)
    assert torch.allclose(y, torch.tensor([1.0 + 0.98 * 2.0, 1.0]))


def fill(ag, n=40):
    for i in range(n):
        ag.remember(np.random.rand(T, F), i % A, float(i), np.random.rand(T, F), i % 7 == 0)


def test_target_fixed_between_syncs():
    ag = make_agent(target_update_every=5)
    fill(ag)
    w0 = copy.deepcopy(ag.target_model.state_dict())
    for _ in range(4):
        ag.replay(8)
    for k, v in ag.target_model.state_dict().items():
        assert torch.equal(v, w0[k])
    assert not all(torch.equal(v, w0[k]) for k, v in ag.online_model.state_dict().items())
    ag.replay(8)  # 5th call syncs
    for k, v in ag.target_model.state_dict().items():
        assert torch.equal(v, ag.online_model.state_dict()[k])


def test_forward_independent_of_previous_call():
    m = myLSTM(F, 16, 2, A)
    x, y = torch.rand(1, T, F), torch.rand(1, T, F)
    o1 = m(x); m(y); o2 = m(x)
    assert torch.equal(o1, o2)
    # legacy carries state (documents the old defect)
    ml = myLSTM(F, 16, 2, A, legacy=True)
    assert not torch.equal(ml(x), ml(x))


def test_replay_one_optimizer_step():
    ag = make_agent()
    fill(ag)
    ag.replay(16)
    assert ag.optimizer_steps == 1
    leg = make_agent(legacy=True); fill(leg); leg.replay(16)
    assert leg.optimizer_steps == 16  # old behaviour, for comparison


def test_fine_tune_runs_for_every_model():
    env = Environment()
    for i in range(5):  # shared env, as in lstm_transfer.py
        ag = make_agent()
        n = ag.fine_tune(env, fine_tune_steps=20, batch_size=4)
        assert n == 20, f"model {i} ran {n} steps"
        env.time = 5000  # emulate finished evaluation episode


# ---------------------------------------------------------------- review follow-up
import importlib, importlib.util, random, subprocess, tempfile, types
from unittest import mock

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_batched_replay_gradient_only_through_chosen_action():
    ag = make_agent()
    fill(ag)
    mb = [ag.memory[i] for i in range(8)]
    states = torch.tensor(np.array([m[0] for m in mb]), dtype=torch.float32)
    actions = torch.tensor([m[1] for m in mb]).unsqueeze(1)
    targets = torch.rand(8)
    ag.online_model.train()
    out = ag.online_model(states)
    out.retain_grad()
    loss = ag.criterion(out.gather(1, actions).squeeze(1), targets)
    loss.backward()
    mask = torch.zeros_like(out, dtype=torch.bool).scatter_(1, actions, True)
    assert torch.all(out.grad[~mask] == 0)
    assert torch.any(out.grad[mask] != 0)


def _load_script(name):
    spec = importlib.util.spec_from_file_location("t_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())  # scripts makedirs their results folder at import
    try:
        os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def _run_main_mocked(name, **kw):
    mod = _load_script(name)
    agent = mock.MagicMock()
    agent.pretrained_model = "m.pth"
    agent.sequence_length, agent.state_size = 10, 6
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    os.makedirs(mod.RESULTS_FOLDER, exist_ok=True)
    try:
        with mock.patch.object(mod, "Environment"), \
             mock.patch.object(mod, "DDQNAgentLSTM", return_value=agent) as ctor, \
             mock.patch.object(mod, "fine_tune_agent"), \
             mock.patch.object(mod, "evaluate_agent", return_value=[0.0]), \
             mock.patch.object(mod, "plot_accumulated_rewards"):
            mod.main(model_steps=(1,), **kw)
    finally:
        os.chdir(cwd)
    return ctor


def test_both_scripts_honour_legacy():
    for name in ("lstm_transfer", "lstm_transfer_quantum_compare"):
        for legacy in (True, False):
            ctor = _run_main_mocked(name, legacy=legacy)
            assert ctor.call_args.kwargs["legacy"] is legacy, name
        with mock.patch.dict(os.environ, {"QRL_LEGACY": "1"}):
            assert _run_main_mocked(name).call_args.kwargs["legacy"] is True
        with mock.patch.dict(os.environ, {"QRL_LEGACY": "0"}):
            assert _run_main_mocked(name).call_args.kwargs["legacy"] is False
        mod = _load_script(name)
        assert mod.legacy_from_env(["--legacy"]) is True


def test_wrappers_do_not_reset_env():
    for name in ("lstm_transfer", "lstm_transfer_quantum_compare"):
        mod = _load_script(name)
        env, agent = mock.MagicMock(), mock.MagicMock()
        agent.pretrained_model = "m.pth"
        cwd = os.getcwd()
        os.chdir(tempfile.mkdtemp())
        os.makedirs(mod.RESULTS_FOLDER, exist_ok=True)
        try:
            mod.fine_tune_agent(agent, env, 1, 1)
        finally:
            os.chdir(cwd)
        env.reset.assert_not_called()


def test_fine_tune_reset_only_when_not_legacy():
    for legacy, n_reset in ((False, 1), (True, 0)):
        ag = make_agent(legacy=legacy)
        env = mock.MagicMock()
        env.reset.return_value = np.zeros(F)
        env.step.return_value = (np.zeros(F), 0.0, 0, 0, 0, 0, 0, True)
        ag.fine_tune(env, 3, 2)
        assert env.reset.call_count == n_reset


def test_legacy_matches_parent_commit_trace():
    repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()
    rel = os.path.relpath(os.path.abspath(HERE), repo)
    d = tempfile.mkdtemp()
    for f in ("agentHuber.py", "networks.py"):
        try:
            src = subprocess.check_output(["git", "show", f"ee99307^:{rel}/{f}"], cwd=repo, text=True)
        except subprocess.CalledProcessError:
            import pytest; pytest.skip("parent commit not available")
        open(os.path.join(d, f), "w").write(src)

    def run(agent_cls):
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        ag = agent_cls()
        rng = np.random.RandomState(5)
        acts = []
        s = rng.rand(T, F)
        for i in range(30):
            a = ag.act(s)
            acts.append(a)
            ns = rng.rand(T, F)
            ag.remember(s, a, float(rng.randn()), ns, i % 9 == 8)
            ag.replay(4)
            s = ns
        return acts, ag.loss_history, {k: v.clone() for k, v in ag.online_model.state_dict().items()}, ag.epsilon

    new = run(lambda: DDQNAgentLSTM(F, A, T, legacy=True))
    # load the old modules under their original names, then restore the current ones
    saved = {k: sys.modules.pop(k) for k in ("agentHuber", "networks") if k in sys.modules}
    sys.path.insert(0, d)
    try:
        old_mod = importlib.import_module("agentHuber")
        old = run(lambda: old_mod.DDQNAgentLSTM(F, A, T))
    finally:
        sys.path.remove(d)
        for k in ("agentHuber", "networks"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    assert new[0] == old[0]
    assert new[1] == old[1] and len(new[1]) > 0
    assert new[3] == old[3]
    for k in old[2]:
        assert torch.equal(new[2][k], old[2][k]), k


# ---------------------------------------------------------------- launcher-run support (QRL_RUN_DIR, seed, telemetry)
import hashlib, json


def _tree_state(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[p] = (os.stat(p).st_mtime_ns, hashlib.md5(open(p, "rb").read()).hexdigest())
    return out


def _real_run(name, run_dir, seed, eval_steps=20):
    mod = _load_script(name)
    env_vars = {"QRL_RUN_DIR": run_dir, "MPLCONFIGDIR": tempfile.mkdtemp()}
    cwd = os.getcwd()
    os.chdir(HERE)  # real pretrained modelCPU_1.pth lives here
    try:
        with mock.patch.dict(os.environ, env_vars):
            mod.main(legacy=False, model_steps=(1,), eval_steps=eval_steps, num_evaluations=1,
                     fine_tune_steps=5, seed=seed)
    finally:
        os.chdir(cwd)


def test_run_dir_isolation_telemetry_and_seed():
    if not os.path.exists(os.path.join(HERE, "modelCPU_1.pth")):
        import pytest; pytest.skip("modelCPU_1.pth missing")
    for name, legacy_dir in (("lstm_transfer", "final_metrics_lstm"),
                             ("lstm_transfer_quantum_compare", "final_quantum_compare_metrics_lstm")):
        orig = os.path.join(HERE, legacy_dir)
        before = _tree_state(orig) if os.path.isdir(orig) else None
        rewards = []
        for _ in range(2):
            run = tempfile.mkdtemp()
            _real_run(name, run, seed=123)
            res = os.path.join(run, "results")
            assert os.path.exists(os.path.join(res, "modelCPU_1_fine_tuned.pth"))
            assert os.path.exists(os.path.join(res, "model_1", "rewards.txt"))
            lines = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
            assert len(lines) == 1
            assert set(lines[0]) == {"step", "model", "reward", "mean_vms"}
            assert lines[0]["step"] == 0 and lines[0]["model"] == 1
            assert json.load(open(os.path.join(run, "progress.json"))) == {"step": 1, "total": 1}
            rewards.append(open(os.path.join(res, "model_1", "rewards.txt")).read())
        assert rewards[0] == rewards[1]
        after = _tree_state(orig) if os.path.isdir(orig) else None
        assert before == after, "legacy results folder was modified"
