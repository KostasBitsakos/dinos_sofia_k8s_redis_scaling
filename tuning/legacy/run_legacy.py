#!/usr/bin/env python3
"""Track L: one run of one agent in the faithful port of the legacy DInos/DERP simulation (tuning/legacy/*.py).

  python3 tuning/legacy/run_legacy.py --agent derp --train-ticks 1000 --seed 0 --protocol both --out r.json

Agents (original architectures + hyperparameters, P = ~/simulations/redisk8s/timeseries/Pytorch):
  derp          dense DDQN (P/dense/ddqnagent.py), trained as P/dense/doubledeep.py:9-82
  lstm_scratch  "RS": LSTM agentHuber (P/agentHuber.py), trained as P/doubledeepCPU.py:9-85
  dinos_t       DInos+transfer as P/lstm_transfer/lstm_transfer.py:122-145: pretrain agentHuber for --train-ticks (as
                doubledeepCPU.py produced modelCPU_N.pth), load into DDQNAgentLSTM (epsilon 0.2), fine-tune 1000 ticks
                on the target env, evaluate
  Env variants (all verbatim): --train-env / --env {dense,dense2,cpu,cpu2,lstm_transfer}. Defaults (TRAIN_ENV/EVAL_ENV
  below) = the paper's final-evaluation environments per .lab/tuning/deep/claude_opus_2.md and codex_astra_2.md.
  hpa_rule      P/HPA/simpleagent.py threshold rule (cpu>50 or mem>50 -> +1 else -1); nothing is trained
  k8s_hpa       our Kubernetes HPA replica formula (sim_v2/policies.py K8sHPA, target --hpa-target, default 50)
                applied to the legacy cpu observation; one +-1 move per tick toward its desired count
  always_down   action 0 every tick (the optimum of the legacy reward, which rewards pods only through -10*pods)
"--train-ticks" N = N/1000 training chunks of 1000 ticks on ONE env object that is never reset (state window zeroed at
each chunk start), as the original "N steps" models. Pretrain ticks (dinos_t, source task) and fine-tune ticks (target)
are reported separately.

Protocols (each gives 5 episode returns, their sum, reward components and mean pods):
  orig  "reproduction of figure protocol" = exactly the scripts' evaluation:
        derp/lstm_scratch: a FRESH agent (epsilon 1.0 -> uniform random actions) loads the trained weights, one env,
        never reset, 5 runs until done (run 1 = 5000 ticks, runs 2-5 = 1 tick each)  [P/dense/doubledeepfinal.py,
        P/doubledeepCPUfinal.py]; dinos_t: epsilon 0.2 (decayed during fine-tuning), env shared between fine-tune and
        eval, reset before each of 5 x 5000-tick runs; the shared-env quirk (models 2..20 meet an env already at
        t>=5000, so fine-tuning stops after 1 tick) is reproduced for --train-ticks != 1000
        [P/lstm_transfer/lstm_transfer.py:12-69,122-145]; hpa_rule/k8s_hpa/always_down: P/HPA/hpa_final.py (reset,
        5 x 5000). The RNG stream continues from training (one process), seeded only by --seed.
  fair  greedy (no exploration draw), env reset before each of 5 paired episodes with eval seeds 1000-1004
        (numpy/random/torch reseeded per episode, LSTM hidden state zeroed), 5000 ticks each, same for all agents.
        Within one env class the exogenous noise is then identical across agents.
Cycle 4 (--env-variant coupled, tuning/legacy/coupled_env.py): the same verbatim envs wrapped with the synthetic
load-sharing coupling g = 6L/(70n) (c, m, l scaled by g; cpu/mem clipped at 100; reward law unchanged). Envs: derp
dense, lstm_scratch and dinos_t lstm_transfer, rule agents dense (COUPLED_ENV). Only greedy evaluation (--protocol fair)
is allowed. Training resets an episode when the env reports done (t >= 5000; legacy training never resets). dinos_t:
pretrain (B - 1000) ticks on the LEGACY (uncoupled) source env (dense), then fine-tune 1000 ticks on the coupled
target, so B counts all of its training ticks.
Cycle 5 (any of --source-variant/--pretrain-ticks/--finetune-ticks/--total-ticks/--checkpoints/--target-env, coupled
only): tuning/legacy/c5_protocol.py -- common coupled lstm_transfer target for all arms, transfer from legacy / coupled /
no (random-init) source, episode-resetting training from the reset observation, private env RNG streams (matched target
traces), checkpoint evaluation on deep copies. See that module's docstring.
Eval splits (--eval-split, comma list, one invocation evaluates the same trained/fine-tuned weights on each):
  fair 1000-1004 (default, legacy behaviour), val 1100-1104, test 2000-2019, test2 3000-3019 (cycle 6);
  greedy, reset, 5000 ticks per episode.
  With any split other than fair alone, or with coupled, each episode also reports p95 latency (of the observed
  latency) and each split mean/sd of the episode return.
Non-numerical deviations from the originals (do not change any number): no plots/tqdm/prints/files; the LSTM target
network forward runs under torch.no_grad() (the original leaks an ever-growing autograd graph through the stateful
hidden state; values are identical, tested in tests/test_legacy.py).
Dashboard observers (M-20261021-008, opt-in, default off, cycle-5 protocol only; tuning/legacy/live_rec.py):
  --live-file PATH   ring buffer of the last 600 ticks (tick, phase train/eval, load, pods, pending, action, reward),
                     rewritten atomically every 25 ticks; the dashboard reads tuning/results/**/live/*.json
  --record-traj N    per-tick trajectory (load, cpu, mem, latency, pods, pending, reward, action) of the first N greedy
                     eval episodes of every checkpoint's last split, inside the result JSON
  Neither draws random numbers or changes the training/evaluation order: every number is identical with or without.
Seeds: --seed in 0-19999 except the eval block 1000-1004; if QRL_SANDBOX_SEEDS is set (sandbox launcher) every seed
used must lie inside it (same rule as sim_v2/env.py _sandbox_seed_guard). Telemetry (PLAN 7): $QRL_RUN_DIR/metrics.jsonl
and progress.json.
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tuning.legacy import agent_huber, agent_huber_transfer, ddqnagent, env_cpu, env_cpu2, env_dense, env_dense2  # noqa: E402
from tuning.legacy import env_transfer  # noqa: E402
from tuning.legacy import simpleagent  # noqa: E402
from tuning.legacy import coupled_env  # noqa: E402
from tuning.legacy import agent_lstm_v2  # noqa: E402

AGENTS = ("derp", "lstm_scratch", "dinos_t", "hpa_rule", "k8s_hpa", "always_down")
TRAIN_TICKS = (1000, 2000, 5000, 10000, 20000)
# Every legacy env with the 8-tuple step() interface, verbatim (P/environment.py and P/environment2.py return a 3-tuple
# and belong to older scripts, so they are not offered). Which variant produced the paper numbers is NOT decided here.
ENVS = {"dense": env_dense.Environment, "dense2": env_dense2.Environment, "cpu": env_cpu.Environment,
        "cpu2": env_cpu2.Environment, "lstm_transfer": env_transfer.Environment}
# Per-agent defaults = the environments of the paper's final evaluation, per the provenance checks
# .lab/tuning/deep/claude_opus_2.md and codex_astra_2.md (coordinator, 2026-10-19). md5 prefixes of the originals:
# dense/environment.py = HPA/environment.py 5ee774f0, lstm_transfer/environment.py 6f8be85e, environmentCPU.py a91f7d77.
#   derp          train dense, eval dense (weights dense/modelCPU_N.pth)
#   lstm_scratch  RS: trained on the 5ee774f env (then the file environmentCPU.py), evaluated on environmentCPU.py a91f7d77
#   dinos_t       base weights = RS weights (pretrained on 5ee774f = dense); fine-tune + eval on lstm_transfer
#   hpa_rule      HPA/environment.py (byte-identical to dense)
#   k8s_hpa, always_down: no original script; dense
TRAIN_ENV = {"derp": "dense", "lstm_scratch": "dense", "dinos_t": "dense"}
EVAL_ENV = {"derp": "dense", "lstm_scratch": "cpu", "dinos_t": "lstm_transfer", "hpa_rule": "dense", "k8s_hpa": "dense",
            "always_down": "dense"}
EVAL_SEEDS = (1000, 1001, 1002, 1003, 1004)
VAL_SEEDS = tuple(range(1100, 1105))
TEST_SEEDS = tuple(range(2000, 2020))
TEST2_SEEDS = tuple(range(3000, 3020))      # cycle 6: fresh final-test seeds (2000-2019 were inspected in c4/c5)
TEST4_SEEDS = tuple(range(5000, 5020))      # cycle 8: fresh final-test seeds (test3 4000-4019 lives in c7_oracles.py)
TEST5_SEEDS = tuple(range(6000, 6020))      # cycle 9: fresh final-test seeds for the original-env family
TEST6_SEEDS = tuple(range(7000, 7020))      # cycle 9b: fresh confirmation seeds (never used before)
TEST7_SEEDS = tuple(range(8000, 8020))      # cycle 10: fresh confirmation seeds for the delay variants
TEST8_SEEDS = tuple(range(9000, 9020))      # cycle 12: fresh confirmation seeds
TEST9_SEEDS = tuple(range(10000, 10020))    # cycle 13: fresh replication seeds
TEST10_SEEDS = tuple(range(11000, 11020))   # cycle 18 (Dinos V.2, C-20261021-02): fresh confirmation seeds for 22/10
TEST11_SEEDS = tuple(range(12000, 12020))   # cycle 22: second held-out confirmation split (test10 used once, c21)
SPLITS = {"fair": EVAL_SEEDS, "val": VAL_SEEDS, "test": TEST_SEEDS, "test2": TEST2_SEEDS, "test4": TEST4_SEEDS,
          "test5": TEST5_SEEDS, "test6": TEST6_SEEDS, "test7": TEST7_SEEDS,
          "test8": TEST8_SEEDS, "test9": TEST9_SEEDS, "test10": TEST10_SEEDS,
          "test11": TEST11_SEEDS}
VARIANTS = ("legacy", "coupled")
# cycle 4 coupled defaults (coordinator task M-20261019-009): dense for DERP, lstm_transfer for the LSTM agents,
# dense for the rule agents; dinos_t pretrains on the LEGACY dense source env
COUPLED_ENV = {"derp": "dense", "lstm_scratch": "lstm_transfer", "dinos_t": "lstm_transfer", "hpa_rule": "dense",
               "k8s_hpa": "dense", "always_down": "dense"}
COUPLED_TRAIN_ENV = {"derp": "dense", "lstm_scratch": "lstm_transfer", "dinos_t": "dense"}
SEED_RANGE = range(0, 20000)
CHUNK = 1000
EP_TICKS = 5000
N_EVAL = 5
STATE, ACTIONS, SEQ, BATCH = 6, 3, 10, 16
FINE_TUNE_TICKS = 1000
COMP = ("3L", "neg5lat", "neg10N", "neg05cpu", "neg05mem")
ORIG_LABEL = "reproduction of figure protocol"


# ---------------------------------------------------------------- seeds / telemetry
def seed_all(s):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def sandbox_seed_guard(seed):
    """Mirror of sim_v2/env.py _sandbox_seed_guard: unset = no check; else seed must be inside one listed range."""
    spec = os.environ.get("QRL_SANDBOX_SEEDS")
    if spec is None:
        return
    s = int(seed)
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        if a and int(a) <= s <= int(b or a):
            return
    raise PermissionError(f"seed {s} is outside the sandbox seed ranges QRL_SANDBOX_SEEDS={spec!r}")


def check_seeds(seed, protocol, splits=("fair",)):
    if int(seed) not in SEED_RANGE:
        raise SystemExit(f"refusing: --seed {seed} outside 0-19999")
    for sp in ("fair",) + tuple(splits):
        if int(seed) in SPLITS[sp]:
            raise SystemExit(f"refusing: --seed {seed} is in the {sp} eval block {SPLITS[sp][0]}-{SPLITS[sp][-1]}")
    sandbox_seed_guard(seed)
    if protocol in ("fair", "both"):
        for sp in splits:
            for s in SPLITS[sp]:
                sandbox_seed_guard(s)


class Telemetry:
    def __init__(self, run_dir, total):
        self.dir, self.total = run_dir, int(total)
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)

    def log(self, step, **m):
        if self.dir:
            rec = {"step": int(step)}
            rec.update({k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating, np.integer))})
            with open(os.path.join(self.dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")

    def progress(self, step):
        if self.dir:
            with open(os.path.join(self.dir, "progress.json"), "w") as f:
                json.dump({"step": int(step), "total": self.total}, f)


# ---------------------------------------------------------------- agents
def no_grad_target(agent):
    """Run the LSTM target network forward under no_grad. Numerically identical (the target only enters the loss via
    target_f.detach()); stops the original's autograd graph from growing through the stateful hidden state."""
    fwd = agent.target_model.forward

    def f(x):
        with torch.no_grad():
            return fwd(x)
    agent.target_model.forward = f
    return agent


def make_lstm_scratch():
    return no_grad_target(agent_huber.DDQNAgent(STATE, ACTIONS, SEQ))


def train_chunks(agent, env, ticks, lstm, tel=None, tick0=0, reset_on_done=False):
    """P/dense/doubledeep.py:24-82 (lstm=False) and P/doubledeepCPU.py:25-85 (lstm=True), without plots/tqdm:
    ticks/1000 chunks on one env (never reset), state zeroed at each chunk start, replay(16) every tick once the
    memory holds > 16 transitions. The original's `done` handling makes every chunk exactly 1000 ticks.
    reset_on_done (cycle 4, not in the originals): after a terminal transition (env t >= 5000) is stored, the env is
    reset and the state window zeroed (dense: the reset observation), so training never runs past t = 5000."""
    rewards = []
    for c in range(ticks // CHUNK):
        state = np.zeros((SEQ, STATE)) if lstm else np.zeros(STATE)
        total_reward = 0
        for _ in range(CHUNK):
            action = agent.act(state, verbose=0)
            new_state, reward, _, _, _, _, _, done = env.step(action)
            total_reward += reward
            if lstm:
                next_state = np.zeros((SEQ, STATE))
                next_state[:-1] = state[1:]
                next_state[-1] = new_state
            else:
                next_state = new_state
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if reset_on_done and done:
                s0 = env.reset()
                state = np.zeros((SEQ, STATE)) if lstm else s0
            if len(agent.memory) > BATCH:
                agent.replay(BATCH)
        rewards.append(float(total_reward))
        if tel:
            tel.log(tick0 + (c + 1) * CHUNK, train_chunk_return=total_reward, epsilon=agent.epsilon)
            tel.progress(tick0 + (c + 1) * CHUNK)
    return rewards


def fine_tune(agent, env, steps=FINE_TUNE_TICKS, batch_size=BATCH):
    """P/lstm_transfer/agentHuber.py:94-113 (DDQNAgentLSTM.fine_tune) without prints; returns ticks actually run."""
    state = np.zeros((agent.sequence_length, agent.state_size))
    n = 0
    for step in range(steps):
        action = agent.act(state)
        next_state, reward, _, _, _, _, _, done = env.step(action)
        n += 1
        nxt = np.zeros((agent.sequence_length, agent.state_size))
        nxt[:-1] = state[1:]
        nxt[-1] = next_state
        agent.remember(state, action, reward, nxt, done)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        state = nxt
        if done:
            break
    return n


def make_transfer(state_dict):
    """DDQNAgentLSTM(pretrained_model=...) of P/lstm_transfer/agentHuber.py:30-35 (epsilon 0.2), from a state dict."""
    a = agent_huber_transfer.DDQNAgentLSTM(STATE, ACTIONS, SEQ)
    a.online_model.load_state_dict(state_dict)
    a.target_model.load_state_dict(state_dict)
    a.epsilon = 0.2
    a.pretrained_model = "in-memory"
    return no_grad_target(a)


def greedy_fn(agent, lstm):
    """The non-random branch of agent.act (no exploration draw, so no RNG is consumed)."""
    def f(state):
        x = np.reshape(state, (1, SEQ, np.shape(state)[-1])) if lstm else np.array(state)   # width 6 (c10 delay: 7)
        agent.online_model.eval()
        with torch.no_grad():
            return int(torch.argmax(agent.online_model.forward(torch.tensor(x, dtype=torch.float32))).item())
    return f


class K8sLegacy:
    """sim_v2 K8sHPA (Kubernetes replica formula, tolerance 0.1, default behaviour) on the legacy cpu observation
    (state[2]); the legacy env only moves +-1 per tick, so the action is one step toward the desired count."""

    def __init__(self, target):
        sys.path.insert(0, os.path.join(ROOT, "sim_v2"))
        from policies import K8sHPA
        from env import OBS_NAMES
        self.ready, self.pending = OBS_NAMES.index("ready"), OBS_NAMES.index("pending")
        self.n = len(OBS_NAMES)
        self.h = K8sHPA(target, min_replicas=1, max_replicas=20)
        self.h.reset()

    def reset(self):
        self.h.reset()

    def act(self, state):
        obs = np.zeros(self.n + 1)
        obs[self.ready], obs[self.n] = state[5], state[2]
        desired = self.h.act(obs)
        cur = int(round(state[5]))
        return 2 if desired > cur else (0 if desired < cur else 1)


# ---------------------------------------------------------------- evaluation
def run_episode(env, policy, lstm, state0, max_ticks, stop_on_done, stats=False):
    """One evaluation run; returns dict with return, ticks, components and mean pods (stats: + p95 of the observed
    latency state[4])."""
    state = np.zeros((SEQ, STATE)) if lstm else state0
    tot, comp, pods, lat, k = 0.0, np.zeros(5), [], [], 0
    while True:
        action = policy(state)
        out = env.step(action)
        new_state, reward, done = out[0], out[1], out[7]
        tot += reward
        comp += np.array(out[2:7], dtype=float)
        pods.append(new_state[5])
        lat.append(new_state[4])
        k += 1
        if lstm:
            nxt = np.zeros((SEQ, STATE))
            nxt[:-1] = state[1:]
            nxt[-1] = new_state
            state = nxt
        else:
            state = new_state
        if (stop_on_done and done) or k >= max_ticks:
            break
    r = dict(ret=float(tot), ticks=k, mean_pods=float(np.mean(pods)), **{c: float(v) for c, v in zip(COMP, comp)})
    if stats:
        r["p95_latency"] = float(np.percentile(lat, 95))
        r["mean_latency"] = float(np.mean(lat))
    return r


def summarize(eps, comp_names=COMP, **extra):
    """comp_names: reward component keys of the episodes (cycle 9 original env: served, vm_cost)."""
    comp = {c: float(sum(e[c] for e in eps)) for c in comp_names}
    ticks = sum(e["ticks"] for e in eps)
    r = dict(sum5=float(sum(e["ret"] for e in eps)), episodes=eps, components_sum5=comp, ticks=ticks,
             mean_pods=float(sum(e["mean_pods"] * e["ticks"] for e in eps) / ticks), **extra)
    if eps and "p95_latency" in eps[0]:      # stats mode: 'sum5' is the sum over ALL episodes of the split
        rets = np.array([e["ret"] for e in eps])
        r.update(n_episodes=len(eps), mean_ret=float(rets.mean()), sd_ret=float(rets.std(ddof=1)) if len(eps) > 1 else 0.0,
                 components_mean={c: comp[c] / len(eps) for c in comp_names},
                 p95_latency_mean=float(np.mean([e["p95_latency"] for e in eps])))
    return r


def eval_noreset(env, policy, lstm):
    """P/dense/doubledeepfinal.py:28-77 / P/doubledeepCPUfinal.py:28-78: one env, never reset, state zeros, each run
    `while not done` capped at 5000 ticks."""
    return [run_episode(env, policy, lstm, np.zeros(STATE), EP_TICKS, True) for _ in range(N_EVAL)]


def eval_reset(env, policy, lstm, state_from_reset, seeds=None, on_reset=None, stats=False):
    """Reset before each of 5 runs x 5000 ticks (P/lstm_transfer/lstm_transfer.py:27-50, P/HPA/hpa_final.py:35-63).
    seeds: reseed numpy/random/torch before each reset (P-fair); one episode per seed."""
    eps = []
    for i in range(N_EVAL if seeds is None else len(seeds)):
        if seeds is not None:
            seed_all(seeds[i])
        if on_reset:
            on_reset()
        s0 = env.reset()
        eps.append(run_episode(env, policy, lstm, s0 if state_from_reset else None, EP_TICKS, False, stats))
    return eps


def dinos_orig_eval(agent, env, quirk):
    """P-orig for dinos_t on an agent already built from pretrained weights: fine-tune on the shared env (quirk: env
    at t=5000 -> stops after 1 tick), then 5 reset runs with the agent's own epsilon-greedy act."""
    if quirk:
        env.time = EP_TICKS
    ft = fine_tune(agent, env)
    eps_eval = float(agent.epsilon)
    eps = eval_reset(env, agent.act, True, False)
    return summarize(eps, label=ORIG_LABEL, finetune_ticks_run=ft, epsilon_eval=eps_eval, shared_env_quirk=bool(quirk))


def dinos_orig_from_weights(path, quirk, seed):
    """Evaluate original DInos base weights (P/lstm_transfer/modelCPU_N.pth) exactly as lstm_transfer.py did."""
    seed_all(seed)
    agent = make_transfer(torch.load(path))
    return dinos_orig_eval(agent, env_transfer.Environment(), quirk)


LEGACY_P = "/home/ubuntu/simulations/redisk8s/timeseries/Pytorch"
# logged 5-run sums (P/<dir>/model_N/rewards.txt summed), models 1/2/5/10/20
LOGGED = {"derp": (286672, 259386, 218389, 197408, 152188),            # dense/final_metrics_dense
          "rs": (253531, 209063, 216908, 324216, 249457),              # final_metrics
          "dinos": (3280882, 3617616, 3616592, 2822113, 3228498),      # lstm_transfer/final_metrics_lstm
          "hpa": (-1036558, -1032121, -1034987, -1031488, -1035403)}   # HPA/final_metrics_hpa
MODELS = (1, 2, 5, 10, 20)


def dinos_orig_logged_model(m, seed, root=LEGACY_P):
    """Replay the policy lstm_transfer.py actually evaluated for model m: model 1 = the saved fine-tuned weights
    (final_metrics_lstm/modelCPU_1_fine_tuned.pth) at the epsilon its 1000-tick fine-tune ends with (0.2 decayed by
    0.995 per replay until <= 0.1); models 2..20 = base weights at epsilon 0.2 (fine-tune stopped after 1 tick)."""
    if m != 1:
        return dinos_orig_from_weights(f"{root}/lstm_transfer/modelCPU_{m}.pth", True, seed)
    seed_all(seed)
    a = make_transfer(torch.load(f"{root}/lstm_transfer/final_metrics_lstm/modelCPU_1_fine_tuned.pth"))
    e = a.epsilon
    while e > a.epsilon_min:
        e *= a.epsilon_decay
    a.epsilon = e
    return summarize(eval_reset(env_transfer.Environment(), a.act, True, False), label=ORIG_LABEL, epsilon_eval=e,
                     finetune_ticks_run=0, shared_env_quirk=False)


# ---------------------------------------------------------------- one run
def parse_splits(spec):
    sp = tuple(x.strip() for x in str(spec).split(",") if x.strip())
    bad = [x for x in sp if x not in SPLITS]
    if not sp or bad or len(set(sp)) != len(sp):
        raise SystemExit(f"--eval-split must be a comma list of distinct {tuple(SPLITS)}, got {spec!r}")
    return sp


def run(agent_name, train_ticks, seed, protocol, env_name="native", hpa_target=50, tel_dir=None, train_env="native",
        env_variant="legacy", eval_splits=("fair",)):
    if agent_name not in AGENTS:
        raise SystemExit(f"unknown agent {agent_name}")
    if train_ticks not in TRAIN_TICKS:
        raise SystemExit(f"--train-ticks must be one of {TRAIN_TICKS}")
    if env_variant not in VARIANTS:
        raise SystemExit(f"--env-variant must be one of {VARIANTS}")
    eval_splits = parse_splits(",".join(eval_splits) if not isinstance(eval_splits, str) else eval_splits)
    coupled = env_variant == "coupled"
    if coupled and protocol != "fair":
        raise SystemExit("--env-variant coupled supports only --protocol fair (greedy split evaluation)")
    if protocol == "orig" and eval_splits != ("fair",):
        raise SystemExit("--eval-split applies to the fair (greedy) protocol; use --protocol fair or both")
    stats = coupled or eval_splits != ("fair",)
    check_seeds(seed, protocol, eval_splits)
    from tuning.legacy import confirm_guard as CG
    CG.authorise(eval_splits, "run_legacy.run")          # test10/test11: grant from the CLI guard required
    tenv = ((COUPLED_ENV if coupled else EVAL_ENV)[agent_name]) if env_name == "native" else env_name
    Env = coupled_env.make(ENVS[tenv]) if coupled else ENVS[tenv]
    trained = agent_name in ("derp", "lstm_scratch", "dinos_t")
    trenv = (((COUPLED_TRAIN_ENV if coupled else TRAIN_ENV)[agent_name]) if train_env == "native" else train_env) \
        if trained else None
    # dinos_t pretrains on the LEGACY source env even under coupled (the transfer is legacy source -> coupled target)
    TrainEnv = None
    if trained:
        TrainEnv = coupled_env.make(ENVS[trenv]) if coupled and agent_name != "dinos_t" else ENVS[trenv]
    # coupled dinos_t: B = pretrain + fine-tune, so pretrain B - 1000
    pre_ticks = train_ticks - FINE_TUNE_TICKS if (coupled and agent_name == "dinos_t") else train_ticks
    if pre_ticks <= 0:
        raise SystemExit("coupled dinos_t needs --train-ticks > 1000 (pretrain = B - 1000)")
    prots = (("orig",) if protocol in ("orig", "both") else ()) + (eval_splits if protocol in ("fair", "both") else ())
    n_ft = (1 if protocol in ("orig", "both") else 0) + (1 if protocol in ("fair", "both") else 0)
    total = (pre_ticks + (FINE_TUNE_TICKS * n_ft if agent_name == "dinos_t" else 0) if trained else 0) \
        + sum(N_EVAL if p == "orig" else len(SPLITS[p]) for p in prots) * EP_TICKS
    tel = Telemetry(tel_dir, total)
    res = dict(agent=agent_name, seed=int(seed), protocol=protocol, env=tenv, train_env=trenv, train_ticks=int(train_ticks) if trained else 0,
               train_ticks_label=int(train_ticks), pretrain_ticks=0, finetune_ticks=0, eval_seeds_fair=list(EVAL_SEEDS),
               torch_threads=torch.get_num_threads(), legacy_root="~/simulations/redisk8s/timeseries/Pytorch")
    if coupled or eval_splits != ("fair",):
        res.update(env_variant=env_variant, eval_splits=list(eval_splits),
                   eval_seeds={sp: list(SPLITS[sp]) for sp in eval_splits},
                   train_env_variant=("legacy" if agent_name == "dinos_t" else env_variant) if trained else None,
                   train_reset_on_done=bool(coupled))
    seed_all(seed)
    t0, c0 = time.time(), time.process_time()
    lstm = agent_name in ("lstm_scratch", "dinos_t")
    weights = None
    if agent_name == "derp":
        ag = ddqnagent.DDQNAgent(STATE, ACTIONS)
        res["train_chunk_returns"] = train_chunks(ag, TrainEnv(), train_ticks, False, tel, reset_on_done=coupled)
        weights = {k: v.clone() for k, v in ag.online_model.state_dict().items()}
        res["epsilon_end_train"] = float(ag.epsilon)
    elif agent_name == "lstm_scratch":
        ag = make_lstm_scratch()
        res["train_chunk_returns"] = train_chunks(ag, TrainEnv(), train_ticks, True, tel, reset_on_done=coupled)
        weights = {k: v.clone() for k, v in ag.online_model.state_dict().items()}
        res["epsilon_end_train"] = float(ag.epsilon)
    elif agent_name == "dinos_t":
        ag = make_lstm_scratch()
        res["pretrain_chunk_returns"] = train_chunks(ag, TrainEnv(), pre_ticks, True, tel, reset_on_done=coupled)
        res["pretrain_ticks"], res["train_ticks"] = int(pre_ticks), 0
        weights = {k: v.clone() for k, v in ag.online_model.state_dict().items()}
    res["train_cpu_s"], res["train_wall_s"] = time.process_time() - c0, time.time() - t0
    rng_after_train = (np.random.get_state(), random.getstate(), torch.get_rng_state())
    tick = pre_ticks if trained else 0

    def mk_policy(fair):
        """Policy for one protocol; returns (policy, lstm, on_reset, extra info)."""
        if agent_name == "derp":
            a = ddqnagent.DDQNAgent(STATE, ACTIONS)            # fresh agent: epsilon 1.0 (doubledeepfinal.py:159-160)
            a.online_model.load_state_dict(weights)
            return (greedy_fn(a, False) if fair else lambda s: a.act(s, verbose=0)), False, None, \
                {"epsilon_eval": 0.0 if fair else float(a.epsilon)}
        if agent_name == "lstm_scratch":
            a = make_lstm_scratch()                          # fresh agent: epsilon 1.0 (doubledeepCPUfinal.py)
            a.online_model.load_state_dict(weights)
            hid = (lambda: setattr(a.online_model, "hidden", a.online_model.init_hidden())) if fair else None
            return (greedy_fn(a, True) if fair else a.act), True, hid, {"epsilon_eval": 0.0 if fair else float(a.epsilon)}
        if agent_name == "hpa_rule":
            a = simpleagent.SimpleAgent()
            return a.act, False, None, {}
        if agent_name == "k8s_hpa":
            a = K8sLegacy(hpa_target)
            return a.act, False, a.reset, {"hpa_target": int(hpa_target)}
        if agent_name == "always_down":
            return (lambda s: 0), False, None, {}
        raise AssertionError(agent_name)

    out = {}
    tuned = None        # dinos_t: (agent, env, ticks) fine-tuned once and shared by every greedy split
    for prot in prots:
        fair = prot != "orig"
        c1 = time.process_time()
        # every protocol starts from the same post-training RNG state (so 'fair' alone == 'fair' within 'both');
        # the greedy splits reseed every episode, so the RNG state between splits does not matter
        if prot in ("orig", eval_splits[0]):
            np.random.set_state(rng_after_train[0]); random.setstate(rng_after_train[1]); torch.set_rng_state(rng_after_train[2])
        if agent_name == "dinos_t":
            if fair:
                first = tuned is None
                if first:
                    agent = make_transfer(weights)
                    env = Env()
                    tuned = (agent, env, fine_tune(agent, env))
                agent, env, ft = tuned
                pol = greedy_fn(agent, True)
                hid = lambda: setattr(agent.online_model, "hidden", agent.online_model.init_hidden())  # noqa: E731
                r = summarize(eval_reset(env, pol, True, False, SPLITS[prot], hid, stats), label=prot,
                              finetune_ticks_run=ft if first else 0, finetune_shared=not first, epsilon_eval=0.0) \
                    if stats else \
                    summarize(eval_reset(env, pol, True, False, EVAL_SEEDS, hid), label="fair", finetune_ticks_run=ft,
                              epsilon_eval=0.0)
                if stats:
                    r["finetune_ticks"] = ft
            else:
                agent = make_transfer(weights)
                env = Env()
                r = dinos_orig_eval(agent, env, quirk=train_ticks != FINE_TUNE_TICKS)
            res["finetune_ticks"] = max(res["finetune_ticks"], r["finetune_ticks_run"])
        else:
            pol, is_lstm, on_reset, extra = mk_policy(fair)
            env = Env()
            if fair and stats:
                r = summarize(eval_reset(env, pol, is_lstm, not is_lstm, SPLITS[prot], on_reset, stats), label=prot, **extra)
            elif fair:
                r = summarize(eval_reset(env, pol, is_lstm, not is_lstm, EVAL_SEEDS, on_reset), label="fair", **extra)
            elif agent_name in ("derp", "lstm_scratch"):
                r = summarize(eval_noreset(env, pol, is_lstm), label=ORIG_LABEL, **extra)
            else:
                if on_reset:
                    on_reset()
                r = summarize(eval_reset(env, pol, False, True), label=ORIG_LABEL, **extra)
        r["cpu_s"] = time.process_time() - c1
        tick += r["ticks"] + r.get("finetune_ticks_run", 0)
        if stats and fair:
            tel.log(tick, **{f"{prot}_mean_ret": r["mean_ret"], f"{prot}_mean_pods": r["mean_pods"],
                             f"{prot}_p95_latency": r["p95_latency_mean"]})
        else:
            tel.log(tick, **{f"{prot}_sum5": r["sum5"], f"{prot}_mean_pods": r["mean_pods"]})
        tel.progress(min(tick, total))
        out[prot] = r
    res["protocols"] = out
    res["cpu_s"], res["wall_s"] = time.process_time() - c0, time.time() - t0
    tel.progress(total)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--agent", required=True, choices=AGENTS)
    p.add_argument("--train-ticks", type=int, default=None, choices=TRAIN_TICKS,
                   help="required except under the cycle-5 protocol (see the c5 flags below)")
    p.add_argument("--seed", type=int, default=int(os.environ.get("QRL_RUN_SEED", 0)))
    p.add_argument("--protocol", choices=("orig", "fair", "both"), default=None,
                   help="default: both (legacy), fair (coupled)")
    p.add_argument("--env", choices=("native",) + tuple(ENVS), default="native",
                   help="evaluation env (dinos_t: fine-tune + eval target env); native = EVAL_ENV default")
    p.add_argument("--train-env", choices=("native",) + tuple(ENVS), default="native",
                   help="training env (dinos_t: pretraining/source env); native = TRAIN_ENV default")
    p.add_argument("--env-variant", choices=VARIANTS, default="legacy",
                   help="legacy = verbatim envs; coupled = cycle-4 load-sharing wrapper (tuning/legacy/coupled_env.py)")
    p.add_argument("--eval-split", default="fair",
                   help="comma list of greedy eval splits: fair (1000-1004), val (1100-1104), test (2000-2019), "
                        "test2 (3000-3019), test4 (5000-5019), test5 (6000-6019), test6 (7000-7019), test7 (8000-8019), "
                        "test8 (9000-9019), test9 (10000-10019), test10 (11000-11019, used: closed), "
                        "test11 (12000-12019); test10/test11 need a confirmation manifest (confirm_guard.py)")
    p.add_argument("--hpa-target", type=int, default=50, help="k8s_hpa target utilisation %% (not tuned)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 1)))
    p.add_argument("--out", default=None)
    c5 = p.add_argument_group("cycle 5 (coupled only; tuning/legacy/c5_protocol.py, any of these selects it)")
    c5.add_argument("--source-variant", choices=("legacy", "coupled", "none"), default=None,
                    help="dinos_t source: dense legacy / dense coupled / none = random-init transfer agent")
    c5.add_argument("--pretrain-ticks", type=int, default=None, help="dinos_t source ticks")
    c5.add_argument("--finetune-ticks", type=int, default=None, help="dinos_t target ticks")
    c5.add_argument("--total-ticks", type=int, default=None, help="derp/lstm_scratch target ticks (free int)")
    c5.add_argument("--checkpoints", default=None, help="comma list of target-tick counts to evaluate at")
    c5.add_argument("--target-env", choices=tuple(ENVS), default=None, help="common target env (default lstm_transfer)")
    c6 = p.add_argument_group("cycle 6 (c5 protocol; lstm_scratch / dinos_t only)")
    c6.add_argument("--agent-variant", choices=("verbatim", "fixed", "dense_fixed") + tuple(agent_lstm_v2.ARCHS),
                    default="verbatim",
                    help="fixed = tuning/legacy/agent_lstm_fixed.py (hidden reset per window, detached Double-DQN "
                         "target, periodic target sync); dense_fixed (lstm_scratch, --env-family orig) = the same "
                         "agent with an MLP over the flattened 10-step window (agent_dense_fixed.py, arm mlp_hist10); "
                         "cycle 18 (--env-family orig, agent_lstm_v2.ARCHS): a1/a1h/a2/a3/a4 = Dinos V.2 architecture "
                         "search, mlp_* = parameter-matched MLPs; --agent lstm_scratch -> arms dinos_<arch> / mlp_*; "
                         "--agent dinos_t + --pretrain-env-family -> transfer_<arch>")
    c6.add_argument("--target-sync", type=int, default=None, help="fixed only: target sync period in replays (250)")
    c6.add_argument("--diag", type=int, default=None, help="log diagnostics every N replays to <out>.diag.json")
    c8 = p.add_argument_group("cycle 8 (c5 protocol, all arms; defaults = cycles 5-7)")
    c8.add_argument("--reward-div", type=float, default=1.0, help="learning targets use r/D (reported returns unscaled)")
    c8.add_argument("--obs-scale", choices=("off", "on"), default="off",
                    help="on: agent observations / [70, 60, 100, 100, 1.2, 6]")
    c8.add_argument("--eps-schedule", choices=("original", "linear10k"), default="original",
                    help="linear10k: epsilon linear from the agent's initial to its floor over 10k training ticks")
    c9 = p.add_argument_group("cycle 9 (c5 protocol, derp / lstm_scratch; tuning/legacy/orig_variants.py)")
    c9.add_argument("--env-family", choices=("coupled", "orig"), default="coupled",
                    help="orig = the original P/environment.py (min(capacity, load) - 3 VMs) for training and eval")
    c9.add_argument("--orig-variant", default=None,
                    help="orig family: orig | noise5 | noise15 | multispike | spike300 | noise15_multispike (default orig); "
                         "cycle 10: append _d<D> for a pod startup delay of D ticks (e.g. orig_d5, spike300_d10)")
    c9b = p.add_argument_group("cycle 9b (--env-family orig, dinos_t only: transfer onto the orig family)")
    c9b.add_argument("--pretrain-env-family", choices=("orig", "coupled"), default=None,
                     help="source family of the pretraining (with --pretrain-ticks; fine-tune = --finetune-ticks)")
    c9b.add_argument("--pretrain-variant", default=None,
                     help="orig source: an --orig-variant name; coupled source: legacy | coupled (dense source env)")
    c9b.add_argument("--finetune-eps", choices=("restart", "low"), default="restart",
                     help="restart: fine-tune epsilon schedule restarts at 1.0; low: starts at 0.2")
    c9b.add_argument("--derp-history", type=int, default=1,
                     help="derp on --env-family orig: dense input = flattened window of the last H observations")
    obsv = p.add_argument_group("dashboard observers (opt-in, cycle-5 protocol only; numbers unchanged)")
    obsv.add_argument("--live-file", default=None, help="write a live ring-buffer JSON here (every 25 ticks)")
    obsv.add_argument("--record-traj", type=int, default=0,
                      help="store per-tick trajectories of N greedy eval episodes (every checkpoint, last split)")
    c18 = p.add_argument_group("cycle 18 (c5 protocol, orig family; researcher checkpoint protocol, opt-in)")
    c18.add_argument("--save-weights", choices=("off", "on"), default="off",
                     help="on: save every target checkpoint's online state_dict (<out>.ck<T>.pt) and the val-selected one "
                          "(<out>.selected.pt)")
    c18.add_argument("--source-checkpoints", default=None,
                     help="orig transfer: evaluate the source on its val split at these pretrain-tick counts (largest = "
                          "--pretrain-ticks) and fine-tune from the val-best one, e.g. 1000,2000,5000,10000,20000")
    c18.add_argument("--confirm-manifest", default=None,
                     help="test10/test11 only: the confirmation manifest frozen for that split by make_c18.py --confirm (same as env "
                          "QRL_CONFIRM_MANIFEST; tuning/legacy/confirm_guard.py)")
    c19 = p.add_argument_group("cycle 19 (c5 protocol, fixed-agent family: Double-DQN add-ons, off by default)")
    c19.add_argument("--n-step", type=int, default=1, help="n-step returns (bootstrap gamma^n; 1 = off)")
    c19.add_argument("--dueling", choices=("off", "on"), default="off", help="dueling Q head (V + A - mean A)")
    c19.add_argument("--per", choices=("off", "on"), default="off",
                     help="proportional prioritized replay (alpha 0.6, beta 0.4 -> 1 over the training ticks)")
    c23 = p.add_argument_group("cycle 23 (c5 protocol, fixed-agent family, pretrain and target phases; off by default)")
    c23.add_argument("--lr", type=float, default=None, help="Adam learning rate override (default: the arm's own)")
    c23.add_argument("--grad-clip", type=float, default=None,
                     help="global grad-norm clip after backward, before each optimizer step (pre-clip norms logged)")
    c23.add_argument("--pending-maturity", choices=("off", "on"), default="off",
                     help="orig family, _d<D> variants: + 5 features = pending pods ready in 1..5 ticks")
    a = p.parse_args(argv)
    a.argv_ = list(sys.argv[1:] if argv is None else argv)
    if (a.save_weights == "on" or a.source_checkpoints) and a.env_family != "orig":
        p.error("--save-weights/--source-checkpoints need --env-family orig")
    if a.env_family == "orig":
        return main_c5(p, a)
    c19_on = a.n_step != 1 or a.dueling != "off" or a.per != "off"
    if a.orig_variant is not None:
        p.error("--orig-variant needs --env-family orig")
    if a.pretrain_env_family is not None or a.pretrain_variant is not None or a.finetune_eps != "restart" \
            or a.derp_history != 1:
        p.error("--pretrain-env-family/--pretrain-variant/--finetune-eps/--derp-history need --env-family orig")
    if any(v is not None for v in (a.source_variant, a.pretrain_ticks, a.finetune_ticks, a.total_ticks,
                                   a.checkpoints, a.target_env)):
        return main_c5(p, a)
    if c19_on:
        p.error("--n-step/--dueling/--per need the cycle-5 protocol flags")
    if a.lr is not None or a.grad_clip is not None or a.pending_maturity != "off":
        p.error("--lr/--grad-clip/--pending-maturity need the cycle-5 protocol flags")
    if a.agent_variant != "verbatim" or a.target_sync is not None or a.diag is not None:
        p.error("--agent-variant/--target-sync/--diag need the cycle-5 protocol flags")
    if a.live_file is not None or a.record_traj:
        p.error("--live-file/--record-traj need the cycle-5 protocol flags")
    if a.reward_div != 1.0 or a.obs_scale != "off" or a.eps_schedule != "original":
        p.error("--reward-div/--obs-scale/--eps-schedule need the cycle-5 protocol flags")
    if a.train_ticks is None:
        p.error("--train-ticks is required")
    if a.protocol is None:
        a.protocol = "fair" if a.env_variant == "coupled" else "both"
    confirm_guard(a)
    torch.set_num_threads(a.threads)
    import contextlib
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):   # cpu2/dense2 print every step (verbatim)
        res = run(a.agent, a.train_ticks, a.seed, a.protocol, a.env, a.hpa_target, os.environ.get("QRL_RUN_DIR"),
                  a.train_env, a.env_variant, a.eval_split)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps({"agent": res["agent"], "train_ticks": a.train_ticks, "seed": a.seed, "cpu_s": round(res["cpu_s"], 1),
                      **{k: round(v.get("mean_ret", v["sum5"])) for k, v in res["protocols"].items()}}))
    return res


def confirm_guard(a, split=None):
    """test10/test11 need a frozen, unused confirmation manifest of that split (tuning/legacy/confirm_guard.py);
    no-op otherwise."""
    from tuning.legacy import confirm_guard as CG
    sp = tuple(x.strip() for x in str(split or a.eval_split).split(","))
    return CG.require(sp, "run_legacy.py", getattr(a, "argv_", []), getattr(a, "confirm_manifest", None), a.seed)


def main_c5(p, a):
    """Cycle-5 protocol entry (tuning/legacy/c5_protocol.py). Coupled (or cycle-9 orig family); --protocol fair only."""
    orig = getattr(a, "env_family", "coupled") == "orig"
    if orig and (a.env_variant != "legacy" or a.target_env is not None or a.train_env != "native"):
        p.error("--env-family orig selects its own env: no --env-variant/--target-env/--train-env")
    if not orig and a.env_variant != "coupled":
        p.error("the cycle-5 flags need --env-variant coupled")
    if a.train_ticks is not None or a.env != "native":
        p.error("cycle 5 uses --total-ticks / --pretrain-ticks + --finetune-ticks and --target-env, not --train-ticks/--env")
    if a.protocol not in (None, "fair"):
        p.error("cycle 5 supports only greedy evaluation (--protocol fair)")
    split = a.eval_split if a.eval_split != "fair" else "val,test"
    confirm_guard(a, split)
    from tuning.legacy import c5_protocol as C5
    cks = [int(x) for x in a.checkpoints.split(",") if x.strip()] if a.checkpoints else None
    torch.set_num_threads(a.threads)
    import contextlib
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        res = C5.run_c5(a.agent, a.seed, a.source_variant, a.pretrain_ticks, a.finetune_ticks, a.total_ticks, cks,
                        a.target_env or C5.DEFAULT_TARGET_ENV,
                        C5.DEFAULT_SOURCE_ENV if a.train_env == "native" else a.train_env, split,
                        os.environ.get("QRL_RUN_DIR"), a.out, agent_variant=a.agent_variant,
                        target_sync=a.target_sync, diag=a.diag, reward_div=a.reward_div, obs_scale=a.obs_scale,
                        eps_schedule=a.eps_schedule,
                        **({"live_file": a.live_file} if a.live_file else {}),
                        **({"record_traj": a.record_traj} if a.record_traj else {}),
                        **({"save_weights": True} if getattr(a, "save_weights", "off") == "on" else {}),
                        **({"source_checkpoints": [int(x) for x in a.source_checkpoints.split(",") if x.strip()]}
                           if getattr(a, "source_checkpoints", None) else {}),
                        **({"ddqn": dict(n_step=a.n_step, dueling=a.dueling == "on", per=a.per == "on")}
                           if (getattr(a, "n_step", 1) != 1 or getattr(a, "dueling", "off") != "off"
                               or getattr(a, "per", "off") != "off") else {}),
                        **({"lr": a.lr} if getattr(a, "lr", None) is not None else {}),
                        **({"grad_clip": a.grad_clip} if getattr(a, "grad_clip", None) is not None else {}),
                        **({"pending_maturity": True} if getattr(a, "pending_maturity", "off") == "on" else {}),
                        **(dict(env_family="orig", orig_variant=a.orig_variant or "orig",
                                pretrain_env_family=a.pretrain_env_family, pretrain_variant=a.pretrain_variant,
                                finetune_eps=a.finetune_eps, derp_history=a.derp_history) if orig else {}))
    print(json.dumps({"arm": res["arm"], "seed": a.seed, "cpu_s": round(res["cpu_s"], 1),
                      "train_cpu_s_per_1k": round(res["train_cpu_s_per_1k_ticks"], 1),
                      **{f"ck{t}_{sp}": round(r[sp]["mean_ret"]) for t, r in res["checkpoint_results"].items()
                         for sp in res["eval_splits"]}}))
    return res


if __name__ == "__main__":
    main()
