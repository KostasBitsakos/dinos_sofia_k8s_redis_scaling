"""Cycle 5 protocol (Track L, coupled path only; design .lab/tuning/cycles/c4_codex.md + c4_synthesis.md).

Cycle 6 additions (c5_codex.md + c5_synthesis.md): --agent-variant verbatim|fixed for lstm_scratch and dinos_t (fixed =
agent_lstm_fixed.py, also used for the dinos_t source pretraining; --target-sync, default 250 replays), --diag N
(c6_diag.py observers, <out>.diag.json) and the test2 split (3000-3019). The verbatim path is unchanged except four
metadata keys (agent_variant, target_sync_replays, double_dqn_target, hidden_reset_per_window) on the LSTM arms.

Reached through run_legacy.py when any cycle-5 flag is given (--source-variant, --pretrain-ticks, --finetune-ticks,
--total-ticks, --checkpoints, --target-env); requires --env-variant coupled. The legacy path of run_legacy.run() is
not touched. The verbatim ported files (env_*.py, agents, networks) are wrapped, never edited.

Arms (all on ONE common target: the coupled lstm_transfer env by default, --target-env):
  derp          dense DDQN (ddqnagent.py, its own epsilon 1 -> 0.01), --total-ticks target ticks
  lstm_scratch  agentHuber LSTM (epsilon 1 -> 0.05, lr 1e-4), --total-ticks target ticks
  dinos_t       --source-variant legacy|coupled: agentHuber pretrained --pretrain-ticks on the dense source env
                (legacy = verbatim, coupled = coupled_env wrapper; same sinusoid/noise/spike input), its online weights
                loaded into DDQNAgentLSTM (epsilon 0.2 -> 0.1, lr 5e-5, fresh replay memory and Adam), fine-tuned
                --finetune-ticks on the target. --source-variant none: the same transfer agent from random weights
                (fresh myLSTM init), same optimiser/epsilon, no pretraining (control for negative transfer).
Epsilon decays x0.995 per replay as in the originals (replay(16) every tick once the memory holds > 16).

Training (source and target alike): episodes of 5000 ticks; on done the env is reset and the state window restarts
from the reset observation (LSTM: zeros with the reset observation as last row; dense: the reset observation). The
first state of every phase is env.reset() as well, so every arm starts from the same initial observation. The LSTM
hidden state is NOT reset during training (as in the originals; known persistent-hidden replay defect).

RNG streams: each env object draws from a private numpy MT19937 stream (StreamEnv swaps the global numpy state around
step()/reset(); the verbatim envs use np.random.*). Target training stream = RandomState([seed, 5, 1]), source stream =
RandomState([seed, 5, 2]); the agent (epsilon draws, replay sampling, init) uses the global numpy/random/torch streams
seeded by --seed. The lstm_transfer env draws a fixed number of values per tick independent of the action, so for a
given seed every arm sees the identical exogenous target trace. Evaluation episode with eval seed s uses a fresh env
on RandomState(s), i.e. exactly the trace of np.random.seed(s) used by coupled_baselines.py.

Checkpoints (--checkpoints, target-tick counts): at each, a deep copy of the online network is evaluated greedily
(no exploration draw, hidden state zeroed per episode, reset observation in the window) on every --eval-split
(default val,test), 5000 ticks per episode; global RNG states are saved and restored, so evaluation does not change
training. One JSON per checkpoint (<out>.ck<T>.json) is written as soon as it is done, plus the full --out JSON.

Cycle 8 conditioning options (c7_codex.md + c7_synthesis.md; all arms, both phases of dinos_t; defaults = cycles 5-7
exactly, recorded as reward_div / obs_scale / eps_schedule in every JSON):
  --reward-div D      the agent stores r/D in replay memory, so its learning target is r/D + gamma*Q'(s') and Q lives
                      in scaled units from initialisation (incl. source pretraining). Reported returns stay unscaled.
                      (D = 1: the reward is stored unchanged, no division.)
  --obs-scale on      every observation the agent sees (training, evaluation, diag probes) is divided elementwise by
                      OBS_SCALE = [70, 60, 100, 100, 1.2, 6] for the env layout [load, capacity, cpu %, mem %, latency,
                      pods] (env_transfer.py / env_dense.py next_state): mean load 50+20, reference capacity 10*6 pods,
                      percent bound, latency amplitude, initial pods. No fitting, no clipping. Reported pods/latency are raw.
  --eps-schedule linear10k  epsilon = e0 + (floor - e0) * min(1, t / 10000) before the action of training tick t+1
                      (t = ticks done in the phase), e0/floor = the agent's own initial/minimum epsilon; the agent's
                      x0.995 per-replay decay is disabled (epsilon_decay = 1). original = x0.995 per replay.
Splits: test4 = 5000-5019 (run_legacy.SPLITS, cycle 8 final test).

Cycle 9 (.lab/tuning/cycles/c9_design.md; defaults = cycle 8 exactly): --env-family orig --orig-variant V trains and
evaluates derp / lstm_scratch on the original P/environment.py through orig_variants.py (target env name "orig:V";
private RandomState per env object instead of StreamEnv, same stream seeds; step() gym-like, reward components
served = min(capacity, load) and vm_cost = -3 VMs). --obs-scale on divides by orig_variants.OBS_SCALE
[100, 100, 60, 70, 2.4, 20] there. Eval episodes additionally report pods behaviour (frac_const_pods = fraction of
ticks whose pods equal the previous tick's, pods_changes). Split test5 = 6000-6019. No --diag.

Cycle 9b (council c9b; defaults = cycle 9 exactly, the coupled path is untouched):
  dinos_t on --env-family orig (--agent-variant fixed only): pretrain_env_family orig|coupled, pretrain_variant
      (orig: an orig_variants name, source env "orig:V"; coupled: legacy|coupled = the dense source env of the
      cycle-5 path). Pretraining: FX.make_scratch, pretrain_ticks on the source stream RandomState([seed, 5, 2]) (the
      eval traces are RandomState(s) for integer split seeds, so disjoint), same reward_div / eps_schedule, obs_scale
      with the SOURCE family divisors. Fine-tune: a fresh FX.make_scratch agent (same architecture AND the scratch
      hyperparameters lr 1e-4 / floor 0.05, so transfer - scratch isolates the initial weights) whose online and
      target nets are loaded with the pretrained weights; fresh replay memory and Adam; finetune_ticks on the target
      stream with the same conditioning flags. finetune_eps: restart = epsilon 1.0 and the scratch schedule;
      low = epsilon 0.2, then eps_schedule (linear10k: linear 0.2 -> floor over 10k ticks; original: x0.995/replay).
      Checkpoints are target (fine-tune) ticks; pretrain_ticks / finetune_ticks reported separately. Arm name
      transfer_<family>_<variant>.
  derp_history H (derp, orig family only; default 1 = unchanged): the dense DDQN sees the flattened window of the
      last H observations (H*6 inputs, zero-padded after a reset exactly like the LSTM window). Arm derp_hist<H>.
  n_params: trainable parameters of the online network, recorded on every orig-family run.

Cycle 10 (pod startup delay; defaults = cycle 9b exactly): --orig-variant / --pretrain-variant '<variant>_d<D>'
(orig_variants.py) selects the delayed env: the observation gains a 7th feature (pending pods; obs-scale divisor 20),
so every network of the run takes 7 inputs per frame (derp_history H: 7H). Transfer needs source and target of the
same state size (e.g. orig_d0 -> spike300_d5); the LSTM arms need --agent-variant fixed (the verbatim agents hard-code
6 inputs). Recorded: startup_delay, state_size, obs features. Window/history code takes the frame width from the
observation itself, so the 6-feature runs are bit-identical to before.

Cycle 10b (matched-mechanics dense control, c10_codex_review.md point 2; defaults = cycle 10 exactly):
--agent lstm_scratch --agent-variant dense_fixed (--env-family orig only) = agent_dense_fixed.DenseFixedAgent: the
fixed LSTM agent with every hyperparameter and training mechanic inherited (gamma .98, replay 5000, Double DQN,
--target-sync, lr 1e-4, epsilon 1 -> 0.05, same conditioning flags), fed the identical 10 x F window, but its
Q-network is an MLP over the flattened window (MyDenseNet, hidden width 540; parameter count within 20% of the
LSTM's). Arm name mlp_hist10; recorded network / dense_hidden / n_params.

Cycle 18 (Dinos V.2 architecture search, council C-20261021-02 + researcher R-20261021-004; defaults = cycle 10b
exactly): --agent-variant <arch> (--env-family orig only; agent_lstm_v2.ARCHS: a1, a1h, a2, a3, a4 = spec-string
sequence networks; mlp_a1, mlp_a1h, mlp_a234, mlp_a2, mlp_a34 = parameter-matched dense controls with the
agent_dense_fixed mechanics). --agent lstm_scratch: arms dinos_<arch> (sequence) / <arch> (mlp_*). --agent dinos_t with
--pretrain-env-family: the cycle-9b orig transfer protocol unchanged (source pretraining with a scratch agent of the
SAME arch, a fresh scratch agent loads the weights, --finetune-eps), arms transfer_<arch>. Recorded on these arms only:
network, net_spec, architecture, variant_label, matched_to, n_params, hyperparameters.
Checkpoint protocol of the researcher (P/doubledeepCPU.py saves modelCPU_N.pth per N x 1000 training ticks;
P/lstm_transfer/lstm_transfer.py:124-146 fine-tunes and evaluates each of N = 1, 2, 5, 10, 20), opt-in, any arm:
  --checkpoints 1000,2000,5000,10000,20000: one run with greedy deep-copy evaluation at each count is identical to
      separate runs of that length (same seed, evaluation restores every RNG state, schedules depend only on t).
  save_weights (--save-weights): the online state_dict of every target checkpoint -> <out>.ck<T>.pt, and the
      checkpoint with the largest mean VAL return of THIS run (ties: earlier) -> <out>.selected.pt; recorded under
      "saved_weights" (selected_checkpoint, its val mean, selection_scope "per_seed"). Test splits never enter any
      selection. The PER-SEED .selected.pt is a diagnostic: the summary (sum_c10.py) locks ONE checkpoint per arm
      across seeds (mean val over seeds) and `sum_c10.py --export-locked` writes that checkpoint for every seed as
      <out>.locked.pt + <out>.locked.json; Transfer V.2 and the final confirmation use the locked rule.
  source_checkpoints (--source-checkpoints, orig transfer only): during source pretraining (--pretrain-ticks = the
      largest one) the source online network is evaluated greedily at each count on the SOURCE env's VAL split
      (val seeds, same obs scaling), its state_dict kept; the checkpoint with the largest source VAL mean return
      (ties: earlier) initialises the fine-tune agent. Recorded under "source_selection" (val by checkpoint, selected,
      its val); with --save-weights also <out>.src.ck<T>.pt. Without the flag the final source weights are used (as
      before).

Dashboard observers (M-20261021-008, opt-in, default off; tuning/legacy/live_rec.py): live_file (--live-file) keeps a
ring buffer of the last ticks (load, pods, pending, action, reward; training and evaluation) in a small JSON rewritten
atomically every 25 ticks; record_traj N (--record-traj) stores per-tick lists of the first N greedy eval episodes of
every checkpoint's last split under checkpoint_results[<T>][<split>]["trajectory"] (the loop selects the checkpoint
by validation afterwards, so each one is kept). They only read values the
protocol already computed: no RNG draw, no extra step, same order; with both off the run is exactly as before.

Cycle 23 (council C-20261023-01; opt-in, defaults = cycle 20 exactly; fixed-agent family only, BOTH the pretrain and
the target phase; recorded only when on):
  lr (--lr X)             Adam learning rate of every fixed-family agent of the run (source and fine-tune). Without it
                          the lr is unchanged: 1e-4 for every fixed-family agent on the orig family (scratch, source
                          pretraining and the orig transfer fine-tune all use make_scratch), coupled dinos_t fine-tune
                          (FX.make_transfer) 5e-5. Key optim_options.
  grad_clip (--grad-clip C) global grad-norm clip after backward, before every optimizer step (FixedLSTMAgent.set_optim);
                          pre-clip norm statistics and clip frequency per phase under grad_norm; telemetry
                          <phase>_grad_norm_mean / <phase>_clip_frac every 1000 ticks.
  pending_maturity (--pending-maturity on, orig family, '_d<D>' target and source): 5 more features per frame
                          (orig_variants.FEATURES_MATURITY: pending pods ready in 1..5 ticks; 0/1, obs-scale divisor 1;
                          zeros at D = 0) -> F = 12 for every network of the run, source and target alike. Matched MLPs:
                          mlp_hist10 hidden agent_dense_fixed.DENSE_HIDDEN_PM (522); V.2 MLP controls must be the
                          F = 12 matches of agent_lstm_v2.MATCH_PM (mlp_a1h_pm = m271), the F = 7 ones are refused.
                          Key pending_maturity.
"""
import copy
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from tuning.legacy import agent_dense_fixed as DFX
from tuning.legacy import agent_lstm_v2 as V2
from tuning.legacy import agent_lstm_fixed as FX
from tuning.legacy import c6_diag
from tuning.legacy import coupled_env, ddqnagent, networks_lstm
from tuning.legacy import live_rec
from tuning.legacy import orig_variants as OV
from tuning.legacy import run_legacy as RL

SOURCE_VARIANTS = ("legacy", "coupled", "none")
C5_AGENTS = ("derp", "lstm_scratch", "dinos_t")
V2_VARIANTS = tuple(V2.ARCHS)                # cycle 18: a1, a1h, a2, a3, a4, mlp_a1, mlp_a1h, mlp_a234, ...
AGENT_VARIANTS = ("verbatim", "fixed", "dense_fixed") + V2_VARIANTS
FIXED_VARIANTS = ("fixed", "dense_fixed") + V2_VARIANTS    # fixed-agent mechanics (dense_fixed: cycle 10b MLP network)


def v2_arm(agent, variant):
    """cycle 18 arm names: lstm_scratch -> dinos_<arch> (sequence archs) / <arch> (mlp_*); dinos_t -> transfer_<arch>"""
    if agent == "dinos_t":
        return f"transfer_{variant}"
    return variant if V2.is_mlp(variant) else f"dinos_{variant}"


def best_ck(val_by_ck):
    """largest mean val return; ties: the earlier checkpoint (deterministic)."""
    ks = sorted(val_by_ck)
    return max(ks, key=lambda c: (val_by_ck[c], -c))


def pt_path(out, tag):
    root, _ = os.path.splitext(out)
    return f"{root}.{tag}.pt"
CYCLE_TAG = 5
TARGET_STREAM, SOURCE_STREAM = 1, 2
DEFAULT_TARGET_ENV = "lstm_transfer"
DEFAULT_SOURCE_ENV = "dense"
LOG_EVERY = 1000
# cycle 8
OBS_FEATURES = ("load", "capacity", "cpu", "mem", "latency", "pods")
OBS_SCALE = np.array([70.0, 60.0, 100.0, 100.0, 1.2, 6.0])
OBS_SCALES = ("off", "on")
EPS_SCHEDULES = ("original", "linear10k")
LINEAR_EPS_TICKS = 10000
# cycle 9
ENV_FAMILIES = ("coupled", "orig")
ORIG_PREFIX = "orig:"
# cycle 9b
PRETRAIN_FAMILIES = ("orig", "coupled")
FINETUNE_EPS = ("restart", "low")
LOW_EPS = 0.2
# cycle 23
PM_SUFFIX = "+pm"           # internal env-name suffix: orig adapter with the pending-maturity features


def obs_divisors(family="coupled", variant=None, pm=False):
    if family != "orig":
        return OBS_SCALE
    if pm:
        return OV.obs_scale(variant, True)
    return OV.OBS_SCALE if variant is None else OV.obs_scale(variant)


def obs_features(family="coupled", variant=None, pm=False):
    if pm and family == "orig":
        return OV.features(variant, True)
    return OBS_FEATURES if family != "orig" or variant is None else OV.features(variant)


def obs_fn(obs_scale, family="coupled", variant=None, pm=False):
    """None (identity, the observation array is passed on untouched) or elementwise division by OBS_SCALE
    (cycle 9 orig family: orig_variants.OBS_SCALE)."""
    if obs_scale == "off":
        return None
    if obs_scale == "on":
        d = obs_divisors(family, variant, pm)
        return lambda s: np.asarray(s, dtype=float) / d
    raise SystemExit(f"--obs-scale must be one of {OBS_SCALES}")


def linear_eps(e0, floor, n=LINEAR_EPS_TICKS):
    """epsilon after t training ticks of the phase: linear from e0 to floor over n ticks, then floor."""
    e0, floor = float(e0), float(floor)
    return lambda t: floor if t >= n else e0 + (floor - e0) * (t / float(n))


def eps_fn(agent, eps_schedule):
    """original: None (agent's own x0.995 per replay). linear10k: disable the agent's decay, return the schedule."""
    if eps_schedule == "original":
        return None
    if eps_schedule != "linear10k":
        raise SystemExit(f"--eps-schedule must be one of {EPS_SCHEDULES}")
    f = linear_eps(agent.epsilon, agent.epsilon_min, LINEAR_EPS_TICKS)
    agent.epsilon_decay = 1.0
    return f


def _sc(obs, s):
    return s if obs is None else obs(s)


def stream_seed(seed, which):
    return [int(seed), CYCLE_TAG, int(which)]


class StreamEnv:
    """Give one env object its own numpy stream: the global numpy state is swapped in/out around step()/reset().
    Other attributes are delegated to the wrapped env."""

    def __init__(self, env, seed):
        self.__dict__["env"] = env
        self.__dict__["rng"] = np.random.RandomState(seed).get_state()

    def __getattr__(self, name):
        return getattr(self.__dict__["env"], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["env"], name, value)

    def _call(self, f, *a):
        outer = np.random.get_state()
        np.random.set_state(self.__dict__["rng"])
        try:
            return f(*a)
        finally:
            self.__dict__["rng"] = np.random.get_state()
            np.random.set_state(outer)

    def step(self, action):
        return self._call(self.__dict__["env"].step, action)

    def reset(self):
        return self._call(self.__dict__["env"].reset)


def make_env(name, variant, seed):
    if name.startswith(ORIG_PREFIX):      # cycle 9: "orig:<variant>", private RandomState(seed) inside the adapter
        v = name[len(ORIG_PREFIX):]
        if v.endswith(PM_SUFFIX):         # cycle 23: + pending-maturity features
            return OV.make(v[:-len(PM_SUFFIX)], seed, True)
        return OV.make(v, seed)
    base = RL.ENVS[name]
    return StreamEnv(coupled_env.make(base)() if variant == "coupled" else base(), seed)


def env_step(env, action):
    """-> (next state, reward, reward components, done) for both env families."""
    if isinstance(env, OV.OrigEnv):
        st, r, done, info = env.step(action)
        return st, r, tuple(info[c] for c in OV.COMP), done
    out = env.step(action)
    return out[0], out[1], out[2:7], out[7]


def comp_names(env):
    return OV.COMP if isinstance(env, OV.OrigEnv) else RL.COMP


def window0(s0, lstm, hist=1):
    """First state after a reset: LSTM window of zeros with the reset observation as its last row; dense: s0
    (cycle 9b, dense hist > 1: the flattened zero window of hist rows with s0 as the last row). The frame width is
    len(s0) (6; cycle-10 delay variants 7)."""
    k = len(s0)
    if not lstm:
        if hist > 1:
            w = np.zeros(hist * k)
            w[-k:] = s0
            return w
        return np.array(s0, dtype=float)
    w = np.zeros((RL.SEQ, k))
    w[-1] = s0
    return w


def push(state, new_state, lstm, hist=1):
    if not lstm:
        if hist > 1:
            k = len(new_state)
            nxt = np.empty(hist * k)
            nxt[:-k] = state[k:]
            nxt[-k:] = new_state
            return nxt
        return new_state
    nxt = np.zeros((RL.SEQ, np.shape(state)[1]))
    nxt[:-1] = state[1:]
    nxt[-1] = new_state
    return nxt


def rng_get():
    return np.random.get_state(), random.getstate(), torch.get_rng_state()


def rng_set(st):
    np.random.set_state(st[0])
    random.setstate(st[1])
    torch.set_rng_state(st[2])


# ---------------------------------------------------------------- evaluation
def eval_episode(env, policy, lstm, s0, ticks=RL.EP_TICKS, obs=None, hist=1, live=None, trace=None):
    """live / trace (opt-in observers, live_rec.py): read the raw state, action and reward of every tick."""
    state = window0(_sc(obs, s0), lstm, hist)
    names = comp_names(env)
    tot, comp, pods, lat, pend = 0.0, np.zeros(len(names)), [], [], []
    for _ in range(ticks):
        a = policy(state)
        st, r, c, _ = env_step(env, a)
        if live is not None:
            live.add("eval", st, a, r)
        if trace is not None:
            trace.add(st, a, r)
        tot += r
        comp += np.array(c, dtype=float)
        pods.append(st[5])
        lat.append(st[4])
        if len(st) > 6:
            pend.append(st[6])
        state = push(state, _sc(obs, st), lstm, hist)
    r = dict(ret=float(tot), ticks=int(ticks), mean_pods=float(np.mean(pods)),
             **{c: float(v) for c, v in zip(names, comp)},
             p95_latency=float(np.percentile(lat, 95)), mean_latency=float(np.mean(lat)))
    if isinstance(env, OV.OrigEnv):       # cycle 9 only (coupled JSON unchanged)
        r.update(pods_behaviour(s0[5], pods))
        if env.delay is not None:         # cycle 10 only
            r["mean_pending"] = float(np.mean(pend))
    return r


def pods_behaviour(p0, pods):
    """frac_const_pods: fraction of ticks whose pods equal the previous tick's (tick 1 vs the reset state)."""
    p = np.concatenate([[p0], np.asarray(pods, dtype=float)])
    ch = int(np.count_nonzero(np.diff(p)))
    return dict(frac_const_pods=1.0 - ch / float(len(pods)), pods_changes=ch)


def copy_model(model):
    """Deep copy of a Q-network. The LSTM's stateful hidden tensor is a non-leaf of the training graph (deepcopy
    refuses it): it is detached from the copy only (the training model keeps the very same object)."""
    h = model.__dict__.get("hidden", None)
    if h is None:
        return copy.deepcopy(model)
    model.__dict__["hidden"] = None
    try:
        m = copy.deepcopy(model)
    finally:
        model.__dict__["hidden"] = h
    m.hidden = tuple(x.detach().clone() for x in h)
    return m


def evaluate(online_model, lstm, target_env, seeds, label, ticks=RL.EP_TICKS, obs=None, hist=1, live=None, traj_n=0):
    """Greedy evaluation of a DEEP COPY of online_model; global RNG states restored afterwards.
    live: LiveRecorder (or None); traj_n: record per-tick trajectories of the first traj_n episodes (key 'trajectory')."""
    st = rng_get()
    m = copy_model(online_model)
    pol = RL.greedy_fn(SimpleNamespace(online_model=m), lstm)
    eps, names, trajs = [], RL.COMP, []
    try:
        for i, s in enumerate(seeds):
            env = make_env(target_env, "coupled", int(s))
            if lstm:
                m.hidden = m.init_hidden()
            names = comp_names(env)
            if live is not None:
                live.set_label(f"{label} seed {int(s)}")
            tr = live_rec.TrajRecorder() if i < int(traj_n) else None
            eps.append(eval_episode(env, pol, lstm, env.reset(), ticks, obs, hist, live, tr))
            if tr is not None:
                trajs.append(dict(seed=int(s), **tr.d))
    finally:
        rng_set(st)
    r = RL.summarize(eps, comp_names=names, label=label)
    if trajs:
        r["trajectory"] = dict(keys=list(live_rec.TRAJ_KEYS), episodes=trajs)
    return r


# ---------------------------------------------------------------- training
def train_episodic(agent, act, env, ticks, lstm, tel=None, tick0=0, phase="target", checkpoints=(), on_ck=None,
                   extra=None, reward_div=1.0, obs=None, eps=None, hist=1, live=None):
    """replay(16) every tick once memory > 16; reset + reset-observation window on done. on_ck(t) at checkpoints.
    Cycle 8: memory stores reward / reward_div (unchanged if 1), observations pass through obs, eps(t) sets epsilon.
    live: opt-in LiveRecorder (reads the raw state/action/reward of every tick)."""
    lphase = "pretrain" if phase == "source" else "train"
    state = window0(_sc(obs, env.reset()), lstm, hist)
    div = float(reward_div)
    ep_ret, ep_returns, cks = 0.0, [], set(int(c) for c in checkpoints)
    cpu_eval = 0.0
    gclip = getattr(agent, "grad_clip", None) is not None      # cycle 23 telemetry (only with --grad-clip)
    g_last = 0
    for t in range(1, int(ticks) + 1):
        if eps is not None:
            agent.epsilon = eps(t - 1)
        action = act(state)
        new_state, reward, _, done = env_step(env, action)
        if live is not None:
            live.add(lphase, new_state, action, reward)
        ep_ret += reward
        nxt = push(state, _sc(obs, new_state), lstm, hist)
        agent.remember(state, action, reward if div == 1.0 else reward / div, nxt, done)
        if t == int(ticks) and not done and getattr(agent, "n_step", 1) > 1:
            agent.flush_nstep(nxt)      # cycle 19: phase end = truncation -> bootstrapped flush BEFORE the last replay
        state = nxt
        if done:
            ep_returns.append(float(ep_ret))
            ep_ret = 0.0
            state = window0(_sc(obs, env.reset()), lstm, hist)
        if len(agent.memory) > RL.BATCH:
            if getattr(agent, "per", False):        # cycle 19: PER beta anneals in t / ticks (1.0 at the last tick)
                agent.per_progress = t / float(ticks)
            agent.replay(RL.BATCH)
        if tel and t % LOG_EVERY == 0:
            gx = {}
            if gclip:
                gw = agent.grad_norms[g_last:]
                g_last = len(agent.grad_norms)
                if gw:
                    gx = {f"{phase}_grad_norm_mean": float(np.mean(gw)),
                          f"{phase}_clip_frac": float(np.mean(np.asarray(gw) > agent.grad_clip))}
            tel.log(tick0 + t, **{f"{phase}_epsilon": agent.epsilon, f"{phase}_partial_ep_return": ep_ret}, **gx)
            tel.progress(tick0 + t + (extra[0] if extra else 0))
        if t in cks and on_ck:
            c = time.process_time()
            on_ck(t)
            cpu_eval += time.process_time() - c
    out = dict(episode_returns=ep_returns, partial_episode_return=float(ep_ret), partial_episode_ticks=int(ticks) % RL.EP_TICKS,
                epsilon_end=float(agent.epsilon), eval_cpu_s=cpu_eval)
    if getattr(agent, "per", False):
        out["per_beta_final"] = agent.per_beta_last
    if gclip:
        out["grad_norm"] = agent.grad_norm_stats()
    return out


def ck_path(out, t):
    root, ext = os.path.splitext(out)
    return f"{root}.ck{int(t)}{ext or '.json'}"


def arm_name(agent, source_variant):
    return f"transfer_{source_variant}" if agent == "dinos_t" else agent


def validate_c6(agent, agent_variant, target_sync, diag):
    """Cycle-6 options: returns the effective target-sync period (None = verbatim: after every replay)."""
    if agent_variant not in AGENT_VARIANTS:
        raise SystemExit(f"--agent-variant must be one of {AGENT_VARIANTS}")
    if agent_variant == "dense_fixed" and (agent != "lstm_scratch" or diag is not None):
        raise SystemExit("--agent-variant dense_fixed applies only to --agent lstm_scratch (no --diag)")
    if agent_variant in V2_VARIANTS and (agent not in ("lstm_scratch", "dinos_t") or diag is not None):
        raise SystemExit(f"--agent-variant {agent_variant} applies only to --agent lstm_scratch / dinos_t (no --diag)")
    if agent == "derp" and (agent_variant != "verbatim" or diag is not None):
        raise SystemExit("derp: --agent-variant fixed and --diag apply only to the LSTM arms (lstm_scratch, dinos_t)")
    if agent_variant == "verbatim":
        if target_sync is not None:
            raise SystemExit("--target-sync applies only to --agent-variant fixed (verbatim syncs every replay)")
        sync = None
    else:
        sync = FX.DEFAULT_TARGET_SYNC if target_sync is None else int(target_sync)
        if sync < 1:
            raise SystemExit("--target-sync must be >= 1")
    if diag is not None and int(diag) < 1:
        raise SystemExit("--diag must be >= 1 (replays between records)")
    return sync


def validate(agent, source_variant, pretrain_ticks, finetune_ticks, total_ticks, checkpoints):
    if agent not in C5_AGENTS:
        raise SystemExit(f"cycle-5 protocol supports --agent {C5_AGENTS}")
    if agent == "dinos_t":
        if source_variant not in SOURCE_VARIANTS:
            raise SystemExit(f"dinos_t needs --source-variant {SOURCE_VARIANTS}")
        p = int(pretrain_ticks or 0)
        if source_variant == "none" and p:
            raise SystemExit("--source-variant none takes no --pretrain-ticks (random-init control)")
        if source_variant != "none" and p <= 0:
            raise SystemExit("--source-variant legacy|coupled needs --pretrain-ticks > 0")
        if not finetune_ticks or int(finetune_ticks) <= 0:
            raise SystemExit("dinos_t needs --finetune-ticks > 0")
        T = int(finetune_ticks)
        if total_ticks is not None and int(total_ticks) != p + T:
            raise SystemExit(f"--total-ticks {total_ticks} != pretrain {p} + finetune {T}")
    else:
        if source_variant is not None or pretrain_ticks or finetune_ticks:
            raise SystemExit(f"{agent}: --source-variant/--pretrain-ticks/--finetune-ticks apply only to dinos_t")
        if not total_ticks or int(total_ticks) <= 0:
            raise SystemExit(f"{agent} needs --total-ticks > 0 (target training ticks)")
        p, T = 0, int(total_ticks)
    cks = sorted(set(int(c) for c in (checkpoints or (T,))))
    if any(c <= 0 or c > T for c in cks):
        raise SystemExit(f"--checkpoints must lie in 1..{T} (target ticks), got {cks}")
    return p, T, cks


def validate_c9(agent, diag, env_family, orig_variant, transfer=False):
    """-> True for the cycle-9 original-env family (transfer: cycle-9b dinos_t, checked by validate_c9b)."""
    if env_family not in ENV_FAMILIES:
        raise SystemExit(f"--env-family must be one of {ENV_FAMILIES}")
    if env_family == "coupled":
        if orig_variant is not None:
            raise SystemExit("--orig-variant needs --env-family orig")
        return False
    if agent not in ("derp", "lstm_scratch") and not (agent == "dinos_t" and transfer):
        raise SystemExit("--env-family orig supports --agent derp / lstm_scratch (dinos_t: --pretrain-env-family)")
    if diag is not None:
        raise SystemExit("--diag is not available on --env-family orig (its probes use the coupled env)")
    OV.parse(orig_variant)
    return True


def validate_c10(agent, agent_variant, orig_variant, pretrain_env_family, pretrain_variant):
    """Cycle 10 (delay variants, 7 inputs). -> state size of the target env (6 = cycle 9 unchanged)."""
    k = OV.state_size(orig_variant)
    if k == RL.STATE:
        if pretrain_env_family == "orig" and OV.state_size(pretrain_variant) != k:
            raise SystemExit("transfer: source and target must have the same state size (a _d<D> source needs a "
                             "_d<D> target)")
        return k
    if agent != "derp" and agent_variant not in FIXED_VARIANTS:
        raise SystemExit("delay variants (_d<D>, 7 inputs) need --agent-variant fixed for the LSTM arms")
    if pretrain_env_family is not None and (pretrain_env_family != "orig" or OV.state_size(pretrain_variant) != k):
        raise SystemExit("transfer onto a delay variant needs an orig-family _d<D> source (same 7-feature state)")
    return k


def validate_c9b(agent, env_family, source_variant, pretrain_env_family, pretrain_variant, finetune_eps,
                 agent_variant, derp_history):
    """Cycle-9b options. -> (orig transfer?, derp history H). Every option at its default is a no-op."""
    h = int(derp_history)
    if h < 1:
        raise SystemExit("--derp-history must be >= 1")
    if finetune_eps not in FINETUNE_EPS:
        raise SystemExit(f"--finetune-eps must be one of {FINETUNE_EPS}")
    transfer = pretrain_env_family is not None
    if h > 1 and (agent != "derp" or env_family != "orig"):
        raise SystemExit("--derp-history applies only to --agent derp on --env-family orig")
    if not transfer:
        if pretrain_variant is not None or finetune_eps != "restart":
            raise SystemExit("--pretrain-variant/--finetune-eps need --pretrain-env-family")
        return False, h
    if env_family != "orig" or agent != "dinos_t":
        raise SystemExit("--pretrain-env-family applies only to --agent dinos_t on --env-family orig")
    if source_variant is not None:
        raise SystemExit("--env-family orig transfer: use --pretrain-env-family/--pretrain-variant, not --source-variant")
    if agent_variant not in ("fixed",) + V2_VARIANTS:
        raise SystemExit(f"--env-family orig transfer needs --agent-variant fixed or one of {V2_VARIANTS}")
    if pretrain_env_family not in PRETRAIN_FAMILIES:
        raise SystemExit(f"--pretrain-env-family must be one of {PRETRAIN_FAMILIES}")
    if pretrain_env_family == "orig":
        OV.parse(pretrain_variant)
    elif pretrain_variant not in ("legacy", "coupled"):
        raise SystemExit("--pretrain-env-family coupled needs --pretrain-variant legacy|coupled")
    return True, h


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


C19_DEFAULTS = dict(n_step=1, dueling=False, per=False)


def validate_c19(ddqn, agent, agent_variant, diag, source_variant=None):
    """cycle 19 Double-DQN add-ons -> normalised dict, or None when every option is at its default (= no change)."""
    d = dict(C19_DEFAULTS)
    for k, v in dict(ddqn or {}).items():
        if k not in d:
            raise SystemExit(f"unknown cycle-19 option {k!r}")
        d[k] = v
    d = dict(n_step=int(d["n_step"]), dueling=bool(d["dueling"]), per=bool(d["per"]))
    if d["n_step"] < 1:
        raise SystemExit("--n-step must be >= 1")
    if d == C19_DEFAULTS:
        return None
    if agent == "derp" or agent_variant not in FIXED_VARIANTS:
        raise SystemExit("--n-step/--dueling/--per apply only to the fixed-agent family (--agent lstm_scratch / dinos_t "
                         f"with --agent-variant in {FIXED_VARIANTS})")
    if diag is not None:
        raise SystemExit("--n-step/--dueling/--per cannot be combined with --diag")
    if d["dueling"] and agent == "dinos_t" and source_variant == "none":
        raise SystemExit("--dueling needs a pretrained source (the 'none' init is a plain myLSTM)")
    return d


def validate_c23(lr, grad_clip, pm, agent, agent_variant, env_family, orig_variant, pretrain_env_family,
                 pretrain_variant):
    """cycle 23 -> optim kwargs for set_optim (None when lr and grad_clip are off) and pm (bool)."""
    op = {}
    if lr is not None:
        op["lr"] = float(lr)
        if not np.isfinite(op["lr"]) or op["lr"] <= 0:
            raise SystemExit("--lr must be a finite number > 0")
    if grad_clip is not None:
        op["grad_clip"] = float(grad_clip)
        if not np.isfinite(op["grad_clip"]) or op["grad_clip"] <= 0:
            raise SystemExit("--grad-clip must be a finite number > 0")
    pm = bool(pm)
    if (op or pm) and (agent == "derp" or agent_variant not in FIXED_VARIANTS):
        raise SystemExit("--lr/--grad-clip/--pending-maturity apply only to the fixed-agent family (--agent "
                         f"lstm_scratch / dinos_t with --agent-variant in {FIXED_VARIANTS})")
    if pm:
        if env_family != "orig":
            raise SystemExit("--pending-maturity on needs --env-family orig")
        OV.state_size(orig_variant, True)               # a '_d<D>' target (SystemExit otherwise)
        if pretrain_env_family is not None:
            if pretrain_env_family != "orig":
                raise SystemExit("--pending-maturity on with transfer needs an orig-family '_d<D>' source")
            OV.state_size(pretrain_variant, True)
        if agent_variant in V2_VARIANTS and V2.is_mlp(agent_variant) and agent_variant not in V2.MATCH_PM:
            raise SystemExit(f"--pending-maturity on: {agent_variant} is matched for 7 inputs; use one of "
                             f"{tuple(V2.MATCH_PM)} (re-matched for {V2.PM_STATE_SIZE})")
    elif agent_variant in V2.MATCH_PM:
        raise SystemExit(f"{agent_variant} is matched for --pending-maturity on ({V2.PM_STATE_SIZE} inputs)")
    return (op or None), pm


def validate_c8(reward_div, obs_scale, eps_schedule):
    d = float(reward_div)
    if not np.isfinite(d) or d <= 0:
        raise SystemExit("--reward-div must be a finite number > 0")
    if obs_scale not in OBS_SCALES:
        raise SystemExit(f"--obs-scale must be one of {OBS_SCALES}")
    if eps_schedule not in EPS_SCHEDULES:
        raise SystemExit(f"--eps-schedule must be one of {EPS_SCHEDULES}")
    return d


def run_c5(agent, seed, source_variant=None, pretrain_ticks=None, finetune_ticks=None, total_ticks=None,
           checkpoints=None, target_env=DEFAULT_TARGET_ENV, source_env=DEFAULT_SOURCE_ENV, eval_splits=("val", "test"),
           tel_dir=None, out=None, eval_ticks=RL.EP_TICKS, agent_variant="verbatim", target_sync=None, diag=None,
           reward_div=1.0, obs_scale="off", eps_schedule="original", env_family="coupled", orig_variant=None,
           pretrain_env_family=None, pretrain_variant=None, finetune_eps="restart", derp_history=1, live_file=None,
           record_traj=0, save_weights=False, source_checkpoints=None, ddqn=None, lr=None, grad_clip=None,
           pending_maturity=False):
    otr, hist = validate_c9b(agent, env_family, source_variant, pretrain_env_family, pretrain_variant, finetune_eps,
                             agent_variant, derp_history)
    # orig transfer: the source checks of validate() are those of a pretrained dinos_t ("legacy" stands for any source)
    P, T, cks = validate(agent, "legacy" if otr else source_variant, pretrain_ticks, finetune_ticks, total_ticks,
                         checkpoints)
    sync = validate_c6(agent, agent_variant, target_sync, diag)
    rdiv = validate_c8(reward_div, obs_scale, eps_schedule)
    dd_opts = validate_c19(ddqn, agent, agent_variant, diag, source_variant)
    op, pm = validate_c23(lr, grad_clip, pending_maturity, agent, agent_variant, env_family, orig_variant,
                          pretrain_env_family, pretrain_variant)
    opk = {"optim": op} if op else {}       # cycle 23 kwargs of every fixed-family make_* (absent = unchanged)

    def dd(ticks, phase):
        """cycle 19: set_ddqn kwargs for an agent trained `ticks` ticks (None = defaults, nothing is called)."""
        if dd_opts is None:
            return None
        return dict(dd_opts, per_anneal_replays=max(1, int(ticks) - RL.BATCH), per_seed=[int(seed), 19, phase])
    orig = validate_c9(agent, diag, env_family, orig_variant, transfer=otr)
    if agent_variant in ("dense_fixed",) + V2_VARIANTS and not orig:
        raise SystemExit(f"--agent-variant {agent_variant} needs --env-family orig")
    sdim = RL.STATE
    if orig:
        target_env = ORIG_PREFIX + orig_variant
        sdim = validate_c10(agent, agent_variant, orig_variant, pretrain_env_family, pretrain_variant)
        if pm:          # cycle 23: + 5 maturity features (source and target alike)
            sdim = OV.state_size(orig_variant, True)
    obs = obs_fn(obs_scale, env_family, orig_variant if orig else None, pm)
    fixed = agent_variant in FIXED_VARIANTS
    dense = agent_variant == "dense_fixed"
    v2 = agent_variant in V2_VARIANTS
    scks = sorted(set(int(c) for c in source_checkpoints)) if source_checkpoints else None
    if scks is not None and (not otr or any(c <= 0 or c > P for c in scks) or scks[-1] != P):
        raise SystemExit("--source-checkpoints needs orig transfer and counts in 1..--pretrain-ticks, the largest = "
                         "--pretrain-ticks")
    if (save_weights or scks) and not out:
        raise SystemExit("--save-weights/--source-checkpoints need --out")
    if save_weights or scks:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if save_weights and "val" not in eval_splits:
        raise SystemExit("--save-weights selects on val: --eval-split must include val")

    def mk_scratch(ddq=None):
        """cycle 18: scratch agent of the V.2 variant; otherwise the fixed LSTM (unchanged). ddq: cycle-19 options."""
        if v2:
            return V2.make_scratch(agent_variant, sdim, RL.ACTIONS, RL.SEQ, sync, **({"ddqn": ddq} if ddq else {}),
                                   **opk)
        return FX.make_scratch(sdim, RL.ACTIONS, RL.SEQ, sync, **({"ddqn": ddq} if ddq else {}), **opk)
    eval_splits = RL.parse_splits(eval_splits if isinstance(eval_splits, str) else ",".join(eval_splits))
    RL.check_seeds(seed, "fair", eval_splits)
    from tuning.legacy import confirm_guard as CG
    CG.authorise(eval_splits, "c5_protocol.run_c5")      # test10/test11: grant from the CLI guard required
    lstm = agent != "derp"
    n_eval_eps = sum(len(RL.SPLITS[s]) for s in eval_splits)
    n_src_eval = len(scks) * len(RL.SPLITS["val"]) * eval_ticks if scks else 0
    tel = RL.Telemetry(tel_dir, P + T + len(cks) * n_eval_eps * eval_ticks + n_src_eval)
    res = dict(protocol="c5", agent=agent, arm=arm_name(agent, source_variant), seed=int(seed),
               env_variant="orig_family" if orig else "coupled", target_env=target_env,
               target_variant=orig_variant if orig else "coupled",
               source_env=source_env if (agent == "dinos_t" and source_variant != "none") else None,
               source_variant=source_variant if agent == "dinos_t" else None,
               pretrain_ticks=P, target_ticks=T, total_ticks=P + T, checkpoints=cks, eval_splits=list(eval_splits),
               eval_seeds={sp: list(RL.SPLITS[sp]) for sp in eval_splits}, eval_episode_ticks=int(eval_ticks),
               target_stream_seed=stream_seed(seed, TARGET_STREAM),
               source_stream_seed=stream_seed(seed, SOURCE_STREAM) if P else None,
               train_reset_on_done=True, reset_observation_window=True, torch_threads=torch.get_num_threads(),
               legacy_root="~/simulations/redisk8s/timeseries/Pytorch")
    res.update(reward_div=rdiv, obs_scale=obs_scale,
               obs_scale_divisors=dict(zip(obs_features(env_family, orig_variant if orig else None, pm),
                                           obs_divisors(env_family, orig_variant if orig else None, pm).tolist()))
               if obs is not None else None,
               eps_schedule=eps_schedule, eps_linear_ticks=LINEAR_EPS_TICKS if eps_schedule == "linear10k" else None)
    if orig:        # cycle 9 keys (absent on the coupled path, so its JSON is unchanged)
        res.update(env_family="orig", orig_variant=orig_variant, reward_components=list(OV.COMP),
                   orig_env="P/environment.py (env_orig.py) via orig_variants.py")
        if OV.startup_delay(orig_variant) is not None:      # cycle 10 keys (absent on cycle-9 runs)
            res.update(startup_delay=OV.startup_delay(orig_variant), state_size=sdim,
                       obs_features=list(OV.features(orig_variant, pm)))
        if agent == "derp":
            res.update(derp_history=hist, input_size=hist * sdim)
            if hist > 1:
                res["arm"] = f"derp_hist{hist}"
        if otr:
            sfam_env = ORIG_PREFIX + pretrain_variant if pretrain_env_family == "orig" else source_env
            spv = pretrain_variant if pretrain_env_family == "orig" else None
            sobs = obs_fn(obs_scale, pretrain_env_family, spv, pm)
            res.update(arm=f"transfer_{pretrain_env_family}_{pretrain_variant}", pretrain_env_family=pretrain_env_family,
                       pretrain_variant=pretrain_variant, source_env=sfam_env, source_variant=None,
                       finetune_ticks=T, finetune_eps=finetune_eps,
                       source_obs_scale_divisors=(dict(zip(obs_features(pretrain_env_family, spv, pm),
                                                           obs_divisors(pretrain_env_family, spv, pm).tolist()))
                                                  if sobs is not None else None),
                       finetune_agent="agent_lstm_fixed.make_scratch (lr 1e-4, eps floor 0.05) + pretrained weights")
    if v2:          # cycle 18 arm name (set before the live recorder reads it)
        res["arm"] = v2_arm(agent, agent_variant)
    if agent != "derp":     # cycle 6 keys (absent for derp, so its JSON is unchanged)
        res.update(agent_variant=agent_variant, target_sync_replays=sync if fixed else 1,
                   double_dqn_target=fixed, hidden_reset_per_window=fixed)
    live, traj_n = None, int(record_traj or 0)
    if traj_n < 0:
        raise SystemExit("--record-traj must be >= 0")
    if traj_n:
        res["record_traj"] = dict(episodes=traj_n, checkpoints=list(cks), split=eval_splits[-1])
    if live_file:
        res["live_file"] = str(live_file)
        live = live_rec.LiveRecorder(live_file, dict(arm=res["arm"], agent=agent, seed=int(seed),
                                                     variant=res["target_variant"], out=out, total_ticks=P + T,
                                                     checkpoints=cks, eval_splits=list(eval_splits)))
    if op:          # cycle 23 keys (absent with the defaults)
        res["optim_options"] = dict(op, applies_to="pretrain and target phases" if P else "target phase")
        if "lr" in op and "finetune_agent" in res:
            res["finetune_agent"] += f" [cycle 23: lr overridden to {op['lr']:g}]"
    if pm:
        res["pending_maturity"] = dict(
            k=OV.MATURITY_K, features=list(OV.FEATURES_MATURITY), obs_scale_divisor=float(OV.OBS_SCALE_MATURITY[0]),
            state_size=sdim, definition="pending pods (controller request/cancel history, startup delay D) that become "
            "active k ticks after the observation, k = 1..5; 0/1 each; zeros at D = 0; no load / time exposed",
            applies_to="pretrain and target phases" if P else "target phase")
        target_env = target_env + PM_SUFFIX            # internal env names (recorded names above stay unchanged)
        if otr:
            sfam_env = sfam_env + PM_SUFFIX
    RL.seed_all(seed)
    dg = None
    if diag is not None:
        st0 = rng_get()
        probes = c6_diag.probe_windows(make_env, lambda s0, l_: window0(_sc(obs, s0), l_),
                                       lambda st, ns, l_: push(st, _sc(obs, ns), l_))   # private streams; RNG restored
        rng_set(st0)
        dg = c6_diag.Diag(diag, probes, seed)
    clock = [0]

    def watch(ag, act, phase, tick0):
        """--diag: attach the observers and count the actions taken (no effect on the computation)."""
        if dg is None:
            return act
        dg.attach(ag, phase, lambda: tick0 + clock[0])

        def f(s):
            a_ = act(s)
            clock[0] += 1
            dg.count_action(a_)
            return a_
        return f
    t0, c0 = time.time(), time.process_time()
    if agent == "derp":
        ag = ddqnagent.DDQNAgent(hist * sdim, RL.ACTIONS)
        act = lambda s: ag.act(s, verbose=0)  # noqa: E731
    elif otr:       # cycle 9b: transfer onto the orig family
        src = mk_scratch(dd(P, 1))
        senv = make_env(sfam_env, pretrain_variant, stream_seed(seed, SOURCE_STREAM))
        src_ck, src_val = {}, {}

        def on_src_ck(t):       # cycle 18 --source-checkpoints: source VAL evaluation + kept weights
            r = evaluate(src.online_model, True, sfam_env, RL.SPLITS["val"], "source_val", eval_ticks, sobs, 1, live)
            src_val[int(t)] = float(r["mean_ret"])
            src_ck[int(t)] = {k: v.detach().clone() for k, v in src.online_model.state_dict().items()}
            if save_weights:
                torch.save(src_ck[int(t)], pt_path(out, f"src.ck{int(t)}"))
            tel.log(int(t), source_tick=int(t), source_val_mean_ret=r["mean_ret"])
        pr = train_episodic(src, lambda s: src.act(s, verbose=0), senv, P, True, tel, 0, "source", reward_div=rdiv,
                            obs=sobs, eps=eps_fn(src, eps_schedule), live=live,
                            **(dict(checkpoints=scks, on_ck=on_src_ck) if scks else {}))
        pr["n_params"] = n_params(src.online_model)
        if op:
            pr["learning_rate"] = float(src.learning_rate)      # cycle 23 (only with --lr/--grad-clip)
        res["pretrain"] = pr
        if pm:
            res["pretrain"]["input_size"] = sdim
        if scks:
            sel = best_ck(src_val)
            w = src_ck[sel]
            res["source_selection"] = dict(checkpoints=scks, split="val", env=sfam_env,
                                           val_by_checkpoint={str(c): src_val[c] for c in scks}, selected=sel,
                                           selected_val_mean_ret=src_val[sel], rule="max source val mean; ties earlier")
        else:
            w = {k: v.clone() for k, v in src.online_model.state_dict().items()}
        res["init"] = f"pretrained {P} ticks on {sfam_env} ({pretrain_env_family}/{pretrain_variant})"
        ag = mk_scratch(dd(T, 2))
        ag.load_weights(w)
        if finetune_eps == "low":
            ag.epsilon = LOW_EPS
        act = lambda s: ag.act(s, verbose=0)  # noqa: E731
    elif agent == "lstm_scratch":
        if v2:          # cycle 18
            ag = mk_scratch(dd(T, 2))
        elif dense:     # cycle 10b
            ag = DFX.make_scratch(sdim, RL.ACTIONS, RL.SEQ, sync, **({"ddqn": dd(T, 2)} if dd_opts else {}), **opk,
                                  **({"hidden": DFX.DENSE_HIDDEN_PM} if pm else {}))
            res.update(arm=f"mlp_hist{RL.SEQ}", network="mlp_flat_window", dense_hidden=ag.dense_hidden,
                       input_size=RL.SEQ * sdim)
        else:
            ag = FX.make_scratch(sdim, RL.ACTIONS, RL.SEQ, sync, **({"ddqn": dd(T, 2)} if dd_opts else {}), **opk) \
                if fixed else RL.make_lstm_scratch()
        act = lambda s: ag.act(s, verbose=0)  # noqa: E731
    else:
        if source_variant == "none":
            w = networks_lstm.myLSTM(RL.STATE, 128, 3, RL.ACTIONS).state_dict()
            res["init"] = "random (fresh myLSTM)"
        else:
            src = FX.make_scratch(RL.STATE, RL.ACTIONS, RL.SEQ, sync, **({"ddqn": dd(P, 1)} if dd_opts else {}),
                                  **opk) if fixed else RL.make_lstm_scratch()
            senv = make_env(source_env, source_variant, stream_seed(seed, SOURCE_STREAM))
            sact = watch(src, lambda s: src.act(s, verbose=0), "source", 0)
            seps = eps_fn(src, eps_schedule)
            pr = train_episodic(src, sact, senv, P, True, tel, 0, "source", reward_div=rdiv, obs=obs, eps=seps,
                                live=live)
            if dg is not None:
                dg.finish()
            if op and fixed:
                pr["learning_rate"] = float(src.learning_rate)  # cycle 23
            res["pretrain"] = pr
            w = src.online_model.state_dict()
            res["init"] = f"pretrained on {source_variant} {source_env}"
        w = {k: v.clone() for k, v in w.items()}
        ag = FX.make_transfer(RL.STATE, RL.ACTIONS, RL.SEQ, w, sync, **({"ddqn": dd(T, 2)} if dd_opts else {}),
                              **opk) if fixed else RL.make_transfer(w)
        act = ag.act
    teps = eps_fn(ag, eps_schedule)
    res["epsilon_start"], res["epsilon_min"], res["epsilon_decay"] = float(ag.epsilon), float(ag.epsilon_min), \
        float(ag.epsilon_decay)
    res["learning_rate"] = float(ag.learning_rate)
    if v2:          # cycle 18 keys (absent on every other arm)
        kind = "mlp" if V2.is_mlp(agent_variant) else "seq"
        res.update(network="spec_seq" if kind == "seq" else "mlp_flat_window", net_spec=V2.ARCHS[agent_variant],
                   architecture=V2.architecture_str(agent_variant, sdim, RL.SEQ, RL.ACTIONS, bool(ag.dueling)),
                   variant_label=V2.LABEL.get(agent_variant, "MLP control " + V2.ARCHS[agent_variant]) + (" + transfer" if otr else ""),
                   matched_to=list(V2.MATCH.get(agent_variant, V2.MATCH_PM.get(agent_variant, ()))) or None,
                   hyperparameters=V2.hyperparameters(ag), input_size=RL.SEQ * sdim if kind == "mlp" else sdim)
        if kind == "mlp":
            res["dense_hidden"] = int(ag.dense_hidden)
        if otr:
            res["finetune_agent"] = (f"agent_lstm_v2.make_scratch({agent_variant!r}) (lr 1e-4, eps floor 0.05) + "
                                     f"pretrained weights of the same variant")
    if orig:
        res["n_params"] = n_params(ag.online_model)
    if dd_opts is not None:     # cycle 19 key (absent with the defaults, so every earlier JSON is unchanged)
        res["ddqn_options"] = dict(ag.ddqn_config(), applies_to="pretrain and target phases" if P else "target phase")
    res["pretrain_cpu_s"] = time.process_time() - c0
    res["checkpoint_results"] = {}
    extra = [n_src_eval]  # eval ticks done so far (progress.json counts training + evaluation ticks)

    def on_ck(t):
        c = time.process_time()
        r = {"target_tick": int(t), "epsilon_at_checkpoint": float(ag.epsilon)}
        for sp in eval_splits:
            tn = traj_n if sp == eval_splits[-1] else 0
            r[sp] = evaluate(ag.online_model, lstm, target_env, RL.SPLITS[sp], sp, eval_ticks, obs, hist, live, tn)
        r["cpu_s"] = time.process_time() - c
        res["checkpoint_results"][str(int(t))] = r
        extra[0] += n_eval_eps * eval_ticks
        tel.progress(P + t + extra[0])
        tel.log(P + t, target_tick=t, **{f"{sp}_mean_ret": r[sp]["mean_ret"] for sp in eval_splits},
                **{f"{sp}_mean_pods": r[sp]["mean_pods"] for sp in eval_splits})
        if save_weights:
            torch.save({k: v.detach().clone() for k, v in ag.online_model.state_dict().items()}, pt_path(out, f"ck{int(t)}"))
        if out:
            meta = {k: v for k, v in res.items() if k not in ("checkpoint_results", "pretrain")}
            write_json(ck_path(out, t), dict(meta, checkpoint=r))

    tenv = make_env(target_env, "coupled", stream_seed(seed, TARGET_STREAM))
    if dg is not None:
        clock[0] = 0
        act = watch(ag, act, "target", P)
    c1 = time.process_time()
    tr = train_episodic(ag, act, tenv, T, lstm, tel, P, "target", cks, on_ck, extra, reward_div=rdiv, obs=obs, eps=teps,
                        hist=hist, live=live)
    res["target_train"] = tr
    res["target_train_cpu_s"] = time.process_time() - c1 - tr["eval_cpu_s"]
    res["eval_cpu_s"] = tr["eval_cpu_s"]
    res["cpu_s"], res["wall_s"] = time.process_time() - c0, time.time() - t0
    res["train_cpu_s_per_1k_ticks"] = (res["pretrain_cpu_s"] + res["target_train_cpu_s"]) / ((P + T) / 1000.0)
    if op and "grad_clip" in op:    # cycle 23: pre-clip grad-norm statistics per phase
        res["grad_norm"] = {k: v["grad_norm"] for k, v in (("pretrain", res.get("pretrain") or {}), ("target", tr))
                            if isinstance(v, dict) and "grad_norm" in v}
    if fixed:
        res["target_syncs"] = int(ag.n_target_syncs)
        res["replays"] = int(ag.n_replays)
    if dg is not None:
        dg.finish()
        if out:
            root, ext = os.path.splitext(out)
            res["diag"] = f"{root}.diag{ext or '.json'}"
            write_json(res["diag"], dict(arm=res["arm"], seed=int(seed), agent_variant=agent_variant,
                                         target_sync_replays=res["target_sync_replays"], **dg.summary()))
        else:
            res["diag"] = dg.summary()
    if save_weights:
        vb = {int(c): float(r["val"]["mean_ret"]) for c, r in res["checkpoint_results"].items()}
        sel = best_ck(vb)
        sd = torch.load(pt_path(out, f"ck{sel}"))
        torch.save(sd, pt_path(out, "selected"))
        res["saved_weights"] = dict(checkpoints={str(c): os.path.basename(pt_path(out, f"ck{c}")) for c in cks},
                                    selected_checkpoint=sel, selected_val_mean_ret=vb[sel],
                                    selected_file=os.path.basename(pt_path(out, "selected")),
                                    rule="max val mean of this run; ties earlier (test never used)",
                                    selection_scope="per_seed",
                                    note="per-seed diagnostic only: the summarised policy, Transfer V.2 and the final "
                                         "confirmation use the across-seed locked checkpoint (sum_c10.py "
                                         "--export-locked -> <run>.locked.pt / <run>.locked.json)")
    tel.progress(P + T + len(cks) * n_eval_eps * eval_ticks + n_src_eval)
    if live is not None:
        live.flush(done=True)
    if out:
        write_json(out, res)
    return res


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
