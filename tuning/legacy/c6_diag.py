"""Cycle 6 instrumentation (--diag N) for the LSTM agents of c5_protocol (verbatim and fixed alike).

Pure observers: nothing here changes the training computation or consumes the global numpy/random/torch streams
(tested: identical weights with and without --diag). Hooks, installed on one agent instance:
  * agent.criterion  -> records the Huber loss and, per update, the TD error of the replayed action
                        (target_f[a] - output[a], before the step) and the largest |target_f[j] - output[j]| over the
                        NON-replayed actions j (exactly 0 when the eval and train forwards agree; non-zero = spurious
                        error from the carried hidden state of the verbatim myLSTM)
  * agent.optimizer.step -> L2 norm of all online-model gradients before each step, step counter
  * <agent module>.random during replay() -> proxy whose sample() delegates to random.sample and records the
                        minibatch (to know each update's action); same RNG calls as without the proxy
  * agent.replay    -> counts replays; every N replays appends a record (see Diag.record)
Probe set: PROBE_N fixed 10x6 windows from the coupled lstm_transfer target env on its own stream (seed PROBE_SEED,
uniform random actions from a private RandomState), identical for every arm and run. Q on probes is computed on a deep
copy with zero hidden state per window. Replay-order probe: 16 transitions from the agent's memory (chosen with a
private RandomState), Q of each state through a copy of the online network, (a) carrying the hidden state across the
batch in sampled order vs a private shuffle, (b) with zero hidden per window in both orders; reports the mean/max abs
per-state output difference between the two orders (b is 0 by construction for a stateless forward).
"""
import copy
import random as _random
import sys

import numpy as np
import torch

PROBE_N, PROBE_TICKS, PROBE_SEED = 64, 400, [0, 6, 9]
QS = (5, 25, 50, 75, 95)


def _flat(model):
    return torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])


def _q_reset(model, windows):
    """Q-values with zero hidden per window, via the shared lstm/fc submodules (works for myLSTM and ResetLSTM)."""
    m = model
    with torch.no_grad():
        x = torch.tensor(np.asarray(windows), dtype=torch.float32)
        h0 = torch.zeros(m.num_layers, x.shape[0], m.hidden_size)
        out, _ = m.lstm(x, (h0, h0.clone()))
        return m.fc(out[:, -1, :]).numpy()


def _q_carried(model, windows):
    """Q-values when the hidden state is carried from window to window (verbatim myLSTM behaviour), from zero."""
    m = model
    h = (torch.zeros(m.num_layers, 1, m.hidden_size), torch.zeros(m.num_layers, 1, m.hidden_size))
    qs = []
    with torch.no_grad():
        for w in windows:
            out, h = m.lstm(torch.tensor(np.asarray(w)[None], dtype=torch.float32), h)
            qs.append(m.fc(out[:, -1, :])[0].numpy())
    return np.array(qs)


def probe_windows(make_env, window0, push, n=PROBE_N, ticks=PROBE_TICKS):
    """Fixed probe set: windows at evenly spaced ticks of one coupled lstm_transfer episode on PROBE_SEED."""
    env = make_env("lstm_transfer", "coupled", PROBE_SEED)
    rng = np.random.RandomState(PROBE_SEED[0] + 7919)
    state = window0(env.reset(), True)
    keep = set(np.linspace(20, ticks - 1, n).astype(int).tolist())
    out = []
    for t in range(ticks):
        state = push(state, env.step(int(rng.randint(3)))[0], True)
        if t in keep:
            out.append(state.copy())
    return np.array(out)


def _quant(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return None
    return {f"p{q}": float(np.percentile(v, q)) for q in QS}


class _SampleProxy:
    def __init__(self, diag):
        self._d = diag

    def sample(self, population, k):
        mb = _random.sample(population, k)
        self._d._minibatch = list(mb)
        return mb

    def __getattr__(self, name):
        return getattr(_random, name)


class Diag:
    def __init__(self, every, probes, seed):
        self.every = int(every)
        self.probes = probes
        self.rng = np.random.RandomState([int(seed), 6, 77])
        self.records, self.order_probes = [], []
        self.phase, self.agent = None, None

    # ------------------------------------------------------------ attach
    def attach(self, agent, phase, tick_fn):
        self.phase, self.agent, self.tick_fn = phase, agent, tick_fn
        self.start = _flat(agent.online_model)
        self.n_replay, self.n_steps = 0, 0
        self._reset_interval()
        self._minibatch, self._k = None, 0
        crit, opt, rep = agent.criterion, agent.optimizer, agent.replay
        mod = sys.modules[type(agent).__module__]
        d = self

        def criterion(output, target):
            loss = crit(output, target)
            k = d._k
            d._k += 1
            diff = (target.detach() - output.detach())[0].numpy()
            if d._minibatch is not None and k < len(d._minibatch):
                a = int(d._minibatch[k][1])
                d.i_td.append(float(diff[a]))
                others = np.delete(np.abs(diff), a)
                d.i_spur.append(float(others.max()) if others.size else 0.0)
            d.i_loss.append(float(loss.detach()))
            return loss

        def step(*a, **kw):
            sq = 0.0
            for p in agent.online_model.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().pow(2).sum())
            d.i_gn.append(sq ** 0.5)
            d.n_steps += 1
            return opt_step(*a, **kw)

        def replay(batch_size):
            old = mod.random
            mod.random = _SampleProxy(d)
            d._minibatch, d._k = None, 0
            try:
                r = rep(batch_size)
            finally:
                mod.random = old
            d.n_replay += 1
            if d.n_replay % d.every == 0:
                d.record()
            return r

        opt_step = opt.step
        agent.criterion, agent.replay = criterion, replay
        opt.step = step
        self.i_act = np.zeros(3, dtype=int)
        self.record(initial=True)

    def finish(self):
        """Final record of a phase if replays happened since the last one."""
        if self.agent is not None and self.n_replay % self.every:
            self.record()

    def count_action(self, a):
        self.i_act[int(a)] += 1

    def _reset_interval(self):
        self.i_loss, self.i_gn, self.i_td, self.i_spur = [], [], [], []
        self.i_act = np.zeros(3, dtype=int)

    # ------------------------------------------------------------ records
    def record(self, initial=False):
        ag = self.agent
        cur = _flat(ag.online_model)
        q = _q_reset(ag.online_model, self.probes)
        srt = np.sort(q, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        greedy = np.bincount(q.argmax(1), minlength=3)
        tgt = _flat(ag.target_model)
        st = [s.get("step") for s in ag.optimizer.state.values() if "step" in s]
        rec = dict(phase=self.phase, replay=int(self.n_replay), tick=int(self.tick_fn()), initial=bool(initial),
                   epsilon=float(ag.epsilon), optimizer_steps=int(self.n_steps),
                   adam_step=float(st[0]) if st else 0.0,
                   loss_mean=float(np.mean(self.i_loss)) if self.i_loss else None,
                   grad_norm_mean=float(np.mean(self.i_gn)) if self.i_gn else None,
                   grad_norm_max=float(np.max(self.i_gn)) if self.i_gn else None,
                   param_delta_norm=float(torch.linalg.norm(cur - self.start)),
                   param_delta_rel=float(torch.linalg.norm(cur - self.start) / torch.linalg.norm(self.start)),
                   target_online_gap=float(torch.linalg.norm(cur - tgt)),
                   td=_quant(self.i_td), abs_td=_quant(np.abs(self.i_td)),
                   spurious_nonaction_abs=_quant(self.i_spur),
                   q_mean=float(q.mean()), q_sd=float(q.std()),
                   q_margin=dict(mean=float(margin.mean()), **_quant(margin)),
                   probe_greedy_hist=greedy.astype(int).tolist(),
                   act_hist=self.i_act.astype(int).tolist())
        self.records.append(rec)
        self._reset_interval()
        if not initial:
            self.order_probe()
        return rec

    def order_probe(self, n=16):
        mem = list(self.agent.memory)
        if len(mem) < n:
            return None
        idx = self.rng.choice(len(mem), n, replace=False)
        perm = self.rng.permutation(n)
        w = np.array([mem[i][0] for i in idx])
        m = _strip(self.agent.online_model)
        a_c, b_c = _q_carried(m, w), _q_carried(m, w[perm])[np.argsort(perm)]
        a_r, b_r = _q_reset(m, w), _q_reset(m, w[perm])[np.argsort(perm)]
        dc, dr = np.abs(a_c - b_c), np.abs(a_r - b_r)
        r = dict(phase=self.phase, replay=int(self.n_replay), carried_mean_abs=float(dc.mean()),
                 carried_max_abs=float(dc.max()), carried_argmax_changed=int((a_c.argmax(1) != b_c.argmax(1)).sum()),
                 reset_mean_abs=float(dr.mean()), reset_max_abs=float(dr.max()),
                 reset_vs_carried_mean_abs=float(np.abs(a_r - a_c).mean()), q_abs_mean=float(np.abs(a_r).mean()))
        self.order_probes.append(r)
        return r

    def summary(self):
        return dict(every_replays=self.every, probe_n=int(len(self.probes)), probe_seed=PROBE_SEED,
                    probe_ticks=PROBE_TICKS, records=self.records, order_probes=self.order_probes)


def _strip(model):
    """Deep copy of a model without its stateful hidden tensor (which may be a non-leaf of the training graph)."""
    h = model.__dict__.get("hidden", None)
    if h is None:
        return copy.deepcopy(model)
    model.__dict__["hidden"] = None
    try:
        m = copy.deepcopy(model)
    finally:
        model.__dict__["hidden"] = h
    return m
