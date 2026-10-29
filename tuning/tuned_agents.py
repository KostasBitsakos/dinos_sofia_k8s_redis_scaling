"""Tuned variants of the frozen DInos / DERP agents (sandbox dinos-tuning). The frozen agent files are NOT edited:
the tuned agents subclass them, and every option defaults to the frozen behaviour, so old runs still reproduce.

Config keys (all optional; see DEFAULT_CFG):
  memory_size          replay buffer size (frozen: DERP 500, LSTM 5000)
  lr, gamma            override the frozen learning rate / discount
  target_update_every  gradient updates between target syncs (LSTM frozen: 100; DERP frozen: sync after EVERY replay)
  batched_double_dqn   DERP only: one batched Double-DQN gradient step per replay (frozen: per-sample steps, max target)
  eps_schedule         "per_replay" (frozen: eps *= decay per replay call) | "linear" | "exp"; the last two are a
                       function of the ENV step: set agent.epsilon = eps_at(step) before every act (same for all agents)
  eps_start, eps_end   schedule bounds (eps_start may also be set with "per_replay" to change the initial epsilon;
                       dinos_t frozen start is 0.2, scratch / DERP 1.0)
  eps_decay_steps      env steps over which eps goes from eps_start to eps_end
  reward_clip          learner only: the reward stored in the replay buffer is clipped to [-c, c] AFTER learn_scale (env reward and
                       reported returns are untouched). Default null = off
  reward_transform     learner only: null | "symlog" (sign(r) * log1p(|r|), applied after learn_scale, before reward_clip;
                       keeps the ordering of very bad rewards while bounding the scale)
  grad_clip            max global grad-norm applied before every optimizer step (null = off)
  huber_beta           SmoothL1 beta of the frozen criterion (frozen = 1.0; null = keep)
  n_step               learner only: n-step return. remember() accumulates r_t + g r_{t+1} + .. over n steps, stores
                       (s_t, a_t, R, s_{t+n}, done) and the agent's gamma becomes gamma**n (so every TD target, frozen
                       per-sample DERP path included, bootstraps with gamma**n). At episode end (done) the buffer is flushed with
                       shorter returns (terminal, so no bootstrap). Reward transform/clip act per step BEFORE accumulation.
                       Default 1 = off
  explore_repeat_max   training only: when exploring, the random action is repeated for k ~ U{1..K} steps in total (K = this).
                       The chance of starting an exploration burst is eps / mean(k), so the long-run random-action fraction
                       stays about eps. Greedy evaluation (epsilon == 0) is unaffected. Default null = off. The episode
                       loop calls agent.reset_exploration() at every env reset
  decision_every       k >= 1 (policy part, training AND greedy eval): the agent picks an action every k env steps and sends
                       "hold" (action 1) in between. The learner stores ONE transition per decision: (s_t, executed a_t,
                       sum of the k learner-scaled rewards, s_{t+k}, done) and discounts with gamma**k. With n_step=n the
                       n-step return runs over n decisions (discount gamma**(k*n)). reward_transform / reward_clip act on
                       the k-step summed reward. replay_every counts decisions. A macro cut by the step budget is not stored.
                       Default 1 = off
  scale_down_cooldown  m >= 0 (policy part, training AND eval): after any executed scale action (down or up) a proposed
                       scale-down is converted to hold for the next m env steps (HPA-like stabilisation). The executed
                       (not the proposed) action is stored in replay. Default 0 = off
  reset_head           LSTM only: re-initialise the output Linear (Q-head) after loading the pretrained trunk
"""
import math
import random
from collections import deque

import numpy as np

DEFAULT_CFG = dict(memory_size=None, lr=None, gamma=None, target_update_every=None, batched_double_dqn=False,
                   eps_schedule="per_replay", eps_start=None, eps_end=None, eps_decay_steps=20000, reset_head=False,
                   reward_clip=None, reward_transform=None, grad_clip=None, huber_beta=None, n_step=1, explore_repeat_max=None,
                   decision_every=1, scale_down_cooldown=0)
SCHEDULES = ("per_replay", "linear", "exp")


def check_cfg(cfg):
    c = dict(DEFAULT_CFG)
    unknown = set(cfg) - set(DEFAULT_CFG)
    if unknown:
        raise ValueError(f"unknown agent options: {sorted(unknown)}")
    c.update(cfg)
    if c["reward_transform"] not in (None, "symlog"):
        raise ValueError("reward_transform must be null or 'symlog'")
    if c["eps_schedule"] not in SCHEDULES:
        raise ValueError(f"eps_schedule must be one of {SCHEDULES}")
    if c["eps_schedule"] != "per_replay" and (c["eps_decay_steps"] is None or c["eps_decay_steps"] <= 0):
        raise ValueError("eps_decay_steps must be positive")
    for k in ("memory_size", "target_update_every"):
        if c[k] is not None and (int(c[k]) != c[k] or c[k] < 1):
            raise ValueError(f"{k} must be a positive integer")
    if int(c["n_step"]) != c["n_step"] or c["n_step"] < 1:
        raise ValueError("n_step must be a positive integer")
    if c["explore_repeat_max"] is not None and (int(c["explore_repeat_max"]) != c["explore_repeat_max"]
                                                or c["explore_repeat_max"] < 1):
        raise ValueError("explore_repeat_max must be a positive integer")
    if int(c["decision_every"]) != c["decision_every"] or c["decision_every"] < 1:
        raise ValueError("decision_every must be a positive integer")
    if int(c["scale_down_cooldown"]) != c["scale_down_cooldown"] or c["scale_down_cooldown"] < 0:
        raise ValueError("scale_down_cooldown must be a non-negative integer")
    for k in ("reward_clip", "grad_clip", "huber_beta"):
        if c[k] is not None and not c[k] > 0:
            raise ValueError(f"{k} must be positive")
    for k in ("lr", "gamma"):
        if c[k] is not None and not c[k] > 0:
            raise ValueError(f"{k} must be positive")
    return c


def eps_at(schedule, step, start, end, decay_steps):
    """Epsilon after `step` env steps. linear: start -> end over decay_steps, then flat. exp: geometric start -> end."""
    f = min(max(step / float(decay_steps), 0.0), 1.0)
    if schedule == "linear":
        return start + (end - start) * f
    if schedule == "exp":
        return start * (end / start) ** f
    raise ValueError(schedule)


def _wrap_optimizer(agent, grad_clip):
    import torch
    opt = agent.optimizer
    inner = opt.step

    def step(*a, **k):
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(agent.online_model.parameters(), float(grad_clip))
        agent.grad_updates += 1
        return inner(*a, **k)
    opt.step = step


def _common(agent, c):
    if c["memory_size"] is not None:
        agent.memory = deque(agent.memory, maxlen=int(c["memory_size"]))
    if c["lr"] is not None:
        agent.learning_rate = float(c["lr"])
        for g in agent.optimizer.param_groups:
            g["lr"] = float(c["lr"])
    if c["gamma"] is not None:
        agent.gamma = float(c["gamma"])
    agent.cfg = c
    agent.grad_updates = 0      # counts every optimizer step of ANY agent (per-sample DERP steps included)
    _wrap_optimizer(agent, c["grad_clip"])
    agent.ctl = Controller(c["decision_every"], c["scale_down_cooldown"])
    agent.gamma = agent.gamma ** agent.ctl.k         # one stored transition spans k env steps
    n = int(c["n_step"])
    agent.n_step = n
    if n > 1:
        agent.gamma = agent.gamma ** n
        g1 = agent.gamma ** (1.0 / n)
        buf = deque()
        _store = agent.remember

        def remember_n(s, a, r, s2, d):
            buf.append((s, a, r))
            if len(buf) == n:
                R = sum(g1 ** i * b[2] for i, b in enumerate(buf))
                _store(buf[0][0], buf[0][1], R, s2, d)
                buf.popleft()
            if d:                       # flush the tail: next state is terminal, no bootstrap
                while buf:
                    R = sum(g1 ** i * b[2] for i, b in enumerate(buf))
                    _store(buf[0][0], buf[0][1], R, s2, True)
                    buf.popleft()
        agent.remember = remember_n
    K = c["explore_repeat_max"]
    agent._held = [None, 0]
    agent.reset_exploration = lambda: agent._held.__setitem__(slice(None), [None, 0])
    if K is not None:
        _act = agent.act
        p_start = 2.0 / (K + 1)          # 1 / E[k]

        def act(state, *a, **k):
            if agent.epsilon <= 0.0:     # greedy / evaluation: untouched
                return _act(state, *a, **k)
            if agent._held[1] > 0:
                agent._held[1] -= 1
                return agent._held[0]
            if np.random.rand() < agent.epsilon * p_start:
                act_ = random.randrange(agent.action_size)
                agent._held[:] = [act_, int(np.random.randint(1, K + 1)) - 1]
                return act_
            saved, agent.epsilon = agent.epsilon, 0.0
            try:
                return _act(state, *a, **k)
            finally:
                agent.epsilon = saved
        agent.act = act
    if c["reward_clip"] is not None or c["reward_transform"] is not None:
        _remember = agent.remember
        lim = None if c["reward_clip"] is None else float(c["reward_clip"])
        sym = c["reward_transform"] == "symlog"

        def remember(s, a, r, s2, d):
            if sym:
                r = float(np.sign(r) * np.log1p(abs(r)))
            if lim is not None:
                r = float(np.clip(r, -lim, lim))
            _remember(s, a, r, s2, d)
        agent.remember = remember
    if c["huber_beta"] is not None:
        import torch.nn as nn
        agent.criterion = nn.SmoothL1Loss(beta=float(c["huber_beta"]))
    if c["eps_start"] is not None:
        agent.epsilon = float(c["eps_start"])
    if c["eps_schedule"] != "per_replay":
        agent.epsilon_min = float(c["eps_end"] if c["eps_end"] is not None else agent.epsilon_min)
        agent.epsilon_decay = 1.0           # the frozen per-replay decay is switched off
        agent.eps_start = float(c["eps_start"] if c["eps_start"] is not None else agent.epsilon)
        agent.eps_end = agent.epsilon_min
    elif c["eps_end"] is not None:
        agent.epsilon_min = float(c["eps_end"])


class Controller:
    """Agent-side control layer (part of the policy): decision period k and scale-down cooldown m. Actions: 0 down,
    1 hold, 2 up."""

    def __init__(self, k=1, m=0):
        self.k, self.m, self.cd = int(k), int(m), 0

    def reset(self):
        self.cd = 0

    def convert(self, a):
        """Proposed -> executable action (scale-down becomes hold while the cooldown runs)."""
        return 1 if (a == 0 and self.cd > 0) else int(a)

    def executed(self, a):
        """Register the action actually sent to the env at one step."""
        self.cd = self.m if a != 1 and self.m > 0 else max(self.cd - 1, 0)


def macro_step(agent, sim, window, state_of, scale, budget, on_step=None, learn=True, n_dec=0, bs=16, every=1):
    """One decision: act, apply the executed action, hold for k-1 more env steps (or until `budget` env steps /
    episode end). Returns (next_window, done, env_steps_used). With learn=True the transition is stored and
    agent.replay runs (k=1, m=0 reproduces the plain act/step/remember/replay loop exactly). on_step(a, r, info)
    is called for every env step."""
    ctl = agent.ctl
    s0 = state_of(window)
    a = ctl.convert(agent.act(s0))
    R, used, done, w = 0.0, 0, False, window
    for j in range(ctl.k):
        ae = a if j == 0 else 1
        w, r, done, info = sim.step(ae)
        ctl.executed(ae)
        R += r
        used += 1
        if on_step:
            on_step(ae, r, info)
        if done or used >= budget:
            break
    if learn and (done or used == ctl.k):
        agent.remember(s0, a, R * scale, state_of(w), done)
        if len(agent.memory) > bs and n_dec % every == 0:
            agent.replay(bs)
    return w, done, used


def set_epsilon(agent, env_step):
    """Drive the env-step epsilon schedule (no-op for per_replay). Call before agent.act()."""
    c = agent.cfg
    if c["eps_schedule"] != "per_replay":
        agent.epsilon = eps_at(c["eps_schedule"], env_step, agent.eps_start, agent.eps_end, c["eps_decay_steps"])


class AdapterNet:
    """Built lazily (needs torch): input Linear(6->6) in front of a myLSTM trunk. Initialised to the affine
    'pretrain' mapping of transfer_map (so it starts where input_map="pretrain" would), then trained with the net."""


def _adapter_class():
    import torch
    import torch.nn as nn

    class WithInputAdapter(nn.Module):
        def __init__(self, net, n_in=None, identity=False):
            super().__init__()
            from tuning import transfer_map
            d = net.lstm.input_size
            n_in = n_in or d
            self.adapter = nn.Linear(n_in, d)       # n_in > d: extra (demand) columns start at 0
            W, b = (np.eye(d), np.zeros(d)) if identity else transfer_map.affine_pretrain(d)
            with torch.no_grad():
                self.adapter.weight.zero_()
                self.adapter.weight[:, :d] = torch.tensor(W, dtype=torch.float32)
                self.adapter.bias.copy_(torch.tensor(b, dtype=torch.float32))
            self.net = net

        def forward(self, x):
            return self.net(self.adapter(x))
    return WithInputAdapter


def make_tuned_lstm(pretrained=None, cfg=None, input_adapter=False, n_features=None):
    """n_features (default = the 6 base features): with input_adapter the trunk keeps its 6 inputs and the adapter is
    Linear(n_features -> 6); without it the LSTM itself takes n_features inputs (scratch only)."""
    import torch
    import torch.nn as nn
    import agents
    c = check_cfg(cfg or {})
    n_in = n_features or agents.N_FEATURES
    path_ckpt = isinstance(pretrained, str)     # state_dict from tuning/pretrain_source.py (V2 features, 6 inputs)
    if n_in != agents.N_FEATURES and not input_adapter:
        if pretrained is not None:
            raise ValueError("extra features with a pretrained trunk need input_map='adapter'")
        agent = _scratch_lstm(n_in)
    else:
        if path_ckpt:
            agent = _scratch_lstm(agents.N_FEATURES)
            sd = torch.load(pretrained)
            agent.online_model.load_state_dict(sd)
            agent.target_model.load_state_dict(sd)
        else:
            agent = agents.make_lstm(pretrained)
        agent.state_size = n_in
    if c["target_update_every"] is not None:
        agent.target_update_every = int(c["target_update_every"])
    if c["reset_head"]:
        if pretrained is None:
            raise ValueError("reset_head needs a pretrained model")
        fresh = nn.Linear(agent.online_model.fc.in_features, agent.online_model.fc.out_features)
        agent.online_model.fc.load_state_dict(fresh.state_dict())
        agent.update_target_model()
    if input_adapter:
        import torch
        cls = _adapter_class()
        agent.online_model = cls(agent.online_model, n_in, identity=path_ckpt)
        agent.target_model = cls(agent.target_model, n_in, identity=path_ckpt)
        agent.update_target_model()
        agent.optimizer = torch.optim.Adam(agent.online_model.parameters(), lr=agent.learning_rate)
    _common(agent, c)
    return agent


def _scratch_lstm(n_in):
    import agents
    mod = agents._import_from(agents.LSTM_DIR, "agentHuber", drop=("networks",))
    return mod.DDQNAgentLSTM(n_in, agents.N_ACTIONS, agents.WINDOW, pretrained_model=None)


def make_tuned_derp(cfg=None, n_features=None):
    import torch
    import torch.nn as nn
    import agents
    c = check_cfg(cfg or {})
    if n_features and n_features != agents.N_FEATURES:
        mod = agents._import_from(agents.DENSE_DIR, "ddqnagent", drop=("environment", "network", "ddqnagent"))
        base = mod.DDQNAgent(n_features, agents.N_ACTIONS)
    else:
        base = agents.make_derp()
    cls = type(base)

    class TunedDERP(cls):
        def compute_targets(self, rewards, next_states, dones):
            with torch.no_grad():
                a_star = self.online_model(next_states).argmax(dim=1, keepdim=True)
                q_next = self.target_model(next_states).gather(1, a_star).squeeze(1)
                return rewards + self.gamma * (1.0 - dones) * q_next

        def replay(self, batch_size):
            if not self.cfg["batched_double_dqn"]:
                super().replay(batch_size)            # frozen per-sample path, target synced every replay
                return
            if len(self.memory) < batch_size:
                return
            import random
            mb = random.sample(self.memory, batch_size)
            st = torch.tensor(np.array([m[0] for m in mb]), dtype=torch.float32)
            ac = torch.tensor([m[1] for m in mb], dtype=torch.int64).unsqueeze(1)
            rw = torch.tensor([m[2] for m in mb], dtype=torch.float32)
            nx = torch.tensor(np.array([m[3] for m in mb]), dtype=torch.float32)
            dn = torch.tensor([float(m[4]) for m in mb], dtype=torch.float32)
            y = self.compute_targets(rw, nx, dn)
            self.online_model.train()
            q = self.online_model(st).gather(1, ac).squeeze(1)
            loss = self.criterion(q, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()               # increments grad_updates (wrapped optimizer)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            every = self.cfg["target_update_every"] or 100
            if self.grad_updates % every == 0:
                self.update_target_model()
            self.loss_history.append(loss.item())

    agent = base
    agent.__class__ = TunedDERP
    _common(agent, c)
    return agent


def q_stats(agent, batch_size, rng):
    """mean |Q| of the online net and mean |TD| on a replay batch drawn with a private rng (training streams untouched)."""
    import torch
    n = len(agent.memory)
    if n < max(batch_size, 2):
        return {}
    idx = rng.choice(n, size=batch_size, replace=False)
    mem = [agent.memory[int(i)] for i in idx]
    st = torch.tensor(np.array([m[0] for m in mem]), dtype=torch.float32)
    nx = torch.tensor(np.array([m[3] for m in mem]), dtype=torch.float32)
    shape = getattr(agent, "sequence_length", None)
    if shape:
        st = st.reshape(batch_size, agent.sequence_length, agent.state_size)
        nx = nx.reshape(batch_size, agent.sequence_length, agent.state_size)
    a = torch.tensor([m[1] for m in mem], dtype=torch.int64).unsqueeze(1)
    rw = torch.tensor([m[2] for m in mem], dtype=torch.float32)
    dn = torch.tensor([float(m[4]) for m in mem], dtype=torch.float32)
    was = agent.online_model.training
    agent.online_model.eval()
    try:
        with torch.no_grad():
            qall = agent.online_model(st)
            y = agent.compute_targets(rw, nx, dn) if hasattr(agent, "compute_targets") else \
                rw + agent.gamma * (1.0 - dn) * agent.target_model(nx).max(dim=1).values
            td = (y - qall.gather(1, a).squeeze(1)).abs()
    finally:
        agent.online_model.train(was)
    out = dict(mean_abs_q=float(qall.abs().mean()), mean_abs_td=float(td.mean()))
    for i, v in enumerate(qall.mean(dim=0).tolist()):
        out[f"mean_q_a{i}"] = float(v)
    return out
