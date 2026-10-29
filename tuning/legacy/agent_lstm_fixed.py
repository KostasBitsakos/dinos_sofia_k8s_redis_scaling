"""Cycle 6 'fixed' LSTM agent (NOT a verbatim port; design .lab/tuning/cycles/c5_codex.md + c5_synthesis.md).

Same as the verbatim agentHuber.DDQNAgent (scratch) / lstm_transfer agentHuber.DDQNAgentLSTM (transfer) except for three
defects, each supported by the DInos paper text (DDQN; Algorithm 1 line 22 periodic sync):
  1. hidden state: every forward pass starts from a zero (h, c) for each 10x6 window (acting and replay alike); the
     verbatim myLSTM carries self.hidden across unrelated calls (networks_lstm.py:25).
  2. Double-DQN target, detached: a* = argmax_a Q_online(s', a), y = r + gamma * Q_target(s', a*), computed under
     torch.no_grad() (verbatim: plain max over the target network).
  3. target network synced every `target_sync` replay() calls (default 250) instead of after every replay.
Unchanged: myLSTM(6, 128, 3 layers, 3) weights/state_dict layout, inputs, reward, SmoothL1 (Huber) loss, Adam with the
verbatim learning rates, epsilon schedules (scratch 1 -> 0.05, transfer 0.2 -> 0.1 after loading; x0.995 per replay),
gamma 0.98, memory 5000, 16 sequential single-sample updates per replay (target_f = online(s) with the action entry
replaced), and the RNG consumption pattern of act()/replay() (np.random.rand, random.randrange, random.sample).

Cycle 19 (opt-in Double-DQN add-ons, .lab/tuning/cycles/c19_policy_strategy.md; all off by default, so the default
agent is bit-identical): set_ddqn(n_step, dueling, per, ...) right after construction (before remember/load_weights).
  n_step N  remember() accumulates R = sum_k gamma^k r_k over the last N transitions (rewards as stored, i.e. after
            --reward-div) and stores (s_t, a_t, R, s_{t+N}, done, gamma^N); on done every pending shorter return is
            flushed as terminal. Target: R + gamma^N * Q_target(s', argmax_a Q_online(s', a)) (Double DQN kept).
  dueling   the final Q Linear(w, A) becomes DuelingHead: V = Linear(w, 1), A = Linear(w, A), Q = V + A - mean_a A
            (same hidden width; +w+1 parameters). Applied to every network kind (myLSTM fc, SpecQ q, FlatMLP fc3).
  per       proportional prioritized replay (Schaul et al. 2016): P(i) ~ p_i^0.6, p_i = |TD| + 1e-6, new transitions
            get the max priority seen, stratified sampling of the batch, then the batch (transitions, indices and IS
            weights together) is shuffled with the same private RNG so the 16 sequential updates are not in ring
            order; IS weights (N P(i))^-beta normalised by the buffer-wide maximum weight (so <= 1); beta linear
            0.4 -> 1.0 in the training tick of the phase (per_progress = t / ticks, set by c5_protocol.train_episodic,
            so the replay at the phase's last tick uses exactly 1.0; without it: replay k of per_anneal_replays); each of the
            16 sequential Huber updates is multiplied by its IS weight; priorities = |TD of the replayed action| taken
            before that sample's update. Sampling uses a private RandomState (per_seed) -> deterministic under the seed.
  phase end (n_step > 1): flush_nstep(state) stores the pending transitions with their shorter returns BOOTSTRAPPED
            (done False, gamma^k) from the last state; truncation is never marked terminal (train_episodic calls it).
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .networks_lstm import myLSTM

DEFAULT_TARGET_SYNC = 250
PER_ALPHA, PER_BETA0, PER_EPS = 0.6, 0.4, 1e-6


class DuelingHead(nn.Module):
    """Q = V + A - mean_a A from the same features the plain Q Linear used."""

    def __init__(self, in_features, n_actions):
        super().__init__()
        self.value = nn.Linear(in_features, 1)
        self.advantage = nn.Linear(in_features, n_actions)

    def forward(self, h):
        a = self.advantage(h)
        return self.value(h) + a - a.mean(dim=-1, keepdim=True)


def _q_head(model):
    """(parent module, attribute) of the final Q Linear: SpecQ.q, FlatMLP.net.fc3, myLSTM/ResetLSTM.fc."""
    if isinstance(getattr(model, "q", None), nn.Linear):
        return model, "q"
    if isinstance(getattr(getattr(model, "net", None), "fc3", None), nn.Linear):
        return model.net, "fc3"
    if isinstance(getattr(model, "fc", None), nn.Linear):
        return model, "fc"
    raise ValueError(f"no plain Q Linear head on {type(model).__name__}")


def make_dueling(model):
    parent, name = _q_head(model)
    lin = getattr(parent, name)
    setattr(parent, name, DuelingHead(lin.in_features, lin.out_features))
    return model


class PrioritizedReplay:
    """Proportional prioritized replay, ring buffer of `maxlen` (O(N) cumsum sampling; N <= 5000)."""

    def __init__(self, maxlen, alpha=PER_ALPHA, eps=PER_EPS, seed=None):
        self.maxlen, self.alpha, self.eps = int(maxlen), float(alpha), float(eps)
        self.data, self.pos = [], 0
        self.prio = np.zeros(self.maxlen, dtype=np.float64)       # raw priorities p_i = |TD| + eps
        self.max_p = 1.0
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.data)

    def __iter__(self):       # oldest first, like the deque
        n = len(self.data)
        start = self.pos if n == self.maxlen else 0
        return iter([self.data[(start + i) % self.maxlen] for i in range(n)])

    def append(self, item):
        if len(self.data) < self.maxlen:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.prio[self.pos] = self.max_p
        self.pos = (self.pos + 1) % self.maxlen

    def probabilities(self):
        pa = self.prio[:len(self.data)] ** self.alpha
        return pa / pa.sum()

    def sample(self, batch_size, beta):
        n = len(self.data)
        pa = self.prio[:n] ** self.alpha
        total = pa.sum()
        u = (np.arange(batch_size) + self.rng.random_sample(batch_size)) / batch_size * total
        idx = np.minimum(np.searchsorted(np.cumsum(pa), u, side="right"), n - 1)
        idx = idx[self.rng.permutation(batch_size)]     # stratified -> ascending slots; randomise the update order
        w = (n * pa[idx] / total) ** (-beta)
        w_max = (n * pa.min() / total) ** (-beta)
        return idx, [self.data[i] for i in idx], w / w_max

    def update(self, idx, td):
        p = np.abs(np.asarray(td, dtype=np.float64)) + self.eps
        self.prio[idx] = p
        self.max_p = max(self.max_p, float(p.max()))


class ResetLSTM(myLSTM):
    """myLSTM whose forward starts every call from zero hidden state (one per batch row). Same parameters/state_dict
    as myLSTM, so weights transfer both ways. self.hidden is kept (unused) for interface compatibility."""

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, dtype=x.dtype)
        lstm_out, _ = self.lstm(x, (h0, h0.clone()))
        return self.fc(lstm_out[:, -1, :])


class FixedLSTMAgent:
    n_step, dueling, per = 1, False, False          # cycle 19 add-ons, off unless set_ddqn() is called

    def __init__(self, state_size, action_size, sequence_length, learning_rate, epsilon_min,
                 target_sync=DEFAULT_TARGET_SYNC):
        if int(target_sync) < 1:
            raise ValueError("target_sync must be >= 1 replay")
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=5000)
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.target_sync = int(target_sync)
        self.n_replays = 0
        self.n_target_syncs = 0
        self.online_model = ResetLSTM(state_size, 128, 3, action_size)
        self.target_model = ResetLSTM(state_size, 128, 3, action_size)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.loss_history = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def set_ddqn(self, n_step=1, dueling=False, per=False, per_anneal_replays=None, per_seed=None):
        """Cycle 19 add-ons (module docstring). Call once, before remember()/load_weights()."""
        n_step = int(n_step)
        if n_step < 1:
            raise ValueError("n_step must be >= 1")
        if len(self.memory) or self.n_replays:
            raise RuntimeError("set_ddqn must be called before training")
        self.n_step, self._nbuf = n_step, deque()
        if dueling:
            self.dueling = True
            make_dueling(self.online_model)
            make_dueling(self.target_model)
            self.update_target_model()
            self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)
        if per:
            self.per = True
            self.per_anneal_replays = max(1, int(per_anneal_replays or 1))
            self.per_progress, self.per_beta_last = None, None
            self.memory = PrioritizedReplay(self.memory.maxlen, seed=per_seed)
        return self

    def ddqn_config(self):
        """The cycle-19 options for the result JSON."""
        d = dict(n_step=int(self.n_step), dueling=bool(self.dueling), per=bool(self.per))
        if self.per:
            d.update(per_alpha=PER_ALPHA, per_beta0=PER_BETA0, per_beta_end=1.0, per_eps=PER_EPS,
                     per_anneal="linear in t / phase ticks (1.0 at the last tick)",
                     per_sampling="proportional, stratified, batch order shuffled",
                     per_is_norm="buffer-wide max weight")
        return d

    def per_beta(self):
        """beta for the replay about to run: per_progress (t / ticks) if the training loop sets it, else the replay
        index k = n_replays (0-based) over per_anneal_replays - 1 (replay number per_anneal_replays gets 1.0)."""
        if self.per_progress is not None:
            f = float(self.per_progress)
        else:
            f = self.n_replays / max(1, self.per_anneal_replays - 1)
        return PER_BETA0 + (1.0 - PER_BETA0) * min(1.0, max(0.0, f))

    def flush_nstep(self, next_state):
        """Phase end without done: store pending n-step transitions with shorter bootstrapped returns (not terminal)."""
        while self.n_step > 1 and self._nbuf:
            self._emit(next_state, False)

    def load_weights(self, state_dict):
        self.online_model.load_state_dict(state_dict)
        self.target_model.load_state_dict(state_dict)

    def remember(self, state, action, reward, next_state, done):
        if self.n_step == 1:
            self.memory.append((state, action, reward, next_state, done))
            return
        self._nbuf.append((state, action, reward))
        if done:                                    # flush every pending (shorter) return as terminal
            while self._nbuf:
                self._emit(next_state, True)
        elif len(self._nbuf) == self.n_step:
            self._emit(next_state, False)

    def _emit(self, next_state, done):
        R = 0.0
        for k, (_, _, r) in enumerate(self._nbuf):
            R += (self.gamma ** k) * r
        s0, a0, _ = self._nbuf.popleft()
        self.memory.append((s0, a0, R, next_state, done, self.gamma ** (len(self._nbuf) + 1)))

    def _x(self, s):
        return torch.tensor(np.reshape(s, (1, self.sequence_length, self.state_size)), dtype=torch.float32)

    def act(self, state, verbose=False):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        self.online_model.eval()
        with torch.no_grad():
            q = self.online_model(self._x(state))
        return torch.argmax(q).item()

    def ddqn_target(self, reward, next_state, done, discount=None):
        """Detached Double-DQN target (python float / 0-dim tensor without grad). discount: gamma^n (n-step)."""
        if done:
            return reward
        with torch.no_grad():
            nx = self._x(next_state)
            self.online_model.eval()
            a_star = int(torch.argmax(self.online_model(nx)[0]).item())
            if discount is None:
                return reward + self.gamma * self.target_model(nx)[0][a_star]
            return reward + discount * self.target_model(nx)[0][a_star]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        if self.per:
            self.per_beta_last = self.per_beta()
            idx, minibatch, isw = self.memory.sample(batch_size, self.per_beta_last)
            tds = []
        else:
            minibatch = random.sample(self.memory, batch_size)
        batch_losses = []
        for k, tr in enumerate(minibatch):
            state, action, reward, next_state, done = tr[:5]
            target = self.ddqn_target(reward, next_state, done) if len(tr) == 5 else \
                self.ddqn_target(reward, next_state, done, tr[5])
            x = self._x(state)
            self.online_model.eval()
            with torch.no_grad():
                target_f = self.online_model(x)
                if self.per:
                    tds.append(float(target) - float(target_f[0][action]))
                target_f[0][action] = target
            self.online_model.train()
            self.optimizer.zero_grad()
            output = self.online_model(x)
            loss = self.criterion(output, target_f.detach())
            if self.per:
                loss = loss * float(isw[k])
            loss.backward()
            self.optimizer.step()
            batch_losses.append(loss.item())
        if self.per:
            self.memory.update(idx, tds)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.n_replays += 1
        if self.n_replays % self.target_sync == 0:
            self.update_target_model()
            self.n_target_syncs += 1
        self.loss_history.append(np.mean(batch_losses))


def make_scratch(state_size, action_size, seq, target_sync=DEFAULT_TARGET_SYNC, ddqn=None):
    """Counterpart of agentHuber.DDQNAgent: lr 1e-4, epsilon 1 -> 0.05. ddqn: kwargs of set_ddqn (cycle 19)."""
    a = FixedLSTMAgent(state_size, action_size, seq, 0.0001, 0.05, target_sync)
    if ddqn:
        a.set_ddqn(**ddqn)
    return a


def make_transfer(state_size, action_size, seq, state_dict, target_sync=DEFAULT_TARGET_SYNC, ddqn=None):
    """Counterpart of lstm_transfer DDQNAgentLSTM(pretrained_model=...): lr 5e-5, epsilon 0.2 -> 0.1."""
    a = FixedLSTMAgent(state_size, action_size, seq, 0.00005, 0.1, target_sync)
    if ddqn:
        a.set_ddqn(**ddqn)
    a.load_weights(state_dict)
    a.epsilon = 0.2
    a.pretrained_model = "in-memory"
    return a
