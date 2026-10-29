"""Dinos V.2 architecture search (council C-20261021-02; researcher scope change R-20261021-004, 2026-10-21).

Every network here is a Q-network for the fixed-agent mechanics of agent_lstm_fixed.FixedLSTMAgent (true detached
Double DQN, terminal masking, gamma 0.98, replay 5000, Huber, 16 sequential single-sample updates per replay, target
sync 250 replays, Adam); ONLY the network differs. Input = the (batch, 10, F) window the agent already gets (scaled by
--obs-scale on like every other arm).

Generic spec string -> network (SpecQ), tokens separated by '-', applied in order; a linear Q head (n actions) is
appended automatically:
  cK   Conv1d(width -> K, kernel 3, padding 1) + ReLU over the time axis (sequence -> sequence)
  cKkNrR causal Conv1d(width -> K, kernel N, dilation R, left padding (N-1)R, no right padding) + ReLU
  eK   per-timestep Dense embedding Linear(width -> K) + ReLU (sequence -> sequence)
  lK   one LSTM layer, hidden K, batch_first, from a zero (h, c) on EVERY forward call (sequence -> sequence)
  cat  last timestep of the sequence features || newest frame x[:, -1, :] (the same scaled observation)
  attcat attention pooling over the sequence: a = softmax_t(h_t . w) (w bias-free, zero-padded frames after a
       reset masked out), sum_t a_t h_t || newest frame
  trendcatN last timestep || newest frame || [L_t - L_{t-1}, (L_t - L_{t-N}) / N] of the scaled load (feature 0),
       causal, 0 where the past frame is padding
  dK   Dense K + ReLU (on a sequence, its last timestep is taken first)
  mW   (alone) the matched dense control: agent_dense_fixed.FlatMLP = MyDenseNet over the flattened 10 x F window,
       W -> W hidden (the mlp_hist10 network and DenseFixedAgent mechanics); params = W^2 + (10F + 5) W + 3.
  mWtN (alone) FlatMLPTrend: the same MLP over the flattened window || the two trendcatN features (V2Agent
       mechanics, identical to DenseFixedAgent); params = W^2 + (10F + 7) W + 3.
Modules are created in spec order (online net first, then target net), so the global torch stream consumption is fixed
by the spec. To add an architecture for try-and-error, add one line to ARCHS (and a matched MLP if the size differs).

Registered architectures (params for F = 7, 3 actions):
  a1        l64-cat-d64-d32           25,475   Dinos V.2-A (Codex primary)
  a1h       l128-cat-d128-d128       104,451   Dinos V.2-H (DERP head, Claude)
  a2        l64-l32-cat-d64-d32       35,971   2 LSTM layers
  a3        c32-l64-cat-d64-d32       32,579   Conv1d front
  a4        e32-l64-cat-d64-d32       32,131   per-step Dense embedding, LSTM as 2nd layer
  a5        c32k3r2-l64-cat-d64-d32   32,579   causal dilated Conv1d front (branch 2, c18_astra_design.md)
  a6        e32-l64-attcat-d64-d32    32,195   a4 + attention readout over the LSTM outputs
  a8        e32-l64-trendcat5-d64-d32 32,259   a4 + explicit causal load trends
  a9        e32-l64-cat-d128-d128     51,459   a4 with the wider head of a1h
  mlp_a1    m127                      25,657   matched to a1 (+0.71%)
  mlp_a1h   m288                     104,547   matched to a1h (+0.09%)
  mlp_a234  m151                      34,129   one control for the 32-36k class (a2 -5.1%, a3 +4.8%, a4 +6.2%)
  mlp_a2    m156                      36,039   exact match for a2 (+0.19%; registered, not in the default screen)
  mlp_a34   m146                      32,269   exact match for a3/a4/a5/a6 (-0.95% / +0.43% / -0.95% / +0.23%)
  mlp_a8    m145t5                    32,193   exact match for a8 (-0.20%), same two trend features
  mlp_a9    m192                      51,267   exact match for a9 (-0.37%)
(tests/test_v2.py asserts these counts.)

Agents: make_scratch(arch, ...) -> lr 1e-4, epsilon 1 -> 0.05 (agent_lstm_fixed.make_scratch). As in
agent_dense_fixed, the myLSTM networks FixedLSTMAgent.__init__ builds are created inside torch.random.fork_rng and
discarded. Transfer = the existing orig-family transfer protocol of c5_protocol (source pretraining with a scratch
agent of the SAME arch, fresh scratch agent loading the online weights into online + target, fresh replay and Adam).
"""
import re

import torch
import torch.nn as nn

from . import agent_dense_fixed as DFX
from . import agent_lstm_fixed as FX
from .network_dense import MyDenseNet

ARCHS = {"a1": "l64-cat-d64-d32", "a1h": "l128-cat-d128-d128", "a2": "l64-l32-cat-d64-d32",
         "a3": "c32-l64-cat-d64-d32", "a4": "e32-l64-cat-d64-d32",
         "a5": "c32k3r2-l64-cat-d64-d32", "a6": "e32-l64-attcat-d64-d32", "a8": "e32-l64-trendcat5-d64-d32",
         "a9": "e32-l64-cat-d128-d128",
         "mlp_a1": "m127", "mlp_a1h": "m288", "mlp_a234": "m151", "mlp_a2": "m156", "mlp_a34": "m146",
         "mlp_a8": "m145t5", "mlp_a9": "m192"}
MATCH = {"mlp_a1": ("a1",), "mlp_a1h": ("a1h",), "mlp_a234": ("a2", "a3", "a4"), "mlp_a2": ("a2",),
         "mlp_a34": ("a3", "a4", "a5", "a6"), "mlp_a8": ("a8",), "mlp_a9": ("a9",)}
LABEL = {"a1": "Dinos V.2-A (LSTM64 -> concat obs -> 64 -> 32)", "a1h": "Dinos V.2-H (LSTM128 -> concat obs -> 128 -> 128)",
         "a2": "Dinos V.2 a2 (LSTM64 -> LSTM32 -> concat obs -> 64 -> 32)",
         "a3": "Dinos V.2 a3 (Conv1d32 -> LSTM64 -> concat obs -> 64 -> 32)",
         "a4": "Dinos V.2 a4 (step Dense32 -> LSTM64 -> concat obs -> 64 -> 32)",
         "a5": "Dinos V.2 a5 (causal dilated Conv1d32 k3 r2 -> LSTM64 -> concat obs -> 64 -> 32)",
         "a6": "Dinos V.2 a6 (step Dense32 -> LSTM64 -> attention pooling || obs -> 64 -> 32)",
         "a8": "Dinos V.2 a8 (step Dense32 -> LSTM64 -> concat obs + load trends (1, 5) -> 64 -> 32)",
         "a9": "Dinos V.2 a9 (step Dense32 -> LSTM64 -> concat obs -> 128 -> 128)",
         "mlp_a8": "MLP control m145 over the window + load trends (1, 5)"}
LOAD_IDX = 0                # env_orig layout [load, capacity, cpu %, mem %, latency, pods(, pending)]
_CONV = re.compile(r"^c(\d+)k(\d+)(?:r(\d+))?$")
_MLP = re.compile(r"^m(\d+)(?:t(\d+))?$")
_TREND = re.compile(r"^trendcat(\d+)$")


def is_mlp(arch):
    return ARCHS[arch].startswith("m")


def parse(spec):
    """spec -> list of op tuples; raises ValueError on a malformed spec. Tuples: ("c", K) symmetric conv (pad 1),
    ("cc", K, kernel, dilation) causal conv, ("e"|"l"|"d", K), ("cat", 0), ("att", 0), ("trend", n), ("m", W),
    ("mt", W, n)."""
    toks = [t for t in str(spec).split("-") if t]
    if len(toks) == 1 and toks[0][0] == "m":
        mm = _MLP.match(toks[0])
        if not mm or int(mm.group(1)) < 1 or (mm.group(2) is not None and int(mm.group(2)) < 1):
            raise ValueError(f"{spec}: bad MLP token")
        if mm.group(2) is None:
            return [("m", int(mm.group(1)))]
        return [("mt", int(mm.group(1)), int(mm.group(2)))]
    ops, seq = [], True
    for t in toks:
        tm = _TREND.match(t)
        if t in ("cat", "attcat") or tm:
            if not seq:
                raise ValueError(f"{spec}: {t!r} after the sequence was reduced")
            if tm and int(tm.group(1)) < 1:
                raise ValueError(f"{spec}: bad token {t!r}")
            ops.append(("cat", 0) if t == "cat" else ("att", 0) if t == "attcat" else ("trend", int(tm.group(1))))
            seq = False
            continue
        cm = _CONV.match(t)
        if cm:
            K, kern, dil = int(cm.group(1)), int(cm.group(2)), int(cm.group(3) or 1)
            if min(K, kern, dil) < 1 or not seq:
                raise ValueError(f"{spec}: bad token {t!r}")
            ops.append(("cc", K, kern, dil))
            continue
        op, k = t[0], t[1:]
        if op not in "celd" or not k.isdigit() or int(k) < 1:
            raise ValueError(f"{spec}: bad token {t!r}")
        if op in "cel" and not seq:
            raise ValueError(f"{spec}: sequence op {t!r} after the sequence was reduced")
        if op == "d":
            seq = False
        ops.append((op, int(k)))
    if not ops:
        raise ValueError("empty spec")
    return ops


def valid_rows(x):
    """(batch, T) bool: True for real frames. After a reset the window is zero-padded at the front (c5_protocol.
    window0); a real frame is never all-zero (pods >= 1). The newest frame always counts as real, so an all-zero
    window (terminal next state) cannot produce an empty softmax."""
    v = x.abs().sum(-1) > 0
    v[:, -1] = True
    return v


def trend_features(x, n, idx=LOAD_IDX):
    """(batch, 2): [L_t - L_{t-1}, (L_t - L_{t-n}) / n] of the (scaled) load column of the window, causal (only rows
    <= t), 0 where the needed past frame is padding."""
    T = x.shape[1]
    if T <= n:
        raise ValueError(f"trend window {n} needs more than {T} frames")
    v, L = valid_rows(x), x[:, :, idx]
    d1 = torch.where(v[:, -2], L[:, -1] - L[:, -2], torch.zeros_like(L[:, -1]))
    dn = torch.where(v[:, -1 - n], (L[:, -1] - L[:, -1 - n]) / n, torch.zeros_like(L[:, -1]))
    return torch.stack([d1, dn], dim=1)


class FlatMLPTrend(nn.Module):
    """FlatMLP (flattened window, MyDenseNet) whose input also gets the two trend_features (matched control of a8)."""

    def __init__(self, input_size, sequence_length, hidden_size, output_size, n):
        super().__init__()
        self.input_size, self.sequence_length, self.n = input_size, sequence_length, int(n)
        self.net = MyDenseNet(input_size * sequence_length + 2, hidden_size, output_size)
        self.hidden = None

    def init_hidden(self):
        return None

    def forward(self, x):
        return self.net(torch.cat([x.reshape(x.shape[0], -1), trend_features(x, self.n)], dim=1))


class SpecQ(nn.Module):
    """Sequence Q-network built from a spec string (see the module docstring). init_hidden/hidden exist only for
    interface compatibility with the evaluation code (it sets m.hidden; unused, every call starts from zero)."""

    def __init__(self, spec, input_size, output_size):
        super().__init__()
        self.spec, self.input_size = str(spec), int(input_size)
        self.ops = parse(spec)
        if self.ops[0][0] in ("m", "mt"):
            raise ValueError("mW specs are built by agent_dense_fixed.FlatMLP / FlatMLPTrend")
        self.layers = nn.ModuleList()
        w = self.input_size
        for o in self.ops:
            op, k = o[0], o[1]
            if op == "c":
                self.layers.append(nn.Conv1d(w, k, kernel_size=3, padding=1))
            elif op == "cc":            # causal: left padding (kernel - 1) * dilation, no right padding
                self.layers.append(nn.Conv1d(w, k, kernel_size=o[2], dilation=o[3]))
            elif op == "e":
                self.layers.append(nn.Linear(w, k))
            elif op == "l":
                self.layers.append(nn.LSTM(w, k, num_layers=1, batch_first=True))
            elif op == "d":
                self.layers.append(nn.Linear(w, k))
            elif op == "att":           # bias-free score vector w: softmax_t(h_t . w)
                self.layers.append(nn.Linear(w, 1, bias=False))
                k = w + self.input_size
            elif op == "trend":
                self.layers.append(nn.Identity())
                k = w + self.input_size + 2
            else:          # cat
                self.layers.append(nn.Identity())
                k = w + self.input_size
            w = k
        self.q = nn.Linear(w, output_size)
        self.relu = nn.ReLU()
        self.hidden = None
        self.last_attention = None

    def init_hidden(self):
        return None

    def forward(self, x):
        h = x
        for o, m in zip(self.ops, self.layers):
            op, k = o[0], o[1]
            if op == "c":
                h = self.relu(m(h.transpose(1, 2))).transpose(1, 2)
            elif op == "cc":
                pad = (o[2] - 1) * o[3]
                h = self.relu(m(nn.functional.pad(h.transpose(1, 2), (pad, 0)))).transpose(1, 2)
            elif op == "e":
                h = self.relu(m(h))
            elif op == "l":
                h0 = torch.zeros(1, h.shape[0], k, dtype=h.dtype)
                h, _ = m(h, (h0, h0.clone()))
            elif op == "att":
                s = m(h).squeeze(-1).masked_fill(~valid_rows(x), float("-inf"))
                a = torch.softmax(s, dim=1)
                self.last_attention = a.detach()
                h = torch.cat([(a.unsqueeze(-1) * h).sum(1), x[:, -1, :]], dim=1)
            elif op == "trend":
                h = torch.cat([h[:, -1, :], x[:, -1, :], trend_features(x, k)], dim=1)
            elif op == "cat":
                h = torch.cat([h[:, -1, :], x[:, -1, :]], dim=1)
            else:
                if h.dim() == 3:
                    h = h[:, -1, :]
                h = self.relu(m(h))
        if h.dim() == 3:
            h = h[:, -1, :]
        return self.q(h)


def build_net(arch, state_size, sequence_length, action_size):
    ops = parse(ARCHS[arch])
    if ops[0][0] == "m":
        return DFX.FlatMLP(state_size, sequence_length, ops[0][1], action_size)
    if ops[0][0] == "mt":
        return FlatMLPTrend(state_size, sequence_length, ops[0][1], action_size, ops[0][2])
    return SpecQ(ARCHS[arch], state_size, action_size)


def input_size(arch, state_size, sequence_length=10):
    """Network input width recorded in the result JSON (MLP: flattened window [+ 2 trend features])."""
    op = parse(ARCHS[arch])[0][0]
    return sequence_length * state_size + (2 if op == "mt" else 0) if op in ("m", "mt") else state_size


def architecture(arch, state_size, sequence_length=10, action_size=3):
    """Human-readable layer string (for the result JSON / dashboard)."""
    ops = parse(ARCHS[arch])
    if ops[0][0] in ("m", "mt"):
        n, W = sequence_length * state_size, ops[0][1]
        if ops[0][0] == "mt":
            t = ops[0][2]
            return (f"flatten window[{sequence_length}x{state_size}] || load trends(L_t-L_t-1, (L_t-L_t-{t})/{t}) "
                    f"-> {n + 2} -> {W} ReLU -> {W} ReLU -> {action_size}")
        return f"flatten window[{sequence_length}x{state_size}] -> {n} -> {W} ReLU -> {W} ReLU -> {action_size}"
    parts, w = [f"window[{sequence_length}x{state_size}]"], state_size
    for o in ops:
        op, k = o[0], o[1]
        if op == "c":
            parts.append(f"Conv1d({w},{k},k3,pad1) ReLU")
        elif op == "cc":
            parts.append(f"causal Conv1d({w},{k},k{o[2]},dil{o[3]},left pad {(o[2] - 1) * o[3]}) ReLU")
        elif op == "e":
            parts.append(f"per-step Dense({w},{k}) ReLU")
        elif op == "l":
            parts.append(f"LSTM({w},{k},1 layer, zero h/c per window)")
        elif op == "cat":
            parts.append(f"concat(last[{w}], newest obs[{state_size}])")
            k = w + state_size
        elif op == "att":
            parts.append(f"concat(attention pool[{w}] (softmax_t(h_t.w), w bias-free, padding masked), "
                         f"newest obs[{state_size}])")
            k = w + state_size
        elif op == "trend":
            parts.append(f"concat(last[{w}], newest obs[{state_size}], load trends[2] (L_t-L_t-1, (L_t-L_t-{k})/{k}))")
            k = w + state_size + 2
        else:
            parts.append(f"Dense({w},{k}) ReLU")
        w = k
    return " -> ".join(parts + [f"Dense({w},{action_size})"])


def architecture_str(arch, state_size, sequence_length=10, action_size=3, dueling=False):
    """architecture() with the cycle-19 dueling head spelled out (identical string when dueling is off)."""
    s = architecture(arch, state_size, sequence_length, action_size)
    if not dueling:
        return s
    head = s.rpartition(" -> ")[0]
    ops = parse(ARCHS[arch])
    w = ops[0][1] if ops[0][0] == "m" else s.rpartition(" -> ")[2].split("(")[1].split(",")[0]
    return f"{head} -> dueling[V: Dense({w},1); A: Dense({w},{action_size}); Q = V + A - mean A]"


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class V2Agent(FX.FixedLSTMAgent):
    """FixedLSTMAgent with a SpecQ network."""

    def __init__(self, arch, state_size, action_size, sequence_length, learning_rate, epsilon_min,
                 target_sync=FX.DEFAULT_TARGET_SYNC):
        if arch not in ARCHS or parse(ARCHS[arch])[0][0] == "m":
            raise ValueError(f"unknown V.2 sequence arch {arch!r}")
        with torch.random.fork_rng(devices=[]):
            super().__init__(state_size, action_size, sequence_length, learning_rate, epsilon_min, target_sync)
        self.variant = arch
        if parse(ARCHS[arch])[0][0] == "mt":
            self.dense_hidden = parse(ARCHS[arch])[0][1]
        self.online_model = build_net(arch, state_size, sequence_length, action_size)
        self.target_model = build_net(arch, state_size, sequence_length, action_size)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)


def make_scratch(arch, state_size, action_size, seq, target_sync=FX.DEFAULT_TARGET_SYNC, ddqn=None):
    """Counterpart of agent_lstm_fixed.make_scratch: lr 1e-4, epsilon 1 -> 0.05. MLP archs = agent_dense_fixed.
    DenseFixedAgent (the mlp_hist10 agent) with hidden width W. ddqn: kwargs of FixedLSTMAgent.set_ddqn (cycle 19;
    the same head/replay/return options apply to the sequence arch and its matched MLP)."""
    if arch not in ARCHS:
        raise ValueError(f"unknown V.2 arch {arch!r}")
    if parse(ARCHS[arch])[0][0] == "m":
        a = DFX.DenseFixedAgent(state_size, action_size, seq, 0.0001, 0.05, target_sync,
                                hidden=parse(ARCHS[arch])[0][1])
        a.variant = arch
    else:
        a = V2Agent(arch, state_size, action_size, seq, 0.0001, 0.05, target_sync)
    if ddqn:
        a.set_ddqn(**ddqn)
    return a


def hyperparameters(agent):
    return dict(gamma=float(agent.gamma), replay_capacity=int(agent.memory.maxlen), batch_size=16,
                updates_per_replay=16, loss="SmoothL1 (Huber)", optimizer="Adam", learning_rate=float(agent.learning_rate),
                epsilon_min=float(agent.epsilon_min), target_sync_replays=int(agent.target_sync),
                double_dqn="detached (online argmax, target value), terminal masked",
                **({"ddqn_addons": agent.ddqn_config()} if (agent.n_step != 1 or agent.dueling or agent.per) else {}))
