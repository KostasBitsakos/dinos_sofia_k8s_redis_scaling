"""Dinos V.2 architecture search (council C-20261021-02; researcher scope change R-20261021-004, 2026-10-21).

Every network here is a Q-network for the fixed-agent mechanics of agent_lstm_fixed.FixedLSTMAgent (true detached
Double DQN, terminal masking, gamma 0.98, replay 5000, Huber, 16 sequential single-sample updates per replay, target
sync 250 replays, Adam); ONLY the network differs. Input = the (batch, 10, F) window the agent already gets (scaled by
--obs-scale on like every other arm).

Generic spec string -> network (SpecQ), tokens separated by '-', applied in order; a linear Q head (n actions) is
appended automatically:
  cK   Conv1d(width -> K, kernel 3, padding 1) + ReLU over the time axis (sequence -> sequence)
  eK   per-timestep Dense embedding Linear(width -> K) + ReLU (sequence -> sequence)
  lK   one LSTM layer, hidden K, batch_first, from a zero (h, c) on EVERY forward call (sequence -> sequence)
  cat  last timestep of the sequence features || newest frame x[:, -1, :] (the same scaled observation)
  dK   Dense K + ReLU (on a sequence, its last timestep is taken first)
  mW   (alone) the matched dense control: agent_dense_fixed.FlatMLP = MyDenseNet over the flattened 10 x F window,
       W -> W hidden (the mlp_hist10 network and DenseFixedAgent mechanics); params = W^2 + (10F + 5) W + 3.
Modules are created in spec order (online net first, then target net), so the global torch stream consumption is fixed
by the spec. To add an architecture for try-and-error, add one line to ARCHS (and a matched MLP if the size differs).

Registered architectures (params for F = 7, 3 actions):
  a1        l64-cat-d64-d32           25,475   Dinos V.2-A (Codex primary)
  a1h       l128-cat-d128-d128       104,451   Dinos V.2-H (DERP head, Claude)
  a2        l64-l32-cat-d64-d32       35,971   2 LSTM layers
  a3        c32-l64-cat-d64-d32       32,579   Conv1d front
  a4        e32-l64-cat-d64-d32       32,131   per-step Dense embedding, LSTM as 2nd layer
  a10       e32-l128-cat-d128-d128   117,507   cross-pollination: branch 2's a4 embedding front + branch 1's a1h
  mlp_a1    m127                      25,657   matched to a1 (+0.71%)
  mlp_a1h   m288                     104,547   matched to a1h (+0.09%)
  mlp_a234  m151                      34,129   one control for the 32-36k class (a2 -5.1%, a3 +4.8%, a4 +6.2%)
  mlp_a2    m156                      36,039   exact match for a2 (+0.19%; registered, not in the default screen)
  mlp_a34   m146                      32,269   exact match for a3/a4 (-0.95% / +0.43%; registered, not default)
  mlp_a10   m307                     117,277   exact match for a10 (-0.20%)
  a11       l128-l128-cat-d128-d128         236,547   cycle 20: 2 stacked LSTM128 + the a1h head
  a12       l128-l128-l128-cat-d128-d128    368,643   cycle 20: 3 stacked LSTM128 (researcher myLSTM depth) + a1h head
  mlp_a11   m450                     236,253   exact match for a11 (-0.12%)
  mlp_a12   m571                     368,869   exact match for a12 (+0.06%)
  mlp_a1h_pm m271  (F = 12)          107,319   cycle 23 --pending-maturity on: match for a1h at F = 12 (107,651, -0.31%)
(tests/test_v2.py asserts these counts.)

Agents: make_scratch(arch, ...) -> lr 1e-4, epsilon 1 -> 0.05 (agent_lstm_fixed.make_scratch). As in
agent_dense_fixed, the myLSTM networks FixedLSTMAgent.__init__ builds are created inside torch.random.fork_rng and
discarded. Transfer = the existing orig-family transfer protocol of c5_protocol (source pretraining with a scratch
agent of the SAME arch, fresh scratch agent loading the online weights into online + target, fresh replay and Adam).
"""
import torch
import torch.nn as nn

from . import agent_dense_fixed as DFX
from . import agent_lstm_fixed as FX

ARCHS = {"a1": "l64-cat-d64-d32", "a1h": "l128-cat-d128-d128", "a2": "l64-l32-cat-d64-d32",
         "a3": "c32-l64-cat-d64-d32", "a4": "e32-l64-cat-d64-d32",
         "a10": "e32-l128-cat-d128-d128",
         "a11": "l128-l128-cat-d128-d128", "a12": "l128-l128-l128-cat-d128-d128",
         "mlp_a1": "m127", "mlp_a1h": "m288", "mlp_a234": "m151", "mlp_a2": "m156", "mlp_a34": "m146", "mlp_a10": "m307",
         "mlp_a11": "m450", "mlp_a12": "m571",
         "mlp_a1h_pm": "m271"}
MATCH = {"mlp_a1": ("a1",), "mlp_a1h": ("a1h",), "mlp_a234": ("a2", "a3", "a4"), "mlp_a2": ("a2",),
         "mlp_a34": ("a3", "a4"), "mlp_a10": ("a10",),
         "mlp_a11": ("a11",), "mlp_a12": ("a12",)}
# cycle 23 --pending-maturity on (F = 12 inputs per frame): matched MLPs re-matched for F = 12 (not in MATCH, whose
# counts are for F = 7). a1h at F = 12 = 107,651; m288 there = 118,947 (+10.5%), m271 = 107,319 (-0.31%).
MATCH_PM = {"mlp_a1h_pm": ("a1h",)}
PM_STATE_SIZE = 12
LABEL = {"a1": "Dinos V.2-A (LSTM64 -> concat obs -> 64 -> 32)", "a1h": "Dinos V.2-H (LSTM128 -> concat obs -> 128 -> 128)",
         "a2": "Dinos V.2 a2 (LSTM64 -> LSTM32 -> concat obs -> 64 -> 32)",
         "a3": "Dinos V.2 a3 (Conv1d32 -> LSTM64 -> concat obs -> 64 -> 32)",
         "a4": "Dinos V.2 a4 (step Dense32 -> LSTM64 -> concat obs -> 64 -> 32)",
         "a10": "Dinos V.2 a10 (step Dense32 -> LSTM128 -> concat obs -> 128 -> 128)",
         "a11": "Dinos V.2 a11 (LSTM128 -> LSTM128 -> concat obs -> 128 -> 128)",
         "a12": "Dinos V.2 a12 (LSTM128 -> LSTM128 -> LSTM128 -> concat obs -> 128 -> 128)"}


def is_mlp(arch):
    return ARCHS[arch].startswith("m")


def parse(spec):
    """spec -> list of (op, size); raises ValueError on a malformed spec."""
    toks = [t for t in str(spec).split("-") if t]
    if len(toks) == 1 and toks[0][0] == "m":
        return [("m", int(toks[0][1:]))]
    ops, seq = [], True
    for t in toks:
        if t == "cat":
            if not seq:
                raise ValueError(f"{spec}: 'cat' after the sequence was reduced")
            ops.append(("cat", 0))
            seq = False
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


class SpecQ(nn.Module):
    """Sequence Q-network built from a spec string (see the module docstring). init_hidden/hidden exist only for
    interface compatibility with the evaluation code (it sets m.hidden; unused, every call starts from zero)."""

    def __init__(self, spec, input_size, output_size):
        super().__init__()
        self.spec, self.input_size = str(spec), int(input_size)
        self.ops = parse(spec)
        if self.ops[0][0] == "m":
            raise ValueError("mW specs are built by agent_dense_fixed.FlatMLP")
        self.layers = nn.ModuleList()
        w, seq = self.input_size, True
        for op, k in self.ops:
            if op == "c":
                self.layers.append(nn.Conv1d(w, k, kernel_size=3, padding=1))
            elif op == "e":
                self.layers.append(nn.Linear(w, k))
            elif op == "l":
                self.layers.append(nn.LSTM(w, k, num_layers=1, batch_first=True))
            elif op == "d":
                self.layers.append(nn.Linear(w, k))
                seq = False
            else:          # cat
                self.layers.append(nn.Identity())
                k, seq = w + self.input_size, False
            w = k
        self.q = nn.Linear(w, output_size)
        self.relu = nn.ReLU()
        self.hidden = None

    def init_hidden(self):
        return None

    def forward(self, x):
        h = x
        for (op, k), m in zip(self.ops, self.layers):
            if op == "c":
                h = self.relu(m(h.transpose(1, 2))).transpose(1, 2)
            elif op == "e":
                h = self.relu(m(h))
            elif op == "l":
                h0 = torch.zeros(1, h.shape[0], k, dtype=h.dtype)
                h, _ = m(h, (h0, h0.clone()))
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
    return SpecQ(ARCHS[arch], state_size, action_size)


def architecture(arch, state_size, sequence_length=10, action_size=3):
    """Human-readable layer string (for the result JSON / dashboard)."""
    ops = parse(ARCHS[arch])
    if ops[0][0] == "m":
        n, W = sequence_length * state_size, ops[0][1]
        return f"flatten window[{sequence_length}x{state_size}] -> {n} -> {W} ReLU -> {W} ReLU -> {action_size}"
    parts, w = [f"window[{sequence_length}x{state_size}]"], state_size
    for op, k in ops:
        if op == "c":
            parts.append(f"Conv1d({w},{k},k3,pad1) ReLU")
        elif op == "e":
            parts.append(f"per-step Dense({w},{k}) ReLU")
        elif op == "l":
            parts.append(f"LSTM({w},{k},1 layer, zero h/c per window)")
        elif op == "cat":
            parts.append(f"concat(last[{w}], newest obs[{state_size}])")
            k = w + state_size
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
        if arch not in ARCHS or is_mlp(arch):
            raise ValueError(f"unknown V.2 sequence arch {arch!r}")
        with torch.random.fork_rng(devices=[]):
            super().__init__(state_size, action_size, sequence_length, learning_rate, epsilon_min, target_sync)
        self.variant = arch
        self.online_model = build_net(arch, state_size, sequence_length, action_size)
        self.target_model = build_net(arch, state_size, sequence_length, action_size)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)


def make_scratch(arch, state_size, action_size, seq, target_sync=FX.DEFAULT_TARGET_SYNC, ddqn=None, optim=None):
    """Counterpart of agent_lstm_fixed.make_scratch: lr 1e-4, epsilon 1 -> 0.05. MLP archs = agent_dense_fixed.
    DenseFixedAgent (the mlp_hist10 agent) with hidden width W. ddqn: kwargs of FixedLSTMAgent.set_ddqn (cycle 19;
    the same head/replay/return options apply to the sequence arch and its matched MLP)."""
    if arch not in ARCHS:
        raise ValueError(f"unknown V.2 arch {arch!r}")
    if is_mlp(arch):
        a = DFX.DenseFixedAgent(state_size, action_size, seq, 0.0001, 0.05, target_sync,
                                hidden=parse(ARCHS[arch])[0][1])
        a.variant = arch
    else:
        a = V2Agent(arch, state_size, action_size, seq, 0.0001, 0.05, target_sync)
    if ddqn:
        a.set_ddqn(**ddqn)
    if optim:           # cycle 23: lr override / grad-norm clip (FixedLSTMAgent.set_optim)
        a.set_optim(**optim)
    return a


def hyperparameters(agent):
    return dict(gamma=float(agent.gamma), replay_capacity=int(agent.memory.maxlen), batch_size=16,
                updates_per_replay=16, loss="SmoothL1 (Huber)", optimizer="Adam", learning_rate=float(agent.learning_rate),
                epsilon_min=float(agent.epsilon_min), target_sync_replays=int(agent.target_sync),
                double_dqn="detached (online argmax, target value), terminal masked",
                **({"ddqn_addons": agent.ddqn_config()} if (agent.n_step != 1 or agent.dueling or agent.per) else {}),
                **({"grad_clip_max_norm": float(agent.grad_clip)} if agent.grad_clip is not None else {}))
