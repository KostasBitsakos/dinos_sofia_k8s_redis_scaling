"""Cycle 10b matched-mechanics dense control (Codex review c10_codex_review.md point 2).

DenseFixedAgent = agent_lstm_fixed.FixedLSTMAgent with ONLY the Q-network replaced: every method (act, replay,
ddqn_target, remember, _x, update_target_model, load_weights) is inherited unchanged, and every hyperparameter
(gamma 0.98, replay capacity 5000, Double-DQN detached target, target sync period, lr, epsilon start/floor/decay,
16 sequential single-sample Huber updates per replay, Adam) is set by FixedLSTMAgent.__init__ itself. The agent is
fed the very same (sequence_length x F) window as the LSTM (c5_protocol lstm=True path); the network flattens it to
sequence_length * F inputs and applies network_dense.MyDenseNet (the DERP architecture: Linear-ReLU-Linear-ReLU-Linear)
with hidden width DENSE_HIDDEN, chosen so the trainable parameter count is within +-20% of the fixed LSTM's
(myLSTM(F, 128, 3 layers, 3)): F = 7: MLP 70->540->540->3 = 332,103 vs LSTM 334,723; F = 6: 326,703 vs 334,211.
The LSTM networks that FixedLSTMAgent.__init__ builds are discarded; they are created inside torch.random.fork_rng,
so the global torch stream is consumed only by the MLP initialisation.
"""
import torch
import torch.nn as nn

from . import agent_lstm_fixed as FX
from .network_dense import MyDenseNet

DENSE_HIDDEN = 540


class FlatMLP(nn.Module):
    """Flatten the (batch, seq, F) window row-major (oldest frame first) and apply MyDenseNet(seq * F, H, out).
    No recurrent state; init_hidden/hidden exist only for interface compatibility with the LSTM evaluation code."""

    def __init__(self, input_size, sequence_length, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.net = MyDenseNet(input_size * sequence_length, hidden_size, output_size)
        self.hidden = None

    def init_hidden(self):
        return None

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class DenseFixedAgent(FX.FixedLSTMAgent):
    def __init__(self, state_size, action_size, sequence_length, learning_rate, epsilon_min,
                 target_sync=FX.DEFAULT_TARGET_SYNC, hidden=DENSE_HIDDEN):
        with torch.random.fork_rng(devices=[]):
            super().__init__(state_size, action_size, sequence_length, learning_rate, epsilon_min, target_sync)
        self.dense_hidden = int(hidden)
        self.online_model = FlatMLP(state_size, sequence_length, self.dense_hidden, action_size)
        self.target_model = FlatMLP(state_size, sequence_length, self.dense_hidden, action_size)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)


def make_scratch(state_size, action_size, seq, target_sync=FX.DEFAULT_TARGET_SYNC, ddqn=None):
    """Counterpart of agent_lstm_fixed.make_scratch: lr 1e-4, epsilon 1 -> 0.05. ddqn: cycle-19 set_ddqn kwargs."""
    a = DenseFixedAgent(state_size, action_size, seq, 0.0001, 0.05, target_sync)
    if ddqn:
        a.set_ddqn(**ddqn)
    return a
