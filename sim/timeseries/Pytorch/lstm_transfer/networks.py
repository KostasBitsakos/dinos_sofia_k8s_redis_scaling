import torch
import torch.nn as nn

class myLSTM(nn.Module):
    """LSTM Q-network.

    legacy=False (default): every forward call starts from a zero hidden state,
    so the output depends only on the input window (fix F3.3).
    legacy=True: reproduces the original behaviour, where self.hidden is carried
    across forward calls (batch size must then be 1).
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, legacy=False):
        super(myLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.legacy = legacy
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, x):
        if self.legacy:
            lstm_out, self.hidden = self.lstm(x, self.hidden)
        else:
            lstm_out, _ = self.lstm(x)  # zero initial state, nothing stored
        out = self.fc(lstm_out[:, -1, :])
        return out
