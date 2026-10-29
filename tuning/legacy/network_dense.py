# LEGACY PORT (Track L) -- verbatim copy of /home/ubuntu/simulations/redisk8s/timeseries/Pytorch/dense/network.py:1-18
# original sha256 3974c048fb4cc2318198c26c651af3c58d9ed364f0ce732f53bfeb2fa8414564
# role: DERP dense Q-network
# changes: none; nothing else. Regenerate/verify with tuning/legacy/port_sources.py and tests/test_legacy.py.
import torch
import torch.nn as nn

class MyDenseNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyDenseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
