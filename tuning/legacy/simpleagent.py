# LEGACY PORT (Track L) -- verbatim copy of /home/ubuntu/simulations/redisk8s/timeseries/Pytorch/HPA/simpleagent.py:1-14
# original sha256 ece841f7d7152fc35adc0d687a757d5dee950deb12665f6d91e5df39484452ea
# role: threshold rule of HPA/hpa_final.py
# changes: import lines 1 made package-relative; nothing else. Regenerate/verify with tuning/legacy/port_sources.py and tests/test_legacy.py.
from .env_dense import Environment


class SimpleAgent:
    def __init__(self, cpu_threshold=50, memory_threshold=50):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold

    def act(self, state):
        _, _, cpu_usage, memory_usage, _, num_vms = state
        if cpu_usage > self.cpu_threshold or memory_usage > self.memory_threshold:
            return 2  # Increase VMs
        else:
            return 0  # Decrease VMs
