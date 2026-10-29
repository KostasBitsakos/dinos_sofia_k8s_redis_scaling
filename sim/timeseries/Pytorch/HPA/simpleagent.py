from environment import Environment


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
