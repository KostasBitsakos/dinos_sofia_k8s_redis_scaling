import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.time = 0
        self.num_vms = 6  # Starting number of VMs
        self.load_amplitude = 50
        self.load_period = 1000 / (2 * np.pi)
        self.cpu_range = (30, 75)
        self.memory_range = (40, 65)
        self.latency_amplitude = 1.2
        self.spike_position = 1500
        self.spike_length = 100
        self.spike_amplitude = 3
        self.load_spike_amplitude = 80
        self.load_offset = 20  # Ensure load doesn't go near zero

    # --- Load Function ---
    def load(self, t):
        load = self.load_amplitude * np.sin(2 * np.pi * t / self.load_period) + self.load_amplitude + self.load_offset
        load += np.random.normal(0, 0.2)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            load += self.load_spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * (t - self.spike_position) / self.spike_length))
        return max(load, 0)

    # --- Capacity Function ---
    def capacity(self, t):
        capacity = 10 * self.num_vms
        if self.spike_position <= t < self.spike_position + self.spike_length:
            capacity += self.spike_amplitude * 0.5 * np.abs(np.sin(2 * np.pi * 20 * (t - self.spike_position) / self.spike_length))
        return max(capacity, 0)

    # --- CPU Usage Function ---
    def cpu_usage(self, t):
        cpu_usage = np.random.uniform(*self.cpu_range)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            cpu_usage += self.spike_amplitude
        return max(cpu_usage, 0)

    # --- Memory Usage Function ---
    def memory_usage(self, t):
        memory_usage = np.random.uniform(*self.memory_range)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            memory_usage += self.spike_amplitude
        return max(memory_usage, 0)

    # --- Latency Function ---
    def latency(self, t):
        latency = self.latency_amplitude * np.sin(2 * np.pi * 2 * t / self.load_period) + self.latency_amplitude
        latency += np.random.normal(0, 0.1)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            latency += self.spike_amplitude * 0.8 * np.abs(np.sin(2 * np.pi * 20 * (t - self.spike_position) / self.spike_length))
        return max(latency, 0)

    # --- Step Function (Fixed to Ensure Full Evaluation) ---
    def step(self, action):
        """
        Perform an action in the environment and calculate the next state, reward, and other metrics.

        Parameters:
        - action: int (0 = scale down, 1 = no change, 2 = scale up)

        Returns:
        - next_state: np.array (contains [load, capacity, cpu_usage, memory_usage, latency, num_vms])
        - reward: float (overall reward combining load, latency, VM, CPU, and memory factors)
        - load_reward: float (reward based on load)
        - latency_penalty: float (penalty for high latency)
        - vm_penalty: float (penalty for number of VMs)
        - cpu_penalty: float (penalty for high CPU usage)
        - memory_penalty: float (penalty for high memory usage)
        - done: bool (whether the simulation is finished)
        """
        # Adjust the number of VMs based on the action
        if action == 0 and self.num_vms > 1:
            self.num_vms -= 1
        elif action == 2 and self.num_vms < 20:
            self.num_vms += 1

        self.time += 1  # Increment time step

        # Retrieve metrics for the next state
        next_load = self.load(self.time)
        next_capacity = self.capacity(self.time)
        next_cpu_usage = self.cpu_usage(self.time)
        next_memory_usage = self.memory_usage(self.time)
        next_latency = self.latency(self.time)

        # Reward function components
        load_reward = 3.0 * next_load
        latency_penalty = -5.0 * next_latency
        vm_penalty = -10.0 * self.num_vms
        cpu_penalty = -0.5 * next_cpu_usage
        memory_penalty = -0.5 * next_memory_usage

        reward = load_reward + latency_penalty + vm_penalty + cpu_penalty + memory_penalty

        # Ensure evaluation runs for the full 5000 steps
        done = self.time >= 5000

        next_state = np.array([next_load, next_capacity, next_cpu_usage, next_memory_usage, next_latency, self.num_vms])

        return next_state, reward, load_reward, latency_penalty, vm_penalty, cpu_penalty, memory_penalty, done

    # --- Reset Function ---
    def reset(self):
        self.time = 0
        self.num_vms = 6
        initial_load = self.load(self.time)
        initial_capacity = self.capacity(self.time)
        initial_cpu_usage = self.cpu_usage(self.time)
        initial_memory_usage = self.memory_usage(self.time)
        initial_latency = self.latency(self.time)
        initial_state = np.array([initial_load, initial_capacity, initial_cpu_usage, initial_memory_usage, initial_latency, self.num_vms])
        return initial_state

# --- Visualization of Environment Metrics ---
if __name__ == "__main__":
    env = Environment()
    time_steps = range(5000)

    metrics = {
        "Load": [env.load(t) for t in time_steps],
        "Capacity": [env.capacity(t) for t in time_steps],
        "CPU Usage": [env.cpu_usage(t) for t in time_steps],
        "Memory Usage": [env.memory_usage(t) for t in time_steps],
        "Latency": [env.latency(t) for t in time_steps],
    }

    plt.figure(figsize=(12, 8))
    for idx, (metric, values) in enumerate(metrics.items(), 1):
        plt.subplot(3, 2, idx)
        plt.plot(time_steps, values, label=metric)
        plt.xlabel("Time")
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()
    plt.savefig("environment_metrics.png")
    plt.show()
