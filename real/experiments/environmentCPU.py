import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.time = 0
        self.num_vms = 6
        self.load_amplitude = 50
        self.load_period = 1000 / (2 * np.pi)
        self.read_amplitude = 0.75
        self.read_period = 1000 / (4 * np.pi)
        self.cpu_range = (30, 75)
        self.memory_range = (40, 65)
        self.latency_amplitude = 1.2
        self.spike_position = 1500
        self.spike_length = 100
        self.spike_amplitude = 3
        self.load_spike_amplitude = 80
        self.load_offset = 20

    def load(self, t):
        load = self.load_amplitude * np.sin(2 * np.pi * t / self.load_period) + self.load_amplitude + self.load_offset
        load += np.random.normal(0, 0.2)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            load += self.load_spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * (t - self.spike_position) / self.spike_length))
        return max(load, 0)

    def read_percentage(self, t):
        read = self.read_amplitude * np.sin(2 * np.pi * t / self.read_period) + self.read_amplitude
        read += np.random.normal(0, 0.1)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            read += self.spike_amplitude * 0.3 * np.abs(np.sin(2 * np.pi * 20 * (t - self.spike_position) / self.spike_length))
        return max(read, 0)

    def capacity(self, t):
        return 10 * self.num_vms * self.read_percentage(t)

    def cpu_usage(self, t):
        cpu = np.random.uniform(*self.cpu_range)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            cpu += self.spike_amplitude
        return max(cpu, 0)

    def memory_usage(self, t):
        memory = np.random.uniform(*self.memory_range)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            memory += self.spike_amplitude
        return max(memory, 0)

    def latency(self, t):
        latency = self.latency_amplitude * np.sin(2 * np.pi * 2 * t / self.load_period) + self.latency_amplitude
        latency += np.random.normal(0, 0.1)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            latency += self.spike_amplitude * 0.8 * np.abs(np.sin(2 * np.pi * 20 * (t - self.spike_position) / self.spike_length))
        return max(latency, 0)

    def step(self, action):
        if action == 0 and self.num_vms > 1:
            self.num_vms -= 1
        elif action == 2 and self.num_vms < 20:
            self.num_vms += 1

        self.time += 1
        next_load = self.load(self.time)
        next_capacity = self.capacity(self.time)
        next_cpu = self.cpu_usage(self.time)
        next_memory = self.memory_usage(self.time)
        next_latency = self.latency(self.time)
        num_vms = self.num_vms

        load_reward = 3.0 * next_load
        latency_penalty = -5.0 * next_latency
        vm_penalty = -10.0 * self.num_vms
        cpu_penalty = -0.5 * next_cpu
        memory_penalty = -0.5 * next_memory

        reward = load_reward + latency_penalty + vm_penalty + cpu_penalty + memory_penalty
        done = self.time >= 5000

        next_state = np.array([next_load, next_capacity, next_cpu, next_memory, next_latency, num_vms])
        return next_state, reward, load_reward, latency_penalty, vm_penalty, cpu_penalty, memory_penalty, done

    def reset(self):
        self.time = 0
        self.num_vms = 6
        return np.array([self.load(self.time), self.capacity(self.time), self.cpu_usage(self.time),
                         self.memory_usage(self.time), self.latency(self.time), self.num_vms])

if __name__ == "__main__":
    env = Environment()
    time_steps = range(5000)

    metrics = {
        "Load": [env.load(t) for t in time_steps],
        "Capacity": [env.capacity(t) for t in time_steps],
        "Read Percentage": [env.read_percentage(t) for t in time_steps],
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
    plt.savefig('final_metrics_original/environment_metrics.png')
    plt.show()
