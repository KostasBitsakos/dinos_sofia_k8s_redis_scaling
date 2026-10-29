import numpy as np
import os
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
        self.load_offset = 20  # Ensures load doesn't go near zero

    def load(self, t):
        load = self.load_amplitude * np.sin(2 * np.pi * t / self.load_period) + self.load_amplitude + self.load_offset
        load += np.random.normal(0, 0.2)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            load += self.load_spike_amplitude * np.abs(np.sin(2 * np.pi * 10 * (t - self.spike_position) / self.spike_length))
        return max(load, 0)

    def capacity(self, t):
        return 10 * self.num_vms  

    def cpu_usage(self, t):
        return np.random.uniform(*self.cpu_range)

    def memory_usage(self, t):
        return np.random.uniform(*self.memory_range)

    def latency(self, t):
        return self.latency_amplitude * np.sin(2 * np.pi * 2 * t / self.load_period) + self.latency_amplitude

    def step(self, action):
        if action == 0 and self.num_vms > 1:
            self.num_vms -= 1
        elif action == 2 and self.num_vms < 20:
            self.num_vms += 1

        self.time += 1
        next_load = self.load(self.time)
        next_capacity = self.capacity(self.time)
        next_cpu_usage = self.cpu_usage(self.time)
        next_memory_usage = self.memory_usage(self.time)
        next_latency = self.latency(self.time)
        num_vms = self.num_vms

        # --- Normalization of Metrics ---
        next_load_norm = next_load / (self.load_amplitude + self.load_offset)
        next_latency_norm = next_latency / (self.latency_amplitude + self.spike_amplitude + 1.0)
        next_cpu_usage_norm = (next_cpu_usage - self.cpu_range[0]) / (self.cpu_range[1] - self.cpu_range[0])
        next_memory_usage_norm = (next_memory_usage - self.memory_range[0]) / (self.memory_range[1] - self.memory_range[0])

        # --- Reward Function ---
        load_factor = 50.0  
        latency_factor = -0.5  
        vm_penalty_factor = -1.0  
        cpu_penalty_factor = -0.05  
        memory_penalty_factor = -0.05  

        load_reward = load_factor * next_load_norm
        latency_penalty = latency_factor * next_latency_norm
        vm_penalty = vm_penalty_factor * num_vms
        cpu_penalty = cpu_penalty_factor * next_cpu_usage_norm
        memory_penalty = memory_penalty_factor * next_memory_usage_norm

        # Large Baseline Reward
        baseline_reward = 10000.0  
        reward = load_reward + latency_penalty + vm_penalty + cpu_penalty + memory_penalty + baseline_reward

        next_state = np.array([next_load, next_capacity, next_cpu_usage, next_memory_usage, next_latency, num_vms])
        done = self.time >= 5000

        return next_state, reward, load_reward, latency_penalty, vm_penalty, cpu_penalty, memory_penalty, done
