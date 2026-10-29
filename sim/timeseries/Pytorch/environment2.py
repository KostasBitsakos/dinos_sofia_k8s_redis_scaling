import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.time = 0
        self.num_vms = 6  # Starting number of VMs
        self.load_amplitude = 50
        self.load_period = 1000 / (2 * np.pi)
        self.read_amplitude = 0.75
        self.read_period = 1000 / (4 * np.pi)
        self.cpu_range = (30, 50)
        self.memory_range = (40, 60)
        self.latency_amplitude = 1.2
        self.spike_position = 1500
        self.spike_length = 100
        self.spike_amplitude = 3
        self.load_spike_amplitude = 80

    def load(self, t):
        load = self.load_amplitude * np.sin(2 * np.pi * t / self.load_period) + self.load_amplitude
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
        capacity = 10 * self.num_vms * self.read_percentage(t)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            capacity += self.spike_amplitude * 0.5 * np.abs(np.sin(2 * np.pi * 20 * (t - self.spike_position) / self.spike_length))
        return max(capacity, 0)

    def cpu_usage(self, t):
        cpu_usage = np.random.uniform(*self.cpu_range)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            cpu_usage += self.spike_amplitude
        return max(cpu_usage, 0)

    def memory_usage(self, t):
        memory_usage = np.random.uniform(*self.memory_range)
        if self.spike_position <= t < self.spike_position + self.spike_length:
            memory_usage += self.spike_amplitude
        return max(memory_usage, 0)

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
        next_cpu_usage = self.cpu_usage(self.time)
        next_memory_usage = self.memory_usage(self.time)
        next_latency = self.latency(self.time)
        num_vms = self.num_vms

        load_factor = 1.0
        latency_factor = -5.0
        vm_penalty_factor = -2.0
        load_reward = load_factor * next_load
        latency_penalty = latency_factor * next_latency
        vm_penalty = vm_penalty_factor * self.num_vms

        # reward = load_reward + latency_penalty + vm_penalty

        # # Clip the reward
        # #reward = np.clip(reward, -10, 10)

        # next_state = np.array([next_load, next_capacity, next_cpu_usage, next_memory_usage, next_latency, num_vms])
        # done = self.time >= 5000

        # return next_state, reward, load_reward, latency_penalty, vm_penalty, done

        reward = load_factor * next_load + latency_factor * next_latency + vm_penalty_factor * self.num_vms

        next_state = np.array([next_load, next_capacity, next_cpu_usage, next_memory_usage, next_latency, num_vms])
        done = self.time >= 5000

        return next_state, reward, done

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


if __name__ == "__main__":
    env = Environment()

    # Plotting
    time_steps = range(5000)
    env_load = [env.load(t) for t in time_steps]
    env_capacity = [env.capacity(t) for t in time_steps]
    env_read = [env.read_percentage(t) for t in time_steps]
    env_cpu = [env.cpu_usage(t) for t in time_steps]
    env_memory = [env.memory_usage(t) for t in time_steps]
    env_latency = [env.latency(t) for t in time_steps]
    env.step(0)
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 2, 1)
    plt.plot(time_steps, env_load, label='Load')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(time_steps, env_capacity, label='Capacity')
    plt.xlabel('Time')
    plt.ylabel('Capacity')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(time_steps, env_read, label='Read Percentage')
    plt.xlabel('Time')
    plt.ylabel('Read Percentage')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(time_steps, env_cpu, label='CPU Usage')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(time_steps, env_memory, label='Memory Usage')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(time_steps, env_latency, label='Latency')
    plt.xlabel('Time')
    plt.ylabel('Latency')
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics.png')
    plt.show()

    # Instead of showing the plot, save it to a file
    # env = Environment()
    env.time = 0
    actions = [0, 1, 2]  # Possible actions

    for _ in range(10):  # Run for 10 steps
        action = np.random.choice(actions)  # Random action
        new_state, reward, done = env.step(action)
        print(f"Time: {env.time}")
        print(f"Load: {new_state[0]}, Capacity: {new_state[1]}, Reward: {reward}")
        print(f"CPU Usage: {new_state[2]}, Memory Usage: {new_state[3]}, Latency: {new_state[4]}")
        print(f"Number of VMs: {new_state[5]}\n")
        # print(f"Load Reward: {load_reward}, Latency Penalty: {latency_penalty}, VM Penalty: {vm_penalty}")


