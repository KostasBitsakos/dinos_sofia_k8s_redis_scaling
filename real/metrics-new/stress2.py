from rediscluster import RedisCluster
import time
import math
import psutil
import matplotlib.pyplot as plt
import numpy as np
from kubernetes import client, config
from tqdm import tqdm
import subprocess

# Parameters for sinusoidal load
period = 30  # seconds for a full cycle
duration = 10 * 60  # total duration of 10 minutes
time_steps = np.arange(0, duration, 5)  # time points every 5 seconds
loads = 0.5 * (1 + np.sin(2 * np.pi * time_steps / period))  # sinusoidal load
rewards = []

# Reward function parameters
load_factor = 3.0
latency_factor = -5.0
vm_penalty_factor = -10.0
cpu_penalty_factor = -0.5
memory_penalty_factor = -0.5

# Connect to Redis cluster
try:
    r = RedisCluster(startup_nodes=[{"host": "10.244.37.156", "port": 6379}], decode_responses=True)
except Exception as e:
    print(f"Failed to connect to Redis cluster: {e}")
    exit(1)

# Function to send requests to Redis cluster based on current load
def send_requests_to_redis(load):
    try:
        for _ in range(int(load * 1000)):  # Sending multiple requests based on load
            r.set("key", "value" * int(load * 100))
    except Exception as e:
        print(f"Error sending requests to Redis: {e}")

# Function to get number of pods from Kubernetes
def get_num_pods():
    result = subprocess.run(['kubectl', 'get', 'pods', '--no-headers'], stdout=subprocess.PIPE)
    pods = result.stdout.decode('utf-8').splitlines()
    return len([p for p in pods if 'redis-autoscale-cluster' in p and 'Running' in p])

# Function to get CPU and memory usage across all nodes
def get_cluster_metrics():
    total_cpu, total_memory = 0.0, 0.0
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        if 'redis-server' in proc.info['name']:
            total_cpu += proc.info['cpu_percent']
            total_memory += proc.info['memory_percent']
    return total_cpu, total_memory

# Initialize metrics lists
cpu_usage = []
memory_usage = []
pod_count = []
latency = []  # Simulating latency for reward function
load_metrics = []
reward_values = []

# Initialize Kubernetes config
config.load_kube_config()

# Progress bar for the stress test
for i, current_load in tqdm(enumerate(loads), total=len(loads), desc="Stress Test Progress"):
    # Send requests to Redis cluster
    send_requests_to_redis(current_load)

    # Get metrics
    total_cpu, total_memory = get_cluster_metrics()
    pods = get_num_pods()

    # Simulate latency (e.g., from Redis response time) - modify this based on real metrics if available
    current_latency = np.random.uniform(0.1, 1.0)  # Simulated latency in seconds
    latency.append(current_latency)

    # Calculate reward
    load_reward = load_factor * current_load
    latency_penalty = latency_factor * current_latency
    vm_penalty = vm_penalty_factor * pods  # Using number of pods in the reward calculation
    cpu_penalty = cpu_penalty_factor * total_cpu
    memory_penalty = memory_penalty_factor * total_memory

    reward = load_reward + latency_penalty + vm_penalty + cpu_penalty + memory_penalty
    rewards.append(reward)
    reward_values.append(reward)

    # Store metrics
    cpu_usage.append(total_cpu)
    memory_usage.append(total_memory)
    pod_count.append(pods)
    load_metrics.append(current_load)

    # Wait for 5 seconds before the next iteration
    time.sleep(5)

# Plot the metrics
plt.figure(figsize=(12, 6))
plt.plot(time_steps, cpu_usage, label="CPU Usage (%)")
plt.plot(time_steps, memory_usage, label="Memory Usage (%)")
plt.plot(time_steps, pod_count, label="Number of Pods")
plt.xlabel("Time (seconds)")
plt.ylabel("Metric Value")
plt.legend()
plt.title("CPU, Memory, and Pod Count over Time")
plt.savefig("combined_metrics.png")
plt.show()

# Plot each metric separately
plt.figure()
plt.plot(time_steps, cpu_usage)
plt.xlabel("Time (seconds)")
plt.ylabel("CPU Usage (%)")
plt.title("CPU Usage over Time")
plt.savefig("cpu_usage.png")

plt.figure()
plt.plot(time_steps, memory_usage)
plt.xlabel("Time (seconds)")
plt.ylabel("Memory Usage (%)")
plt.title("Memory Usage over Time")
plt.savefig("memory_usage.png")

plt.figure()
plt.plot(time_steps, pod_count)
plt.xlabel("Time (seconds)")
plt.ylabel("Number autoscaling-redis-cluster-hpa-kubernetesof Pods")
plt.title("Number of Pods over Time")
plt.savefig("pod_count.png")

# Plot the load over time
plt.figure()
plt.plot(time_steps, load_metrics)
plt.xlabel("Time (seconds)")
plt.ylabel("Load")
plt.title("Load over Time")
plt.savefig("load.png")

# Plot rewards over time
plt.figure()
plt.plot(time_steps, rewards)
plt.xlabel("Time (seconds)")
plt.ylabel("Reward")
plt.title("Reward over Time")
plt.savefig("rewards.png")

# Show final accumulated reward
total_reward = np.sum(rewards)
print(f"Total Accumulated Reward: {total_reward}")
