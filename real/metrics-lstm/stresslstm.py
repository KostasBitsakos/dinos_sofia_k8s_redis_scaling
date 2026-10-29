import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from kubernetes import client, config
from tqdm import tqdm
from rediscluster import RedisCluster
import subprocess
import psutil

# Define the myLSTM model class
class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(myLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Load the model
state_size = 6
sequence_length = 10
action_size = 3
model = myLSTM(input_size=state_size, hidden_size=128, num_layers=3, output_size=action_size)
model.load_state_dict(torch.load("modelCPU_10.pth"))
model.eval()

# Function to get Redis pod IP addresses dynamically
def get_redis_pod_ips():
    result = subprocess.run(['kubectl', 'get', 'pods', '-o', 'wide', '--no-headers'], stdout=subprocess.PIPE)
    pods = result.stdout.decode('utf-8').splitlines()
    redis_pods = [p.split() for p in pods if 'redis-autoscale-cluster' in p and 'Running' in p]
    
    # Return a list of dicts with host and port
    redis_nodes = [{"host": pod[5], "port": 6379} for pod in redis_pods]
    return redis_nodes

# Function to establish a Redis Cluster connection
def connect_to_redis_cluster():
    redis_nodes = get_redis_pod_ips()
    if not redis_nodes:
        print("No running Redis pods found.")
        return None

    try:
        redis_cluster = RedisCluster(startup_nodes=redis_nodes, decode_responses=True)
        print("Connected to Redis cluster.")
        return redis_cluster
    except Exception as e:
        print(f"Failed to connect to Redis cluster: {e}")
        return None

# Function to get current pod count from Kubernetes
def get_pod_count():
    result = subprocess.run(['kubectl', 'get', 'pods', '--no-headers'], stdout=subprocess.PIPE)
    pods = result.stdout.decode('utf-8').splitlines()
    return len([p for p in pods if 'redis-autoscale-cluster' in p and 'Running' in p])

# Function to gracefully retry Redis requests if TTL exhausted error occurs
def send_requests_to_redis(load, redis_cluster):
    if not redis_cluster:
        print("No Redis cluster connection.")
        return

    retries = 3  # Retry up to 3 times
    for _ in range(retries):
        try:
            for _ in range(int(load * 1000)):  # Sending multiple requests based on load
                redis_cluster.set("key", "value" * int(load * 100))
            return  # Exit function if successful
        except Exception as e:
            if "TTL exhausted" in str(e):
                print(f"Error sending requests to Redis: TTL exhausted. Retrying...")
                time.sleep(5)  # Wait 5 seconds before retrying
            else:
                print(f"Error sending requests to Redis: {e}")
                break  # Exit on non-TTL errors

# Function to scale the number of pods using the Kubernetes API
def scale_pods(chosen_action, current_pod_count):
    v1 = client.AppsV1Api()
    deployment_name = "redis-autoscale-cluster"
    namespace = "default"

    new_pod_count = current_pod_count
    # Ensure a minimum of 3 pods to avoid Redis cluster failure
    if chosen_action == 0 and current_pod_count > 3:
        new_pod_count = max(3, current_pod_count - 1)
        print(f"Scaling down to {new_pod_count} pods")
    elif chosen_action == 2 and current_pod_count < 20:
        new_pod_count = min(20, current_pod_count + 1)
        print(f"Scaling up to {new_pod_count} pods")

    if new_pod_count != current_pod_count:
        body = {'spec': {'replicas': new_pod_count}}
        v1.patch_namespaced_deployment_scale(name=deployment_name, namespace=namespace, body=body)
        print(f"Pod scaling applied: {new_pod_count} replicas")

        # Introduce a delay after scaling to give Redis time to rebalance
        print("Waiting for Redis cluster to rebalance...")
        time.sleep(30)  # Wait 30 seconds for Redis to rebalance

# Function to get average CPU and memory usage across Redis nodes
def get_avg_cpu_memory():
    total_cpu, total_memory = 0.0, 0.0
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        if 'redis-server' in proc.info['name']:
            total_cpu += proc.info['cpu_percent']
            total_memory += proc.info['memory_percent']
    return total_cpu, total_memory

# Connect to Redis cluster
redis_cluster = connect_to_redis_cluster()

# Initialize Kubernetes config
config.load_kube_config()

# Parameters for sinusoidal load
# Parameters for sinusoidal load
# Parameters for sinusoidal load
period = 30  # seconds for a full cycle
duration = 10 * 60  # total duration of 10 minutes
time_steps = np.arange(0, duration, 15)  # time points every 15 seconds

# Sinusoidal load varying between 0.2 and 0.8
min_load = 0.2
max_load = 0.8
amplitude = (max_load - min_load) / 2
mean_load = (max_load + min_load) / 2
loads = mean_load + amplitude * np.sin(2 * np.pi * time_steps / period)


# Debugging line to check the load range
print(f"Load varies between {loads.min()} and {loads.max()}")


# Initialize metrics lists
cpu_usage = []
memory_usage = []
pod_count = []
load_list = []
reward_list = []

# Initialize the first state
current_state = np.zeros((sequence_length, state_size))

# Autoscaling loop with time progression bar
for i, load in tqdm(enumerate(loads), total=len(loads), desc="Autoscaling Progress", unit="step"):
    print(f"Current load being applied: {load}")  # Debugging line to ensure load is varying
    send_requests_to_redis(load, redis_cluster)  # Apply load to Redis cluster
    
    next_load = load
    load_list.append(next_load)
    
    next_cpu_usage, next_memory_usage = get_avg_cpu_memory()  # Get avg CPU and memory usage
    next_latency = np.random.uniform(0.1, 1.0)  # Simulated latency
    next_num_pods = get_pod_count()  # Get number of pods
    next_capacity = next_load * next_num_pods * 0.1  # Simulate capacity or any other relevant feature

    next_state = np.array([next_load, next_cpu_usage, next_memory_usage, next_latency, next_num_pods, next_capacity])
    current_state = np.roll(current_state, -1, axis=0)  # Shift the sequence
    current_state[-1] = next_state  # Add the new state

    # Reshape the state for LSTM input
    lstm_input = np.reshape(current_state, (1, sequence_length, state_size))
    lstm_input_tensor = torch.tensor(lstm_input, dtype=torch.float32)
    
    with torch.no_grad():
        action = model(lstm_input_tensor)
        chosen_action = torch.argmax(action).item()

    scale_pods(chosen_action, next_num_pods)

    cpu_usage.append(next_cpu_usage)
    memory_usage.append(next_memory_usage)
    pod_count.append(next_num_pods)

    reward = next_load * 3.0 - next_latency * 5.0 - next_num_pods * 10.0 - next_cpu_usage * 0.5 - next_memory_usage * 0.5
    reward_list.append(reward)

    time.sleep(15)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(time_steps, cpu_usage, label="CPU Usage (%)")
plt.plot(time_steps, memory_usage, label="Memory Usage (%)")
plt.plot(time_steps, pod_count, label="Pod Count")
plt.plot(time_steps, load_list, label="Load")
plt.plot(time_steps, reward_list, label="Reward")
plt.xlabel("Time (seconds)")
plt.ylabel("Metrics")
plt.legend()
plt.title("Autoscaling Metrics over Time")
plt.savefig("autoscaling_metricsnew.png")
plt.show()

total_reward = np.sum(reward_list)
print(f"Total Accumulated Reward: {total_reward}")
