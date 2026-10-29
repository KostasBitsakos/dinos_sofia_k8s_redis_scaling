import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from kubernetes import client, config
from tqdm import tqdm
from rediscluster import RedisCluster

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

# Function to get current pod count from Kubernetes
def get_pod_count():
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace="default", label_selector="app=redis-autoscale-cluster")
    return len(pods.items)

# Function to scale the number of pods using the Kubernetes API
def scale_pods(chosen_action, current_pod_count):
    v1 = client.AppsV1Api()
    deployment_name = "redis-autoscale-cluster"
    namespace = "default"
    
    # Get current deployment
    deployment = v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    new_pod_count = current_pod_count
    
    # Scale up or down based on chosen_action
    if chosen_action == 0 and current_pod_count > 2:  # Minimum of 2 pods
        new_pod_count = max(2, current_pod_count - 1)
        print(f"Scaling down to {new_pod_count} pods")
    elif chosen_action == 2 and current_pod_count < 20:
        new_pod_count = min(20, current_pod_count + 1)
        print(f"Scaling up to {new_pod_count} pods")

    # Update the deployment's replica count if needed
    if new_pod_count != current_pod_count:
        body = {'spec': {'replicas': new_pod_count}}
        v1.patch_namespaced_deployment_scale(name=deployment_name, namespace=namespace, body=body)
        print(f"Pod scaling applied: {new_pod_count} replicas")

# Function to get average CPU and memory usage across Redis nodes
def get_avg_cpu_memory():
    v1 = client.CustomObjectsApi()
    metrics = v1.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", "default", "pods")
    total_cpu = 0
    total_memory = 0
    num_pods = 0
    
    for pod in metrics['items']:
        if 'redis-autoscale-cluster' in pod['metadata']['name']:
            cpu_str = pod['containers'][0]['usage']['cpu']
            memory_str = pod['containers'][0]['usage']['memory']
            total_cpu += convert_cpu_to_percent(cpu_str)
            total_memory += convert_memory_to_percent(memory_str)
            num_pods += 1
    
    avg_cpu = total_cpu / num_pods if num_pods > 0 else 0
    avg_memory = total_memory / num_pods if num_pods > 0 else 0
    
    return avg_cpu, avg_memory

def convert_cpu_to_percent(cpu_str):
    if 'n' in cpu_str:  # nanocores to cores
        return int(cpu_str.replace('n', '')) / 1000000
    elif 'u' in cpu_str:  # microcores to cores
        return int(cpu_str.replace('u', '')) / 1000000000
    elif 'm' in cpu_str:  # millicores to cores
        return int(cpu_str.replace('m', '')) / 1000
    return float(cpu_str)

def convert_memory_to_percent(memory_str):
    if 'Ki' in memory_str:  # KiB to MiB
        return int(memory_str.replace('Ki', '')) / 1024
    elif 'Mi' in memory_str:  # MiB
        return int(memory_str.replace('Mi', ''))
    elif 'Gi' in memory_str:  # GiB to MiB
        return int(memory_str.replace('Gi', '')) * 1024
    elif 'Ti' in memory_str:  # TiB to MiB
        return int(memory_str.replace('Ti', '')) * 1024 * 1024
    return int(memory_str)

# Function to send requests to Redis cluster with retry logic for TTL errors
def send_requests_to_redis(load, max_retries=5, retry_delay=2):
    for retry in range(max_retries):
        try:
            for _ in range(int(load * 1000)):  # Sending multiple requests based on load
                redis_cluster.set("key", "value" * int(load * 100))
            return  # Exit if the request is successful
        except Exception as e:
            if "TTL exhausted" in str(e):
                print(f"Error sending requests to Redis: TTL exhausted. Retrying... (attempt {retry+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Error sending requests to Redis: {e}")
                break  # Break for non-TTL errors

# Initialize Redis Cluster connection with correct IP addresses
try:
    redis_cluster = RedisCluster(startup_nodes=[
        {"host": "10.244.37.150", "port": 6379},
        {"host": "10.244.37.131", "port": 6379},
        {"host": "10.244.37.137", "port": 6379}
    ], decode_responses=True)
    print("Connected to Redis cluster.")
except Exception as e:
    print(f"Failed to connect to Redis cluster: {e}")
    exit(1)

# Initialize Kubernetes config
config.load_kube_config()

# Initialize metrics lists
cpu_usage = []
memory_usage = []
pod_count = []
load_list = []
reward_list = []

# Parameters for sinusoidal load
period = 30  # seconds for a full cycle
duration = 10 * 60  # total duration of 10 minutes
time_steps = np.arange(0, duration, 5)  # time points every 5 seconds

# Sinusoidal load varying between 0.2 and 0.8
min_load = 0.2
max_load = 0.8
amplitude = (max_load - min_load) / 2  # This will give us the range of the variation
mean_load = (max_load + min_load) / 2  # This will be the center (middle) value

# Now apply the sinusoidal function to time_steps to get the desired variation between 0.2 and 0.8
loads = mean_load + amplitude * np.sin(2 * np.pi * time_steps / period)

# Initialize the first state
current_state = np.zeros((sequence_length, state_size))

# Autoscaling loop with time progression bar
for i, load in tqdm(enumerate(loads), total=len(loads), desc="Autoscaling Progress", unit="step"):
    # Apply the load to Redis
    print(f"Current load being applied: {load}")
    send_requests_to_redis(load)  # Apply sinusoidal load to Redis cluster

    # Track the actual load applied
    load_list.append(load)  # Store load for plotting
    
    # Get avg CPU and memory usage
    next_cpu_usage, next_memory_usage = get_avg_cpu_memory()
    
    # Simulated latency and number of pods
    next_latency = np.random.uniform(0.1, 1.0)
    next_num_pods = get_pod_count()
    next_capacity = load * next_num_pods * 0.1  # Simulate capacity

    # Update the current state
    next_state = np.array([load, next_cpu_usage, next_memory_usage, next_latency, next_num_pods, next_capacity])
    current_state = np.roll(current_state, -1, axis=0)  # Shift the sequence
    current_state[-1] = next_state  # Add the new state

    # Reshape the state for LSTM input
    lstm_input = np.reshape(current_state, (1, sequence_length, state_size))
    lstm_input_tensor = torch.tensor(lstm_input, dtype=torch.float32)
    
    # Get the model's action (resize decision)
    with torch.no_grad():
        action = model(lstm_input_tensor)
        chosen_action = torch.argmax(action).item()

    # Scale pods based on chosen action
    scale_pods(chosen_action, next_num_pods)

    # Store CPU, memory, and pod count metrics
    cpu_usage.append(next_cpu_usage)
    memory_usage.append(next_memory_usage)
    pod_count.append(next_num_pods)

    # Calculate reward based on the current metrics
    reward = load * 3.0 - next_latency * 5.0 - next_num_pods * 10.0 - next_cpu_usage * 0.5 - next_memory_usage * 0.5
    reward_list.append(reward)

    # Wait for 5 seconds before the next scaling decision
    time.sleep(5)

# Plotting the results after the autoscaling process
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
plt.savefig("autoscaling_metrics2.png")
plt.show()

# Print total accumulated reward
total_reward = np.sum(reward_list)
print(f"Total Accumulated Reward: {total_reward}")

