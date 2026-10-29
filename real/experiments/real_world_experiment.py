import torch
import numpy as np
import os
import time
import subprocess
import matplotlib.pyplot as plt
from rediscluster import RedisCluster
from agentHuber import DDQNAgentLSTM
from redis.exceptions import ConnectionError, ResponseError
from rediscluster.exceptions import RedisClusterException, ClusterError

# --- CONFIGURATION ---
RESULTS_FOLDER = "new_metrics"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
REDIS_LABEL = "redis=redis-cluster-pods-lbl"
MAX_HEALTH_ATTEMPTS = 5  # Max attempts to fix cluster before skipping

# --- REDIS CONNECTION SETUP ---
def get_redis_pod_ips():
    """Fetch IP addresses of Redis pods dynamically."""
    result = subprocess.run(
        ["kubectl", "get", "pods", "-l", REDIS_LABEL, "-o", "jsonpath={.items[*].status.podIP}"],
        capture_output=True, text=True
    )
    redis_ips = result.stdout.strip().split()
    if not redis_ips:
        print("❌ No Redis pods found.")
    return redis_ips

def create_redis_client():
    """Create Redis Cluster client dynamically."""
    redis_ips = get_redis_pod_ips()
    startup_nodes = [{"host": ip, "port": "6379"} for ip in redis_ips]
    try:
        client = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)
        print(f"✅ Connected to Redis Cluster with nodes: {', '.join(redis_ips)}")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Redis Cluster: {e}")
        return None

redis_client = create_redis_client()

# --- AGENT SETUP ---
model_path = "modelCPU_10.pth"
print(f"🔍 Loading pre-trained model: {model_path}")
agent = DDQNAgentLSTM(state_size=6, action_size=3, sequence_length=10)
agent.online_model.load_state_dict(torch.load(model_path))
agent.epsilon = 0.05  # Reduce exploration for real-world testing
print("✅ Pre-trained model loaded successfully!")

# --- METRIC COLLECTION ---
timesteps = 2000
load_data, latency_data, cpu_usage_data, memory_usage_data, num_workers_data, rewards_data = [], [], [], [], [], []

# Track previous CPU and memory usage
previous_cpu_usage = 0
previous_memory_usage = 0

def get_redis_stats():
    """Fetch and calculate average CPU, memory, latency, and connected clients from Redis."""
    try:
        nodes_info = redis_client.info()

        # Fetch CPU and memory usage from all nodes
        cpu_usages = [float(node.get('used_cpu_sys', 0)) for node in nodes_info.values()]
        memory_usages = [float(node.get('used_memory', 0)) / (1024 * 1024) for node in nodes_info.values()]  # Convert to MB
        latencies = [float(node.get('instantaneous_ops_per_sec', 0)) for node in nodes_info.values()]
        connected_clients_list = [int(node.get('connected_clients', 0)) for node in nodes_info.values()]

        num_nodes = len(nodes_info)
        if num_nodes > 0:
            # Average CPU, memory, latency, and connected clients across nodes
            avg_cpu_usage = np.mean(cpu_usages)
            avg_memory_usage = np.mean(memory_usages)
            avg_latency = np.mean(latencies)
            avg_connected_clients = np.mean(connected_clients_list)

            return avg_cpu_usage, avg_memory_usage, avg_latency, avg_connected_clients
        else:
            print("⚠️ No Redis nodes found.")
            return 0, 0, 0, 0

    except (ConnectionError, ResponseError, ClusterError) as e:
        print(f"⚠️ Redis connection issue: {e}")
        return 0, 0, 0, 0

def remove_failed_nodes():
    """Automatically remove failed nodes from the cluster."""
    try:
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
        failed_nodes = [line.split()[0] for line in cluster_nodes_info.splitlines() if 'fail' in line]

        if failed_nodes:
            print(f"❌ Removing failed nodes: {failed_nodes}")
            for node_id in failed_nodes:
                subprocess.run([
                    "redis-cli", "-c", "-h", "10.244.37.144", "-p", "6379",
                    "CLUSTER", "FORGET", node_id
                ], check=True)
            print("✅ Failed nodes removed successfully.")
        else:
            print("✅ No failed nodes detected.")
    except Exception as e:
        print(f"❌ Failed to remove nodes: {e}")


def is_cluster_healthy(redis_client):
    """Check if the Redis cluster is healthy."""
    try:
        cluster_info = redis_client.cluster_info()
        print(f"📊 Cluster info: {cluster_info}")

        # If at least one node reports 'ok', consider the cluster healthy
        healthy_nodes = 0
        for node, info in cluster_info.items():
            if isinstance(info, dict):
                cluster_state = info.get('cluster_state', 'unknown')
                if cluster_state == 'ok':
                    healthy_nodes += 1
        
        if healthy_nodes == len(cluster_info):
            print("✅ All nodes report healthy. Cluster is fully healthy.")
            return True
        elif healthy_nodes > 0:
            print("⚠️ Some nodes report healthy. Proceeding with caution.")
            return True  # Proceed even if some nodes are ok
        else:
            print(f"⚠️ Cluster state: UNKNOWN, waiting for it to stabilize...")
            return False

    except Exception as e:
        print(f"⚠️ Failed to fetch cluster info: {e}")
        return False

def assign_missing_slots():
    """Automatically detect and assign missing slots to a healthy master node."""
    try:
        # Get the current slot coverage
        cluster_slots = redis_client.execute_command('CLUSTER SLOTS')
        covered_slots = set()
        for slot_range in cluster_slots:
            start_slot, end_slot = slot_range[0], slot_range[1]
            covered_slots.update(range(start_slot, end_slot + 1))

        all_slots = set(range(16384))
        missing_slots = all_slots - covered_slots

        if missing_slots:
            print(f"🔧 Assigning {len(missing_slots)} missing slots.")

            # Find healthy master nodes
            cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
            masters = [line for line in cluster_nodes_info.splitlines() if 'master' in line and 'fail' not in line]

            if not masters:
                print("❌ No available master nodes to assign slots.")
                return

            # Assign missing slots to the first healthy master node
            master_ip_port = masters[0].split()[1]
            master_ip, master_port = master_ip_port.split(':')

            for slot in missing_slots:
                subprocess.run([
                    "redis-cli", "-c", "-h", master_ip, "-p", master_port,
                    "CLUSTER", "ADDSLOTS", str(slot)
                ], check=True)

            print("✅ Missing slots assigned successfully.")
        else:
            print("✅ No missing slots detected.")

    except Exception as e:
        print(f"❌ Failed to assign missing slots: {e}")




def reconnect_redis():
    """Reconnect to the Redis cluster using the first healthy node."""
    global redis_client
    redis_client = find_healthy_node()
    if redis_client:
        print("🔄 Reconnected to Redis using a healthy node.")
    else:
        print("❌ Failed to reconnect to any healthy Redis node.")


# --- AUTOMATED CLUSTER FIXES ---
# Corrected slot assignment function
def fix_cluster_slots():
    """Detect and assign missing slots to healthy master nodes."""
    try:
        cluster_slots = redis_client.execute_command('CLUSTER SLOTS')
        covered_slots = set()

        # Collect all currently covered slots
        for slot_range in cluster_slots:
            start_slot, end_slot = slot_range[0], slot_range[1]
            covered_slots.update(range(start_slot, end_slot + 1))

        all_slots = set(range(16384))
        missing_slots = all_slots - covered_slots

        if missing_slots:
            print(f"🔧 Assigning {len(missing_slots)} missing slots.")

            # Get available master nodes
            cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
            if isinstance(cluster_nodes_info, list):
                cluster_nodes_info = "\n".join(cluster_nodes_info)
            
            masters = [line for line in cluster_nodes_info.splitlines() if 'master' in line and 'fail' not in line]

            if not masters:
                print("❌ No available master nodes to assign slots.")
                return

            # Assign missing slots to the first healthy master node
            master_ip_port = masters[0].split()[1]
            master_ip, master_port = master_ip_port.split(':')

            # Batch add slots for performance
            missing_slots_list = list(missing_slots)
            batch_size = 1000  # Assign slots in batches
            for i in range(0, len(missing_slots_list), batch_size):
                batch_slots = missing_slots_list[i:i + batch_size]
                subprocess.run([
                    "redis-cli", "-c", "-h", master_ip, "-p", master_port,
                    "CLUSTER", "ADDSLOTS", *map(str, batch_slots)
                ], check=True)

            print("✅ Missing slots assigned successfully.")
        else:
            print("✅ No missing slots detected.")

    except Exception as e:
        print(f"❌ Failed to fix slot issues: {e}")




def fix_cluster_issues():
    """Automatically detect and fix Redis cluster issues."""
    try:
        # Step 1: Assign missing slots
        remove_failed_nodes()

        assign_missing_slots()

        # Step 2: Perform cluster rebalance only if there are enough nodes
        cluster_info = redis_client.execute_command('CLUSTER NODES')
        node_count = len([line for line in cluster_info.splitlines() if 'master' in line])

        if node_count > 1:
            print("🔧 Rebalancing cluster...")
            subprocess.run([
                "redis-cli", "--cluster", "rebalance", "10.244.37.144:6379",
                "--cluster-use-empty-masters", "--cluster-yes"
            ], check=True)
            print("✅ Cluster rebalanced successfully.")
        else:
            print("⚠️ Not enough master nodes to perform rebalance.")

        # Reconnect to refresh state after rebalancing
        reconnect_redis()

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to fix cluster issues: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during cluster fix: {e}")





def find_healthy_node():
    """Scan through all Redis nodes and connect to the first healthy node."""
    redis_ips = get_redis_pod_ips()
    
    for ip in redis_ips:
        try:
            print(f"🔍 Checking node: {ip}")
            node_client = RedisCluster(startup_nodes=[{"host": ip, "port": "6379"}],
                                       decode_responses=True, skip_full_coverage_check=True,
                                       socket_timeout=5, retry_on_timeout=True)
            
            cluster_info = node_client.cluster_info()
            print(f"📊 Cluster info for {ip}: {cluster_info}")
            
            # Adjusted logic to check the nested cluster state
            for node, info in cluster_info.items():
                if isinstance(info, dict):
                    cluster_state = info.get('cluster_state', 'unknown')
                    if cluster_state == 'ok':
                        print(f"✅ Found healthy node: {ip} ({node})")
                        return node_client

        except Exception as e:
            print(f"⚠️ Failed to connect to node {ip}: {e}")
    
    print("❌ No healthy Redis nodes found in the cluster.")
    return None



def fix_cluster_issues():
    """Automatically detect and fix Redis cluster issues."""
    try:
        fix_cluster_slots()
        print("🔧 Rebalancing cluster...")
        subprocess.run([
            "redis-cli", "--cluster", "rebalance", "10.244.37.144:6379",
            "--cluster-use-empty-masters", "--cluster-yes"
        ], check=True)
        
        # Reconnect to refresh state after rebalancing
        reconnect_redis()
        print("✅ Cluster issues attempted to be fixed.")

    except Exception as e:
        print(f"❌ Failed to fix cluster issues: {e}")



# --- LOAD SIMULATION ---
def generate_sinusoidal_load(t, amplitude=100, period=1000, noise_level=5, spike_amplitude=150, spike_duration=50):
    """Generate sinusoidal load with noise and spikes."""
    load = amplitude * np.sin(2 * np.pi * t / period) + amplitude
    load += np.random.normal(0, noise_level)
    if 1500 <= t < 1500 + spike_duration:
        load += spike_amplitude * np.abs(np.sin(2 * np.pi * (t - 1500) / spike_duration))
    return max(load, 0)

def get_current_pod_count():
    """Get the current number of Redis pods."""
    pods_output = subprocess.getoutput("kubectl get pods -l redis=redis-cluster-pods-lbl --no-headers | wc -l")
    return int(pods_output.strip())

# --- CLUSTER SCALING ---
last_scale_time = 0
def scale_redis(action):
    """Scale Redis pods up or down based on the agent's action."""
    global last_scale_time
    current_time = time.time()

    if current_time - last_scale_time < 60:
        print("⏳ Throttling scaling to prevent rapid changes.")
        return

    current_pods = get_current_pod_count()
    new_pods = current_pods + 1 if action == 2 else max(1, current_pods - 1 if action == 0 else current_pods)

    if new_pods != current_pods:
        try:
            subprocess.run(["kubectl", "scale", "deployment/redis-autoscale-cluster", f"--replicas={new_pods}"], check=True)
            time.sleep(15)  # Allow time for pods to start
            reconnect_redis()
            fix_cluster_issues()  # Automatically fix cluster after scaling
        except subprocess.CalledProcessError as e:
            print(f"❌ Scaling command failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error during scaling: {e}")

    last_scale_time = current_time
    time.sleep(5)



def rebalance_slots():
    """Automatically rebalance slots after scaling operations."""
    try:
        subprocess.run([
            "redis-cli", "--cluster", "rebalance", "10.244.37.144:6379",
            "--cluster-use-empty-masters", "--cluster-yes"
        ], check=True)
        print("✅ Slots rebalanced successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Slot rebalancing failed: {e}")


# --- MAIN LOOP ---
health_attempts = 0
state = np.zeros((agent.sequence_length, agent.state_size))  # Initialize the state for LSTM

for t in range(timesteps):
    # --- Check Cluster Health ---
    if not is_cluster_healthy(redis_client):
        print(f"⚠️ Cluster not healthy, attempting to fix... (Attempt {health_attempts + 1})")
        fix_cluster_issues()
        health_attempts += 1
        
        if health_attempts >= MAX_HEALTH_ATTEMPTS:
            print("❌ Max health attempts reached. Continuing experiment with partial data.")
            break  # Instead of skipping, break and proceed to plotting
        
        time.sleep(5)
        continue
    else:
        health_attempts = 0  # Reset health attempts if cluster is healthy

    # --- Generate Load ---
    load = generate_sinusoidal_load(t)
    try:
        redis_client.set(f"load_key_{t}", load)
        stored_value = redis_client.get(f"load_key_{t}")
        if float(stored_value) != load:
            raise ValueError("Mismatch between written and read load values.")
    except Exception as e:
        print(f"⚠️ Error writing to Redis: {e}. Skipping timestep {t}.")
        continue

    # --- Collect Metrics from Redis ---
    cpu_usage, memory_usage, latency, connected_clients = get_redis_stats()
    current_pod_count = get_current_pod_count()
    print(f"📊 Metrics - Load: {load:.2f}, CPU: {cpu_usage:.2f}, Memory: {memory_usage:.2f}, Latency: {latency:.2f}, Connected Clients: {connected_clients}, Pods: {current_pod_count}")

    # --- Prepare State for Agent ---
    new_state = np.array([load, cpu_usage, memory_usage, latency, connected_clients, current_pod_count])
    next_state = np.zeros((agent.sequence_length, agent.state_size))
    next_state[:-1] = state[1:]  # Shift previous states
    next_state[-1] = new_state   # Add current metrics to the state

    # --- Agent Decides on Scaling Action ---
    action = agent.act(next_state)
    action_description = {0: "Scale Down", 1: "No Change", 2: "Scale Up"}
    print(f"🔍 Timestep {t}: Load={load:.2f}, CPU={cpu_usage:.2f}, Mem={memory_usage:.2f}, Latency={latency:.2f}, Pods={current_pod_count}, Action={action_description[action]}")

    # --- Execute Scaling Action ---
    scale_redis(action)

    # --- Record Data for Analysis ---
    load_data.append(load)
    cpu_usage_data.append(cpu_usage)
    memory_usage_data.append(memory_usage)
    latency_data.append(latency)
    num_workers_data.append(current_pod_count)

    # --- Calculate Reward ---
    reward = 3.0 * load - 5.0 * latency - 10.0 * current_pod_count - 0.5 * cpu_usage - 0.5 * memory_usage
    rewards_data.append(reward)

    # Update state for next iteration
    state = next_state

    # --- Wait for Next Timestep ---
    time.sleep(0.5)



# --- PLOTTING RESULTS ---
if load_data:
    # Proceed with plotting if data exists
    plt.figure(figsize=(12, 8))

    def plot_metric(ax, data, ylabel, title, num_workers):
        ax.plot(data, label=ylabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        axb = ax.twinx()
        axb.plot(num_workers, color='red', linestyle='--', label="Redis Workers")
        axb.set_ylabel("Number of Redis Workers")

    plot_metric(plt.subplot(2, 2, 1), load_data, "Load", "Load Over Time", num_workers_data)
    plot_metric(plt.subplot(2, 2, 2), cpu_usage_data, "CPU Usage", "CPU Usage Over Time", num_workers_data)
    plot_metric(plt.subplot(2, 2, 3), memory_usage_data, "Memory Usage", "Memory Usage Over Time", num_workers_data)
    plot_metric(plt.subplot(2, 2, 4), latency_data, "Latency", "Latency Over Time", num_workers_data)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/real_world_metrics.jpg")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(rewards_data, label='Reward')
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Accumulated Rewards Over Time")
    plt.legend()
    plt.savefig(f"{RESULTS_FOLDER}/accumulated_rewards_plot.jpg")
    plt.close()

    with open(f"{RESULTS_FOLDER}/accumulated_rewards.txt", "w") as f:
        f.write(f"Total Accumulated Reward: {sum(rewards_data)}\n")

    print(f"✅ Real-world experiment completed! Results are saved in '{RESULTS_FOLDER}'.")
else:
    print("❌ No data collected during the experiment. Check cluster health and try again.")
