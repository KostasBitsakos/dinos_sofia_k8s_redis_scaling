import torch
import numpy as np
import os
import time
import subprocess
import matplotlib.pyplot as plt
from rediscluster import RedisCluster
from agentHuber import DDQNAgentLSTM
from redis.exceptions import ConnectionError, ResponseError
from rediscluster.exceptions import ClusterError

# --- CONFIGURATION ---
RESULTS_FOLDER = "transfer_learning_metrics"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
REDIS_LABEL = "redis=redis-cluster-pods-lbl"
MAX_HEALTH_ATTEMPTS = 5  # Max attempts to fix cluster before skipping
FINE_TUNE_INTERVAL = 50  # Fine-tune every 50 timesteps
BATCH_SIZE = 16

def validate_cluster_before_experiment():
    """Ensure the cluster is fully healthy before starting the experiment."""
    print("🔍 Validating cluster before starting the experiment...")
    if not is_cluster_healthy():
        print("⚠️ Cluster not healthy. Attempting to fix before running the experiment.")
        fix_cluster_issues()
        if not is_cluster_healthy():
            print("❌ Cluster remains unhealthy after fixes. Aborting the experiment.")
            exit(1)
    print("✅ Cluster is healthy. Starting the experiment.")

def get_coordinator_ip():
    """Fetch the IP of a healthy Redis coordinator node."""
    try:
        pods_output = subprocess.getoutput(
            "kubectl get pods -l redis=redis-cluster-pods-lbl -o jsonpath='{.items[*].status.podIP}'"
        )
        pod_ips = pods_output.strip().replace("'", "").split()

        if not pod_ips:
            print("❌ No Redis pods found.")
            return None

        for ip in pod_ips:
            result = subprocess.run(
                ["redis-cli", "-c", "-h", ip, "-p", "6379", "CLUSTER", "INFO"],
                capture_output=True, text=True
            )
            if "cluster_state:ok" in result.stdout:
                print(f"✅ Coordinator IP detected: {ip}")
                return ip  # Return the first healthy node

        print("❌ No healthy coordinator found. Aborting scaling/rebalancing.")
        return None  # Do not fall back to unhealthy nodes

    except Exception as e:
        print(f"❌ Failed to get coordinator IP: {e}")
        return None

def check_redis_connection(ip):
    """Attempt to connect to a specific Redis node, with retries."""
    for attempt in range(3):
        try:
            result = subprocess.run(["redis-cli", "-c", "-h", ip, "-p", "6379", "PING"], capture_output=True, text=True)
            if "PONG" in result.stdout:
                print(f"✅ Node {ip} is responsive.")
                return True
            else:
                print(f"⚠️ Attempt {attempt + 1}: Node {ip} not responsive. Retrying...")
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}: Error connecting to {ip}: {e}")
        time.sleep(5)
    
    print(f"❌ Node {ip} failed after 3 attempts.")
    return False



def forget_removed_nodes():
    """Ensure nodes that are removed during scaling are forgotten from the cluster."""
    try:
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
        if isinstance(cluster_nodes_info, bytes):
            cluster_nodes_info = cluster_nodes_info.decode('utf-8')

        for line in cluster_nodes_info.strip().split('\n'):
            parts = line.split()
            node_id = parts[0]
            ip_port = parts[1]
            flags = parts[2]

            if 'fail' in flags or 'handshake' in flags:
                ip = ip_port.split('@')[0].split(':')[0]
                print(f"⚠️ Node {ip} has failed status. Attempting to forget it.")
                subprocess.run(["redis-cli", "-c", "-h", get_coordinator_ip(), "-p", "6379", "CLUSTER", "FORGET", node_id], check=True)
                print(f"✅ Node {ip} successfully forgotten from cluster.")
    except Exception as e:
        print(f"❌ Failed to forget removed nodes: {e}")




def ensure_all_nodes_responsive():
    """Check if all Redis nodes are responsive before any scaling or rebalancing."""
    redis_ips = get_redis_pod_ips()
    healthy_ips = []
    for ip in redis_ips:
        try:
            result = subprocess.run(["redis-cli", "-c", "-h", ip, "-p", "6379", "PING"], capture_output=True, text=True)
            if "PONG" in result.stdout:
                print(f"✅ Node {ip} is responsive.")
                healthy_ips.append(ip)
            else:
                print(f"⚠️ Node {ip} is not responsive. Skipping...")
        except Exception as e:
            print(f"❌ Error connecting to node {ip}: {e}")
    
    if not healthy_ips:
        print("❌ No healthy nodes found. Aborting operation.")
        exit(1)
    
    return healthy_ips

   

def ensure_node_removal(ip_to_remove):
    """Ensure that the node has been successfully removed from the cluster."""
    try:
        time.sleep(5)  # Allow some buffer time after scaling
        cluster_nodes_info = subprocess.getoutput(f"redis-cli -c -h {get_coordinator_ip()} -p 6379 CLUSTER NODES")

        if ip_to_remove in cluster_nodes_info:
            print(f"⚠️ Node {ip_to_remove} still exists in cluster. Attempting to forget it.")
            subprocess.run(["redis-cli", "-c", "-h", get_coordinator_ip(), "-p", "6379", "CLUSTER", "FORGET", ip_to_remove], check=True)
            print(f"✅ Node {ip_to_remove} removed from cluster.")
        else:
            print(f"✅ Node {ip_to_remove} successfully removed from the cluster.")
    except Exception as e:
        print(f"❌ Failed to ensure node removal: {e}")


def force_cluster_sync():
    """Force the cluster to resynchronize and rebalance before starting."""
    coordinator_ip = get_coordinator_ip()
    if not coordinator_ip:
        print("❌ Cannot force sync without a healthy coordinator.")
        return

    print("🔄 Forcing full cluster sync and rebalance...")
    rebalance_result = subprocess.run(
        ["redis-cli", "--cluster", "rebalance", f"{coordinator_ip}:6379", "--cluster-use-empty-masters", "--cluster-yes"],
        capture_output=True, text=True
    )

    if rebalance_result.returncode != 0:
        print(f"⚠️ Rebalance encountered an error: {rebalance_result.stderr}")
    else:
        print("✅ Cluster rebalanced successfully.")

# Call this before the experiment starts



def remove_failed_nodes():
    """Automatically remove failed nodes from the Redis cluster."""
    try:
        cluster_nodes_info = redis_client.execute_command("CLUSTER NODES")
        if isinstance(cluster_nodes_info, list):
            cluster_nodes_info = "\n".join(map(str, cluster_nodes_info))
        elif isinstance(cluster_nodes_info, dict):
            cluster_nodes_info = "\n".join(f"{k}: {v}" for k, v in cluster_nodes_info.items())

        failed_nodes = []
        for line in cluster_nodes_info.splitlines():
            parts = line.split()
            if 'fail' in line and len(parts) > 1:
                node_id = parts[0]
                if len(node_id) == 40:  # Redis Node IDs are 40 characters long
                    failed_nodes.append(node_id)

        if not failed_nodes:
            print("✅ No failed nodes detected.")
            return

        print(f"❌ Removing failed nodes: {failed_nodes}")
        coordinator_ip = get_coordinator_ip()
        if not coordinator_ip:
            print("❌ No valid coordinator found. Cannot remove failed nodes.")
            return

        for node_id in failed_nodes:
            try:
                subprocess.run(["redis-cli", "-c", "-h", coordinator_ip, "-p", "6379", "CLUSTER", "FORGET", node_id], check=True)
                print(f"✅ Removed failed node: {node_id}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to remove node {node_id}: {e}")

    except Exception as e:
        print(f"❌ Failed to remove nodes: {e}")




def redistribute_slots_before_scaling_down():
    """Redistribute slots from the node that will be removed before scaling down."""
    try:
        pod_ips = get_redis_pod_ips()
        if len(pod_ips) <= 1:
            print("⚠️ Only one pod left. No need to redistribute slots.")
            return

        coordinator_ip = get_coordinator_ip()
        pods_excluding_coordinator = [ip for ip in pod_ips if ip != coordinator_ip]

        if not pods_excluding_coordinator:
            print("❌ All nodes are acting as coordinators. Cannot safely scale down.")
            return

        pod_to_remove = pods_excluding_coordinator[-1]
        remaining_pods = [ip for ip in pod_ips if ip != pod_to_remove]

        print(f"🔄 Preparing to redistribute slots from {pod_to_remove} before scaling down.")

        # Run rebalance to move slots off the pod to be removed
        rebalance_command = [
            "redis-cli", "--cluster", "rebalance", f"{remaining_pods[0]}:6379",
            "--cluster-use-empty-masters", "--cluster-yes"
        ]
        rebalance_result = subprocess.run(rebalance_command, capture_output=True, text=True)
        if rebalance_result.returncode == 0:
            print(f"✅ Successfully redistributed slots before removing {pod_to_remove}.")
        else:
            print(f"⚠️ Rebalance failed before scaling down: {rebalance_result.stderr}")

    except Exception as e:
        print(f"❌ Error during slot redistribution: {e}")










def validate_slot_distribution():
    """Ensure all 16384 slots are correctly distributed after scaling."""
    try:
        cluster_slots = redis_client.execute_command('CLUSTER SLOTS')
        covered_slots = set()
        for slot_range in cluster_slots:
            start_slot, end_slot = slot_range[0], slot_range[1]
            covered_slots.update(range(start_slot, end_slot + 1))
        
        if len(covered_slots) == 16384:
            print("✅ All slots are correctly distributed.")
        else:
            print(f"⚠️ Slot distribution issue detected. Only {len(covered_slots)} slots covered.")
            assign_missing_slots()
            subprocess.run(["redis-cli", "--cluster", "rebalance", f"{get_coordinator_ip()}:6379", "--cluster-use-empty-masters", "--cluster-yes"], check=True)
    except Exception as e:
        print(f"❌ Slot validation failed: {e}")

# Call it inside fix_cluster_issues()
def fix_cluster_issues():
    """Fix Redis cluster issues dynamically."""
    try:
        print("🔧 Checking and fixing cluster issues...")
        remove_failed_nodes()
        assign_missing_slots()

        coordinator_ip = get_coordinator_ip()
        if not coordinator_ip:
            print("❌ No valid coordinator found. Attempting to force rebalance from available nodes.")
            redis_ips = get_redis_pod_ips()
            if not redis_ips:
                print("❌ No Redis nodes available for forced rebalance.")
                return

            # Force rebalance from any available node
            for ip in redis_ips:
                rebalance_result = subprocess.run(
                    ["redis-cli", "--cluster", "rebalance", f"{ip}:6379", "--cluster-use-empty-masters", "--cluster-yes"],
                    capture_output=True, text=True
                )
                if rebalance_result.returncode == 0:
                    print(f"✅ Forced rebalance successful from node: {ip}")
                    return
                else:
                    print(f"⚠️ Forced rebalance failed from node {ip}: {rebalance_result.stderr}")

            print("❌ All forced rebalance attempts failed.")
            return

        cluster_info = subprocess.getoutput(f"redis-cli -c -h {coordinator_ip} -p 6379 CLUSTER INFO")
        if "cluster_state:ok" in cluster_info:
            print("✅ Cluster is already healthy. No need to rebalance.")
            return

        print("🔄 Running cluster rebalance...")
        rebalance_result = subprocess.run(
            ["redis-cli", "--cluster", "rebalance", f"{coordinator_ip}:6379", "--cluster-use-empty-masters", "--cluster-yes"],
            capture_output=True, text=True
        )

        if rebalance_result.returncode != 0:
            print(f"⚠️ Rebalance encountered an error: {rebalance_result.stderr}")
        else:
            print("✅ Cluster rebalanced successfully.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to fix cluster issues: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during cluster fix: {e}")





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

def create_redis_client_with_retries(max_attempts=5):
    """Retry connecting to Redis cluster multiple times before failing."""
    attempts = 0
    while attempts < max_attempts:
        try:
            client = create_redis_client()
            if client:
                return client
        except Exception as e:
            print(f"⚠️ Attempt {attempts + 1} to reconnect to Redis failed: {e}")
        attempts += 1
        time.sleep(5)

    print("❌ Failed to reconnect to Redis after multiple attempts.")
    return None

# Use this version after scaling or fixing cluster issues


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


redis_client = create_redis_client_with_retries()

# --- AGENT SETUP ---
model_path = "modelCPU_5.pth"  # Using transfer learning model
print(f"🔍 Loading transfer learning model: {model_path}")
agent = DDQNAgentLSTM(state_size=6, action_size=3, sequence_length=10, pretrained_model=model_path)
agent.epsilon = 0.05  # Reduce exploration for real-world testing
print("✅ Transfer learning model loaded successfully!")

# --- METRIC COLLECTION ---
timesteps = 2000
load_data, latency_data, cpu_usage_data, memory_usage_data, num_workers_data, rewards_data = [], [], [], [], [], []

# --- Redis Stats ---
def get_redis_stats():
    """Fetch and calculate average CPU, memory, latency, and connected clients from Redis."""
    try:
        nodes_info = redis_client.info()
        cpu_usages = [float(node.get('used_cpu_sys', 0)) for node in nodes_info.values()]
        memory_usages = [float(node.get('used_memory', 0)) / (1024 * 1024) for node in nodes_info.values()]  # Convert to MB
        latencies = [float(node.get('instantaneous_ops_per_sec', 0)) for node in nodes_info.values()]
        connected_clients_list = [int(node.get('connected_clients', 0)) for node in nodes_info.values()]

        if nodes_info:
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

# --- Cluster Health and Management ---




def assign_missing_slots():
    """Automatically detect and assign missing slots to a healthy master node."""
    try:
        cluster_slots = redis_client.execute_command('CLUSTER SLOTS')
        if not cluster_slots:
            print("❌ No slot information available.")
            return

        covered_slots = set()
        for slot_range in cluster_slots:
            start_slot, end_slot = slot_range[0], slot_range[1]
            covered_slots.update(range(start_slot, end_slot + 1))

        all_slots = set(range(16384))
        missing_slots = all_slots - covered_slots

        if missing_slots:
            print(f"🔧 Assigning {len(missing_slots)} missing slots.")

            # Find a valid master to assign the slots
            cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
            masters = [
                line for line in cluster_nodes_info.splitlines()
                if 'master' in line and 'fail' not in line
            ]

            if not masters:
                print("❌ No available master nodes to assign slots.")
                return

            master_ip_port = masters[0].split()[1]
            master_ip, master_port = master_ip_port.split(':')
            print(f"🔄 Assigning slots to master: {master_ip}:{master_port}")

            # Reattempt assigning slots
            for slot in missing_slots:
                subprocess.run(["redis-cli", "-c", "-h", master_ip, "-p", master_port, "CLUSTER", "ADDSLOTS", str(slot)], check=True)

            print("✅ Missing slots assigned successfully.")
        else:
            print("✅ No missing slots detected.")
    except Exception as e:
        print(f"❌ Failed to assign missing slots: {e}")



def is_cluster_healthy():
    """Check if Redis cluster is in a healthy state."""
    try:
        coordinator_ip = get_coordinator_ip()
        if not coordinator_ip:
            print("❌ No valid Redis node found for health check.")
            return False

        cluster_info = subprocess.getoutput(f"redis-cli -c -h {coordinator_ip} -p 6379 CLUSTER INFO")

        if 'cluster_state:ok' in cluster_info:
            print("✅ Cluster state is OK.")
            return True
        else:
            print("⚠️ Cluster state is not 'ok'. Investigating...")
            return False
    except Exception as e:
        print(f"❌ Failed to check cluster health: {e}")
        return False




def recover_failed_nodes():
    """Identify and remove failed nodes from the cluster."""
    try:
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')

        if isinstance(cluster_nodes_info, list):
            cluster_nodes_info = "\n".join(str(node) for node in cluster_nodes_info)
        elif isinstance(cluster_nodes_info, dict):
            cluster_nodes_info = "\n".join(f"{k}: {v}" for k, v in cluster_nodes_info.items())

        failed_nodes = [line.split()[0] for line in cluster_nodes_info.splitlines() if 'fail' in line]

        if failed_nodes:
            print(f"⚠️ Detected failed nodes: {failed_nodes}")
            for node_id in failed_nodes:
                coordinator_ip = get_coordinator_ip()
                if coordinator_ip:
                    subprocess.run(["redis-cli", "-c", "-h", coordinator_ip, "-p", "6379", "CLUSTER", "FORGET", node_id], check=True)
                    print(f"✅ Removed failed node {node_id}")
                else:
                    print("❌ No healthy coordinator found to remove failed nodes.")
        else:
            print("✅ No failed nodes detected.")
    except Exception as e:
        print(f"❌ Failed to recover nodes: {e}")



def fix_cluster_issues():
    """Fix Redis cluster issues dynamically."""
    try:
        print("🔧 Checking and fixing cluster issues...")
        remove_failed_nodes()
        assign_missing_slots()

        coordinator_ip = get_coordinator_ip()
        if not coordinator_ip:
            print("❌ No valid coordinator found. Skipping rebalance.")
            return

        cluster_info = subprocess.getoutput(f"redis-cli -c -h {coordinator_ip} -p 6379 CLUSTER INFO")
        if "cluster_state:ok" in cluster_info:
            print("✅ Cluster is already healthy. No need to rebalance.")
            return

        print("🔄 Running cluster rebalance...")
        rebalance_result = subprocess.run(["redis-cli", "--cluster", "rebalance", f"{coordinator_ip}:6379", "--cluster-use-empty-masters", "--cluster-yes"], capture_output=True, text=True)

        if rebalance_result.returncode != 0:
            print(f"⚠️ Rebalance encountered an error: {rebalance_result.stderr}")
        else:
            print("✅ Cluster rebalanced successfully.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to fix cluster issues: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during cluster fix: {e}")





def reconnect_redis():
    """Reconnect to Redis after scaling or fixing issues."""
    global redis_client
    redis_ips = get_redis_pod_ips()  # Refresh the list of pod IPs
    startup_nodes = [{"host": ip, "port": "6379"} for ip in redis_ips]
    
    try:
        redis_client = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)
        print(f"✅ Connected to Redis Cluster with nodes: {', '.join(redis_ips)}")
    except Exception as e:
        print(f"❌ Failed to reconnect to Redis Cluster: {e}")


# --- Load Simulation ---
def generate_sinusoidal_load(t, amplitude=100, period=1000, noise_level=5, spike_amplitude=150, spike_duration=50):
    load = amplitude * np.sin(2 * np.pi * t / period) + amplitude
    load += np.random.normal(0, noise_level)
    if 1500 <= t < 1500 + spike_duration:
        load += spike_amplitude * np.abs(np.sin(2 * np.pi * (t - 1500) / spike_duration))
    return max(load, 0)

# --- Scaling Redis Pods ---
def get_current_pod_count():
    pods_output = subprocess.getoutput("kubectl get pods -l redis=redis-cluster-pods-lbl --no-headers | wc -l")
    return int(pods_output.strip())

def scale_redis(action):
    current_pods = get_current_pod_count()
    new_pods = current_pods + 1 if action == 2 else max(1, current_pods - 1 if action == 0 else current_pods)

    if new_pods != current_pods:
        print(f"🔄 Preparing to scale Redis: Current Pods={current_pods}, New Pods={new_pods}")

        healthy_nodes = ensure_all_nodes_responsive()
        print(f"✅ Healthy nodes before scaling: {healthy_nodes}")

        if new_pods < current_pods:
            redistribute_slots_before_scaling_down()

        try:
            subprocess.run(["kubectl", "scale", "deployment/redis-autoscale-cluster", f"--replicas={new_pods}"], check=True)
            print("⏳ Waiting for Redis pods to stabilize after scaling...")
            time.sleep(30)
            
            print("🔄 Reconnecting to Redis after scaling...")
            reconnect_redis()

            # Skip rebalance if only one pod is left
            if new_pods == 1:
                print("✅ Only one node remains. Skipping rebalancing.")
                return

            # Post-scaling rebalancing
            print("🔄 Running post-scaling rebalancing...")
            rebalance_result = subprocess.run(
                ["redis-cli", "--cluster", "rebalance", f"{get_coordinator_ip()}:6379", "--cluster-use-empty-masters", "--cluster-yes"],
                capture_output=True, text=True
            )

            if rebalance_result.returncode != 0:
                print(f"❌ Rebalance failed: {rebalance_result.stderr}")
            else:
                print("✅ Post-scaling rebalancing successful.")

            if not is_cluster_healthy():
                print("⚠️ Cluster unhealthy after scaling. Fixing issues...")
                fix_cluster_issues()

        except subprocess.CalledProcessError as e:
            print(f"❌ Scaling command failed: {e}")
            reconnect_redis()
            fix_cluster_issues()









force_cluster_sync()
validate_cluster_before_experiment()

# --- Ensure Cluster is Healthy Before Experiment Starts ---
validate_cluster_before_experiment()

# --- MAIN LOOP with Dynamic Transfer Learning ---
state = np.zeros((agent.sequence_length, agent.state_size))
health_attempts = 0

for t in range(timesteps):
    if not is_cluster_healthy():
        print(f"⚠️ Cluster not healthy, attempting to fix... (Attempt {health_attempts + 1})")
        fix_cluster_issues()
        health_attempts += 1
        if health_attempts >= MAX_HEALTH_ATTEMPTS:
            print("❌ Max health attempts reached. Exiting experiment.")
            break
        time.sleep(5)
        continue
    else:
        health_attempts = 0

    load = generate_sinusoidal_load(t)

    # 🆕 Retry mechanism for Redis writes
    for attempt in range(3):
        try:
            redis_client.set(f"load_key_{t}", load)
            break  # Exit the retry loop if successful
        except Exception as e:
            print(f"⚠️ Retry {attempt + 1}: Error writing to Redis: {e}")
            time.sleep(5)  # Wait before retrying
    else:
        print(f"❌ Failed to write load at timestep {t} after 3 attempts. Skipping.")
        continue


    cpu_usage, memory_usage, latency, connected_clients = get_redis_stats()
    current_pod_count = get_current_pod_count()

    new_state = np.array([load, cpu_usage, memory_usage, latency, connected_clients, current_pod_count])
    next_state = np.zeros((agent.sequence_length, agent.state_size))
    next_state[:-1] = state[1:]
    next_state[-1] = new_state

    action = agent.act(next_state)
    scale_redis(action)

    reward = 3.0 * load - 5.0 * latency - 10.0 * current_pod_count - 0.5 * cpu_usage - 0.5 * memory_usage
    agent.remember(state, action, reward, next_state, False)

    if t % FINE_TUNE_INTERVAL == 0 and len(agent.memory) > BATCH_SIZE:
        print(f"🔄 Fine-tuning the model at timestep {t}")
        agent.replay(BATCH_SIZE)

    load_data.append(load)
    cpu_usage_data.append(cpu_usage)
    memory_usage_data.append(memory_usage)
    latency_data.append(latency)
    num_workers_data.append(current_pod_count)
    rewards_data.append(reward)

    state = next_state
    time.sleep(0.5)

# --- Save Fine-Tuned Model ---
fine_tuned_model_path = os.path.join(RESULTS_FOLDER, "modelCPU_5_finetuned.pth")
agent.save_model(fine_tuned_model_path)
print(f"✅ Fine-tuned model saved to: {fine_tuned_model_path}")

# --- Plotting Results ---
if load_data:
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
    plt.savefig(f"{RESULTS_FOLDER}/real_world_metrics_transfer_learning.jpg")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(rewards_data, label='Reward')
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Accumulated Rewards Over Time")
    plt.legend()
    plt.savefig(f"{RESULTS_FOLDER}/accumulated_rewards_transfer_learning.jpg")
    plt.close()

    with open(f"{RESULTS_FOLDER}/accumulated_rewards.txt", "w") as f:
        f.write(f"Total Accumulated Reward: {sum(rewards_data)}\n")

    print(f"✅ Transfer learning real-world experiment completed! Results are saved in '{RESULTS_FOLDER}'.")
else:
    print("❌ No data collected during the experiment. Check cluster health and try again.")
