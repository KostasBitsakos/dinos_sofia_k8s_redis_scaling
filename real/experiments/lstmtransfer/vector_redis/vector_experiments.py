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
import logging

# --- Configuration ---
RESULTS_FOLDER = "new_metrics2"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
REDIS_LABEL = "redis=redis-cluster-pods-lbl"
MAX_HEALTH_ATTEMPTS = 5  # Max attempts to fix cluster before skipping
timesteps = 2000  # Define the number of timesteps for the experiment

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{RESULTS_FOLDER}/autoscaler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_coordinator_ip():
    """Get the IP address of a Redis node to use as the coordinator for rebalancing."""
    try:
        # Fetch the list of Redis pod IPs
        redis_ips = get_redis_pod_ips()
        if not redis_ips:
            logger.error("❌ No Redis pods found.")
            return None

        # Return the first IP in the list as the coordinator
        coordinator_ip = redis_ips[0]
        logger.info(f"🔧 Using {coordinator_ip} as the coordinator for rebalancing.")
        return coordinator_ip
    except Exception as e:
        logger.error(f"❌ Failed to get coordinator IP: {e}")
        return None
    

def debug_cluster_nodes():
    """Debug the response format of CLUSTER NODES."""
    try:
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
        logger.info("\n📝 DEBUG: CLUSTER NODES response format:")
        logger.info(f"Type: {type(cluster_nodes_info)}")
        logger.info(f"Content: {cluster_nodes_info}")

        # If the response is a string, parse it into a list of dictionaries
        if isinstance(cluster_nodes_info, str):
            logger.info("Parsing CLUSTER NODES string response...")
            nodes = []
            for line in cluster_nodes_info.splitlines():
                parts = line.split()
                if len(parts) >= 8:
                    node_info = {
                        'id': parts[0],
                        'host': parts[1].split(':')[0],
                        'port': int(parts[1].split(':')[1]),
                        'flags': tuple(parts[2].split(',')),
                        'master': parts[3] if parts[3] != '-' else None,
                        'ping-sent': int(parts[4]),
                        'pong-recv': int(parts[5]),
                        'link-state': parts[6],
                        'slots': tuple(map(int, parts[7].split('-')) if '-' in parts[7] else (int(parts[7]), int(parts[7]))),
                        'migrations': parts[8:] if len(parts) > 8 else []
                    }
                    nodes.append(node_info)
            logger.info(f"Parsed nodes: {nodes}")
            return nodes
        else:
            return cluster_nodes_info

    except Exception as e:
        logger.error(f"❌ Failed to fetch CLUSTER NODES: {e}")
        return []



def assign_missing_slots():
    """Detect and assign missing slots to healthy master nodes."""
    try:
        cluster_slots = debug_cluster_slots()  # Use the debug function to get parsed slots
        if not cluster_slots:
            logger.error("❌ No slot information found. Your cluster may not be initialized correctly.")
            return

        covered_slots = set()
        for slot_range, info in cluster_slots.items():
            start_slot, end_slot = slot_range
            covered_slots.update(range(start_slot, end_slot + 1))

        all_slots = set(range(16384))
        missing_slots = all_slots - covered_slots

        if missing_slots:
            logger.info(f"🔧 Assigning {len(missing_slots)} missing slots.")
            # Rest of the logic...
        else:
            logger.info("✅ No missing slots detected.")

    except Exception as e:
        logger.error(f"❌ Failed to assign missing slots: {e}")

def create_vector_index(index_name="vec_index", dim=1536):
    try:
        redis_client.execute_command(
            f"FT.CREATE {index_name} ON HASH PREFIX 1 doc: "
            f"SCHEMA embedding VECTOR FLAT 6 TYPE FLOAT32 DIM {dim} DISTANCE_METRIC COSINE"
        )
        logger.info(f"✅ Vector index '{index_name}' created.")
    except Exception as e:
        logger.warning(f"⚠️ Index might already exist or creation failed: {e}")

def search_similar_vectors(query_vector, k=5, index_name="vec_index"):
    try:
        blob = query_vector.astype(np.float32).tobytes()
        result = redis_client.execute_command(
            f"FT.SEARCH {index_name} '*=>[KNN {k} @embedding $vec AS score]' "
            "PARAMS 2 vec $blob DIALECT 2"
        )
        return result
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        return []

def write_vector_to_redis(key, vector, metadata=None):
    """Write vector embedding and optional metadata to Redis as HASH."""
    try:
        vector_bytes = vector.astype(np.float32).tobytes()
        payload = {
            "embedding": vector_bytes
        }
        if metadata:
            payload.update(metadata)

        redis_client.hset(key, mapping=payload)

    except Exception as e:
        logger.error(f"❌ Failed to write vector: {e}")




# --- Redis Client Initialization ---
def refresh_redis_connection():
    """Force a Redis Cluster connection refresh after node removal or scaling."""
    global redis_client
    redis_client = create_redis_client()  # Reinitialize connection
    time.sleep(5)  # Give time for updates
    logger.info("🔄 Redis client refreshed and reconnected.")

def get_redis_pod_ips():
    """Fetch IP addresses of Redis pods dynamically."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", REDIS_LABEL, "-o", "jsonpath={.items[*].status.podIP}"],
            capture_output=True, text=True
        )
        redis_ips = result.stdout.strip().split()
        if not redis_ips:
            logger.error("❌ No Redis pods found.")
        return redis_ips
    except Exception as e:
        logger.error(f"❌ Failed to fetch Redis pod IPs: {e}")
        return []

def create_redis_client():
    """Create Redis Cluster client dynamically with retries."""
    redis_ips = get_redis_pod_ips()
    if not redis_ips:
        logger.error("❌ No Redis pods found. Exiting.")
        exit(1)

    startup_nodes = [{"host": ip, "port": "6379"} for ip in redis_ips]

    for attempt in range(3):  # Retry mechanism
        try:
            client = RedisCluster(
                startup_nodes=startup_nodes,
                decode_responses=True,
                skip_full_coverage_check=True
            )
            logger.info(f"✅ Connected to Redis Cluster with nodes: {', '.join(redis_ips)}")

            # Debug: Verify connection by running a simple command
            try:
                client.ping()
                logger.info("✅ Redis cluster is responsive.")
            except Exception as e:
                logger.error(f"❌ Redis cluster is not responsive: {e}")

            return client
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt + 1}/3: Failed to connect to Redis Cluster: {e}")
            time.sleep(5)

    logger.error("❌ All attempts to connect to Redis Cluster failed. Exiting.")
    exit(1)

# Initialize Redis client
redis_client = create_redis_client()

# --- Agent Setup ---
model_path = "modelCPU_5.pth"  # Path to the pre-trained model
logger.info(f"🔍 Loading transfer learning model: {model_path}")

# Initialize the agent with transfer learning
agent = DDQNAgentLSTM(state_size=6, action_size=3, sequence_length=10, pretrained_model=model_path)
agent.epsilon = 0.1  # Reduce exploration for real-world testing
logger.info("✅ Transfer learning model loaded successfully!")

# --- Metric Collection ---
load_data, latency_data, cpu_usage_data, memory_usage_data, num_workers_data, rewards_data = [], [], [], [], [], []

# --- Helper Functions ---
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
            logger.warning("⚠️ No Redis nodes found.")
            return 0, 0, 0, 0

    except (ConnectionError, ResponseError, ClusterError) as e:
        logger.warning(f"⚠️ Redis connection issue: {e}")
        return 0, 0, 0, 0

def remove_failed_nodes():
    """Remove failed nodes from the Redis cluster."""
    try:
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
        if not isinstance(cluster_nodes_info, list):
            logger.error(f"❌ Unexpected CLUSTER NODES response type: {type(cluster_nodes_info)}")
            return

        failed_nodes = []
        for node_info in cluster_nodes_info:
            flags = node_info.get('flags', [])
            if 'fail' in flags:
                failed_nodes.append(node_info['id'])

        if failed_nodes:
            logger.info(f"🔧 Removing failed nodes: {failed_nodes}")
            # Use one of the known healthy node addresses (first node in the list)
            healthy_host = cluster_nodes_info[0]['host']
            for node_id in failed_nodes:
                subprocess.run([
                    "redis-cli", "-c", "-h", healthy_host, "-p", "6379",
                    "CLUSTER", "FORGET", node_id
                ], check=True)
            logger.info("✅ Failed nodes removed successfully.")
        else:
            logger.info("✅ No failed nodes detected.")

    except Exception as e:
        logger.error(f"❌ Failed to remove failed nodes: {e}")


def is_cluster_healthy():
    """Check if the Redis cluster is healthy by examining the cluster info."""
    try:
        time.sleep(5)  # Allow cluster to stabilize
        cluster_info = redis_client.execute_command('CLUSTER INFO')
        logger.debug(f"Cluster info response (type: {type(cluster_info)}): {cluster_info}")
        
        # If we get a dict, check that every node reports healthy state.
        if isinstance(cluster_info, dict):
            if all(info.get('cluster_state') == 'ok' for info in cluster_info.values()):
                logger.info("✅ Cluster is healthy.")
                return True
            else:
                logger.warning("⚠️ Cluster is not healthy.")
                return False
        else:
            logger.error(f"❌ Unexpected CLUSTER INFO response type: {type(cluster_info)}")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to fetch cluster info: {e}")
        return False





def fix_cluster_issues():
    """Automatically detect and fix Redis cluster issues."""
    try:
        logger.info("🔍 Checking for failed nodes...")
        remove_failed_nodes()

        logger.info("🔍 Checking for missing slots...")
        assign_missing_slots()

        logger.info("🔄 Refreshing Redis connection before checking master nodes...")
        refresh_redis_connection()
        time.sleep(5)

        # Fetch cluster nodes info
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
        logger.debug(f"CLUSTER NODES response (type: {type(cluster_nodes_info)}): {cluster_nodes_info}")  # Debugging line

        # Ensure the response is a list of dictionaries
        if not isinstance(cluster_nodes_info, list):
            logger.error(f"❌ Unexpected CLUSTER NODES response type: {type(cluster_nodes_info)}")
            return

        # Parse the response (it's a list of dictionaries)
        active_masters = []
        replicas = []

        for node_info in cluster_nodes_info:
            flags = node_info.get('flags', [])
            master_id = node_info.get('master', None)

            # Check if the node is a master
            if 'master' in flags and 'fail' not in flags:
                active_masters.append((node_info['id'], f"{node_info['host']}:{node_info['port']}"))

            # Check if the node is a replica
            if 'slave' in flags and 'fail' not in flags:
                replicas.append((node_info['id'], f"{node_info['host']}:{node_info['port']}", master_id))

        # Check the number of active masters
        if len(active_masters) < 3:
            logger.warning(f"⚠️ Only {len(active_masters)} master nodes found. Attempting recovery...")

            if not replicas:
                logger.error("❌ No replicas available to promote. Scaling up the cluster...")
                scale_redis(2)  # Scale up the cluster to add more replicas
                time.sleep(15)  # Wait for new pods to start
                refresh_redis_connection()
                return

            # Promote the first replica to master
            replica_id, replica_ip_port, _ = replicas[0]
            replica_ip = replica_ip_port.split(':')[0]
            logger.info(f"🔧 Promoting {replica_ip} to master...")

            # Run CLUSTER FAILOVER to promote the replica
            subprocess.run(["redis-cli", "-c", "-h", replica_ip, "-p", "6379", "CLUSTER", "FAILOVER"], check=True)
            time.sleep(10)  # Wait for the promotion to complete

            # Refresh the connection and check the number of masters again
            refresh_redis_connection()
            updated_masters = redis_client.execute_command('CLUSTER NODES')
            num_updated_masters = len([node_info for node_info in updated_masters if 'master' in node_info.get('flags', []) and 'fail' not in node_info.get('flags', [])])

            if num_updated_masters >= 3:
                logger.info(f"✅ Master promotion successful! {num_updated_masters} masters available.")
            else:
                logger.error(f"❌ Master promotion failed. Only {num_updated_masters} masters detected.")

        logger.info("✅ Cluster issues fixed.")
    except Exception as e:
        logger.error(f"❌ Unexpected error during cluster fix: {e}")

                            

def force_fix_cluster_slots():
    """Forcefully fix missing slots in the Redis cluster."""
    try:
        cluster_slots = redis_client.execute_command('CLUSTER SLOTS')
        if not cluster_slots or len(cluster_slots) == 0:
            logger.warning("⚠️ No slot information found. Your cluster may not be initialized correctly.")
            return

        covered_slots = set()
        # Iterate over the dictionary returned by CLUSTER SLOTS
        for slot_range, info in cluster_slots.items():
            start_slot, end_slot = slot_range
            covered_slots.update(range(start_slot, end_slot + 1))

        all_slots = set(range(16384))
        missing_slots = all_slots - covered_slots

        if missing_slots:
            logger.info(f"🔧 {len(missing_slots)} missing slots detected. Attempting to assign...")
            assign_missing_slots()
        else:
            logger.info("✅ No missing slots detected.")
    except Exception as e:
        logger.error(f"❌ Failed to fix cluster slots: {e}")



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


def scale_redis(action):
    """Scale Redis pods up or down based on the agent's action."""
    current_pods = get_current_pod_count()
    new_pods = current_pods + 1 if action == 2 else max(3, current_pods - 1 if action == 0 else current_pods)

    if new_pods != current_pods:
        logger.info(f"🔧 Scaling Redis pods from {current_pods} to {new_pods}...")
        try:
            subprocess.run(["kubectl", "scale", "deployment/redis-autoscale-cluster", f"--replicas={new_pods}"], check=True)
            time.sleep(15)  # Allow time for pods to start
            refresh_redis_connection()
            fix_cluster_issues()  # Automatically fix cluster after scaling

            # Rebalance the cluster
            coordinator_ip = get_coordinator_ip()
            if coordinator_ip:
                logger.info("🔄 Rebalancing the cluster...")
                rebalance_result = subprocess.run(
                    ["redis-cli", "--cluster", "rebalance", f"{coordinator_ip}:6379", "--cluster-use-empty-masters", "--cluster-yes"],
                    capture_output=True, text=True
                )
                if rebalance_result.returncode != 0:
                    logger.error(f"❌ Rebalance failed: {rebalance_result.stderr}")
                    logger.error(f"❌ Rebalance stdout: {rebalance_result.stdout}")
                else:
                    logger.info("✅ Cluster rebalanced successfully.")
            else:
                logger.error("❌ No valid coordinator found. Skipping rebalance.")

            logger.info(f"✅ Successfully scaled Redis pods to {new_pods}.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Scaling command failed: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error during scaling: {e}")
    else:
        logger.info("🔧 No scaling action required.")

def plot_results():
    """Plot the experiment results."""
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

    logger.info(f"✅ Real-world experiment completed! Results are saved in '{RESULTS_FOLDER}'.")
# --- Main Loop ---
def debug_cluster_slots():
    """Debug the response format of CLUSTER SLOTS."""
    try:
        cluster_slots = redis_client.execute_command('CLUSTER SLOTS')
        logger.info("\n📝 DEBUG: CLUSTER SLOTS response format:")
        logger.info(f"Type: {type(cluster_slots)}")
        logger.info(f"Content: {cluster_slots}")

        # If the response is a list, parse it into a dictionary
        if isinstance(cluster_slots, list):
            logger.info("Parsing CLUSTER SLOTS list response...")
            slots_dict = {}
            for slot_info in cluster_slots:
                start_slot = slot_info[0]
                end_slot = slot_info[1]
                master_info = slot_info[2]
                slots_dict[(start_slot, end_slot)] = {
                    'master': (master_info[0], master_info[1]),
                    'slaves': [slave[:2] for slave in slot_info[3:]]
                }
            logger.info(f"Parsed slots: {slots_dict}")
            return slots_dict
        else:
            return cluster_slots

    except Exception as e:
        logger.error(f"❌ Failed to fetch CLUSTER SLOTS: {e}")
        return {}

# Call the debug function
debug_cluster_nodes()
def main():
    """Main execution loop for the Redis autoscaler."""
    # Debug cluster nodes and slots
    debug_cluster_nodes()
    debug_cluster_slots()

    health_attempts = 0
    state = np.zeros((agent.sequence_length, agent.state_size))  # Initialize the state for LSTM

    for t in range(timesteps):
        logger.info(f"\n⏳ Starting timestep {t}...")
        # Rest of the main logic...

        # Ensure Cluster is Healthy
        for attempt in range(3):  # Retry mechanism
            if is_cluster_healthy():
                break
            logger.warning(f"⚠️ Cluster not healthy, attempting recovery... (Attempt {attempt + 1})")
            fix_cluster_issues()
            time.sleep(10)  # Wait for cluster to stabilize

        if not is_cluster_healthy():
            logger.error(f"❌ Cluster still unhealthy after retries. Skipping timestep {t}.")
            health_attempts += 1
            if health_attempts >= MAX_HEALTH_ATTEMPTS:
                logger.error("❌ Max health attempts reached. Continuing experiment with partial data.")
                break
            continue
        else:
            logger.info("✅ Cluster recovered. Proceeding with load generation.")
            health_attempts = 0

        # Generate Load
        load = generate_sinusoidal_load(t)

        # Fix Missing Slots
        logger.info("🔍 Ensuring all slots are assigned before writing...")
        assign_missing_slots()

        # Retry Writing Load to Redis
        write_to_redis(f"load_key_{t}", load)

        # Collect Metrics
        cpu_usage, memory_usage, latency, connected_clients = get_redis_stats()
        current_pod_count = get_current_pod_count()
        logger.info(f"📊 Metrics - Load: {load:.2f}, CPU: {cpu_usage:.2f}, Memory: {memory_usage:.2f}, Latency: {latency:.2f}, Connected Clients: {connected_clients}, Pods: {current_pod_count}")

        # Prepare State for Agent
        new_state = np.array([load, cpu_usage, memory_usage, latency, connected_clients, current_pod_count])
        next_state = np.zeros((agent.sequence_length, agent.state_size))
        next_state[:-1] = state[1:]  # Shift previous states
        next_state[-1] = new_state   # Add current metrics to the state

        # Agent Decides on Scaling Action
        start_time = time.time()
        action = agent.act(next_state)
        inference_time = time.time() - start_time
        logger.info(f"🕒 Model inference time: {inference_time:.4f} seconds")
        action_description = {0: "Scale Down", 1: "No Change", 2: "Scale Up"}
        logger.info(f"🔍 Timestep {t}: Load={load:.2f}, CPU={cpu_usage:.2f}, Mem={memory_usage:.2f}, Latency={latency:.2f}, Pods={current_pod_count}, Action={action_description[action]}")

        # Execute Scaling Action
        scale_redis(action)

        # Post-Scaling Health Check
        logger.info("⏳ Waiting for Redis cluster to stabilize...")
        time.sleep(15)

        if not is_cluster_healthy():
            logger.warning("⚠️ Cluster unhealthy after scaling. Attempting to fix...")
            fix_cluster_issues()
            time.sleep(10)

            if not is_cluster_healthy():
                logger.error("❌ Cluster still unhealthy after fix attempts.")
                continue

        # Post-Scaling Rebalancing
        logger.info("🔄 Running post-scaling rebalancing...")
        coordinator_ip = get_coordinator_ip()
        if coordinator_ip:
            rebalance_result = subprocess.run(
                ["redis-cli", "--cluster", "rebalance", f"{coordinator_ip}:6379", "--cluster-use-empty-masters", "--cluster-yes"],
                capture_output=True, text=True
            )

            if rebalance_result.returncode != 0:
                logger.error(f"❌ Rebalance failed: {rebalance_result.stderr}")
            else:
                logger.info("✅ Post-scaling rebalancing successful.")
        else:
            logger.error("❌ No valid coordinator found. Skipping rebalance.")

        # Ensure No Failed Nodes Exist After Rebalancing
        logger.info("🔧 Checking for failed nodes after scaling...")
        remove_failed_nodes()

        # Record Data for Analysis
        load_data.append(load)
        cpu_usage_data.append(cpu_usage)
        memory_usage_data.append(memory_usage)
        latency_data.append(latency)
        num_workers_data.append(current_pod_count)

        # Calculate Reward
        reward = 3.0 * load - 5.0 * latency - 10.0 * current_pod_count - 0.5 * cpu_usage - 0.5 * memory_usage
        rewards_data.append(reward)

        # Update State for Next Iteration
        state = next_state

        # Delay Between Timesteps
        time.sleep(0.5)

    # Plot Results
    if load_data:
        plot_results()
    else:
        logger.error("❌ No data collected during the experiment. Check cluster health and try again.")

# --- Run the Experiment ---



if __name__ == "__main__":
    # Initialize Redis client
    redis_client = create_redis_client()

    # Debug cluster nodes
    debug_cluster_nodes()

    # Run the main experiment
    main()