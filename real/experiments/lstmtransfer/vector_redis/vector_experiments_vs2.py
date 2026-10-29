import torch
import numpy as np
import os
import time
import subprocess
import matplotlib.pyplot as plt
import csv
from rediscluster import RedisCluster
from agentHuber import DDQNAgentLSTM
from redis.exceptions import ConnectionError, ResponseError
from rediscluster.exceptions import RedisClusterException, ClusterError
import logging

# --- Configuration ---
RESULTS_FOLDER = "new_metrics2"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
REDIS_LABEL = "redis=redis-cluster-pods-lbl"
MAX_HEALTH_ATTEMPTS = 5
VECTOR_DIM = 1536
INDEX_NAME = "vec_index"
timesteps = 2000

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

# --- Vector DB ---
def create_vector_index(index_name=INDEX_NAME, dim=VECTOR_DIM):
    try:
        redis_client.execute_command(
            f"FT.CREATE {index_name} ON HASH PREFIX 1 doc:{{vec}}: SCHEMA embedding VECTOR FLAT 6 TYPE FLOAT32 DIM {dim} DISTANCE_METRIC COSINE"
        )
        logger.info(f"✅ Vector index '{index_name}' created.")
    except Exception as e:
        logger.warning(f"⚠️ Index might already exist or creation failed: {e}")


def write_vector_to_redis(key, vector, metadata=None):
    try:
        logger.info(f"📝 Writing vector to Redis with key: {key}")
        vector_bytes = vector.astype(np.float32).tobytes()
        payload = {"embedding": vector_bytes}
        if metadata:
            payload.update(metadata)
        redis_client.hset(key, mapping=payload)
    except Exception as e:
        logger.error(f"❌ Failed to write vector: {e}")

def assign_missing_slots():
    try:
        logger.info("🔍 Checking for missing slots...")
        result = subprocess.run(
            ["redis-cli", "--cluster", "fix", f"{get_redis_pod_ips()[0]}:6379", "--cluster-yes"],
            capture_output=True, text=True
        )
        logger.info(result.stdout)
        if "slots not covered" in result.stdout:
            logger.warning("⚠️ Some slots were not covered and were reassigned.")
        else:
            logger.info("✅ All cluster slots are assigned.")
    except Exception as e:
        logger.error(f"❌ Failed to assign missing slots: {e}")

def search_similar_vectors(query_vector, k=5, index_name=INDEX_NAME):
    try:
        logger.info(f"🔍 Performing KNN vector search with hash tag: {{vec}}")
        blob = query_vector.astype(np.float32).tobytes()
        result = redis_client.execute_command(
            f"FT.SEARCH {index_name} '{{vec}}=>[KNN {k} @embedding $vec AS score]' PARAMS 2 vec $blob DIALECT 2"
        )
        logger.info(f"✅ Vector search successful. Returned {len(result)} items.")
        return result
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        return []


# --- Redis Cluster Management ---
def get_redis_pod_ips():
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
    redis_ips = get_redis_pod_ips()
    if not redis_ips:
        logger.error("❌ No Redis pods found. Exiting.")
        exit(1)

    startup_nodes = [{"host": ip, "port": "6379"} for ip in redis_ips]
    for attempt in range(3):
        try:
            client = RedisCluster(
                startup_nodes=startup_nodes,
                decode_responses=False,
                skip_full_coverage_check=True
            )
            logger.info(f"✅ Connected to Redis Cluster with nodes: {', '.join(redis_ips)}")
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt + 1}/3: Failed to connect to Redis Cluster: {e}")
            time.sleep(5)
    logger.error("❌ All attempts to connect to Redis Cluster failed. Exiting.")
    exit(1)

def refresh_redis_connection():
    global redis_client
    redis_client = create_redis_client()
    time.sleep(5)
    logger.info("🔄 Redis client refreshed and reconnected.")

def rebalance_cluster():
    coordinator_ip = get_redis_pod_ips()[0]
    if coordinator_ip:
        logger.info("🔄 Rebalancing cluster...")
        result = subprocess.run(
            ["redis-cli", "--cluster", "rebalance", f"{coordinator_ip}:6379", "--cluster-use-empty-masters", "--cluster-yes"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("✅ Cluster rebalanced successfully.")
        else:
            logger.error(f"❌ Rebalance failed: {result.stderr}")

# --- Monitoring ---
def get_redis_stats():
    try:
        nodes_info = redis_client.info()
        cpu_usages = [float(node.get(b'used_cpu_sys', 0)) for node in nodes_info.values()]
        memory_usages = [float(node.get(b'used_memory', 0)) / (1024 * 1024) for node in nodes_info.values()]
        latencies = [float(node.get(b'instantaneous_ops_per_sec', 0)) for node in nodes_info.values()]
        connected_clients_list = [int(node.get(b'connected_clients', 0)) for node in nodes_info.values()]

        if nodes_info:
            return np.mean(cpu_usages), np.mean(memory_usages), np.mean(latencies), np.mean(connected_clients_list)
        return 0, 0, 0, 0
    except Exception as e:
        logger.warning(f"⚠️ Redis connection issue: {e}")
        return 0, 0, 0, 0

def get_current_pod_count():
    pods_output = subprocess.getoutput("kubectl get pods -l redis=redis-cluster-pods-lbl --no-headers | wc -l")
    return int(pods_output.strip())

def scale_redis(action):
    current_pods = get_current_pod_count()
    new_pods = current_pods + 1 if action == 2 else max(3, current_pods - 1 if action == 0 else current_pods)

    if new_pods != current_pods:
        logger.info(f"🔧 Scaling Redis pods from {current_pods} to {new_pods}...")
        try:
            subprocess.run(["kubectl", "scale", "deployment/redis-autoscale-cluster", f"--replicas={new_pods}"], check=True)
            time.sleep(20)
            refresh_redis_connection()
            assign_missing_slots()
            time.sleep(10)
            rebalance_cluster()
            time.sleep(10)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Scaling command failed: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error during scaling: {e}")
    else:
        logger.info("🔧 No scaling action required.")

# --- Load Generator ---
def generate_embedding_load():
    return np.random.rand(VECTOR_DIM).astype(np.float32)

# --- Plotting ---
def plot_results():
    plt.figure(figsize=(12, 8))

    def plot_metric(ax, data, ylabel, title, num_workers):
        ax.plot(data, label=ylabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        axb = ax.twinx()
        axb.plot(num_workers, color='red', linestyle='--', label="Redis Workers")
        axb.set_ylabel("Redis Workers")

    plot_metric(plt.subplot(2, 2, 1), cpu_usage_data, "CPU", "CPU Usage", num_workers_data)
    plot_metric(plt.subplot(2, 2, 2), memory_usage_data, "Memory", "Memory Usage", num_workers_data)
    plot_metric(plt.subplot(2, 2, 3), latency_data, "Latency", "Latency", num_workers_data)
    plot_metric(plt.subplot(2, 2, 4), rewards_data, "Reward", "Reward", num_workers_data)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/metrics.jpg")
    plt.close()

# --- CSV Export ---
def export_metrics_to_csv():
    csv_file = os.path.join(RESULTS_FOLDER, "metrics.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestep", "CPU", "Memory", "Latency", "QueryLatency", "Pods", "Reward"])
        for i in range(len(cpu_usage_data)):
            writer.writerow([
                i,
                cpu_usage_data[i],
                memory_usage_data[i],
                latency_data[i],
                query_latency_data[i],
                num_workers_data[i],
                rewards_data[i]
            ])
    logger.info(f"📁 Metrics exported to CSV: {csv_file}")

# --- Main ---
def main():
    global redis_client
    redis_client = create_redis_client()
    create_vector_index()

    agent = DDQNAgentLSTM(state_size=6, action_size=3, sequence_length=10, pretrained_model="modelCPU_5.pth")
    agent.epsilon = 0.1

    state = np.zeros((agent.sequence_length, agent.state_size))

    for t in range(timesteps):
        logger.info(f"\n⏳ Timestep {t} / {timesteps}")

        embedding = generate_embedding_load()
        assign_missing_slots()
        time.sleep(5)
        write_vector_to_redis(f"doc:{{vec}}:{t}", embedding)


        start = time.time()
        _ = search_similar_vectors(embedding)
        query_latency = time.time() - start

        cpu, mem, latency, clients = get_redis_stats()
        pods = get_current_pod_count()

        metrics = np.array([query_latency, cpu, mem, latency, clients, pods])
        next_state = np.zeros((agent.sequence_length, agent.state_size))
        next_state[:-1] = state[1:]
        next_state[-1] = metrics

        action = agent.act(next_state)
        logger.info(f"🔁 Action taken: {action}")

        scale_redis(action)

        cpu_usage_data.append(cpu)
        memory_usage_data.append(mem)
        latency_data.append(latency)
        query_latency_data.append(query_latency)
        num_workers_data.append(pods)

        reward = 3.0 * query_latency - 5.0 * latency - 10.0 * pods - 0.5 * cpu - 0.5 * mem
        rewards_data.append(reward)
        state = next_state
        time.sleep(0.5)

    logger.info("🎯 Experiment complete. Exporting plots and metrics...")
    plot_results()
    export_metrics_to_csv()

# --- Metrics Containers ---
cpu_usage_data, memory_usage_data, latency_data = [], [], []
num_workers_data, rewards_data, query_latency_data = [], [], []

if __name__ == "__main__":
    main()
