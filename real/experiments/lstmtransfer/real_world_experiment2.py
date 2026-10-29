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


# ✅ Define the original three master nodes at the start of the script
# Define original master nodes (update these IPs if they change)
# ✅ Define the original three master nodes at the start of the script
ORIGINAL_MASTERS = {
    "10.244.37.156:6379",
    "10.244.37.151:6379",
    "10.244.37.182:6379"
}









# --- CONFIGURATION ---
RESULTS_FOLDER = "new_metrics2"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
REDIS_LABEL = "redis=redis-cluster-pods-lbl"
MAX_HEALTH_ATTEMPTS = 5  # Max attempts to fix cluster before skipping

def pre_experiment_cluster_check():
    """Ensure the cluster has at least 3 masters before running the experiment."""
    print("🔍 Performing pre-experiment cluster health check...")

    # ✅ Retry mechanism for handling Redis connection errors
    attempts = 5  # Max retries
    for attempt in range(attempts):
        try:
            cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
            if isinstance(cluster_nodes_info, list):
                cluster_nodes_info = "\n".join([str(node) for node in cluster_nodes_info])

            active_masters = [line for line in cluster_nodes_info.splitlines() if 'master' in line and 'fail' not in line]

            if len(active_masters) >= 3:
                print("✅ Cluster is healthy with at least 3 master nodes.")
                return
            else:
                print(f"⚠️ Only {len(active_masters)} master nodes found. Attempting to scale up...")

                # ✅ Scale up to 3 master nodes
                subprocess.run(["kubectl", "scale", "deployment/redis-autoscale-cluster", "--replicas=3"], check=True)
                time.sleep(30)  # Allow time for new nodes to start

                # ✅ Reconnect Redis client after scaling
                refresh_redis_connection()
                time.sleep(10)  # Allow for cluster stabilization
                
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/{attempts} failed: {e}")
            time.sleep(5)  # Wait before retrying

    print("❌ Failed to verify cluster health after multiple attempts. Exiting.")
    exit(1)


import re
def debug_cluster_nodes():
    try:
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
        print("\n📝 FULL RAW CLUSTER NODES OUTPUT (Direct from Redis):")
        print(cluster_nodes_info)
        print("\n✅ END OF RAW OUTPUT\n")
    except Exception as e:
        print(f"❌ ERROR: Failed to fetch cluster nodes: {e}")


def parse_cluster_nodes(cluster_nodes_info):
    """Parses the Redis CLUSTER NODES output and removes slot details from debug output."""
    print("\n📝 ENTERED `parse_cluster_nodes` FUNCTION")
    
    parsed_nodes = {}

    if isinstance(cluster_nodes_info, list):
        print(f"✅ CLUSTER NODES received as list with {len(cluster_nodes_info)} entries.")
    elif isinstance(cluster_nodes_info, dict):
        print(f"🛑 ERROR: CLUSTER NODES returned a dict (unexpected). Contents:\n{cluster_nodes_info}")
        return {}
    else:
        print(f"⚠️ Unexpected format received for CLUSTER NODES: {type(cluster_nodes_info)}")
        return {}

    # ✅ Process each node and exclude slots from debug print
    for node in cluster_nodes_info:
        node_id = node.get("id", "UNKNOWN_ID")
        ip_address = node.get("host", "UNKNOWN_HOST")
        is_master = "master" in node.get("flags", [])  
        master_id = node.get("master", None)

        parsed_nodes[ip_address] = {
            "node_id": node_id,
            "ip": ip_address,
            "is_master": is_master,
            "master_id": master_id if master_id else None,
        }

    # ✅ Debugging: Print cleaned node information without slots
    print("\n✅ FINAL PARSED CLUSTER NODES (Slots Removed):")
    for ip, info in parsed_nodes.items():
        print(f"🔹 IP: {ip}, Node ID: {info['node_id']}, Master: {info['is_master']}, Master ID: {info['master_id']}")

    return parsed_nodes


















def scale_down_cluster():
    """
    Scale down the cluster by removing non-original nodes.
    """
    cluster_info = get_cluster_info()  # Fetch current cluster state
    all_nodes = list(cluster_info.keys())  # Get all nodes in the cluster
    removable_nodes = [node for node in all_nodes if node not in ORIGINAL_MASTERS]

    if not removable_nodes:
        print("⚠️ No removable nodes found. Scale-down aborted.")
        return

    node_to_remove = removable_nodes[0]  # Pick a node to remove
    print(f"🔻 Scaling down: Removing node {node_to_remove}")

    try:
        redis_cli = get_redis_client()
        redis_cli.execute_command(f"CLUSTER FORGET {node_to_remove}")
        print(f"✅ Successfully removed {node_to_remove} from the cluster.")
    except Exception as e:
        print(f"❌ Failed to remove {node_to_remove}: {e}")


def can_scale_down(cluster_info):
    """
    Determines if the Redis cluster can scale down.
    - Ensures the original three master nodes remain.
    - Prefers removing non-masters first.
    """
    if cluster_info is None:
        print("❌ Cluster info is None. Cannot determine if scale-down is possible.")
        return False

    master_nodes = [node for node, info in cluster_info.items() if info["is_master"]]
    replica_nodes = [node for node, info in cluster_info.items() if not info["is_master"]]

    removable_nodes = [node for node in cluster_info.keys() if node not in ORIGINAL_MASTERS]

    print(f"📊 DEBUG: Masters={len(master_nodes)}, Replicas={len(replica_nodes)}, Removable Nodes={removable_nodes}")

    # ✅ Prioritize removing replicas first
    if replica_nodes:
        print(f"✅ Scale-down possible. Can remove replica {replica_nodes[0]}.")
        return True

    # ❌ Prevent removal of masters if only 3 remain
    if len(master_nodes) <= 3:
        print("❌ Scale-down aborted. Cannot remove any more master nodes.")
        return False

    # ✅ If more than 3 masters exist, allow removing one
    if removable_nodes:
        print(f"✅ Scale-down possible. Can remove extra master {removable_nodes[0]}.")
        return True

    print("❌ Scale-down aborted. No removable nodes available.")
    return False






def verify_slot_distribution():
    """Ensures that all 16384 slots are correctly assigned after scaling."""
    try:
        cluster_slots = redis_client.execute_command('CLUSTER SLOTS')
        assigned_slots = set()

        for slot_range in cluster_slots:
            start_slot, end_slot = slot_range[:2]
            assigned_slots.update(range(start_slot, end_slot + 1))

        if len(assigned_slots) == 16384:
            print("✅ Slot distribution is correct.")
            return True
        else:
            print(f"⚠️ Only {len(assigned_slots)}/16384 slots assigned! Running rebalance...")
            subprocess.run([
                "redis-cli", "--cluster", "rebalance", f"{get_coordinator_ip()}:6379",
                "--cluster-use-empty-masters", "--cluster-yes"
            ], check=True)
            return False
    except Exception as e:
        print(f"❌ Failed to verify slot distribution: {e}")
        return False


def wait_for_pods_ready(expected_pods):
    """Wait for the expected number of Redis pods to be in Running state AND responsive."""
    print(f"⏳ Waiting for {expected_pods} Redis pods to become Ready...")

    max_retries = 10
    for i in range(max_retries):
        pods_output = subprocess.getoutput("kubectl get pods -l redis=redis-cluster-pods-lbl --no-headers")
        running_pods = sum(1 for line in pods_output.split("\n") if "Running" in line)

        if running_pods >= expected_pods:
            print(f"✅ {running_pods}/{expected_pods} Redis pods are Running.")

            # ✅ Check if Redis responds to PING
            redis_ips = get_active_cluster_nodes()
            healthy_count = 0

            for ip in redis_ips:
                try:
                    result = subprocess.run(["redis-cli", "-c", "-h", ip, "-p", "6379", "PING"], capture_output=True, text=True)
                    if "PONG" in result.stdout:
                        healthy_count += 1
                except Exception:
                    continue

            if healthy_count >= expected_pods:
                print(f"✅ {healthy_count}/{expected_pods} Redis nodes are responsive.")
                return True
            else:
                print(f"⚠️ Only {healthy_count}/{expected_pods} Redis nodes responded to PING.")

        time.sleep(5)

    print("❌ Timed out waiting for Redis pods to be Ready.")
    return False


import subprocess

import subprocess

def get_cluster_info(retries=3, delay=5):
    """
    Retrieves detailed Redis cluster node info and identifies master/slave nodes.
    """
    redis_nodes = ["10.244.37.200:6379", "10.244.37.203:6379", "10.244.37.204:6379"]  # Hardcoded nodes (WRONG)

    
    for attempt in range(retries):
        for node in redis_nodes:
            try:
                print(f"🔍 Attempting to get cluster info from {node} (Attempt {attempt + 1})")
                
                # Run Redis CLI command to get cluster nodes
                result = subprocess.run(
                    ["redis-cli", "-h", node.split(":")[0], "-p", node.split(":")[1], "cluster", "nodes"],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    parsed_info = parse_cluster_nodes(result.stdout)
                    if parsed_info:
                        return parsed_info  # Return valid cluster info
                    else:
                        print(f"⚠️ Warning: Parsed cluster info is empty from {node}. Retrying...")

            except Exception as e:
                print(f"⚠️ Error retrieving cluster info from {node}: {e}")

        time.sleep(delay)  # Wait before retrying

    print("❌ Failed to retrieve cluster info after multiple attempts.")
    return None  # Return None if all attempts fail


def get_active_cluster_nodes():
    """Fetch the list of currently active Redis pods from Kubernetes."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", "redis=redis-cluster-pods-lbl", "-o", "jsonpath={.items[*].status.podIP}"],
            capture_output=True, text=True
        )
        pod_ips = result.stdout.strip().split()
        if not pod_ips:
            print("❌ No active Redis pods found.")
            return []
        return [f"{ip}:6379" for ip in pod_ips]
    except Exception as e:
        print(f"❌ Failed to retrieve active cluster nodes: {e}")
        return []

def get_cluster_info(retries=3, delay=5):
    """
    Retrieves detailed Redis cluster node info dynamically.
    """
    redis_nodes = get_active_cluster_nodes()  # Get current active nodes
    
    if not redis_nodes:
        print("❌ No active Redis nodes available. Cannot fetch cluster info.")
        return None
    
    for attempt in range(retries):
        for node in redis_nodes:
            try:
                print(f"🔍 Attempting to get cluster info from {node} (Attempt {attempt + 1})")
                
                # Run Redis CLI command to get cluster nodes
                result = subprocess.run(
                    ["redis-cli", "-h", node.split(":")[0], "-p", node.split(":")[1], "CLUSTER", "NODES"],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    parsed_info = parse_cluster_nodes(result.stdout)
                    if parsed_info:
                        return parsed_info  # Return valid cluster info
                    else:
                        print(f"⚠️ Warning: Parsed cluster info is empty from {node}. Retrying...")

            except Exception as e:
                print(f"⚠️ Error retrieving cluster info from {node}: {e}")

        time.sleep(delay)  # Wait before retrying

    print("❌ Failed to retrieve cluster info after multiple attempts.")
    return None  # Return None if all attempts fail

# import redis

# def get_cluster_nodes(redis_client):
#     nodes_info = redis_client.execute_command('CLUSTER NODES')
#     nodes = []
#     for line in nodes_info.splitlines():
#         parts = line.split()
#         node_id = parts[0]
#         addr = parts[1]
#         flags = parts[2].split(',')
#         master_id = parts[3]
#         if 'slave' in flags:
#             role = 'replica'
#         elif 'master' in flags:
#             role = 'master'
#         else:
#             role = 'unknown'
#         nodes.append({'id': node_id, 'addr': addr, 'role': role, 'master_id': master_id})
#     return nodes

# def select_replica_node_for_removal(nodes):
#     for node in nodes:
#         if node['role'] == 'replica':
#             return node
#     return None

# def remove_node(redis_client, node_id):
#     # Ensure the node is empty before removal
#     # Migrate slots if necessary
#     redis_client.execute_command('CLUSTER FORGET', node_id)





coordinator_ip = None  # Global variable


def get_coordinator_ip():
    """
    Find a healthy Redis node to use as the coordinator.
    """
    try:
        pod_ips = subprocess.getoutput(
            "kubectl get pods -l redis=redis-cluster-pods-lbl -o jsonpath='{.items[*].status.podIP}'"
        ).strip().split()

        if not pod_ips:
            print("❌ No Redis pods found in Kubernetes.")
            return None

        for ip in pod_ips:
            result = subprocess.run(
                ["redis-cli", "-c", "-h", ip, "-p", "6379", "CLUSTER", "INFO"],
                capture_output=True, text=True
            )

            if "cluster_state:ok" in result.stdout:
                print(f"✅ Coordinator IP found: {ip}")
                return ip

        print("❌ No explicitly healthy coordinator found, but all nodes report `cluster_state:ok`.")
        return pod_ips[0] if pod_ips else None

    except Exception as e:
        print(f"❌ Failed to get coordinator IP: {e}")
        return None







def refresh_redis_connection():
    """Force a Redis Cluster connection refresh after node removal or scaling."""
    global redis_client
    redis_client = create_redis_client()  # Reinitialize connection
    time.sleep(5)  # Give time for updates
    print("🔄 Redis client refreshed and reconnected.")

import subprocess
import os
import time

def force_fix_cluster_slots():
    """Forcefully assign missing slots to active master nodes in the Redis Cluster."""
    print("🔍 Checking for missing slots...")

    for attempt in range(5):
        try:
            cluster_slots = redis_client.execute_command('CLUSTER SLOTS')

            if not cluster_slots or len(cluster_slots) == 0:
                print(f"⚠️ Attempt {attempt + 1}: No slot information found. Retrying in 5 seconds...")
                time.sleep(5)
                continue  # Retry if no slots found
            break  # ✅ Exit loop if slot data is retrieved
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}: Failed to fetch slot data: {e}")
            time.sleep(5)

    # ✅ Emergency cluster reset if slots are still missing
    if not cluster_slots or len(cluster_slots) == 0:
        print("❌ Cluster slot information is empty after retries. Attempting emergency cluster reset.")

        # Ensure at least 3 Redis nodes are running
        print("🔄 Ensuring 3 Redis master nodes exist...")
        os.system("kubectl scale deployment redis-autoscale-cluster --replicas=3")
        time.sleep(10)

        # Re-initialize the cluster
        redis_ips = subprocess.getoutput(
            "kubectl get pods -l redis=redis-cluster-pods-lbl -o jsonpath='{.items[*].status.podIP}' | tr ' ' '\n' | awk '{print $1\":6379\"}' | paste -sd ' '"
        ).strip()

        print(f"🔄 Recreating Redis Cluster with nodes: {redis_ips}")
        os.system(f"kubectl exec -it $(kubectl get pods -l redis=redis-cluster-pods-lbl -o jsonpath=\"{{.items[0].metadata.name}}\") -- redis-cli --cluster create {redis_ips} --cluster-replicas 0 --cluster-yes")
        
        time.sleep(10)  # Wait for stabilization
        refresh_redis_connection()
        return force_fix_cluster_slots()  # Retry fixing slots after reset

    print("✅ Cluster slots retrieved successfully.")
    
    covered_slots = set()
    for slot_range in cluster_slots:
        if len(slot_range) >= 2:
            start_slot, end_slot = slot_range[:2]
            covered_slots.update(range(start_slot, end_slot + 1))

    all_slots = set(range(16384))
    missing_slots = all_slots - covered_slots

    if not missing_slots:
        print("✅ No missing slots detected.")
        return

    print(f"⚠️ {len(missing_slots)} slots are missing! Assigning them now...")

    # ✅ Get available master nodes
    try:
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')

        if isinstance(cluster_nodes_info, list):
            cluster_nodes_info = "\n".join([str(node) for node in cluster_nodes_info])

        masters = [
            line.split()[1] for line in cluster_nodes_info.splitlines()
            if 'master' in line and 'fail' not in line
        ]

    except Exception as e:
        print(f"❌ Failed to retrieve cluster nodes: {e}")
        return

    if not masters:
        print("❌ No available master nodes to assign slots. Attempting failover...")

        # Try promoting a replica to master
        replicas = [
            line.split()[1] for line in cluster_nodes_info.splitlines()
            if 'slave' in line and 'fail' not in line
        ]

        if replicas:
            replica_ip = replicas[0].split(":")[0]
            print(f"🔧 Promoting replica {replica_ip} to master...")
            try:
                subprocess.run(["redis-cli", "-c", "-h", replica_ip, "-p", "6379", "CLUSTER", "FAILOVER"], check=True)
                time.sleep(10)  # Allow time for failover
                refresh_redis_connection()
                return force_fix_cluster_slots()  # Re-attempt slot assignment
            except Exception as e:
                print(f"❌ Failover attempt failed: {e}")
                return
        else:
            print("❌ No replicas available to promote. Manual intervention required.")
            return

    # ✅ Assign missing slots evenly among all masters
    num_masters = len(masters)
    missing_slots_list = list(missing_slots)

    for i, master_ip_port in enumerate(masters):
        try:
            master_ip, master_port = master_ip_port.split(':')
        except ValueError:
            print(f"❌ Invalid master format: {master_ip_port}. Skipping...")
            continue

        assigned_slots = missing_slots_list[i::num_masters]  # Round-robin slot assignment

        if assigned_slots:
            try:
                subprocess.run([
                    "redis-cli", "-c", "-h", master_ip, "-p", master_port,
                    "CLUSTER", "ADDSLOTS", *map(str, assigned_slots)
                ], check=True)
                print(f"✅ Assigned {len(assigned_slots)} slots to {master_ip}:{master_port}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to assign slots to {master_ip}:{master_port}: {e}")

    print("✅ Missing slots assigned successfully. Running final rebalance...")
    rebalance_slots()






def fix_cluster_slots():
    """Detect and assign missing slots to healthy master nodes."""
    max_retries = 5
    for attempt in range(max_retries):
        print(f"🔍 Attempt {attempt + 1}/{max_retries}: Checking for missing slots...")

        try:
            cluster_slots = redis_client.execute_command('CLUSTER SLOTS')

            if not cluster_slots or len(cluster_slots) == 0:
                print(f"⚠️ Attempt {attempt + 1}: No slot information found. Retrying in 5 seconds...")
                time.sleep(5)
                continue  # Retry if no slots found

            break  # Exit loop if slot data is retrieved

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}: Failed to fetch slot data: {e}")
            time.sleep(5)

    if not cluster_slots:
        print("❌ Cluster slot information is empty after retries. Manual intervention required.")
        return

    covered_slots = set()
    for slot_range in cluster_slots:
        start_slot, end_slot = slot_range[:2]
        covered_slots.update(range(start_slot, end_slot + 1))

    all_slots = set(range(16384))
    missing_slots = all_slots - covered_slots

    if not missing_slots:
        print("✅ No missing slots detected.")
        return

    print(f"🔧 Assigning {len(missing_slots)} missing slots.")

    # Get available master nodes
    cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
    if isinstance(cluster_nodes_info, list):
        cluster_nodes_info = "\n".join([str(node) for node in cluster_nodes_info])

    masters = [
        line.split()[1] for line in cluster_nodes_info.splitlines()
        if 'master' in line and 'fail' not in line
    ]

    if not masters:
        print("❌ No available master nodes to assign slots. Attempting failover...")

        # Try promoting a replica to master
        replicas = [
            line.split()[1] for line in cluster_nodes_info.splitlines()
            if 'slave' in line and 'fail' not in line
        ]

        if replicas:
            replica_ip = replicas[0].split(":")[0]
            print(f"🔧 Promoting replica {replica_ip} to master...")
            subprocess.run(["redis-cli", "-c", "-h", replica_ip, "-p", "6379", "CLUSTER", "FAILOVER"], check=True)
            time.sleep(10)  # Allow time for failover
            refresh_redis_connection()

            # Re-attempt slot assignment
            return fix_cluster_slots()
        else:
            print("❌ No replicas available to promote. Manual intervention required.")
            return

    # Assign missing slots to the first healthy master node
    master_ip_port = masters[0]
    master_ip, master_port = master_ip_port.split(':')

    for slot in missing_slots:
        subprocess.run([
            "redis-cli", "-c", "-h", master_ip, "-p", master_port,
            "CLUSTER", "ADDSLOTS", str(slot)
        ], check=True)

    print("✅ Missing slots assigned successfully.")

    # 🔄 Final Rebalancing
    print("🔄 Running final cluster rebalance to ensure even slot distribution...")
    rebalance_slots()




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


def ensure_all_nodes_responsive():
    """Ensure all Redis nodes are responsive before performing scaling operations."""
    redis_ips = get_redis_pod_ips()
    healthy_ips = []

    for ip in redis_ips:
        try:
            result = subprocess.run(["redis-cli", "-c", "-h", ip, "-p", "6379", "PING"], capture_output=True, text=True)
            if "PONG" in result.stdout:
                print(f"✅ Node {ip} is responsive.")
                healthy_ips.append(ip)
            else:
                print(f"⚠️ Node {ip} is unresponsive.")
        except Exception as e:
            print(f"❌ Error connecting to node {ip}: {e}")

    if len(healthy_ips) < 2:  # Ensure at least 2 nodes before scaling
        print("❌ Not enough healthy nodes to proceed. Fixing cluster...")
        fix_cluster_issues()
    
    return healthy_ips

def create_redis_client():
    """Create Redis Cluster client dynamically with retries."""
    redis_ips = get_redis_pod_ips()
    startup_nodes = [{"host": ip, "port": "6379"} for ip in redis_ips]

    attempts = 3  # Retry mechanism
    for attempt in range(attempts):
        try:
            client = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)
            print(f"✅ Connected to Redis Cluster with nodes: {', '.join(redis_ips)}")
            return client

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/{attempts}: Failed to connect to Redis Cluster: {e}")
            time.sleep(5)

    print("❌ All attempts to connect to Redis Cluster failed. Exiting.")
    exit(1)

redis_client = create_redis_client()


# --- AGENT SETUP ---
model_path = "modelCPU_5.pth"  # Using transfer learning model
print(f"🔍 Loading transfer learning model: {model_path}")
agent = DDQNAgentLSTM(state_size=6, action_size=3, sequence_length=10, pretrained_model=model_path)
agent.epsilon = 0.1  # Reduce exploration for real-world testing
print("✅ Transfer learning model loaded successfully!")


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
    """Remove failed nodes from the Redis cluster."""
    try:
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
        logger.debug(f"CLUSTER NODES response type: {type(cluster_nodes_info)}")
        logger.debug(f"CLUSTER NODES response content: {cluster_nodes_info}")

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
        time.sleep(5)  # Wait for cluster to stabilize
        cluster_info = redis_client.execute_command('CLUSTER INFO')
        logger.debug(f"Cluster info response (type: {type(cluster_info)}): {cluster_info}")
        
        # If we receive a dict, check that every node's info has cluster_state 'ok'
        if isinstance(cluster_info, dict):
            # In our output, CLUSTER INFO returns a dict mapping node ip:port to its info.
            # We'll consider the cluster healthy if every node reports cluster_state 'ok'
            if all(info.get('cluster_state') == 'ok' for info in cluster_info.values()):
                logger.info("✅ Cluster is healthy.")
                return True
            else:
                logger.warning("⚠️ Cluster is not healthy.")
                return False

        # Fallback: if a string is returned, parse it
        elif isinstance(cluster_info, str):
            cluster_info_dict = {}
            for line in cluster_info.splitlines():
                key, value = line.split(':', 1)
                cluster_info_dict[key.strip()] = value.strip()
            if cluster_info_dict.get('cluster_state') == 'ok':
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



def assign_missing_slots():
    """Detect and assign missing slots to healthy master nodes."""
    try:
        # CLUSTER SLOTS now returns a dict mapping slot-range tuples to slot info
        cluster_slots = redis_client.execute_command('CLUSTER SLOTS')
        covered_slots = set()
        for slot_range, info in cluster_slots.items():
            start_slot, end_slot = slot_range  # slot_range is a tuple (start, end)
            covered_slots.update(range(start_slot, end_slot + 1))

        all_slots = set(range(16384))
        missing_slots = all_slots - covered_slots

        if missing_slots:
            logger.info(f"🔧 Assigning {len(missing_slots)} missing slots.")

            # Get available master nodes (CLUSTER NODES returns a list of dictionaries)
            cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
            if not isinstance(cluster_nodes_info, list):
                logger.error(f"❌ Unexpected CLUSTER NODES response type: {type(cluster_nodes_info)}")
                return

            # Filter for healthy masters (flags contains 'master' and does not contain 'fail')
            masters = [
                node_info for node_info in cluster_nodes_info 
                if 'master' in node_info.get('flags', []) and 'fail' not in node_info.get('flags', [])
            ]

            if not masters:
                logger.error("❌ No available master nodes to assign slots.")
                return

            # Use the first healthy master node
            master_ip = masters[0]['host']
            master_port = str(masters[0]['port'])

            missing_slots_list = list(missing_slots)
            batch_size = 1000  # Process slots in batches
            for i in range(0, len(missing_slots_list), batch_size):
                batch_slots = missing_slots_list[i:i + batch_size]
                try:
                    subprocess.run([
                        "redis-cli", "-c", "-h", master_ip, "-p", master_port,
                        "CLUSTER", "ADDSLOTS", *map(str, batch_slots)
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Failed to assign slots {batch_slots}: {e}")
                    continue

            logger.info("✅ Missing slots assigned successfully.")
        else:
            logger.info("✅ No missing slots detected.")

    except Exception as e:
        logger.error(f"❌ Failed to assign missing slots: {e}")





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
                cluster_nodes_info = "\n".join([str(node) for node in cluster_nodes_info])
            
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
    """Automatically detect and fix Redis cluster issues, ensuring at least 3 master nodes."""
    try:
        print("🔍 Checking for failed nodes...")
        remove_failed_nodes()

        print("🔍 Checking for missing slots...")
        assign_missing_slots()

        print("🔄 Refreshing Redis connection before checking master nodes...")
        refresh_redis_connection()
        time.sleep(5)

        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
        active_masters = [line for line in cluster_nodes_info.splitlines() if 'master' in line and 'fail' not in line]

        if len(active_masters) < 3:
            print(f"⚠️ Only {len(active_masters)} master nodes found. Attempting recovery...")

            replicas = [line for line in cluster_nodes_info.splitlines() if 'slave' in line and 'fail' not in line]
            if not replicas:
                print("❌ No replicas available to promote. Manual intervention required.")
                return

            replica_ip = replicas[0].split()[1].split(':')[0]
            print(f"🔧 Promoting {replica_ip} to master...")

            subprocess.run(["redis-cli", "-c", "-h", replica_ip, "-p", "6379", "CLUSTER", "FAILOVER"], check=True)
            time.sleep(10)

            refresh_redis_connection()
            updated_masters = redis_client.execute_command('CLUSTER NODES')
            num_updated_masters = len([line for line in updated_masters.splitlines() if 'master' in line and 'fail' not in line])

            if num_updated_masters >= 3:
                print(f"✅ Master promotion successful! {num_updated_masters} masters available.")
            else:
                print(f"❌ Master promotion failed. Only {num_updated_masters} masters detected.")

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



def redistribute_slots_before_scaling_down():
    """Ensure we never remove one of the original 3 masters."""
    try:
        cluster_slots = redis_client.execute_command('CLUSTER SLOTS')
        if not cluster_slots or len(cluster_slots) == 0:
            print("⚠️ No slot information found. Skipping redistribution.")
            return

        # ✅ Identify current master and non-master nodes
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')
        active_masters = [line for line in cluster_nodes_info.splitlines() if 'master' in line and 'fail' not in line]
        non_master_nodes = [line for line in cluster_nodes_info.splitlines() if 'master' not in line and 'fail' not in line]

        print(f"📊 Master Nodes: {len(active_masters)}, Non-Master Nodes: {len(non_master_nodes)}")

        # ✅ Ensure at least 3 master nodes remain
        if len(active_masters) <= 3:
            print("❌ Cannot scale down further. Only 3 master nodes left.")
            return

        # ✅ If a non-master exists, remove it first
        if non_master_nodes:
            node_to_remove = non_master_nodes[0].split()[1].split(':')[0]
            print(f"🔧 Removing non-master node: {node_to_remove}")
            remove_node(redis_client, node_to_remove)
            return

        print("⚠️ No non-master nodes available for removal. Skipping redistribution.")

    except Exception as e:
        print(f"❌ Failed to redistribute slots before scaling down: {e}")



last_scale_time = 0
import subprocess
import time

import subprocess
import time

import subprocess
import time


import subprocess
import time

def get_coordinator_ip():
    """
    Dynamically find a valid coordinator for Redis cluster operations.
    """
    try:
        pod_ips = get_active_cluster_nodes()
        print("🔍 Active pod IPs:", pod_ips)

        for ip in pod_ips:
            result = subprocess.run(
                ["redis-cli", "-c", "-h", ip, "-p", "6379", "CLUSTER", "INFO"],
                capture_output=True, text=True
            )

            if "cluster_state:ok" in result.stdout:
                print(f"✅ Coordinator IP detected: {ip}")
                return ip

        print("❌ No explicitly healthy coordinator found, but all nodes report `cluster_state:ok`.")
        print("🔄 Defaulting to the first available node.")
        return pod_ips[0] if pod_ips else None

    except Exception as e:
        print(f"❌ Failed to get coordinator IP: {e}")
        return None


def scale_redis(action):
    """
    Scale the Redis cluster up or down while ensuring:
    - The original masters remain.
    - Slot rebalancing is performed before and after scaling.
    """

    current_pods = get_current_pod_count()

    # 🔄 Refresh Redis connection before checking state
    print("🔄 Refreshing Redis connection before reading cluster state...")
    refresh_redis_connection()
    time.sleep(5)

    # ✅ Fetch cluster state
    try:
        cluster_nodes_info = redis_client.execute_command('CLUSTER NODES')

        if isinstance(cluster_nodes_info, bytes):
            cluster_nodes_info = cluster_nodes_info.decode("utf-8")

        if isinstance(cluster_nodes_info, list):
            print(f"⚠️ Received unexpected list format. Converting to string.")
            cluster_nodes_info = "\n".join([" ".join(str(value) for value in node.values()) for node in cluster_nodes_info])

        if not isinstance(cluster_nodes_info, str):
            print(f"⚠️ Unexpected format received: {type(cluster_nodes_info)}")
            return

        print("\n📝 RAW CLUSTER NODES (filtered):")
        for node in cluster_nodes_info.splitlines():
            print(node)

        print("🔄 Parsing `CLUSTER NODES` output...")
        parsed_cluster = parse_cluster_nodes(cluster_nodes_info)

    except Exception as e:
        print(f"❌ ERROR: Failed to fetch cluster nodes: {e}")
        return

    # ✅ Fetch Kubernetes pod IPs correctly
    pod_ips = subprocess.getoutput(
        "kubectl get pods -l redis=redis-cluster-pods-lbl -o jsonpath='{.items[*].status.podIP}'"
    ).strip().split()

    # ✅ Extract only **IP addresses** from parsed cluster nodes
    parsed_cluster_ips = {node_data["ip"] for node_data in parsed_cluster.values() if node_data.get("ip")}

    print(f"\n🔍 DEBUG INFO:")
    print(f"🔹 Parsed Cluster Nodes: {parsed_cluster.keys()}")
    print(f"🔹 Cluster Node IPs: {parsed_cluster_ips}")
    print(f"🔹 Original Masters: {ORIGINAL_MASTERS}")
    print(f"🔹 Extracted Pod IPs: {pod_ips}\n")

    if action == 0:  # **Scale Down**
        print("🛑 Initiating SCALE DOWN process...")

        removable_nodes = [
            node for node in parsed_cluster_ips
            if f"{node}:6379" not in ORIGINAL_MASTERS and node in pod_ips
        ]

        print(f"\n🔍 Removable Nodes Found: {removable_nodes}\n")

        if not removable_nodes:
            print("⚠️ No removable nodes found. Scale-down aborted.")
            return

        node_to_remove = removable_nodes[0]
        print(f"🔻 Scaling down: Preparing to remove node {node_to_remove}")

        # ✅ Step 1: Migrate slots before removal
        try:
            coordinator_ip = get_coordinator_ip()
            if not coordinator_ip:
                print("❌ No valid coordinator found. Cannot proceed with scaling down.")
                return

            print(f"🔄 Migrating slots from {node_to_remove} before removal...")
            subprocess.run([
                "redis-cli", "--cluster", "reshard", f"{coordinator_ip}:6379",
                "--cluster-yes", "--from", node_to_remove, "--to", coordinator_ip, "--slots", "16384"
            ], check=True)
            print("✅ Slot migration successful.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Slot migration failed: {e}")
            return

        # ✅ Step 2: Remove the node from Redis Cluster
        try:
            redis_client.execute_command(f"CLUSTER FORGET {node_to_remove}")
            print(f"✅ Successfully removed {node_to_remove} from Redis cluster.")
        except Exception as e:
            print(f"❌ Failed to remove node from Redis cluster: {e}")

        # ✅ Step 3: Remove the corresponding Kubernetes pod
        pod_name = subprocess.getoutput(
            f"kubectl get pods -l redis=redis-cluster-pods-lbl -o json | jq -r '.items[] | select(.status.podIP==\"{node_to_remove}\") | .metadata.name'"
        ).strip()

        if pod_name:
            try:
                subprocess.run(["kubectl", "delete", "pod", pod_name], check=True)
                print(f"✅ Successfully removed pod {pod_name}.")
            except Exception as e:
                print(f"❌ Failed to remove pod {pod_name}: {e}")
                return
        else:
            print(f"⚠️ No matching Kubernetes pod found for {node_to_remove}. Skipping pod removal.")

        print("🔄 Running post-scale-down slot rebalance...")
        rebalance_slots()
        return

    elif action == 1:  # **No Change**
        print("🟡 No change requested. Skipping scaling.")
        return

    elif action == 2:  # **Scale Up**
        new_pods = current_pods + 1
        print(f"🔄 Scaling Redis up: {current_pods} -> {new_pods}")

        try:
            subprocess.run(["kubectl", "scale", "deployment/redis-autoscale-cluster", f"--replicas={new_pods}"], check=True)
            print("⏳ Waiting for Redis pods to stabilize...")
            if not wait_for_pods_ready(new_pods):
                print("❌ Scaling failed. Rolling back changes...")
                return

            print("🔄 Reconnecting to Redis after scaling...")
            refresh_redis_connection()
            time.sleep(10)

            print("🔧 Removing stale nodes...")
            remove_failed_nodes()

            rebalance_slots()

            if not is_cluster_healthy(redis_client):
                print("⚠️ Cluster unhealthy after scaling. Fixing issues...")
                fix_cluster_issues()
                time.sleep(10)

                if not is_cluster_healthy(redis_client):
                    print("❌ Cluster still unhealthy after fix attempts.")
                    return

        except subprocess.CalledProcessError as e:
            print(f"❌ Scaling command failed: {e}")
            refresh_redis_connection()
            fix_cluster_issues()

















def rebalance_slots():
    """Rebalance slots after scaling, ensuring the original masters remain intact."""
    try:
        coordinator_ip = get_coordinator_ip()
        if not coordinator_ip:
            print("❌ No valid coordinator found for rebalance. Skipping...")
            return

        print(f"🔄 Rebalancing cluster using {coordinator_ip}:6379...")

        max_attempts = 3
        for attempt in range(max_attempts):
            result = subprocess.run([
                "redis-cli", "--cluster", "rebalance", f"{coordinator_ip}:6379",
                "--cluster-use-empty-masters", "--cluster-yes"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Slots rebalanced successfully.")
                refresh_redis_connection()
                return
            else:
                print(f"⚠️ Attempt {attempt + 1}: Slot rebalancing failed. Retrying in 5 seconds...")
                time.sleep(5)

        print("❌ Slot rebalancing failed after multiple attempts.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Slot rebalancing command failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during rebalancing: {e}")









# --- MAIN LOOP ---

# --- MAIN LOOP ---

health_attempts = 0
state = np.zeros((agent.sequence_length, agent.state_size))  # Initialize the state for LSTM

for t in range(timesteps):
    print(f"\n⏳ Starting timestep {t}...")

    # --- Step 1: Ensure Cluster is Healthy Before Writing Load ---
    if not is_cluster_healthy(redis_client):
        print(f"⚠️ Cluster not healthy, attempting recovery... (Attempt {health_attempts + 1})")
        fix_cluster_issues()
        time.sleep(10)

        if not is_cluster_healthy(redis_client):
            print(f"❌ Cluster still unhealthy after retries. Skipping timestep {t}.")
            health_attempts += 1
            if health_attempts >= MAX_HEALTH_ATTEMPTS:
                print("❌ Max health attempts reached. Continuing experiment with partial data.")
                break  # Exit experiment if cluster remains unstable
            continue  # Skip to next timestep
        else:
            print("✅ Cluster recovered. Proceeding with load generation.")
            health_attempts = 0  # Reset health attempts if cluster is healthy

    # --- Step 2: Generate Load ---
    load = generate_sinusoidal_load(t)  # ✅ FIXED: Ensure load is assigned before use

    # --- Step 3: Fix Missing Slots Before Writing Load ---
    print("🔍 Ensuring all slots are assigned before writing...")
    force_fix_cluster_slots()

    # ✅ **Check if slots are assigned before writing**
    try:
        cluster_slots = redis_client.execute_command("CLUSTER SLOTS")
        if not cluster_slots:
            print("❌ No slots assigned. Skipping this timestep.")
            continue  # Skip writing load
    except Exception as e:
        print(f"❌ Failed to check cluster slots: {e}. Skipping timestep {t}.")
        continue  # Skip writing load

    # --- Step 4: Retry Writing Load to Redis ---
    for attempt in range(3):
        try:
            redis_client.set(f"load_key_{t}", load)
            stored_value = redis_client.get(f"load_key_{t}")
            if float(stored_value) != load:
                raise ValueError("Mismatch between written and read load values.")
            break  # Exit retry loop if successful
        except Exception as e:
            print(f"⚠️ Retry {attempt + 1}: Error writing to Redis: {e}")
            refresh_redis_connection()
            time.sleep(5)
    else:
        print(f"❌ Failed to write load at timestep {t} after 3 attempts. Skipping.")
        continue  # Skip to next timestep

    # --- Step 5: Collect Metrics from Redis ---
    cpu_usage, memory_usage, latency, connected_clients = get_redis_stats()
    current_pod_count = get_current_pod_count()
    print(f"📊 Metrics - Load: {load:.2f}, CPU: {cpu_usage:.2f}, Memory: {memory_usage:.2f}, Latency: {latency:.2f}, Connected Clients: {connected_clients}, Pods: {current_pod_count}")

    # --- Step 6: Prepare State for Agent ---
    new_state = np.array([load, cpu_usage, memory_usage, latency, connected_clients, current_pod_count])
    next_state = np.zeros((agent.sequence_length, agent.state_size))
    next_state[:-1] = state[1:]  # Shift previous states
    next_state[-1] = new_state   # Add current metrics to the state

    # --- Step 7: Agent Decides on Scaling Action ---
    start_time = time.time()
    action = agent.act(next_state)
    inference_time = time.time() - start_time
    print(f"🕒 Model inference time: {inference_time:.4f} seconds")
    action_description = {0: "Scale Down", 1: "No Change", 2: "Scale Up"}
    print(f"🔍 Timestep {t}: Load={load:.2f}, CPU={cpu_usage:.2f}, Mem={memory_usage:.2f}, Latency={latency:.2f}, Pods={current_pod_count}, Action={action_description[action]}")

    # --- Step 8: Ensure All Nodes Are Responsive Before Scaling ---
    healthy_nodes = ensure_all_nodes_responsive()
    if len(healthy_nodes) < 2:  # Ensure at least 2 healthy nodes before scaling
        print("❌ Not enough healthy nodes to scale safely. Skipping scaling.")
        continue

    # --- Step 9: Execute Scaling Action ---
    scale_redis(action)  # 🔄 Unified call for scaling up/down

    # --- Step 10: Post-Scaling Health Check ---
    print("⏳ Waiting for Redis cluster to stabilize...")
    time.sleep(15)

    if not is_cluster_healthy(redis_client):
        print("⚠️ Cluster unhealthy after scaling. Attempting to fix...")
        fix_cluster_issues()
        time.sleep(10)

        if not is_cluster_healthy(redis_client):
            print("❌ Cluster still unhealthy after fix attempts.")
            continue  # Skip to next timestep

    # --- Step 11: Post-Scaling Rebalancing ---
    print("🔄 Running post-scaling rebalancing...")
    coordinator_ip = get_coordinator_ip()
    if coordinator_ip:
        rebalance_result = subprocess.run(
            ["redis-cli", "--cluster", "rebalance", f"{coordinator_ip}:6379", "--cluster-use-empty-masters", "--cluster-yes"],
            capture_output=True, text=True
        )

        if rebalance_result.returncode != 0:
            print(f"❌ Rebalance failed: {rebalance_result.stderr}")
        else:
            print("✅ Post-scaling rebalancing successful.")
    else:
        print("❌ No valid coordinator found. Skipping rebalance.")

    # --- Step 12: Ensure No Failed Nodes Exist After Rebalancing ---
    print("🔧 Checking for failed nodes after scaling...")
    remove_failed_nodes()

    # --- Step 13: Record Data for Analysis ---
    load_data.append(load)
    cpu_usage_data.append(cpu_usage)
    memory_usage_data.append(memory_usage)
    latency_data.append(latency)
    num_workers_data.append(current_pod_count)

    # --- Step 14: Calculate Reward ---
    reward = 3.0 * load - 5.0 * latency - 10.0 * current_pod_count - 0.5 * cpu_usage - 0.5 * memory_usage
    rewards_data.append(reward)

    # --- Step 15: Update State for Next Iteration ---
    state = next_state

    # --- Step 16: Delay Between Timesteps ---
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
