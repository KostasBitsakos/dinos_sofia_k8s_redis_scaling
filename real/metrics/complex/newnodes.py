from rediscluster import RedisCluster
from kubernetes import client, config
import re

# Load Kubernetes configuration
config.load_kube_config()

# Kubernetes client
v1 = client.CoreV1Api()

# Define your Redis cluster nodes
startup_nodes = [
    {"host": "10.97.62.196", "port": 6379}  # Example; adjust based on your actual setup
]

# Initialize RedisCluster client
rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)

def fetch_pod_name_by_ip(ip_address):
    try:
        pods = v1.list_pod_for_all_namespaces(watch=False)
        for pod in pods.items:
            if pod.status.pod_ip == ip_address:
                return pod.metadata.name
    except Exception as e:
        print(f"Error fetching pod name for IP {ip_address}: {e}")
    return None

def map_redis_nodes_to_pods():
    # Fetch cluster nodes from Redis
    nodes = rc.cluster_nodes()
    for node in nodes:
        # Assuming 'node' dict contains 'host' and 'port' keys
        ip = node['host']
        port = node['port']
        pod_name = fetch_pod_name_by_ip(ip)
        print(f"Redis Node: {ip}:{port}, Kubernetes Pod Name: {pod_name}")

if __name__ == "__main__":
    map_redis_nodes_to_pods()
