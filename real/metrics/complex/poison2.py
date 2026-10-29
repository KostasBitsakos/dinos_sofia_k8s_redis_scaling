import numpy as np
import time
from rediscluster import RedisCluster
from kubernetes import client, config
from nodes5 import get_current_nodes
from k8smetrics import fetch_pod_metrics

# Load Kubernetes configuration
config.load_kube_config()

# Define your initial cluster nodes
startup_nodes = [
    {"host": "10.244.37.164", "port": "6379"},
    {"host": "10.244.37.176", "port": "6379"},
    {"host": "10.244.37.168", "port": "6379"},
    {"host": "10.244.37.142", "port": "6379"},
    {"host": "10.244.37.174", "port": "6379"},
    {"host": "10.244.37.187", "port": "6379"},
]

# Initialize RedisCluster client
rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)

def perform_poisson_requests(lam=50, interval=5):
    """
    Perform requests on the Redis cluster based on Poisson distribution.
    """
    operations = np.random.poisson(lam)  # Number of operations to perform
    latencies = []  # Store latencies for each operation
    for _ in range(operations):
        op_start = time.time()
        # Example Redis operation
        key = f'key-{np.random.randint(1000)}'
        rc.set(key, 'value')
        latencies.append(time.time() - op_start)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    return operations, avg_latency

def monitor_redis_cluster_metrics(interval=5):
    """
    Monitor and print metrics for the Redis cluster.
    """
    print("Starting Redis cluster metrics monitoring...")
    while True:
        # Perform Poisson distribution requests
        operations, avg_latency = perform_poisson_requests()
        
        # Fetch and print cluster nodes and their metrics
        current_nodes = get_current_nodes()
        for node in current_nodes:
            cpu_usage, memory_usage = fetch_pod_metrics(node['pod_name'])
            print(f"Node: {node['pod_name']}, Operations: {operations}, Avg Latency: {avg_latency:.5f}s, CPU Usage: {cpu_usage}, Memory Usage: {memory_usage}")
        
        time.sleep(interval)

if __name__ == "__main__":
    monitor_redis_cluster_metrics()

