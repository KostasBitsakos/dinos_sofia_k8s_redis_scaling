from rediscluster import RedisCluster
import numpy as np
import time
from kubernetes import client, config

# Define your cluster nodes
startup_nodes = [
    {"host": "10.244.37.164", "port": 6379},
    {"host": "10.244.37.176", "port": 6379},
    {"host": "10.244.37.168", "port": 6379},
    {"host": "10.244.37.142", "port": 6379},
    {"host": "10.244.37.174", "port": 6379},
    {"host": "10.244.37.187", "port": 6379},
]

# Initialize Kubernetes client
config.load_kube_config()
v1_metrics = client.CustomObjectsApi()

def get_avg_cpu_mem_usage(namespace='default', label_selector='app=redis'):
    metrics = v1_metrics.list_namespaced_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        namespace=namespace,
        plural="pods",
        label_selector=label_selector
    )

    total_cpu_nano_cores = 0
    total_memory_bytes = 0
    count = 0

    for item in metrics['items']:
        for container in item['containers']:
            cpu_usage = container['usage']['cpu']
            mem_usage = container['usage']['memory']
            total_cpu_nano_cores += int(cpu_usage.rstrip('n'))
            total_memory_bytes += int(mem_usage.rstrip('Ki')) * 1024  # Convert Ki to bytes
            count += 1

    avg_cpu_usage = total_cpu_nano_cores / count / 1e6 if count else 0  # Convert to millicores
    avg_memory_usage = total_memory_bytes / count / 1024 / 1024 if count else 0  # Convert bytes to MiB

    return avg_cpu_usage, avg_memory_usage

# Connect to Redis Cluster
rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)

# Configuration
duration = 300  # Duration of the test in seconds (5 minutes)
lambda_rate = 50  # Average requests per second for Poisson distribution

start_time = time.time()

while time.time() - start_time < duration:
    elapsed = time.time() - start_time
    operations = np.random.poisson(lambda_rate)  # Number of operations to perform based on Poisson distribution
    latencies = []  # Store latencies for each operation

    for _ in range(operations):
        op_start = time.time()
        rc.set(f'key-{np.random.randint(1000)}', 'value')  # Perform a Redis operation
        latencies.append(time.time() - op_start)

    # Calculate throughput and average latency
    throughput = operations
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    avg_cpu_usage, avg_memory_usage = get_avg_cpu_mem_usage()

    print(f"Time: {elapsed:.2f}s, Throughput: {throughput} ops/s, Avg Latency: {avg_latency:.5f} s, Avg CPU Usage: {avg_cpu_usage:.2f}m, Avg Memory Usage: {avg_memory_usage:.2f}Mi")

    # Sleep for 1 second before next iteration
    time.sleep(1)

