from rediscluster import RedisCluster
import numpy as np
import time
from kubernetes import client, config
import json

# Initialize Kubernetes client
config.load_kube_config()
v1_metrics = client.CustomObjectsApi()

# Define your cluster nodes
startup_nodes = [
    {"host": "10.244.37.170", "port": "6379"},
    {"host": "10.244.37.138", "port": "6379"},
    {"host": "10.244.37.191", "port": "6379"},
    # Add the rest of your nodes here...
]

# Connect to Redis Cluster
rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)

# Configuration
duration = 300  # Duration of the test in seconds (5 minutes)
max_rate = 100  # Max requests per second
period = 10  # Period of the sinusoid in seconds

def get_avg_cpu_mem_usage(namespace='default', label_selector='redis=redis-cluster-pods-lbl'):
    metrics = v1_metrics.list_namespaced_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        namespace=namespace,
        plural="pods",
        label_selector=label_selector
    )

    total_cpu_nano_cores = 0
    total_memory_kib = 0
    count = 0

    for item in metrics.get('items', []):
        for container in item['containers']:
            cpu_usage = container['usage']['cpu']
            mem_usage = container['usage']['memory']
            # Remove the trailing 'n' and 'Ki' and convert to integers
            total_cpu_nano_cores += int(cpu_usage.rstrip('n'))
            total_memory_kib += int(mem_usage.rstrip('Ki'))
            count += 1

    if count > 0:
        avg_cpu_usage_milli_cores = (total_cpu_nano_cores / count) / 1e6  # Convert to milliCores
        avg_memory_usage_mi = (total_memory_kib / count) / 1024.0  # Convert to MiB
    else:
        avg_cpu_usage_milli_cores = 0
        avg_memory_usage_mi = 0

    return avg_cpu_usage_milli_cores, avg_memory_usage_mi

start_time = time.time()

while time.time() - start_time < duration:
    operations = 0  # Number of operations this second
    latencies = []  # Store latencies for each operation

    elapsed = time.time() - start_time
    rate = (np.sin(elapsed * (2 * np.pi / period)) + 1) / 2 * max_rate  # Sinusoidal modulation

    avg_cpu_usage, avg_memory_usage = get_avg_cpu_mem_usage()

    loop_start = time.time()
    while time.time() - loop_start < 1:
        op_start = time.time()
        rc.set('key', 'value')  # Example operation
        latencies.append(time.time() - op_start)
        operations += 1

    # Calculate throughput and average latency for the past second
    throughput = operations
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"Time: {elapsed:.2f}s, Throughput: {throughput} ops/s, Avg Latency: {avg_latency:.5f} s, Avg CPU Usage: {avg_cpu_usage}m, Avg Memory Usage: {avg_memory_usage}Mi")

    # Adjust sleep to synchronize with the start of the next second
    time.sleep(max(0, 1 - (time.time() - loop_start)))

