from rediscluster import RedisCluster
import numpy as np
import time
from kubernetes import client, config

# Function to calculate average CPU and Memory usage for pods matching a label selector
def get_avg_cpu_mem_usage(label_selector, namespace='default'):
    config.load_kube_config()  # Loads kubeconfig for accessing the Kubernetes API
    metrics_api = client.CustomObjectsApi()
    pod_metrics_list = metrics_api.list_namespaced_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        namespace=namespace,
        plural="pods",
        label_selector=label_selector
    )

    total_cpu_usage = 0
    total_memory_usage = 0
    count = 0

    for pod in pod_metrics_list.get("items", []):
        for container in pod.get("containers", []):
            cpu_usage_str = container.get("usage", {}).get("cpu", "0")
            mem_usage_str = container.get("usage", {}).get("memory", "0Mi")

            # Convert CPU usage to millicores and memory usage to Mi
            cpu_usage = int(cpu_usage_str.rstrip("m")) if cpu_usage_str.endswith("m") else int(cpu_usage_str) / 1e6
            mem_usage = int(mem_usage_str.rstrip("Mi"))

            total_cpu_usage += cpu_usage
            total_memory_usage += mem_usage
            count += 1

    avg_cpu_usage = total_cpu_usage / count if count else 0
    avg_memory_usage = total_memory_usage / count if count else 0

    return avg_cpu_usage, avg_memory_usage

# Your existing Redis cluster configuration
startup_nodes = [
    {"host": "10.244.37.170", "port": "6379"},
    {"host": "10.244.37.138", "port": "6379"},
    {"host": "10.244.37.191", "port": "6379"},
    # Add the rest of your nodes here...
]
rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)

# Test configuration
duration = 300  # Duration of the test in seconds (5 minutes)
max_rate = 100  # Max requests per second
period = 10  # Period of the sinusoid in seconds

start_time = time.time()

while time.time() - start_time < duration:
    operations = 0
    latencies = []

    elapsed = time.time() - start_time
    rate = (np.sin(elapsed * (2 * np.pi / period)) + 1) / 2 * max_rate

    loop_start = time.time()
    while time.time() - loop_start < 1:
        op_start = time.time()
        rc.set('key', 'value')  # Example operation
        latencies.append(time.time() - op_start)
        operations += 1

    throughput = operations
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Fetch average CPU and Memory usage
    avg_cpu_usage, avg_memory_usage = get_avg_cpu_mem_usage("app=redis", namespace="default")
    
    print(f"Time: {elapsed:.2f}s, Throughput: {throughput} ops/s, Avg Latency: {avg_latency:.5f} s, Avg CPU Usage: {avg_cpu_usage}m, Avg Memory Usage: {avg_memory_usage}Mi")

    time.sleep(max(0, 1 - (time.time() - loop_start)))

