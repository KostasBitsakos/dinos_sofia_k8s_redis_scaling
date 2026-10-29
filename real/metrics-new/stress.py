from rediscluster import RedisCluster
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from kubernetes import client, config
from tqdm import tqdm  # For the progress bar

# Constants for the experiment
TOTAL_DURATION = 600  # 10 minutes in seconds
INTERVAL = 5  # Measure every 5 seconds
PERIOD = 30  # 30 seconds period for the sinusoidal load
REDIS_CLUSTER_HOST = '10.109.35.136'  # Redis cluster service IP from kubectl get svc
REDIS_CLUSTER_PORT = 6379  # Default Redis port

# Connect to Redis cluster
def connect_to_redis_cluster():
    try:
        # Use RedisCluster to connect to the cluster
        startup_nodes = [{"host": REDIS_CLUSTER_HOST, "port": REDIS_CLUSTER_PORT}]
        return RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)
    except Exception as e:
        print(f"Failed to connect to Redis cluster: {e}")
        exit(1)

# Send requests to Redis in a loop
def send_requests_to_redis(load):
    r = connect_to_redis_cluster()
    for _ in range(int(load)):
        r.set('key', 'value')

# Measure CPU, memory usage, and number of pods
def measure_k8s_metrics():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    metrics_client = client.CustomObjectsApi()

    pods = v1.list_namespaced_pod(namespace='default', label_selector="redis=redis-cluster-pods-lbl")
    pod_count = len(pods.items)

    node_metrics = metrics_client.list_cluster_custom_object(
        group="metrics.k8s.io", version="v1beta1", plural="nodes"
    )

    total_cpu = 0
    total_mem = 0

    for item in node_metrics["items"]:
        usage_cpu = item["usage"]["cpu"]
        usage_mem = item["usage"]["memory"]
        total_cpu += int(usage_cpu[:-1])  # Strip 'n' at the end
        total_mem += int(usage_mem[:-2])  # Strip 'Ki' at the end

    return total_cpu, total_mem, pod_count

# Generate load following a sinusoidal distribution
def generate_load():
    time_points = np.arange(0, TOTAL_DURATION, INTERVAL)
    load = 100 * (np.sin(2 * np.pi * time_points / PERIOD) + 1) / 2
    return time_points, load

# Plot metrics
def plot_metrics(time_points, load, cpu_usage, mem_usage, pod_count):
    # Plot load
    plt.figure()
    plt.plot(time_points, load, label="Load (Requests)", color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Load (Requests)")
    plt.title("Sinusoidal Load Over Time")
    plt.savefig('load_plot.png')

    # Plot CPU usage
    plt.figure()
    plt.plot(time_points, cpu_usage, label="CPU Usage", color='green')
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (n)")
    plt.title("CPU Usage Over Time")
    plt.savefig('cpu_usage_plot.png')

    # Plot Memory usage
    plt.figure()
    plt.plot(time_points, mem_usage, label="Memory Usage", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (Ki)")
    plt.title("Memory Usage Over Time")
    plt.savefig('memory_usage_plot.png')

    # Plot number of pods
    plt.figure()
    plt.plot(time_points, pod_count, label="Number of Pods", color='purple')
    plt.xlabel("Time (s)")
    plt.ylabel("Pods Count")
    plt.title("Pods Count Over Time")
    plt.savefig('pods_count_plot.png')

if __name__ == "__main__":
    time_points, load = generate_load()

    # Initialize lists for metrics
    cpu_usage = []
    mem_usage = []
    pod_count = []

    # Run the experiment for 10 minutes
    start_time = time.time()

    # Use tqdm for progress bar
    with tqdm(total=TOTAL_DURATION, desc="Experiment Progress", unit="s") as pbar:
        for t, l in zip(time_points, load):
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Stop the loop after 10 minutes
            if elapsed_time >= TOTAL_DURATION:
                break

            # Apply load and measure metrics
            send_requests_to_redis(l)
            cpu, mem, pods = measure_k8s_metrics()
            cpu_usage.append(cpu)
            mem_usage.append(mem)
            pod_count.append(pods)

            # Update the progress bar
            pbar.update(INTERVAL)

            # Ensure the loop runs every INTERVAL seconds
            time.sleep(INTERVAL)

    # Plot results
    plot_metrics(time_points[:len(cpu_usage)], load[:len(cpu_usage)], cpu_usage, mem_usage, pod_count)
