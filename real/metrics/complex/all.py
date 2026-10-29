import redis
import numpy as np
import csv
import re
import time
from kubernetes import client, config
from rediscluster import RedisCluster

# Define your startup nodes here
startup_nodes = [
    {"host": "10.244.37.149", "port": 6379},
    {"host": "10.244.37.183", "port": 6379},
    {"host": "10.244.37.182", "port": 6379},
    {"host": "10.244.37.150", "port": 6379},
    {"host": "10.244.37.169", "port": 6379},
    {"host": "10.244.37.154", "port": 6379},
]

config.load_kube_config()
v1 = client.CoreV1Api()
metrics_api = client.CustomObjectsApi()
r = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)

def parse_quantity(quantity):
    """
    Parse Kubernetes quantity strings into a unified unit (cores for CPU, MiB for memory).
    """
    unit_factors = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "m": 1/1000, "": 1, "n": 1/10**9}
    match = re.match(r"([0-9]+)([a-zA-Z]*)", quantity)
    number, unit = match.groups() if match else (0, "")
    return float(number) * unit_factors.get(unit, 1)

def get_avg_pod_metrics(namespace='default', label_selector='redis=redis-cluster-pods-lbl'):
    """
    Fetch and calculate average CPU and memory usage for pods matching the label selector.
    """
    pod_metrics = metrics_api.list_namespaced_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        namespace=namespace,
        plural="pods",
        label_selector=label_selector)

    total_cpu_usage_millicores, total_memory_usage_mib, pod_count = 0, 0, 0

    for pod in pod_metrics.get('items', []):
        for container in pod.get('containers', []):
            cpu_usage_cores = parse_quantity(container['usage']['cpu'])  # CPU usage in cores
            memory_usage_bytes = parse_quantity(container['usage']['memory'])  # Memory usage in bytes
            
            total_cpu_usage_millicores += cpu_usage_cores * 1000  # Convert cores to millicores
            total_memory_usage_mib += memory_usage_bytes / (2**20)  # Convert bytes to MiB
            pod_count += 1

    avg_cpu_usage_millicores = total_cpu_usage_millicores / pod_count if pod_count else 0
    avg_memory_usage_mib = total_memory_usage_mib / pod_count if pod_count else 0

    return avg_cpu_usage_millicores, avg_memory_usage_mib


def send_poisson_requests_and_log_metrics(rate_lambda=10, duration_seconds=300):
    start_time = time.time()
    last_print_time = start_time  # Initialize last_print_time here
    end_time = start_time + duration_seconds
    total_requests = 0
    latencies = []

    with open('metrics4.csv', 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'total_requests', 'throughput', 'average_latency', 'avg_cpu_usage', 'avg_memory_usage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while time.time() < end_time:
            # Sinusoidal-distributed interval
            interval = abs(np.sin(time.time())) * rate_lambda
            time.sleep(interval)

            # Perform Redis operations
            key = f"key_{np.random.randint(1000)}"
            value = "value"
            operation_start = time.time()
            r.set(key, value)
            r.get(key)
            operation_end = time.time()

            latencies.append(operation_end - operation_start)
            total_requests += 1

            current_time = time.time()
            # Check if at least 5 seconds have passed since the last print
            if current_time - last_print_time >= 5:
                throughput = total_requests / (current_time - start_time)
                avg_latency = np.mean(latencies) if latencies else 0
                avg_cpu_usage, avg_memory_usage = get_avg_pod_metrics(namespace='default', label_selector='redis=redis-cluster-pods-lbl')

                print(f"Timestamp: {current_time}, Total requests: {total_requests}, Throughput: {throughput:.2f} requests/s, Average latency: {avg_latency:.5f} s, Average CPU Usage: {avg_cpu_usage:.2f}%, Average Memory Usage: {avg_memory_usage:.2f}%")

                writer.writerow({
                    'timestamp': current_time,
                    'total_requests': total_requests,
                    'throughput': throughput,
                    'average_latency': avg_latency,
                    'avg_cpu_usage': avg_cpu_usage,
                    'avg_memory_usage': avg_memory_usage})

                last_print_time = current_time  # Update the last print time
                # More code here...

                # Fetch Kubernetes metrics for average CPU and memory usage
                # avg_cpu_usage, avg_memory_usage = get_avg_pod_metrics(namespace='default', label_selector='app=redis')

                # # Calculate throughput and average latency
                # total_time_elapsed = time.time() - start_time
                # throughput = total_requests / total_time_elapsed
                # avg_latency = sum(latencies) / len(latencies) if latencies else 0

                # # Print and log the metrics
                # print(f"Total requests: {total_requests}, Throughput: {throughput:.2f} requests/s, Average latency: {avg_latency:.5f} s, Average CPU Usage: {avg_cpu_usage:.3f}%, Average Memory Usage: {avg_memory_usage:.3f}%")
                # writer.writerow({
                #     'timestamp': time.time(),
                #     'total_requests': total_requests,
                #     'throughput': throughput,
                #     'average_latency': avg_latency,
                #     'avg_cpu_usage': avg_cpu_usage,
                #     'avg_memory_usage': avg_memory_usage
                # })

        # Consider adding logic here to close any open resources or handle any finalization required after the loop

if __name__ == "__main__":
    send_poisson_requests_and_log_metrics(rate_lambda=10, duration_seconds=300)