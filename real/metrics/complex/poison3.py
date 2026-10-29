import time
import numpy as np
from nodes5 import get_current_nodes
from k8smetrics import fetch_pod_metrics

# Node capacity (for calculations)
# Ensure these values reflect your actual cluster's total resources
total_cpu_cores = 40  # Example: total CPU cores as from `kubectl describe nodes`
total_memory_bytes = 201301324 * 1024  # Example: converted Ki to bytes for total memory

def simulate_load_on_redis():
    # Placeholder for simulating load on Redis using Poisson distribution
    pass

def calculate_throughput_and_latency():
    # Placeholder for calculating throughput and latency
    # You'll need to implement the logic based on how you simulate the load
    return np.random.rand(), np.random.rand()  # Example random throughput and latency

# def print_metrics_every_node(interval=5):
#     while True:
#         current_nodes = get_current_nodes()
#         for node in current_nodes:
#             cpu_usage, memory_usage = fetch_pod_metrics(node['pod_name'], 'default')
#             # Ensure cpu_usage and memory_usage are converted to integers before calculation
#             cpu_percentage = (int(cpu_usage.rstrip('n')) / (total_cpu_cores * 1e9)) * 100
#             memory_percentage = (int(memory_usage.rstrip('Ki')) * 1024 / total_memory_bytes) * 100
            
#             simulate_load_on_redis()
#             throughput, latency = calculate_throughput_and_latency()
            
#             print(f"Node: {node['pod_name']}, CPU: {cpu_percentage:.2f}%, Memory: {memory_percentage:.2f}%, Throughput: {throughput} ops/s, Latency: {latency} s")
        
#         time.sleep(interval)

def print_metrics_every_node(interval=5):
    while True:
        current_nodes = get_current_nodes()
        for node in current_nodes:
            # Fetching metrics for each node
            cpu_usage, memory_usage = fetch_pod_metrics(node['pod_name'], 'default')

            # Converting CPU usage from nanocores to a percentage of total CPU cores
            # Assuming 1 core = 1e9 nanocores
            cpu_usage_percentage = (int(cpu_usage.rstrip('n')) / (total_cpu_cores * 1e9)) * 100

            # Converting memory usage from bytes to a percentage of total memory
            memory_usage_percentage = (int(memory_usage.rstrip('Ki')) * 1024 / total_memory_bytes) * 100

            print(f"Node: {node['pod_name']}, CPU: {cpu_usage_percentage:.2f}%, Memory: {memory_usage_percentage:.2f}%")
        
        time.sleep(interval)


if __name__ == "__main__":
    print_metrics_every_node()

