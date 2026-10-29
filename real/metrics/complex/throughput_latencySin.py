import time
import csv
import numpy as np
from rediscluster import RedisCluster
import cpumemory  # Ensure this module contains the cpu_memory_avg function

from rediscluster import RedisCluster

def get_redis_cluster_client():
    # Updated list of nodes, avoiding those marked as 'fail?' and using operational masters
    startup_nodes = [
        # Including available master nodes without fail status
        {"host": "10.244.37.133", "port": 6379},  # Even if flagged, attempting to include unless fully failed
        {"host": "10.244.37.161", "port": 6379},  # Even if flagged, attempting to include unless fully failed
        {"host": "10.244.37.185", "port": 6379},  # Even if flagged, attempting to include unless fully failed
    ]
    # Assuming RedisCluster is imported from redis.cluster
    return RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)


















def perform_requests(rate_lambda, metrics_interval_seconds, total_duration_seconds, max_requests_per_interval):
    r = get_redis_cluster_client()

    with open('metrics/metricsnew23.csv', 'w', newline='') as csvfile:
        fieldnames = ['Time', 'Average Throughput', 'Average Latency', 'CPU Usage %', 'Memory Usage %', 'Node Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        start_time = time.time()
        total_requests = 0
        total_latency = 0.0

        while time.time() - start_time < total_duration_seconds:
            elapsed_time = time.time() - start_time
            num_requests_this_interval = int((np.sin(elapsed_time * rate_lambda) * max_requests_per_interval / 2) + (max_requests_per_interval / 2)) + 1

            for _ in range(num_requests_this_interval):
                operation_start = time.time()
                key = f"key_{np.random.randint(1000)}"
                r.set(key, "value")
                operation_end = time.time()
                total_latency += (operation_end - operation_start)
                total_requests += 1

            if time.time() >= start_time + metrics_interval_seconds:
                avg_throughput = total_requests / metrics_interval_seconds
                avg_latency = total_latency / total_requests if total_requests else 0
                cpu_usage, memory_usage = cpumemory.cpu_memory_avg()  # Retrieve CPU and memory averages
                node_count = len(r.cluster_nodes())

                metrics = {
                    'Time': time.time(),
                    'Average Throughput': avg_throughput,
                    'Average Latency': avg_latency,
                    'CPU Usage %': cpu_usage,
                    'Memory Usage %': memory_usage,
                    'Node Count': node_count
                }
                print(metrics)
                writer.writerow(metrics)

                # Reset for next interval
                start_time += metrics_interval_seconds
                total_requests = 0
                total_latency = 0.0

if __name__ == "__main__":
    perform_requests(0.01, 30, 600, 250)
