import redis
import numpy as np
import time
import csv
from rediscluster import RedisCluster
import math

# Define startup nodes
startup_nodes = [
    {"host": "10.244.37.149", "port": 6379},
    {"host": "10.244.37.183", "port": 6379},
    {"host": "10.244.37.182", "port": 6379},
    {"host": "10.244.37.150", "port": 6379},
    {"host": "10.244.37.169", "port": 6379},
    {"host": "10.244.37.154", "port": 6379}
]

# Initialize Redis Cluster client
r = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)

def send_sinusoidal_requests_and_log_metrics(amplitude=1, period=60, duration_seconds=300):
    start_time = time.time()
    end_time = start_time + duration_seconds
    total_requests = 0
    latencies = []

    with open('metrics2.csv', 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'total_requests', 'throughput', 'average_latency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while time.time() < end_time:
            current_time = time.time()
            # Sinusoidal interval
            interval = amplitude * (1 + math.sin(2 * math.pi * (current_time - start_time) / period))
            time.sleep(max(0.01, interval))  # Ensure there's a minimum interval

            # Simulate Redis operations
            key = f"key_{total_requests}"
            value = f"value_{total_requests}"
            operation_start = time.time()
            r.set(key, value)  # Example Redis operation: set a value
            r.get(key)  # Example Redis operation: get a value
            operation_end = time.time()

            latencies.append(operation_end - operation_start)
            total_requests += 1

            # Compute metrics periodically
            if int(current_time - start_time) % 5 == 0:
                throughput = total_requests / (current_time - start_time)
                avg_latency = sum(latencies) / len(latencies) if latencies else 0

                print(f"Total requests: {total_requests}, Throughput: {throughput:.2f} requests/s, Average latency: {avg_latency:.5f} s")

                writer.writerow({'timestamp': current_time, 'total_requests': total_requests, 'throughput': throughput, 'average_latency': avg_latency})

if __name__ == "__main__":
    send_sinusoidal_requests_and_log_metrics(amplitude=1, period=60, duration_seconds=300)
