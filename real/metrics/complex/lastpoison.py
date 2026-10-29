import redis
import numpy as np
import time
import csv
from rediscluster import RedisCluster


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
r = RedisCluster(startup_nodes=startup_nodes, decode_responses=True,skip_full_coverage_check=True)

def send_poisson_requests_and_log_metrics(rate_lambda=100, duration_seconds=300):
    start_time = time.time()
    end_time = start_time + duration_seconds
    total_requests = 0
    latencies = []

    # Prepare CSV file for logging
    with open('metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'total_requests', 'throughput', 'average_latency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while time.time() < end_time:
            # Poisson-distributed interval
            interval = np.random.exponential(1/rate_lambda)
            time.sleep(interval)

            # Simulate Redis operations
            key = f"key_{total_requests}"
            value = f"value_{total_requests}"
            operation_start = time.time()
            r.set(key, value)  # Example Redis operation: set a value
            r.get(key)  # Example Redis operation: get a value
            operation_end = time.time()

            latencies.append(operation_end - operation_start)
            total_requests += 1

            # Compute metrics every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                # Compute throughput and average latency
                throughput = total_requests / (time.time() - start_time)
                avg_latency = sum(latencies) / len(latencies) if latencies else 0

                print(f"Total requests: {total_requests}, Throughput: {throughput:.2f} requests/s, Average latency: {avg_latency:.5f} s")

                # Log metrics to CSV
                writer.writerow({'timestamp': time.time(), 'total_requests': total_requests, 'throughput': throughput, 'average_latency': avg_latency})

if __name__ == "__main__":
    send_poisson_requests_and_log_metrics(rate_lambda=10, duration_seconds=300)
