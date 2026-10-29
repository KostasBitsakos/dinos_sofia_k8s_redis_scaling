from rediscluster import RedisCluster
import numpy as np
import time
import csv

# Setup for connecting to the Redis Cluster
startup_nodes = [
    {"host": "10.244.37.149", "port": 6379},
    {"host": "10.244.37.183", "port": 6379},
    {"host": "10.244.37.182", "port": 6379},
    {"host": "10.244.37.154", "port": 6379},
    {"host": "10.244.37.169", "port": 6379},
    {"host": "10.244.37.150", "port": 6379}
]

# Connect to Redis Cluster
r = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)

def send_poisson_requests_and_log_metrics(rate_lambda=5, duration_seconds=5):
    """
    Send Redis requests at intervals following a Poisson distribution, print and log metrics every 5 seconds.
    
    :param rate_lambda: The rate (lambda) parameter of the Poisson distribution, average number of events per second.
    :param duration_seconds: The duration to run the simulation, in seconds.
    """
    end_time = time.time() + duration_seconds
    total_requests = 0
    latencies = []
    
    # Open CSV file for writing
    with open('redis_metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Throughput (req/sec)", "Average Latency (sec)"])

        while time.time() < end_time:
            # Poisson-distributed interval (in seconds)
            interval = np.random.poisson(rate_lambda)
            next_request_time = time.time() + interval / rate_lambda

            # Waiting for the next request time or 5 seconds to print metrics
            while time.time() < next_request_time:
                if time.time() >= end_time:
                    break
                time.sleep(max(0, min(5, next_request_time - time.time())))

                # Every 5 seconds, calculate and print metrics so far
                if latencies:
                    throughput = total_requests / sum(latencies)
                    avg_latency = sum(latencies) / len(latencies)
                    print(f"Throughput: {throughput:.2f} requests/sec, Average Latency: {avg_latency:.4f} sec")
                    # Write metrics to CSV
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), throughput, avg_latency])
                else:
                    print("Waiting for the first request...")

            # Simulate a Redis operation
            start_time = time.time()
            r.set(f'key{total_requests}', f'value{total_requests}')
            r.get(f'key{total_requests}')
            operation_end = time.time()

            latencies.append(operation_end - start_time)
            total_requests += 1

if __name__ == "__main__":
    send_poisson_requests_and_log_metrics(rate_lambda=5, duration_seconds=300)  # Run for 5 minutes
