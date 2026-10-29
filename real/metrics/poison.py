from rediscluster import RedisCluster
import numpy as np
import time

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

    print(f"Time: {elapsed:.2f}s, Throughput: {throughput} ops/s, Avg Latency: {avg_latency:.5f} s")

    # Sleep for 1 second before next iteration
    time.sleep(1)


