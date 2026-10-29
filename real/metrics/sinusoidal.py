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
max_rate = 100  # Max requests per second
period = 10  # Period of the sinusoid in seconds

start_time = time.time()

while time.time() - start_time < duration:
    operations = 0  # Number of operations this second
    latencies = []  # Store latencies for each operation

    elapsed = time.time() - start_time
    rate = (np.sin(elapsed * (2 * np.pi / period)) + 1) / 2 * max_rate  # Sinusoidal modulation

    loop_start = time.time()
    while time.time() - loop_start < 1:
        op_start = time.time()
        rc.set('key', 'value')  # Example operation
        latencies.append(time.time() - op_start)
        operations += 1

    # Calculate throughput and average latency for the past second
    throughput = operations
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"Time: {elapsed:.2f}s, Throughput: {throughput} ops/s, Avg Latency: {avg_latency:.5f} s")

    # Adjust sleep to synchronize with the start of the next second
    time.sleep(max(0, 1 - (time.time() - loop_start)))

