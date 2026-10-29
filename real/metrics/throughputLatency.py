from rediscluster import RedisCluster
import time

startup_nodes = [
    {"host": "10.244.37.170", "port": 6379},
    {"host": "10.244.37.138", "port": 6379},
    {"host": "10.244.37.191", "port": 6379},
    {"host": "10.244.37.162", "port": 6379},
    {"host": "10.244.37.159", "port": 6379},
    {"host": "10.244.37.153", "port": 6379},
    # Add the rest of your IPs here as needed
]

# Initialize the Redis cluster client
rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)

def measure_performance(client, operation_count=10000):
    # Measure throughput and latency for SET operations
    start_time = time.time()
    for i in range(operation_count):
        client.set(f"key{i}", f"value{i}")
    end_time = time.time()

    elapsed_time_set = end_time - start_time
    throughput_set = operation_count / elapsed_time_set
    latency_set = elapsed_time_set / operation_count

    # Measure throughput and latency for GET operations
    start_time = time.time()
    for i in range(operation_count):
        value = client.get(f"key{i}")
    end_time = time.time()

    elapsed_time_get = end_time - start_time
    throughput_get = operation_count / elapsed_time_get
    latency_get = elapsed_time_get / operation_count

    return (throughput_set, latency_set), (throughput_get, latency_get)

(throughput_set, latency_set), (throughput_get, latency_get) = measure_performance(rc)

print(f"SET Throughput: {throughput_set} operations per second, Latency: {latency_set} seconds per operation")
print(f"GET Throughput: {throughput_get} operations per second, Latency: {latency_get} seconds per operation")

