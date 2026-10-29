import time
import csv
import numpy as np
import rediscluster
import cpumemory

def get_redis_cluster_nodes():
    client = rediscluster.RedisCluster(startup_nodes=[{"host": "10.244.37.177", "port": "6379"}], decode_responses=True)
    nodes_info = client.cluster_nodes()
    print(nodes_info)  # Debug: Print the structure of nodes_info
    master_nodes = []
    for node_id, node_info in nodes_info.items():
        if 'master' in node_info['flags']:
            master_nodes.append({"host": node_info['host'], "port": node_info['port']})
    return master_nodes


def perform_requests(rate_lambda, metrics_interval_seconds, total_duration_seconds, max_requests_per_interval):
    startup_nodes = get_redis_cluster_nodes()  # Fetch the current cluster nodes dynamically
    r = rediscluster.RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)

    with open('metricsnew.csv', 'w', newline='') as csvfile:
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
                r.get(key)
                operation_end = time.time()
                total_latency += (operation_end - operation_start)
                total_requests += 1

            if time.time() >= start_time + metrics_interval_seconds:
                avg_throughput = total_requests / metrics_interval_seconds
                avg_latency = total_latency / total_requests if total_requests else 0
                cpu_usage, memory_usage = cpumemory.cpu_memory_avg()
                node_count = len(get_redis_cluster_nodes())  # Recalculate node count dynamically

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

                # Reset for the next interval
                start_time += metrics_interval_seconds
                total_requests = 0
                total_latency = 0.0

if __name__ == "__main__":
    perform_requests(0.01, 30, 600, 250)
