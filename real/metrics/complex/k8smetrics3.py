# metrics.py
import time
from kubernetes import client, config
from nodes5 import get_current_nodes  # Adjust the import statement as needed

# Load Kubernetes configuration
config.load_kube_config()

# Create an instance of the API for metrics
v1_metrics = client.CustomObjectsApi()

# Node capacity (for calculations)
total_cpu_cores = 40  # Total CPU cores as from `kubectl describe nodes`
total_memory_bytes = 201301324 * 1024  # Converted Ki to bytes

def fetch_pod_metrics(pod_name, namespace='default'):
    """
    Fetch CPU and memory usage metrics for the given pod name.
    """
    try:
        metrics = v1_metrics.get_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods",
            name=pod_name,
        )
        cpu_usage = int(metrics['containers'][0]['usage']['cpu'].rstrip('n'))  # Removing 'n' from nanocores and converting to int
        memory_usage = int(metrics['containers'][0]['usage']['memory'].rstrip('Ki')) * 1024  # Converting Ki to bytes
        return cpu_usage, memory_usage
    except Exception as e:
        print(f"Error fetching metrics for pod {pod_name}: {e}")
        return None, None

def print_cluster_metrics(interval=5):
    """
    Continuously print CPU and memory metrics for the current Redis cluster nodes as percentages of total capacity.
    """
    print("Starting metrics monitoring for Redis cluster nodes...")
    while True:
        total_cpu_usage = 0
        total_memory_usage = 0
        current_nodes = get_current_nodes()
        node_count = len(current_nodes)
        for node in current_nodes:
            cpu_usage, memory_usage = fetch_pod_metrics(node['pod_name'])
            if cpu_usage is not None and memory_usage is not None:
                total_cpu_usage += cpu_usage
                total_memory_usage += memory_usage
        avg_cpu_usage_percentage = (total_cpu_usage / (total_cpu_cores * 1e9)) * 100  # Assuming cpu_usage is in nanocores
        avg_memory_usage_percentage = (total_memory_usage / total_memory_bytes) * 100
        print(f"Average CPU Usage: {avg_cpu_usage_percentage:.3f}%, Average Memory Usage: {avg_memory_usage_percentage:.3f}%")
        time.sleep(interval)

if __name__ == "__main__":
    print_cluster_metrics()

