# metrics.py
import time
from kubernetes import client, config
from nodes5 import get_current_nodes  # Adjust the import statement as needed

# Load Kubernetes configuration
config.load_kube_config()

# Create an instance of the API for metrics
v1_metrics = client.CustomObjectsApi()

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
        cpu_usage = metrics['containers'][0]['usage']['cpu']
        memory_usage = metrics['containers'][0]['usage']['memory']
        return cpu_usage, memory_usage
    except Exception as e:
        print(f"Error fetching metrics for pod {pod_name}: {e}")
        return "N/A", "N/A"

def print_cluster_metrics(interval=5):
    """
    Continuously print CPU and memory metrics for the current Redis cluster nodes.
    """
    print("Starting metrics monitoring for Redis cluster nodes...")
    while True:
        current_nodes = get_current_nodes()
        for node in current_nodes:
            cpu_usage, memory_usage = fetch_pod_metrics(node['pod_name'])
            print(f"Node {node['host']}:{node['port']} - CPU: {cpu_usage}, Memory: {memory_usage}")
        time.sleep(interval)

if __name__ == "__main__":
    print_cluster_metrics()

