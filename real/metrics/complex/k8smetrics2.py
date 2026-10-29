# metrics_average_corrected.py
import time
from kubernetes import client, config
from nodes5 import get_current_nodes  # Adjust the import statement as needed

# Load Kubernetes configuration
config.load_kube_config()

# Create an instance of the API for metrics
v1_metrics = client.CustomObjectsApi()

def parse_resource_quantity(resource_string):
    """
    Convert Kubernetes resource quantity string to a numeric value.
    """
    if resource_string.endswith("n"):
        return int(resource_string[:-1]) / 1e9  # Convert nanocores to cores for CPU
    elif resource_string.endswith("m"):
        return int(resource_string[:-1]) / 1000  # Convert millicores to cores for CPU
    elif resource_string.endswith("Ki"):
        return int(resource_string[:-2]) * 1024
    elif resource_string.endswith("Mi"):
        return int(resource_string[:-2]) * 1024**2
    elif resource_string.endswith("Gi"):
        return int(resource_string[:-2]) * 1024**3
    else:
        return int(resource_string)

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
        cpu_usage_str = metrics['containers'][0]['usage']['cpu']
        memory_usage_str = metrics['containers'][0]['usage']['memory']
        cpu_usage = parse_resource_quantity(cpu_usage_str)
        memory_usage = parse_resource_quantity(memory_usage_str)
        return cpu_usage, memory_usage
    except Exception as e:
        print(f"Error fetching metrics for pod {pod_name}: {e}")
        return 0, 0

def print_cluster_metrics(interval=5):
    """
    Continuously print the average CPU and memory metrics for the current Redis cluster nodes.
    """
    print("Starting metrics monitoring for Redis cluster nodes...")
    while True:
        current_nodes = get_current_nodes()
        total_cpu_usage = 0
        total_memory_usage = 0
        num_nodes = len(current_nodes)
        
        if num_nodes == 0:
            print("No nodes found.")
            continue
        
        for node in current_nodes:
            cpu_usage, memory_usage = fetch_pod_metrics(node['pod_name'])
            total_cpu_usage += cpu_usage
            total_memory_usage += memory_usage
        
        avg_cpu_usage = total_cpu_usage / num_nodes
        avg_memory_usage = total_memory_usage / num_nodes
        
        print(f"Average CPU Usage: {avg_cpu_usage:.4f} cores, Average Memory Usage: {avg_memory_usage / (1024**2):.4f} MiB")
        time.sleep(interval)

if __name__ == "__main__":
    print_cluster_metrics()

