from kubernetes import client, config
import time
import re

# Function to parse Kubernetes quantity strings (like '200Mi' or '2Gi') into bytes
def parse_quantity(quantity):
    unit_factors = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "m": 1/1000, "": 1, "n": 1/10**9}
    match = re.match(r"([0-9]+)([a-zA-Z]*)", quantity)
    if match:
        number, unit = match.groups()
        return float(number) * unit_factors.get(unit, 1)
    return 0

# Function to calculate and print the CPU and memory usage percentages
def print_avg_pod_metrics_every_5_seconds():
    config.load_kube_config()  # Load kubeconfig
    v1 = client.CoreV1Api()
    custom_api = client.CustomObjectsApi()

    while True:
        total_cpu_usage = total_memory_usage = total_cpu_limit = total_memory_limit = 0
        pod_count = 0

        pods = v1.list_namespaced_pod(namespace="default", label_selector="redis=redis-cluster-pods-lbl").items
        for pod in pods:
            pod_name = pod.metadata.name
            try:
                metrics = custom_api.get_namespaced_custom_object(
                    "metrics.k8s.io", "v1beta1", "default", "pods", pod_name)
                for container in pod.spec.containers:
                    container_metrics = next((x for x in metrics['containers'] if x['name'] == container.name), None)
                    if container_metrics:
                        cpu_usage = parse_quantity(container_metrics['usage']['cpu'])  # Convert to core units
                        memory_usage = parse_quantity(container_metrics['usage']['memory'])  # Convert to bytes
                        
                        # Accumulate resource limits and usages
                        cpu_limit = parse_quantity(container.resources.limits.get('cpu', '0m')) if container.resources.limits else 0
                        memory_limit = parse_quantity(container.resources.limits.get('memory', '0Mi')) if container.resources.limits else 0
                        total_cpu_usage += cpu_usage
                        total_memory_usage += memory_usage
                        total_cpu_limit += cpu_limit
                        total_memory_limit += memory_limit
                        pod_count += 1

            except Exception as e:
                print(f"Error fetching metrics for pod {pod_name}: {e}")

        # Calculate and print averages
        avg_cpu_usage_percentage = (total_cpu_usage / total_cpu_limit * 100) if total_cpu_limit > 0 else 0
        avg_memory_usage_percentage = (total_memory_usage / total_memory_limit * 100) if total_memory_limit > 0 else 0
        print(f"Average CPU Usage: {avg_cpu_usage_percentage:.2f}%, Average Memory Usage: {avg_memory_usage_percentage:.2f}%")
        
        print("-----")
        time.sleep(5)

if __name__ == "__main__":
    print_avg_pod_metrics_every_5_seconds()
