from kubernetes import client, config
import time
import re

# Function to parse Kubernetes quantity strings (like '200Mi' or '2Gi') into bytes
def parse_quantity(quantity):
    unit_factors = {
        "Ki": 2**10,
        "Mi": 2**20,
        "Gi": 2**30,
        "Ti": 2**40,
        "m": 1 / 1000,
        "": 1,
        "n": 1 / 10**9,
    }
    match = re.match(r"([0-9]+)([a-zA-Z]*)", quantity)
    if match:
        number, unit = match.groups()
        return float(number) * unit_factors.get(unit, 1)
    return 0


# Function to calculate and print the CPU and memory usage percentages
# def cpu_memory_avg():
#     config.load_kube_config()  # Load kubeconfig
#     v1 = client.CoreV1Api()
#     custom_api = client.CustomObjectsApi()
#     cpu_percentage = 0
#     memody_usage_percentage = 0
#     pods = v1.list_namespaced_pod(
#         namespace="default", label_selector="redis=redis-cluster-pods-lbl"
#     ).items
#     for pod in pods:
#         pod_name = pod.metadata.name

#         metrics = custom_api.get_namespaced_custom_object(
#             "metrics.k8s.io", "v1beta1", "default", "pods", pod_name
#         )
#         for container in pod.spec.containers:
#             container_metrics = next(
#                 (x for x in metrics["containers"] if x["name"] == container.name), None
#             )
#             if container_metrics:
#                 cpu_usage = parse_quantity(
#                     container_metrics["usage"]["cpu"]
#                 )  # Convert to core units
#                 memory_usage = parse_quantity(
#                     container_metrics["usage"]["memory"]
#                 )  # Convert to bytes

#                 # Fetch the resource limits
#                 cpu_limit = (
#                     parse_quantity(container.resources.limits.get("cpu", "0m"))
#                     if container.resources.limits
#                     else 0
#                 )
#                 memory_limit = (
#                     parse_quantity(container.resources.limits.get("memory", "0Mi"))
#                     if container.resources.limits
#                     else 0
#                 )

#                 # Calculate percentages
#                 cpu_usage_percentage = (
#                     (cpu_usage / cpu_limit * 100) if cpu_limit > 0 else 0
#                 )
#                 memory_usage_percentage = (
#                     (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
#                 )
#     return cpu_usage_percentage, memory_usage_percentage


def get_avg_pod_metrics(
    namespace="default", label_selector="redis=redis-cluster-pods-lbl"
):
    """
    Fetch and calculate average CPU and memory usage for pods matching the label selector.
    """
    pod_metrics = metrics_api.list_namespaced_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        namespace=namespace,
        plural="pods",
        label_selector=label_selector,
    )

    total_cpu_usage_millicores, total_memory_usage_mib, pod_count = 0, 0, 0

    for pod in pod_metrics.get("items", []):
        for container in pod.get("containers", []):
            cpu_usage_cores = parse_quantity(
                container["usage"]["cpu"]
            )  # CPU usage in cores
            memory_usage_bytes = parse_quantity(
                container["usage"]["memory"]
            )  # Memory usage in bytes

            total_cpu_usage_millicores += (
                cpu_usage_cores * 1000
            )  # Convert cores to millicores
            total_memory_usage_mib += memory_usage_bytes / (
                2**20
            )  # Convert bytes to MiB
            pod_count += 1

    avg_cpu_usage_millicores = (
        total_cpu_usage_millicores / pod_count if pod_count else 0
    )
    avg_memory_usage_mib = total_memory_usage_mib / pod_count if pod_count else 0

    return avg_cpu_usage_millicores, avg_memory_usage_mib


if __name__ == "__main__":
    print(cpu_memory_avg())
    print(get_avg_pod_metrics)
