from kubernetes import client, config
import time
import re

# Function to parse Kubernetes quantities (e.g., CPU and memory specifications).
def parse_quantity(quantity):
    unit_factors = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
                    "m": 1/1000, "": 1, "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4, "n": 1/10**9}
    number, unit = re.match(r'([0-9.]+)([a-zA-Z]*)', quantity).groups()
    return float(number) * unit_factors.get(unit, 1)

# Main function to calculate and print the average CPU and memory usage.
def calculate_average_metrics(namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    metrics_api = client.CustomObjectsApi()
    group = 'metrics.k8s.io'
    version = 'v1beta1'
    plural = 'pods'

    while True:
        total_cpu_usage = 0
        total_cpu_limit = 0
        total_containers = 0

        pod_metrics_list = metrics_api.list_namespaced_custom_object(group, version, namespace, plural)
        pods_list = v1.list_namespaced_pod(namespace)

        for pod_metrics in pod_metrics_list['items']:
            for container_metrics in pod_metrics['containers']:
                container_name = container_metrics['name']
                if "redis" in container_name:
                    cpu_usage = parse_quantity(container_metrics['usage']['cpu'])
                    # Find the corresponding pod and container to get the CPU limit.
                    pod_name = pod_metrics['metadata']['name']
                    pod_spec = next(filter(lambda pod: pod.metadata.name == pod_name, pods_list.items), None)
                    if pod_spec:
                        container_spec = next(filter(lambda c: c.name == container_name, pod_spec.spec.containers), None)
                        if container_spec and container_spec.resources and container_spec.resources.limits and 'cpu' in container_spec.resources.limits:
                            cpu_limit = parse_quantity(container_spec.resources.limits['cpu'])
                            total_cpu_usage += cpu_usage
                            total_cpu_limit += cpu_limit
                            total_containers += 1

        if total_containers > 0 and total_cpu_limit > 0:
            avg_cpu_usage_percentage = (total_cpu_usage / total_cpu_limit) * 100
            print(f"Average CPU Usage Percentage: {avg_cpu_usage_percentage:.2f}%")
        else:
            print("No containers with 'redis' in their name were found or no CPU limits set.")

        time.sleep(5)

if __name__ == '__main__':
    try:
        calculate_average_metrics('default')
    except KeyboardInterrupt:
        print("Script interrupted by user.")

