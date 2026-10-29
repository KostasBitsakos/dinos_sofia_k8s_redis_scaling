from kubernetes import client, config
import time
import re

def parse_quantity(quantity):
    """
    Parse Kubernetes quantity string and convert it to a common unit.
    """
    unit_factors = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
                    "m": 1/1000, "": 1, "n": 1/10**9}
    number, unit = re.match(r'([0-9.]+)([a-zA-Z]*)', quantity).groups()
    return float(number) * unit_factors.get(unit, 1)

def calculate_average_metrics(namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    metrics_api = client.CustomObjectsApi()
    group = 'metrics.k8s.io'
    version = 'v1beta1'
    plural = 'pods'

    while True:
        total_cpu_usage_percentage = 0
        total_memory_usage_percentage = 0
        total_containers = 0

        pod_metrics_list = metrics_api.list_namespaced_custom_object(group, version, namespace, plural)
        pods_list = v1.list_namespaced_pod(namespace)

        for pod_metrics in pod_metrics_list['items']:
            pod_name = pod_metrics['metadata']['name']
            pod_spec = next((pod for pod in pods_list.items if pod.metadata.name == pod_name), None)

            if pod_spec:
                for container_metrics, container_spec in zip(pod_metrics['containers'], pod_spec.spec.containers):
                    if "redis" in container_spec.name:
                        if container_spec.resources.limits:
                            cpu_limit = parse_quantity(container_spec.resources.limits.get('cpu', '0'))
                            memory_limit = parse_quantity(container_spec.resources.limits.get('memory', '0'))
                            cpu_usage = parse_quantity(container_metrics['usage']['cpu'])
                            memory_usage = parse_quantity(container_metrics['usage']['memory'])

                            if cpu_limit > 0 and memory_limit > 0:
                                cpu_usage_percentage = (cpu_usage / cpu_limit) * 100
                                memory_usage_percentage = (memory_usage / memory_limit) * 100

                                total_cpu_usage_percentage += cpu_usage_percentage
                                total_memory_usage_percentage += memory_usage_percentage
                                total_containers += 1

        if total_containers > 0:
            avg_cpu_usage_percentage = total_cpu_usage_percentage / total_containers
            avg_memory_usage_percentage = total_memory_usage_percentage / total_containers
            print(f"Average CPU Usage: {avg_cpu_usage_percentage:.2f}%")
            print(f"Average Memory Usage: {avg_memory_usage_percentage:.2f}%")
        else:
            print("No matching containers found.")

        time.sleep(5)

if __name__ == '__main__':
    try:
        calculate_average_metrics('default')
    except KeyboardInterrupt:
        print("Script interrupted by user.")

