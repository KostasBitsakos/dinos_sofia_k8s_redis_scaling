from kubernetes import client, config
import re

def parse_quantity(quantity):
    unit_factors = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4, "m": 1/1000, "": 1, "n": 1/10**9}
    number, unit = re.match(r'([0-9.]+)([a-zA-Z]*)', quantity).groups()
    return float(number) * unit_factors.get(unit, 1)

def calculate_average_usage(namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    metrics_api = client.CustomObjectsApi()
    group = 'metrics.k8s.io'
    version = 'v1beta1'
    plural = 'pods'
    
    total_cpu_usage = 0
    total_memory_usage = 0
    total_containers = 0
    
    pod_metrics_list = metrics_api.list_namespaced_custom_object(group, version, namespace, plural)
    pods_list = v1.list_namespaced_pod(namespace)
    
    for pod_metrics in pod_metrics_list['items']:
        pod_name = pod_metrics['metadata']['name']
        if "redis" in pod_name:
            for container_metrics in pod_metrics['containers']:
                cpu_usage = parse_quantity(container_metrics['usage']['cpu'])
                memory_usage = parse_quantity(container_metrics['usage']['memory'])
                
                total_cpu_usage += cpu_usage
                total_memory_usage += memory_usage
                total_containers += 1
                
    if total_containers > 0:
        avg_cpu_usage = total_cpu_usage / total_containers
        avg_memory_usage = total_memory_usage / total_containers
        return avg_cpu_usage, avg_memory_usage
    else:
        return 0, 0

