from kubernetes import client, config
import time
import re

def parse_quantity(quantity):
    """
    Parse Kubernetes quantity string (e.g., "100Mi", "1Gi", "100m") and convert it to a common unit (bytes for memory, cores for CPU).
    This version adds support for CPU in nanocores (n) and corrects the handling of milliCPUs (m).
    """
    unit_factors = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
                    "m": 1/1000, "": 1, "n": 1/10**9}
    number, unit = re.match(r'([0-9.]+)([a-zA-Z]*)', quantity).groups()
    return float(number) * unit_factors.get(unit, 1)


def calculate_average_metrics(namespace='default'):
    config.load_kube_config()
    api_instance = client.CustomObjectsApi()
    group = 'metrics.k8s.io'
    version = 'v1beta1'
    plural = 'pods'

    while True:
        total_cpu_usage = 0
        total_memory_usage = 0
        total_containers = 0

        try:
            api_response = api_instance.list_namespaced_custom_object(group, version, namespace, plural)
            for pod_metrics in api_response['items']:
                for container in pod_metrics['containers']:
                    cpu_usage = parse_quantity(container['usage']['cpu'])
                    memory_usage = parse_quantity(container['usage']['memory'])
                    total_cpu_usage += cpu_usage
                    total_memory_usage += memory_usage
                    total_containers += 1

            if total_containers > 0:
                avg_cpu_usage = total_cpu_usage / total_containers
                avg_memory_usage = total_memory_usage / total_containers
                print(f"Average CPU Usage: {avg_cpu_usage} cores")
                print(f"Average Memory Usage: {avg_memory_usage} bytes")
            else:
                print("No containers found.")

        except client.exceptions.ApiException as e:
            print(f"Exception when calling CustomObjectsApi->list_namespaced_custom_object: {e}\n")

        time.sleep(5)

if __name__ == '__main__':
    try:
        calculate_average_metrics('default')
    except KeyboardInterrupt:
        print("Script interrupted by user.")

