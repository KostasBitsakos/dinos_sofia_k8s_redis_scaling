import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Load Kubernetes configuration
config.load_kube_config()

# Create an instance of the CoreV1Api
v1 = client.CoreV1Api()

def get_redis_pod_names(namespace='default', label_selector='app=redis'):
    """
    Fetch Redis pod names based on a given label selector.
    """
    pod_names = []
    try:
        pod_list = v1.list_namespaced_pod(namespace, label_selector=label_selector)
        for pod in pod_list.items:
            pod_names.append(pod.metadata.name)
    except ApiException as e:
        print("Exception when calling CoreV1Api->list_namespaced_pod: %s\n" % e)

    return pod_names

def monitor_redis_pods(interval=5, namespace='default', label_selector='app=redis'):
    """
    Monitor Redis pods in the cluster based on the given label selector.
    """
    print("Starting Redis pod monitoring...")
    while True:
        pod_names = get_redis_pod_names(namespace, label_selector)
        print(f"Redis Pods at {time.strftime('%Y-%m-%d %H:%M:%S')}: {pod_names}")
        time.sleep(interval)

if __name__ == "__main__":
    monitor_redis_pods()

