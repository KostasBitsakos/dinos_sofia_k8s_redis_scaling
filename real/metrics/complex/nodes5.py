import time
from rediscluster import RedisCluster
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure access to the Kubernetes cluster
config.load_kube_config()

# Define your initial cluster nodes
startup_nodes = [
    {"host": "10.244.37.164", "port": 6379},
    {"host": "10.244.37.176", "port": 6379},
    {"host": "10.244.37.168", "port": 6379},
    {"host": "10.244.37.142", "port": 6379},
    {"host": "10.244.37.174", "port": 6379},
    {"host": "10.244.37.187", "port": 6379},
]

# Initialize RedisCluster client
rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)

v1 = client.CoreV1Api()

def fetch_pod_name_by_ip(ip_address):
    """
    Fetch Kubernetes pod name by pod IP address.
    """
    try:
        pods = v1.list_pod_for_all_namespaces(watch=False)
        for pod in pods.items:
            if pod.status.pod_ip == ip_address:
                return pod.metadata.name
    except ApiException as e:
        print(f"Exception when calling CoreV1Api->list_pod_for_all_namespaces: {e}")
    return None

def get_current_nodes():
    """
    Fetch the current nodes in the Redis cluster and their corresponding Kubernetes pod names.
    """
    nodes_info = rc.cluster_nodes()
    current_nodes = []
    for node_info in nodes_info:
        pod_name = fetch_pod_name_by_ip(node_info['host'])
        current_nodes.append({
            'host': node_info['host'],
            'port': node_info['port'],
            'role': 'master' if 'master' in node_info['flags'] else 'slave',
            'pod_name': pod_name,  # Kubernetes pod name fetched by IP
        })
    return current_nodes

def monitor_redis_cluster_nodes(interval=5):
    """
    Monitor changes in the Redis cluster nodes every 'interval' seconds, including their Kubernetes pod names.
    """
    print("Starting Redis cluster node monitoring...")
    while True:
        current_nodes = get_current_nodes()
        for node in current_nodes:
            print(f"Node: {node['host']}:{node['port']}, Role: {node['role']}, Pod Name: {node['pod_name']}")
        time.sleep(interval)

if __name__ == "__main__":
    monitor_redis_cluster_nodes()

