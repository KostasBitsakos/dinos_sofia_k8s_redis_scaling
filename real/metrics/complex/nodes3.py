import time
from rediscluster import RedisCluster

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

def get_current_nodes():
    """
    Fetch the current nodes in the Redis cluster and their corresponding Kubernetes pod names.
    """
    nodes_info = rc.cluster_nodes()
    current_nodes = []
    for node_info in nodes_info:
        # Example IP to Pod Name Mapping (Adjust according to your actual naming convention)
        ip = node_info['host']
        pod_ip_segment = ip.replace('.', '-')
        pod_name = f"redis-autoscale-cluster-{pod_ip_segment}-{node_info['port']}"  # Example naming convention
        current_nodes.append({
            'host': node_info['host'],
            'port': node_info['port'],
            'role': 'master' if 'master' in node_info['flags'] else 'slave',
            'pod_name': pod_name,  # This assumes you have a way to map node info to pod names
        })
    return current_nodes

def monitor_redis_cluster_nodes(interval=5):
    """
    Monitor changes in the Redis cluster nodes every 'interval' seconds.
    """
    print("Starting Redis cluster node monitoring...")
    while True:
        current_nodes = get_current_nodes()
        for node in current_nodes:
            print(f"Node: {node['host']}:{node['port']}, Role: {node['role']}, Pod Name: {node['pod_name']}")
        time.sleep(interval)

if __name__ == "__main__":
    monitor_redis_cluster_nodes()

