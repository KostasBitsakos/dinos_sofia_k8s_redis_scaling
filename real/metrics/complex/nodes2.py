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
    Fetch the current nodes in the Redis cluster.
    """
    nodes_info = rc.cluster_nodes()
    current_nodes = []
    for node in nodes_info:
        host = node.get('host')
        port = node.get('port')
        node_id = node.get('node_id', node.get('name'))  # Adjust based on your cluster_nodes() output
        role = 'master' if 'master' in node.get('flags', []) else 'slave'
        
        node_info = {
            'id': node_id,
            'host': host,
            'port': port,
            'role': role,
        }
        current_nodes.append(node_info)
    return current_nodes

def monitor_redis_cluster_nodes(interval=5):
    """
    Monitor changes in the Redis cluster nodes every 'interval' seconds.
    """
    print("Starting Redis cluster node monitoring...")
    while True:
        current_nodes = get_current_nodes()
        print(f"Current nodes at {time.strftime('%Y-%m-%d %H:%M:%S')}:", current_nodes)
        time.sleep(interval)

if __name__ == "__main__":
    monitor_redis_cluster_nodes()

