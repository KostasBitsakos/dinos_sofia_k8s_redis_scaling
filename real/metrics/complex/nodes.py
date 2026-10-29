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

def fetch_cluster_nodes(startup_nodes):
    """
    Fetch the current nodes in the Redis cluster.
    """
    rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)
    current_nodes = rc.cluster_nodes()
    # Adjusted to directly iterate over the list of node dicts
    return [(node["host"], node["port"]) for node in current_nodes]

def monitor_redis_cluster_nodes(startup_nodes, interval=5):
    """
    Monitor changes in the Redis cluster nodes every 'interval' seconds.
    """
    print("Starting Redis cluster node monitoring...")
    previous_nodes = set(fetch_cluster_nodes(startup_nodes))
    print("Initial nodes:", previous_nodes)

    while True:
        time.sleep(interval)
        current_nodes = set(fetch_cluster_nodes(startup_nodes))
        if current_nodes != previous_nodes:
            added_nodes = current_nodes - previous_nodes
            removed_nodes = previous_nodes - current_nodes
            print("Change detected:")
            if added_nodes:
                print("Added nodes:", added_nodes)
            if removed_nodes:
                print("Removed nodes:", removed_nodes)
            previous_nodes = current_nodes

if __name__ == "__main__":
    monitor_redis_cluster_nodes(startup_nodes)

