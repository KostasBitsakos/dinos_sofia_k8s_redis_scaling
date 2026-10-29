from rediscluster import RedisCluster
import traceback

def connect_to_redis_cluster(startup_nodes):
    try:
        # Attempt to create a Redis Cluster connection
        r = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)
        print("Connected to Redis Cluster successfully.")
        return r
    except Exception as e:
        print(f"Failed to connect to Redis Cluster: {str(e)}")
        traceback.print_exc()
        return None

def perform_basic_operations(redis_cluster):
    try:
        # Perform basic set and get operations
        redis_cluster.set("test_key", "hello")
        value = redis_cluster.get("test_key")
        print(f"Value for 'test_key': {value}")
    except Exception as e:
        print(f"Error performing operations on Redis Cluster: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    startup_nodes = [
    {"host": "10.244.37.173", "port": 6379},
    {"host": "10.244.37.186", "port": 6379},
    {"host": "10.244.37.147", "port": 6379},
    {"host": "10.244.37.158", "port": 6379},
    {"host": "10.244.37.191", "port": 6379},
    {"host": "10.244.37.174", "port": 6379},
]


    redis_cluster = connect_to_redis_cluster(startup_nodes)
    if redis_cluster:
        perform_basic_operations(redis_cluster)
