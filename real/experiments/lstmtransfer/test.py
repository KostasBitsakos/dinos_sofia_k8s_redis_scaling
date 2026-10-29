import logging
from rediscluster import RedisCluster

# Configure basic logging (optional)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_redis_client():
    # Replace these IPs with your actual Redis node IPs
    redis_ips = ["10.244.37.148", "10.244.37.134", "10.244.37.146"]
    startup_nodes = [{"host": ip, "port": "6379"} for ip in redis_ips]
    
    try:
        client = RedisCluster(
            startup_nodes=startup_nodes,
            decode_responses=True,
            skip_full_coverage_check=True
        )
        logging.info(f"Connected to Redis Cluster with nodes: {', '.join(redis_ips)}")
        return client
    except Exception as e:
        logging.error(f"Failed to create Redis client: {e}")
        exit(1)

def main():
    redis_client = create_redis_client()

    commands = {
        "CLUSTER SLOTS": None,
        "CLUSTER NODES": None,
        "CLUSTER INFO": None,
    }

    for cmd in commands.keys():
        try:
            result = redis_client.execute_command(cmd)
            print(f"\n=== Command: {cmd} ===")
            print("Type:", type(result))
            print("Output:")
            print(result)
            print("=" * 60)
        except Exception as e:
            print(f"Error executing {cmd}: {e}")

if __name__ == "__main__":
    main()
