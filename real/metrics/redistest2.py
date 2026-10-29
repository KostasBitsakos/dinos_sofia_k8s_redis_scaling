from rediscluster import RedisCluster

startup_nodes = [{"host": "127.0.0.1", "port": "6379"}]
r = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)

r.set("foo", "bar")
print(r.get("foo"))

