import redis
from datetime import datetime
import time
from kubernetes import client, config
import re

# Function to parse Kubernetes quantity string
def parse_quantity(quantity):
    unit_factors = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
                    "m": 1/1000, "": 1, "n": 1/10**9}
    number, unit = re.match(r'([0-9.]+)([a-zA-Z]*)', quantity).groups()
    return float(number) * unit_factors.get(unit, 1)

# Function to measure Redis throughput and latency
def measure_redis_performance(redis_host='localhost', redis_port=6379, test_duration_seconds=5):
    r = redis.Redis(host=redis_host, port=redis_port)
    start_time = datetime.now()
    operations = 0
    total_latency = 0

    while (datetime.now() - start_time).seconds < test_duration_seconds:
        start_operation_time = datetime.now()
        r.set('test_key', 'test_value')
        r.get('test_key')
        end_operation_time = datetime.now()

        operation_latency = (end_operation_time - start_operation_time).total_seconds() * 1000  # Convert to milliseconds
        total_latency += operation_latency
        operations += 1

    average_latency = total_latency / operations
    throughput = operations / test_duration_seconds

    return throughput, average_latency

def calculate_average_metrics(namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    metrics_api = client.CustomObjectsApi()

    while True:
        # Existing code to calculate average CPU and memory usage
        # ...

        # Measure Redis performance
        throughput, average_latency = measure_redis_performance(redis_host='your-redis-service', redis_port=6379)
        print(f"Redis Throughput: {throughput:.2f} ops/sec")
        print(f"Redis Average Latency: {average_latency:.2f} ms")

        time.sleep(5)

if __name__ == '__main__':
    try:
        calculate_average_metrics('default')
    except KeyboardInterrupt:
        print("Script interrupted by user.")

