import redis
import random
import string

def generate_random_string(length=20):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

r = redis.Redis(host='localhost', port=6379, db=0)

# Example of setting large values
for _ in range(10000):
    key = generate_random_string(10)
    value = generate_random_string(2048)  # 2 KB values
    r.set(key, value)

# Example of working with lists
for _ in range(1000):
    list_key = generate_random_string(10)
    for __ in range(100):
        r.lpush(list_key, generate_random_string(1024))  # 1 KB values

# Example of working with sets
for _ in range(1000):
    set_key = generate_random_string(10)
    for __ in range(100):
        r.sadd(set_key, generate_random_string(1024))  # 1 KB values

