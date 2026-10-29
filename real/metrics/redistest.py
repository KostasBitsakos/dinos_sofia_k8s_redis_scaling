import redis

# Connect to Redis server
r = redis.Redis(host='localhost', port=6379, db=0)

# Set a key
r.set('foo', 'bar')

# Get the key
value = r.get('foo')
print(value)  # Output: b'bar'

