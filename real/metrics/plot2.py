import matplotlib.pyplot as plt
import pandas as pd

# Initialize a dictionary to hold your data
data = {
    'Time': [],
    'Throughput': [],
    'Avg Latency': [],
    'Avg CPU Usage': [],
    'Avg Memory Usage': [],
}

# Open and read the data file
with open('data.txt', 'r') as file:
    for line in file:
        if not line.strip():  # Skip empty lines
            continue
        try:
            parts = line.strip().split(', ')
            time = float(parts[0].split(': ')[1].replace('s', ''))
            throughput = int(parts[1].split(': ')[1].split(' ')[0])
            avg_latency = float(parts[2].split(': ')[1].split(' ')[0])
            avg_cpu_usage = float(parts[3].split(': ')[1].replace('m', ''))
            avg_memory_usage = float(parts[4].split(': ')[1].replace('Mi', ''))

            data['Time'].append(time)
            data['Throughput'].append(throughput)
            data['Avg Latency'].append(avg_latency)
            data['Avg CPU Usage'].append(avg_cpu_usage)
            data['Avg Memory Usage'].append(avg_memory_usage)
        except Exception as e:
            print(f"Error processing line: '{line.strip()}'. Error: {e}")

# Convert the data dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(df['Time'], df['Throughput'], label='Throughput (ops/s)', color='red')
plt.plot(df['Time'], df['Avg Latency'], label='Avg Latency (s)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Throughput / Latency')
plt.title('Throughput and Latency over Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['Time'], df['Avg CPU Usage'], label='Avg CPU Usage', color='green')
plt.plot(df['Time'], df['Avg Memory Usage'], label='Avg Memory Usage (Mi)', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Usage')
plt.title('CPU and Memory Usage over Time')
plt.legend()

plt.tight_layout()
plt.show()

