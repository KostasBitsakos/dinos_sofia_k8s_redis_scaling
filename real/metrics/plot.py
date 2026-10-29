import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming data is stored in 'data.txt' and each line is formatted as you described
data = {
    'Time': [],
    'Throughput': [],
    'Avg Latency': [],
    'Avg CPU Usage': [],
    'Avg Memory Usage': [],
}

with open('data.txt', 'r') as f:
    for line in f.readlines():
        parts = line.strip().split(', ')
        data['Time'].append(float(parts[0].split(': ')[1][:-1]))
        data['Throughput'].append(int(parts[1].split(': ')[1].split(' ')[0]))
        data['Avg Latency'].append(float(parts[2].split(': ')[1].split(' ')[0]))
        data['Avg CPU Usage'].append(float(parts[3].split(': ')[1][:-1]))
        data['Avg Memory Usage'].append(float(parts[4].split(': ')[1][:-2]))

df = pd.DataFrame(data)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Throughput (ops/s)', color=color)
ax1.plot(df['Time'], df['Throughput'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Avg Latency (s)', color=color)
ax2.plot(df['Time'], df['Avg Latency'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Throughput and Latency over Time')
plt.show()

fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Avg CPU Usage (m)', color=color)
ax1.plot(df['Time'], df['Avg CPU Usage'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:purple'
ax2.set_ylabel('Avg Memory Usage (Mi)', color=color)
ax2.plot(df['Time'], df['Avg Memory Usage'], color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('CPU and Memory Usage over Time')
plt.show()

