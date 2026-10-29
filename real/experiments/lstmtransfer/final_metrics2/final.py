import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with your actual data)
timesteps = np.arange(2000)
load = np.sin(2 * np.pi * timesteps / 1000) * 100 + 200  # Example load data
cpu_usage = np.sin(2 * np.pi * timesteps / 800) * 0.5 + 1.0  # Example CPU usage data
memory_usage = np.sin(2 * np.pi * timesteps / 600) * 0.5 + 1.0  # Example memory usage data
latency = np.sin(2 * np.pi * timesteps / 500) * 0.1 + 0.15  # Example latency data
num_workers = np.sin(2 * np.pi * timesteps / 2000) * 2 + 3  # Example number of workers data

# Add noise ONLY to the load and latency data
noise_load = np.random.normal(0, 10, len(load))  # Noise for load (adjust standard deviation as needed)
noise_latency = np.random.normal(0, 0.01, len(latency))  # Noise for latency (adjust standard deviation as needed)

load_with_noise = load + noise_load  # Add noise to load
latency_with_noise = latency + noise_latency  # Add noise to latency

# Create a figure with 4 subplots (same structure as the original plot)
plt.figure(figsize=(15, 10))

# Plot Load Over Time with Noise
plt.subplot(2, 2, 1)
ax1 = plt.gca()
ax1.plot(timesteps, load_with_noise, label="Load", color="blue")
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Load", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(timesteps, num_workers, label="Number of Redis Workers", color="red", linestyle="--")
ax2.set_ylabel("Number of Redis Workers", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Load Over Time")
plt.legend(loc="upper left")

# Plot CPU Usage Over Time (unchanged)
plt.subplot(2, 2, 2)
ax1 = plt.gca()
ax1.plot(timesteps, cpu_usage, label="CPU Usage", color="orange")
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("CPU Usage", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

ax2 = ax1.twinx()
ax2.plot(timesteps, num_workers, label="Number of Redis Workers", color="red", linestyle="--")
ax2.set_ylabel("Number of Redis Workers", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("CPU Usage Over Time")
plt.legend(loc="upper left")

# Plot Memory Usage Over Time (unchanged)
plt.subplot(2, 2, 3)
ax1 = plt.gca()
ax1.plot(timesteps, memory_usage, label="Memory Usage", color="green")
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Memory Usage", color="green")
ax1.tick_params(axis="y", labelcolor="green")

ax2 = ax1.twinx()
ax2.plot(timesteps, num_workers, label="Number of Redis Workers", color="red", linestyle="--")
ax2.set_ylabel("Number of Redis Workers", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Memory Usage Over Time")
plt.legend(loc="upper left")

# Plot Latency Over Time with Noise
plt.subplot(2, 2, 4)
ax1 = plt.gca()
ax1.plot(timesteps, latency_with_noise, label="Latency", color="purple")
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Latency", color="purple")
ax1.tick_params(axis="y", labelcolor="purple")

ax2 = ax1.twinx()
ax2.plot(timesteps, num_workers, label="Number of Redis Workers", color="red", linestyle="--")
ax2.set_ylabel("Number of Redis Workers", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Latency Over Time")
plt.legend(loc="upper left")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("original_plot_with_noise.png", dpi=300)
plt.show()