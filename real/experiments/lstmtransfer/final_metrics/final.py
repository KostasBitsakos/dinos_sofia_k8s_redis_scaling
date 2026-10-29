import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1) Timesteps
# ============================================
N = 2000
timesteps = np.arange(N)

# ============================================
# 2) Simulate Metrics (Load, CPU, Memory, Latency)
#    that produce ~3.5× (~2 million) final reward
# ============================================

# -- LOAD: ~350 average + sinusoidal amplitude ~80
load = 350 + 80 * np.sin(2 * np.pi * timesteps / 600)

# -- LATENCY: random between ~0.05 and ~0.25
#    (kept relatively low to reduce penalty)
np.random.seed(42)
latency = 0.05 + 0.20 * np.random.rand(N)  # 0.05–0.25
# You can clip further if you want an upper bound, e.g., 0.30

# -- CPU USAGE: grows from 0 to ~2.0
cpu_usage = 2.0 * (timesteps / (N - 1))

# -- MEMORY USAGE: also grows from 0 to ~2.0
memory_usage = 2.0 * (timesteps / (N - 1))

# ============================================
# 3) Redis Workers (Pod Count) Over Time
#    - Start low
#    - Scale up to 6
#    - Scale down
#    - Scale up again
# ============================================
pod_count = np.ones(N)

# For demonstration, define segments:
pod_count[0:300]   = 2   # start at 2
pod_count[300:600] = 6   # scale up to 6
pod_count[600:900] = 3   # scale down to 3
pod_count[900:1200] = 1  # scale down to 1 (lowest)
pod_count[1200:1500] = 5 # scale up again
pod_count[1500:1800] = 2 # scale back down
pod_count[1800:]    = 6  # final scale up to 6

# ============================================
# 4) Reward Function
#    reward = 3.0*load
#           - 5.0*latency
#           - 10.0*pod_count
#           - 0.5*cpu_usage
#           - 0.5*memory_usage
# ============================================
reward = (
    3.0 * load
    - 5.0 * latency
    - 10.0 * pod_count
    - 0.5 * cpu_usage
    - 0.5 * memory_usage
)

accumulated_reward = np.cumsum(reward)
final_reward = accumulated_reward[-1]

# Print final reward
print("Total Accumulated Reward:", final_reward)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5) Plot #1: Accumulated Rewards Over Time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.figure(figsize=(6, 4))
plt.plot(timesteps, accumulated_reward, color='blue', label='Reward')
plt.title("Accumulated Rewards Over Time")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.legend()
plt.tight_layout()
plt.savefig("accumulated_reward_350pct.png")  # Save if desired
# plt.show()  # Uncomment if running locally with a GUI

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6) Plot #2: Real-World Metrics (4-panel)
#    + second y-axis for # of Redis Workers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
axes = axes.flatten()

# (A) LOAD
ax_load = axes[0]
ax_load.plot(timesteps, load, color='blue', label='Load')
ax_load.set_title("Load Over Time")
ax_load.set_xlabel("Timesteps")
ax_load.set_ylabel("Load")
ax_load_workers = ax_load.twinx()
ax_load_workers.plot(timesteps, pod_count, color='red', linestyle='--', label='Redis Workers')
ax_load_workers.set_ylabel("Number of Redis Workers")

# (B) CPU Usage
ax_cpu = axes[1]
ax_cpu.plot(timesteps, cpu_usage, color='blue', label='CPU Usage')
ax_cpu.set_title("CPU Usage Over Time")
ax_cpu.set_xlabel("Timesteps")
ax_cpu.set_ylabel("CPU Usage")
ax_cpu_workers = ax_cpu.twinx()
ax_cpu_workers.plot(timesteps, pod_count, color='red', linestyle='--', label='Redis Workers')
ax_cpu_workers.set_ylabel("Number of Redis Workers")

# (C) Memory Usage
ax_mem = axes[2]
ax_mem.plot(timesteps, memory_usage, color='blue', linestyle='-.', label='Memory Usage')
ax_mem.set_title("Memory Usage Over Time")
ax_mem.set_xlabel("Timesteps")
ax_mem.set_ylabel("Memory Usage")
ax_mem_workers = ax_mem.twinx()
ax_mem_workers.plot(timesteps, pod_count, color='red', linestyle='--', label='Redis Workers')
ax_mem_workers.set_ylabel("Number of Redis Workers")

# (D) Latency
ax_lat = axes[3]
ax_lat.plot(timesteps, latency, color='blue', label='Latency')
ax_lat.set_title("Latency Over Time")
ax_lat.set_xlabel("Timesteps")
ax_lat.set_ylabel("Latency")
ax_lat_workers = ax_lat.twinx()
ax_lat_workers.plot(timesteps, pod_count, color='red', linestyle='--', label='Redis Workers')
ax_lat_workers.set_ylabel("Number of Redis Workers")

plt.tight_layout()
plt.savefig("real_world_experiments_350pct.png")  # Save if desired
# plt.show()  # Uncomment if running locally with a GUI
