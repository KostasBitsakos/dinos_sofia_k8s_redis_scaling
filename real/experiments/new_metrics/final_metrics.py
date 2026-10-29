import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1) Define time range
# --------------------------------------------------
timesteps = np.arange(0, 2001)  # 0..2000 inclusive

# --------------------------------------------------
# 2) Define Redis workers (piecewise intervals)
# --------------------------------------------------
redis_workers = np.zeros_like(timesteps, dtype=float)
for i, t in enumerate(timesteps):
    if t < 25:
        redis_workers[i] = 2
    elif t < 600:
        redis_workers[i] = 6
    elif t < 900:
        redis_workers[i] = 3
    elif t < 1200:
        redis_workers[i] = 1
    elif t < 1600:
        redis_workers[i] = 5
    elif t < 1800:
        redis_workers[i] = 2
    else:
        redis_workers[i] = 6

# --------------------------------------------------
# 3) Create Load signal (sinusoidal + stronger noise)
# --------------------------------------------------
# Center ~400, amplitude ~100, ~4 cycles over 2000 steps
# Then add noise with amplitude ~60 to increase randomness
base_load = 400 + 100 * np.sin(4 * np.pi * timesteps / 2000.0)
noise_load = 60 * np.random.randn(len(timesteps))  # bigger noise
load_noisy = base_load + noise_load

# --------------------------------------------------
# 4) Create CPU Usage (unchanged linear ramp)
# --------------------------------------------------
cpu_usage = np.linspace(0, 100, len(timesteps))

# --------------------------------------------------
# 5) Create Memory Usage (unchanged step function)
# --------------------------------------------------
# Example: 1 from 0..500, 1.5 from 500..1000, 1.8 from 1000..1500, 2.0 from 1500..2000
memory_usage = np.piecewise(
    timesteps,
    [timesteps < 500, (timesteps >= 500) & (timesteps < 1000),
     (timesteps >= 1000) & (timesteps < 1500), timesteps >= 1500],
    [1.0, 1.5, 1.8, 2.0]
)

# --------------------------------------------------
# 6) Create Latency signal (sinusoidal + stronger noise)
# --------------------------------------------------
# Center ~3, amplitude ~3, ~4 cycles over 2000 steps
# Then add noise with amplitude ~3
base_latency = 3 + 3 * np.sin(4 * np.pi * timesteps / 2000.0)
noise_latency = 3 * np.random.randn(len(timesteps))
latency_noisy = base_latency + noise_latency

# --------------------------------------------------
# 7) Plot the 2×2 subplots with twin y-axes
# --------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 6))

# -------------------- Subplot (0,0): Load --------------------
ax_load = axs[0, 0]
ax_load.plot(timesteps, load_noisy, 'b-', label='Load')
ax_load.set_title("Load Over Time")
ax_load.set_xlabel("Timesteps")
ax_load.set_ylabel("Load")

# Twin y-axis for Redis workers (red dashed)
ax_load_r = ax_load.twinx()
ax_load_r.plot(timesteps, redis_workers, 'r--', label='Number of Redis Workers')
ax_load_r.set_ylabel("Number of Redis Workers")

# -------------------- Subplot (0,1): CPU ---------------------
ax_cpu = axs[0, 1]
ax_cpu.plot(timesteps, cpu_usage, 'b-', label='CPU Usage')
ax_cpu.set_title("CPU Usage Over Time")
ax_cpu.set_xlabel("Timesteps")
ax_cpu.set_ylabel("CPU Usage")

# Twin y-axis for Redis workers
ax_cpu_r = ax_cpu.twinx()
ax_cpu_r.plot(timesteps, redis_workers, 'r--', label='Number of Redis Workers')
ax_cpu_r.set_ylabel("Number of Redis Workers")

# -------------------- Subplot (1,0): Memory -------------------
ax_mem = axs[1, 0]
ax_mem.plot(timesteps, memory_usage, 'b-', label='Memory Usage')
ax_mem.set_title("Memory Usage Over Time")
ax_mem.set_xlabel("Timesteps")
ax_mem.set_ylabel("Memory Usage")

# Twin y-axis for Redis workers
ax_mem_r = ax_mem.twinx()
ax_mem_r.plot(timesteps, redis_workers, 'r--', label='Number of Redis Workers')
ax_mem_r.set_ylabel("Number of Redis Workers")

# -------------------- Subplot (1,1): Latency ------------------
ax_lat = axs[1, 1]
ax_lat.plot(timesteps, latency_noisy, 'b-', label='Latency')
ax_lat.set_title("Latency Over Time")
ax_lat.set_xlabel("Timesteps")
ax_lat.set_ylabel("Latency")

# Twin y-axis for Redis workers
ax_lat_r = ax_lat.twinx()
ax_lat_r.plot(timesteps, redis_workers, 'r--', label='Number of Redis Workers')
ax_lat_r.set_ylabel("Number of Redis Workers")

# --------------------------------------------------
# Final layout adjustments
# --------------------------------------------------
plt.tight_layout()
plt.savefig("plot_noisy_load_latency.png", dpi=150)
plt.show()
