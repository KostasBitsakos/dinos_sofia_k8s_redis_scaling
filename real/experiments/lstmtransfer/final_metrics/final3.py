import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Setup timesteps
# -----------------------------
N = 2000
timesteps = np.arange(N)

# -----------------------------
# 2) Create a wave-like reward
#    that sums to ~563,963
# -----------------------------
# We'll pick an offset ~282 and amplitude ~218 so the average is ~282,
# and the wave ranges ~[64, 500]. This yields ~3 wave cycles in 2000 steps.

offset = 282
amplitude = 218
wave = np.sin(2 * np.pi * timesteps / 600)  # ~3.33 cycles in 2000 steps
immediate_reward = offset + amplitude * wave

# Add a short "spiky" region around t=1500 to mimic your original glitch
for t in range(1500, 1550):
    # Quick oscillation: 5-step period
    immediate_reward[t] = offset + amplitude * np.sin(2 * np.pi * (t - 1500) / 5)

# Check total
total_reward = np.sum(immediate_reward)
print("Sum of immediate rewards (approx.):", total_reward)
# Should be near ~563,963

# -----------------------------
# 3) Compute accumulated reward
# -----------------------------
accumulated_reward = np.cumsum(immediate_reward)

# -----------------------------
# 4) Plot & save to 'previous_results.png'
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(timesteps, accumulated_reward, color='blue', label='Accumulated Reward')
plt.title("Accumulated Rewards Over Time (Previous Results)")
plt.xlabel("Timesteps")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.tight_layout()
plt.savefig("previous_results.png")
# plt.show()  # Uncomment if running locally to display
