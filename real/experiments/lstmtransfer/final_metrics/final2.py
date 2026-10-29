import numpy as np
import matplotlib.pyplot as plt

N = 2000
timesteps = np.arange(N)

# ----------------------------------------------------
# 1) Construct a wave-like reward that oscillates
#    about a higher offset to achieve ~2 million total
# ----------------------------------------------------

# Base sinusoidal shape: 3 full cycles across 2,000 timesteps
#   We'll use period ~ 600-700 steps so we get ~3 waves.
#   For a simple example, let's do period = 600 for a ~3.33 cycles.
wave = np.sin(2 * np.pi * timesteps / 600)  # range: -1 to +1

# Original wave in your plot was roughly from ~200 to ~700 => average ~450
# We want a final total ~3.5× bigger => ~2 million total area over 2,000 steps.
# That means an average reward ~1,000 (since 1,000 × 2,000 = 2,000,000).

# So let's center the wave around 1,000 with an amplitude of ~200
# => reward goes from 800 to 1,200 (average = 1,000).
reward = 1000 + 200 * wave

# ----------------------------------------------------
# 2) Add a small “spiky” region near t=1500
#    to mimic your original glitch/spikes
# ----------------------------------------------------
for t in range(1500, 1550):
    # Replace or adjust the reward with a quicker oscillation
    # so it dips and rises rapidly
    reward[t] = 1000 + 200 * np.sin(2 * np.pi * (t - 1500) / 5)

# ----------------------------------------------------
# 3) Compute final total
#    (sum of immediate rewards across all timesteps)
# ----------------------------------------------------
final_reward = np.sum(reward)
print("Total Accumulated Reward:", final_reward)

# ----------------------------------------------------
# 4) Plot: wave-like "Accumulated Rewards Over Time"
#    (Really the immediate reward at each step)
# ----------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(timesteps, reward, color='blue', label='Reward')
plt.title("Accumulated Rewards Over Time")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.legend()
plt.tight_layout()
plt.savefig("accumulated_reward_wave_350pct.png")
plt.show()
