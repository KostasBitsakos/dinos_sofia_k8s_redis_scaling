import matplotlib.pyplot as plt

# Common training steps for illustration
steps = [1, 2, 5, 10, 20]

# --------------------------------------------------------------------
# 1) HPA Data
labels_hpa = [f"HPA with {s} steps" for s in steps]
rewards_hpa = [-1.05e6, -1.02e6, -1.01e6, -1.00e6, -0.95e6]  # Approx

# 2) DERP Data (dense)
labels_derp = [f"DERP with {s} steps" for s in steps]
rewards_derp = [200000, 180000, 240000, 210000, 220000]     # Approx

# 3) RS Data (LSTM)
labels_rs = [f"RS with {s} steps" for s in steps]
rewards_rs = [240000, 260000, 280000, 300000, 310000]       # Approx

# 4) LSTM_transfer Data
labels_lstm_transfer = [f"RS transfer with {s} steps" for s in steps]
rewards_lstm_transfer = [3.55e6, 3.65e6, 3.80e6, 3.30e6, 3.05e6]  # Approx

# --------------------------------------------------------------------
# Create one figure with 2×2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# ------------------ Subplot (0,0): HPA ------------------
axes[0, 0].bar(labels_hpa, rewards_hpa, color='red')
axes[0, 0].set_title("HPA")
axes[0, 0].set_ylabel("Total Accumulated Reward")
axes[0, 0].tick_params(axis='x', rotation=45)

# ------------------ Subplot (0,1): DERP ------------------
axes[0, 1].bar(labels_derp, rewards_derp, color='green')
axes[0, 1].set_title("DERP (Dense)")
axes[0, 1].tick_params(axis='x', rotation=45)

# ------------------ Subplot (1,0): RS ------------------
axes[1, 0].bar(labels_rs, rewards_rs, color='blue')
axes[1, 0].set_title("RS (LSTM)")
axes[1, 0].set_ylabel("Total Accumulated Reward")
axes[1, 0].tick_params(axis='x', rotation=45)

# ------------------ Subplot (1,1): LSTM_transfer ------------------
axes[1, 1].bar(labels_lstm_transfer, rewards_lstm_transfer, color='purple')
axes[1, 1].set_title("RS Transfer (LSTM_transfer)")
axes[1, 1].tick_params(axis='x', rotation=45)

# Automatically adjust spacing so labels/titles fit well
plt.tight_layout()

# Save the entire figure to a PNG file
plt.savefig("all_four_algorithms.png", format="png")

# (Optional) Display the plot on screen
plt.show()
