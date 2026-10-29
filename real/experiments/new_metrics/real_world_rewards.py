import matplotlib.pyplot as plt

# Approaches and corresponding total rewards
approaches = ["RS (LSTM)", "RS Transfer (LSTM_transfer)"]
rewards = [56000, 56000 * 5.5]  # 56,000 vs. 56,000 × 5.5 = 308,000

# Create a bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(approaches, rewards, color=["blue", "orange"])

# Add a title and label the y-axis
plt.title("Real-World Experiments with Redis & K8s (2000 Timesteps)")
plt.ylabel("Total Reward")

# Optionally, add numeric labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,  # x-position (center of bar)
        height + 5000,                       # y-position slightly above bar
        f"{int(height):,}",                  # format as an integer with commas
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

# Tidy layout, save, and show
plt.tight_layout()
plt.savefig("real_world_experiments_rewards.png", dpi=150)
plt.show()