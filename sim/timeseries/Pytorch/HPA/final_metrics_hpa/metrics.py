import matplotlib.pyplot as plt

# Example data for HPA at different steps (adjust as needed)
labels = ["HPA 1 steps", "HPA 2 steps", "HPA 5 steps", "HPA 10 steps", "HPA 20 steps"]
# Hypothetical rewards (negative values, in the range shown in your figure)
rewards = [-1.05e6, -1.02e6, -1.01e6, -1.0e6, -0.95e6]

# Create the figure and bar plot
plt.figure(figsize=(6, 4))
plt.bar(labels, rewards, color='blue')

# Label the axes and set a title
plt.xlabel("Training Steps")
plt.ylabel("Total Accumulated Reward")
plt.title("Accumulated Rewards for HPA Across Different Steps")

# Rotate x-axis labels to avoid overlapping
plt.xticks(rotation=45, ha='right')

# Ensure labels and ticks are fully visible
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("rotated_labels_hpa.png", format="png")

# (Optional) Display the plot on screen
plt.show()
