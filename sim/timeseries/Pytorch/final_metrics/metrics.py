import matplotlib.pyplot as plt

# Approximate data from the screenshot:
#   Model1 ~ 250k
#   Model2 ~ 200k
#   Model5 ~ 180k
#   Model10 ~ 300k (highest)
#   Model20 ~ 130k
labels = ["Model1", "Model2", "Model5", "Model10", "Model20"]
rewards = [250000, 200000, 180000, 300000, 130000]

# Create the figure and bar plot
plt.figure(figsize=(6, 4))
plt.bar(labels, rewards, color='blue')

# Label the axes and set a title
plt.xlabel("Training Steps")
plt.ylabel("Total Accumulated Reward")
plt.title("Accumulated Rewards for Models Trained with Different Steps")

# Rotate x-axis labels to avoid overlapping
plt.xticks(rotation=45, ha='right')

# Make sure labels and ticks are fully visible
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("rotated_labels_models_corrected.png", format="png")

# (Optional) Display the plot on screen
plt.show()
