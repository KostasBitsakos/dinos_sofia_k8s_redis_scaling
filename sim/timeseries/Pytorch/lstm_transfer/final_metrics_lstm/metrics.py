import matplotlib.pyplot as plt

# Approximate data from the screenshot:
#   Model1 ~ 3.6e6
#   Model2 ~ 3.7e6
#   Model5 ~ 3.8e6
#   Model10 ~ 3.4e6
#   Model20 ~ 3.0e6
labels = ["Model1", "Model2", "Model5", "Model10", "Model20"]
rewards = [3.6e6, 3.7e6, 3.8e6, 3.4e6, 3.0e6]

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
plt.savefig("rotated_labels_models_million.png", format="png")

# (Optional) Display the plot on screen
plt.show()
