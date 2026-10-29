import matplotlib.pyplot as plt

# Example data (adjust as needed)
models = ["Model1", "Model2", "Model5", "Model10", "Model20"]
rewards = [280000, 270000, 230000, 200000, 100000]

# Create the figure and bar plot
plt.figure(figsize=(6, 4))
plt.bar(models, rewards, color='blue')

# Label the axes and set a title
plt.xlabel("Models")
plt.ylabel("Total Accumulated Reward")
plt.title("Accumulated Rewards for Models Trained with Different Steps")

# Rotate x-axis labels to avoid overlapping
plt.xticks(rotation=45, ha='right')

# Make sure labels and ticks are fully visible
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("rotated_labels.png", format="png")

# (Optional) Display the plot on screen
plt.show()
