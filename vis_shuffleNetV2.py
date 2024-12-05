import matplotlib.pyplot as plt
import numpy as np

# Data
epochs = [1, 2, 3, 4, 5]
train_times = [313.41, 293.73, 294.59, 286.56, 292.14]
test_accuracies = [81.51, 88.61, 91.43, 92.73, 93.96]

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot bar chart for training time
ax1.bar(epochs, train_times, color='lightblue', alpha=0.7, label='Training Time (s)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Time (s)')
ax1.set_title('ShuffleNet Training Time and Test Accuracy')
ax1.legend(loc='upper left')

# Create a second y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(epochs, test_accuracies, color='orange', marker='o', label='Test Accuracy (%)')
ax2.set_ylabel('Test Accuracy (%)')
ax2.legend(loc='upper right')

# Show grid and plot
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
