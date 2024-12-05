import torch
import torch.nn as nn
import torchvision.models as models

# Define the custom CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.bn1 = nn.BatchNorm2d(8)  # Batch Normalization
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Apply Batch Normalization
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Function to calculate parameters and memory
def calculate_parameters_and_memory(model, input_shape):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights_memory = total_params * 4 / 1024**2  # Memory for weights in MB (float32)

    # Calculate activation memory
    activation_memory = 0
    input_tensor = torch.randn(input_shape)
    
    def hook(module, input, output):
        nonlocal activation_memory
        if isinstance(output, (tuple, list)):
            for out in output:
                activation_memory += out.numel() * 4 / 1024**2  # Memory in MB
        else:
            activation_memory += output.numel() * 4 / 1024**2

    hooks = []
    for layer in model.children():
        hooks.append(layer.register_forward_hook(hook))

    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    total_memory = weights_memory + activation_memory
    return total_params, trainable_params, weights_memory, activation_memory, total_memory

# Define input sizes
input_size_custom = (1, 1, 28, 28)
input_size_shuffle = (1, 1, 28, 28)

# Instantiate models
custom_model = Net()
shufflenet_model = models.shufflenet_v2_x0_5(pretrained=True)
shufflenet_model.eval()

# Modify ShuffleNetV2 for single-channel input
shufflenet_model.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
shufflenet_model.fc = nn.Linear(shufflenet_model.fc.in_features, 10)

# Calculate for Custom CNN
custom_params, custom_trainable_params, custom_weights_memory, custom_activation_memory, custom_total_memory = calculate_parameters_and_memory(custom_model, input_size_custom)

# Calculate for ShuffleNetV2
shuffle_params, shuffle_trainable_params, shuffle_weights_memory, shuffle_activation_memory, shuffle_total_memory = calculate_parameters_and_memory(shufflenet_model, input_size_shuffle)

# Print results
print("Custom CNN:")
print(f"Total Parameters: {custom_params:,}")
print(f"Trainable Parameters: {custom_trainable_params:,}")
print(f"Weights Memory (MB): {custom_weights_memory:.2f}")
print(f"Activation Memory (MB): {custom_activation_memory:.2f}")
print(f"Total Memory (MB): {custom_total_memory:.2f}\n")

print("ShuffleNetV2:")
print(f"Total Parameters: {shuffle_params:,}")
print(f"Trainable Parameters: {shuffle_trainable_params:,}")
print(f"Weights Memory (MB): {shuffle_weights_memory:.2f}")
print(f"Activation Memory (MB): {shuffle_activation_memory:.2f}")
print(f"Total Memory (MB): {shuffle_total_memory:.2f}")

