import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time


# Define the training function
def train(model, device, train_loader, optimizer, epoch, scheduler):
    model.train()
    epoch_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        epoch_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)

        if batch_idx % 10 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

    scheduler.step()
    accuracy = 100. * correct / total_samples
    print(f"Epoch {epoch}: Training Loss = {epoch_loss / total_samples:.4f}, Accuracy = {accuracy:.2f}%")
    return epoch_loss / total_samples, accuracy


# Define the testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.2f}%)")
    return test_loss, accuracy


def main():
    # Hyperparameters
    batch_size = 64
    test_batch_size = 1000
    epochs = 1
    learning_rate = 0.001
    gamma = 0.7
    seed = 1

    # Device configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Seed for reproducibility
    torch.manual_seed(seed)

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)

    # Modify the first convolutional layer to accept 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Freeze all layers except fc layer and the first convolutional layer (conv1)
    for name, param in model.named_parameters():
        if "fc" in name:  # Fully connected layer remains trainable
            param.requires_grad = True
        elif "conv1" in name:  # First layer remains trainable
            param.requires_grad = True
        else:
            if "weight" in name:  # Freeze weights
                param.requires_grad = False
            elif "bias" in name:  # Keep biases trainable
                param.requires_grad = True
    # Modify the final fully connected layer for MNIST (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)

    model = model.to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Training and testing loop
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    total_training_time = 0  # Variable to track total training time

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        start_time = time.time()
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch, scheduler)
        end_time = time.time()  # End timing
        epoch_training_time = end_time - start_time  # Calculate epoch time
        total_training_time += epoch_training_time  # Accumulate total training time
        test_loss, test_accuracy = test(model, device, test_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    average_training_time = total_training_time / epochs
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
    print(f"Average Training Time per Epoch: {average_training_time:.2f} seconds")
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
