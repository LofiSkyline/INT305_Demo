import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm  # 用于显示进度条

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(train_loader, desc="Training", leave=False)  # TQDM进度条
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})  # 在进度条上显示当前loss
    end_time = time.time()
    avg_time = end_time - start_time
    return total_loss / len(train_loader), avg_time

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy

def main():
    # Define transformations for MNIST
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize MNIST images to 224x224 for ShuffleNetV2
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0.5, std 0.5
    ])

    # Create train and test datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load ShuffleNetV2 model
    model = models.shufflenet_v2_x0_5(pretrained=True)

    # Modify input layer for single-channel input
    model.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)

    # Modify the fully connected layer for MNIST (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Freeze weights in intermediate layers, but not biases or the fc layer
    for name, param in model.named_parameters():
        if 'weight' in name and 'fc' not in name:
            param.requires_grad = False

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training loop
    num_epochs = 5
    epoch_times = []

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss, train_time = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        
        epoch_times.append(train_time)
        
        print(f"Train Loss: {train_loss:.4f}, Train Time: {train_time:.2f}s")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%\n")

    # Average training time per epoch
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Average Training Time per Epoch: {avg_epoch_time:.2f}s")

if __name__ == "__main__":
    main()
