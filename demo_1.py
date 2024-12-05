from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time



# # Adjust the model to get a higher performance
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 8, 3, 1)
#         self.bn1 = nn.BatchNorm2d(8)  # Batch Normalization
#         self.conv2 = nn.Conv2d(8, 16, 3, 1)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(2304, 64)
#         self.fc2 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)  # Apply Batch Normalization
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
    
    
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        
          

        self.fcs = nn.Sequential(
            nn.Linear(2304, 1152),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1152, 576),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(576, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x



def train(args, model, device, train_loader, optimizer, epoch, train_loss_history, train_accuracy_history):
    model.train()
    epoch_loss = 0  # Cumulative loss
    total_batches = 0
    correct = 0  # Total correct predictions
    total_samples = 0  # Total samples processed

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Record loss
        epoch_loss += loss.item()
        total_batches += 1

        # Calculate accuracy for this batch
        pred = output.argmax(dim=1, keepdim=True)  # Predictions
        correct += pred.eq(target.view_as(pred)).sum().item()  # Correct predictions
        total_samples += len(target)  # Total samples in this batch

        # Log progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    # Record average loss and accuracy for the epoch
    average_loss = epoch_loss / total_batches
    accuracy = 100. * correct / total_samples
    train_loss_history.append(average_loss)
    train_accuracy_history.append((epoch, accuracy))  # Record accuracy
    print(f"Epoch {epoch}: Training Accuracy = {accuracy:.2f}%, Average Loss = {average_loss:.2f}")


def visualize_misclassified_images(model, device, test_loader):
    """
    Visualize the first 50 misclassified test images along with their ground truth and predicted labels.
    """
    model.eval()  # Set model to evaluation mode
    misclassified_images = []
    ground_truth = []
    predictions = []

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # Get predictions
            
            # Collect misclassified samples
            for i in range(len(data)):
                if pred[i].item() != target[i].item():  # If prediction is wrong
                    if len(misclassified_images) < 20:
                        misclassified_images.append(data[i].cpu())  # Move image to CPU
                        ground_truth.append(target[i].item())  # Add ground truth label
                        predictions.append(pred[i].item())  # Add predicted label
                    else:
                        break
            if len(misclassified_images) >= 20:
                break

    # Plot the images in a grid (e.g., 5x10)
    num_images = len(misclassified_images)
    rows = num_images // 10 + (1 if num_images % 10 != 0 else 0)  # Determine number of rows
    fig, axes = plt.subplots(rows, 10, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes[:num_images]):
        ax.imshow(misclassified_images[i].squeeze(), cmap='gray')  # Squeeze to remove extra dimensions
        ax.set_title(f"GT: {ground_truth[i]}\nPred: {predictions[i]}")
        ax.axis('off')

    # Turn off unused axes
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def test(model, device, test_loader, epoch, test_accuracy_history, digit_accuracy_history):
    """
    Test the model and track accuracy for each digit (0-9).
    """
    model.eval()
    test_loss = 0
    correct = 0
    digit_correct = [0] * 10  # Correct predictions for each digit
    digit_total = [0] * 10    # Total occurrences for each digit

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            # Update total and correct counts for each digit
            for i in range(10):
                digit_mask = (target == i)  # Mask for current digit
                digit_total[i] += digit_mask.sum().item()
                digit_correct[i] += pred[digit_mask].eq(target[digit_mask].view_as(pred[digit_mask])).sum().item()

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # Calculate accuracy for each digit
    digit_accuracies = [
        (100. * digit_correct[i] / digit_total[i]) if digit_total[i] > 0 else 0.0
        for i in range(10)
    ]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    for i, acc in enumerate(digit_accuracies):
        print(f"Digit {i}: Accuracy: {acc:.2f}%")

    # Save epoch-wise accuracy history
    test_accuracy_history.append((epoch, accuracy, correct))
    digit_accuracy_history.append((epoch, digit_accuracies))  # Save accuracy for each digit

import matplotlib.pyplot as plt

def plot_digit_accuracies(digit_accuracy_history):
    """
    Plot line charts for each digit's accuracy over epochs.
    """
    epochs = [record[0] for record in digit_accuracy_history]
    accuracies = [record[1] for record in digit_accuracy_history]

    # Transpose the list of accuracies to group by digits
    digit_accuracies = list(zip(*accuracies))

    plt.figure(figsize=(12, 8))
    for digit, acc in enumerate(digit_accuracies):
        plt.plot(epochs, acc, label=f"Digit {digit}", marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy of Each Digit Over Epochs")
    plt.legend(title="Digits", loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_mean_std(dataset):
    """
    计算数据集的均值和标准差
    """
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for data, _ in data_loader:
        # 数据是 [batch_size, channels, height, width]
        batch_samples = data.size(0)  # 当前 batch 的样本数
        total_samples += batch_samples
        data = data.view(batch_samples, -1)  # 展平为 [batch_size, num_pixels]
        mean += data.mean(1).sum().item()  # 按像素求均值，再求总和
        std += data.std(1).sum().item()  # 按像素求标准差，再求总和

    mean /= total_samples
    std /= total_samples
    return mean, std

import plotly.graph_objects as go

import plotly.graph_objects as go

def plot_loss(train_loss_history):
    # 绘制训练损失折线图
    fig = go.Figure()
    epochs = list(range(1, len(train_loss_history) + 1))  # 生成 epoch 列表
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss_history,
        mode='lines+markers',
        name='Average Loss',
        line=dict(color='blue', width=2),
        marker=dict(size=6, color='blue')
    ))
    fig.update_layout(
        title='Average Training Loss Per Epoch',
        xaxis_title='Epoch',
        yaxis_title='Average Loss',
        template='plotly_white'
    )
    fig.show()


import plotly.graph_objects as go


def plot_accuracy(train_accuracy_history, test_accuracy_history):
    """
    Plot the training and testing accuracy over epochs on the same graph.
    """
    # Extract epoch and accuracy for training
    train_epochs = [x[0] for x in train_accuracy_history]
    train_accuracies = [x[1] for x in train_accuracy_history]

    # Extract epoch and accuracy for testing
    test_epochs = [x[0] for x in test_accuracy_history]
    test_accuracies = [x[1] for x in test_accuracy_history]

    fig = go.Figure()

    # Add training accuracy line
    fig.add_trace(go.Scatter(
        x=train_epochs,
        y=train_accuracies,
        mode='lines+markers',
        name='Training Accuracy',
        hoverinfo='text',
        text=[f'Epoch: {e}, Training Accuracy: {a:.2f}%' for e, a in zip(train_epochs, train_accuracies)]
    ))

    # Add testing accuracy line
    fig.add_trace(go.Scatter(
        x=test_epochs,
        y=test_accuracies,
        mode='lines+markers',
        name='Testing Accuracy',
        hoverinfo='text',
        text=[f'Epoch: {e}, Testing Accuracy: {a:.2f}%' for e, a in zip(test_epochs, test_accuracies)]
    ))

    # Update layout
    fig.update_layout(
        title='Training and Testing Accuracy Over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Accuracy (%)',
        template='plotly_white',
        legend_title="Legend"
    )

    fig.show()


import numpy as np

def test_with_confusion_matrix(model, device, test_loader):
    """
    Test the model and return predictions and true labels for the confusion matrix.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=False)  # Get predictions
            all_preds.extend(pred.cpu().numpy())  # Store predictions
            all_targets.extend(target.cpu().numpy())  # Store true labels

    return np.array(all_preds), np.array(all_targets)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(preds, targets, class_names):
    """
    Generate and display a confusion matrix.
    """
    # Compute confusion matrix
    cm = confusion_matrix(targets, preds, labels=range(len(class_names)))

    # Display confusion matrix as a heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap="viridis", values_format='d')
    plt.title("Confusion Matrix")
    plt.show()



def main():
    train_accuracy_history = []  # Track epoch-wise training accuracy

    digit_accuracy_history = []

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Visualize image grid after each epoch')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # DataLoader configurations
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Load MNIST data for normalization calculation
    raw_transform = transforms.Compose([transforms.ToTensor()])
    raw_dataset = datasets.MNIST('./data', train=True, download=True, transform=raw_transform)

    # Compute mean and standard deviation
    mean, std = compute_mean_std(raw_dataset)
    print(f"Computed Mean: {mean}, Computed Std: {std}")

    # Transform with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    # Create train and test datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Initialize model, optimizer, and scheduler
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Initialize history tracking
    train_loss_history = []  # Track batch-wise training loss
    test_accuracy_history = []  # Track epoch-wise test accuracy
    class_names = [str(i) for i in range(10)]
    total_training_time = 0  # Variable to track total training time
    # Training and testing loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        start_time = time.time()  # Start timing
        train(args, model, device, train_loader, optimizer, epoch, train_loss_history, train_accuracy_history)
        end_time = time.time()  # End timing
        epoch_training_time = end_time - start_time  # Calculate epoch time
        total_training_time += epoch_training_time  # Accumulate total training time
        test(model, device, test_loader, epoch, test_accuracy_history, digit_accuracy_history)  # Test the model
        scheduler.step()  # Adjust learning rate
    # # Generate confusion matrix after all epochs
    # preds, targets = test_with_confusion_matrix(model, device, test_loader)
    # plot_confusion_matrix(preds, targets, class_names)
    # After training and testing
    plot_digit_accuracies(digit_accuracy_history)


    # print("\nVisualizing Misclassified Images...")
    # visualize_misclassified_images(model, device, test_loader)
        # Calculate average training time per epoch
    average_training_time = total_training_time / args.epochs
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
    print(f"Average Training Time per Epoch: {average_training_time:.2f} seconds")

    # Plot results
    plot_loss(train_loss_history)
    plot_accuracy(train_accuracy_history, test_accuracy_history)
 # Plot test accuracy

    # Save the model (if enabled)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Model saved to mnist_cnn.pt")


if __name__ == '__main__':
    main()