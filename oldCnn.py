from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import csv
import os



# Adjust the model to get a higher performance
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
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
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
        test_loss, correct, len(test_loader.dataset),accuracy))

    # for i, acc in enumerate(digit_accuracies):
    #     print(f"Digit {i}: Accuracy: {acc:.2f}%")

    # Save epoch-wise accuracy history
    test_accuracy_history.append((epoch, accuracy, correct))
    digit_accuracy_history.append((epoch, digit_accuracies))  # Save accuracy for each digit




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

def write_to_csv(log_file, data):
    """
    Append data to a CSV file. Create the file with headers if it doesn't exist.
    :param log_file: Path to the CSV file
    :param data: Dictionary containing the data to write
    """
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)




def main():
    train_accuracy_history = []  # Track epoch-wise training accuracy
    digit_accuracy_history = []
    train_loss_history = []  # Track batch-wise training loss
    test_accuracy_history = []  # Track epoch-wise test accuracy
    total_training_time = 0  # Variable to track total training time

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
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-file', type=str, default='training_log.csv',
                        help='path to the log file (default: training_log.csv)')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # DataLoader configurations
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}


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

    # Create a CSV log file
    log_file = args.log_file

    # Training and testing loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        start_time = time.time()  # Start timing
        train(args, model, device, train_loader, optimizer, epoch, train_loss_history, train_accuracy_history)
        end_time = time.time()  # End timing
        epoch_training_time = end_time - start_time  # Calculate epoch time
        print(epoch_training_time)
        total_training_time += epoch_training_time  # Accumulate total training time
        test(model, device, test_loader, epoch, test_accuracy_history, digit_accuracy_history)  # Test the model
        # Retrieve the last recorded values for train loss and accuracy
        train_loss = train_loss_history[-1] if train_loss_history else None
        train_accuracy = train_accuracy_history[-1][1] if train_accuracy_history else None
        test_accuracy = test_accuracy_history[-1][1] if test_accuracy_history else None
        # Log epoch results
        log_data = {
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "epoch_time": epoch_training_time
        }
        write_to_csv(log_file, log_data)

    average_training_time = total_training_time / args.epochs
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
    print(f"Average Training Time per Epoch: {average_training_time:.2f} seconds")






if __name__ == '__main__':
    main()