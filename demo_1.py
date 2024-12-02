from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


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

def train(args, model, device, train_loader, optimizer, epoch, train_loss_history):
    model.train()
    epoch_loss = 0  # 用于记录当前 epoch 的累计损失
    total_batches = 0  # 用于记录当前 epoch 的 batch 数量

    for batch_idx, (data, target) in enumerate(train_loader):
        # 模型训练部分
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # 累加当前 batch 的损失
        epoch_loss += loss.item()
        total_batches += 1

        # 打印日志
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    # 计算并记录当前 epoch 的平均损失
    average_loss = epoch_loss / total_batches
    train_loss_history.append(average_loss)
    print(f"Epoch {epoch}: Average Training Loss = {average_loss:.4f}")


def visualize_images(data_loader):
    pic = None  # 初始化拼接图片容器
    row_images = None  # 每一行拼接的图片
    col_count = 0  # 列计数器
    row_count = 0  # 行计数器

    for batch_idx, (data, target) in enumerate(data_loader):
        # 拼接前 100 张图片
        if row_count < 10:
            for i in range(data.size(0)):  # 遍历当前批次所有图片
                if col_count == 0:
                    row_images = data[i, 0, :, :]  # 初始化当前行
                else:
                    row_images = torch.cat((row_images, data[i, 0, :, :]), dim=1)  # 横向拼接

                col_count += 1
                if col_count == 10:  # 一行拼接完成
                    col_count = 0
                    if pic is None:
                        pic = row_images  # 初始化拼接的第一行
                    else:
                        pic = torch.cat((pic, row_images), dim=0)  # 垂直拼接到大图
                    row_images = None  # 重置当前行
                    row_count += 1

                if row_count == 10:  # 大图拼接完成
                    break
        if row_count == 10:  # 大图拼接完成
            break

    # 显示拼接的图片
    if pic is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(pic.cpu(), cmap='gray')
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
    else:
        print("Warning: No images were selected for visualization.")




def test(model, device, test_loader, epoch, test_accuracy_history):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    # 保存每个 epoch 的正确率和正确分类数量
    test_accuracy_history.append((epoch, accuracy, correct))


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


def plot_accuracy(test_accuracy_history):
    # 绘制测试正确率折线图
    epochs = [x[0] for x in test_accuracy_history]
    accuracies = [x[1] for x in test_accuracy_history]
    correct_counts = [x[2] for x in test_accuracy_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy',
        hoverinfo='text',
        text=[f'Epoch: {e}, Accuracy: {a:.2f}%, Correct: {c}' for e, a, c in zip(epochs, accuracies, correct_counts)]
    ))
    fig.update_layout(
        title='Test Accuracy Over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Accuracy (%)',
        template='plotly_white'
    )
    fig.show()


def main():
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

    # Training and testing loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train(args, model, device, train_loader, optimizer, epoch, train_loss_history)
        
        if args.visualize:
            visualize_images(train_loader)  # Visualize images (if enabled)
        
        test(model, device, test_loader, epoch, test_accuracy_history)  # Test the model
        scheduler.step()  # Adjust learning rate

    # Plot results
    plot_loss(train_loss_history)
    plot_accuracy(test_accuracy_history)  # Plot test accuracy

    # Save the model (if enabled)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Model saved to mnist_cnn.pt")



if __name__ == '__main__':
    main()