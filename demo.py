from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR,ExponentialLR,CosineAnnealingLR
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import itertools

# Adjust the model to get a higher performance
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()# batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        #in_channels=1,out_channels=8,kernel_size=3,stride=1
        self.conv1 = nn.Conv2d(1, 8, 3, 1)# 输出数据大小变为28-3+1=26.所以batchx1x28x28 -> batchx8x26x26   
        self.conv2 = nn.Conv2d(8, 16, 3, 1)#第一个卷积层的输出通道数等于第二个卷积层的输入通道数。
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)#（激活函数ReLU不改变形状）
        x = self.conv2(x)  
        x = F.relu(x)#（激活函数ReLU不改变形状）
        x = F.max_pool2d(x, 2)# batch*8x26x26  -> batch*8*13*13（2*2的池化层会减半，步长为2）此时输出数据大小变为13-3+2=12（卷积核大小为3），所以 batchx8x13x13 -> batchx16x12x12。
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
train_accs = []
def train(args, model, device, train_loader, optimizer, epoch):# 定义每个epoch的训练细节
    total = 0 # 总样本数量
    running_loss = 0 # 记录当前的损失值
    accuracy = 0 #记录每次epoch的accuracy
    model.train() # 设置为trainning模式
    plt.figure()
    pic = None

    for batch_idx, (data, target) in enumerate(train_loader): 
        if batch_idx in (1,2,3,4,5): # 图像拼接 针对前五个图像进行拼接
            if batch_idx == 1:
                pic = data[0,0,:,:] # 第一个样本， 第一个通到，整个宽高所有的图像
            else:
                pic = torch.cat((pic,data[0,0,:,:]),dim=1) # 按照高度进行拼接
        data, target = data.to(device), target.to(device) # 部署标签和模型
        optimizer.zero_grad()# 优化器梯度初始化为零
        # forword + backward + update
        output = model(data)# 把数据输入网络并得到输出，即进行前向传播
        loss = F.cross_entropy(output, target) # 计算损失函数 
        _, predicted = torch.max(output.data, dim=1)
        if batch_idx == 1:
            images = utils.make_grid(data,padding = 0)
            image_show(images)
            print('GroundTruth: ', ' '.join('%d' % target[j] for j in range(64)))
            print('Predicted: ', ' '.join('%d' % predicted[j] for j in range(64)))
        accuracy += (predicted == target).sum().item()
        total += target.size(0)

        # Calculate gradients
        loss.backward()# 反向传播梯度
        
        # Optimize the parameters according to the calculated gradients
        optimizer.step()# 结束一次前传+反传之后，更新优化器参数
        
        running_loss += loss.item()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    train_accs.append(100 * accuracy/total)    
    plt.imshow(pic.cpu(), cmap='gray')
    plt.show()

def image_show(images):
    images = images.numpy()
    images = images.transpose((1, 2, 0))
    print(images.shape)
    plt.imshow(images)
    plt.show()

test_accs = [] 
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss= 0
    conf_matrix = torch.zeros(10, 10)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            #conf_matrix = confusion_matrix(output, target, conf_matrix)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            _, predicted = torch.max(output.data, dim=1)  # get the index of the max log-probability

            if batch_idx == 1:
                #images = utils.make_grid(data,padding = 0)
                #image_show(images)
                print('GroundTruth: ', ' '.join('%d' % target[j] for j in range(64)))
                print('Predicted: ', ' '.join('%d' % predicted[j] for j in range(64)))
            total += target.size(0)
            correct += (predicted == target).sum().item()# 对预测正确的数据个数进行累加
            # 不同类别的数量统计（区别于总体）
            c = (predicted == target)
            for i in range(10):
                lable = target[i]
                class_correct[lable] += c[i].sum().item()
                class_total[lable] += 1
            print(class_correct[i])
            
    test_loss /= len(test_loader.dataset)   # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均los
                                            # 将变量值或属性值除以表达式值，并将浮点数结果赋给该变量或属性
                                            # variableorproperty /= expression  
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    for i in range(10):
        print('Accuracy of %0f : %2d %%' % 
                (i, 100* class_correct[i]
                / class_total[i]))
        
    print(conf_matrix)
    #plt.figure(figsize=(10,10))
    #plot_confusion_matrix(conf_matrix, names)
    test_accs.append(100 * correct / total)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
    
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
    
        fmt = '.2f' if normalize else '.0f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

names = ("0","1","2","3","4", "5","6","7","8","9")

def confusion_matrix(preds, labels, conf_matrix):
        preds = torch.argmax(preds, 1)
        for p,t in zip(preds,labels):
            conf_matrix[p,t] += 1
        return conf_matrix

def main():
    # Training settings
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example') 
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',  # 训练的时候每次喂入的样本数量
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', # 训练的时候每次喂入的样本数
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',   # 训练轮次
                        help='number of epochs to train (default: 14)')  
    parser.add_argument('--lr', type=float, default= 0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status') # 每训练多少轮次以后记录一次状态
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('-f', type=str, default="读取额外的参数")
    args = parser.parse_args()  # 存储终端输入的参数
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed) # 设置torch的值

    device = torch.device("cpu") # 运算使用GPU或者CPU

    # batch_size is a crucial hyper-parameter
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}


    # Normalize the input (black and white image)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Make train dataset split
    dataset1 = datasets.MNIST("mnist-data", train=True, download=True,
                       transform=transform)
    # Make test dataset split
    dataset2 = datasets.MNIST("mnist-data", train=False,download = True,
                       transform=transform)

    # Convert the dataset to dataloader, including train_kwargs and test_kwargs
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    # Put the model on the GPU or CPU
    model = Net().to(device)
     
    '''
   #这部分还没有跑通 但思路是这样
    # 导入Pytorch中自带的resnet18网络模型
    model_ft = models.resnet18(pretrained=True)
    
    # 将网络模型的各层的梯度更新置为False
    for param in model_ft.parameters():
        param.requires_grad = False
 
    # 修改网络模型的最后一个全连接层
    # 获取最后一个全连接层的输入通道数
    num_ftrs = model_ft.fc.in_features
    # 修改最后一个全连接层的的输出数为10（0-9的数字）
    model_ft.fc = nn.Linear(num_ftrs, 10)
    # 是否使用gpu
    if use_gpu:
        model_ft = model_ft.cuda()
 
    # 定义网络模型的损失函数
    criterion = nn.CrossEntropyLoss()
 
    # 只训练最后一个层
    # Create optimizer
    optimizer = optim.Adadelta(model_ft.fc.parameters(), lr=args.lr/10)#The general approach is to make the initial learning rate 10 times smaller than that of Training from scratch.

    '''
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr,)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.99))
    # 定义四个不同的优化器
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(),lr=args.lr, momentum=0.8)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.99))

    # Create a schedule for the optimizer
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    scheduler= CosineAnnealingLR(optimizer,T_max=20,eta_min=0.05)
    # Begin training and testing
    epochs = []
    for epoch in range(1, args.epochs + 1):
        epochs.append(epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

              
    print(epochs) 
    print(test_accs)
    print(train_accs)
    
   
    plt.plot(epochs,train_accs,color='r',label='train_acc')        
    plt.plot(epochs,test_accs,color='b',label='test_acc')  
    plt.xlabel('epochs')    
    plt.ylabel('accuracy')   
    plt.title("change of train(test) accuracy")      
    plt.legend()     
    plt.savefig('test.jpg')  
    plt.show()               

    # Save the model
    if args.save_model == True:
        torch.save(model.state_dict(), "D://INT305//model//mnist_cnn.pt")
        

if __name__ == '__main__':
    main()
    