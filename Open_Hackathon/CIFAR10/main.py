import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from utils import load_custom_state_dict

import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

num_batches = 10
batch_size = 100
image_list = list(range(num_batches*batch_size))

# load train set
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

# select 3 classes
y = np.array(trainset.targets)
pos_i = np.argwhere( (y==0) | (y==1) | (y==2))
pos_i = list(pos_i[:,0])
trainset.data = [trainset.data[j] for j in pos_i]
trainset.targets = [trainset.targets[j] for j in pos_i]

# select a subset
trainset_1 = torch.utils.data.Subset(trainset, image_list)
# create train loader
trainloader = torch.utils.data.DataLoader(
    trainset_1, batch_size=batch_size, shuffle=True, num_workers=2)
# load test set
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
# select 3 classes
y = np.array(testset.targets)
pos_i = np.argwhere( (y==0) | (y==1) | (y==2))
pos_i = list(pos_i[:,0])
testset.data = [testset.data[j] for j in pos_i]
testset.targets = [testset.targets[j] for j in pos_i]
# select a subset
testset_1 = torch.utils.data.Subset(testset, image_list)
# create test loader
testloader = torch.utils.data.DataLoader(
    testset_1, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

# net = ResNet18()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = QuobileNet()
# net = SimpleNet()
net = QuanvNet()

net = net.to(device)
if device == 'cuda':
    print ("GPU Use Enabled")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('pretrained_models'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./pretrained_models/Classical_1_layer_depth_1-epoch_274.pth')
    load_custom_state_dict(net, checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=4e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if not os.path.exists('tools/eval_stats'):
    os.makedirs('tools/eval_stats')

try:
    os.remove('tools/eval_stats/log_validation_Quanv_1_layer_depth_1.csv')
except OSError:
    pass


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    with open('tools/eval_stats/log_validation_Quanv_1_layer_depth_1.csv', 'a') as f:
        f.write('%.4f, %.4f\n' %(100.*correct/total, test_loss))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('pretrained_models'):
            os.mkdir('pretrained_models')
        torch.save(state, './pretrained_models/Quanv_1_layer_depth_1-epoch_{}.pth'.format(epoch + 1))
        best_acc = acc

for epoch in range(start_epoch, start_epoch+350):
    train(epoch)
    test(epoch)
    scheduler.step()
