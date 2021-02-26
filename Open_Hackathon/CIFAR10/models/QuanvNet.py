import torch
import torch.nn as nn
from models.CustomLayers import *


class QuanvNet(nn.Module):
    def __init__(self):
        super(QuanvNet, self).__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # Same Max pool as above
        self.convPool = nn.Conv2d(32, 16, kernel_size=1)

        # Classical Alts to Quanv Layers
        self.QuanvALT1 = nn.Conv2d(16, 4, kernel_size=3)
        self.QuanvALT2 = nn.Conv2d(4, 4, kernel_size=3)

        # Quanv Layers
        self.quanv1 = Quanv(kernal_size=3, output_depth=4, circuit_layers=1)
        self.quanv2 = Quanv(kernal_size=3, output_depth=4, circuit_layers=1)

        # Flatten
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 3)


        # Activation Functions

        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.pool(out)
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.pool(out)
        out = self.convPool(out)

        out = self.relu(self.QuanvALT1(out))
        # out = self.relu(self.QuanvALT2(out))

        # out = self.relu(self.quanv1(out))
        out = self.relu(self.quanv2(out))

        out = out.view(-1, 64)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.softmax(self.fc2(out))

        return out
