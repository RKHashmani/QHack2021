import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CustomLayers import *


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Same Max pool as above
        # Flatten
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, 3)

        # Quanvolution Layer
        # self.Quanv1 = Quanv(kernal_size=3, output_depth=4, circuit_layers=1)

        # Activation Functions

        # self.AdaptPool = nn.AdaptiveMaxPool2d(3)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.pool(out)
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.pool(out)
        out = out.view(-1, 4096)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.softmax(self.fc2(out),)

        return out
