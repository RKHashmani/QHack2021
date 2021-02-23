import torch.nn as nn
import torch
import pennylane as qml
from pennylane import numpy as np
# import pennylane_qulacs

from networks.backbone.CustomLayers.FlexibleQuanvLayer import Quanv


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)

        self.AdaptPoolQuan = nn.AdaptiveMaxPool2d(5)
        self.CConv = nn.Conv2d(5, 4, kernel_size=2)

        self.conv2 = nn.Conv2d(4, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=2)
        self.pool = nn.MaxPool2d(2)
        self.AdaptPool = nn.AdaptiveMaxPool2d(4)
        self.fc1 = nn.Linear(640, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.Quanv = Quanv(2, 4, 1)

    def forward(self, x):
        x = self.AdaptPoolQuan(self.relu(self.conv1(x)))

        x = self.sigmoid(self.Quanv(x))

        # x = self.sigmoid(self.CConv(x))

        x = self.relu(self.conv2(x))
        x = self.AdaptPool(self.relu(self.conv3(x)))  # The output shape will always be 40*4*4 = 640.
        x = x.view(-1, 640)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x