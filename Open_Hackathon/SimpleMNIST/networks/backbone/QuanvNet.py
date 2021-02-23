import torch.nn as nn
import torch
import pennylane as qml
from pennylane import numpy as np
# import pennylane_qulacs

from networks.backbone.CustomLayers.FlexibleQuanvLayer import Quanv


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)

        self.convALT = nn.Conv2d(20, 4, kernel_size=2)

        self.Quanv1 = Quanv(2, 4, 1)
        self.AdaptPool = nn.AdaptiveMaxPool2d(3)

        self.fc1 = nn.Linear(36, 12)
        self.fc2 = nn.Linear(12, 3)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # x = self.sigmoid(self.convALT(x))
        x = self.sigmoid(self.Quanv1(x))
        x = self.AdaptPool(x)
        x = x.view(-1, 36)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)




        # x = self.sigmoid(self.CConv(x))

        return x
