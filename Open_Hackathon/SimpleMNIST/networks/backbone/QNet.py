import torch.nn as nn
import torch
import pennylane as qml
from pennylane import numpy as np

class SimpleNet(nn.Module):
  def __init__(self):
    super(SimpleNet, self).__init__()
    
    #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(4, 20, kernel_size=5)
    self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
    self.pool = nn.MaxPool2d(2)
    self.AdaptPool = nn.AdaptiveMaxPool2d(4)
    self.fc1 = nn.Linear(640, 64)
    self.fc2 = nn.Linear(64, 10)
    self.relu = nn.ReLU()

    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (3, n_qubits, 3)}
    self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

  def qconv(self, x):
    def flatten(t):
      t = t.reshape(1, -1)
      t = t.squeeze()
      return t

    s = 2 # kernel_size
    f = 4 # depth
    q_out = torch.zeros(1,f,x.shape[3]-s+1, x.shape[3]-s+1)
    for idx in range(x.shape[3]-s+1):
      for idy in range(x.shape[3]-s+1):
        q_out[:,:,idx,idy] = self.qlayer(flatten(x[0,0,idx:idx+s,idy:idy+s]))
    
    return q_out

  def forward(self, x):

    #x = self.pool(self.relu(self.conv1(q_out))) 
    x = self.pool(self.sigmoid(self.qconv(x)))       
    x = self.pool(self.relu(self.conv2(x)))
    x = self.AdaptPool(self.relu(self.conv3(x))) # The output shape will always be 40*4*4 = 640.
    x = x.view(-1, 640)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x
