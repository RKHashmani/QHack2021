import torch.nn as nn
import torch
import pennylane as qml
from pennylane import numpy as np
# import pennylane_qulacs

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

    n_qubits = 4
    dev1 = qml.device("default.qubit", wires=n_qubits)

    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    @qml.qnode(dev1)
    def circuit(inputs, weights):
      for j in range(n_qubits):
        qml.RY(np.pi * inputs[j], wires=j)
      qml.templates.RandomLayers(weights, wires=list(range(n_qubits)))
      return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    weight_shapes = {"weights": (3, n_qubits, 3)}
    # self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)  # Cenks

    n_layers = 2
    params = {"weights": (n_layers, 4)}
    self.qlayer = qml.qnn.TorchLayer(circuit, params)

  def qconv(self, x):
    def flatten(t):
      t = t.reshape(1, -1)
      t = t.squeeze()
      return t

    s = 2 # kernel_size
    f = 4 # depth
    #q_out = torch.zeros(1,f,x.shape[3]-s+1, x.shape[3]-s+1)
    q_out = torch.zeros((x.shape[3]-s+1),(x.shape[3]-s+1), s**2)
    count = 0
    for idx in range(x.shape[3]-s+1):
      for idy in range(x.shape[3]-s+1):
        for idz in range(x.shape[2]):
          q_out[idx,idy] += self.qlayer(flatten(x[0,idz,idx:idx+s,idy:idy+s]))

    return torch.reshape(q_out, (1,f,x.shape[3]-s+1,x.shape[3]-s+1))

  def forward(self, x):

    x = self.AdaptPoolQuan(self.relu(self.conv1(x)))

    x = self.sigmoid(self.qconv(x))
    #x = self.sigmoid(self.CConv(x))

    x = self.relu(self.conv2(x))
    x = self.AdaptPool(self.relu(self.conv3(x))) # The output shape will always be 40*4*4 = 640.
    x = x.view(-1, 640)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x
