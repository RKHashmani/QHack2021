import torch.nn as nn
import torch
import pennylane as qml
from pennylane import numpy as np


class Quanv(nn.Module):
    def __init__(self, n_qubits, n_layers, kernal_size, output_depth):
        super().__init__()

        self.n_qubits = n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev)
        def circuit(inputs, weights):
            for j in range(n_qubits):
                qml.RY(np.pi * inputs[j], wires=j)
            qml.templates.RandomLayers(weights, wires=list(range(n_qubits)))
            return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

        self.n_layers = n_layers
        params = {"weights": (n_layers, 4)} # 4 for area of kernel, 2x2
        self.qlayer = qml.qnn.TorchLayer(circuit, params)

        self.s = kernal_size  # kernel_size
        self.f = output_depth  # depth

    def forward(self, x):
        q_out = torch.zeros((x.shape[3] - self.s + 1), (x.shape[3] - self.s + 1), self.s ** 2)

        for idx in range(x.shape[3] - self.s + 1):
            for idy in range(x.shape[3] - self.s + 1):
                for idz in range(x.shape[2]):
                    q_out[idx, idy] += self.qlayer(self.flatten(x[0, idz, idx:idx + self.s, idy:idy + self.s]))

        return torch.reshape(q_out, (1, self.f, x.shape[3] - self.s + 1, x.shape[3] - self.s + 1))

    def flatten(self, t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t
