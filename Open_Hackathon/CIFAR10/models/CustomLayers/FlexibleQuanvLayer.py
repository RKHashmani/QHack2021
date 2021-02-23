import torch.nn as nn
import torch
import pennylane as qml
from pennylane import numpy as np


class Quanv(nn.Module):
    def __init__(self, kernal_size, output_depth, circuit_layers):
        super().__init__()


        self.kernal_size = kernal_size  # kernel_size
        self.f = output_depth  # depth
        self.kernal_area = self.kernal_size ** 2
        self.n_qubits = self.kernal_area # n_qubits

        dev = qml.device("default.qubit", wires=self.n_qubits)
        @qml.qnode(dev)
        def circuit(inputs, weights):
            for j in range(self.n_qubits):
                qml.RY(np.pi * inputs[j], wires=j)
            qml.templates.RandomLayers(weights, wires=list(range(self.n_qubits)))
            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]

        self.circuit_layers = circuit_layers
        params = {"weights": (self.circuit_layers, self.kernal_area)} # 4 for area of kernel, 2x2
        self.qlayer = qml.qnn.TorchLayer(circuit, params)


    def forward(self, x):
        q_out = torch.zeros((x.shape[3] - self.kernal_size + 1), (x.shape[3] - self.kernal_size + 1), self.kernal_area)

        for idx in range(x.shape[3] - self.kernal_size + 1):
            for idy in range(x.shape[2] - self.kernal_size + 1):
                for idz in range(x.shape[1]):
                    q_out[idx, idy] += self.qlayer(self.flatten(x[0, idz, idx:idx + self.kernal_size, idy:idy + self.kernal_size]))

        return torch.reshape(q_out, (1, self.f, x.shape[3] - self.kernal_size + 1, x.shape[3] - self.kernal_size + 1))

    def flatten(self, t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t
