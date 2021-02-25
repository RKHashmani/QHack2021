import remote_cirq
import pennylane as qml
import numpy as np
import sys, time
wires = 32
np.random.seed(0)


API_KEY = "AIzaSyCyEpDpnnBO5Z1BaPWMCRyzFC_9redBQ4Q"
sim = remote_cirq.RemoteSimulator(API_KEY)

dev = qml.device("cirq.simulator",
                 wires=wires,
                 simulator=sim,
                 analytic=False)

@qml.qnode(dev)
def circuit(inputs, weights):
    qml.templates.embeddings.AngleEmbedding(inputs, wires=list(range(wires)), rotation='Y')
    qml.templates.layers.BasicEntanglerLayers(weights, wires=list(range(wires)), rotation='Y')
    return [qml.expval(qml.PauliZ(j)) for j in range(wires)]

inputs = np.random.rand(wires)
for i in range(9,32):
    inputs[i] = np.pi/2
weights = np.random.randn(1, wires)
t0 = time.time()
print(circuit(inputs, weights))
print('It took %.4f seconds!'%(time.time()-t0))