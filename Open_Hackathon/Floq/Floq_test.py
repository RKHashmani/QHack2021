import remote_cirq
import pennylane as qml
import numpy as np
import sys, time
wires = 32
np.random.seed(0)

weights = np.random.randn(1, wires, 3)
API_KEY = "AIzaSyCyEpDpnnBO5Z1BaPWMCRyzFC_9redBQ4Q"
sim = remote_cirq.RemoteSimulator(API_KEY)

dev = qml.device("cirq.simulator",
                 wires=wires,
                 simulator=sim,
                 analytic=False)

@qml.qnode(dev)
def my_circuit(weights):
	qml.templates.layers.StronglyEntanglingLayers(weights,
	                                              wires=range(wires))
	return qml.expval(qml.PauliZ(0))

t0 = time.time()
print(my_circuit(weights))
print('It took %.4f seconds!'%(time.time()-t0))