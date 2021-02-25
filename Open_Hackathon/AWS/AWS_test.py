import pennylane as qml
import numpy as np
import sys, time
wires = 30
np.random.seed(0)

#device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
# Aspen-9 QPU
#device_arn = "arn:aws:braket:::device/qpu/rigetti/Aspen-9"
# Please enter the S3 bucket you created during onboarding
# (or any other S3 bucket starting with 'amazon-braket-' in your account) in the code below

my_bucket = f"amazon-braket-b011687def13" # the name of the bucket
my_prefix = "pennylane-test" # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)

dev = qml.device('braket.aws.qubit', device_arn=device_arn, wires=wires, s3_destination_folder=s3_folder)

weights = np.random.randn(1, wires, 3)

@qml.qnode(dev)
def my_circuit(weights):
	qml.templates.layers.StronglyEntanglingLayers(weights,
	                                              wires=range(wires))
	return qml.expval(qml.PauliZ(0))

t0 = time.time()
print(my_circuit(weights))
print('It took %.4f seconds!'%(time.time()-t0))