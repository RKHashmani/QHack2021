#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #

    LAYERS = 2
    WIRES = 3

    dev = qml.device('default.qubit', wires=WIRES)

    # Minimize the circuit
    def variational_circuit(inputs, params):
        parameters = params.reshape((LAYERS, WIRES, 3))
        qml.templates.embeddings.AngleEmbedding(inputs, wires=[0, 1, 2], rotation='Y')
        qml.templates.StronglyEntanglingLayers(parameters, wires=range(WIRES))
        return qml.expval(qml.PauliZ(0))



    circuit = qml.QNode(variational_circuit, dev)

    def square_loss(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss

    def cost(params, X_train, Y_train):
        predictions = [circuit(x, params) for x in X_train ]
        return square_loss(Y_train, predictions)

    opt = qml.AdamOptimizer(stepsize=0.1)

    steps = 90
    conv_tolerance = 0.01

    training_params= np.random.rand((LAYERS * WIRES * 3))

    for i in range (steps):
        training_params, prev_cost = opt.step_and_cost(lambda v: cost(v, X_train, Y_train), training_params)

        curr_cost = cost(training_params, X_train, Y_train)

        conv = np.abs(curr_cost - prev_cost)

        if (i + 1) % 2 == 0:
            print("Cost after step {:5d}: {: .7f}".format(i + 1, curr_cost))

        if conv <= conv_tolerance:
            break

    predictions = [circuit(x, training_params) for x in X_test]

    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
