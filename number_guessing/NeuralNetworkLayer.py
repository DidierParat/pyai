import numpy as np

from number_guessing.MathUtil import sigmoid, sigmoid_derivative


class NeuralNetworkLayer:
    def __init__(self, nb_of_neurons, preceding_layer_size):
        self.nb_of_neurons = nb_of_neurons
        self.weights = np.zeros(shape=(nb_of_neurons, preceding_layer_size))
        self.biases = np.zeros(shape=(nb_of_neurons, 1))

    def feed_forward(self, nnl_input):
        return sigmoid(np.dot(self.weights, nnl_input) + self.biases)

    def back_propagate(self, last_input, last_output, partial_diff_to_output):
        partial_diff_to_weight = last_input * sigmoid_derivative(last_output) * partial_diff_to_output
        partial_diff_to_previous_output = self.weights * sigmoid_derivative(last_output) * partial_diff_to_output
        self.weights += partial_diff_to_weight
        return partial_diff_to_previous_output
