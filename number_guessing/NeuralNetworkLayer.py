import numpy as np

from number_guessing.MathUtil import sigmoid, sigmoid_derivative


class NeuralNetworkLayer:
    def __init__(self, nb_of_neurons, preceding_layer_size):
        self.nb_of_neurons = nb_of_neurons
        self.weights = np.zeros(shape=(nb_of_neurons, preceding_layer_size))
        #self.weights = np.random.rand(nb_of_neurons, preceding_layer_size)
        self.biases = np.zeros(shape=(nb_of_neurons, 1))

    def feed_forward(self, nnl_input):
        z = np.dot(self.weights, nnl_input) + self.biases
        return sigmoid(z)

    def back_propagate(self, last_input, last_output, partial_diff_to_output):
        inter = partial_diff_to_output * sigmoid_derivative(last_output)
        if np.shape(last_input) == np.shape(inter):
            partial_diff_to_weight = np.dot(last_input.T, inter)
        else:
            partial_diff_to_weight = np.dot(last_input, inter)
        inter = np.dot(self.weights.T, partial_diff_to_output)
        if np.shape(last_output) == np.shape(inter):
            partial_diff_to_previous_output = np.dot(sigmoid_derivative(last_output).T, inter)
        else:
            partial_diff_to_previous_output = np.dot(sigmoid_derivative(last_output).T, inter.T)
        self.weights += partial_diff_to_weight.T
        return partial_diff_to_previous_output
