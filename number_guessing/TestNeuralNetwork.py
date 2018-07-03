import unittest
import numpy as np
import logging

from number_guessing.FlexibleNeuralNetwork import FlexibleNeuralNetwork
from number_guessing.NeuralNetworkLayer import NeuralNetworkLayer

FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
LOG = logging.getLogger('TestNeuralNetwork')
training_rounds = 10000


class Test1Dimension1Weight(unittest.TestCase):
    def test1(self):
        training_input = np.matrix([1])
        expected_output = 1
        output_layer = NeuralNetworkLayer(1, 1)
        final_output = None
        for i in range(training_rounds):
            final_output = output_layer.feed_forward(training_input)
            partial_diff_of_cost_to_output = 2 * (expected_output - final_output)
            output_layer.back_propagate(training_input, final_output, partial_diff_of_cost_to_output)
        #np.testing.assert_almost_equal(final_output, expected_output, 1)
        LOG.info("1D-1W - #1 - final output: {}".format(final_output))

    def test1bis(self):
        training_input = np.matrix([1])
        expected_output = 1
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        output = None
        for i in range(training_rounds):
            output = flexible_neural_network.train([(training_input, expected_output)])
        #np.testing.assert_almost_equal(output, expected_output, 1)
        LOG.info("1D-1W - #1bis - final output: {}".format(output))

    def test2(self):
        training_input = np.matrix([1])
        expected_output = 0
        output_layer = NeuralNetworkLayer(1, 1)
        final_output = None
        for i in range(training_rounds):
            final_output = output_layer.feed_forward(training_input)
            partial_diff_of_cost_to_output = 2 * (expected_output - final_output)
            output_layer.back_propagate(training_input, final_output, partial_diff_of_cost_to_output)
        #np.testing.assert_almost_equal(final_output, expected_output, 1)
        LOG.info("1D-1W - #2 - final output: {}".format(final_output))


# class Test2Dimensions1Weight(unittest.TestCase):
#     def test1(self):
#         training_input = np.matrix([[1],
#                                     [1]])
#         expected_output = 1
#         output_layer = NeuralNetworkLayer(1, 2)
#         final_output = None
#         for i in range(1000):
#             final_output = output_layer.feed_forward(training_input)
#             partial_diff_of_cost_to_output = 2 * (expected_output - final_output)
#             output_layer.back_propagate(training_input, final_output, partial_diff_of_cost_to_output)
#         LOG.info("2D-1W - #1 - final output: {}".format(final_output))

#     def test2(self):
#         training_input = np.matrix([[1],
#                                     [1]])
#         expected_output = 0
#         output_layer = NeuralNetworkLayer(1, 2)
#         final_output = None
#         for i in range(1000):
#             final_output = output_layer.feed_forward(training_input)
#             #print("weight: {}".format(output_layer.weights))
#             #print("biases: {}".format(output_layer.biases))
#             #print("output: {}".format(final_output))
#             output_layer.back_propagate(training_input, final_output, expected_output)
#         LOG.info("2D-1W - #2 - Final output: {}".format(final_output))
#
#     def test3(self):
#         training_input = np.matrix([[1],
#                                     [1]])
#         expected_output = np.matrix([[1],
#                                      [1]])
#         output_layer = NeuralNetworkLayer(2, 2)
#         final_output = None
#         for i in range(1000):
#             final_output = output_layer.feed_forward(training_input)
#             #print("weight: {}".format(output_layer.weights))
#             #print("biases: {}".format(output_layer.biases))
#             #print("output: {}".format(final_output))
#             output_layer.back_propagate(training_input, final_output, expected_output)
#         LOG.info("2D-1W - #3 - Final output: {}".format(final_output))
#
#
class Test1Dimension2Weights(unittest.TestCase):
    def test1(self):
        training_input = np.matrix([1])
        expected_output = 1
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        output = None
        for i in range(training_rounds):
            output = flexible_neural_network.train([(training_input, expected_output)])
#        np.testing.assert_almost_equal(output, expected_output, 1)
        LOG.info("1D-2W - #1 - final output: {}".format(output))


class Test1Dimension5Weights(unittest.TestCase):
    def test1(self):
        training_input = np.matrix([1])
        expected_output = 1
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        output = None
        for i in range(training_rounds):
            output = flexible_neural_network.train([(training_input, expected_output)])
#        np.testing.assert_almost_equal(output, expected_output, 1)
        LOG.info("1D-5W - #1 - final output: {}".format(output))


# class Test4Dimensions2Weights(unittest.TestCase):
#     def test1(self):
#         X = np.array([[0, 0, 1],
#                       [0, 1, 1],
#                       [1, 0, 1],
#                       [1, 1, 1]])
#         y = np.array([[0], [1], [1], [0]])
#         hidden_layer1 = NeuralNetworkLayer(4, 3)
#         output_layer = NeuralNetworkLayer(4, 4)
#         final_output = None
#         for i in range(1500):
#             hidden_layer1_output = hidden_layer1.feed_forward(X)
#             final_output = output_layer.feed_forward(hidden_layer1_output)
#
#             output_layer.back_propagate(hidden_layer1_output, final_output, y)
#             hidden_layer1.back_propagate(X, hidden_layer1_output, y)
#         LOG.info("4D-2W - #1 - final output: {}".format(final_output))


if __name__ == "__main__":
    unittest.main()
