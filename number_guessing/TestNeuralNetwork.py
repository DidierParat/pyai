import unittest
import numpy as np
import logging

from number_guessing.FlexibleNeuralNetwork import FlexibleNeuralNetwork
from number_guessing.NeuralNetworkLayer import NeuralNetworkLayer

FORMAT = '[%(asctime)-15s] - %(name)s.%(funcName)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


class Test1Dimension(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.training_rounds = 200

    def test_1_layer(self):
        training_input = np.array([[1]])
        expected_output = np.array([[1]])
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        loss = None
        for i in range(self.training_rounds):
            loss = flexible_neural_network.train([(training_input, expected_output)])
        self.log.info("Loss: {}".format(loss))
        prediction = flexible_neural_network.predict(training_input)
        self.log.info("Prediction, Expected: {}, {}".format(prediction, expected_output))
        np.testing.assert_almost_equal(loss, 0, 2)

    def test_1_layer_goal_0(self):
        training_input = np.array([[1]])
        expected_output = np.array([[0]])
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        loss = None
        for i in range(self.training_rounds):
            loss = flexible_neural_network.train([(training_input, expected_output)])
        self.log.info("Loss: {}".format(loss))
        prediction = flexible_neural_network.predict(training_input)
        self.log.info("Prediction, Expected: {}, {}".format(prediction, expected_output))
        np.testing.assert_almost_equal(loss, 0, 2)

    def test_2_layers(self):
        training_input = np.array([[0]])
        expected_output = np.array([[0]])
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        loss = None
        for i in range(self.training_rounds):
            loss = flexible_neural_network.train([(training_input, expected_output)])
        self.log.info("Loss: {}".format(loss))
        prediction = flexible_neural_network.predict(training_input)
        self.log.info("Prediction, Expected: {}, {}".format(prediction, expected_output))
        np.testing.assert_almost_equal(loss, 0, 2)

    def test_5_layers(self):
        training_input = np.array([[1]])
        expected_output = np.array([[1]])
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 1))
        loss = None
        for i in range(self.training_rounds):
            loss = flexible_neural_network.train([(training_input, expected_output)])
        self.log.info("Loss: {}".format(loss))
        prediction = flexible_neural_network.predict(training_input)
        self.log.info("Prediction, Expected: {}, {}".format(prediction, expected_output))
        np.testing.assert_almost_equal(loss, 0, 2)


class Test2Dimensions(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.training_rounds = 200

    def test_1_layer_1D_output(self):
        training_input = np.array([[1],
                                   [1]])
        expected_output = np.array([[1]])
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(1, 2))
        loss = None
        for i in range(self.training_rounds):
            loss = flexible_neural_network.train([(training_input, expected_output)])
        self.log.info("Loss: {}".format(loss))
        prediction = flexible_neural_network.predict(training_input)
        self.log.info("Prediction, Expected: {}, {}".format(prediction, expected_output))
        np.testing.assert_almost_equal(loss, 0, 2)

    def test_1_layer_2D_output(self):
        training_input = np.array([[1],
                                   [1]])
        expected_output = np.array([[1],
                                    [1]])
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(2, 2))
        loss = None
        for i in range(self.training_rounds):
            loss = flexible_neural_network.train([(training_input, expected_output)])
        self.log.info("Loss: {}".format(loss))
        prediction = flexible_neural_network.predict(training_input)
        self.log.info("Prediction, Expected: {}, {}".format(prediction, expected_output))
        np.testing.assert_almost_equal(loss, 0, 2)

    def test_1_layer_2D_output_bis(self):
        training_input = np.array([[0],
                                   [1]])
        expected_output = np.array([[1],
                                    [0]])
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(2, 2))
        loss = None
        for i in range(self.training_rounds):
            loss = flexible_neural_network.train([(training_input, expected_output)])
        self.log.info("Loss: {}".format(loss))
        prediction = flexible_neural_network.predict(training_input)
        self.log.info("Prediction, Expected: {}, {}".format(prediction, expected_output))
        np.testing.assert_almost_equal(loss, 0, 2)


class Test4Dimensions(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.training_rounds = 100

    # From: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
    def test_2_layers(self):
        #training_input = np.array([[0, 0, 1],
        #                           [0, 1, 1],
        #                           [1, 0, 1],
        #                           [1, 1, 1]])
        training_input = np.array([[0],
                                   [0],
                                   [1],
                                   [1]])
        expected_output = np.array([[0],
                                    [1],
                                    [1],
                                    [0]])
        flexible_neural_network = FlexibleNeuralNetwork()
        flexible_neural_network.add_layer(NeuralNetworkLayer(4, 4))
        flexible_neural_network.add_layer(NeuralNetworkLayer(4, 4))
        loss = None
        for i in range(self.training_rounds):
            loss = flexible_neural_network.train([(training_input, expected_output)])
        self.log.info("Loss: {}".format(loss))
        prediction = flexible_neural_network.predict(training_input)
        self.log.info("Prediction, Expected: {}, {}".format(prediction, expected_output))
        #np.testing.assert_almost_equal(loss, 0, 2)


if __name__ == "__main__":
    unittest.main()
