import numpy as np
from PIL import Image
from number_guessing.MnistDbReader import MnistDbReader
from number_guessing.NeuralNetwork import NeuralNetwork
from number_guessing.NeuralNetworkLayer import NeuralNetworkLayer


def pretty_print(label, image):
    # TODO fix matrix
    Image.fromarray(np.asmatrix(image), 'L').show()
    print("Image displayed of number {}.".format(label))


if __name__ == "__main__":
    # Load training data
    data_reader = MnistDbReader("resources/train-images-idx3-ubyte", "resources/train-labels-idx1-ubyte", 100)
    data_reader.initialize()
    label, image = data_reader.get_next_tuple()
    #pretty_print(label, image)

    # Create neural network
    hidden_layer1 = NeuralNetworkLayer(16, 28*28)
    output_layer = NeuralNetworkLayer(10, 16)

    # Train model
    while (label is not None) and (image is not None):
        hidden_layer1_output = hidden_layer1.feed_forward(image)
        final_output = output_layer.feed_forward(hidden_layer1_output)
        # TODO
        label, image = data_reader.get_next_tuple()

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)

    for i in range(500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
