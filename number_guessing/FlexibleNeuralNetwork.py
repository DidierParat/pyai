import logging

LOG = logging.getLogger('FlexibleNeuralNetwork')


class FlexibleNeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, neural_network_layer):
        self.layers.append(neural_network_layer)

    def train(self, training_set):
        if not self.layers:
            print("No neural network layers set.")
            return

        layer_output = None
        final_output = None
        for (training_input, expected_output) in training_set:
            # feed forward
            layer_output = training_input
            layers_input = []
            layers_output = []
            for i in range(len(self.layers)):
                LOG.debug("ff: i: {}".format(i))
                layer = self.layers[i]
                layers_input.append(layer_output)
                LOG.debug("ff: layer_input: {}".format(layer_output))
                layer_output = layer.feed_forward(layer_output)
                layers_output.append(layer_output)
                LOG.debug("ff: layer_output: {}".format(layer_output))
            final_output = layer_output

            # back propagate
            partial_diff_of_cost_to_output = 2 * (expected_output - layer_output)
            for i in reversed(range(len(self.layers))):
                LOG.debug("bp: i: {}".format(i))
                layer = self.layers[i]
                layer_output = layers_output[i]
                layer_input = layers_input[i]
                LOG.debug("bp: layer_output: {}".format(layer_output))
                LOG.debug("bp: layer_input: {}".format(layer_input))
                partial_diff_of_cost_to_output = layer.back_propagate(
                    layer_input, layer_output, partial_diff_of_cost_to_output)
        return final_output
