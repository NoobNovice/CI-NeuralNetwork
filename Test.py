import numpy

class Test:
    def feed_forward(self, weight, bias, layer):
        for cur_layer in range(1, len(layer)):  # except layer 0 input layer
            for cur_node in range(0, len(layer[cur_layer])):
                v = numpy.dot(layer[cur_layer - 1], weight[cur_layer - 1][cur_node])
                v += bias[cur_layer - 1][cur_node]
                layer[cur_layer][cur_node] = self.activation_func(v)
        return