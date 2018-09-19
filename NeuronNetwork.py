import random
import math
import numpy
import copy

def innit_array(number_arr):
    return [0] * number_arr

class NeuronNetwork:
    # create structure
    layer = []
    weight = []
    past_weight = []
    gradient = []
    bias = []
    def __init__(self, num_inputNode, num_outputNode, arr_hiddenNode):
        # append input node layer
        self.layer.append(innit_array(num_inputNode))

        # append hidden node layer
        for cur_layer in range(0, len(arr_hiddenNode)):
            self.layer.append(innit_array(arr_hiddenNode[cur_layer]))
            self.bias.append(innit_array(arr_hiddenNode[cur_layer]))

        # append output node layer
        self.layer.append(innit_array(num_outputNode))
        self.bias.append(innit_array(num_outputNode))

        # append gradient arr
        self.gradient = copy.deepcopy(self.bias)

        # append weight
        for cur_layer in range(1, len(self.layer)):
            weight_of_node = []
            for cur_node in range(0, len(self.layer[cur_layer])):
                weight_of_node.append(innit_array(len(self.layer[cur_layer - 1])))
            self.weight.append(weight_of_node)
        return

    # set weight and bias value
    def set_value(self):
        # weight
        for cur_layer in range(0, len(self.weight)):
            for cur_node in range(0, len(self.weight[cur_layer])):
                for cur_wire in range(0, len(self.weight[cur_layer][cur_node])):
                    self.weight[cur_layer][cur_node][cur_wire] = random.uniform(0, 1)

        self.past_weight = copy.deepcopy(self.weight)
        # bias
        for cur_layer in range(0, len(self.bias)):
            for cur_node in range(0, len(self.bias[cur_layer])):
                self.bias[cur_layer][cur_node] = random.uniform(0, 1)
        return

    # sigmoid function
    @staticmethod
    def activation_func(v):
        y = 1 / (1 + math.exp(-v))
        return y

    # diff sigmoid function
    @staticmethod
    def diff_func(y):
        y_diff = y * (1 - y)
        return y_diff

    cut_row = 0 # number of input when train must equal test
    # train neuron network
    def train(self, data_set, num_parameter, learningRate, alpha, hopCount):
        self.cut_row = num_parameter
        E = 1
        count = 0
        data_row = 0
        print("hop 0")
        print("===============================================================================================")
        while E >= math.pow(10, -6) and count <= hopCount:
            print("E is {}".format(E))
            print("Progress  {}%".format(round((100*data_row)/len(data_set), 2)))
            # set input
            if data_row == len(data_set):
                data_row = 0
                count += 1
                print("hop {}".format(count))
                print("===============================================================================================")
            for data_col in range(0, num_parameter):
                self.layer[0][data_col] = data_set[data_row][data_col]

            # feed forward step
            for cur_layer in range(1, len(self.layer)): # except layer 0 input layer
                for cur_node in range(0, len(self.layer[cur_layer])):
                    v = numpy.dot(self.layer[cur_layer - 1], self.weight[cur_layer - 1][cur_node])
                    v += self.bias[cur_layer - 1][cur_node]
                    self.layer[cur_layer][cur_node] = self.activation_func(v)

            # back propagation step
            design_output = []
            for data_col in range(num_parameter, len(data_set[data_row])):
                design_output.append(data_set[data_row][data_col])
            error_output = numpy.subtract(design_output, self.layer[len(self.layer) - 1])
            E = numpy.sum(numpy.power(error_output, 2))/len(error_output)
            # compute gradient
            for cur_layer in range(len(self.weight) - 1, -1, -1):
                # case output node (gradient)
                if cur_layer == len(self.weight) - 1:
                    for cur_node in range(0, len(self.weight[cur_layer])):
                        equation = error_output[cur_node] * self.diff_func(self.layer[cur_layer + 1][cur_node])
                        self.gradient[cur_layer][cur_node] = equation
                # case hidden node (local gradient)
                else:
                    for cur_node in range(0, len(self.weight[cur_layer])):
                        sum = 0
                        for next_node in range(0, len(self.weight[cur_layer + 1])):
                            sum += self.weight[cur_layer + 1][next_node][cur_node] * self.gradient[cur_layer + 1][next_node]
                        self.gradient[cur_layer][cur_node] = self.diff_func(self.layer[cur_layer + 1][cur_node]) * sum
            # update weight
            for cur_layer in range(0, len(self.weight)):
                for cur_node in range(0, len(self.weight[cur_layer])):
                    for cur_wire in range(0, len(self.weight[cur_layer][cur_node])):
                        momentum = alpha * (self.weight[cur_layer][cur_node][cur_wire] - self.past_weight[cur_layer][cur_node][cur_wire])
                        delta_w = learningRate * self.gradient[cur_layer][cur_node] * self.layer[cur_layer][cur_wire]
                        # save w(t - 1)
                        self.past_weight[cur_layer][cur_node][cur_wire] = self.weight[cur_layer][cur_node][cur_wire]
                        self.weight[cur_layer][cur_node][cur_wire] += momentum + delta_w
            # update bias
            for cur_layer in range(0, len(self.bias)):
                for cur_node in range(0, len(self.bias[cur_layer])):
                    self.bias[cur_layer][cur_node] += learningRate * self.gradient[cur_layer][cur_node] * 1

            data_row += 1
        return

    # test neuron network
    def test(self, data_set):
        result = 0
        for cur_row in range(0, len(data_set)):
            for data_col in range(0, self.cut_row):
                self.layer[0][data_col] = data_set[cur_row][data_col]

            for cur_layer in range(1, len(self.layer)): # except layer 0 input layer
                for cur_node in range(0, len(self.layer[cur_layer])):
                    v = numpy.dot(self.layer[cur_layer - 1], self.weight[cur_layer - 1][cur_node])
                    v += self.bias[cur_layer - 1][cur_node]
                    self.layer[cur_layer][cur_node] = self.activation_func(v)

            print("\n")
            print(self.layer[len(self.layer) - 1])
            for cur_output in range(0, len(self.layer[len(self.layer) - 1])):
                if self.layer[len(self.layer) - 1][cur_output] > 0.5:
                    self.layer[len(self.layer) - 1][cur_output] = 1
                else:
                    self.layer[len(self.layer) - 1][cur_output] = 0
            print(self.layer[len(self.layer) - 1])
            design_output = []
            for data_col in range(self.cut_row, len(data_set[cur_row])):
                design_output.append(data_set[cur_row][data_col])
            print(design_output)
            if numpy.array_equal(design_output, self.layer[len(self.layer) - 1]):
                result += 1

        print("accuracy is {}".format(result * 100 / len(data_set)))
