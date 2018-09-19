from NeuronNetwork import *
from ReadFile import *

# test read data
file_data = ReadFile()
file_data.excel("iris.xls")
file_data.cut_data(10)
print(len(file_data.train_data))
print(len(file_data.test_data))

arr = [[1, 1, 0, 1]]
# test neuron network
model = NeuronNetwork(4, 3, [3, 2])
model.set_value()
temp_weight = copy.deepcopy(model.weight)
temp_bias = copy.deepcopy(model.bias)
print(model.bias)
model.train(file_data.train_data, 4, 0.2, 0.001, 50000)
model.test(file_data.test_data)

