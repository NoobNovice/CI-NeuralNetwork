from NeuronNetwork import *
from ReadExcelFile import *
import copy
import numpy

file_data = ReadExcelFile("flood.xls")
file_data.ten_fold_data()

result_file = open("CrossValidateResult.txt", "w")
model = NeuronNetwork(8, 1, [7, 6, 5, 4, 3, 2])
# 10 cross validation
for i in range(0, 10):
    fold_data = copy.deepcopy(file_data.fold_data)
    test_data = fold_data.pop(i)
    train_data = numpy.concatenate(fold_data)
    model.set_value()
    model.train(train_data, 8, 0.2, 0.001, pow(10, -7), 1000)
    result_file.write("fold {} is {}\n".format(i + 1, round(model.test_mean_square_error(test_data), 3)))


