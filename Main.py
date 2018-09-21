from NeuronNetwork import *
from ReadExcelFile import *
import copy
import numpy

# flood prediction
# file_data = ReadExcelFile("flood.xls")
# file_data.decimal_scaling()
# file_data.ten_fold_data()
#
# result_file = open("CrossValidateResult1.txt", "w")
# model = NeuronNetwork(8, 1, [7, 6, 5, 4, 3, 2])
# # 10 cross validation
# for i in range(0, 10):
#     fold_data = copy.deepcopy(file_data.fold_data)
#     test_data = fold_data.pop(i)
#     train_data = numpy.concatenate(fold_data)
#     model.set_value()
#     model.train(train_data, 8, 0.2, 0.001, pow(10, -7), 1000)
#     result_file.write("fold {} is {}\n".format(i + 1, round(model.test_mean_square_error(test_data), 3)))

# cross validation
file_data2 = ReadExcelFile("cross.xls")
file_data2.ten_fold_data()

result_file = open("CrossValidateResult2.txt", "w")
model = NeuronNetwork(2, 2, [2])
# 10 cross validation
roundRun = []
for i in range(0, 10):
    fold_data = copy.deepcopy(file_data2.fold_data)
    test_data = fold_data.pop(i)
    train_data = numpy.concatenate(fold_data)
    model.set_value()
    roundRun.append(model.train(train_data, 2, 0.15, 0.25, pow(10, -6), 5000))
    result_file.write("fold {} is {}\n".format(i + 1, model.test_classification(test_data)))
print(roundRun)


