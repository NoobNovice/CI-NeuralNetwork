import xlrd
import random
import math
import numpy

class ReadExcelFile:
    table_data = []
    fold_data = []
    num_row = 0
    num_col = 0

    min = 0
    max = 0
    mean = 0
    SD = 0
    j = 0
    def __init__(self, file_name):
        # read excel file
        workbook = xlrd.open_workbook(file_name)
        worksheet = workbook.sheets()

        self.num_row = worksheet[0].nrows
        self.num_col = worksheet[0].ncols

        for cur_row in range(1, self.num_row):
            row = []
            for cur_col in range(0, self.num_col):
                value = worksheet[0].cell_value(cur_row, cur_col)
                row.append(value)
            self.table_data.append(row)

        # find statistics value
        self.max = numpy.max(self.table_data)
        self.min = numpy.min(self.table_data)
        self.mean = numpy.average(self.table_data)
        self.SD = numpy.std(self.table_data)
        temp = self.max
        while True:
            temp = temp / 10
            self.j += 1
            if temp < 1:
                break

        # transform date
        for cur_row in range(0, len(self.table_data)):
            for cur_col in range(0, len(self.table_data[cur_row])):
                self.table_data[cur_row][cur_col] = self.decimal_scaling(self.table_data[cur_row][cur_col])

        # random data
        for i in range(0, len(self.table_data)):
            rand = random.randrange(0, len(self.table_data) - 1)
            temp = self.table_data[i]
            self.table_data[i] = self.table_data[rand]
            self.table_data[rand] = temp
        return

    def ten_fold_data(self):
        # cut to test set
        num_of_train = math.ceil((len(self.table_data) * 10) / 100)

        for i in range(0, 9):
            temp_arr = []
            for j in range(0, num_of_train):
                temp_arr.append(self.table_data.pop(random.randrange(0, len(self.table_data) - 1)))
            self.fold_data.append(temp_arr)
        self.fold_data.append(self.table_data)
        return

    def minmax_normalize(self, v, new_min, new_max):
        return (v - self.min)/(self.max - self.min)*(new_max - new_min) + new_min

    def z_score(self, v):
        return (v - self.mean)/self.SD

    def decimal_scaling(self, v):
        return v/math.pow(10, self.j)
