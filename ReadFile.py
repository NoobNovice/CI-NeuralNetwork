import xlrd
import random
class ReadFile:
    train_data = []
    test_data = []
    num_row = 0
    num_col = 0

    def __init__(self):
        return

    def excel(self, file_name):
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
            self.train_data.append(row)

    def cut_data(self, test_percent):
        for i in range(0, len(self.train_data)):
            rand = random.randrange(0, len(self.train_data))
            rand2 = random.randrange(len(self.train_data)/2, len(self.train_data))
            temp = self.train_data[i]
            temp2 = self.train_data[rand]
            temp3 = self.train_data[rand2]
            self.train_data[i] = temp2
            self.train_data[rand] = temp3
            self.train_data[rand2] = temp
        num_of_train = int((len(self.train_data) * test_percent) / 100)
        for i in range(0, num_of_train):
            self.test_data.append(self.train_data[i])
            self.train_data.pop(i)
        return
