import numpy as np
import math
import torch
import pickle

def special_normalize(filename, size):
    save = open("means.npy", 'rb')
    means = np.load(save)
    save.close()

    save = open("stdevs.npy", 'rb')
    stdevs = np.load(save)
    save.close()

    file = open(filename, 'r')
    #data = np.zeros(shape=(size, 121))
    data = []

    row_num = 0
    ans = -1
    for row in file:
        if(row_num >= size):
            break
        values1 = row.split(',')
        values = []
        for i in range(121):
            if i == 0:
                try:
                    ans = int(values1[i])
                except:
                    ans = 0
            else:
                values.append((float(values1[i]) - means[i - 1]) / stdevs[i - 1])
        a, b, c, d = create_hyper_params(values)
        all = []
        for i in range(4):
            all.append(a[i])
            all.append(b[i])
            all.append(c[i])
            all.append(d[i])
        all = list(map(float, all))
        data.append((torch.tensor(all), ans))
        row_num += 1

    return data
    #[([0,...,15),(1,0)), ...., ()]

def createData(file_name, size):
    # mean and stdev
    means = np.zeros(shape=(120))
    stdevs = np.zeros(shape=(120))
    data = np.zeros(shape=(size, 120))
    file = open(file_name, 'r')
    for line in file:
        values1 = line.split(',')
        values = np.zeros(shape=(120))
        for i in range(1, 121):
            values[i - 1] = float(values1[i])
        for i in range(120):
            means[i] = means[i] + values[i]
            stdevs[i] += values[i] * values[i]
    for i in range(120):
        means[i] /= size
        stdevs[i] = math.sqrt(stdevs[i] / size - means[i] * means[i])

    ## Z score
    line_num = 0
    for line in file:
        values = line.split(',')
        values = np.array(values)
        for i in range(120):
            data[line_num][i] = (values[i] - means[i]) / stdevs[i]
        line_num += 1
    file.close()

    # pickle means and stdevs
    save = open("means.npy", 'wb')
    np.save(save, means)
    save.close()

    save = open("stdevs.npy", 'wb')
    np.save(save, stdevs)
    save.close()


def create_hyper_params(row):
    # create params
    param_avg = np.zeros(shape=(4))
    param_stdev = np.zeros(shape=(4))
    diff_avg = np.zeros(shape=(4))
    diff_stdev = np.zeros(shape=(4))
    for cell in range(120):
        param_avg[cell % 4] += row[cell]
        param_stdev[cell % 4] += row[cell] * row[cell]
        if cell > 3:
            diff_avg[cell % 4] += row[cell] - row[cell - 4]
            diff_stdev[cell % 4] += (row[cell] - row[cell - 4]) * (row[cell] - row[cell - 4])

    for i in range(4):
        param_avg[i] /= 30
        param_stdev[i] = math.sqrt(param_stdev[i]/30 - param_avg[i] * param_avg[i])
        diff_avg[i] /= 29
        diff_stdev[i] = math.sqrt(diff_stdev[i]/29 - diff_avg[i] * diff_avg[i])

    return param_avg, param_stdev, diff_avg, diff_stdev


def main():
    createData("train.csv", 700000)


if __name__ == "__main__":
    main()