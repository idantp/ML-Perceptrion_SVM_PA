import numpy as np
import sys
from scipy import stats
import random
import matplotlib.pyplot as plt
from scipy.stats import zscore


# TODO: normalization z-score ?

def normalization(x_arr):
    transposed_x_arr = np.transpose(x_arr)
    for row in transposed_x_arr:
        max_in_col = max(row)
        min_in_col = min(row)
        if (max_in_col - min_in_col != 0):
            row = (row - min_in_col) / (max_in_col - min_in_col)
        else:
            continue
    return np.transpose(transposed_x_arr)


def svm(x_arr, y_arr):
    iterations = 10
    eta = 0.01
    var_lambda = 0.3
    weights_arr = np.zeros((3, 8))
    for i in range(iterations):
        arr_zip = list(zip(x_arr, y_arr))
        np.random.shuffle(arr_zip)
        x_arr, y_arr = zip(*arr_zip)
        for x, y in zip(x_arr, y_arr):
            # predict.
            y_hat = np.argmax(np.dot(weights_arr, x))
            # update
            if y != y_hat:
                for matrix_line_no in range(weights_arr.shape[0]):
                    if matrix_line_no == y:
                        weights_arr[y, :] = (1 - (var_lambda * eta)) * weights_arr[y,
                                                                       :] + np.multiply(eta, x)
                    elif matrix_line_no == y_hat:
                        weights_arr[y_hat, :] = (1 - (var_lambda * eta)) * weights_arr[y_hat,
                                                                           :] + np.multiply(eta, x)
                    else:
                        weights_arr[i, :] = (1 - (var_lambda * eta)) * weights_arr[i,
                                                                       :] + np.multiply(eta, x)
    return weights_arr


def perceptron_algorithm(x_arr, y_arr):
    iterations = 10
    eta = 0.1
    weights_arr = np.zeros((3, 8))
    for i in range(iterations):
        arr_zip = list(zip(x_arr, y_arr))
        np.random.shuffle(arr_zip)
        x_arr, y_arr = zip(*arr_zip)
        for x, y in zip(x_arr, y_arr):
            # predict.
            y_hat = np.argmax(np.dot(weights_arr, x))
            # update
            if y != y_hat:
                weights_arr[y, :] = weights_arr[y, :] + np.multiply(eta, x)
                weights_arr[y_hat, :] = weights_arr[y_hat, :] - np.multiply(eta, x)
    print(weights_arr)
    return weights_arr


def loss_function_pa(weights_arr, y, y_hat, x):
    return max(0.0, 1 - np.dot(weights_arr[y], x) + np.dot(weights_arr[y_hat], x))


def pa_algorithm(x_arr, y_arr):
    iterations = 10
    weights_arr = np.zeros((3, 8))
    for i in range(iterations):
        arr_zip = list(zip(x_arr, y_arr))
        np.random.shuffle(arr_zip)
        x_arr, y_arr = zip(*arr_zip)
        for x, y in zip(x_arr, y_arr):
            # predict.
            y_hat = np.argmax(np.dot(weights_arr, x))
            # update
            if y != y_hat:
                loss = loss_function_pa(weights_arr, y, y_hat, x)
                divide_by = ((np.power(np.linalg.norm(x, ord=2), 2)) * 2)
                if divide_by != 0:
                    loss /= divide_by
                    weights_arr[y, :] = weights_arr[y, :] + np.multiply(loss, x)
                    weights_arr[y_hat, :] = weights_arr[y_hat, :] - np.multiply(loss, x)
    return weights_arr


def read_from_files(x_arr, y_arr):
    # go over the x training file and turn it into an array to append to the original array.
    counter = 0
    with open(sys.argv[1]) as path:
        line = path.readline()
        while line:
            counter += 1
            # split the line into a list by commas.
            temp = line.split(',')
            # change the letter to fit a certain number on a scale.
            if temp[0] == 'M':
                temp[0] = 0.25
            elif temp[0] == 'F':
                temp[0] = 0.50
            else:
                temp[0] = 0.75
            x_arr.append(list(map(float, temp)))
            line = path.readline()
    # read from the Y file and put either 0 1 or 2 into the array.
    with open(sys.argv[2]) as path2:
        line = path2.readline()
        while line:
            if line[0] == '0':
                y_arr.append(0)
            elif line[0] == '1':
                y_arr.append(1)
            else:
                y_arr.append(2)
            line = path2.readline()
    return counter

def array_splitter(start_index,end_index,array):
    temp_arr = []
    for row in range(start_index,end_index):
        temp_arr.append(array[row])
    return temp_arr

def main():
    x_arr = []
    y_arr = []
    sets_amount = read_from_files(x_arr, y_arr)
    training_sets_amount = int((sets_amount * 5) / 6)
    testing_sets_index = training_sets_amount + 1
    y_arr = list(map(int, y_arr))
    x_arr = normalization(x_arr)
    arr_zipped = list(zip(x_arr, y_arr))
    np.random.shuffle(arr_zipped)
    x_training_sets = array_splitter(0,training_sets_amount,x_arr)
    x_testing_sets = array_splitter(testing_sets_index,(sets_amount+1),x_arr)
    perceptron_algorithm(x_training_sets, y_arr)


if __name__ == "__main__":
    main()
