import numpy as np
import sys
import random
import numpy.core.defchararray as npy_dy
import matplotlib.pyplot as plt
from scipy.stats import zscore


def perceptron_algorithm(x_arr, y_arr):
    iterations = 1000
    eta = 0.1
    weights_arr = np.zeros(3, 8)
    for i in range(iterations):
        arr_zip = list(zip(x_arr,y_arr))
        np.random.shuffle(arr_zip)
        x_arr, y_arr = zip(*arr_zip)
        for x, y in zip(x_arr, y_arr):
            # predict.
            y_hat = np.argmax(np.dot(weights_arr, x))
            # update
            if y != y_hat:
                weights_arr[y, :] = weights_arr[y, :] + eta * x
                weights_arr[y_hat, :] = weights_arr[y_hat, :] - eta * x


def loss_function_pa(weights_arr, y, y_hat, x):
    return max(0.0, 1 - np.dot(weights_arr[y], x) + np.dot(weights_arr[y_hat], x))


def pa_algorithm(x_arr, y_arr):
    iterations = 1000
    weights_arr = np.zeros(3, 8)
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
                loss /= ((np.power(np.linalg.norm(x, ord=2),2))*2)
                weights_arr[y, :] = weights_arr[y, :] + loss * x
                weights_arr[y_hat, :] = weights_arr[y_hat, :] - loss * x


def read_from_files(x_arr,y_arr):
    # go over the x training file and turn it into an array to append to the original array.
    with open(sys.argv[1]) as path:
        line = path.readline()
        while line:
            # split the line into a list by commas.
            temp = line.split(',')
            # change the letter to fit a certain number on a scale.
            if temp[0] == 'M':
                temp[0] = 0.25
            elif temp[0] == 'F':
                temp[0] = 0.50
            else:
                temp[0] = 0.75
            x_arr.append(temp)
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


def main():
    x_arr = []
    y_arr = []
    read_from_files(x_arr,y_arr)


if __name__ == "__main__":
    main()