# Idan Twito, 311125249
# Roy Hevrony, 312498272

import numpy as np
import sys
from scipy.stats import zscore


# normalization of type Zero_Score as we learned in class.
def zero_score_norm(x_arr):
    for i in range(0, len(x_arr)):
        x_arr[i] = zscore(x_arr[i])
    return x_arr


# runs the algorithm with the test set and the weights that the training algorithm returned.
# adds each prediction to an array and finally returns the array when finished prediction.
def predict(test_x, w):
    prediction_results = []
    test_length = len(test_x)
    for i in range(0, test_length):
        y_hat = np.argmax(np.dot(w, test_x[i]))
        prediction_results.append(y_hat)
    return prediction_results


# second normalization technique learned in class.
def normalization(x_arr):
    transposed_x_arr = np.transpose(x_arr)
    temp_arr = []
    for row in transposed_x_arr:
        max_in_col = max(row)
        min_in_col = min(row)
        if max_in_col - min_in_col != 0:
            temp_arr.append((row - min_in_col) / (max_in_col - min_in_col))
        else:
            continue
    return np.transpose(temp_arr)


# SVM algorithm.
def svm(x_arr, y_arr):
    iterations = 10
    eta = 0.01
    var_lambda = 0.5
    weights_arr = np.zeros((3, 8))
    # go over 10 iterations.
    for i in range(iterations):
        # make sure that each X and Y are in the same place when shuffling.
        arr_zip = list(zip(x_arr, y_arr))
        np.random.shuffle(arr_zip)
        x_arr, y_arr = zip(*arr_zip)
        # go over each pair x, y in the zipped arrays.
        for x, y in zip(x_arr, y_arr):
            # predict.
            y_hat = np.argmax(np.dot(weights_arr, x))
            # update
            if y != y_hat:
                for matrix_line_no in range(weights_arr.shape[0]):
                    if matrix_line_no == y:
                        weights_arr[y, :] = np.multiply((1 - (var_lambda * eta)),
                                                        weights_arr[y, :]) + np.multiply(eta, x)
                    elif matrix_line_no == y_hat:
                        weights_arr[y_hat, :] = np.multiply((1 - (var_lambda * eta)),
                                                            weights_arr[y_hat, :]) - np.multiply(
                            eta, x)
                    else:
                        weights_arr[matrix_line_no, :] = np.multiply((1 - (var_lambda * eta)),
                                                                     weights_arr[matrix_line_no, :])
            else:
                weights_arr[0, :] = np.multiply((1 - (var_lambda * eta)), weights_arr[0, :])
                weights_arr[1, :] = np.multiply((1 - (var_lambda * eta)), weights_arr[1, :])
                weights_arr[2, :] = np.multiply((1 - (var_lambda * eta)), weights_arr[2, :])
        # divide the ETA by the iteration number each iteration.
        if i > 0:
            eta /= i
    return weights_arr


# Perceptron algorithm.
def perceptron_algorithm(x_arr, y_arr):
    iterations = 10
    eta = 0.1
    # initialize the weights array with zeros.
    weights_arr = np.zeros((3, 8))
    # go over 10 iterations to train the algorithm.
    for i in range(iterations):
        # zip the arrays together so that in shuffle mode it won't change.
        arr_zip = list(zip(x_arr, y_arr))
        np.random.shuffle(arr_zip)
        x_arr, y_arr = zip(*arr_zip)
        # go over each pair of x,y in the zipped arrays.
        for x, y in zip(x_arr, y_arr):
            # predict - returns the most probable label
            y_hat = np.argmax(np.dot(weights_arr, x))
            # update
            if y != y_hat:
                weights_arr[y, :] = weights_arr[y, :] + np.multiply(eta, x)
                weights_arr[y_hat, :] = weights_arr[y_hat, :] - np.multiply(eta, x)
        # divide the ETA by the iteration number each iteration.
        if i > 0:
            eta /= i
    return weights_arr


# calculate the loss of the PA algorithm.
def loss_function_pa(weights_arr, y, y_hat, x):
    return max(0.0, 1 - np.dot(weights_arr[y], x) + np.dot(weights_arr[y_hat], x))


# Passive Aggressive algorithm.
def pa_algorithm(x_arr, y_arr):
    iterations = 10
    loss_counter = 0
    # initialize the weights array to zero.
    weights_arr = np.zeros((3, 8))
    temp_weights = weights_arr
    # go over 10 iterations.
    for i in range(iterations):
        # zip the two arrays together to not lose the places of the training set and its prediction.
        arr_zip = list(zip(x_arr, y_arr))
        np.random.shuffle(arr_zip)
        x_arr, y_arr = zip(*arr_zip)
        # go over each pair in the zipped arrays.
        for x, y in zip(x_arr, y_arr):
            # predict.
            y_hat = np.argmax(np.dot(weights_arr, x))
            # update
            if y != y_hat:
                loss = loss_function_pa(weights_arr, y, y_hat, x)
                divide_by = ((np.power(np.linalg.norm(x, ord=2), 2)) * 2)
                # to rule out division by zero.
                if divide_by != 0:
                    loss_counter += 1
                    tau = loss / divide_by
                    weights_arr[y, :] = weights_arr[y, :] + np.multiply(tau, x)
                    weights_arr[y_hat, :] = weights_arr[y_hat, :] - np.multiply(tau, x)
                    # in order to calculate the average and return it.
                    temp_weights = np.add(temp_weights, weights_arr)
    # calculate the average in case that the loss_counter is not zero.
    if loss_counter != 0:
        return temp_weights / loss_counter
    else:
        return weights_arr


# create 3 arrays, each one from the file that was given in the command line.
# returns the number of items in the training set array.
def read_from_files(x_arr, y_arr, test_arr):
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

    with open(sys.argv[3]) as path3:
        line = path3.readline()
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
            test_arr.append(list(map(float, temp)))
            line = path3.readline()
    return counter


# splits the array.
def array_splitter(start_index, end_index, array):
    temp_arr = []
    for row in range(start_index, end_index):
        temp_arr.append(array[row])
    return temp_arr


# print the predictions of each algo.
def print_predictions(perceptron_arr, svm_arr, pa_arr):
    for i in range(len(perceptron_arr)):
        print("perceptron: {0}, svm: {1}, pa: {2}".format((perceptron_arr[i]), (svm_arr[i]),
                                                          (pa_arr[i])))


# main function.
def main():
    x_arr = []
    y_arr = []
    testing_set_arr = []
    # read from the files.
    read_from_files(x_arr, y_arr, testing_set_arr)
    # zip the arrays.
    y_arr = list(map(int, y_arr))
    arr_zipped = list(zip(x_arr, y_arr))
    np.random.shuffle(arr_zipped)
    # turn it into an numpy array.
    x_arr = np.array(x_arr)
    # calculate the weights for each algorithm.
    w = perceptron_algorithm(x_arr, y_arr)
    w2 = svm(x_arr, y_arr)
    w3 = pa_algorithm(x_arr, y_arr)
    # calculate the predictions from each algorithm using the weights from the training.
    perceptron_arr = predict(testing_set_arr, w)
    svm_arr = predict(testing_set_arr, w2)
    pa_arr = predict(testing_set_arr, w3)
    # print the results.
    print_predictions(perceptron_arr, svm_arr, pa_arr)


if __name__ == "__main__":
    main()
