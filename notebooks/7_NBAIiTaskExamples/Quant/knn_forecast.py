# import requests
# import time, datetime
# from matplotlib import pyplot
import numpy as np


# import pandas as pd
# from . import data_helper
# import data_helper
# from pandas import Series, concat, read_csv, datetime, DataFrame
# from sklearn.preprocessing import MinMaxScaler


def canberra_distance(train_X, test_x):
    train_size = train_X.shape[0]
    features = train_X.shape[1]
    temp_test_x = np.repeat(test_x, train_size, axis=0)
    distance = np.abs(train_X - temp_test_x) / np.add(np.abs(train_X), np.abs(temp_test_x))
    # print('1', distance.shape)
    distance = np.sum(distance, axis=1)  # (train_size, )
    return distance


class KNN():
    def __init__(self, train_X, train_Y, test_X, k=11):
        self.k = k
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X

    def canberra_distance(self, train_X, test_x):
        train_size = train_X.shape[0]
        features = train_X.shape[1]
        temp_test_x = np.repeat(test_x, train_size, axis=0)
        distance = np.abs(train_X - temp_test_x) / np.add(np.abs(train_X), np.abs(temp_test_x))
        distance_matrix = np.sum(distance, axis=1)
        return distance_matrix
        # output: (train_size, )

    def knn_predict_1test(self, knn_matrix):
        # knn_matrix: k * y_feature
        pred = np.mean(knn_matrix, axis=0)
        pred = pred.reshape(1, pred.shape[0])
        # pred: 1 * y_feature
        return pred

    def knn_predict(self):
        predictions = []
        test_size = self.test_X.shape[0]
        # print('dev size', test_size)
        for i in range(test_size):  # for each test sample
            dis_array = self.canberra_distance(self.train_X, self.test_X[i:i + 1, :])  # test_x[i:i+1,:] (1,8)

            index_sort_dis = np.argsort(dis_array)
            # sorted_dis = np.sort(dis_array)
            knn_matrix = self.train_Y[index_sort_dis[0:self.k], :]

            prediction = self.knn_predict_1test(knn_matrix)
            # print('pred shape', self.prediction.shape) #(1, 8)
            predictions.append(prediction)

        predictions = np.array(predictions)
        if test_size == 1:
            predictions = np.reshape(predictions, [1, -1])
        else:
            predictions = np.reshape(predictions, [predictions.shape[0], predictions.shape[2]])

        # print('knn pred:', predictions.shape)
        return predictions
