# from sklearn.preprocessing import MinMaxScaler
# import multiprocessing
import multiprocessing

# import knn_forecast
import numpy as np
from matplotlib import pyplot
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

import data_helper
import knn_forecast

n_lag = 4
n_seq = 4
n_feature = 2
n_xfeature = n_feature * n_lag
n_yfeature = n_feature * n_seq

dev_size = 10000


def get_RSME(actual, predict):  # (?,1)
    MSE = np.sum(np.square(actual - predict))
    RMSE = np.sqrt(MSE)
    result = np.sum(RMSE)
    return result


class STACK_ENSEMBLE():
    def __init__(self, n_lag=4, n_seq=4, n_feature=2):
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.n_feature = n_feature
        self.n_xfeature = self.n_lag * self.n_feature
        self.n_yfeature = self.n_seq * self.n_feature

    def build_datasets(self):
        series = data_helper.download_data()
        series.drop(series.index[:-20000], inplace=True)

        orig_data = series.values

        diff_data = data_helper.difference(orig_data, 1)

        # min max scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(diff_data)

        supervised = data_helper.series_to_supervised(scaled_data, n_lag, n_seq, dropnan=True)

        train, dev = supervised.values[0:-dev_size, :], supervised.values[-dev_size:, :]

        return series, scaler, train, dev

    def GB_predict(self, train_X, train_Y, test_X):
        model_GB = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100), n_jobs=1)
        model_GB.fit(train_X, train_Y)
        predict_GB = model_GB.predict(test_X)
        return predict_GB

    def KNN_predict(self, train_X, train_Y, test_X):
        knn_model = knn_forecast.KNN(train_X, train_Y, test_X)
        prediction_knn = knn_model.knn_predict()
        return prediction_knn

    def inverse_transform(self, series, forecast, scaler, n_test):
        # forecase = (1, 8)
        forecast = np.array(forecast)
        stack_forecast = forecast.reshape(-1, self.n_feature)
        invert_forecast = scaler.inverse_transform(stack_forecast)

    def main_execute(self):
        n_test = 1
        series, scaler, train, dev = self.build_datasets()

        train_X, train_Y = train[:, :n_xfeature], train[:, -n_yfeature:]
        dev_X, dev_Y = dev[:-3, :n_xfeature], dev[:-3, -n_yfeature:]

        test_X_1 = dev_X[-1, :]  # (8, )
        test_X_1 = test_X_1.reshape(1, -1)  # last dev_X
        test_Y_1 = dev_Y[-1, :]  # (8, )
        test_Y_1 = test_Y_1.reshape(1, -1)  # last dev_Y

        function_list = [self.GB_predict, self.KNN_predict, self.GB_predict, self.KNN_predict]
        args_list = [(train_X, train_Y, dev_X), (train_X, train_Y, dev_X),
                     (train_X, train_Y, test_Y_1), (train_X, train_Y, test_Y_1)]

        results = []
        pool = multiprocessing.Pool(4)
        for (func, args) in zip(function_list, args_list):
            result = pool.apply_async(func, args=args)
            results.append(result)
        pool.close()
        pool.join()

        predict_dev_GB = results[0].get()
        predict_dev_knn = results[1].get()
        predict_test_GB = results[2].get()
        predict_test_knn = results[3].get()
        # print('predict_dev_GB', predict_dev_GB.shape)
        # print('predict_test_GB', predict_test_GB.shape)
        # # predictions for dev
        # predict_dev_GB = self.GB_predict(train_X, train_Y, dev_X)  #(10000, 8)
        # predict_dev_knn = self.KNN_predict(train_X, train_Y, dev_X) #(10000, 8)
        #
        # # predcitons for test
        # predict_test_GB = self.GB_predict(train_X, train_Y, test_Y_1)  # (1,8)
        # predict_test_knn = self.KNN_predict(train_X, train_Y, test_Y_1) #(1,8)

        # stack predictions of dev, test separately
        stack_dev_predict = np.concatenate((predict_dev_knn, predict_dev_GB), axis=1)
        stack_test_predict = np.concatenate((predict_test_knn, predict_test_GB), axis=1)

        predict_stack = self.GB_predict(stack_dev_predict, dev_Y, stack_test_predict)  # (1,8)
        # print('predict_stack',predict_stack.shape)

        temp_predict_stack = predict_stack.reshape(-1, self.n_feature)
        invert_predict_stack = scaler.inverse_transform(temp_predict_stack)
        invert_predict_stack = np.add(invert_predict_stack, series.values[-temp_predict_stack.shape[0]:, :])

        max1 = np.max(invert_predict_stack, axis=1)
        max1 = max1.reshape(-1, 1)

        min1 = np.min(invert_predict_stack, axis=1)
        min1 = min1.reshape(-1, 1)

        invert_predict_stack = np.concatenate((max1, min1), axis=1)
        # test_Y_1 = test_Y_1.reshape(-1, self.n_feature)
        # invert_test_Y_1  = scaler.inverse_transform(test_Y_1)
        # invert_test_Y_1 = np.add(invert_test_Y_1, series.values[-test_Y_1.shape[0]-4:-4, :])

        #
        # test_X_1 = test_X_1.reshape(-1, self.n_feature)
        # invert_test_X_1 = scaler.inverse_transform(test_X_1)
        # invert_test_X_1 = np.add(invert_test_X_1, series.values[-test_X_1.shape[0] -8 :-8, :])
        # print('invert_test_X_1')
        # print(invert_test_X_1)

        # print('invert_predict_stack')
        # print(invert_predict_stack)

        # print('invert_test_Y_1')
        # print(invert_test_Y_1)
        #
        # print('orig_data')
        # print(series.values[-10:, :])
        #

        timestamp_last = series.index.values[-1]

        forecasts = []
        # forecasts = (dict, dict, dict, dict)
        # print('json forecast')
        for i in range(4):
            dic = dict()
            dic['effect_time'] = np.asscalar(timestamp_last + 1 + i * 300)
            dic['expire_time'] = np.asscalar(timestamp_last + (i + 1) * 300)
            dic['high'] = np.asscalar(invert_predict_stack[i, 0])
            dic['low'] = np.asscalar(invert_predict_stack[i, 1])
            dic['median'] = (dic['low'] + dic['high']) / 2
            dic['time_range'] = 300
            forecasts.append(dic)

        pyplot.figure()
        x1 = np.arange(1, 31, 1)
        len1 = x1.shape[0]
        x2 = np.arange(31, 35, 1)
        pyplot.plot(x1, series.values[-len1:, 0], color='g', marker='*', label='actual high')
        pyplot.scatter(x2, invert_predict_stack[:, 0], color='r', marker='*', label='pred high')

        pyplot.plot(x1, series.values[-len1:, 1], color='b', marker='*', label='actual low')
        pyplot.scatter(x2, invert_predict_stack[:, 1], color='b', marker='*', label='pred low')

        pyplot.legend()
        # pyplot.savefig("Results/fig.png")
        # pyplot.show()

        return forecasts, pyplot

        # pyplot.figure()
        # x1=np.arange(1,31,1)
        # len1 = x1.shape[0]
        # x2=np.arange(31,35,1)
        # pyplot.plot(x1, series.values[-len1:, 0], color='g', marker='*', label='actual high')
        # pyplot.scatter(x2, invert_predict_stack[:,0], color='r', marker='*', label='pred high')
        #
        # pyplot.plot(x1, series.values[-len1:, 1], color='b', marker='*', label='actual low')
        # pyplot.scatter(x2, invert_predict_stack[:, 1], color='b', marker='*', label='pred low')
        #
        # # pyplot.legend()
        # pyplot.show()
