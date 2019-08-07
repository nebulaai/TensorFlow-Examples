import numpy as np
import pandas as pd
from pandas import DataFrame, concat
import requests
import time


def data_handler():
    FETCH_URL = "https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=300"  # unix timestamp
    end_time = int(time.time())
    start_time = end_time - 20002 * 300
    url = FETCH_URL % ("USDT_ETH", start_time, end_time)
    # print(url)
    # print(start_time)
    # print("\n",end_time)
    # print("\nDelta = ",end_time - start_time)
    response = requests.get(url)
    response_text = response.text
    return response_text


def download_data():
    data_in_json = data_handler()
    data = pd.read_json(data_in_json, convert_dates=False)
    time = data['date']
    # # pandas series tolist, return list of time at each collection
    # time = time.tolist()
    # x, date = [], []
    # for tm in time:
    #     date.append(datetime.datetime.fromtimestamp(tm).strftime('%Y-%m-%d %H:%M'))
    # for i in range(len(time)):
    #     x.append(i)
    # print(date)

    date = np.transpose(time)
    open = data["open"].values
    close = data['close'].values
    high = data["high"].values
    low = data["low"].values
    d = {"open": open, 'close': close, "high": high, "low": low}

    data = pd.DataFrame(data=d, index=date)

    data.drop(columns=["open", "close"], axis=1, inplace=True)

    # print('2')
    # print(data.shape)
    # print(data.tail())
    return data

def difference(dataset, interval=1):
    diff = list()

    n_size = dataset.shape[0]
    n_feature = dataset.shape[1]
    for i in range(interval, n_size):
        if n_feature == 1:
            value = dataset[i] - dataset[i - interval]
        else:
            value = dataset[i, :] - dataset[(i - interval), :]

        value = value.reshape(1, value.shape[0])
        diff.append(value)

    diff = np.array(diff)
    diff = diff.reshape(diff.shape[0], diff.shape[2])
    return diff


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg