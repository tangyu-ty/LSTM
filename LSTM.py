import os

import pandas_datareader.data as pd_data
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

def dataload():
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2021, 11, 18)
    df = pd_data.DataReader('GOOGL', 'stooq', start, end)
    return df


def Stock_Price_LSTM_Data_Precesing(df, mem_his_days, pre_days):
    df.dropna(inplace=True)  # 删除0
    df.sort_index(inplace=True)
    # inplace(原地)排序，根据之前传进来的日期
    df['label'] = df['Close'].shift(-pre_days)  # Close收盘价往上移pre_days个再加到新的label列作为标签
    scaler = StandardScaler()
    sca_X = scaler.fit_transform(df.iloc[:, :-1])  #

    deq = deque(maxlen=mem_his_days)
    # 队列的最大长度为记忆天数

    X = []
    for i in sca_X:
        list(i)
        deq.append(list(i))
        if len(deq) == mem_his_days:
            X.append(list(deq))
    # X的shape是(4330,mem_his_days,5);每个样本存的是(mem_his_days，5)的shape
    # 每次fori的时候，i是一个5的张量，然后每次X.append5个张量，
    # 也就是说mem_his_days天的数据*每天的5个特征
    X_latey = X[-pre_days:]
    # 少的是(0,men_his_days-1)个，序列未满时的值
    X = X[:-pre_days]
    # 删掉nan的值。
    # 得到纯粹的训练集
    y = df['label'][mem_his_days - 1:-pre_days]

    X = np.array(X)
    # 4330个样本，每个样本有(5天*每天5个特征)
    y = np.array(y)

    print(len(X))
    print(len(y))
    print(len(X_latey))
    return X, y, X_latey


def init_model(the_lstm_layers, the_dense_layers, the_units):
    model = Sequential()
    # 序列添加
    model.add(LSTM(the_units, input_shape=X.shape[1:], activation='relu', return_sequences=True))
    model.add(Dropout(0.1))

    for i in range(the_lstm_layers):
        model.add(LSTM(the_units, activation='relu', return_sequences=True))
        model.add(Dropout(0.1))

    model.add(LSTM(the_units, activation='relu'))
    model.add(Dropout(0.1))

    for i in range(the_dense_layers):
        model.add(Dense(the_units, activation='relu'))
        model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(optimizer="Adam", loss="mse", metrics=["mape"])
    # MeanAbsolutePercentageError 平均绝对百分比误差 metrics(度量)
    # MeanSquaredError 均方误差
    # LSTM，10个神经元
    return model



if __name__ == "__main__":
    df = dataload()

    # model = model()
    # model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

    # 多模型训练
    mem_days = [5, 10, 15]
    lstm_layers = [1, 2, 3, 4]
    dense_layers = [1, 2, 3]
    units = [8, 16, 32]
    pre_days = 10
    # 一共训练3*4*3*3个模型
    #os.removedirs('./models')
    for the_mem_days in mem_days:
        for the_lstm_layers in lstm_layers:
            for the_dense_layers in dense_layers:
                for the_units in units:
                    filepath = './models/{val_mape:.2f}_{epoch:02d}-' + f'mem_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_units_{the_units}'
                    checkpoint_callback = ModelCheckpoint(
                        filepath=filepath,
                        save_weights_only=False,
                        monitor='val_mape',
                        mode='min',
                        save_best_only=True)
                    X, y, X_latey = Stock_Price_LSTM_Data_Precesing(df, the_mem_days, pre_days)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)
                    model = init_model(the_lstm_layers, the_dense_layers, the_units)
                    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint_callback])
    best_model = load_model('./models/5.10_12-mem_5_lstm_3_dense_1_units_8')
    print(best_model.summary())

    with tf.Session() as sess:
        # 网络结构写入
        summary_writer = tf.summary.FileWriter('./log/', sess.graph)
        # summary_writer = tf.summary.FileWriter('./log/', tf.get_default_graph())




