# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import metrics
"""
多元线性回归需要对各变量进行标准化，因为在求系数wj梯度时，每个样本计算值与其标签值的差要与每个样本对应的第j个属性值相乘，然后求和
因此，如果属性值之间的差异太大，会造成系数无法收敛
"""

#  最小二乘法直接求解权重系数
def least_square(train_x, train_y):
    """
    input：训练数据（样本*属性）和标签
    """
    weights = (train_x.T * train_x).I * train_x.T * train_y
    return weights


# 梯度下降算法
def gradient_descent(train_x, train_y, maxCycle, alpha):
    numSamples, numFeatures = np.shape(train_x)
    weights = np.zeros((numFeatures, 1))

    for i in range(maxCycle):
        h = train_x * weights
        err = h - train_y
        weights = weights - (alpha * err.T * train_x).T
    return weights


def stochastic_gradient_descent(train_x, train_y, maxCycle, alpha):
    numSamples, numFeatures = np.shape(train_x)
    weights = np.zeros((numFeatures, 1))

    for i in range(maxCycle):
        for j in range(numSamples):
            h = train_x[j, :] * weights
            err = h - train_y[j, 0]
            weights = weights - (alpha * err.T * train_x[j, :]).T

    return weights


def load_data():
    df = pd.read_csv('C:\\Users\\subshine\\Desktop\\train1.csv', sep=',', header=0, index_col=0, encoding='ANSI')
    label = df['totals.transactionRevenue']
    data = df.drop('totals.transactionRevenue', axis=1)
    #data = np.array(df)

    return data, label


def show_results(predict_y, test_y):
    plt.scatter(np.array(test_y), np.array(predict_y), marker='x', s=30, c='red')  # 画图的数据需要是数组而不能是矩阵
    plt.plot(np.arange(0, 50), np.arange(0, 50))
    plt.xlabel("original_label")
    plt.ylabel("predict_label")
    plt.title("LinerRegression")
    plt.show()


if __name__ == "__main__":
    data, label = load_data()

    data = preprocessing.normalize(data.T).T

    train_x, test_x, train_y, test_y = train_test_split(data, label, train_size=0.75, random_state=33)

    linreg = LinearRegression()
    linreg.fit(train_x, train_y)
    y_pred = linreg.predict(test_x)
    rmse = np.sqrt(metrics.mean_squared_error(test_y, y_pred))
    print(rmse)
    '''
    train_x = np.mat(train_x)
    test_x = np.mat(test_x)
    train_y = np.mat(train_y).T  # (3,)转为矩阵变为行向量了,需要转置
    test_y = np.mat(test_y).T
    
    #     weights = least_square(train_x, train_y)
    #     predict_y = test_x * weights
    #     show_results(predict_y, test_y)
    #
    weights = gradient_descent(train_x, train_y, 1000, 0.01)
    predict_y = test_x * weights
    show_results(predict_y, test_y)

#     weights = stochastic_gradient_descent(train_x, train_y, 100, 0.01)
#     predict_y = test_x * weights
#     show_results(predict_y, test_y)
    '''