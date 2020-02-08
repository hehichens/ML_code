'''
Adaboost 回归算法
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BiTree():
    '''
    用简单的最小二乘树做弱学习器
    '''
    def __init__(self, W):
        self.W = W

    def cal_error(self, X, y, split):

        mask_left = X < split
        mask_right = X >= split
        X_left = X[mask_left]
        X_right = X[mask_right]
        c_left = np.mean(y[mask_left])
        c_right = np.mean((y[mask_right]))

        error_left = np.dot(self.W[mask_left], np.abs((X_left - c_left)**2))
        error_right = np.dot(self.W[mask_right], np.abs(X_right - c_right)**2)
        return  error_right + error_left


    def fit(self, X, y):
        self.y = y
        self.X = X
        X_sorted = np.sort(X)
        split_list = (X_sorted[1:] + X_sorted[:-1]) / 2
        best_split = split_list[0]
        min_error = np.inf

        for split in split_list:
            error = self.cal_error(X, y, split)
            if error < min_error:
                min_error = error
                best_split = split

        self.split_index = best_split

    def predict(self, X):
        y_pred = []
        for x in X:
            if x < self.split_index:
                y_pred.append(np.mean(self.y[self.X < self.split_index])) # 在哪边用哪边的平均值
            else:
                y_pred.append(np.mean(self.y[self.X >= self.split_index]))
        return np.array(y_pred)


class AdaboostRegresiion():
    def __init__(self, max_itr=4):
        self.max_itr = max_itr

    def fit(self, X, y):
        self.rgs_list = []
        self.W = np.ones(len(X)) / len(X)

        for itr in range(self.max_itr):
            rgs = BiTree(self.W)
            rgs.fit(X, y)
            G_ = rgs.predict(X)
            SquaError = (y - G_)**2 / np.max((y - G_)**2)
            e_ = np.sum(self.W * SquaError)  # 误差值
            e_ = np.round(e_, 4)
            alpha_ = e_ / (1 - e_)
            alpha_ = np.round(alpha_, 4)
            Z_ = np.sum(self.W * alpha_**(1 - SquaError))

            self.W = self.W * alpha_**(1 - SquaError) / Z_
            self.W = np.round(self.W, 5)

            self.rgs_list.append([np.log(1 / alpha_), rgs])

        self.rgs_list = np.array(self.rgs_list)


    def predict(self, X):
        y_pred = 0
        for rgs in self.rgs_list:
            y_pred += rgs[0] * rgs[1].predict(X)
        return np.array(y_pred / len(self.rgs_list)) # 平均值代替预测值

def load_data(path_="./Input/data_8-2.txt"):

    df = pd.read_csv(path_)
    x = df["x"].values
    y = df["y"].values
    return x, y

if __name__ == "__main__":
    X, y = load_data()
    rgs = AdaboostRegresiion(max_itr=1)
    rgs.fit(X, y)
    X_test = np.arange(X.min(), X.max(), 0.1)
    y_pred = rgs.predict(X_test)
    # print(y_pred)
    plt.scatter(X, y, c='r')
    plt.plot(X_test, y_pred)
    plt.show()




