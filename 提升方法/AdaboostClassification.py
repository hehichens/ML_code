'''
Adaboost 分类算法
'''

import numpy as np
import pandas as pd

def clf_great_than_(x_, v_):
    """
    weak learner

    :param x_:
    :param v_: threshold
    :return: classify results
    """
    y_ = np.zeros(x_.size, dtype=int)
    y_[x_ > v_] = 1
    y_[x_ < v_] = -1
    return y_


def clf_less_than_(x_, v_):
    """
    weak learner
    :param x_:
    :param v_: threshold
    :return: classify results
    """
    y_ = np.zeros(x_.size, dtype=int)
    y_[x_ < v_] = 1
    y_[x_ > v_] = -1
    return y_

class BiSection():
    def __init__(self, func_list, W):
        '''

        :param func_list: 待选的弱分类器列表
        :param W: 权重
        '''
        self.func = None
        self.split_index = None
        self.func_list = func_list
        self.W = W

    def fit(self, X, y):
        split_list = np.arange(X.min()-0.5, X.max()+0.5)
        min_error = np.inf
        for func in self.func_list:
            for v in split_list:
                y_pred = func(X, v)
                error = np.sum(self.W[y != y_pred])
                if error < min_error:
                    min_error = error
                    self.func = func
                    self.split_index = v

    def predict(self, X):
        return self.func(X, self.split_index)


class AdaboostClassification():
    def __init__(self, max_itr=10, func_list=[clf_great_than_, clf_less_than_]):
        self.max_itr = max_itr
        self.func_list = func_list


    def fit(self, X, y):
        self.clf_list = []
        self.W = np.ones(len(X)) / len(X)

        for itr in range(self.max_itr):
            clf = BiSection(self.func_list, self.W)
            clf.fit(X, y)
            G_ = clf.predict(X)

            e_ = np.sum(self.W[G_ != y]) # 误差值
            e_ = np.round(e_, 4)
            alpha_ = np.log((1 - e_) / e_) / 2
            alpha_ = np.round(alpha_, 4)
            self.W = self.W * np.exp(-alpha_ * y * G_) / np.sum(self.W * np.exp(-alpha_ * y * G_))
            self.W = np.round(self.W, 5)

            self.clf_list.append([alpha_, clf])


        self.clf_list = np.array(self.clf_list)

    def predcit(self, X):
        y_pred = 0
        for clf in self.clf_list:
            y_pred += clf[0] * clf[1].predict(X)
        return np.sign(y_pred)


def load_data(path_="./Input/data_8-1.txt"):

    df = pd.read_csv(path_)
    x = df["x"].values
    y = df["y"].values
    return x, y


if __name__ == "__main__":
    X, y = load_data()
    clf = AdaboostClassification(max_itr=4)
    clf.fit(X, y)

    print("training result: ")
    for clf_ in clf.clf_list:
        print(clf_[0], clf_[1].func.__name__)

    X_test = np.array([1, 4, 6])
    print("\npredict result: ")
    print(clf.predcit(X_test))

