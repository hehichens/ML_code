'''
用感知机实现回归算法
'''
import numpy as np
import matplotlib.pyplot as plt

class ProceptronRegression():
    def __init__(self, max_itr=100, lr_rate=0.1, eps=0.01):
        self.max_itr = max_itr
        self.lr_rate = lr_rate
        self.eps = eps

    def fit(self, X, y):
        w = 1
        b = 0

        for itr in range(self.max_itr):
            # print(len(X)**2)

            if 1 < self.eps:
                break
            for d in range(len(X)):
                x_ = X[d]
                y_ = y[d]
                temp = abs((w * x_ + b) - y_)
                w = self.lr_rate * 2 * temp * (-x_)
                b = 2*temp

        self.w = w
        self.b = b

    def predict(self, X):
        return  np.dot(X, self.w) + self.b

    def score(self, X):
        y_pred = self.predict(X)
        return (y_pred-(np.sum(self.w * X+self.b)**2)) / len(X)**2

if __name__ == "__main__":
    from Data.make_regression import  load_data
    X_train, X_test, y_train, y_test = load_data()
    rgs = ProceptronRegression(max_itr=100, lr_rate=0.1, eps=0.01)
    rgs.fit(X_train, y_train)
    # y_pred = rgs.predict(X_test)
    print(rgs.score(X_test))

    # plt.scatter(X_train, y_train)
    # xx = np.arange(X_train.min(), X_train.max(), 0.01)
    # plt.plot(xx, -(rgs.w[0] * xx + rgs.b) / rgs.w[1])
