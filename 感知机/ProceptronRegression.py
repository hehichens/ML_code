'''
用感知机实现回归算法
'''
import numpy as np
import matplotlib.pyplot as plt

class ProceptronRegression():
    def __init__(self, max_itr=100, lr_rate=0.001, eps=0.01):
        self.max_itr = max_itr
        self.lr_rate = lr_rate
        self.eps = eps

    def fit(self, X, y):
        w = np.ones(2)

        for itr in range(self.max_itr):
            # print(len(X)**2)

            if 1 < self.eps:
                break
            temp = 0
            for d in range(len(X)):
                '''更新方式有点问题'''
                x_ = np.array([1, X[d]])
                y_ = y[d]
                temp += (y_ - np.dot(w, x_)) * x_

            print(temp)
            w += self.lr_rate * temp

        self.w = w

    def predict(self, X):
        return  np.dot(X, self.w[0]) + self.w[1]

    def score(self, X):
        y_pred = self.predict(X)
        return (y_pred-(np.sum(self.w[0] * X+self.w[1])**2)) / len(X)**2

if __name__ == "__main__":
    from Data.make_regression import  load_data
    X_train, X_test, y_train, y_test = load_data()
    rgs = ProceptronRegression(max_itr=10, lr_rate=0.1, eps=0.01)
    rgs.fit(X_train, y_train)
    y_pred = rgs.predict(X_test)
    print(rgs.score(X_test))

    plt.scatter(X_train, y_train)
    xx = np.arange(X_train.min(), X_train.max(), 0.01)
    plt.plot(X_test, y_pred)
