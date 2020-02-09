'''
感知机分类
'''

import numpy as np

# 数据线性可分，二分类数据
# 此处为一元一次线性方程
class ProceptronClassification():
    def __init__(self, max_itr=100, l_rate=0.1):
        self.max_itr = max_itr

        self.l_rate = l_rate
        # self.data = data

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 随机梯度下降法
    def fit(self, X, y):
        self.w = np.zeros(len(X[0]))
        self.b = 0
        for itr in range(self.max_itr):
            for d in range(len(X)):
                x_ = X[d]
                y_ = y[d]
                if y_ * (np.dot(x_, self.w) + self.b) <= 0:
                    self.w += self.l_rate * y_ * x_
                    self.b += self.l_rate * y_

        return 'PerceptronClassification Finished!'

    def score(self, X):
        y_pred = self.predict(X)
        return np.sum(y_pred == y_test) / len(y_test)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

if __name__ == "__main__":
    from Data.load_iris import load_data
    X_train, X_test, y_train, y_test = load_data()
    clf = ProceptronClassification(max_itr=100, l_rate=0.01)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: ", clf.score(X_test))

    import matplotlib.pyplot as plt
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    xx = np.arange(X_train.min(), X_train.max(), 0.01)
    plt.plot(xx, -(clf.w[0] * xx + clf.b) / clf.w[1])
    plt.show()
