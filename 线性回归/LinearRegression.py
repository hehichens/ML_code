'''
标准线性回归
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class StandLinearRegression(object):
    def __init__(self, max_itr=100, lr_rate=0.01, eps=0.1):
        self.max_itr = max_itr
        self.lr_rate = lr_rate
        self.eps = eps

    def SquareLoss(self, y, y_pred):
        return np.sum((y - y_pred) ** 2) / (len(y) ** 2)

    def fit(self, X, y):
        self.w = w = np.random.rand(len(X[0]) + 1)  # b, a, 构造y = a*x + b

        for itr in range(self.max_itr):
            # print(len(X)**2)
            temp = 0
            for d in range(len(X)):
                x_ = np.hstack([1, X[d]])
                y_ = y[d]
                temp += (y_ - np.dot(w, x_)) * x_

            # print(temp)
            w += self.lr_rate * temp/(len(y)**2)
            # print(w)
            self.w = w
            y_pred = self.predict(X)
            if self.SquareLoss(y, y_pred) < self.eps:
                print("iterations:", itr + 1)
                break

        print("Train Finished !")
        return

    def predict(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.SquareLoss(y, y_pred)

if __name__ == "__main__":
    from Data.load_boston import load_data
    X_train, X_test, y_train, y_test = load_data() # 加载数据, 4控制离散程度
    rgs = StandLinearRegression(max_itr=100, lr_rate=1e-3, eps=0.01)
    rgs.fit(X_train, y_train)

    print("Training loss: ", rgs.score(X_test, y_test))
    y_pred = rgs.predict(X_test)
    print("Predict: ", y_pred)
    print("Predict Loss: ", rgs.score(X_test, y_test))

    X_train = X_train[:, 0]
    X_test = X_test[:, 0]
    print(X_train.shape, y_train.shape)
    plt.scatter(X_train, y_train, label="train")
    xx = np.arange(X_train.min(), X_train.max(), 0.01)
    plt.plot(xx, rgs.w[1]*xx + rgs.w[0], 'r')
    plt.scatter(X_test, y_pred, label='predict')
    plt.legend()
    plt.show()