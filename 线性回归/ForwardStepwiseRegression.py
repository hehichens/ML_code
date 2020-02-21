'''
岭回归
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class RidgeRegression(object):
    def __init__(self, max_itr=100, lr_rate=0.01, eps=0.1):
        self.max_itr = max_itr
        self.lr_rate = lr_rate
        self.eps = eps

    def SquareLoss(self, y, y_pred):
        '''区别就在损失函数不同'''
        return np.sum((y - y_pred) ** 2) / (len(y) ** 2)

    def fit(self, X, y):
        w = np.random.rand(2)  # b, a, 构造y = a*x + b
        y_pred = np.dot(X, w[1]) + w[0]
        loss = self.SquareLoss(y, y_pred)
        for itr in range(self.max_itr):
            step = self.lr_rate * loss*1e2*5
            for i in range(len(w)):
                new_w = w.copy()
                for go in [-step, step]:
                    new_w[i] = w[i] + go
                    y_pred = np.dot(X, new_w[1]) + new_w[0]
                    new_loss = self.SquareLoss(y, y_pred)
                    if new_loss < loss:
                        loss = new_loss
                        w[i] = new_w[i]

            if loss < self.eps:
                print("iterations:", itr + 1)
                break

        self.w = w
        print("Train Finished !")
        return

    def predict(self, X):
        return np.dot(X, self.w[1]) + self.w[0]

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.SquareLoss(y, y_pred)

if __name__ == "__main__":
    from Data.make_regression import  load_data
    X_train, X_test, y_train, y_test = load_data(4) # 加载数据, 4控制离散程度
    rgs = RidgeRegression(max_itr=100, lr_rate=1e-3, eps=0.04)
    rgs.fit(X_train, y_train)
    print("training loss: ", rgs.score(X_test, y_test))

    y_pred = rgs.predict(X_test)
    print("predict: ", y_pred)

    score = rgs.score(X_test, y_test)
    print("Predict Loss: ", score)

    plt.scatter(X_train, y_train, label="train")
    xx = np.arange(X_train.min(), X_train.max(), 0.01)
    plt.plot(xx, rgs.w[1]*xx + rgs.w[0], 'r')
    plt.scatter(X_test, y_pred, label='predict')
    plt.legend()
    plt.show()