from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split

iris = load_iris()
data = iris.data[:100, :2]
target = iris.target[:100]
target[target == 0] = -1


def load_data():
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    return X_train, X_test, y_train, y_test

def show_data():
    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], data[:, 1], c=target)
    plt.show()
