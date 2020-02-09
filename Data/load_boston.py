from sklearn.datasets import load_boston
from sklearn.model_selection import  train_test_split

boston = load_boston()
data = boston.data
target = boston.target

def load_data():
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    return X_train, X_test, y_train, y_test

def show_data():
    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], target)
    plt.show()
