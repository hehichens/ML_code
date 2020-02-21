import numpy as np

from sklearn.model_selection import train_test_split

def load_data(n):
    X = np.arange(0, 10, 0.1)
    y = X + (np.random.rand(len(X)) - 0.5) * n
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def show_data():
    import matplotlib.pyplot as plt
    print(X.shape)
    plt.scatter(X, y)
    plt.plot(X, X)
    plt.show()
