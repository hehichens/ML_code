from sklearn.datasets import  make_classification
from sklearn.model_selection import train_test_split

data, target = make_classification(n_samples=100,
                                   n_features=2,
                                   n_redundant=0,
                                   n_informative=1,
                                   n_clusters_per_class=1)

print(data.shape)

def load_data():
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    return X_train, X_test, y_train, y_test

def show_data():
    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], data[:, 1], c=target)
    plt.show()


show_data()