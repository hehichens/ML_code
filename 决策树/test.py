from  Data.load_iris import load_data
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test= load_data()


plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()

