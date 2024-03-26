from sklearn.datasets import make_moons
import numpy as np

"""# Data Generation"""
def load_data():
    X, y = make_moons(n_samples=600, noise=0.55, random_state=0)

    # shuffle data
    idx = np.random.permutation(X.shape[0])
    X, y = X[idx,:], y[idx]

    # train/test split
    ratio = 0.8
    X_train, y_train = X[:int (ratio*X.shape[0])], y[:int (ratio*X.shape[0])].reshape(-1,1)
    X_test, y_test = X[int (ratio*X.shape[0]):], y[int (ratio*X.shape[0]):].reshape(-1,1)
    return X_train, y_train, X_test, y_test