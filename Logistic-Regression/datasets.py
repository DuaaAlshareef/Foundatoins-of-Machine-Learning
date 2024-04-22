import numpy as np
from sklearn.datasets import make_classification


def generate_data():

  X, y = make_classification(n_features=2, n_redundant=0, random_state=1, n_clusters_per_class=1)
  
  np.random.seed(0) 
  train_size = 0.8
  n = int(len(X)*train_size)
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  train_idx = indices[: n]
  test_idx = indices[n:]
  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  return X_train, y_train, X_test, y_test

