import numpy as np
def loss(y_pred, Y):
  return  -((np.sum(Y*(np.log(y_pred)) + (1-Y)*(np.log(1-y_pred)))) / Y.shape[1])


"""## Accuracy"""

def accuracy(y_pred, y):
  m=y.shape[1]
  correct_pred = np.sum(y_pred == y)
  return  correct_pred / m
