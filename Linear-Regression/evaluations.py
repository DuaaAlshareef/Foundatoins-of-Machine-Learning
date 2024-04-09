import numpy as np

def mean_squared_error(ytrue, ypred):
  return np.mean((ytrue-ypred)**2)