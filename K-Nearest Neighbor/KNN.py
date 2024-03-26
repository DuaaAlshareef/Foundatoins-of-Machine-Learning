import numpy as np
from collections import Counter
from dataset import *
from operations import *

class KnnClassifier:
  def __init__(self,k):
    self.k = k
    self.X_train= None
    self.y_train= None

  def fit(self, X_train, y_train):
    self.X_train= X_train
    self.y_train= y_train



  """## Predict labels"""
  def predict_labels(self,X_test):
    dist= get_distance_vectorized(self.X_train,X_test).T
    # Initialize y_pred
    y_pred = np.zeros((X_test.shape[0],1))
    #predict the label for each example in x_test
    for i in range(X_test.shape[0]):
      # get the closest k examples
      # Get indices of k nearest neighbors for each point in X_test
      knn_indices = np.argsort(dist[i])[:self.k]

      #Get the labels for the closest k
      knn_labels = self.y_train[knn_indices].flatten()

      #Use the majority vote to predict the label
      y_pred[i]= Counter(knn_labels).most_common()[0][0]

    return y_pred

  def predict(self, X_test):
    predictions = []

    ## TODO: Implement this method. You should use the functions you wrote above
    # for computing distances and to predict output labels.

    predictions = self.predict_labels(X_test)

    return predictions


  def check_accuracy(self, X_test, y_test):
    self.y_test = y_test

    m = self.y_test.shape[0]
    y_pred = self.predict(X_test)
    accuracy = np.sum(y_pred == self.y_test) /m
    return accuracy





