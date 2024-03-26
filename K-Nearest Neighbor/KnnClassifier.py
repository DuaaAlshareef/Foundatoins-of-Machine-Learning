import numpy as np
from collections import Counter
from dataset import *
from operations import *

class KnnClassifier:
  def __init__(self,k):
    self.k = k
    # self.X_train= None
    # self.y_train= None

  def fit(self, X_train, y_train):
    self.X_train= X_train
    self.y_train= y_train


  def predict(self, X_test):
    self.X_test = X_test
    distances = []
    predictions = []

    ## TODO: Implement this method. You should use the functions you wrote above
    # for computing distances and to predict output labels.

    distances = get_distance_vectorized(self.X_train,self.X_test)
    predictions = predict_labels(distances,self.y_train, self.k)

    return predictions


  def check_accuracy(self, x_test, y_test):
    self.y_test = y_test

    m = self.y_test.shape[0]
    y_pred = self.predict(X_test)
    accuracy = np.sum(y_pred == self.y_test) /m
    return accuracy

# accuracy = []
# for i in range(1, 50):
#   knn = KnnClassifier(i)
#   knn.fit(X_train,y_train)
#   accuracy.append(knn.check_accuracy(X_test, y_test))
# print(accuracy)



