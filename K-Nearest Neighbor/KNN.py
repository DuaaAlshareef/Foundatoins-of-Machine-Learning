import numpy as np
from collections import Counter



"""# KNN Implementation
## Distance functions
### Loops Implementaion
"""
def get_distance_with_loops(X_train,X_test):

 ## TODO: Implement this function using a pair of nested loops over the training data and the test data.
  euc_dist=np.zeros((X_train.shape[0],X_test.shape[0]))
  for i in range(X_train.shape[0]):
    for j in range(X_test.shape[0]):
      euc_dist[i,j] = np.sqrt(np.sum((X_train[i]-X_test[j])**2))


  return euc_dist


"""###Vectorized Implementation(No Loop)"""

def get_distance_vectorized(X_train,X_test):

    ## TODO: Implement this function without loop.
    # HINT: Try to formulate the Euclidean distance using two broadcast sums and a matrix multiply.
    ## we decompose the squared difference into sqrt(sum(xtrain**2 + xtest**2 - 2*xtrain*xtest))
    ## term1,term2,term3 inside the sqrt
    term1= np.sum((X_train)**2,axis=1, keepdims=True)
    term2= np.sum((X_test)**2,axis=1, keepdims=True)
    term3= 2*np.dot(X_train,X_test.T)
    euc_dist = np.sqrt(term1+term2.T-term3)

    return euc_dist

"""## Predict labels"""

from collections import Counter
def predict_labels(dists, y_train, k=1):

  dist= get_distance_vectorized(X_train,X_test).T
  # Initialize y_pred
  y_pred = np.zeros((X_test.shape[0],1))
  #predict the label for each example in x_test
  for i in range(X_test.shape[0]):
    # get the closest k examples
    # Get indices of k nearest neighbors for each point in X_test
    knn_indices = np.argsort(dist[i])[:k]

    #Get the labels for the closest k
    knn_labels = y_train[knn_indices].flatten()

    #Use the majority vote to predict the label
    y_pred[i]= Counter(knn_labels).most_common()[0][0]

  return y_pred

## Test your code here ##
x = np.array(X_test[0]).reshape(1,-1)
dist = get_distance_with_loops(X_train,x)
pred = predict_labels(dist, y_train, k=3)
# pred

"""## Put everything in a class"""

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



