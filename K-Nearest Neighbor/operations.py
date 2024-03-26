from collections import Counter
import numpy as np
from dataset import *


# """# KNN Implementation
# ## Distance functions
# ### Loops Implementaion
# """
# def get_distance_with_loops(X_train,X_test):
#   euc_dist=np.zeros((X_train.shape[0],X_test.shape[0]))
#   for i in range(X_train.shape[0]):
#     for j in range(X_test.shape[0]):
#       euc_dist[i,j] = np.sqrt(np.sum((X_train[i]-X_test[j])**2))

#   return euc_dist

"""###Vectorized Implementation(No Loop)"""

def get_distance_vectorized(X_train,X_test):
    ## we decompose the squared difference into sqrt(sum(xtrain**2 + xtest**2 - 2*xtrain*xtest))
    ## term1,term2,term3 inside the sqrt
    term1= np.sum((X_train)**2,axis=1, keepdims=True)
    term2= np.sum((X_test)**2,axis=1, keepdims=True)
    term3= 2*np.dot(X_train,X_test.T)
    euc_dist = np.sqrt(term1+term2.T-term3)

    return euc_dist

"""## Predict labels"""
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