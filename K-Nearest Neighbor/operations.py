from collections import Counter
import numpy as np
from dataset import *

"""###Vectorized Implementation(No Loop)"""

def get_distance_vectorized(X_train,X_test):
    ## we decompose the squared difference into sqrt(sum(xtrain**2 + xtest**2 - 2*xtrain*xtest))
    ## term1,term2,term3 inside the sqrt
    term1= np.sum((X_train)**2,axis=1, keepdims=True)
    term2= np.sum((X_test)**2,axis=1, keepdims=True)
    term3= 2*np.dot(X_train,X_test.T)
    euc_dist = np.sqrt(term1+term2.T-term3)

    return euc_dist
