import numpy as np
from sigmoid import *

class SimpleNN:
    def __init__(self,h0,h1,h2):
        self.W1= np.random.randn(h1,h0)*0.01
        self.W2= np.random.randn(h2,h1)*0.01
        self.b1= np.random.randn(h1,1)
        self.b2= np.random.randn(1,1)

    def forward_pass(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = sigmoid(Z2)

        return A2, Z2, A1, Z1
    
    def backward_pass(self, X, Y, A2, A1, Z1):
        m=Y.shape[1]
        dZ2 = A2-Y
        dW2 = (np.dot(dZ2,A1.T))/m
        db2 = ((np.sum(dZ2, axis =1,keepdims=True)) /m)
        dZ1 = (self.W2.T@dZ2) * d_sigmoid(Z1)
        dW1 = (np.dot(dZ1,X.T))/m
        db1 = ((np.sum(dZ1, axis =1,keepdims=True))/m)
        return dW1, dW2, db1, db2
    

    def update(self, dW1, dW2, db1, db2, alpha):
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
    
    def predict(self, X):
        A2, _, _, _ = self.forward_pass(X)
        predictions = (A2>=0.5).astype(int)
        return predictions