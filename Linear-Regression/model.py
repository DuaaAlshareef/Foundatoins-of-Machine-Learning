import numpy as np
from evaluations import *
class LinearRegression:
    def __init__(self):
        self.theta = None
    
    def forward_pass(self, X):
        return np.dot(X, self.theta)
        
    def backward_pass(self, X, y):
        yhat = self.forward_pass(X)
        dtheta =  -2 * np.dot(X.T, (y - yhat))
        return dtheta
    
    def update_param(self, grad, step_size=0.1):
        self.theta -= step_size * grad
        

    def fit(self, X, y, num_epochs = 1000):

        D = X.shape[1]
        self.theta = np.zeros((D,1))
        
        for epoch in range(num_epochs): 
            ypred = self.forward_pass(X)
            loss = mean_squared_error(y, ypred)
            grad = self.backward_pass(X, y)
            self.update_param(grad)

            print(f"\nEpoch {epoch}, loss {loss}")
            