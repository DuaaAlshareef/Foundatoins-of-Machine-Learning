import numpy as np

class LinearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        _, self.D = self.X.shape
        self.theta = np.zeros((self.D,1))
    
    def forward_pass(self):
        return np.dot(self.X, self.theta)
        
    def backward_pass(self):
        yhat = self.forward_pass()
        dtheta =  -2 * np.dot(self.X.T, (self.y - yhat))
        return dtheta
    
    def update_param(self, step_size=0.1):
        grads = self.backward_pass()
        self.theta = self.theta - step_size * grads
        
