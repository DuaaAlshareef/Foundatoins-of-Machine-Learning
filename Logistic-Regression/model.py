import numpy as np
from evaluations import cross_entropy_loss
class LogisticRegression():

    def __init__(self):
        self.w = None

    def add_ones(self, X):
        ones = np.ones((X.shape[0],1))
        return np.hstack((ones,X))

    def sigmoid(self, X):
        z = X @ self.w
        return 1/(1+np.exp(-z))


    def predict_proba(self, X):  
        X = self.add_ones(X)
        proba = self.sigmoid(X)
        return proba

    def predict(self, X):
        probas = self.predict_proba(X)
        output = (probas>=0.5).astype(int)
        return output
  

    def fit(self, X, y, lr= 0.1, n_epochs=10000):

        X = self.add_ones(X)
        y = y.reshape(-1,1)

        self.w = np.zeros((X.shape[1],1))

        for epoch in range(n_epochs):
            y_pred = self.sigmoid(X)

            grad = - (1/ X.shape[0]) * (X.T @ (y-y_pred))

            self.w -= lr * grad

            loss = cross_entropy_loss(y,y_pred)

            if epoch%1000 == 0:
                print(f'loss for epoch {epoch}  : {loss}')

