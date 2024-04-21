import numpy as np


class LogisticRegression():

  def __init__(self,lr,n_epochs):
    self.lr = lr
    self.n_epochs = n_epochs
    self.train_losses = []
    self.w = None
    self.weight = []

  def add_ones(self, x):
    ones = np.ones((x.shape[0],1))
    return np.hstack((ones,x))

  def sigmoid(self, x):
    z = x @ self.w
    return 1/(1+np.exp(-z))

  def cross_entropy(self, x, y_true):
    y_pred = self.sigmoid(x)
    loss = -np.mean(y_true * np.log(y_pred) + (1-y_true)* np.log(1-y_pred))
    return loss


  def predict_proba(self,x):  
    x=self.add_ones(x)
    proba = self.sigmoid(x)
    return proba

  def predict(self,x):
    probas = self.predict_proba(x)
    output = (probas>=0.5).astype(int)
    return output

  def fit(self,x,y):

    x = self.add_ones(x)
    y = y.reshape(-1,1)

    self.w = np.zeros((x.shape[1],1))

    for epoch in range(self.n_epochs):
      y_pred = self.sigmoid(x)

      grad = - (1/ x.shape[0]) * (x.T @ (y-y_pred))

      self.w -= self.lr * grad

      loss = self.cross_entropy(x,y)
      self.train_losses.append(loss)

      if epoch%1000 == 0:
        print(f'loss for epoch {epoch}  : {loss}')

  def accuracy(self,y_true, y_pred):
    acc = np.mean(y_true.reshape(-1,1) == y_pred) * 100
    return acc