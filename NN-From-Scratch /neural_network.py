import numpy as np
from dataset import train_test_split
from sigmoid import *
from evaluations import *
from model import SimpleNN
from tqdm import trange

X_train, Y_train, X_test, Y_test= train_test_split()

model = SimpleNN(h0=2,h1=10,h2=1)

alpha = 0.1
n_epochs = 10000
for i in trange(n_epochs):
  ## forward pass
  A2, Z2, A1, Z1 = model.forward_pass(X_train)
  ## backward pass
  dW1, dW2, db1, db2 = model.backward_pass(X_train, Y_train, A2, A1, Z1)
  ## update parameters
  model.update(dW1, dW2, db1, db2, alpha)
  ## save the train loss
  if i%1000==0:
    print(f"training loss after {i} epochs is {loss(A2, Y_train)}")
    ## compute test loss
    AT2, ZT2, AT1, ZT1 = model.forward_pass(X_test)
    print(f"test loss after {i} epochs is {loss(AT2, Y_test)}")


y_pred = model.predict(X_train)
train_accuracy = accuracy(y_pred, Y_train)
print ("train accuracy :", train_accuracy)

y_pred = model.predict(X_test)
test_accuracy = accuracy(y_pred, Y_test)
print ("test accuracy :", test_accuracy)

