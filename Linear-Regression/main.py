import numpy as np
from dataset import generate_data
from model import LinearRegression
from evaluations import *


Xtrain, ytrain = generate_data()
model = LinearRegression(Xtrain, ytrain)

losses = []
num_epochs = 10
for epoch in range(num_epochs): # Do some iterations
    ypred = model.forward_pass()# make predictions with current parameters
    loss = mean_squared_error(ytrain,ypred)# Compute mean squared error
    grads = model.backward_pass()# compute gradients of loss wrt parameters
    model.update_param()# Update your parameters with the gradients

    losses.append(loss)
    print(f"\nEpoch {epoch}, loss {loss}")
    
print(losses)