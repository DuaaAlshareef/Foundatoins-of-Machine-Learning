import numpy as np
from dataset import generate_data
from model import LinearRegression
from evaluations import *


Xtrain, ytrain = generate_data()
model = LinearRegression(Xtrain, ytrain)

losses = []
num_epochs = 10
for epoch in range(num_epochs): 
    ypred = model.forward_pass()
    loss = mean_squared_error(ytrain,ypred)
    grads = model.backward_pass()
    model.update_param()

    losses.append(loss)
    print(f"\nEpoch {epoch}, loss {loss}")
    
print(losses)