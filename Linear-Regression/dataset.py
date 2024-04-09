import numpy as np

def generate_data():
    np.random.seed(10)
    xtrain = np.linspace(0,1, 10)
    ytrain = xtrain + np.random.normal(0, 0.1, (10,))

    xtrain = xtrain.reshape(-1, 1)
    ytrain = ytrain.reshape(-1, 1)
    return xtrain, ytrain