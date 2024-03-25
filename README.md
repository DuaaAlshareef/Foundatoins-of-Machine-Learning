# Foundations of Machine Learning

This repository contains foundational algorithms and models commonly used in machine learning. It serves as a resource for beginners to understand the core concepts and implementation of essential techniques. The repository is organized into two main folders: PCA and Simple NN Model.


## NN-From-Scratch (Simple Neural Network Model)

The NN-From-Scratch folder contains a basic implementation of a neural network from scratch using Python. This model serves as a foundation for understanding the fundamental concepts of neural networks, including forward propagation, backward propagation, and gradient descent.


# Forward Pass for One Hidden Layer Neural Network (Classification)

This is an implementation of the forward pass for a one hidden layer neural network with the output layer for classification tasks. The forward pass is a fundamental step in neural network computation, where input data is propagated through the network to generate predictions.

## Overview

![Neural Network Architecture](neural_network_architecture.png)

The neural network architecture consists of three layers:

1. **Input Layer**: The input layer receives the feature values of the input data.
2. **Hidden Layer**: The hidden layer applies a set of weights and biases to the input data, followed by an activation function (e.g., sigmoid, ReLU).
3. **Output Layer**: The output layer produces the final predictions based on the transformed features from the hidden layer.


# Backward Pass

## Overview
![Neural Network Backward Pass](neural_network_backward_pass.png)


## Implementation

The implementation of the forward pass for the one hidden layer neural network can be found in the following files:

- **model.py**: This file contains the code for performing the forward and backward pass operations. It includes functions to compute the activations of the hidden layer and the output layer, given the input features and the model parameters, it also includes calculating the gradients of the functions to be used for gradient descent.

### Gradient Descent:


### Files
- **dataset.py**:
  
- **sigmoid.py**:
  
- **evaluations.py**:



- **main.py**:

  
## PCA (Principal Component Analysis)

The PCA folder contains implementations of the Principal Component Analysis algorithm, a widely used technique for dimensionality reduction and data visualization in machine learning. PCA helps in identifying patterns in data and reducing its complexity while preserving important information.

### Files

- **PCA.py**: This file contains the implementation of the PCA algorithm. It includes functions for calculating principal components, transforming data, and reconstructing original features from reduced dimensions.

- **PCA.py**: This file contains the implementation of the PCA algorithm. It includes functions for calculating principal components, transforming data, and reconstructing original features from reduced dimensions.
  
- **PCA_CLASS.py**: The file contains the implementation of the same PCA algorithm by using class.


## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your_username/foundations-of-machine-learning.git
