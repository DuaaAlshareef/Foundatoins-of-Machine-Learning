# Foundations of Machine Learning

This repository contains foundational algorithms and models commonly used in machine learning. It serves as a resource for beginners to understand the core concepts and implementation of essential techniques. The repository is organized into two main folders: PCA and Simple NN Model.


## NN-From-Scratch (Simple Neural Network Model)

The NN-From-Scratch folder contains a basic implementation of a neural network from scratch using Python. This model serves as a foundation for understanding the fundamental concepts of neural networks, including forward propagation, backward propagation, and gradient descent.

### Files:

# - **model.py**:
The implementation of the forward pass for the one hidden layer neural network can be found in the above file.

This file contains the code for performing the forward and backward pass operations. It includes functions to compute the activations of the hidden layer and the output layer, given the input features and the model parameters, it also includes calculating the gradients of the functions to be used for gradient descent.

   1. ## Forward Pass

      ## Overview
       ![Neural Network Architecture](https://www.nosco.ch/ai/ml/inc/img/neural_network.png)

This is a demonstration of the forward pass for a one-hidden-layer neural network with the output layer for classification tasks. The forward pass is a fundamental step in neural network computation, where input data is propagated through the network to generate predictions.

The neural network architecture consists of three layers:

1. **Input Layer**: The input layer receives the feature values of the input data.
2. **Hidden Layer**: The hidden layer applies a set of weights and biases to the input data, followed by an activation function (sigmoid in our example).
3. **Output Layer**: The output layer produces the final predictions based on the transformed features from the hidden layer.



2. ## Backward Pass

      ## Overview
   ![Neural Network Backward Pass](https://miro.medium.com/max/908/1*ahiviCqq6B0R_XWBmgvHkA.png)

Backpropagation is a core algorithm in neural network training, enabling adjustments of weights and biases to minimize prediction errors. It iteratively propagates errors backward through the network layers, using the chain rule of calculus to efficiently compute gradients. This process enables the optimization of network parameters via gradient descent, forming the foundation of modern deep learning frameworks and facilitating training of complex neural architectures for diverse machine learning tasks.



   3. ## Gradient Descent: 

      ## Overview
       ![Gradient Descent](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*qLSq-P_4iwNPWQTo.png)

Gradient descent is a fundamental optimization algorithm used to minimize the loss function by iteratively adjusting the model parameters. In our context, we update the weights W1,W2,b1,b2 of a neural network for classification.



- W1 <- W1 - alpha * dL_dW1
- W2 <- W2 - alpha * dL_dW2
- b1 <- b1 - alpha * dL_db1
- b2 <- b2 - alpha * dL_db2

## Usage

To use this implementation:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/DuaaAlshareif/one-hidden-layer-nn-classification.git


### Files
- **dataset.py**:
  
- **sigmoid.py**:
  
- **evaluations.py**:



- **main.py**:

  
## PCA (Principal Component Analysis)

The PCA folder contains implementations of the Principal Component Analysis algorithm, a widely used technique for dimensionality reduction and data visualization in machine learning. PCA helps in identifying patterns in data and reducing its complexity while preserving important information.

### Files

- **PCA.py**: This file contains the implementation of the PCA algorithm. It includes functions for calculating principal components, transforming data, and reconstructing original features from reduced dimensions.
  
- **PCA_CLASS.py**: The file contains the implementation of the same PCA algorithm by using class.


## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/duaaAslahreef/foundations-of-machine-learning.git
