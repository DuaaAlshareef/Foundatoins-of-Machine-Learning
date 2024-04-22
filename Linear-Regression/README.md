# Linear Regression

This repository contains the implementation of a simple linear regression model, along with scripts for data handling and training.

## Files

- `main.py`: This is the main script for training and evaluating the linear regression model. It contains the code to load datasets, initialize and train the model, and evaluate its performance. 



- `model.py`: This file contains the implementation of the linear regression model. The `LinearRegression` class includes the following methods:

  - `forward_pass(self, X)`: Computes the forward pass of the model to generate predictions for input data `X`.

    The mathematical function for the linear regression hypothesis is computed as:
  $$ \text{hypothesis} = X \cdot \theta $$


  - `backward_pass(self, X, y)`: Computes the backward pass of the model to calculate gradients with respect to the model parameters, given input data `X` and true target values `y`.
  - `update_param(self, grad, step_size=0.1)`: Updates the model parameters using the computed gradients and a specified step size for gradient descent optimization.
  - `fit(self, X, y, num_epochs=1000)`: Fits the model to the training data `X` and target values `y`. During training, the model minimizes the mean squared error loss between predicted and true target values.



- `datasets.py`: This file contains functions for generating synthetic datasets used for training and testing linear regression models. The `generate_data()` function generates a synthetic dataset with one feature (`xtrain`) and one target variable (`ytrain`). The dataset is created using a linear relationship with added Gaussian noise for randomness.

- `evaluations.py`: This file contains evaluation metrics for assessing the performance of linear regression models. The Mean Squared Error (MSE) is computed using the following mathematical equation:

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2 $$




## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/DuaaAlshareef/Foundatoins-of-Machine-Learning.git
   cd Linear-Regression

2. Run the main script:
    ```bash
    python main.py
    ```
