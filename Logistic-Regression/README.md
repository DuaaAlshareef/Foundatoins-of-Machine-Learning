# Logistic Regression

This repository contains the implementation of logistic regression, a fundamental machine learning algorithm used for binary classification tasks. 

## Sigmoid Function

The sigmoid function, also known as the logistic function, is a key component of logistic regression. It maps any real-valued number to the range [0, 1], making it suitable for modeling probabilities. The sigmoid function is defined as:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$


## Files

- `main.py`: This is the main script for training and evaluating logistic regression models. It contains the code to load datasets, initialize and train the model, and evaluate its performance. 

- `model.py`: This file contains the implementation of the logistic regression model. The `LogisticRegression` class implements logistic regression for binary classification tasks. It includes methods for:
  - Adding bias terms to input features (`add_ones`).
  - Computing the sigmoid function to obtain predicted probabilities (between 0 and 1) (`sigmoid`).
  - Predicting probabilities of input samples (`predict_proba`).
  - Making binary predictions based on predicted probabilities using a threshold of 0.5 (`predict`).
  - Training the model using gradient descent (`fit`), where the cross-entropy loss is minimized.
  


- `datasets.py`: This file contains functions for generating the datasets used for training and testing the logistic regression model.

     The `generate_data()` function creates a dataset with two features using the `make_classification` function from the scikit-learn library. It splits the dataset into 80% training set and 20% testing sets with a specified ratio and returns the data splits for further processing.

- `evaluations.py`: This file contains functions for evaluating the performance of logistic regression models. It includes functions for calculating the cross-entropy loss and accuracy metrics.
 The cross-entropy loss, often used as the loss function in logistic regression, is calculated using the formula:

    $$ L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} (y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)) $$

  

    The `cross_entropy_loss()` function in this file computes the cross-entropy loss between the true labels (`y_true`) and the predicted probabilities (`y_pred`).



## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/DuaaAlshareef/Foundatoins-of-Machine-Learning.git
    cd Logistic-Regression
    ```


2. Run the main script:
    ```bash
    python main.py
    ```
