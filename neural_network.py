import numpy as np

# generate data
var = 0.2
n = 800
class_0_a = var * np.random.randn(n//4,2)
class_0_b =var * np.random.randn(n//4,2) + (2,2)

class_1_a = var* np.random.randn(n//4,2) + (0,2)
class_1_b = var * np.random.randn(n//4,2) +  (2,0)

X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])
X.shape, Y.shape

# shuffle the data
rand_perm = np.random.permutation(n)

X = X[rand_perm, :]
Y = Y[rand_perm, :]

X = X.T
Y = Y.T
X.shape, Y.shape[0]

# train test split
ratio = 0.8
X_train = X [:, :int (n*ratio)]
Y_train = Y [:, :int (n*ratio)]

X_test = X [:, int (n*ratio):]
Y_test = Y [:, int (n*ratio):]

plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[0,:])
plt.show()

## Fill this cell

def sigmoid(z):

  return 1/(1+ np.exp(-z))


def d_sigmoid(z):

  return sigmoid(z)*(1-sigmoid(z))

def loss(y_pred, Y):

  return  -((np.sum(Y*(np.log(y_pred)) + (1-Y)*(np.log(1-y_pred)))) / Y.shape[1])

"""## Initialize parameters"""

h0, h1, h2 = 2, 10, 1

def init_params():
  W1= np.random.randn(h1,h0)*0.01
  W2= np.random.randn(h2,h1)*0.01
  b1= np.random.randn(h1,1)
  b2= np.random.randn(1,1)

  return W1, W2, b1, b2

"""## Forward pass"""

def forward_pass(X, W1,W2, b1, b2):
  Z1 = W1.dot(X) + b1
  A1 = sigmoid(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = sigmoid(Z2)

  return A2, Z2, A1, Z1

"""## Backward pass"""

def backward_pass(X,Y, A2, Z2, A1, Z1, W1, W2, b1, b2):
    m=Y.shape[1]
    dZ2 = A2-Y
    dW2 = (np.dot(dZ2,A1.T))/m
    db2 = ((np.sum(dZ2, axis =1,keepdims=True)) /m)
    dZ1 = (W2.T@dZ2) * d_sigmoid(Z1)
    dW1 = (np.dot(dZ1,X.T))/m
    db1 = ((np.sum(dZ1, axis =1,keepdims=True))/m)
    return dW1, dW2, db1, db2

"""## Accuracy"""

def accuracy(y_pred, y):
  m=y.shape[1]
  correct_pred = np.sum(y_pred == y)
  return  correct_pred / m


def predict(X,W1,W2, b1, b2):
  A2, Z2, A1, Z1 = forward_pass(X, W1,W2, b1, b2)
  predictions = (A2>=0.5).astype(int)
  return predictions

"""## Update parameters"""

def update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha ):
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W1 -= alpha * dW1
    b1 -= alpha * db1

    return W1, W2, b1, b2

"""## Plot decision boundary"""

def plot_decision_boundary(W1, W2, b1, b2):
  x = np.linspace(-0.5, 2.5,100 )
  y = np.linspace(-0.5, 2.5,100 )
  xv , yv = np.meshgrid(x,y)
  xv.shape , yv.shape
  X_ = np.stack([xv,yv],axis = 0)
  X_ = X_.reshape(2,-1)
  A2, Z2, A1, Z1 = forward_pass(X_, W1, W2, b1, b2)
  plt.figure()
  plt.scatter(X_[0,:], X_[1,:], c= A2)
  plt.show()

"""## Training loop"""

alpha = 0.1
W1, W2, b1, b2 = init_params()
n_epochs = 10000
train_loss = []
test_loss = []
for i in range(n_epochs):
  ## forward pass
  A2, Z2, A1, Z1 = forward_pass(X_train, W1,W2, b1, b2)
  ## backward pass
  dW1, dW2, db1, db2 = backward_pass(X_train,Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
  ## update parameters
  W1, W2, b1, b2 = update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha )

  ## save the train loss
  train_loss.append(loss(A2, Y_train))
  ## compute test loss
  AT2, ZT2, AT1, ZT1 = forward_pass(X_test, W1, W2, b1, b2)
  test_loss.append(loss(AT2, Y_test))

  # plot boundary
  if i %1000 == 0:
    plot_decision_boundary(W1, W2, b1, b2)

## plot train and test losses
plt.plot(train_loss)
plt.plot(test_loss)

y_pred = predict(X_train, W1, W2, b1, b2)
train_accuracy = accuracy(y_pred, Y_train)
print ("train accuracy :", train_accuracy)

y_pred = predict(X_test, W1, W2, b1, b2)
test_accuracy = accuracy(y_pred, Y_test)
print ("test accuracy :", test_accuracy)

