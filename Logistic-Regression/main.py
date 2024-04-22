from datasets import generate_data
from model import LogisticRegression
from evaluations import *



X_train, y_train, X_test, y_test = generate_data()

model = LogisticRegression()

model.fit(X_train, y_train)




ypred_train = model.predict(X_train)
acc = accuracy(y_train,ypred_train)
print(f"The training accuracy is: {acc}")

ypred_test = model.predict(X_test)
acc = accuracy(y_test,ypred_test)
print(f"The test accuracy is: {acc}")