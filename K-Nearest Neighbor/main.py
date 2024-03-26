import numpy as np
from dataset import *
from operations import *
from KNN import KnnClassifier

X_train, y_train, X_test, y_test= load_data()
accuracy = []
for i in range(1, 20):
  knn = KnnClassifier(i)
  knn.fit(X_train,y_train)
  accuracy.append(knn.check_accuracy(X_test, y_test))
  
print(accuracy)