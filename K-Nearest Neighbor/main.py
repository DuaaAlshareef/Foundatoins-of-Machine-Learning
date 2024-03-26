import numpy as np
from collections import Counter
from dataset import *
from operations import *
from KNN import KnnClassifier


accuracy = []
for i in range(1, 50):
  knn = KnnClassifier(i)
  knn.fit(X_train,y_train)
  accuracy.append(knn.check_accuracy(X_test, y_test))
  
print(accuracy[:10])