from datasets import train_test_split
from sklearn import make_classification


X, y = make_classification(n_features=2, n_redundant=0, random_state=1, n_clusters_per_class=1)
