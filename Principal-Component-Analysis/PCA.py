import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import PCA
from numpy.linalg import eig

iris = load_iris()
X = iris['data']
y = iris['target']


n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
    )
df["label"] = iris.target

df

df.info()


# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


n=X.shape[0]
def mean(X): # np.mean(X, axis = 0)
  mean = (1/n) * np.sum(X,axis=0)
  return mean


def std(X): # np.std(X, axis = 0
  std = ((1/(n-1))* np.sum(((X-mean(X))**2),axis=0))**0.5
  print(std)
  return std


def Standardize_data(X):
  X_std = (X - mean(X)) / std(X)
  return X_std


mean(X)
X_std = Standardize_data(X)

assert (np.round(mean(X_std)) == np.array([0., 0., 0., 0.])).all(), "Your mean computation is incorrect"
assert (np.round(std(X_std)) == np.array([1., 1., 1., 1.])).all(), "Your std computation is incorrect"


def covariance(X):
  n=X.shape[0]
  cov = (1/ (n-1)) * (np.dot(X.T , X))
  return cov

Cov_mat = covariance(X_std)
Cov_mat


eigen_values, eigen_vectors =  np.linalg.eig(covariance(X_std)) # return eigen values and eigen vectors

print(eigen_values)
print(eigen_vectors)

print(eigen_values)
idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]
print(idx)

print("---------------------------------------------------")

eigen_values_sorted = eigen_values[idx]
eigen_vectors_sorted = eigen_vectors.T[:,idx]

print(eigen_vectors_sorted)


explained_variance = [(i / sum(eigen_values))*100 for i in eigen_values_sorted]
explained_variance = np.round(explained_variance, 2)
cum_explained_variance = np.cumsum(explained_variance)

print('Explained variance: {}'.format(explained_variance))
print('Cumulative explained variance: {}'.format(cum_explained_variance))


"""#### Project our data onto the subspace"""

# Get our projection matrix
c = 2
P = eigen_vectors_sorted[:c, :] # Projection matrix


X_proj = X_std.dot(P.T)
X_proj.shape





"""## Using sklearn"""

from sklearn.decomposition import PCA

#define PCA model to use
pca = PCA(n_components=4)

#fit PCA model to data
pca.fit(X_std)

explained_variance = pca.explained_variance_
print(f"Explained_variance: {explained_variance}")
explained_variance_ratio_percent = pca.explained_variance_ratio_ * 100
print(f"Explained_variance_ratio: {explained_variance_ratio_percent}")
cum_explained_variance = np.cumsum(explained_variance_ratio_percent)



"""*  Kaiser'rule witch keep all the components with eigenvalues greater than 1."""

## Transform data
X_proj = pca.transform(X_std)






# my_pca = PCA(n_component=2)
# my_pca.fit(X)
# new_X = my_pca.transform()

