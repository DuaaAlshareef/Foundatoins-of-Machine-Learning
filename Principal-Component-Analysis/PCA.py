import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

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

  # Your code here
  mean = (1/n) * np.sum(X,axis=0)

  return mean


def std(X): # np.std(X, axis = 0)

  # Your code here
  std = ((1/(n-1))* np.sum(((X-mean(X))**2),axis=0))**0.5
  print(std)
  return std


def Standardize_data(X):

  # Your code here
  X_std = (X - mean(X)) / std(X)

  return X_std

mean(X)

X_std = Standardize_data(X)

assert (np.round(mean(X_std)) == np.array([0., 0., 0., 0.])).all(), "Your mean computation is incorrect"
assert (np.round(std(X_std)) == np.array([1., 1., 1., 1.])).all(), "Your std computation is incorrect"

"""### 2.   compute the covariance matrix

Determine the covariance matrix of the data set

$\text{Cov}(X_i,X_j) = \frac{1}{n-1}\sum_{k=1}^{n}(X_{i}^{k}-\bar{X}i)(X_{j}^{k}-\bar{X}_j)$
\begin{equation*}
\mathbf{S} = \frac{1}{n-1}\mathbf{X}^\top\mathbf{X},
\end{equation*}

where $\mathbf{X}$ is the $n \times p$ matrix of standardized data, and $\mathbf{S}$ is the $p \times p$ sample covariance matrix. The $(i,j)$th entry of $\mathbf{S}$ is given by

\begin{equation*}
s_{i,j} = \frac{1}{n-1}\sum_{k=1}^{n} x_{k,i}x_{k,j},
\end{equation*}

where $x_{k,i}$ and $x_{k,j}$ are the $i$th and $j$th standardized variables, respectively, for the $k$th observation.


It is important to note that the covariance matrix is a square, postive definate ,symmetric matric of dimension p by p where p is the number of variables
"""

def covariance(X):
  n=X.shape[0]
  ## Your code here
  cov = (1/ (n-1)) * (np.dot(X.T , X))

  return cov

Cov_mat = covariance(X_std)
Cov_mat

"""### 3.   Compute the eigenvalue and eigenvector of our covariance matrix
Compute eigen values and standardised eigen vectors of the covariance matrix
Let $A$ be the covariance matrix of a dataset $X$, then the eigenvalue equation is given by:

\begin{equation*}
A\mathbf{v} = \lambda \mathbf{v}
\end{equation*}

where $\mathbf{v}$ is the eigenvector of $A$ and $\lambda$ is the corresponding eigenvalue.

To find the eigenvalues and eigenvectors, we can solve this equation using the characteristic polynomial of $A$:

\begin{equation*}
\det(A - \lambda I) = 0
\end{equation*}

where $I$ is the identity matrix of the same size as $A$. Solving for $\lambda$ gives the eigenvalues, and substituting each eigenvalue back into the equation $A\mathbf{v} = \lambda \mathbf{v}$ gives the corresponding eigenvectors.

"""

from numpy.linalg import eig

# Your code here
eigen_values, eigen_vectors =  np.linalg.eig(covariance(X_std)) # return eigen values and eigen vectors

print(eigen_values)
print(eigen_vectors)

"""*   rank the eigenvalues and their associated eigenvectors in decreasing order"""

print(eigen_values)
idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]
print(idx)

print("---------------------------------------------------")

eigen_values_sorted = eigen_values[idx]
eigen_vectors_sorted = eigen_vectors.T[:,idx]

print(eigen_vectors_sorted)

"""
######   Choose the number component that will the number of dimensions of the new feature subspace  

*   To be able to visualize our data on the new subspace we will choose 2  
*   Retain at least 95% percent from the cumulayive explained variance

"""

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






"""Build your own pca class using function we did in this tutorial. Your class should have at least the following methods:


*   def fit(X)
*   def transform(X)

"""

class PCA:
  pass

# my_pca = PCA(n_component=2)
# my_pca.fit(X)
# new_X = my_pca.transform()

