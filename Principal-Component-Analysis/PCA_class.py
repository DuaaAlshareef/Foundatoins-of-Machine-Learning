import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        #sort eigenvalues and eigenvectors in descending order
        indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, indices]
        eigenvalues = eigenvalues[indices]
        #select the top values
        self.components = eigenvectors[:, :self.n_components]
    
    def transform(self, X):
        X_centered = X-self.mean
        projected_data = np.dot(X_centered, self.components)
        
        return projected_data
    

