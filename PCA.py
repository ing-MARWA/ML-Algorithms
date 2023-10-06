# import numpy as np
# from sklearn.decomposition import PCA

# # Create a sample dataset
# X = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])

# # Initialize PCA model with 2 principal components
# pca = PCA(n_components=2)

# # Fit and transform the data
# pca.fit(X)
# X_pca = pca.transform(X)

# # Print the original data and transformed data
# print("Original Data:")
# print(X)

# print("\nTransformed Data using PCA:")
# print(X_pca)

import numpy as np
def standardize_data(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    return (X - mean) / std_dev
def compute_covariance_matrix(X):
    n_samples = X.shape[0]
    return (1 / (n_samples - 1)) * np.dot(X.T, X)
def compute_eigenvectors_and_eigenvalues(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, eigenvectors
def sort_eigenvectors_by_eigenvalues(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]
def pca(X, n_components):
    X_standardized = standardize_data(X)
    cov_matrix = compute_covariance_matrix(X_standardized)
    eigenvalues, eigenvectors = compute_eigenvectors_and_eigenvalues(cov_matrix)
    sorted_eigenvalues, sorted_eigenvectors = sort_eigenvectors_by_eigenvalues(eigenvalues, eigenvectors)
    return np.dot(X_standardized, sorted_eigenvectors[:, :n_components])
# Generate a sample dataset
np.random.seed(42)
X = np.random.randn(10, 3)

# Apply PCA to reduce the dataset to 2 dimensions
X_pca = pca(X, n_components=2)
print(X_pca)


