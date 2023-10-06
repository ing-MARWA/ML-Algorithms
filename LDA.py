# import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# # Create a sample dataset
# X = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 10]
# ])

# y = np.array([0, 1, 0]) # Class labels for each sample in X

# # Initialize LDA model with 1 discriminant
# lda = LinearDiscriminantAnalysis(n_components=1)

# # Fit the data and transform it using LDA
# X_lda = lda.fit_transform(X, y)

# # Print the original data and transformed data
# print("Original Data:")
# print(X)

# print("\nTransformed Data using LDA:")
# print(X_lda)

import numpy as np

# Sample data
X = np.array([[4, 2],
              [2, 4],
              [2, 3],
              [3, 6],
              [4, 4]])

y = np.array([0, 0, 1, 1, 1])

# Calculate the mean of each class
mean_vectors = []
classes = np.unique(y)
for cls in classes:
    mean_vectors.append(np.mean(X[y == cls], axis=0))

# Calculate the within-class scatter matrix
S_W = np.zeros((X.shape[1], X.shape[1]))
for cls, mean_vec in zip(classes, mean_vectors):
    class_sc_mat = np.zeros((X.shape[1], X.shape[1]))
    for row in X[y == cls]:
        row, mean_vec = row.reshape(X.shape[1], 1), mean_vec.reshape(X.shape[1], 1)
        class_sc_mat += (row - mean_vec).dot((row - mean_vec).T)
    S_W += class_sc_mat

# Calculate the between-class scatter matrix
overall_mean = np.mean(X, axis=0)
S_B = np.zeros((X.shape[1], X.shape[1]))
for mean_vec in mean_vectors:
    n = X[y == cls].shape[0]
    mean_vec = mean_vec.reshape(X.shape[1], 1)
    overall_mean = overall_mean.reshape(X.shape[1], 1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

# Calculate the eigenvectors and eigenvalues of the matrix (S_W^-1 * S_B)
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Sort the eigenvectors by decreasing eigenvalues
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Choose the top k eigenvectors
k = 1
W = np.hstack([eig_pairs[i][1].reshape(X.shape[1], 1) for i in range(0, k)])

# Transform the data using W
X_lda = X.dot(W)

print("Original data:")
print(X)
print("\nTransformed data:")
print(X_lda)
