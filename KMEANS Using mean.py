# import numpy as np
# from sklearn.cluster import KMeans

# # Create a sample dataset
# X = np.array([
#     [1, 2],
#     [1.5, 1.8],
#     [5, 8],
#     [8, 8],
#     [1, 0.6],
#     [9, 11]
# ])

# # Initialize K-means model with 2 clusters
# kmeans = KMeans(n_clusters=2)

# # Fit the data and predict the cluster labels
# kmeans.fit(X)
# y_pred = kmeans.predict(X)

# # Print the original data and the predicted cluster labels
# print("Original Data:")
# print(X)

# print("\nPredicted Cluster Labels using K-means:")
# print(y_pred)


import random

def euclidean_distance(a, b):
    return sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)) ** 0.5

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        closest_centroid = min(range(len(centroids)), key=lambda i: euclidean_distance(point, centroids[i]))
        clusters[closest_centroid].append(point)
    return clusters

def update_centroids(clusters):
    return [tuple(sum(cluster_dim) / len(cluster) for cluster_dim in zip(*cluster)) for cluster in clusters]

def kmeans(data, k, max_iterations=100):
    centroids = random.sample(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters

# Example data
data = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (8, 8),
    (8, 9),
    (9, 8),
    (9, 9),
]

# Run K-means algorithm
k = 2
centroids, clusters = kmeans(data, k)

# Print results
print("Centroids:", centroids)
print("Clusters:", clusters)