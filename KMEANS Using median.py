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

# # Define a custom function to calculate the distance between each point and the median
# def euclidean_distance_median(point, median):
#     return np.sqrt(np.sum((point - median) ** 2, axis=1))

# # Initialize K-means model with 2 clusters and custom function for calculating distances
# kmeans = KMeans(n_clusters=2, init='k-means++', metric=euclidean_distance_median)

# # Fit the data and predict the cluster labels
# kmeans.fit(X)
# y_pred = kmeans.predict(X)

# # Print the original data and the predicted cluster labels
# print("Original Data:")
# print(X)

# print("\nPredicted Cluster Labels using K-means with median:")
# print(y_pred)


import random
def kmeans_median(data, k):
    # Initialize centroids randomly
    centroids = random.sample(list(data), k)
    while True:
        clusters = [[] for _ in range(k)]
        # Assign each data point to the closest centroid
        for point in data:
            distances = [abs(point - c) for c in centroids]
            closest_centroid_idx = distances.index(min(distances))
            clusters[closest_centroid_idx].append(point)
        # Update centroids as the median of their respective cluster
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                new_centroids.append(centroids[i])
                continue
            new_centroids.append(int(sum(cluster)/len(cluster)))
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters
# Example usage
data_points = [1, 2, 3, 10, 11, 12]
k_clusters = kmeans_median(data_points, 2)
print("Clustered Data:")
for i,c in enumerate(k_clusters):
    print(f"Cluster {i}: {c}")