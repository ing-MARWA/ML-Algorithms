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

# # Define a custom function to calculate the distance between each point and the variance
# def euclidean_distance_variance(point, variance):
#     return np.sqrt(np.sum((point - variance) ** 2, axis=1))

# # Initialize K-means model with 2 clusters and custom function for calculating distances using variance
# kmeans = KMeans(n_clusters=2, init='k-means++', metric=euclidean_distance_variance)

# # Fit the data and predict the cluster labels
# kmeans.fit(X)
# y_pred = kmeans.predict(X)

# # Print the original data and the predicted cluster labels
# print("Original Data:")
# print(X)

# print("\nPredicted Cluster Labels using K-means with variance:")
# print(y_pred)



import random
def kmeans_variance(data, k):
    # Initialize centroids randomly
    centroids = random.sample(list(data), k)
    while True:
        clusters = [[] for _ in range(k)]
        # Assign each data point to the closest centroid
        for point in data:
            distances = [abs(point - c) for c in centroids]
            closest_centroid_idx = distances.index(min(distances))
            clusters[closest_centroid_idx].append(point)
        # Update centroids as the mean of their respective cluster
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                new_centroids.append(centroids[i])
                continue
            new_centroids.append(sum(cluster)/len(cluster))
        if new_centroids == centroids:
            break
        centroids = new_centroids
    # Calculate sum of squared error (SSE) as a measure of clustering quality 
    sse = 0.0
    for i, centroid in enumerate(centroids):
      sse += sum((x - centroid) ** 2 for x in clusters[i])
    return (clusters, sse)
# Example usage
data_points = [1, 2, 3, 10, 11, 12]
k_clusters,sse_value= kmeans_variance(data_points ,2)
print("Clustered Data:")
for i,c in enumerate(k_clusters ):
   print(f"Cluster {i}: {c}")
print(f"\nSum of Squared Error: {sse_value}")