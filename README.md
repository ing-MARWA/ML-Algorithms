# Pattern-Recognition
# Kmeans using variance :
# K-means Clustering with Variance

This is a Python code that demonstrates how to perform K-means clustering with a custom function for calculating distances using variance. K-means clustering is an unsupervised machine learning algorithm used to partition data points into distinct groups or clusters based on their similarities.

## Usage

1. Import the necessary libraries:

```python
import numpy as np
from sklearn.cluster import KMeans
import random
```

2. Define the custom function `euclidean_distance_variance` to calculate the distance between each data point and the variance:

```python
def euclidean_distance_variance(point, variance):
    return np.sqrt(np.sum((point - variance) ** 2, axis=1))
```

3. Create a sample dataset:

```python
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])
```

4. Initialize the K-means model with 2 clusters and the custom function for calculating distances using variance:

```python
kmeans = KMeans(n_clusters=2, init='k-means++', metric=euclidean_distance_variance)
```

5. Fit the data and predict the cluster labels:

```python
kmeans.fit(X)
y_pred = kmeans.predict(X)
```

6. Print the original data and the predicted cluster labels:

```python
print("Original Data:")
print(X)

print("\nPredicted Cluster Labels using K-means with variance:")
print(y_pred)
```

7. Define the `kmeans_variance` function to perform K-means clustering with variance:

```python
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
```

8. Example usage of the `kmeans_variance` function:

```python
data_points = [1, 2, 3, 10, 11, 12]
k_clusters, sse_value = kmeans_variance(data_points, 2)
print("Clustered Data:")
for i, c in enumerate(k_clusters):
    print(f"Cluster {i}: {c}")
print(f"\nSum of Squared Error: {sse_value}")
```

## Explanation

The code demonstrates two approaches to perform K-means clustering with variance. The first approach uses the `KMeans` class from the `sklearn.cluster` module, while the second approach defines a custom function `kmeans_variance` to perform the clustering.

The `euclidean_distance_variance` function calculates the Euclidean distance between each data point and the variance. This function is used as a metric for the K-means model in the first approach.

The `kmeans_variance` function takes a dataset (`data`) and the number of clusters (`k`) as input. It initializes centroids randomly and assigns each data point to the closest centroid. It then updates the centroids as the mean of their respective clusters and repeats the process until convergence. Finally, it calculates the sum of squared error (SSE) as a measure of clustering quality.

## Result

The code outputs the original data and the predicted cluster labels using K-means with variance. It also displays the clustered data and the sum of squared error (SSE) for the custom `kmeans_variance` function.

Please note that the code provided is a simplified example and may need modifications for your specific use case.

# Kmeans using mean :
# K-Means Clustering

This repository contains an implementation of the K-Means clustering algorithm in Python. The code allows you to cluster data points into a specified number of clusters using the K-Means algorithm.

## Usage

To use this code, follow the instructions below:

1. Install the required dependencies by running the following command:

   ```
   pip install numpy sklearn
   ```

2. Import the necessary libraries:

   ```python
   import numpy as np
   from sklearn.cluster import KMeans
   ```

3. Create a sample dataset:

   ```python
   X = np.array([
       [1, 2],
       [1.5, 1.8],
       [5, 8],
       [8, 8],
       [1, 0.6],
       [9, 11]
   ])
   ```

4. Initialize the K-Means model with the desired number of clusters:

   ```python
   kmeans = KMeans(n_clusters=2)
   ```

5. Fit the data and predict the cluster labels:

   ```python
   kmeans.fit(X)
   y_pred = kmeans.predict(X)
   ```

6. Print the original data and the predicted cluster labels:

   ```python
   print("Original Data:")
   print(X)

   print("\nPredicted Cluster Labels using K-means:")
   print(y_pred)
   ```

## Custom Implementation

If you prefer to use a custom implementation of the K-Means algorithm, you can use the provided functions `euclidean_distance`, `assign_clusters`, `update_centroids`, and `kmeans`.

Here is an example of how to use the custom implementation:

```python
import random

def euclidean_distance(a, b):
    # Function code here...

def assign_clusters(data, centroids):
    # Function code here...

def update_centroids(clusters):
    # Function code here...

def kmeans(data, k, max_iterations=100):
    # Function code here...

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
```

Feel free to modify the code to suit your needs and integrate it into your own projects.

## License

This code is licensed under the MIT License. You can find more information in the [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## Acknowledgements

This code was inspired by the K-Means implementation in the scikit-learn library.

# Kmeans using median :
# K-means Clustering with Median

This repository contains an implementation of K-means clustering with a custom function for calculating distances using the median. It provides an alternative approach to the traditional K-means algorithm.

## Installation

To use this code, you need to have Python installed. You can clone this repository using the following command:

```
git clone https://github.com/your-username/your-repository.git
```

## Usage

The code is written in Python and requires the following dependencies:

- numpy
- sklearn

You can install these dependencies using pip:

```
pip install numpy sklearn
```

To run the code, you can simply execute the `kmeans_median.py` file. Here's an example of how to use the code:

```python
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
```

Make sure to replace `data_points` with your own data and adjust the value of `k_clusters` according to your needs.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/your-username/your-repository/blob/master/LICENSE) file for more details.

## Acknowledgements

This implementation was inspired by the K-means algorithm and the concept of calculating distances using the median.

That's it! You can customize this README file to include more information about your project, such as its purpose, features, and any additional instructions for users.

