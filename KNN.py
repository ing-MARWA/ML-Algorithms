# import numpy as np

# class KNN:
#     def __init__(self, k=5):
#         self.k = k
    
#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y
    
#     def euclidean_distance(self, x1, x2):
#         return np.sqrt(np.sum((x1 - x2)**2))
    
#     def predict(self, X):
#         y_pred = []
        
#         for i in range(len(X)):
#             distances = []
            
#             for j in range(len(self.X_train)):
#                 dist = self.euclidean_distance(X[i], self.X_train[j])
#                 distances.append((dist, self.y_train[j]))
            
#             distances.sort()
#             neighbors = distances[:self.k]
            
#             labels = [neighbor[1] for neighbor in neighbors]
#             label = max(set(labels), key=labels.count)
#             y_pred.append(label)
        
#         return y_pred
# # create a KNN object with k=3
# knn = KNN(k=3)

# # train on some data
# X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9,11]])
# y_train = ['red', 'red', 'blue', 'blue', 'red', 'blue']

# knn.fit(X_train, y_train)

# # make predictions on new data
# X_test = np.array([[1, 4], [2.5, 2.8], [3, 6], [7, 8], [0, 0.6]])
# y_pred = knn.predict(X_test)

# print(y_pred)  # output: ['red', 'red', 'red', 'blue', 'red']



import math
import operator

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Define the KNN function
def knn(training_data, test_data, k):
    distances = []
    for i in range(len(training_data)):
        distance = euclidean_distance(training_data[i][:-1], test_data)
        distances.append((training_data[i], distance))
    distances.sort(key=operator.itemgetter(1))
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# Define a function to predict the class of a test instance
def predict_class(neighbors):
    class_votes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

# Example usage
training_data = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1]
]

test_data = [5.0, 3.5, None] # None is a placeholder for the class label
k = 3

neighbors = knn(training_data, test_data, k)
predicted_class = predict_class(neighbors)

print(f"Test instance: {test_data[:-1]}")
print(f"Neighbors: {neighbors}")
print(f"Predicted class: {predicted_class}")   
