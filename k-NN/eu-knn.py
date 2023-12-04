import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define a custom KNN classifier
class MyKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.to_numpy()

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Calculate Euclidean distances
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Find the indices of the k-nearest neighbors
        k_indices = np.argpartition(distances, self.k)[:self.k]
        # Get the labels of the k-nearest neighbors and find the most common one
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Load and preprocess the data
data = pd.read_csv("toy.txt", sep=", ", names=["age", "sector"], engine="python")
train_data = pd.read_csv("income.train.txt.5k", sep=",", names=["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country", "target"], engine="python")
dev_data = pd.read_csv("income.dev.txt", sep=",", names=["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country", "target"], engine="python")

# Data preprocessing
num_processor = MinMaxScaler(feature_range=(0, 2))
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')
preprocessor = ColumnTransformer([("num", num_processor, ["age", "hours"]), ("cat", cat_processor, ["sector", "edu", "marriage", "occupation", "race", "sex", "country"])])
preprocessor.fit(train_data)
train_processed_data = preprocessor.transform(train_data)
dev_processed_data = preprocessor.transform(dev_data)

# Split the data into training and testing sets
X = train_processed_data
y = train_data['target'].map({' <=50K': 0, ' >50K': 1})
X_dev = dev_processed_data
y_dev = dev_data['target'].map({' <=50K': 0, ' >50K': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Find the best k value
best_k = 0
best_accuracy = 0

for k in range(1, 101, 2):
    knn = MyKNN(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_dev = knn.predict(X_dev)
    accuracy_dev = accuracy_score(y_dev, y_pred_dev)

    if accuracy_dev > best_accuracy:
        best_accuracy = accuracy_dev
        best_k = k

    print(f"k = {k} train_err: {(1-accuracy) * 100:.2f}% dev_err: {(1-accuracy_dev) * 100:.2f}%")

print(f"best k: {best_k} best accuracy: {best_accuracy}\n-----------------------------------------")

# Find the closest data points to a sample using Euclidean and Manhattan distances
person1_dev = X_dev[0]
euclidean_distances = np.linalg.norm(X_train - person1_dev, axis=1)
manhattan_distances = np.sum(np.abs(X_train - person1_dev), axis=1)

top_k = 3
top_3_euclidean = np.argpartition(euclidean_distances, top_k)[:top_k]
top_3_manhattan = np.argpartition(manhattan_distances, top_k)[:top_k]

print("Top 3 closest data points for Euclidean distance:")
for i, idx in enumerate(top_3_euclidean):
    print(f"Index {i + 1}: Index {idx}, Distance: {euclidean_distances[idx]:.2f}")

print("\nTop 3 closest data points for Manhattan distance:")
for i, idx in enumerate(top_3_manhattan):
    print(f"Index {i + 1}: Index {idx}, Distance: {manhattan_distances[idx]:.2f}")
