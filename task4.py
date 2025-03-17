import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import seaborn as sns

# Generate synthetic data for clustering
# We create four Gaussian-distributed clusters with known mean and covariance
np.random.seed(42)
data1 = np.random.multivariate_normal([10, 10], [[4.5, 0], [0, 4.5]], size=100)
data2 = np.random.multivariate_normal([30, 10], [[4.5, 0], [0, 4.5]], size=100)
data3 = np.random.multivariate_normal([10, 30], [[4.5, 0], [0, 4.5]], size=100)
data4 = np.random.multivariate_normal([30, 30], [[4.5, 0], [0, 4.5]], size=100)

# Combine all clusters into a single dataset and assign ground-truth labels
data = np.vstack([data1, data2, data3, data4])
labels_true = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100)

# Define a color palette to use for visualization
n_clusters_original = 12  # Number of clusters for the original KMeans model
colors = sns.color_palette("husl", n_colors=n_clusters_original)

# Initialize KMeans models with different cluster configurations
kmeans_original = KMeans(n_clusters=n_clusters_original, random_state=42)  # Model with 12 clusters
kmeans_single = KMeans(n_clusters=8, random_state=42)  # Reduced model with 8 clusters
kmeans_double = KMeans(n_clusters=4, random_state=42)  # Reduced model with 4 clusters

# Fit KMeans to the dataset and measure execution time
start = time.time()
labels_original = kmeans_original.fit_predict(data)
original_time = (time.time() - start) * 1000  # Time in milliseconds

start = time.time()
labels_single = kmeans_single.fit_predict(data)
single_time = (time.time() - start) * 1000

start = time.time()
labels_double = kmeans_double.fit_predict(data)
double_time = (time.time() - start) * 1000

# Split data into training and test sets for evalution
X_train, X_test, y_train, y_test = train_test_split(data, labels_true, test_size=0.3, random_state=42)

# Evaluate the accuracy of the original KMeans model
train_preds_original = kmeans_original.predict(X_train)
test_preds_original = kmeans_original.predict(X_test)
original_accuracy = accuracy_score(y_test, test_preds_original)

# Evaluate the accuracy of the single-stage clustering model (8clusters)
train_preds_single = kmeans_single.predict(X_train)
test_preds_single = kmeans_single.predict(X_test)
single_accuracy = accuracy_score(y_test, test_preds_single)

# Evaluate the accuracy of the double-stage clustering model (4clusters)
train_preds_double = kmeans_double.predict(X_train)
test_preds_double = kmeans_double.predict(X_test)
double_accuracy = accuracy_score(y_test, test_preds_double)

# Create a 2x2 plot for visualizing the data and clustering results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Visualization of the original data (all points with out clastering)
axes[0, 0].scatter(data[:, 0], data[:, 1], c='green', alpha=0.6)
axes[0, 0].set_title("Original Data (Unclustered)")

# Visualization of the original clustering results with 12 clusters
for i in range(n_clusters_original):
    cluster_data = data[labels_original == i]  # Extract points in cluster i
    axes[0, 1].scatter(cluster_data[:, 0], cluster_data[:, 1], 
                       c=[colors[i]], label=f"Cluster {i}", alpha=0.6)
    axes[0, 1].scatter(kmeans_original.cluster_centers_[i, 0], 
                       kmeans_original.cluster_centers_[i, 1], 
                       c='black', marker='x', s=200, linewidths=3)
axes[0, 1].set_title("Clustering with 12 Clusters")

# Visualization of singlestage clustering results with 8 clusters
for i in range(8):
    cluster_data = data[labels_single == i]  # Extract points in cluster i
    color_idx = int(i * (n_clusters_original / 8))  # Map to original color palette
    axes[1, 0].scatter(cluster_data[:, 0], cluster_data[:, 1], 
                       c=[colors[color_idx]], alpha=0.6)
axes[1, 0].set_title("Clustering with 8 Clusters (Single-Stage)")

# Visualization of double-stage clustering results with 4 clusters
for i in range(4):
    cluster_data = data[labels_double == i]  # Extract points in cluster i
    color_idx = int(i * (n_clusters_original / 4))  # Map to original color palette
    # Randomly sample half of the points in the cluster for visualization
    sample_size = len(cluster_data) // 2
    sampled_indices = np.random.choice(len(cluster_data), size=sample_size, replace=False)
    sampled_data = cluster_data[sampled_indices]
    axes[1, 1].scatter(sampled_data[:, 0], sampled_data[:, 1], 
                       c=[colors[color_idx]], alpha=0.6)
axes[1, 1].set_title("Clustering with 4 Clusters (Double-Stage)")

# Addjust the axes limits for better visualization
for ax in axes.flat:
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)

plt.tight_layout()
plt.show()

# Print the evaluation results
print(f"(Original Data) Mean Testing Accuracy: {original_accuracy:.2f}, Training Time: {original_time:.2f} ms")
print(f"(Single-stage Clustering) Mean Testing Accuracy: {single_accuracy:.2f}, Training Time: {single_time:.2f} ms")
print(f"(Double-stage Clustering) Mean Testing Accuracy: {double_accuracy:.2f}, Training Time: {double_time:.2f} ms")
