# Exp9: K-Means Clustering Algorithm

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data  # features
y = iris.target  # true labels (for reference only, not used in clustering)

# Optional: Scale features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters for 3 species
kmeans.fit(X_scaled)

# Cluster labels assigned by the algorithm
labels = kmeans.labels_

# Add cluster labels to a DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = labels

# Print first 10 rows
print(df.head(10))

# Visualize clusters (using first two features)
plt.figure(figsize=(8,6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='viridis', s=50)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-Means Clustering on Iris Dataset")
plt.show()
