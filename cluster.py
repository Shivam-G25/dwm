# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Create manual dataset
data = {
    'Maths': [35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    'Science': [30, 42, 46, 52, 58, 62, 66, 72, 76, 85]
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)
print("Dataset:\n", df)

# Step 3: Create and fit KMeans model
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(df)

# Step 4: Get cluster labels
df['Cluster'] = kmeans.labels_
print("\nClustered Data:\n", df)

# Step 5: Visualize the clusters
plt.scatter(df['Maths'], df['Science'], c=df['Cluster'], cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='black', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering (Students based on Marks)')
plt.xlabel('Maths Marks')
plt.ylabel('Science Marks')
plt.legend()
plt.show()

