# -*- coding: utf-8 -*-
"""
Created on Sat May  4 07:42:13 2024

@author: aas0041
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set the number of OMP threads to avoid MKL-related memory leak
os.environ["OMP_NUM_THREADS"] = "5"  # Adjust the number as needed based on your machine's core count

# Define the file name
file_name = 'C:/Users/aas0041/Desktop/590 Project/clustering_data.xlsx'


# Specify columns for clustering
columns_for_clustering = ['MaxCustomersOut', 'CustomerHoursOutTotal', 'Outage Density', 'Wind Gust (kt)']

# Load the dataset assuming the first row is the header
data = pd.read_excel(file_name, header=0)

# Display data types to identify non-numeric columns
print(data.dtypes)

# Preprocessing
# Replace NaN values with the mean of each column for numeric data only
for col in data.select_dtypes(include=np.number).columns:
    data[col].fillna(data[col].mean(), inplace=True)

# Scaling the data to normalize the effect of each feature
scaler = StandardScaler()
numeric_cols = data.select_dtypes(include=[np.number]).columns
data_scaled = scaler.fit_transform(data[numeric_cols])

# PCA for dimensionality reduction to retain 95% of the variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(data_scaled)
print("Reduced feature space to:", X_reduced.shape[1], "dimensions")

# Choosing the number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plot the inertia to find the optimal number of clusters
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Assuming optimal clusters from the elbow plot
optimal_clusters = 10  # This should be set based on the elbow plot visually
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
clusters = kmeans.fit_predict(data_scaled)

# Assign the cluster back to the original data
data['Cluster'] = clusters


# Visualization of clusters
sns.pairplot(data, hue='Cluster', vars=columns_for_clustering)
plt.show()

# Analyzing clusters by computing the means of each cluster for selected variables
print(data.groupby('Cluster')[columns_for_clustering].mean())  # Adjust columns as per need for better understanding of cluster centroids

# Silhouette Score and Calinski-Harabasz Index
silhouette_avg = silhouette_score(data_scaled, clusters)
calinski_harabasz_idx = calinski_harabasz_score(data_scaled, clusters)

print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_idx:.2f}")

# Save the clustered data if needed
data.to_excel('Wind speed K_cluster.xlsx', index=False)
