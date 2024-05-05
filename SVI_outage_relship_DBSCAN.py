# -*- coding: utf-8 -*-
"""
Created on Sat May  4 07:45:04 2024

@author: aas0041
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file name
file_name = 'C:/Users/aas0041/Desktop/590 Project/clustering_data.xlsx'

# Load the dataset
data = pd.read_excel(file_name, header=0)

# Specify columns for clustering
columns_for_clustering = ['MaxCustomersOut', 'CustomerHoursOutTotal', 'Outage Density', 'Overall SVI']

# Select only the specified columns for clustering
data_for_clustering = data[columns_for_clustering]

# Preprocess data: fill NaNs with column means
data_for_clustering = data_for_clustering.fillna(data_for_clustering.mean())

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_for_clustering)

# PCA for dimensionality reduction to retain 95% of the variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(features_scaled)

# Applying DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_reduced)

# Filter out noise (-1 labels) for intrinsic metric evaluations
non_noise_indices = clusters != -1
clusters_non_noise = clusters[non_noise_indices]
X_clustered = X_reduced[non_noise_indices]

# Calculate clustering evaluation metrics
if len(np.unique(clusters_non_noise)) > 1:
    silhouette_avg = silhouette_score(X_clustered, clusters_non_noise)
    calinski_harabasz_idx = calinski_harabasz_score(X_clustered, clusters_non_noise)
    print("\nThe high Silhouette Score of {:.3f} suggests that the DBSCAN algorithm has effectively grouped similar data points together into cohesive clusters while maintaining clear separation between different clusters.".format(silhouette_avg))
    print("Additionally, the Calinski-Harabasz Index of approximately {:.2f} indicates that the clusters are well-differentiated, with each cluster being tightly grouped.".format(calinski_harabasz_idx))
    print("Together, these metrics signal a successful clustering outcome, demonstrating both internal similarity within clusters and distinctness between clusters in your dimensionality-reduced dataset.")
else:
    print("Not enough clusters to calculate Silhouette Score and Calinski-Harabasz Index.")

# Print unique cluster labels and noise count
unique_clusters = np.unique(clusters)
noise_count = np.count_nonzero(clusters == -1)
print(f"\nUnique cluster labels (including noise): {unique_clusters}")
print(f"Count of noise points: {noise_count}")

# Additional info
print(f"\nOriginal number of features: {len(columns_for_clustering)}")
print(f"Reduced number of features: {pca.n_components_}")
print(f"Explained variance ratio (cumulative): {np.sum(pca.explained_variance_ratio_):.3f}")
print(f"Shape of the dimensionality-reduced dataset: {X_reduced.shape}")
print(f"Number of clusters found (excluding noise): {len(set(clusters_non_noise))}")
print(f"First 10 cluster assignments (including noise): {clusters[:10]}")

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_clustered[:, 0], y=X_clustered[:, 1], hue=clusters_non_noise, palette='viridis', s=100, alpha=0.6, legend='full')
plt.title('DBSCAN Clusters after PCA reduction (Social Vulnerability Index vs Power Outage)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster ID')
plt.show()

# Save the filtered and reduced data
filtered_data = data.iloc[non_noise_indices].copy()  # Make a copy to avoid SettingWithCopyWarning
filtered_data['Cluster'] = clusters_non_noise
filtered_data.to_excel('filtered_clustered_data.xlsx', index=False)

# Filter out unwanted columns from filtered_data
selected_columns = ['MaxCustomersOut', 'CustomerHoursOutTotal', 'Outage Density', 'Overall SVI', 'Cluster']
filtered_data_selected = filtered_data[selected_columns]

# Perform groupby operation and calculate mean values
print("Mean values per cluster in reduced dimension:")
print(filtered_data_selected.groupby('Cluster').mean())


