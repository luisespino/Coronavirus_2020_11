import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('201213177_data.csv', engine='python')

# Remove the column named 'department' because it's not part of data analysis.
data_var = data.drop(['municipio'], axis=1)

# Normalize the data
data_norm = (data_var-data_var.min())/(data_var.max() - data_var.min())
del data_norm['2020-11-11']

# Search for the optimal number of clusters
# Calculating how similar the individuals are within the clusters.
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300)
    # Apply KMeans to the database
    kmeans.fit(data_norm)
    wcss.append(kmeans.inertia_)

# Plotting WCSS Results to Form the Jambu Elbow
# WCSS.  It is an indicator of how similar individuals are within the clusters.
# Uncomment the following lines if you want to see how Jambu Elbow looks like.
# plt.plot(range(1,11), wcss)
# plt.title('Jambu Elbow')
# plt.xlabel('# Clusters')
# plt.ylabel('WCSS')
# plt.show()