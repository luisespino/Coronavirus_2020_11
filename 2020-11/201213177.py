import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = pd.read_csv('201213177_data.csv', engine='python')

# Remove the column named 'department' because it's not part of data analysis.
data_var = data.drop(['municipio'], axis=1)
del data_var['2020-11-11']

# Normalize the data
data_norm = (data_var-data_var.min())/(data_var.max() - data_var.min())

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

# Applying the k-means method to the database
# Create the model
clustering = KMeans(n_clusters=3, max_iter=300)
# Apply the model to the database
clustering.fit(data_norm)

# Adding the classification to the original file
# The results of clustering are saved in labels_ inside the model
data['KMeans_Clusters'] = clustering.labels_
# print(data)

# Visualizing the clusters that were formed
# We will apply principal component analysis to get an idea of ​​how the clusters were formed
pca = PCA(n_components=2)
pca_covid = pca.fit_transform(data_norm)
pca_covid_df = pd.DataFrame(data=pca_covid, columns=['Component_1', 'Component_2'])
pca_covid_towns = pd.concat([pca_covid_df, data[['KMeans_Clusters']]], axis=1)
# print(pca_covid_towns)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Component 1', fontsize=15)
ax.set_ylabel('Component 2', fontsize=15)
ax.set_title('Principal Components', fontsize=20)

color_theme = np.array(['blue', 'green', 'orange'])
ax.scatter(x=pca_covid_towns.Component_1, y=pca_covid_towns.Component_2, c=color_theme[pca_covid_towns.KMeans_Clusters], s=50)
plt.show()