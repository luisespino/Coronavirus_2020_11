import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

datos = []
with open('201114705_month.csv') as csv_datos:
    for linea in csv_datos:
        datos.append(linea.strip().split(','))

tam_conf = []

for linea in datos:
    tam_conf.append(linea[1:3])

X = np.array(tam_conf)

#plt.scatter(X[:,0],X[:,1], label='True Position')
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(kmeans.cluster_centers_)
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1], color='black')

plt.show()
