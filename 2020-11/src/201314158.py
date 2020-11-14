
#201314158
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


#with variable X
#the first value represents number deaths
#the second value represents number of positives covid 19
#the third value represents average ages
X = np.array([[31,3045,5],[21,5790,15],[107,28883,25],[209,28398,35],[479,18996,45],[757,12701,55],[1032,7673,65],[575,2317,75],[241,827,85],[41,134,95],])
#para mostrar sin aplicar clustering 
plt.scatter(X[:,0],X[:,1], label='True Position')
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X)
print(kmeans.cluster_centers_)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap ='rainbow')
#para imprimir centroides
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')
plt.show()
