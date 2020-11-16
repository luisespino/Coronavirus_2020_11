import requests
import json
import datetime
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

response = requests.get("https://api.covidtracking.com/v1/us/daily.json")
data_source = response.json()

input_data = []

for d in data_source:
    testIncrease = d['totalTestResultsIncrease'] or 0
    positiveIncrease = d['positiveIncrease'] or 0
    
    input_data.append([testIncrease, positiveIncrease])


X = np.array(input_data)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print(kmeans.cluster_centers_)

plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')

plt.show()