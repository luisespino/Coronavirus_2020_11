import pandas as pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#online data
#data=pandas.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')

#get data  https://ourworldindata.org/coronavirus
data=pandas.read_csv('201504480.csv')


#last date 
data=data[data['date']=='2020-11-15']

#selected countries
data=data[(data['continent']=='North America') | (data['continent']=='South America')]
data=data[['location','total_cases_per_million','total_deaths_per_million']]


#set 0 in NaN values
data['total_cases_per_million'] = data['total_cases_per_million'].fillna(0)
data['total_deaths_per_million'] = data['total_deaths_per_million'].fillna(0)


#selected columns
data1 = data[['total_cases_per_million','total_deaths_per_million']]


X = np.array(data1 )
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(kmeans.cluster_centers_)

centroides = kmeans.cluster_centers_
etiquetas = kmeans.labels_

plt.plot(X[etiquetas==0,0],X[etiquetas==0,1],'r.', label='cluster 1')
plt.plot(X[etiquetas==1,0],X[etiquetas==1,1],'b.', label='cluster 2')
plt.plot(X[etiquetas==2,0],X[etiquetas==2,1],'g.', label='cluster 3')

plt.plot(centroides[:,0],centroides[:,1],'mo',markersize=8, label='centroides')
plt.legend(loc='best')
plt.show()

cluster = kmeans.fit_predict(data1)
data['cluster'] = cluster

print(data.to_string(index=False))



