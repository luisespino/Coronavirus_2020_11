import matplotlib.pyplot as plt 
import numpy as np 
from csv import reader
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid

# Read a CSV with data with Department = Guatemala, Town = Mixco
# Since February 13, 2020 - Until November 14

# Day of the Year , Postive cases that day

array_data = []

day_initial = 34

with open(os.getcwd()+'\\2020-11\\data\\201504448_confirmados.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            # Read only data from Mixco, Guatemala
            if row[2] == "MIXCO":
                j = 0
                for i in row:
                    if j > 4:
                        array_data.append([day_initial, int(i)])
                    day_initial = day_initial + 1
                    j = j + 1
                break

X = np.array(array_data)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(kmeans.cluster_centers_)
plt.xlabel("Days of the year")
plt.ylabel("Positive cases in Mixco")
plt.title("Covid 19 in Mixco, Guatemala")
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')

plt.show()

#centroides
#[[ 95.14159292   4.26548673]
# [191.30379747  71.03797468]
# [272.02380952  29.89285714]]

array_clases = []


X = np.array(array_data)

for i in array_data:
    if i[0] < 160 :
        array_clases.append("C1")
    elif i[0] < 210:
        array_clases.append("C2")
    elif i[0] < 400 and i[1] < 75:
        array_clases.append("C3")
    else:
        array_clases.append("C2")


Y = np.array( array_clases)

clf = NearestCentroid()
clf.fit(X, Y)

# We want to predict if the December 25, the number of positive cases its equal or
# approximatly 100, if this is correct the class of the brediction could be C2 

print(clf.predict( [ [360, 100 ] ]))
