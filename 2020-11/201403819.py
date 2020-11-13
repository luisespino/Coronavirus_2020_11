#Yoselin Lemus

from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Data set
# Datos de 13/02/2020 a 10/11/2020
# De acuerdo con: https://tablerocovid.mspas.gob.gt/
region_guatemala = ['Region2 Norte', 'Region2 Norte', 'Region 5 Central', 'Region3 Nor-Oriente', 'Region 8 Peten',
                    'Region3 Nor-Oriente', 'Region 7 Nor-Occidente', 'Region 5 Central', 'Region1 Metropolitana',
                    'Region 7 Nor-Occidente', 'Region3 Nor-Oriente', 'Region 4 Sur-Oriente', 'Region 4 Sur-Oriente',
                    'Region 6 Sur-Occidente', 'Region 6 Sur-Occidente', 'Region 5 Central', 'Region 6 Sur-Occidente',
                    'Region 4 Sur-Oriente', 'Region 6 Sur-Occidente', 'Region 6 Sur-Occidente', 'Region 6 Sur-Occidente',
                    'Region3 Nor-Oriente']
# departamento = ['Alta Verapaz', 'Baja Verapaz', 'Chimaltenango', 'Chiquimula', 'Petén', 'El Progreso', 'Quiché',
#                 'Escuintla', 'Guatemala', 'Huehuetenango', 'Izabal', 'Jalapa', 'Jutiapa', 'Quetzaltenango', 'Retalhuleu',
#                 'Sacatepequez', 'San Marcos', 'Santa Rosa', 'Sololá', 'Suchitepéquez', 'Totonicapán', 'Zacapa']

no_fallecidos = [54, 34, 102, 30, 82, 40, 37, 257, 2000, 57, 139, 13, 34, 259, 56, 150, 136, 30, 43, 112, 84, 59]

# Para cambiar letras por numeros
label = preprocessing.LabelEncoder()

# Convertir strings en numeros
region_guatemala2 = label.fit_transform(region_guatemala)
x = np.array([
    [region_guatemala2[0],no_fallecidos[0]],[region_guatemala2[1],no_fallecidos[1]],[region_guatemala2[2],no_fallecidos[2]],
    [region_guatemala2[3],no_fallecidos[3]],[region_guatemala2[4],no_fallecidos[4]],[region_guatemala2[5],no_fallecidos[5]],
    [region_guatemala2[6],no_fallecidos[6]],[region_guatemala2[7],no_fallecidos[7]],[region_guatemala2[8],no_fallecidos[8]],
    [region_guatemala2[9],no_fallecidos[9]],[region_guatemala2[10],no_fallecidos[10]],[region_guatemala2[11],no_fallecidos[11]],
    [region_guatemala2[12],no_fallecidos[12]],[region_guatemala2[13],no_fallecidos[13]],[region_guatemala2[14],no_fallecidos[14]],
    [region_guatemala2[15],no_fallecidos[15]],[region_guatemala2[16],no_fallecidos[16]],[region_guatemala2[17],no_fallecidos[17]],
    [region_guatemala2[18],no_fallecidos[18]],[region_guatemala2[19],no_fallecidos[19]],[region_guatemala2[20],no_fallecidos[20]],
    [region_guatemala2[21],no_fallecidos[21]],
])
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
print("Clusters: Fallecidos de acuerdo a regiones de Guatemala\n ",kmeans.cluster_centers_)
plt.scatter(x[:,0], x[:,1], c=kmeans.labels_, cmap='rainbow')
plt.show()