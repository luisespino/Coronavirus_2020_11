#Carlos Ruperto Rodríguez Zea
#200714832
#Inteligencia Artificial

#Imports
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

#Lectura de Datos
datos = pd.read_excel('200714832.xlsx')

#Etiquetado
lb = LabelEncoder()
datos['alerta']=lb.fit_transform(datos['alerta'])
print(datos.head(10))
print()

#Separacion de datos
dx = datos[['tasa_100k_habitantes','%_positividad','tasa_pruebas_1000_habitantes']].values
dy = datos[['alerta']].values.ravel()

#Datos de Prueba
prueba = np.array([[119.44,11.50,0.74],[25.65,11.20,0.16],[50,5,0.20]])

#Modelo
modelo = GaussianNB()
modelo.fit(dx, dy)

#Resultados
print("Prediccion: ", *modelo.predict(prueba), sep='\n')
print()
print("Precision: ", modelo.score(dx, dy))