#Data
#https://www.worldometers.info/coronavirus/country/guatemala/
#https://www.worldometers.info/coronavirus/country/el-salvador/
#https://www.worldometers.info/coronavirus/country/nicaragua/
#https://www.worldometers.info/coronavirus/country/honduras/

from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Meses Julio, Agosto, Septiembre, Octubre
#prediccion de si esta bajando el nivel de porcentaje de activo vs muertes
pais = ['El Salvador', 'El Salvador', 'El Salvador', 'El Salvador',
 'Guatemala', 'Guatemala', 'Guatemala', 'Guatemala',
  'Nicaragua', 'Nicaragua','Nicaragua', 'Nicaragua',
    'Honduras','Honduras','Honduras','Honduras']

#Casos    
activos = [2590, 8096,  10526, 4397,
 18096, 50979, 79893, 92409,
  2519, 3672, 4668, 5170,
  19558,42014,61014,76900]
#Muertes
death = [182,459,724,848 
,773,1959,2778,3261,
83,116,141,151,
497,1337,1873,2353]

#Rango
range = ['si', 'si', 'si', 'no',
 'no', 'no', 'si', 'si',
  'no', 'no', 'si', 'si',
  'si','no','no', 'si']

# Crear el codificador de etiquetas
le = preprocessing.LabelEncoder()

pais_encoded = le.fit_transform(pais) 
range_encoded = le.fit_transform(range) 

#code
print("pais_encoded: ", pais_encoded)
print("range_encoded: ", range_encoded)


# Combinar atributos en una misma lista
features = list(zip(pais_encoded, activos, death))
#print(features)

arbol = tree.DecisionTreeClassifier().fit(features, range_encoded)

# Graficar arbol de desicion
plt.figure(figsize=(10,8))
tree.plot_tree(arbol, filled=True, class_names=['si', 'no'], rounded=True)

plt.legend(loc='lower right', borderpad=3, handletextpad=3)
plt.axis("tight")
plt.show()

# Probar modelo
prediction = arbol.predict([[0,34015,979]]); #

print("prediccion",prediction)
