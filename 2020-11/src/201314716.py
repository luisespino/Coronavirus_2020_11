#Data
#https://www.worldometers.info/coronavirus/country/guatemala/
#https://www.worldometers.info/coronavirus/country/el-salvador/
#https://www.worldometers.info/coronavirus/country/nicaragua/
#https://www.worldometers.info/coronavirus/country/honduras/

from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Months July, August, September, October
#prediction of whether the level of percentage of assets vs deaths is decreasing
pais = ['El Salvador', 'El Salvador', 'El Salvador', 'El Salvador',
 'Guatemala', 'Guatemala', 'Guatemala', 'Guatemala',
  'Nicaragua', 'Nicaragua','Nicaragua', 'Nicaragua',
    'Honduras','Honduras','Honduras','Honduras',
    'Costa Rica', 'Costa Rica','Costa Rica','Costa Rica']

#Months July, August, September, October   
activos = [2590, 8096,  10526, 4397,
 18096, 50979, 79893, 92409,
  2519, 3672, 4668, 5170,
  19558,42014,61014,76900,
  3753,18187,42184,76828]

#Months July, August, September, October
death = [182,459,724,848 
,773,1959,2778,3261,
83,116,141,151,
497,1337,1873,2353,
17,154,443,917]

#Months July, August, September, October
range = ['si', 'si', 'si', 'no',
 'no', 'no', 'si', 'si',
  'no', 'no', 'si', 'si',
  'si','no','no', 'si',
  'si', 'si','si','si']

# Create the tag encoder
le = preprocessing.LabelEncoder()

pais_encoded = le.fit_transform(pais) 
range_encoded = le.fit_transform(range) 

#code
print("pais_encoded: ", pais_encoded)
print("range_encoded: ", range_encoded)


# Combine attributes in the same list
features = list(zip(pais_encoded, activos, death))
#print(features)

arbol = tree.DecisionTreeClassifier().fit(features, range_encoded)

# Graph decision tree
plt.figure(figsize=(10,8))
tree.plot_tree(arbol, filled=True, class_names=['si', 'no'], rounded=True)

plt.legend(loc='lower right', borderpad=3, handletextpad=3)
plt.axis("tight")
plt.show()

# Test model
prediction = arbol.predict([[0,34015,979]]); 

print("prediccion",prediction)
