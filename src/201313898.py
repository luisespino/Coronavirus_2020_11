"""
Copyright 2020 Eduardo Javier Hevia Calderon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as grafica

#Datos Guatemala
#https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
#Datos referncia
#https://www.google.com/search?q=covid+guatemala+estadisticas&oq=covid+gu&aqs=chrome.0.69i59l2j69i57j0l4j69i60.3103j0j4&sourceid=chrome&ie=UTF-8

Cases = [
    0,1,1,2,6,6,9,12,17,19,20,21,24,25,28,34,34,36,38,39,47,50,61,61,70,77,
    87,95,126,137,155,156,167,180,196,214,235,257,289,294,316,384,430,473,500,530,530,
    557,599,644,688,703,730,763,798,832,900,967,1052,1114,1199,1342,1518,1643,1763,1763,
    1912,2133,2265,2512,2743,3054,3424,3760,3954,4145,4348,4607,4739,5087,5336,5586,5760,
    6154,6485,6792,7055,7502,7866,8221,8561,8982,9491,9845,10272,10706,11251,11868,12509,
    12755,13145,13769,14540,14819,15619,15828,16397,16930,17409,18096,19011,20072,21293,
    22501,23248,23972,24787,25411,26658,27619,28598,29355,29742,30872,32074,32939,33809,
    38042,38677,39039,40229,41135,42192,43283,44492,45053,45309,46451,47605,48826,49789,
    50979,51306,51542,52365,53509,54339,55270,56189,56605,56987,57966,59089,60284,61428,
    62313,62562,62944,63847,64881,65983,66941,67856,68188,68533,69651,70714,71856,72921,
    73679,73912,74074,74893,75644,76358,77040,77481,77683,77828,78721,79622,80306,81009,
    81658,81909,82172,82684,82924,83664,84344,85152,85444,85681,86623,87442,87933,88878,
    89702,90092,90263,90968,91746,92409,93090,93748,93963,94182,94870,95704,96480,96935,
    97544,97715,97836,99094,99715,99765,101028,102415,109849,111262,113600,114123
]

# X = Days

X = np.array(range(len(Cases)))
sequencia = np.linspace(X.min(), X.max(), len(Cases)).reshape(-1, 1)

#grade
grade = 10
#grade = 10
polyreg=make_pipeline(PolynomialFeatures(grade), LinearRegression())
polyreg.fit(X[:, np.newaxis], Cases)
#grafica
grafica.figure()
grafica.scatter(X, Cases) #Forming the graph 
grafica.plot(sequencia, polyreg.predict(sequencia), color = "red") # prediction line
grafica.title("Linear Regression Grade 10")
grafica.savefig('Covid Graph')
#view
grafica.show()
