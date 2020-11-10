# JERSON VILLATORO
# 201442819
# TAREA 7

import pandas as pd
import numpy as np
import matplotlib.pyplot as graficador
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Los datos fueron tomados de: https://tablerocovid.mspas.gob.gt/
# En el set de datos toma como fecha inicial 14-02-2020
# y fecha final 11-10-2020 

df = pd.read_excel('fallecidos_fecha.xlsx', sheet_name='Sheet1')

#Se extrea de la data la información que nos interesa
dias = df[['dia']]
fallecidos = df[['acumulado']]

#Fallecidos
"""fallecidos = np.array([1,3,4,6,8,9,11,12,13,14,15,18,19,20,21,22,23,24,25,27,28,32,34,36,39,41,45,51,57,63,71,81,
        89,98,112,120,131,144,156,179,191,205,221,237,266,279,295,318,332,358,379,402,432,457,480,
        503,535,564,590,626,648,665,695,723,751,785,816,850,895,941,983,1014,1051,1089,1129,1165,
        1211,1273,1309,1365,1414,1476,1539,1585,1651,1689,1729,1778,1818,1872,1912,1965,1996,2039,
        2073,2114,2153,2188,2225,2256,2300,2329,2351,2381,2427,2459,2494,2527,2549,2589,2607,2631,
        2666,2697,2725,2758,2784,2795,2825,2850,2885,2910,2938,2958,2971,2983,3007,3034,3061,3069,
        3089,3108,3120,3134,3150,3162,3186,3199,3216,3231,3244,3265,3281,3290,3301,3309,3324,3341,
        3351,3364,3374,3386,3405,3414,3429,3436,3445,3459,3470,3481,3488,3496,3507,3519,3531,3540,
        3552,3566,3574,3591,3598,3605,3620,3630,3639,3647,3657,3665,3669,3675,3683,3695,3705,3708,
        3720,3728,3735,3739,3749,3757,3764,3773,3778,3786,3792,3799,3806,3809,3812,3815,3821,3823]
)

dias = np.array(range(len(fallecidos)))"""

print(dias)
print(fallecidos)

#Se hace una proyección para el día 252
sequencia = np.linspace(dias.min(),dias.max() + 40,300).reshape(-1,1)

# Creación del modelo
model = make_pipeline(PolynomialFeatures(7), LinearRegression())
model.fit(dias, fallecidos)
graficador.figure()

graficador.scatter(dias, fallecidos)

# Preparando y mostrando la gráfica
graficador.plot(sequencia, model.predict(sequencia), color = "blue")
graficador.title("CASOS COVID-19 GT 2020")
graficador.show()
