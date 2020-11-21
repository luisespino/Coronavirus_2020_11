# http://epidemiologia.mspas.gob.gt/phocadownloadpap/boletin-semana-epidemiologica/Semepi-13.pdf
# 200312459
# Gustavo Ichel

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import numpy as np
import random


regiones = [1, 2, 3, 4, 5, 6, 7, 8]
casos = [170, 23, 31, 17, 11, 24, 9, 2]

regiones = np.asarray(regiones)
casos = np.asarray(casos)

regiones = regiones[:, np.newaxis]
casos = casos[:, np.newaxis]


plt.scatter(regiones, casos)
plt.show()


xseq = np.linspace(regiones.min(), regiones.max(), 300).reshape(-1,1)
regr = make_pipeline(PolynomialFeatures(9), LinearRegression())
regr.fit(regiones, casos)

plt.scatter(regiones, casos)
plt.plot(xseq, regr.predict(xseq), color = "red")
plt.show()
