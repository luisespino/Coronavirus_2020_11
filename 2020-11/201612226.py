from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import numpy as np
import random

#======================================================================================#
# Number of suspected cases of covid 19 by region in Guatemala
# during the first week of infections (March 22 to March 28, 2020)

# Regions 
# [Metropolitana, Peten, Nororiental, Central, Norte, Suroccidental, Noroccidental, Suroriental]

regions = [1, 2, 3, 4, 5, 6, 7, 8]
cases = [170, 22, 29, 18, 10, 21, 8, 3]

regions = np.asarray(regions)
cases = np.asarray(cases)

regions = regions[:, np.newaxis]
cases = cases[:, np.newaxis]


plt.scatter(regions, cases)
plt.show()


xseq = np.linspace(regions.min(), regions.max(), 300).reshape(-1,1)
regr = make_pipeline(PolynomialFeatures(9), LinearRegression())
regr.fit(regions, cases)

plt.scatter(regions, cases)
plt.plot(xseq, regr.predict(xseq), color = "red")
plt.show()

#======================================================================================#
# Ref
# http://epidemiologia.mspas.gob.gt/phocadownloadpap/boletin-semana-epidemiologica/Semepi-13.pdf