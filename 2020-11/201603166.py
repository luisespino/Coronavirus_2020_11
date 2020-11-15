from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import json
# Fuente de los datos https://github.com/CSSEGISandData/COVID-19
f = open('covid_salvador.json', 'r')
es_array = json.loads(f.read())['El Salvador']
es_confirmed = list(map(lambda x: x['confirmed'], es_array))
print(len(es_confirmed))
f.close()

es_confirmed.insert(0, 0)
Y = np.array(es_confirmed)
X = np.array(range(len(es_confirmed)))
X_seq = np.linspace(X.min(), 320, 350).reshape(-1, 1)

degree=9
polyreg=make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X[:, np.newaxis], Y)
plt.figure()
plt.scatter(X, Y)
plt.plot(X_seq, polyreg.predict(X_seq), color="black")
plt.title("Casos de covid Confirmados El Salvador 320 dias")
plt.savefig('poly_graph', dpi=100)
plt.show()

