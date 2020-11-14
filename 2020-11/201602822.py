from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import io
import os
from datetime import datetime

data_path = os.path.join(
    'https://raw.githubusercontent.com',
    'CSSEGISandData',
    'COVID-19',
    'master',
    'csse_covid_19_data',
    'csse_covid_19_time_series'
)

URL_CONFIRMED = os.path.join(data_path, 'time_series_covid19_confirmed_global.csv')
print('Confirmed Cases Source URL:', URL_CONFIRMED)
data_confirmed = requests.get(URL_CONFIRMED).content
df_cases_contries = pd.read_csv(io.StringIO(data_confirmed.decode('utf-8')))

#Data based on worldometers.info and the latest United Nations Population Division estimates
ca_country_names = ['Guatemala', 'Belize', 'Honduras', 'El Salvador', 'Nicaragua', 'Costa Rica', 'Panama']
popu_ca_countries = {
    'Country': ca_country_names,
    'Population': [17915568, 397628, 9904607, 6486205, 6624554, 5094118, 4314767], #Absolute
    'Population_m': [17916, 398, 9905, 6486, 6625, 5094, 4315], #P/1000
    'Density': [167, 17, 89, 313, 55, 65, 58], #P/Km
}
df_popu_ca_countries = pd.DataFrame(popu_ca_countries, columns=['Country', 'Population', 'Density'])

ca_population_total = df_popu_ca_countries['Population'].sum()
print('CA Population:', ca_population_total)

df_cases_ca_countries = df_cases_contries[df_cases_contries['Country/Region'].isin(ca_country_names)]
data_cases_ca_countries = df_cases_ca_countries.values

dates = df_cases_ca_countries.columns.values[4:]
formatted_dates = dates
#formatted_dates = list(map(lambda date_str: datetime(2000+int(date_str.split('/')[2]), int(date_str.split('/')[0]), int(date_str.split('/')[1])).strftime('%d/%m/%Y'), dates))# strptime(date_str, '%-m/%-d/%-y').strftime('%d/%m/%Y'), dates))

degree = 9
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
plots = []
for i in range(len(data_cases_ca_countries)):
    X = np.array(range(len(dates)))
    x_max = X.max()
    X_seq = np.linspace(X.min(), x_max, x_max+1).reshape(-1, 1)
    Y = data_cases_ca_countries[i][4:]
    
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(X[:, np.newaxis], Y)
    Y_predicted = polyreg.predict(X_seq)
    
    #plt.scatter(range(len(dates)), data_ca_countries[i][4:], color='gray')
    plot, = plt.plot(X_seq, Y_predicted, color=colors[i], linewidth=3)
    plots.append(plot)
    
plt.grid()
plt.legend(plots, ca_country_names)
plt.xlabel('Days from 01/22/2020')
plt.ylabel('Confirmed cases')
plt.title('Confirmed cases by Central America countries')
plt.show()