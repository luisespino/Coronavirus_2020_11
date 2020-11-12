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
print(URL_CONFIRMED)

data_confirmed = requests.get(URL_CONFIRMED).content
df = pd.read_csv(io.StringIO(data_confirmed.decode('utf-8')))

df_ca_countries = df[df['Country/Region'].isin(['Guatemala', 'Honduras', 'El Salvador', 'Costa Rica', 'Nicaragua', 'Panama', 'Belize'])]
data_ca_countries = df_ca_countries.values
print(data_ca_countries)

dates = df_ca_countries.columns.values[4:]
formatted_dates = dates
#formatted_dates = list(map(lambda date_str: datetime(2000+int(date_str.split('/')[2]), int(date_str.split('/')[0]), int(date_str.split('/')[1])).strftime('%d/%m/%Y'), dates))# strptime(date_str, '%-m/%-d/%-y').strftime('%d/%m/%Y'), dates))

degree=9
for i in range(len(data_ca_countries)):
    X = np.array(range(len(dates)))
    x_max = X.max()
    X_seq = np.linspace(X.min(), x_max, x_max+1).reshape(-1, 1)
    Y = data_ca_countries[i][4:]
    
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(X[:, np.newaxis], Y)
    Y_predicted = polyreg.predict(X_seq)
    
    plt.scatter(range(len(dates)), data_ca_countries[i][4:], color='gray')
    plt.plot(X_seq, Y_predicted, color='coral', linewidth=4)
    plt.grid()
    plt.xlabel('Days from 01/22/2020')
    plt.ylabel('Confirmed cases')
    plt.title('%s' % data_ca_countries[i][1])
    plt.show()