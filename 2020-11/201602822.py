from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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

df_ca = df[df['Country/Region'].isin(['Guatemala', 'Honduras', 'El Salvador', 'Costa Rica', 'Nicaragua', 'Panama', 'Belice'])]
data_ca = df_ca.values
print(data_ca)

dates = df_ca.columns.values[4:]
formatted_dates = list(map(lambda date_str: datetime(2000+int(date_str.split('/')[2]), int(date_str.split('/')[0]), int(date_str.split('/')[1])).strftime('%d/%m/%Y'), dates))# strptime(date_str, '%-m/%-d/%-y').strftime('%d/%m/%Y'), dates))