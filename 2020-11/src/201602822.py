from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import io
import os
from datetime import datetime, timedelta

training = False
model_path = '../data/201602822.model'
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
data_confirmed = requests.get(URL_CONFIRMED).content #Need internet connection
df_cases_contries = pd.read_csv(io.StringIO(data_confirmed.decode('utf-8')))

la_country_names = [
    'Guatemala', 
    'Belize', 
    'Honduras', 
    'El Salvador', 
    'Nicaragua', 
    'Costa Rica', 
    'Panama',
    'Mexico',
    'Argentina',
    'Bolivia',
    'Brazil',
    'Chile',
    'Colombia',
    'Ecuador',
    'Guyana',
    'Paraguay',
    'Peru',
    'Suriname',
    'Uruguay',
    'Venezuela',
    'Cuba',
    'Dominican Republic',
    'Haiti'
]

# CLEANING DATA
df_cases_countries = df_cases_contries[df_cases_contries['Country/Region'].isin(la_country_names)]
df_cases_countries.drop(columns=['Province/State', 'Country/Region'], inplace=True)

first_case_date = datetime.strptime('01/22/20', '%m/%d/%y')
data_cases_countries = np.array([0, 0, 0, 0])

for _, row in df_cases_countries.iterrows():
    latitude = row['Lat']
    longitude = row['Long']
    for label, value in row.iteritems():
        if label in ['Lat', 'Long']:
            continue
        days_since_first_case = (datetime.strptime(label, '%m/%d/%y') - first_case_date).days
        d = np.array([latitude, longitude, days_since_first_case, value])
        data_cases_countries = np.vstack([data_cases_countries, d])

data_cases_countries = np.delete(data_cases_countries, 0, 0)
data_cases_countries = data_cases_countries[data_cases_countries[:, 2].argsort()] #By date

input_data = data_cases_countries[:, :-1]
output_data = data_cases_countries[:, -1]
input_train, _, output_train, _ = train_test_split(input_data, output_data)

print(datetime.now())
if not training:
    regr = pickle.load(open(model_path, 'rb'))
else:
    regr = MLPRegressor(
            hidden_layer_sizes=(10,10,10,10),
            solver='lbfgs',
            activation='relu',
            alpha=0.0001,
            learning_rate_init=0.1,
            max_iter=20000
        ).fit(input_train, output_train)
    pickle.dump(regr, open(model_path, 'wb+'))
predicted_data = regr.predict(input_data)
print(datetime.now())

test = [15.7835, -90.2308, 297] # 297 = 11/14/2020 - 01/22/20
print('Lat:', test[0], 'Long:', test[1], 
        'Date:', (first_case_date + timedelta(days=test[2])).strftime('%m/%d/%y'),
        ' -> Confirmed cases:',regr.predict([test]))

real_data = output_data

plt.plot(real_data, label='Real')
plt.plot(predicted_data, label='Predicted')
plt.legend()
plt.xlabel('Days since first case (01/22/2020)')
plt.ylabel('Confirmed cases')
plt.grid()
plt.title('Confirmed cases by Latin America countries')
plt.show()

iday = 0
confirmed_list = []
real_data_trend = []
predicted_data_trend = []
data_cases_countries_predicted = []
for i in range(len(data_cases_countries)):
    data_cases_countries_predicted.append([
        data_cases_countries[i, 0],
        data_cases_countries[i, 1],
        data_cases_countries[i, 2],
        data_cases_countries[i, 3],
        predicted_data[i]
    ])
for lat, long, day, real_confirmed, predicted_confirmed in data_cases_countries_predicted:
    if day == iday:
       confirmed_list.append([real_confirmed, predicted_confirmed])
    else:
        real_data_trend.append(np.max(np.array(confirmed_list)[:,0]))
        predicted_data_trend.append(np.max(np.array(confirmed_list)[:,1]))
        confirmed_list = []
        confirmed_list.append([real_confirmed, predicted_confirmed])
        iday += 1

plt.plot(real_data_trend, label='Real trend')
plt.plot(predicted_data_trend, label='Predicted trend')
plt.legend()
plt.xlabel('Days since first case (01/22/2020)')
plt.ylabel('Confirmed cases')
plt.grid()
plt.title('Confirmed cases by Latin America countries')
plt.show()