from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# STEP 1 - Get data, extracted on November 13th

# Quarantine days extracted from https://es.wikipedia.org/wiki/Cuarentena_por_la_pandemia_de_COVID-19#cite_note-8
quarentine_days = [140, 14, 36, 22, 205, 21, 32, 28, 30, 31, 74, 86, 160, 118, 96, 98, 54, 33, 128, 70, 106, 130, 32, 35]

# Travel restrictions extracted from https://www.kayak.com.gt/restricciones-de-viaje
travel_restriction = ["without restrictions", "parcial closure", "parcial closure", "total closure", "total closure", "without restrictions", "parcial closure", "parcial closure", "parcial closure", "parcial closure", "without restrictions", "parcial closure", "without restrictions", "parcial closure", "without restrictions", "parcial closure", "parcial closure", "without restrictions", "prompt reopening", "parcial closure", "parcial closure", "without restrictions", "parcial closure", "parcial closure"]

# Health investment extracted from https://datosmacro.expansion.com/estado/gasto/salud
health_investment = [14.68, 19.88, 10.06, 10.73, 16.05, 5.27, 17.8, 15.31, 2.99, 15.31, 10.26, 9.07, 17.49, 15.87, 19.24, 15.28, 15.47, 11.71, 3.38, 13.42, 14.9, 18.74, 8.78, 13.34 ]

# Population density extracted from https://datosmacro.expansion.com/demografia/poblacion
population_density = [99, 233, 16, 18, 16, 100, 3, 106, 1093, 378, 25, 146, 44, 103, 316, 94, 122, 85, 411, 200, 25, 275, 9, 48]

# Rise in COVID 19 cases extracted from https://news.google.com/covid19/map?hl=es-419&gl=US&ceid=US%3Aes-419
rise_in_cases = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1]


le = preprocessing.LabelEncoder()

# STEP 2 - Transform tuples
quarentine_days_encoded = le.fit_transform(quarentine_days)
travel_restriction_encoded = le.fit_transform(travel_restriction)
health_investment_encoded = le.fit_transform(health_investment)
population_density_encoded = le.fit_transform(population_density)
label = le.fit_transform(rise_in_cases)

# STEP 3 - Combine attributes into single list of tuples
features = list(zip(quarentine_days_encoded, travel_restriction_encoded, health_investment_encoded, population_density_encoded))
print(features)

# STEP 4 - Define and train model
model = GaussianNB()
model.fit(features, label)

# STEP 5 - Data prediction for Guatemala
predicted = model.predict([[103, 1, 17.21, 159]])
print("Predicted Value: ", predicted)
