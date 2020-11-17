
from sklearn import tree
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Behavioral analysis using sklearn to view # if the percentage level of Active cases vrs Deaths is declining
#           June,      July,      August,    September,  October
countries = ['Jamaica', 'Jamaica', 'Jamaica', 'Jamaica', 'Jamaica',
            'Haiti',  'Haiti',    'Haiti',    'Haiti',   'Haiti',
            'Cuba',   'Cuba',     'Cuba',     'Cuba',   'Cuba'
            'Bahamas', 'Bahamas', 'Bahamas', 'Bahamas', 'Bahamas']

#       June, July, August, September, October
actives = [266,  139,  142,  1548, 4453,
           2056, 4939, 2657, 2199, 1645,
           174,  46,   179,  575,  626,
           42,  4,   494,  1425,  1821]

#       June, July, August, September, October
deaths = [ 9,   10,  10,  21,  111,
           44,  105, 161, 203, 229,
           83,  86,  87,  95,  122,
           11,  11,  14,  50,  96]

#Razon of exchange for [assets/deaths]
#           29.55,  13.9,  14.2,  73.71, 40.17,
#           46.72,  47.03, 16.50, 10.83, 7.18,
#           2.09,   0.53,  2.05,  6.05,  5.13
#           3.81,   0.36,  35.28,  28.5,  18.96

#There is a decrease in cases (active/death) compared to the previous month?
decrease = ['N', 'N', 'Y', 'Y', 'N',
            'Y', 'N', 'N', 'N', 'N',
            'Y', 'N', 'Y', 'Y', 'N',
            'N', 'N', 'Y', 'N', 'N']

# Creation of the libraries that serves us to encoder or labels elements with labels for distinction
le = preprocessing.LabelEncoder()

pais_encoded = le.fit_transform(countries)
range_encoded = le.fit_transform(decrease)

# Combine attributes into the same list
features = list(zip(pais_encoded, actives, deaths))

#Gaussing model
model = GaussianNB()
# Train
clf = model.fit(features, decrease)

# Testing model(using Jamaica, actives cases 26011, deaths 1526)
prediction = clf.predict([[0,26011,1526]])

# The prediction is that for that test data, it means that the result is above average which there is no decrease or the country should take some steps to be able to reduce the number of deaths within active cases
# Is there evidence of collective immunity to coronavirus disease?
print("prediccion",prediction)

