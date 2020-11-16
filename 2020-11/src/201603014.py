# Sebastián Sánchez Túchez
# 201603014

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# sex: 1 -> woman, 2 -> man
# age: age of the patient
# diabetes: 1 -> the patient has diabetes, 2 -> the patient does not has diabetes
# obesity: 1 -> the patient has obesity, 2 -> the patient does not has obesity
# tobacco: 1 -> the patient is a smoker, 2 -> the patient is not a smoker
# RESULT
# death: 1 -> the patient has died, 2 -> the patient has not died

data_set = pd.read_csv('../data/201603014_dataset.csv')
X = data_set.iloc[:, :5].values
Y = data_set.iloc[:, 5].values

# test_size -> 20% on test set, 80% training set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

model = GaussianNB()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

print('Original\t', Y_test)
print('Prediction\t', Y_predict)

cm = confusion_matrix(Y_test, Y_predict)
print('Confusion Matrix:')
print(cm)
print('Accuracy Rate:', (cm[0, 0] + cm[1, 1]) / 1250 * 100)
print('Error Rate:', (cm[0, 1] + cm[1, 0]) / 1250 * 100)


plot_confusion_matrix(model, X_test, Y_test, display_labels=["death", "alive"])
plt.show()