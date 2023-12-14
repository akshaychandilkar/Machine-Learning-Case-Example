import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from warnings import simplefilter 

simplefilter(action='ignore', category=FutureWarning)

print("----Marvellous Infosystems by piyush khairnar----")

print("----Diadetes Predictor Using Logistic Regreesion----")

diabetes = pd.read_csv('diabetes.csv')

print("Columns of Dataset")
print(diabetes.columns)

print("First 5 records of dataset")
print(diabetes.head())

print("Dimension of diabetes data: {}".format(diabetes.shape))
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.loc[:, diabetes.columns != 'Outcome'],  # Features (all columns except 'Outcome')
    diabetes['Outcome'],  # Target variable ('Outcome' column)
    stratify=diabetes['Outcome'],  # Ensures that the class distribution is similar in both sets
    random_state=66  # Seed for reproducibility
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, solver='liblinear').fit(X_train, y_train)

print("Training set accuracy: {:.3f}".format(logreg.score(X_train, y_train)))

print("Test set accuracy: {:.3f}".format(logreg.score(X_test, y_test)))

logreg001 = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear').fit(X_train, y_train)

print("Training set accuracy: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(logreg001.score(X_test, y_test)))

