# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:30:25 2019

@author: Ravi Keerthi
"""

# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


dataset['country']=dataset['country'].astype('category')
dataset['category']=dataset['category'].astype('category')

dataset['currency']=dataset['currency'].astype('category')
dataset['usd_type']=dataset['usd_type'].astype('category')
dataset['spotlight']=dataset['spotlight'].astype('category')
# Dependent variable "State" outcomes 0 or 1
dataset = pd.read_csv('C:/Users//Ravi Keerthi//Desktop//Disserattion//Cleaned_KIVA_Data.csv')
X = dataset.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19]].values
y = dataset.iloc[:, 14].values


dataset=dataset[['backers_count', 'converted_pledged_amount', 'country', 'currency',
       'category', 'currency_trailing_code', 'current_currency', 'deadline',
       'disable_communication', 'goal', 'is_starrable', 'spotlight',
       'staff_pick', 'state', 'static_usd_rate', 
       'name_len', 'name_exclaim', 'name_question', 'name_words',
       'name_is_upper']]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Accuracy Score using random forest:", metrics.accuracy_score(y_test, y_pred) )

print(" Result ")
print(metrics.classification_report(y_test, y_pred))