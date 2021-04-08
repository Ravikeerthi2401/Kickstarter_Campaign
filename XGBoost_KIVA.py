# -*- coding: utf-8 -*-
"""


"""

from numpy import loadtxt
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd


# load data
data= pd.read_csv("C:/Users//Ravi Keerthi//Desktop//Disserattion//Cleaned_KIVA_Data.csv") 


list(data.columns.values)

X = data.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12,15,16,17,18]].values
y = data.iloc[:, 13].values

LiveProjdata= pd.read_csv("C:/Users//Ravi Keerthi//Desktop//Disserattion//TestLiveData.csv")
X_Live = LiveProjdata.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12,15,16,17,18]].values


# split data into train and test sets
seed = 7
test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model no training data 

model = XGBClassifier(learning_rate=0.1, n_estimators=5, objective='binary:logistic', max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.6,
 colsample_bytree=0.8, silent=True, nthread=-1)

model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)

print("Accuracy Score using XG-Boost:", metrics.accuracy_score(y_test, predictions) )

print(" Result ")
print(metrics.classification_report(y_test, y_pred))

#Testing the live projects data
y_pred_Live = model.predict(X_Live)
type(y_pred_Live)
unique_elements, counts_elements = np.unique(y_pred_Live, return_counts=True)
print("Frequency of successful(1) and unsuccesful (0) values for LiveProjects  :")
print(np.asarray((unique_elements, counts_elements)))
