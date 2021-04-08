# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import lightgbm as lgb
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('C:/Users//Ravi Keerthi//Desktop//Disserattion//Cleaned_KIVA_Data.csv')
TestLiveProjectsData = pd.read_csv('C:/Users//Ravi Keerthi//Desktop//Disserattion//KickstarterLiveData.csv')
dataset=dataset[['backers_count', 'converted_pledged_amount', 'country', 'currency',
       'category', 'currency_trailing_code', 'current_currency', 'deadline',
       'disable_communication', 'goal', 'is_starrable', 'spotlight',
       'staff_pick', 'state', 'static_usd_rate', 
       'name_len', 'name_exclaim', 'name_question', 'name_words',
       'name_is_upper']]
TestLiveProjectsData=TestLiveProjectsData[['backers_count', 'converted_pledged_amount', 'country', 'currency',
       'category', 'currency_trailing_code', 'current_currency', 'deadline',
       'disable_communication', 'goal', 'is_starrable', 'spotlight',
       'staff_pick', 'state', 'static_usd_rate', 
       'name_len', 'name_exclaim', 'name_question', 'name_words',
       'name_is_upper']]


X = dataset.iloc[:, [1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19]].values
y = dataset.iloc[:, 13].values

    
X_Live = TestLiveProjectsData.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19]].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 8
params['min_data'] = 100
params['max_depth'] = 8
clf = lgb.train(params, d_train, 100)

#Prediction
y_pred=clf.predict(X_test)

#convert into binary values
for i in range(0,len(y_pred)):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)

print("Accuracy Score using LightGBM:", metrics.accuracy_score(y_test, y_pred) )

print(" Result ")
print(metrics.classification_report(y_test, y_pred))

#Testing the live projects data

#Prediction
y_pred_Live=clf.predict(X_Live)
type(y_pred_Live)
#y_pred_Live=np.round(y_pred_Live, 2)
#convert into binary values
for i in range(0,len(y_pred_Live)):
    if y_pred_Live[i]>=.5:       # setting threshold to .5
       y_pred_Live[i]=1
    else:  
       y_pred_Live[i]=0

unique_elements, counts_elements = np.unique(y_pred_Live, return_counts=True)
print("Frequency of successful(1) and unsuccesful (0) values for LiveProjects  :")
print(np.asarray((unique_elements, counts_elements)))





