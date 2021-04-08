# -*- coding: utf-8 -*-
"""

"""
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import lightgbm as lgb
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('C:/Users//Ravi Keerthi//Desktop//Disserattion//Cleaned_KIVA_Data.csv')
dataset=dataset[['backers_count', 'converted_pledged_amount', 'country', 'currency',
       'category', 'currency_trailing_code', 'current_currency', 'deadline',
       'disable_communication', 'goal', 'is_starrable', 'spotlight',
       'staff_pick', 'state', 'static_usd_rate', 
       'name_len', 'name_exclaim', 'name_question', 'name_words',
       'name_is_upper']]
#K-fold Cross Validatin Technique
X = dataset.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19]].values
y = dataset.iloc[:, 13].values




#K-fold Cross Validatin Technique
from sklearn.model_selection import train_test_split

# Prepare for LightGBM

# Parameters
N_FOLDS =10 #No. of Folds
MAX_BOOST_ROUNDS = 8000
LEARNING_RATE = .0022

#X_train = X_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(X, label=y)



# 10-fold cross-validation with K=5 for KNN (the n_neigh    bors parameter)
# k = 5 for KNeighborsClassifier

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

# Cross-validate
cv_results = lgb.cv(params, d_train, num_boost_round=MAX_BOOST_ROUNDS, nfold=N_FOLDS, 
                    verbose_eval=20, early_stopping_rounds=40)
# Display results
print('Current parameters:\n', params)
print('\nBest num_boost_round:', len(cv_results['binary_logloss-mean']))
print('Best CV score:', cv_results['binary_logloss-mean'][-1])
#Testing the live projects data


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)

print("Accuracy Score using LightGBM:", metrics.accuracy_score(y_test, y_pred) )

print(" Result ")
print(metrics.classification_report(y_test, y_pred))





#Prediction
y_pred_Live=clf.predict(X_Live)

#convert into binary values
for i in range(0,len(y_pred_Live)):
    if y_pred_Live[i]>=.5:       # setting threshold to .5
       y_pred_Live[i]=1
    else:  
       y_pred_Live[i]=0

unique_elements, counts_elements = np.unique(y_pred_Live, return_counts=True)
print("Frequency of successful(1) and unsuccesful (0) values for LiveProjects  :")
print(np.asarray((unique_elements, counts_elements)))