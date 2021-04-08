# -*- coding: utf-8 -*-
"""

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Importing the dataset 
# Dependent variable "State" outcomes 0 or 1
dataset = pd.read_csv('C:/Users//Ravi Keerthi//Desktop//Disserattion//Cleaned_KIVA_Data.csv')
X = dataset.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19]].values
y = dataset.iloc[:, 14].values

LiveProjdata= pd.read_csv("C:/Users//Ravi Keerthi//Desktop//Disserattion//KickstarterLiveData.csv")
X_Live = LiveProjdata.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12,13,16,17,18,19]].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training sethttps://mail.google.com/mail/u/0?ui=2&ik=a573fac37e&attid=0.3&permmsgid=msg-a:r333576996764040028&th=16795a633d13ba9c&view=att&disp=safe&realattid=f_jphfghsg1
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Accuracy Score using SVM:", metrics.accuracy_score(y_test, y_pred) )

print(" Result ")
print(metrics.classification_report(y_test, y_pred))

#Testing the live projects data
y_pred_Live = classifier.predict(X_Live)
type(y_pred_Live)
unique_elements, counts_elements = np.unique(y_pred_Live, return_counts=True)
print("Frequency of successful(1) and unsuccesful (0) values for LiveProjects  :")
print(np.asarray((unique_elements, counts_elements)))

corr = cleaned_data.corr()

sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.show()

sns.to_file("heatmap.png")    
# Create correlation matrix
corr_matrix = cleaned_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

