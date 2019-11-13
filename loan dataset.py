#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#IMPORT ALL THE NECESSARY LIBRARIES
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsOneClassifier 
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[ ]:


#CHECK AND EXPLORE YOUR DATA
data=pd.read_csv('train_ctrUa4k.csv')
#pd.set_option('display.max_rows', 614)
data.describe()
data.drop('Loan_ID', axis=1, inplace=True)
#Check for missing data
data.isnull().sum()
#Let sort out the 3+ in the dependent column
data['Dependents']=data['Dependents'].replace(to_replace='3+',value=3)
#lets fill missing values for categorical data
data['Gender'].fillna('others', inplace=True)
data['Married'].fillna('Unsure', inplace=True)
data['Self_Employed'].fillna('Both', inplace=True)
#lets fill missing numerical values
data['Dependents'].fillna(3, inplace=True)
vol=['LoanAmount','Loan_Amount_Term','Credit_History']
data[vol] =data[vol].replace([0,'NaN',], np.NaN)
data.fillna(data.mean(), inplace=True)
data.isnull().sum()
#Now, we label encode our categorical data
cols = ['Gender', 'Married', 'Self_Employed','Education','Property_Area']
for col in data[cols]:
    encoder = preprocessing.LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
#Select best Features 
X=data.iloc[:, :-1]
y=data.iloc[:,-1]
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']
print(featureScores.nlargest(5,'Score')) 
#Split Our training data
X=data.iloc[:, :-1]
y=data.iloc[:,-1]
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2)
#data Preprocessing
stdscale=preprocessing.StandardScaler().fit(X_train)
X_train_std=stdscale.transform(X_train)
X_test_std=stdscale.transform(X_test)



#USING SVC 
param_grid={'C':[0.1,1,10,0.01,0.001,100,1000], 'gamma':[1,0.1,0.01,0.001,100,1000]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train_std, y_train)
grid.best_params_
grid.best_estimator_
prediction = grid.predict(X_test_std)
pred=metrics.accuracy_score(y_test, prediction)
pred


# In[ ]:


#KNN NEAREST NEIGHBOUR CLASSIFIER
knn=KNeighborsClassifier()
knn.fit(X_train_std, y_train)
train_pred=knn.predict(X_train_std)
print('Accuracy Score for train set : {}'.format(accuracy_score(y_train, train_pred)))
test_pred=knn.predict(X_test_std)
print('Accuracy Score for test set : {}'.format(accuracy_score(y_test, test_pred)))


# In[ ]:


model1=OneVsOneClassifier(LinearSVC(random_state=0))
model1.fit(X_train_std,y_train)
prediction1 = model1.predict(X_test_std)
pred1=metrics.accuracy_score(y_test, prediction1)
print(pred1)


# In[ ]:




