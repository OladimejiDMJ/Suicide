#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#load the data
data=pd.read_csv('master.csv')
#check the shape of the dat
data.shape
data
#check the number of void spaces we have in the data set
data.isnull().sum()


# # List out the columns
cols = data.columns.tolist()


#
cols

# Rearrange the columns in order to put suicides_no on the last column since it is going to be the predictor
new_cols = cols[:4] + cols[5:]
new_cols
new_cols.append(cols[4])


new_cols


# As you can see, suicide no is now on the last column

data = data[new_cols]
data


# # Drop "HDI for year" since it has a lot of void spaces thereby making it irrelevant to our analysis

data.drop('HDI for year', axis=1, inplace=True)
# # I am also dropping the "country-year" column
data.drop('country-year', axis=1, inplace=True)
data


# We need to convert some of our data into numbers and this can be done through label encoding
# country=data['country']
# encoder = preprocessing.LabelEncoder()
# encoder.fit(country)
# print("\nLabel mapping:")
# for i, item in enumerate(encoder.classes_):   
#     print(item, '-->', i) 
cols = ['country', 'sex', 'age', 'generation']
for col in data[cols]:
    encoder = preprocessing.LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
#     print(encoder.classes_)
#     print(dir(encoder))


# # Remove the commas in "gdp_for_year ($)" data

# data
data.iloc[:, 6] = data.iloc[:, 6].map(lambda x:x.replace(',',''))


# # lets normalize our data

data_normalized_l1 = preprocessing.MinMaxScaler(feature_range=(0,2))
data_normalized_l1 = data_normalized_l1.fit_transform(data)
print(data_normalized_l1)
X = data_normalized_l1
X


# # Now lets train our data

X = data.iloc[:,0:9]
y=data.iloc[:,-1]
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.8, random_state=42)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
# predictions = model.predict(X_test)
# score = accuracy_score(Y_test,predictions)
#train_features, test_features, train_targets, test_targets = #train_test_split( 18 features, targets, 19 train_size=0.8, 20 test_size=0.2, 21 # ra stratify=targets 27 )
data.iloc[:, 6] = data.iloc[:, 6].map(lambda x: int(x))


score = model.score(X_test, y_test)
# score = accuracy_score(Y_test,predictions)
print(score)


# # This is bad, lets try if we can improve it a little bit

# # Lets use select best to rank our best features

bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(5,'Score'))  #print 10 best features


# # Now we have our top five select features with "gdp_for_year ($)" being the best

data.iloc[:, 6]


A = data.iloc[:, 6]
B = data.iloc[:, -1]
# A = X.iloc[:, 6]
# B = X.iloc[:, -1]
A= np.array([A]).reshape(-1, 1)
A_train, A_test, B_train,B_test = train_test_split(A,B, test_size=0.2)


model = LinearRegression()
model.fit(A_train,B_train)


score = model.score(A_test, B_test)
# score = accuracy_score(Y_test,predictions)
print(score)