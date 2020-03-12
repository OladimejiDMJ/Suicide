#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

data=pd.read_csv('House_train.csv')


data.shape

data.isnull().sum().sort_values(ascending=False)

#We can see that we need to drop some columns since they contain a lot missing values


data.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Id'],axis=1, inplace=True)

data.describe()


# SalePrice is our target variable.Let visualize it to obtain more details

# WORKING ON CATEGORICAL FEATURES

#TAKE CARE OF ALL MISSING VALUES IN THE CATEGORICAL VARIABLES

missing_cat_var=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','GarageCond','Electrical','MasVnrType']
for each in data[missing_cat_var]:
    data[each].fillna('missing', inplace=True)


data.isnull().sum().sort_values(ascending=False)
#ENCODING ORDINAL CATEGORICAL VARIABLES 


cat1=['ExterCond','ExterQual','BsmtQual','BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond']
#THE ABOVE CATEGORICAL VARIABLES ARE GROUPED TEGETHER BECAUSE THEIR VALUES ARE RANKED IN SIMILAR ORDER
values={'missing':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
for val in cat1:
    data[val]=data[val].map(values)

#SAME WITH THESE
cat2=['BsmtFinType1','BsmtFinType2']
values={'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'missing':0}
for val in cat2:
    data[val]=data[val].map(values)

cat3=['BsmtExposure']
values={'Gd':4,'Av':3,'Mn':2,'No':1,'missing':0}
for val in cat3:
    data[val]=data[val].map(values)
cat4=['Functional']
values={'Typ':8,'Min1':7,'Min2':6,'Mod':5,'Maj1':4,'Maj2':3,'Sev':2,'Sal':1}
for val in cat4:
    data[val]=data[val].map(values) 

cat5=['GarageFinish']
values={'Fin':3,'RFn':2,'Unf':1,'missing':0}
for val in cat5:
    data[val]=data[val].map(values)  

cat6=['Electrical']
values={'SBrkr':5,'FuseA':4,'FuseF':3,'FuseP':2,'Mix':1,'missing':0}
for val in cat6:
    data[val]=data[val].map(values) 

cat7=['LandSlope']
values={'Gtl':3,'Mod':2,'Sev':1}
for val in cat7:
    data[val]=data[val].map(values) 

data.isnull().sum().sort_values(ascending=False)

cat8=['Utilities']
values={'AllPub':4,'NoSewr':3,'NoSeWa':2,'ELO':1}
for val in cat8:
    data[val]=data[val].map(values) 

data.isnull().sum().sort_values(ascending=False)

cat9=['LandContour']
values={'Lvl':4,'Bnk':3,'HLS':2,'Low':1}
for val in cat9:
    data[val]=data[val].map(values) 

cat10=['LotShape']
values={'Reg':4,'IR1':3,'IR2':2,'IR3':1}
for val in cat10:
    data[val]=data[val].map(values) 

other_cat=['MSZoning','Street','LotConfig','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir','GarageType','PavedDrive','SaleType','SaleCondition']
for col in data[other_cat]:
    encoder = preprocessing.LabelEncoder()
    data[col] = encoder.fit_transform(data[col])

#NUMERICAL VARIABLES
missing_num_var=['LotFrontage','GarageYrBlt','MasVnrArea']
for each in data[missing_num_var]:
    data[each].fillna(0, inplace=True)

top_feature=corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12,8))
top_corr=data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show


#sns.boxplot(data['OverallQual'])


# In[28]:


#X=data[['OverallQual','YearBuilt','YearRemodAdd','ExterQual','BsmtQual','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','KitchenQual','TotRmsAbvGrd','GarageCars','GarageFinish','GarageArea']]
X=data.iloc[:, :-1]
y=data['SalePrice']


# In[29]:


X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)


stdscale=preprocessing.StandardScaler()
etr=ExtraTreesRegressor()

pipe=Pipeline([('stdscale',stdscale),('etr',etr)])
pipe.fit(X_train,y_train)

pipe.score(X_test,y_test)

stdscale2=preprocessing.StandardScaler()
XGB=XGBRegressor()
pipe2=Pipeline([('stdscale',stdscale),('XGB',XGB)])
pipe2.fit(X_train,y_train)

pipe2.score(X_test,y_test)
