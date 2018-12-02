# -*- coding: utf-8 -*-
'''
Data cleansing File for 823 Assign A2 

Author: Andrew Garvey
Date : Dec 2nd, 2018

'''

#import standard packages
import os
import time
import random as rd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import other packages 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Output'

#------------------------------------------------------------------------------
# IMPORT FILES

## Read file(s) / directory management
os.chdir(inputdir)
test = pd.read_excel('A2testData_MMAI.xlsx')
train = pd.read_excel('A2trainData_MMAI.xlsx')
os.chdir(outputdir)


#------------------------------------------------------------------------------
# SPLITS AND NORMALIZATION

random_state=123

## drop year, I can't even
train = train.drop('Year',axis =1)
test = test.drop('Year',axis =1)

#drop the 1 super noticable outlier we saw in data exploration from train 
train = train.loc[train['Output Return %']!=1100,:]


## split test/train X/Y
x_train = train.drop('Output Return %',axis =1)
y_train = train.loc[:,['Output Return %']]

x_test = test.drop('Output Return %',axis =1)
y_test = test.loc[:,['Output Return %']]

## use standard scaler to normalalize
scaler = StandardScaler().fit(x_train)

x_train_norm = scaler.transform(x_train)
x_test_norm = scaler.transform(x_test)

x_train_names = pd.DataFrame(x_train_norm, columns = x_train.columns)
x_test_names = pd.DataFrame(x_test_norm, columns = x_test.columns)

#------------------------------------------------------------------------------
## FEATURE SELECTION

## Done via recursive feature seleciton.

estimator = LinearRegression() 
rfe = RFE(estimator, n_features_to_select = 12)  # debatable if wanted to keep everything, this is just me #learning stuff  
selector = rfe.fit(x_train_names, np.ravel(y_train))


index = selector.support_  
x_train_rfe = x_train_names.iloc[:,index]
x_test_rfe = x_test_names.iloc[:,index]


## Done via primary component analysis 

### explained variance graph
covar_mat = PCA(len(x_train_names.columns),random_state=random_state)
covar_mat.fit(x_train_names)

variance = covar_mat.explained_variance_ratio_
var = np.cumsum(np.round(variance,3))
var

#plot explained variance
plt.clf
plt.ylabel(' variance explained')
plt.xlabel('# of features')
plt.title('Variance explained vs # of features')
plt.plot(var)
plt.savefig('Variance vs # features.png')
            
###doing pca with 11 gives 95% of variance still 
n_comp =11 
pca = PCA(n_components = n_comp)
pca_fit = pca.fit(x_train_names)

x_train_pca = pca_fit.transform(x_train_names)
x_test_pca = pca_fit.transform(x_test_names)

#------------------------------------------------------------------------------
# GENERATING OUTPUT
# choosing to use PCA because it does better for everything, could easily also have chosen nothing!

## cbind X and Y 



## output to csv





'''
NOTES:

Should totally learn a pipeline for this one when modeling


'''




















