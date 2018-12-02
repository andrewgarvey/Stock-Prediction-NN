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
rfe = RFE(estimator, n_features_to_select = 13)
selector = rfe.fit(x_train_names, np.ravel(y_train))


index = selector.support_  
x_train_rfe = x_train_names.iloc[:,index]
x_test_rfe = x_test_names.iloc[:,index]


## Done via primary component analysis 

### explained variance 
n_comp =15
pca = PCA(n_components = n_comp)
weights = pca.fit(x_train_names)

x_train_pca = weights.transform(x_train_names)
x_test_pca = weights.transform(x_test_names)
#------------------------------------------------------------------------------
# GENERATING OUTPUT

#choosing to use RFE for readability!! 

## cbind X and Y 



## output to csv





'''
NOTES:

Should totally learn a pipeline for this one when modeling


'''




















