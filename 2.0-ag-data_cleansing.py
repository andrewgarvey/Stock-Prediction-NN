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



#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Output'

#-------------------------------------------------------------------------------
# IMPORT FILES

## Read file(s) / directory management
os.chdir(inputdir)
test = pd.read_excel('A2testData_MMAI.xlsx')
train = pd.read_excel('A2trainData_MMAI.xlsx')
os.chdir(outputdir)


# drop year, can't even b
train = train.drop('Year',axis =1)
test = test.drop('Year',axis =1)



#split test/train X/Y
x_train = train.drop('Output Return %',axis =1)
y_train = train.loc[:,['Output Return %']]

x_test = test.drop('Output Return %',axis =1)
y_test = test.loc[:,['Output Return %']]

# create standard scaler
scaler = StandardScaler().fit(x_train)

scaler.transform(x_train)

#normalize via some sort of preproccessing thing



#




#remove any really similarly correlated columns , done via RFE 
#Outliers ok with me, we have some real distinct data here 
























