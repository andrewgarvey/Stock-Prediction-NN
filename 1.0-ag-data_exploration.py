# -*- coding: utf-8 -*-
'''
Data exploration File for 823 Assign A2 

Author: Andrew Garvey
Date : Dec 2nd, 2018

'''

#setup packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Output'

#-------------------------------------------------------------------------------
# IMPORT FILES

## Read file(s) / directory management
os.chdir(inputdir)
train = pd.read_excel('A2trainData_MMAI.xlsx')
os.chdir(outputdir)

#BASIC EXPLORING, using train stuff 

## head 
train.head()

## Summary
train.describe().transpose()

## Count the missings
train.isna().sum()

## Correlation Matrix, shamelessly stolen [1] (but this time a better one)

f, ax = plt.subplots(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
plt.savefig('1.0-ag-Correlation Matrix.png')
plt.clf()
#risk and return obv correlated,  productivity and Tobin's Q, nothing correlated with output


## Some basic looking at output 

returns=train.loc[:,'Output Return %']
returns.describe().transpose()


plt.hist(returns, bins=100, range= (-50,50))
plt.savefig('1.0-ag-Output Return %.png')
plt.clf()

'''
NOTES

normalize via some sort of preproccessing thing, make it apply to both
remove any really similarly correlated columns , done via RFE 
Outliers ok with me, we have some real distinct data here 

REFERENCES
[1] - https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas/31384328
'''