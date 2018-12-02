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
plt.savefig('Correlation Matrix.png')

#risk and return obv correlated,  


## Some basic looking at output 
returns=train.loc[:,'Output Return %']
returns.describe().transpose()

plt.hist(returns, bins=100, range= (-50,50))
plt.savefig('Output Return %.png')


'''
NOTES

append stuff together so its all done at once !!

normalize via some sort of preproccessing thing

remove any really similarly correlated columns , done via RFE ? else PCA or something

Outliers ok with me, we have some real distinct data here 

Consider YEAR of company? consider companies LAST years earnings? IDK might be good might be trash in a trash bag, and we'd have to get rid of the first year consistently ? 


REFERENCES
[1] - https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas/31384328
'''