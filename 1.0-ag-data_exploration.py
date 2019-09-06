# -*- coding: utf-8 -*-
'''
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
test = pd.read_excel('A2testData_MMAI.xlsx')
os.chdir(outputdir)

#BASIC EXPLORING, using train stuff 

train_year=train.loc[:,'Year']

test_year=test.loc[:,'Year']

plt.clf()
plt.hist(train_year, bins=15)
plt.clf()
plt.hist(test_year, bins=15)

## head 
train.head()

## Summary
train.describe().transpose()

## Count the missings
train.isna().sum() 

## Correlation Matrix, shamelessly stolen again [1] (but this time a better one)
plt.clf()
f, ax = plt.subplots(figsize=(15, 15))
corr = train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot =True)
plt.savefig('1.0-ag-Correlation Matrix.png')




## Some basic looking at output 

returns=train.loc[:,'Output Return %']
returns.describe().transpose()  

Z = (returns-np.mean(returns))/np.std(returns)
Z = Z.sort_values()
Z.head()
Z.tail()
#yeah we got a single value ~93 std away, and the next one is ~17 away... 
#thats almost certainly a typo or something equally crazy, should remove it

plt.clf()
plt.ylabel(' Frequency')
plt.xlabel('Output Return %')
plt.title('Frequency of Output Return %s')
plt.hist(returns, bins=50, range= (-50,50))
plt.savefig('1.0-ag-Output Return %.png')


'''
NOTES

'''


