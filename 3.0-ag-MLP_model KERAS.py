# -*- coding: utf-8 -*-
'''
Data cleansing File for 823 Assign A2, using keras this time  

Author: Andrew Garvey
Date : Dec 3rd, 2018

'''

#import standard packages
import os
import time
import random as rd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import other packages 




#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Output'


## Read file(s) / directory management
os.chdir(inputdir)
test = pd.read_csv('2.0-ag-Test_Cleaned.csv',index_col = 0)
train = pd.read_csv('2.0-ag-Train_Cleaned.csv',index_col = 0)
os.chdir(outputdir)


#------------------------------------------------------------------------------
# Global Vars
random_state=123

#------------------------------------------------------------------------------
# Splits
x_train = train.drop('Output Return %',axis =1)
y_train = train.loc[:,['Output Return %']]

x_test = test.drop('Output Return %',axis =1)
y_test = test.loc[:,['Output Return %']]

#------------------------------------------------------------------------------
# Make Regression Model
start_time = time.time()











print("--- %s seconds ---" % (time.time() - start_time)) #output time for curiosity 