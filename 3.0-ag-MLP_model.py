'''
Modeling MLP Model File for 823 Assign A2 

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


#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Output'

#import other packages 
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


#------------------------------------------------------------------------------
# IMPORT FILES


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
# Regression Model
start_time = time.time()

mlp = MLPRegressor(random_state = random_state)
mlp.fit(x_train,np.ravel(y_train))

prediction = mlp.predict(x_test)

plt.plot(prediction, y_test, '.')

mean_squared_error(y_test,prediction), r2_score(y_test, prediction)

print("--- %s seconds ---" % (time.time() - start_time))