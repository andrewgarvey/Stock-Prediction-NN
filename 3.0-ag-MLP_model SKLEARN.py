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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV 
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

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
# Make Regression Model
start_time = time.time()

mlp = MLPRegressor(random_state = random_state)  #initialize step(s)

pipe = Pipeline(steps=[('mlp',mlp)]) # group step(s) into pipe

param_grid = {                       #params for various step(s) 
    'mlp__hidden_layer_sizes' : [(100)],                            
    'mlp__activation':  ['logistic'],
    'mlp__solver': ['adam'],
    'mlp__learning_rate' : ['constant'],
    'mlp__max_iter' : [200],
    'mlp__batch_size' : ['auto']
}               


reg = GridSearchCV(pipe,param_grid=param_grid,cv=2,scoring = 'neg_mean_squared_error',n_jobs=-1, verbose =1000000) # make GridSearch

reg.fit(x_train,np.ravel(y_train))  # fit model 

pred = reg.predict(x_test)
#------------------------------------------------------------------------------
# Evaluate Model performance 

print(reg.best_params_)
print(-1*reg.best_score_)

print(mean_squared_error(y_test,pred))
print(r2_score(y_test,pred))

print("--- %s seconds ---" % (time.time() - start_time)) #output time for curiosity 