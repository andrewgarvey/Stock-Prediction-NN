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
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

#------------------------------------------------------------------------------
## Read file(s) / directory management

# CAC 
test = pd.read_csv('2.0-ag-Test_Cleaned_6sd.csv',index_col = 0)
train = pd.read_csv('2.0-ag-Train_Cleaned_6sd.csv',index_col = 0)

#Local 
'''
#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Output'

## Read file(s) / directory management
os.chdir(inputdir)
test = pd.read_csv('2.0-ag-Test_Cleaned_6sd.csv',index_col = 0)
train = pd.read_csv('2.0-ag-Train_Cleaned_6sd.csv',index_col = 0)
os.chdir(outputdir)
'''


#------------------------------------------------------------------------------
# X/Y Splits
x_train = train.drop('Output Return %',axis =1)
y_train = train.loc[:,['Output Return %']]

x_test = test.drop('Output Return %',axis =1)
y_test = test.loc[:,['Output Return %']]

#------------------------------------------------------------------------------
# Make NN Setup

def create_model(momentum, n_hidden_layers, n_neurons_L1, n_neurons_L2, activation, dropout_rate,loss, epochs, batch_size,learn_rate):
    if n_hidden_layers ==1 :
        #make
        model = Sequential()
        model.add(Dense(n_neurons_L1, input_dim=15, activation= activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation=activation))
        #compile
        optimizer = SGD(lr=learn_rate, momentum = momentum)
        model.compile(loss= loss, optimizer= optimizer, metrics=['mse'])
        return model

    else:  
        #make
        model = Sequential()
        model.add(Dense(n_neurons_L1, input_dim=15, activation= activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(n_neurons_L2, activation= activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation=activation))
        #compile
        optimizer = SGD(lr=learn_rate, momentum = momentum)
        model.compile(loss= loss, optimizer= optimizer, metrics=['mse'])
        return model

#------------------------------------------------------------------------------
## MAKE REGRESSION MODEL + GRID

# Seed
seed = 123
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=create_model,  epochs=10, batch_size = 10, verbose=0)

# define the grid search parameters
n_hidden_layers = [1,2]
n_neurons_L1 = [2,5] 
n_neurons_L2 = [2,5]
activation = ['sigmoid','relu']
learn_rate = [0.1,0.01]
dropout_rate = [0.0,0.3]
epochs = [10,50]
batch_size = [200,500]
loss = ['mean_squared_error']
momentum = [0.0]


#param grid
param_grid = dict(n_hidden_layers = n_hidden_layers,
                  n_neurons_L1 = n_neurons_L1,
                  n_neurons_L2 = n_neurons_L2,
                  activation = activation,
                  learn_rate = learn_rate,
                  momentum = momentum,
                  dropout_rate = dropout_rate,
                  epochs = epochs,
                  batch_size = batch_size,
                  loss = loss
                  )

#setup grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=4,verbose=1, cv=2)

#Train Model 
grid = grid.fit(x_train,np.ravel(y_train))


pred = grid.predict(x_test)

print('best params ->',grid.best_params_)
print('mse train ->',-1*grid.best_score_)

print('mse test ->',mean_squared_error(y_test,pred))
print('r2 test ->',r2_score(y_test,pred))