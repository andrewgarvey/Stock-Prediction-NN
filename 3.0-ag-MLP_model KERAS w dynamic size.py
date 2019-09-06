# -*- coding: utf-8 -*-
'''
Data Modeling 
Date : Dec 6th, 2018
'''

#import standard packages
import os
import datetime
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

# THIS WAS RUN ON THE CAC
# -----------------------------------------------------------------------------
## Read file(s) / directory management

test = pd.read_csv('2.0-ag-Test_Cleaned_6sd.csv',index_col = 0)
train = pd.read_csv('2.0-ag-Train_Cleaned_6sd.csv',index_col = 0)


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

def create_model(momentum, n_hidden_layers, n_neurons_L1, n_neurons_middle, activation, dropout_rate,loss, epochs, batch_size,learn_rate):
    #make input layer
    model = Sequential()
    model.add(Dense(n_neurons_L1, input_dim=15, activation= activation))
    model.add(Dropout(dropout_rate))
    
    
    #make hidden layers
    for i in range(n_hidden_layers):
        model.add(Dense(n_neurons_middle, activation=activation))
        model.add(Dropout(dropout_rate))
    
    
    #make output layers
    model.add(Dense(1, activation=activation))
    #compile
    optimizer = SGD(lr=learn_rate, momentum = momentum)
    model.compile(loss= loss, optimizer= optimizer, metrics=['mse'])
    return model


#------------------------------------------------------------------------------
# Seed
seed = 123
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=create_model,  epochs=10, batch_size = 10, verbose=0)

# define the grid search parameters
n_hidden_layers = [0,1]
n_neurons_L1 = [5,10] 
n_neurons_middle = [5,10]
activation = ['sigmoid','relu']
learn_rate = [0.1,0.01]
dropout_rate = [0.0,0.2]
epochs = [50,100]
batch_size = [200,500]
loss = ['mean_squared_error']
momentum = [0.0]


#param grid
param_grid = dict(n_hidden_layers = n_hidden_layers,
                  n_neurons_L1 = n_neurons_L1,
                  n_neurons_middle = n_neurons_middle,
                  activation = activation,
                  learn_rate = learn_rate,
                  momentum = momentum,
                  dropout_rate = dropout_rate,
                  epochs = epochs,
                  batch_size = batch_size,
                  loss = loss
                  )


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,verbose=1, cv=2)


fit_model = grid.fit(x_train,np.ravel(y_train))
#------------------------------------------------------------------------------
#Results

pred = fit_model.predict(x_test) 


print(fit_model.best_params_)
print('mse train ->',-1*fit_model.best_score_)

print('mse test->',mean_squared_error(y_test,pred))
print('r2 ->' ,r2_score(y_test,pred))
