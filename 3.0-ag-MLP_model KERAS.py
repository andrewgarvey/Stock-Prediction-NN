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
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Output'


## Read file(s) / directory management
os.chdir(inputdir)
test = pd.read_csv('2.0-ag-Test_Cleaned_6sd.csv',index_col = 0)
train = pd.read_csv('2.0-ag-Train_Cleaned_6sd.csv',index_col = 0)
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

#STOLEN MOSTLY RIGHT NOW 

# Function to create model, required for KerasClassifier
def create_model():
	#make
	model = Sequential()
	model.add(Dense(12, input_dim=15, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	#compile
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
	return model

# fix random seed for reproducibility
seed = 123
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=create_model,  epochs=10, batch_size = 10, verbose=0)

# define the grid search parameters


#hidden_size = [1,2]
n_neurons_L1 = [1,5,10] 
#n_neurons_L2 = [1,5,10]

activation = ['relu','sigmoid']
learn_rate = [0.001, 0.01]
dropout_rate = [0.0, 0.2]
epochs = [10]
batch_size = [200]


#param grid
param_grid = dict(batch_size=batch_size, epochs=epochs)


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,verbose=10)
grid_result = grid.fit(x_train,np.ravel(y_train))

pred = grid.predict(x_test)

print(reg.best_params_)
print(-1*reg.best_score_)

print(mean_squared_error(y_test,pred))
print(r2_score(y_test,pred))




print("--- %s seconds ---" % (time.time() - start_time)) #output time for curiosity 
