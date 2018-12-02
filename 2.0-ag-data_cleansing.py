# -*- coding: utf-8 -*-
'''
Data cleansing File for 823 Assign A2 

Author: Andrew Garvey
Date : Dec 2nd, 2018

'''

#setup packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Assign\Assign2\Output'

#-------------------------------------------------------------------------------
# IMPORT FILES

## Read file(s) / directory management
os.chdir(inputdir)
train = pd.read_excel('A2trainData_MMAI.xlsx')
os.chdir(outputdir)