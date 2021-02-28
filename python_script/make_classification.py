# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:33:38 2021

@author: Work
"""

import pandas as pd 
import numpy as np
import sklearn

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix,brier_score_loss
from matplotlib import pyplot as plt

def normalise(prob):
    prob = (prob - prob.min()) / (prob.max() - prob.min())
    return prob

# generate 2 class dataset

colnames_df = ['y','x1','x2','x3','x4','x5']

X, y = make_classification(n_samples=500, n_classes=2, weights=[1,1], random_state=1,n_features=5, n_informative=5, n_redundant=0)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
df_platt_1 = pd.concat([y,X], axis=1)
df_platt_1.columns = colnames_df

#%%

X, y = make_classification(n_samples=500, n_classes=2, weights=[0.7,0.3], random_state=1,n_features=5, n_informative=5, n_redundant=0)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
df_platt_2 = pd.concat([y,X], axis=1)
df_platt_2.columns = colnames_df

df_platt_1.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_make_1.csv', index = False, header=True)
df_platt_2.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_make_2.csv', index = False, header=True)


