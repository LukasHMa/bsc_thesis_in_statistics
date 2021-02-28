# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:21:34 2021

@author: Work
"""
import pandas as pd 
import numpy as np
import sklearn

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix,brier_score_loss
from matplotlib import pyplot as plt

import os
os.chdir("C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data")


def normalise(prob):
    prob = (prob - prob.min()) / (prob.max() - prob.min())
    return prob


#%% 

#small scenario b0=-3

data_train = pd.read_csv("test_train_split//df_platt_1_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//df_platt_1_test.txt", sep='\t', low_memory = False)


#%%

#small scenario b0=-5

data_train = pd.read_csv("test_train_split//df_platt_2_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//df_platt_2_test.txt", sep='\t', low_memory = False)


#%%
Train_Y = pd.DataFrame(data_train, columns=['y'])

Test_Y = pd.DataFrame(data_test, columns=['y'])

Train_X = data_train.drop('y', axis=1) 

Test_X = data_test.drop('y', axis=1) 

#%%

LinearSVM = SVC(kernel='linear', probability= True, random_state = 1) #linear kernel

GauSVM = SVC(kernel='rbf', probability= True, random_state = 1)

# PolySVM = SVC(kernel='poly', degree = 3, probability= True, random_state = 1)

# parameter tunning 
param_grid_gau = {'C': [0.001, 0.1, 1, 5, 10, 50], 
              'kernel': ['rbf']}

param_grid_lin = {'C': [0.001, 0.1, 1, 5, 10, 50], 
              'kernel': ['linear']}

param_grid_pol = {'C': [0.001, 0.1, 1, 5, 10, 50], 
              'kernel': ['poly']}

#standard SVM
grid_gau = GridSearchCV(GauSVM, param_grid_gau, refit = True, verbose = 3) 

grid_lin = GridSearchCV(LinearSVM, param_grid_lin, refit = True, verbose = 3)

classifiers = [grid_lin,grid_gau]

# grid_pol = GridSearchCV(PolySVM, param_grid_pol, refit = True, verbose = 3, n_jobs=-1)

# classifiers = [grid_lin,grid_gau,grid_pol]

#%%
#cost sensitive SVM

LinearSVM_w = SVC(kernel='linear',class_weight='balanced', probability= True, random_state = 1) #linear kernel

GauSVM_w = SVC(kernel='rbf', class_weight='balanced', probability= True, random_state = 1)

grid_gau_w = GridSearchCV(GauSVM_w, param_grid_gau, refit = True, verbose = 3) 

grid_lin_w = GridSearchCV(LinearSVM_w, param_grid_lin, refit = True, verbose = 3)

classifiers_w = [grid_lin_w,grid_gau_w]
#%%

#fit with tuning

prediction_train = []
prediction_test = []
best_model = []
probability_train = []
probability_test = []
svm_out_train = []
svm_out_test = []

for clf in classifiers:
    clf.fit(Train_X, Train_Y.values.ravel())
    best_model.append(clf.best_estimator_)
    
    pred_train = clf.predict(Train_X)
    pred_test = clf.predict(Test_X)
    pred_train = pd.Series(index=Train_Y.index, data=pred_train)
    pred_test = pd.Series(index=Test_Y.index, data=pred_test)
    
    prediction_train.append(pred_train)
    prediction_test.append(pred_test)
    
    pred_p_train = clf.predict_proba(Train_X)
    pred_p_test = clf.predict_proba(Test_X)
    
    pred_p_train = pd.DataFrame(index=Train_Y.index, data=pred_p_train)
    pred_p_test = pd.DataFrame(index=Test_Y.index, data=pred_p_test)
    
    probability_train.append(pred_p_train)
    probability_test.append(pred_p_test)
    
    out_train = clf.decision_function(Train_X)
    out_test = clf.decision_function(Test_X)
    
    out_train = pd.DataFrame(index=Train_Y.index, data=out_train)
    out_test = pd.DataFrame(index=Test_Y.index, data=out_test)
    
    svm_out_train.append(out_train)
    svm_out_test.append(out_test)
    
    
#%%

#fitting cost-sensitive SVM

prediction_train_w = []
prediction_test_w = []
best_model_w = []
probability_train_w = []
probability_test_w = []
svm_out_train_w = []
svm_out_test_w = []

    
for clf in classifiers_w:
    clf.fit(Train_X, Train_Y.values.ravel())
    best_model_w.append(clf.best_estimator_)
    
    pred_train_w = clf.predict(Train_X)
    pred_test_w = clf.predict(Test_X)
    pred_train_w = pd.Series(index=Train_Y.index, data=pred_train_w)
    pred_test_w = pd.Series(index=Test_Y.index, data=pred_test_w)
    
    prediction_train_w.append(pred_train_w)
    prediction_test_w.append(pred_test_w)
    
    pred_p_train_w = clf.predict_proba(Train_X)
    pred_p_test_w = clf.predict_proba(Test_X)
    
    pred_p_train_w = pd.DataFrame(index=Train_Y.index, data=pred_p_train_w)
    pred_p_test_w = pd.DataFrame(index=Test_Y.index, data=pred_p_test_w)
    
    probability_train_w.append(pred_p_train_w)
    probability_test_w.append(pred_p_test_w)
    
    out_train_w = clf.decision_function(Train_X)
    out_test_w = clf.decision_function(Test_X)
    
    out_train_w = pd.DataFrame(index=Train_Y.index, data=out_train_w)
    out_test_w = pd.DataFrame(index=Test_Y.index, data=out_test_w)
    
    svm_out_train_w.append(out_train_w)
    svm_out_test_w.append(out_test_w)
    

#%%

# Convert SVM-scores to probability using normalisation (raw probability)

df_svm_prob_train = []
df_svm_prob_test = []
for i in range(0,len(svm_out_train)):
    convert_prob = normalise(svm_out_train[i])
    new_col = 1-convert_prob
    convert_prob = pd.DataFrame(index=Train_Y.index, data=convert_prob)
    convert_prob.insert(0, column='negative', value=new_col)
    df_svm_prob_train.append(convert_prob)
    
    convert_prob_test = normalise(svm_out_test[i])
    new_col_test = 1-convert_prob_test
    convert_prob_test = pd.DataFrame(index=Test_Y.index, data=convert_prob_test)
    convert_prob_test.insert(0, column='negative', value=new_col_test)
    df_svm_prob_test.append(convert_prob_test)

#%%

# Convert cost-sensitive SVM-scores to probability using normalisation (raw probability)

df_svm_prob_train_w = []
df_svm_prob_test_w = []
for i in range(0,len(svm_out_train_w)):
    convert_prob_w = normalise(svm_out_train_w[i])
    new_col_w = 1-convert_prob_w
    convert_prob_w = pd.DataFrame(index=Train_Y.index, data=convert_prob_w)
    convert_prob_w.insert(0, column='negative', value=new_col_w)
    df_svm_prob_train_w.append(convert_prob_w)
    
    convert_prob_test_w = normalise(svm_out_test_w[i])
    new_col_test_w = 1-convert_prob_test_w
    convert_prob_test_w = pd.DataFrame(index=Test_Y.index, data=convert_prob_test_w)
    convert_prob_test_w.insert(0, column='negative', value=new_col_test_w)
    df_svm_prob_test_w.append(convert_prob_test_w)

#%%
# probability

calib_lin = CalibratedClassifierCV(grid_lin, method='sigmoid', cv=5)

calib_gau = CalibratedClassifierCV(grid_gau, method='sigmoid', cv=5)

classifiers_calib = [calib_lin,calib_gau]


# calib_pol = CalibratedClassifierCV(grid_pol, method='sigmoid', cv=5, n_jobs=-1)

# classifiers_calib = [calib_lin,calib_gau, calib_pol]

prob_prediction_train = []
prob_prediction_test = []

calib_probability_train = []
calib_probability_test = []

for clf in classifiers_calib:
    clf.fit(Train_X, Train_Y.values.ravel())
    #prediction
    prob_pred_train = clf.predict(Train_X)
    prob_pred_test = clf.predict(Test_X)
    
    prob_pred_train = pd.Series(index=Train_Y.index, data=prob_pred_train)
    prob_pred_test = pd.Series(index=Test_Y.index, data=prob_pred_test)
    
    prob_prediction_train.append(prob_pred_train)
    prob_prediction_test.append(prob_pred_test)
    #calibrated probability
    cali_prob_train = clf.predict_proba(Train_X)
    cali_prob_test = clf.predict_proba(Test_X)
    
    cali_prob_train = pd.DataFrame(index=Train_Y.index, data=cali_prob_train)
    cali_prob_test = pd.DataFrame(index=Test_Y.index, data=cali_prob_test)
    
    calib_probability_train.append(cali_prob_train)
    calib_probability_test.append(cali_prob_test)
#%%

# Probability cost-senstive SVM

calib_lin_w = CalibratedClassifierCV(grid_lin_w, method='sigmoid', cv=5)

calib_gau_w = CalibratedClassifierCV(grid_gau_w, method='sigmoid', cv=5)

classifiers_calib_w = [calib_lin_w,calib_gau_w]

prob_prediction_train_w = []
prob_prediction_test_w = []

calib_probability_train_w = []
calib_probability_test_w = []

for clf in classifiers_calib_w:
    clf.fit(Train_X, Train_Y.values.ravel())
    #prediction
    prob_pred_train_w = clf.predict(Train_X)
    prob_pred_test_w = clf.predict(Test_X)
    
    prob_pred_train_w = pd.Series(index=Train_Y.index, data=prob_pred_train_w)
    prob_pred_test_w = pd.Series(index=Test_Y.index, data=prob_pred_test_w)
    
    prob_prediction_train_w.append(prob_pred_train_w)
    prob_prediction_test_w.append(prob_pred_test_w)
    #calibrated probability
    cali_prob_train_w = clf.predict_proba(Train_X)
    cali_prob_test_w = clf.predict_proba(Test_X)
    
    cali_prob_train_w = pd.DataFrame(index=Train_Y.index, data=cali_prob_train_w)
    cali_prob_test_w = pd.DataFrame(index=Test_Y.index, data=cali_prob_test_w)
    
    calib_probability_train_w.append(cali_prob_train_w)
    calib_probability_test_w.append(cali_prob_test_w)

    
    
#%%

fop, mpv = calibration_curve(Train_Y, calib_probability_train[0][1], n_bins=10)
fop, mpv = calibration_curve(Train_Y, df_svm_prob_train[0][0], n_bins=10)

#cost sensitive versions
fop, mpv = calibration_curve(Train_Y, calib_probability_train_w[0][1], n_bins=10)
fop, mpv = calibration_curve(Train_Y, df_svm_prob_train_w[0][0], n_bins=10)


# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
plt.plot(mpv, fop, marker='.')

#%%

fop, mpv = calibration_curve(Test_Y, calib_probability_test[0][1], n_bins=10)
fop, mpv = calibration_curve(Test_Y, df_svm_prob_test[0][0], n_bins=10)

#cost sensitive versions
fop, mpv = calibration_curve(Test_Y, calib_probability_test_w[0][1], n_bins=10)
fop, mpv = calibration_curve(Test_Y, df_svm_prob_test_w[0][0], n_bins=10)

# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
plt.plot(mpv, fop, marker='.')



#%%

#Export result

colnames_df = ['Negative','Positive','Group']

#Linear Standard SVM
df_lin_uncal_tr = df_svm_prob_train[0] # uncalibrated probability 
df_lin_pro_tr = calib_probability_train[0] # calibrated probability using platt scaling

data_frame_list_lin_tr = [df_lin_uncal_tr, df_lin_pro_tr]

df_lin_uncal_tr['Group'] = 'Uncalibrated'
df_lin_pro_tr['Group'] = 'Calibrated'

#compute brier score
brier_svm_tr_lin = []

for data in data_frame_list_lin_tr:
    data.columns = colnames_df
    brier = brier_score_loss(Train_Y, data['Positive'])
    brier_svm_tr_lin.append(brier)

df_platt_lin_tr = pd.concat([df_lin_uncal_tr, df_lin_pro_tr])

#%%

#Gaussian Standard SVM
df_gau_uncal_tr = df_svm_prob_train[1] # uncalibrated probability 
df_gau_pro_tr = calib_probability_train[1] # calibrated probability using platt scaling


data_frame_list_gau_tr = [df_gau_uncal_tr, df_gau_pro_tr]

df_gau_uncal_tr['Group'] = 'Uncalibrated'
df_gau_pro_tr['Group'] = 'Calibrated'

#compute brier score
brier_svm_tr_gau = []

for data in data_frame_list_gau_tr:
    data.columns = colnames_df
    brier = brier_score_loss(Train_Y, data['Positive'])
    brier_svm_tr_gau.append(brier)

df_platt_gau_tr = pd.concat([df_gau_uncal_tr, df_gau_pro_tr])

#%%
#----------------------------------------------------------
#Test Linear Standard SVM
df_lin_uncal_test = df_svm_prob_test[0] # uncalibrated probability 
df_lin_pro_test = calib_probability_test[0] # calibrated probability using platt scaling

data_frame_list_lin_test = [df_lin_uncal_test,df_lin_pro_test]

df_lin_uncal_test['Group'] = 'Uncalibrated'
df_lin_pro_test['Group'] = 'Calibrated'


brier_svm_lin_test = []

for data in data_frame_list_lin_test:
    data.columns = colnames_df
    brier = brier_score_loss(Test_Y, data['Positive'])
    brier_svm_lin_test.append(brier)                             

df_platt_lin_test = pd.concat([df_lin_uncal_test,df_lin_pro_test])

#%%
# Test Gaussian Standard SVM
df_gau_uncal_test = df_svm_prob_test[1] # uncalibrated probability 
df_gau_pro_test = calib_probability_test[1] # calibrated probability using platt scaling


data_frame_list_gau_test = [df_gau_uncal_test, df_gau_pro_test]

df_gau_uncal_test['Group'] = 'Uncalibrated'
df_gau_pro_test['Group'] = 'Calibrated'

#compute brier score
brier_svm_test_gau = []

for data in data_frame_list_gau_test :
    data.columns = colnames_df
    brier = brier_score_loss(Test_Y, data['Positive'])
    brier_svm_tr_gau.append(brier)

df_platt_gau_test = pd.concat([df_gau_uncal_test, df_gau_pro_test])
#%%
#Export to CSV
df_platt_lin_tr.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_lin_svm_tr.csv', index = False, header=True)
df_platt_gau_tr.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_gau_tr.csv', index = False, header=True)
df_platt_lin_test.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_lin_svm_test.csv', index = False, header=True)
df_platt_gau_test.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_gau_svm_test.csv', index = False, header=True)


#%%

# Results for the cost-sensitive version (apply only when imbalanced data was used)

#Export result

colnames_df = ['Negative','Positive','Group']

#Linear Standard SVM
df_lin_uncal_tr_w = df_svm_prob_train_w[0] # uncalibrated probability 
df_lin_pro_tr_w = calib_probability_train_w[0] # calibrated probability using platt scaling

data_frame_list_lin_tr_w = [df_lin_uncal_tr_w, df_lin_pro_tr_w]

df_lin_uncal_tr_w['Group'] = 'Uncalibrated'
df_lin_pro_tr_w['Group'] = 'Calibrated'

#compute brier score
brier_svm_tr_lin_w = []

for data in data_frame_list_lin_tr_w:
    data.columns = colnames_df
    brier = brier_score_loss(Train_Y, data['Positive'])
    brier_svm_tr_lin_w.append(brier)

df_platt_lin_tr_w = pd.concat([df_lin_uncal_tr_w, df_lin_pro_tr_w])

#%%

#Gaussian Standard SVM
df_gau_uncal_tr_w = df_svm_prob_train_w[1] # uncalibrated probability 
df_gau_pro_tr_w = calib_probability_train_w[1] # calibrated probability using platt scaling


data_frame_list_gau_tr_w = [df_gau_uncal_tr_w, df_gau_pro_tr_w]

df_gau_uncal_tr_w['Group'] = 'Uncalibrated'
df_gau_pro_tr_w['Group'] = 'Calibrated'

#compute brier score
brier_svm_tr_gau_w = []

for data in data_frame_list_gau_tr_w:
    data.columns = colnames_df
    brier = brier_score_loss(Train_Y, data['Positive'])
    brier_svm_tr_gau_w.append(brier)

df_platt_gau_tr_w = pd.concat([df_gau_uncal_tr_w, df_gau_pro_tr_w])

#%%
#----------------------------------------------------------
#Test Linear Standard SVM
df_lin_uncal_test_w = df_svm_prob_test_w[0] # uncalibrated probability 
df_lin_pro_test_w = calib_probability_test_w[0] # calibrated probability using platt scaling

data_frame_list_lin_test_w = [df_lin_uncal_test_w,df_lin_pro_test_w]

df_lin_uncal_test_w['Group'] = 'Uncalibrated'
df_lin_pro_test_w['Group'] = 'Calibrated'


brier_svm_lin_test_w = []

for data in data_frame_list_lin_test_w:
    data.columns = colnames_df
    brier = brier_score_loss(Test_Y, data['Positive'])
    brier_svm_lin_test_w.append(brier)                             

df_platt_lin_test_w = pd.concat([df_lin_uncal_test_w,df_lin_pro_test_w])

#%%
# Test Gaussian Standard SVM
df_gau_uncal_test_w = df_svm_prob_test_w[1] # uncalibrated probability 
df_gau_pro_test_w = calib_probability_test_w[1] # calibrated probability using platt scaling


data_frame_list_gau_test_w = [df_gau_uncal_test_w, df_gau_pro_test_w]

df_gau_uncal_test_w['Group'] = 'Uncalibrated'
df_gau_pro_test_w['Group'] = 'Calibrated'

#compute brier score
brier_svm_test_gau_w = []

for data in data_frame_list_gau_test_w :
    data.columns = colnames_df
    brier = brier_score_loss(Test_Y, data['Positive'])
    brier_svm_tr_gau_w.append(brier)

df_platt_gau_test_w = pd.concat([df_gau_uncal_test_w, df_gau_pro_test_w])
#%%
#Export to CSV
df_platt_lin_tr_w.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_lin_svm_tr_w.csv', index = False, header=True)
df_platt_gau_tr_w.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_gau_tr_w.csv', index = False, header=True)
df_platt_lin_test_w.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_lin_svm_test_w.csv', index = False, header=True)
df_platt_gau_test_w.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//platt_gau_svm_test_w.csv', index = False, header=True)


