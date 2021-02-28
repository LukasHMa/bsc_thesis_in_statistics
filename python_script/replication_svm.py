# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:50:12 2020

@author: Work
"""
# SVM reliability diagram with calibration
import pandas as pd 
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, brier_score_loss
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import GridSearchCV
import time #measuring elapsed time
import matplotlib.pyplot as plt


import os
os.chdir("C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA")

def normalise(prob):
    prob = (prob - prob.min()) / (prob.max() - prob.min())
    return prob


#%%
start_time = time.time() #Start1

data_train = pd.read_csv("replication_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("replication_test.txt", sep='\t', low_memory = False)

Train_Y = pd.DataFrame(data_train, columns=['y'])

Test_Y = pd.DataFrame(data_test, columns=['y'])

Train_X = pd.DataFrame(data_train, columns=['x2','x4','x5','x6','z1','z2','z5','z6','x11','x1'])

Test_X = pd.DataFrame(data_test, columns=['x2','x4','x5','x6','z1','z2','z5','z6','x11','x1'])

#%%
# Initialise and define different SVMs
LinearSVM = SVC(kernel='linear', probability= True, random_state = 1) #linear kernel

GauSVM = SVC(kernel='rbf', probability= True, random_state = 1) #gaussian kernel

SigmSVM = SVC(kernel='sigmoid', probability= True, random_state = 1) #sigmoid kernel

PolySVM = SVC(kernel='poly', degree = 3, probability= True, random_state = 1) #polynomial kernel

# Initialise and define different SVMs (cost-sensitive)
LinearSVM_w = SVC(kernel='linear',class_weight='balanced', probability= True, random_state = 1) #linear kernel - cost-sensitive version

GauSVM_w = SVC(kernel='rbf', class_weight='balanced', probability= True, random_state = 1) #gaussian kernel - cost-sensitive version

SigmSVM_w = SVC(kernel='sigmoid', class_weight='balanced', probability= True, random_state = 1) #sigmoid kernel - cost-sensitive version

PolySVM_w = SVC(kernel='poly', degree = 3, class_weight='balanced', probability= True, random_state = 1) #polynomial kernel - cost-sensitive version

# parameter tunning 
param_grid_gau = {'C': [0.001, 0.1, 1, 5, 10, 50], 
              'kernel': ['rbf']}

param_grid_lin = {'C': [0.001, 0.1, 1, 5, 10, 50], 
              'kernel': ['linear']}

param_grid_sig = {'C': [0.001, 0.1, 1, 5, 10, 50], 
              'kernel': ['sigmoid']}

param_grid_pol = {'C': [0.001, 0.1, 1, 5, 10, 50], 
              'kernel': ['poly']}

#standard SVM
grid_gau = GridSearchCV(GauSVM, param_grid_gau, refit = True, verbose = 3, n_jobs=-1) 

grid_lin = GridSearchCV(LinearSVM, param_grid_lin, refit = True, verbose = 3, n_jobs=-1)

grid_sig = GridSearchCV(SigmSVM, param_grid_sig, refit = True, verbose = 3, n_jobs=-1) 

grid_pol = GridSearchCV(PolySVM, param_grid_pol, refit = True, verbose = 3, n_jobs=-1)



#Cost sensitive SVM
grid_gau_w = GridSearchCV(GauSVM_w, param_grid_gau, refit = True, verbose = 3, n_jobs=-1) 

grid_lin_w = GridSearchCV(LinearSVM_w, param_grid_lin, refit = True, verbose = 3, n_jobs=-1)

grid_sig_w = GridSearchCV(SigmSVM_w, param_grid_sig, refit = True, verbose = 3, n_jobs=-1) 

grid_pol_w = GridSearchCV(PolySVM_w, param_grid_pol, refit = True, verbose = 3, n_jobs=-1)    

classifiers = [grid_lin,grid_gau,grid_sig,grid_pol]

classifiers_w = [grid_lin_w,grid_gau_w,grid_sig_w,grid_pol_w]
#fit with tuning

#%%

# fit SVM
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
    
    #prediction
    pred_train = clf.predict(Train_X)
    pred_test = clf.predict(Test_X)
    pred_train = pd.Series(index=Train_Y.index, data=pred_train)
    pred_test = pd.Series(index=Test_Y.index, data=pred_test)
    
    prediction_train.append(pred_train)
    prediction_test.append(pred_test)
    
    #probability output (alternative calibration)
    pred_p_train = clf.predict_proba(Train_X)
    pred_p_test = clf.predict_proba(Test_X)
    
    pred_p_train = pd.DataFrame(index=Train_Y.index, data=pred_p_train)
    pred_p_test = pd.DataFrame(index=Test_Y.index, data=pred_p_test)
    
    probability_train.append(pred_p_train)
    probability_test.append(pred_p_test)
    
    #SVM score (distance to hyperplane)
    out_train = clf.decision_function(Train_X)
    out_test = clf.decision_function(Test_X)
    
    out_train = pd.DataFrame(index=Train_Y.index, data=out_train)
    out_test = pd.DataFrame(index=Test_Y.index, data=out_test)
    
    svm_out_train.append(out_train)
    svm_out_test.append(out_test)


# fit cost-sensitive SVMs    
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
    
     #prediction
    pred_train_w = clf.predict(Train_X)
    pred_test_w = clf.predict(Test_X)
    pred_train_w = pd.Series(index=Train_Y.index, data=pred_train_w)
    pred_test_w = pd.Series(index=Test_Y.index, data=pred_test_w)
    
    prediction_train_w.append(pred_train_w)
    prediction_test_w.append(pred_test_w)
    
    # probability output
    pred_p_train_w = clf.predict_proba(Train_X)
    pred_p_test_w = clf.predict_proba(Test_X)
    
    pred_p_train_w = pd.DataFrame(index=Train_Y.index, data=pred_p_train_w)
    pred_p_test_w = pd.DataFrame(index=Test_Y.index, data=pred_p_test_w)
    
    probability_train_w.append(pred_p_train_w)
    probability_test_w.append(pred_p_test_w)
    
    # SVM score (distance to hyperplane)
    out_train_w = clf.decision_function(Train_X)
    out_test_w = clf.decision_function(Test_X)
    
    out_train_w = pd.DataFrame(index=Train_Y.index, data=out_train_w)
    out_test_w = pd.DataFrame(index=Test_Y.index, data=out_test_w)
    
    svm_out_train_w.append(out_train_w)
    svm_out_test_w.append(out_test_w)
    

elapsed_time = (time.time() - start_time)/60

print(elapsed_time)


#%%

# Convert SVM-scores to probability using normalisation (raw probability)

df_svm_prob_train = []
df_svm_prob_test = []
colnames_scores = ['Negative','Positive']
for i in range(0,len(svm_out_train)):
    convert_prob = normalise(svm_out_train[i])
    new_col = 1-convert_prob
    convert_prob = pd.DataFrame(index=Train_Y.index, data=convert_prob)
    convert_prob.insert(0, column='Negative', value=new_col)
    convert_prob.columns = colnames_scores
    df_svm_prob_train.append(convert_prob)
    
    convert_prob_test = normalise(svm_out_test[i])
    new_col_test = 1-convert_prob_test
    convert_prob_test = pd.DataFrame(index=Test_Y.index, data=convert_prob_test)
    convert_prob_test.insert(0, column='Negative', value=new_col_test)
    convert_prob_test.columns = colnames_scores
    df_svm_prob_test.append(convert_prob_test)

# Convert cost-sensitive SVM-scores to probability using normalisation (raw probability)

df_svm_prob_train_w = []
df_svm_prob_test_w = []
for i in range(0,len(svm_out_train_w)):
    convert_prob_w = normalise(svm_out_train_w[i])
    new_col_w = 1-convert_prob_w
    convert_prob_w = pd.DataFrame(index=Train_Y.index, data=convert_prob_w)
    convert_prob_w.insert(0, column='Negative', value=new_col_w)
    convert_prob_w.columns = colnames_scores
    df_svm_prob_train_w.append(convert_prob_w)
    
    convert_prob_test_w = normalise(svm_out_test_w[i])
    new_col_test_w = 1-convert_prob_test_w
    convert_prob_test_w = pd.DataFrame(index=Test_Y.index, data=convert_prob_test_w)
    convert_prob_test_w.insert(0, column='Negative', value=new_col_test_w)
    convert_prob_test_w.columns = colnames_scores
    df_svm_prob_test_w.append(convert_prob_test_w)

#%%

#Export uncalibrated probabilities

colnames_df = ['Negative','Positive','Group']

#Train Standard SVM
df_lin_pro_tr = df_svm_prob_train[0]
df_gau_pro_tr = df_svm_prob_train[1]
df_sig_pro_tr = df_svm_prob_train[2]
df_poly_pro_tr = df_svm_prob_train[3]

data_frame_list_tr = [df_lin_pro_tr, df_gau_pro_tr, df_sig_pro_tr, df_poly_pro_tr]

df_lin_pro_tr['Group'] = 'Linear SVM'
df_gau_pro_tr['Group'] = 'Gaussian SVM'
df_sig_pro_tr['Group'] = 'Sigmoid SVM'
df_poly_pro_tr['Group'] = 'Polynomial SVM'

#compute brier score
brier_svm_tr = []

for data in data_frame_list_tr:
    data.columns = colnames_df
    brier = brier_score_loss(Train_Y, data['Positive'])
    brier_svm_tr.append(brier)

df_svm_pro_tr = pd.concat([df_lin_pro_tr, df_gau_pro_tr, df_sig_pro_tr,df_poly_pro_tr])

#%%
#----------------------------------------------------------
#Test Standard SVM
df_lin_pro_test = df_svm_prob_test[0]
df_gau_pro_test = df_svm_prob_test[1]
df_sig_pro_test = df_svm_prob_test[2]
df_poly_pro_test = df_svm_prob_test[3]

data_frame_list_test = [df_lin_pro_test,df_gau_pro_test,df_sig_pro_test,df_poly_pro_test]


df_lin_pro_test['Group'] = 'Linear SVM'
df_gau_pro_test['Group'] = 'Gaussian SVM'
df_sig_pro_test['Group'] = 'Sigmoid SVM'
df_poly_pro_test['Group'] = 'Polynomial SVM'

brier_svm_test = []

for data in data_frame_list_test:
    data.columns = colnames_df
    brier = brier_score_loss(Test_Y, data['Positive'])
    brier_svm_test.append(brier)                             

df_svm_pro_test = pd.concat([df_lin_pro_test,df_gau_pro_test,df_sig_pro_test,df_poly_pro_test])
#%%
#------------------------------------------------------------------
#Train Cost-sensitive SVM
df_lin_pro_tr_w = df_svm_prob_train_w[0]
df_gau_pro_tr_w = df_svm_prob_train_w[1]
df_sig_pro_tr_w = df_svm_prob_train_w[2]
df_poly_pro_tr_w = df_svm_prob_train_w[3]

data_frame_list_tr_w = [df_lin_pro_tr_w,df_gau_pro_tr_w,df_sig_pro_tr_w,df_poly_pro_tr_w]

df_lin_pro_tr_w['Group'] = 'Linear SVM-DEC'
df_gau_pro_tr_w['Group'] = 'Gaussian SVM-DEC'
df_sig_pro_tr_w['Group'] = 'Sigmoid SVM-DEC'
df_poly_pro_tr_w['Group'] = 'Polynomial SVM-DEC'

brier_svm_tr_w = []

for data in data_frame_list_tr_w:
    data.columns = colnames_df
    brier = brier_score_loss(Train_Y, data['Positive'])
    brier_svm_tr_w.append(brier)

df_svm_pro_tr_DEC = pd.concat([df_lin_pro_tr_w,df_gau_pro_tr_w,df_sig_pro_tr_w,df_poly_pro_tr_w])

#%%
#-------------------------------------------------------------------
#Test Cost-sensitive SVM
df_lin_pro_test_w = df_svm_prob_test_w[0]
df_gau_pro_test_w = df_svm_prob_test_w[1]
df_sig_pro_test_w = df_svm_prob_test_w[2]
df_poly_pro_test_w = df_svm_prob_test_w[3]

data_frame_list_test_w = [df_lin_pro_test_w,df_gau_pro_test_w,df_sig_pro_test_w,df_poly_pro_test_w]

df_lin_pro_test_w['Group'] = 'Linear SVM-DEC'
df_gau_pro_test_w['Group'] = 'Gaussian SVM-DEC'
df_sig_pro_test_w['Group'] = 'Sigmoid SVM-DEC'
df_poly_pro_test_w['Group'] = 'Polynomial SVM-DEC'

brier_svm_test_w = []

for data in data_frame_list_test_w:
    data.columns = colnames_df
    brier = brier_score_loss(Test_Y, data['Positive'])
    brier_svm_test_w.append(brier)   

df_svm_pro_test_DEC = pd.concat([df_lin_pro_test_w,df_gau_pro_test_w,df_sig_pro_test_w,df_poly_pro_test_w])

#%%
#Export to CSV
df_svm_pro_tr.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//standard_svm_tr.csv', index = False, header=True)
df_svm_pro_test.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//standard_svm_test.csv', index = False, header=True)
df_svm_pro_tr_DEC.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//DEC_svm_tr.csv', index = False, header=True)
df_svm_pro_test_DEC.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//DEC_svm_test.csv', index = False, header=True)

#%%

#Extract information about support vectors etc. 

prob_best_model = []
prob_model_bias = []
prob_support = []
prob_support_num = []

for clf in classifiers:
    prob_best_model.append(clf.best_estimator_.dual_coef_)
    prob_model_bias.append(clf.best_estimator_.intercept_)
    prob_support.append(clf.best_estimator_.support_vectors_)
    prob_support_num.append(clf.best_estimator_.n_support_)

prob_best_model_w = []
prob_model_bias_w = []
prob_support_w = []
prob_support_num_w = []

for clf in classifiers_w:
    prob_best_model_w.append(clf.best_estimator_.dual_coef_)
    prob_model_bias_w.append(clf.best_estimator_.intercept_)
    prob_support_w.append(clf.best_estimator_.support_vectors_)
    prob_support_num_w.append(clf.best_estimator_.n_support_)

#Print results
#In-sample

#%%

#Classification reports

class_train_result = []
class_train_result_w = []
class_test_result = []
class_test_result_w = []


for i in range(4):
    train_result = classification_report(Train_Y,prediction_train[i])
    print(train_result)
    class_train_result.append(train_result)
    
    train_result_w = classification_report(Train_Y,prediction_train_w[i])
    print(train_result_w)
    class_train_result_w.append(train_result_w)
    
    test_result = classification_report(Test_Y,prediction_test[i])
    print(test_result)
    class_test_result.append(test_result)
    
    test_result_w = classification_report(Test_Y,prediction_test_w[i])
    print(test_result_w)
    class_test_result_w.append(test_result_w)

    
    #Output

#%%
# probability

# Define SVMs with the resulted C caliberated

# Standard SVM
print(best_model)

LinearSVM_prob = SVC(kernel='linear', C=0.001, probability= True, random_state = 1) #linear kernel

GauSVM_prob = SVC(kernel='rbf', C=0.001, probability= True, random_state = 1)

SigmSVM_prob = SVC(kernel='sigmoid', C=0.001, probability= True, random_state = 1)

PolySVM_prob = SVC(kernel='poly', C=0.001, degree = 3, probability= True, random_state = 1)


# SVM (weighted)
print(best_model_w)

LinearSVM_prob_w = SVC(kernel='linear', C=0.001, class_weight='balanced', probability= True, random_state = 1) #linear kernel

GauSVM_prob_w = SVC(kernel='rbf', C=50, class_weight='balanced', probability= True, random_state = 1)

SigmSVM_prob_w = SVC(kernel='sigmoid', C=0.001, class_weight='balanced', probability= True, random_state = 1)

PolySVM_prob_w = SVC(kernel='poly', C=50, degree = 3, class_weight='balanced', probability= True, random_state = 1)

#%%

calib_lin = CalibratedClassifierCV(LinearSVM_prob, method='sigmoid', cv=5, n_jobs=-1)

calib_gau = CalibratedClassifierCV(GauSVM_prob, method='sigmoid', cv=5, n_jobs=-1)

calib_sig = CalibratedClassifierCV(SigmSVM_prob, method='sigmoid', cv=5, n_jobs=-1)

calib_pol = CalibratedClassifierCV(PolySVM_prob, method='sigmoid', cv=5, n_jobs=-1)

calib_lin_w = CalibratedClassifierCV(LinearSVM_prob_w, method='sigmoid', cv=5, n_jobs=-1)

calib_gau_w = CalibratedClassifierCV(GauSVM_prob_w, method='sigmoid', cv=5, n_jobs=-1)

calib_sig_w = CalibratedClassifierCV(SigmSVM_prob_w, method='sigmoid', cv=5, n_jobs=-1)

calib_pol_w = CalibratedClassifierCV(PolySVM_prob_w, method='sigmoid', cv=5, n_jobs=-1)

classifiers_calib = [calib_lin,calib_gau,calib_sig,calib_pol]

classifiers_calib_w = [calib_lin_w,calib_gau_w,calib_sig_w,calib_pol_w]


#%%

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
#Export result

colnames_df = ['Negative','Positive','Group']

#Train Standard SVM
df_lin_pro_tr = calib_probability_train[0]
df_gau_pro_tr = calib_probability_train[1]
df_sig_pro_tr = calib_probability_train[2]
df_poly_pro_tr = calib_probability_train[3]

data_frame_list_tr = [df_lin_pro_tr, df_gau_pro_tr, df_sig_pro_tr, df_poly_pro_tr]

df_lin_pro_tr['Group'] = 'Linear SVM'
df_gau_pro_tr['Group'] = 'Gaussian SVM'
df_sig_pro_tr['Group'] = 'Sigmoid SVM'
df_poly_pro_tr['Group'] = 'Polynomial SVM'

#compute brier score
brier_svm_tr = []

for data in data_frame_list_tr:
    data.columns = colnames_df
    brier = brier_score_loss(Train_Y, data['Positive'])
    brier_svm_tr.append(brier)

df_svm_pro_tr = pd.concat([df_lin_pro_tr, df_gau_pro_tr, df_sig_pro_tr,df_poly_pro_tr])

#%%
#----------------------------------------------------------
#Test Standard SVM
df_lin_pro_test = calib_probability_test[0]
df_gau_pro_test = calib_probability_test[1]
df_sig_pro_test = calib_probability_test[2]
df_poly_pro_test = calib_probability_test[3]

data_frame_list_test = [df_lin_pro_test,df_gau_pro_test,df_sig_pro_test,df_poly_pro_test]


df_lin_pro_test['Group'] = 'Linear SVM'
df_gau_pro_test['Group'] = 'Gaussian SVM'
df_sig_pro_test['Group'] = 'Sigmoid SVM'
df_poly_pro_test['Group'] = 'Polynomial SVM'

brier_svm_test = []

for data in data_frame_list_test:
    data.columns = colnames_df
    brier = brier_score_loss(Test_Y, data['Positive'])
    brier_svm_test.append(brier)                             

df_svm_pro_test = pd.concat([df_lin_pro_test,df_gau_pro_test,df_sig_pro_test,df_poly_pro_test])
#%%
#------------------------------------------------------------------
#Train Cost-sensitive SVM
df_lin_pro_tr_w = calib_probability_train_w[0]
df_gau_pro_tr_w = calib_probability_train_w[1]
df_sig_pro_tr_w = calib_probability_train_w[2]
df_poly_pro_tr_w = calib_probability_train_w[3]

data_frame_list_tr_w = [df_lin_pro_tr_w,df_gau_pro_tr_w,df_sig_pro_tr_w,df_poly_pro_tr_w]

df_lin_pro_tr_w['Group'] = 'Linear SVM-DEC'
df_gau_pro_tr_w['Group'] = 'Gaussian SVM-DEC'
df_sig_pro_tr_w['Group'] = 'Sigmoid SVM-DEC'
df_poly_pro_tr_w['Group'] = 'Polynomial SVM-DEC'

brier_svm_tr_w = []

for data in data_frame_list_tr_w:
    data.columns = colnames_df
    brier = brier_score_loss(Train_Y, data['Positive'])
    brier_svm_tr_w.append(brier)

df_svm_pro_tr_DEC = pd.concat([df_lin_pro_tr_w,df_gau_pro_tr_w,df_sig_pro_tr_w,df_poly_pro_tr_w])

#%%
#-------------------------------------------------------------------
#Test Cost-sensitive SVM
df_lin_pro_test_w = calib_probability_test_w[0]
df_gau_pro_test_w = calib_probability_test_w[1]
df_sig_pro_test_w = calib_probability_test_w[2]
df_poly_pro_test_w = calib_probability_test_w[3]

data_frame_list_test_w = [df_lin_pro_test_w,df_gau_pro_test_w,df_sig_pro_test_w,df_poly_pro_test_w]

df_lin_pro_test_w['Group'] = 'Linear SVM-DEC'
df_gau_pro_test_w['Group'] = 'Gaussian SVM-DEC'
df_sig_pro_test_w['Group'] = 'Sigmoid SVM-DEC'
df_poly_pro_test_w['Group'] = 'Polynomial SVM-DEC'

brier_svm_test_w = []

for data in data_frame_list_test_w:
    data.columns = colnames_df
    brier = brier_score_loss(Test_Y, data['Positive'])
    brier_svm_test_w.append(brier)   

df_svm_pro_test_DEC = pd.concat([df_lin_pro_test_w,df_gau_pro_test_w,df_sig_pro_test_w,df_poly_pro_test_w])

#%%
#Export to CSV
df_svm_pro_tr.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//standard_svm_tr.csv', index = False, header=True)
df_svm_pro_test.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//standard_svm_test.csv', index = False, header=True)
df_svm_pro_tr_DEC.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//DEC_svm_tr.csv', index = False, header=True)
df_svm_pro_test_DEC.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//DEC_svm_test.csv', index = False, header=True)

#%%

#calibration curve train

fop, mpv = calibration_curve(Train_Y, calib_probability_train[1]['Positive'], n_bins=10) #train
fop, mpv = calibration_curve(Train_Y, calib_probability_train_w[3]['Positive'], n_bins=10) #train_w

# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
plt.plot(mpv, fop, marker='.')

#%%

#calibration curve test

fop, mpv = calibration_curve(Test_Y, calib_probability_test[1]['Positive'], n_bins=10) #test
fop, mpv = calibration_curve(Test_Y, calib_probability_test_w[3]['Positive'], n_bins=10) #test
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
plt.plot(mpv, fop, marker='.')

#%%
#calibration curve train

fop, mpv = calibration_curve(Train_Y, df_svm_prob_train[1]['Positive'], n_bins=20,strategy='quantile') #train
fop, mpv = calibration_curve(Train_Y, df_svm_prob_train[4]['Positive'], n_bins=20,strategy='quantile') #train_w

# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
plt.plot(mpv, fop, marker='.')

#%%

#calibration curve test

fop, mpv = calibration_curve(Test_Y, df_svm_prob_test[0]['Positive'], n_bins=10) #test
fop, mpv = calibration_curve(Test_Y, df_svm_prob_test[4]['Positive'], n_bins=10) #test
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
plt.plot(mpv, fop, marker='.')
