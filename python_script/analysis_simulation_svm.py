
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
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix,brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve 
import time


import os
os.chdir("C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data")

start_time = time.time() #Start1
#%% 

#small scenario b0=-3

data_train = pd.read_csv("test_train_split//simulation_df_small3_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//simulation_df_small3_test.txt", sep='\t', low_memory = False)

#%%

#small scenario b0=-5

data_train = pd.read_csv("test_train_split//simulation_df_small5_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//simulation_df_small5_test.txt", sep='\t', low_memory = False)

#%%

#medium scenario b0=-3

data_train = pd.read_csv("test_train_split//simulation_df_med3_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//simulation_df_med3_test.txt", sep='\t', low_memory = False)

#%%

#medium scenario b0=-5

data_train = pd.read_csv("test_train_split//simulation_df_med5_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//simulation_df_med5_test.txt", sep='\t', low_memory = False)
#%%

#medium scenario b0=-7

data_train = pd.read_csv("test_train_split//simulation_df_med7_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//simulation_df_med7_test.txt", sep='\t', low_memory = False)

#%%

#Large scenario b0=-3

data_train = pd.read_csv("test_train_split//simulation_df_lar3_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//simulation_df_lar3_test.txt", sep='\t', low_memory = False)

#%%

#Large scenario b0=-5

data_train = pd.read_csv("test_train_split//simulation_df_lar5_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//simulation_df_lar5_test.txt", sep='\t', low_memory = False)

#%%

#Large scenario b0=-7

data_train = pd.read_csv("test_train_split//simulation_df_lar7_train.txt", sep='\t', low_memory = False)

data_test = pd.read_csv("test_train_split//simulation_df_lar7_test.txt", sep='\t', low_memory = False)


#%%
Train_Y = pd.DataFrame(data_train, columns=['y'])

Test_Y = pd.DataFrame(data_test, columns=['y'])

Train_X = data_train.drop('y', axis=1) 

Test_X = data_test.drop('y', axis=1) 

#%%
# fit a model
LinearSVM = SVC(kernel='linear', probability= True, random_state = 1) #linear kernel

GauSVM = SVC(kernel='rbf', probability= True, random_state = 1)

SigmSVM = SVC(kernel='sigmoid', probability= True, random_state = 1)

PolySVM = SVC(kernel='poly', degree = 3, probability= True, random_state = 1)

# fit a model (weighted)
LinearSVM_w = SVC(kernel='linear',class_weight='balanced', probability= True, random_state = 1) #linear kernel

GauSVM_w = SVC(kernel='rbf', class_weight='balanced', probability= True, random_state = 1)

SigmSVM_w = SVC(kernel='sigmoid', class_weight='balanced', probability= True, random_state = 1)

PolySVM_w = SVC(kernel='poly', degree = 3, class_weight='balanced', probability= True, random_state = 1)

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
grid_gau = GridSearchCV(GauSVM, param_grid_gau, refit = True, verbose = 3) 

grid_lin = GridSearchCV(LinearSVM, param_grid_lin, refit = True, verbose = 3)

grid_sig = GridSearchCV(SigmSVM, param_grid_sig, refit = True, verbose = 3) 

grid_pol = GridSearchCV(PolySVM, param_grid_pol, refit = True, verbose = 3)



#Cost sensitive SVM
grid_gau_w = GridSearchCV(GauSVM_w, param_grid_gau, refit = True, verbose = 3) 

grid_lin_w = GridSearchCV(LinearSVM_w, param_grid_lin, refit = True, verbose = 3)

grid_sig_w = GridSearchCV(SigmSVM_w, param_grid_sig, refit = True, verbose = 3) 

grid_pol_w = GridSearchCV(PolySVM_w, param_grid_pol, refit = True, verbose = 3)    

classifiers = [grid_lin,grid_gau,grid_sig,grid_pol]

classifiers_w = [grid_lin_w,grid_gau_w,grid_sig_w,grid_pol_w]
#fit with tuning

prediction_train = []
prediction_test = []
best_model = []
probability_train = []
probability_test = []

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
    
prediction_train_w = []
prediction_test_w = []
best_model_w = []
probability_train_w = []
probability_test_w = []
    
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

elapsed_time = (time.time() - start_time)/60

print(elapsed_time)


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

print(best_model)
print(best_model_w)
#Print results
#In-sample

#%%
#create confusion matrix for in-sample classification
axis_title = ['Linear kernel, C=0.001',
              'Gaussian kernel, C=0.001',
              'Sigmoid kernel, C=0.001',
              'Polynomial kernel, C=0.1']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
# fig.suptitle('In-sample accuracy (Scenario II, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario II, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario II, b = -7)',fontsize=20)
fig.suptitle('In-sample accuracy (Scenario I, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario I, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario III, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario III, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario III, b = -7)',fontsize=20)


for cls, ax, title in zip(classifiers, axes.flatten(), axis_title):
    plot_confusion_matrix(cls, 
                          Train_X, 
                          Train_Y,
                          labels=[1, 0],
                          ax=ax, 
                         cmap=plt.cm.Blues)
    ax.title.set_text(title)
            
plt.tight_layout()  
# plt.savefig('svm_in_med_3.png', dpi=300)
# plt.savefig('svm_in_med_5.png', dpi=300) 
# plt.savefig('svm_in_med_7.png', dpi=300) 
plt.savefig('svm_in_small_3.png', dpi=300) 
# plt.savefig('svm_in_small_5.png', dpi=300)  
plt.show()

#%%
#Out-of-sample

#create confusion matrix for in-sample classification
axis_title = ['Linear kernel, C=0.001',
              'Gaussian kernel, C=0.001',
              'Sigmoid kernel, C=0.001',
              'Polynomial kernel, C=0.1']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -5)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -7)',fontsize=20)
fig.suptitle('Out-of-sample accuracy (Scenario I, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario I, b = -5)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario III, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy(Scenario III, b = -5)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario III, b = -7)',fontsize=20)


for cls, ax, title in zip(classifiers, axes.flatten(), axis_title):
    plot_confusion_matrix(cls, 
                          Test_X, 
                          Test_Y,
                          labels=[1, 0], 
                          ax=ax, 
                         cmap=plt.cm.Blues)
    ax.title.set_text(title)
            
plt.tight_layout()  
# plt.savefig('svm_out_med_3.png', dpi=300)
# plt.savefig('svm_out_med_5.png', dpi=300) 
# plt.savefig('svm_out_med_7.png', dpi=300) 
plt.savefig('svm_out_small_3.png', dpi=300) 
# plt.savefig('svm_out_small_5.png', dpi=300) 
plt.show()

#%%

#Weighted SVM  

#create confusion matrix for in-sample classification
axis_title = ['Linear kernel, C=1 (DEC)',
              'Gaussian kernel, C=0.1  (DEC)',
              'Sigmoid kernel, C=0.1 (DEC)',
              'Polynomial kernel, C=0.001 (DEC)']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
# fig.suptitle('In-sample accuracy (Scenario II, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario II, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario II, b = -7)',fontsize=20)
fig.suptitle('In-sample accuracy, DEC (Scenario I, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario I, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario III, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario III, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario III, b = -7)',fontsize=20)



for cls, ax, title in zip(classifiers_w, axes.flatten(), axis_title):
    plot_confusion_matrix(cls, 
                          Train_X, 
                          Train_Y, 
                          ax=ax,
                          labels=[1, 0], 
                         cmap=plt.cm.Blues,
                         values_format="d")
    ax.title.set_text(title)
            
plt.tight_layout()  
# plt.savefig('svm_in_med_3_w.png', dpi=300)
# plt.savefig('svm_in_med_5_w.png', dpi=300) 
# plt.savefig('svm_in_med_7_w.png', dpi=300) 
plt.savefig('svm_in_small_3_w.png', dpi=300) 
# plt.savefig('svm_in_small_5_w.png', dpi=300)
# plt.savefig('svm_in_large_3_w.png', dpi=300) 
# plt.savefig('svm_in_large_5_w.png', dpi=300) 
# plt.savefig('svm_in_large_7_w.png', dpi=300)   
plt.show()

#%%
#Out-of-sample

#create confusion matrix for in-sample classification
axis_title = ['Linear kernel, C=1 (DEC)',
              'Gaussian kernel, C=0.1  (DEC)',
              'Sigmoid kernel, C=0.1 (DEC)',
              'Polynomial kernel, C=0.001 (DEC)']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -5)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -7)',fontsize=20)
fig.suptitle('Out-of-sample accuracy (Scenario I, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario I, b = -5)',fontsize=20)

for cls, ax, title in zip(classifiers_w, axes.flatten(), axis_title):
    plot_confusion_matrix(cls, 
                          Test_X, 
                          Test_Y, 
                          ax=ax, 
                          labels=[1, 0], 
                         cmap=plt.cm.Blues,
                         values_format="d")
    ax.title.set_text(title)
            
plt.tight_layout()  
# plt.savefig('svm_out_med_3_w.png', dpi=300)
# plt.savefig('svm_out_med_5_w.png', dpi=300) 
# plt.savefig('svm_out_med_7_w.png', dpi=300) 
plt.savefig('svm_out_small_3_w.png', dpi=300) 
# plt.savefig('svm_out_small_5_w.png', dpi=300)  
plt.show()


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
    
# Confusion matrices

confusion_train_result = []
confusion_train_result_w = []
confusion_test_result = []
confusion_test_result_w = []


for i in range(4):
    train_result = confusion_matrix(Train_Y,prediction_train[i])
    print(train_result)
    confusion_train_result.append(train_result)
    
    train_result_w = confusion_matrix(Train_Y,prediction_train_w[i])
    print(train_result_w)
    confusion_train_result_w.append(train_result_w)
    
    test_result = confusion_matrix(Test_Y,prediction_test[i])
    print(test_result)
    confusion_test_result.append(test_result)
    
    test_result_w = confusion_matrix(Test_Y,prediction_test_w[i])
    print(test_result_w)
    confusion_test_result_w.append(test_result_w)



#%%
# probability

calib_lin = CalibratedClassifierCV(grid_lin, method='sigmoid', cv="prefit")

calib_gau = CalibratedClassifierCV(grid_gau, method='sigmoid', cv="prefit")

calib_sig = CalibratedClassifierCV(grid_sig, method='sigmoid', cv="prefit")

calib_pol = CalibratedClassifierCV(grid_pol, method='sigmoid', cv="prefit")

calib_lin_w = CalibratedClassifierCV(grid_lin_w, method='sigmoid', cv="prefit")

calib_gau_w = CalibratedClassifierCV(grid_gau_w, method='sigmoid', cv="prefit")

calib_sig_w = CalibratedClassifierCV(grid_sig_w, method='sigmoid', cv="prefit")

calib_pol_w = CalibratedClassifierCV(grid_pol_w, method='sigmoid', cv="prefit")

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
    #caliberated probability
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
    #caliberated probability
    cali_prob_train_w = clf.predict_proba(Train_X)
    cali_prob_test_w = clf.predict_proba(Test_X)
    
    cali_prob_train_w = pd.DataFrame(index=Train_Y.index, data=cali_prob_train_w)
    cali_prob_test_w = pd.DataFrame(index=Test_Y.index, data=cali_prob_test_w)
    
    calib_probability_train_w.append(cali_prob_train_w)
    calib_probability_test_w.append(cali_prob_test_w)

#%%
#In-sample

#create PR for in-sample classification
axis_title = ['Linear kernel, C=0.001',
              'Gaussian kernel, C=0.001',
              'Sigmoid kernel, C=0.001',
              'Polynomial kernel, C=0.1']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
# fig.suptitle('In-sample accuracy (Scenario II, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario II, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario II, b = -7)',fontsize=20)
fig.suptitle('In-sample accuracy (Scenario I, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario I, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario III, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario III, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy (Scenario III, b = -7)',fontsize=20)


for clf, ax, title in zip(classifiers_calib, axes.flatten(), axis_title):
    plot_precision_recall_curve(clf, 
                         Train_X, 
                          Train_Y,
                          pos_label = 1,
                          response_method='predict_proba',
                          ax=ax)
    ax.title.set_text(title)
            
plt.tight_layout()  
# plt.savefig('svm_in_med_3_pr.png', dpi=300)
# plt.savefig('svm_in_med_5_pr.png', dpi=300) 
# plt.savefig('svm_in_med_7_pr.png', dpi=300) 
plt.savefig('svm_in_small_3_pr.png', dpi=300) 
# plt.savefig('svm_in_small_5_pr.png', dpi=300)  
plt.show()

#%%

#Out-of-sample

#create PR curve for in-sample classification
axis_title = ['Linear kernel, C=0.001',
              'Gaussian kernel, C=0.001',
              'Sigmoid kernel, C=0.001',
              'Polynomial kernel, C=0.1']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -5)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -7)',fontsize=20)
fig.suptitle('Out-of-sample accuracy (Scenario I, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario I, b = -5)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario III, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy(Scenario III, b = -5)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario III, b = -7)',fontsize=20)


for clf, ax, title in zip(classifiers_calib, axes.flatten(), axis_title):
    plot_precision_recall_curve(clf, 
                          Test_X, 
                          Test_Y,
                          pos_label = 1,
                          response_method='predict_proba',
                          ax=ax)
    ax.title.set_text(title)
            
plt.tight_layout()  
# plt.savefig('svm_out_med_3_pr.png', dpi=300)
# plt.savefig('svm_out_med_5_pr.png', dpi=300) 
# plt.savefig('svm_out_med_7_pr.png', dpi=300) 
plt.savefig('svm_out_small_3_pr.png', dpi=300) 
# plt.savefig('svm_out_small_5_pr.png', dpi=300) 
plt.show()
#%%

#In-sample

#create  PR curve for in-sample classification
axis_title = ['Linear kernel, C=1 (DEC)',
              'Gaussian kernel, C=0.1  (DEC)',
              'Sigmoid kernel, C=0.1 (DEC)',
              'Polynomial kernel, C=0.001 (DEC)']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
# fig.suptitle('In-sample accuracy (Scenario II, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario II, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario II, b = -7)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario I, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario I, b = -5)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario III, b = -3)',fontsize=20)
# fig.suptitle('In-sample accuracy, DEC (Scenario III, b = -5)',fontsize=20)
fig.suptitle('In-sample accuracy, DEC (Scenario III, b = -7)',fontsize=20)


for clf, ax, title in zip(classifiers_calib_w, axes.flatten(), axis_title):
    plot_precision_recall_curve(clf, 
                         Train_X, 
                          Train_Y,
                          pos_label = 1,
                          response_method='predict_proba',
                          ax=ax)
    ax.title.set_text(title)
            
plt.tight_layout()  
# plt.savefig('svm_in_med_3_pr_w.png', dpi=300)
# plt.savefig('svm_in_med_5_pr_w.png', dpi=300) 
# plt.savefig('svm_in_med_7_pr_w.png', dpi=300) 
# plt.savefig('svm_in_small_3_pr_w.png', dpi=300) 
# plt.savefig('svm_in_small_5_pr_w.png', dpi=300)
# plt.savefig('svm_in_lar_3_pr_w.png', dpi=300)
# plt.savefig('svm_in_lar_5_pr_w.png', dpi=300) 
plt.savefig('svm_in_lar_7_pr_w.png', dpi=300)  
plt.show()

#%%

#Out-of-sample

#create  PR curve for in-sample classification
axis_title = ['Linear kernel, C=1 (DEC)',
              'Gaussian kernel, C=0.1  (DEC)',
              'Sigmoid kernel, C=0.1 (DEC)',
              'Polynomial kernel, C=0.001 (DEC)']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -5)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario II, b = -7)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario I, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario I, b = -5)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario III, b = -3)',fontsize=20)
# fig.suptitle('Out-of-sample accuracy (Scenario III, b = -5)',fontsize=20)
fig.suptitle('Out-of-sample accuracy (Scenario III, b = -7)',fontsize=20)


for clf, ax, title in zip(classifiers_calib_w, axes.flatten(), axis_title):
    plot_precision_recall_curve(clf, 
                          Test_X, 
                          Test_Y,
                          pos_label = 1,
                          response_method='predict_proba',
                          ax=ax)
    ax.title.set_text(title)
            
plt.tight_layout()  
# plt.savefig('svm_out_med_3_pr_w.png', dpi=300)
# plt.savefig('svm_out_med_5_pr_w.png', dpi=300) 
# plt.savefig('svm_out_med_7_pr_w.png', dpi=300) 
# plt.savefig('svm_out_small_3_pr_w.png', dpi=300) 
# plt.savefig('svm_out_small_5_pr_w.png', dpi=300)
# plt.savefig('svm_out_lar_3_pr_w.png', dpi=300)
# plt.savefig('svm_out_lar_5_pr_w.png', dpi=300) 
plt.savefig('svm_out_lar_7_pr_w.png', dpi=300) 
plt.show()
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
df_svm_pro_tr.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//standard_svm_tr.csv', index = False, header=True)
df_svm_pro_test.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//standard_svm_test.csv', index = False, header=True)
df_svm_pro_tr_DEC.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//DEC_svm_tr.csv', index = False, header=True)
df_svm_pro_test_DEC.to_csv (r'C://Users//Work//OneDrive//Universitetsstudier//Kurser//HT2019//Kandidatuppsats//Rare events//DATA//Simulated_data//DEC_svm_test.csv', index = False, header=True)
