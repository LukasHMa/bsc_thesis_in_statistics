rm(list = ls())

library(foreign)
library(ggplot2)
library(MASS)
library(stargazer)
library(tidyverse)
library(devtools)
library(dplyr)
library(sandwich)
library(Zelig)
library(ggplot2)
library(MASS)
library(pROC) # to construct ROC curve
library(ROCR) # to construct ROC curve
library(e1071)  # SVM methodology
library(caret) #confusion matrix
library(MLeval)

Sys.setlocale("LC_ALL","English")


data_model <- read.table(file.choose(),header=TRUE)

source_gist("https://gist.github.com/mrdwab/6424112") #import the stratification function

# set.seed(12345) #small sample
set.seed(123) #medium sample

#split the dataset into training and test sets randomly 

#70% training set
data_model.train <- stratified(data_model, "y", 0.7, replace = FALSE)

#30% test set
#1: use rownames to extract index
#2: specify the observations in full sample whose index NOT in train data
data_model.test <- data_model[which(!rownames(data_model) %in% rownames(data_model.train)),]


table(data_model.train$y)

table(data_model.test$y)

#rarity comparison

table(data_model$y)[2]/sum(table(data_model$y)[1:2])

table(data_model.train$y)[2]/sum(table(data_model.train$y)[1:2])

table(data_model.test$y)[2]/sum(table(data_model.test$y)[1:2])

# store rarity 
r.train <- mean(data_model.train$y)
r.test <- mean(data_model.test$y)

##### data export ######

# # df_small -3
# write.table(data_model.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_small3_train.txt", sep="\t")
# write.table(data_model.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_small3_test.txt", sep="\t")

# df_small -5
# write.table(data_model.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_small5_train.txt", sep="\t")
# write.table(data_model.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_small5_test.txt", sep="\t")

# df_medium -3
# write.table(data_model.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_med3_train.txt", sep="\t")
# write.table(data_model.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_med3_test.txt", sep="\t")

# # df_medium -5
# write.table(data_model.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_med5_train.txt", sep="\t")
# write.table(data_model.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_med5_test.txt", sep="\t")

# # df_medium -7
# write.table(data_model.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_med7_train.txt", sep="\t")
# write.table(data_model.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_med7_test.txt", sep="\t")

# # df_large -3
# write.table(data_model.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_lar3_train.txt", sep="\t")
# write.table(data_model.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_lar3_test.txt", sep="\t")

# # df_large -5
# write.table(data_model.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_lar5_train.txt", sep="\t")
# write.table(data_model.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_lar5_test.txt", sep="\t")

# # df_large -7
# write.table(data_model.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_lar7_train.txt", sep="\t")
# write.table(data_model.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_df_lar7_test.txt", sep="\t")


########################## logit  #######################
#Estimate logistic model (small)
m.logit <- glm(as.factor(y)~x1+x2+x3+x4+x5,family = "binomial"(link = "logit"),data = data_model.train)
summary(m.logit)


#Estimate logistic model (medium)
m.logit <- glm(as.factor(y)~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10,family = "binomial"(link = "logit"),data = data_model.train)
summary(m.logit)


#Estimate logistic model (large
m.logit <- glm(as.factor(y)~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19+x20,family = "binomial"(link = "logit"),data = data_model.train)
summary(m.logit)

############# King & Zengs correction ################

# Estimate the relogit model (small)
m.relogit <- zelig(y~x1+x2+x3+x4+x5, 
                   data = data_model.train, model = "relogit", bias.correct = TRUE, case.control = "weighting")
summary(m.relogit)

# Estimate the relogit model (medium)
m.relogit <- zelig(y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10, 
                   data = data_model.train, model = "relogit", bias.correct = TRUE, case.control = "weighting")
summary(m.relogit)

# Estimate the relogit model (large)
m.relogit <- zelig(y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19+x20, 
                   data = data_model.train, model = "relogit", bias.correct = TRUE, case.control = "weighting")
summary(m.relogit)

############ cloglog ##################

#Estimate the cloglog model (small)
m.cloglog <- glm(as.factor(y)~x1+x2+x3+x4+x5,family = "binomial"(link = "cloglog"),data = data_model.train)
summary(m.cloglog)

#Estimate the cloglog model (medium)
m.cloglog <- glm(as.factor(y)~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10,family = "binomial"(link = "cloglog"),data = data_model.train)
summary(m.cloglog)

#Estimate the cloglog model (large)
m.cloglog <- glm(as.factor(y)~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19+x20,family = "binomial"(link = "cloglog"),data = data_model.train, maxit=100)
summary(m.cloglog)

############# confusion matrix (in sample) ###########
#mlogit 
plogit.tr <- predict(m.logit,data_model.train, type = 'response')
confusionMatrix(data = as.factor(as.numeric(plogit.tr>0.5)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(plogit.tr>r.train)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")

#relogit
# Sigmoid function
sigmoid <- function(x) {
  exp(x)/(1 + exp(x))
}

coef.relogit <- coef(m.relogit)
train.matrix <- subset(data_model.train, select = -y)
design.relogit.manual.tr <- cbind(1, as.matrix(train.matrix))
pred.relogit.manual.tr <- design.relogit.manual.tr %*% coef.relogit
pred.relogit.manual.tr <- sigmoid(pred.relogit.manual.tr)

confusionMatrix(data = as.factor(as.numeric(pred.relogit.manual.tr>0.5)), mode = "prec_recall",reference = as.factor(data_model.train$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(pred.relogit.manual.tr>r.train)), mode = "prec_recall",reference = as.factor(data_model.train$y),positive="1")

#cloglog
pcloglog.tr <- predict(m.cloglog,data_model.train, type = 'response')
confusionMatrix(data = as.factor(as.numeric(pcloglog.tr>0.5)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(pcloglog.tr>r.train)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")


############ confusion matrix (out-of-sample) ##########
#mlogit
plogit <- predict(m.logit,data_model.test, type = 'response')
confusionMatrix(data = as.factor(as.numeric(plogit>0.5)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(plogit>r.test)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")


#relogit
coef.relogit <- coef(m.relogit)
test.matrix <- subset(data_model.test, select = -y)
design.relogit.manual <- cbind(1, as.matrix(test.matrix))
pred.relogit.manual <- design.relogit.manual %*% coef.relogit
pred.relogit.manual <- sigmoid(pred.relogit.manual)

confusionMatrix(data = as.factor(as.numeric(pred.relogit.manual>0.5)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(pred.relogit.manual>r.test)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")

#cloglog
pcloglog <- predict(m.cloglog,data_model.test, type = 'response')
confusionMatrix(data = as.factor(as.numeric(pcloglog>0.5)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(pcloglog>r.test)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")


############# ROC & AUC (in-sample) ##############

#mlogit
pred.logit <- prediction(fitted(m.logit), data_model.train$y)  
perf.logit <- performance(pred.logit, "tpr", "fpr") 
plot(perf.logit) #plot ROC
auc.tmp <- performance(pred.logit,"auc"); auc.logit <- as.numeric(auc.tmp@y.values) #calculate AUC

#relogit
pred.relogit <- prediction(fitted(m.relogit), data_model.train$y)  
perf.relogit <- performance(pred.relogit, "tpr", "fpr") 
plot(perf.relogit) #plot ROC
auc.tmp.relogit <- performance(pred.relogit,"auc"); auc.relogit <- as.numeric(auc.tmp.relogit@y.values) #calculate AUC

#mcloglog
pred.cloglog <- prediction(fitted(m.cloglog), data_model.train$y)  
perf.cloglog <- performance(pred.cloglog, "tpr", "fpr") 
plot(perf.cloglog) #plot ROC
auc.tmp.cloglog <- performance(pred.cloglog,"auc"); auc.cloglog <- as.numeric(auc.tmp.cloglog@y.values) #calculate AUC

########## ROC & AUC (out-of-sample) ###############

#mlogit
pred.logit.out <- prediction(predict(m.logit,data_model.test), data_model.test$y)  
perf.logit.out <- performance(pred.logit.out, "tpr", "fpr") 
plot(perf.logit.out) #plot ROC
auc.tmp.out <- performance(pred.logit.out,"auc"); auc.logit.out <- as.numeric(auc.tmp.out@y.values) #calculate AUC

#relogit
coef.relogit <- coef(m.relogit)
design.relogit.manual <- cbind(1, as.matrix(test.matrix))
pred.relogit.manual <- design.relogit.manual %*% coef.relogit
pred.relogit.manual <- sigmoid(pred.relogit.manual)

pred.relogit.out <- prediction(pred.relogit.manual, data_model.test$y) 
perf.relogit.out<- performance(pred.relogit.out, "tpr", "fpr") 
plot(perf.relogit.out) #plot ROC
auc.tmp.relogit.out<- performance(pred.relogit.out,"auc"); auc.relogit.out<- as.numeric(auc.tmp.relogit.out@y.values) #calculate AUC

#mcloglog
pred.cloglog.out <- prediction(predict(m.cloglog,data_model.test), data_model.test$y) 
perf.cloglog.out <- performance(pred.cloglog.out, "tpr", "fpr") 
plot(perf.cloglog.out) #plot ROC
auc.tmp.cloglog.out <- performance(pred.cloglog.out,"auc"); auc.cloglog.out <- as.numeric(auc.tmp.cloglog.out@y.values) #calculate AUC


#####Probability ######
#brier score

#logit
br.mlogit.tr <- mean((plogit.tr-data_model.train$y)^2) #train
print(br.mlogit.tr)

br.mlogit <- mean((plogit-data_model.test$y)^2) #test
print(br.mlogit)

#relogit
br.relogit.tr <- mean((pred.relogit.manual.tr-data_model.train$y)^2) #train
print(br.relogit.tr)

br.relogit <- mean((pred.relogit.manual-data_model.test$y)^2) #test
print(br.relogit)

#mcloglog
br.cloglog.tr <- mean((pcloglog.tr-data_model.train$y)^2) #train
print(br.cloglog.tr)

br.cloglog <- mean((pcloglog-data_model.test$y)^2) #test
print(br.cloglog)


#Probability histogram

#if small, use 0.05 as binwidth

plogit.tr.data <- as.data.frame(plogit.tr)
pred.relogit.manual.tr.data <- as.data.frame(pred.relogit.manual.tr)
pcloglog.tr.data <- as.data.frame(pcloglog.tr)

colnames(plogit.tr.data) <- 'y'
colnames(pred.relogit.manual.tr.data) <- 'y'
colnames(pcloglog.tr.data) <- 'y'

plogit.tr.data$link <- 'Logit'  #create a column of identification tag
pred.relogit.manual.tr.data$link <- 'Relogit'
pcloglog.tr.data$link <- 'C log-log'

GLM.pred.tr <- rbind.data.frame(plogit.tr.data, pred.relogit.manual.tr.data, pcloglog.tr.data)

prob.plot.tr <- ggplot(GLM.pred.tr, aes(x=y, fill= link)) + geom_histogram(binwidth=0.01, alpha=0.7, position="identity") + geom_vline(aes(xintercept = 0.5), size=0.5, linetype="dashed")
print(prob.plot.tr)


#test
plogit.data <- as.data.frame(plogit)
pred.relogit.manual.data <- as.data.frame(pred.relogit.manual)
pcloglog.data <- as.data.frame(pcloglog)

colnames(plogit.data) <- 'y'
colnames(pred.relogit.manual.data) <- 'y'
colnames(pcloglog.data) <- 'y'

plogit.data$link <- 'Logit'  #create a column of identification tag
pred.relogit.manual.data$link <- 'Relogit'
pcloglog.data$link <- 'C log-log'

GLM.pred <- rbind.data.frame(plogit.data, pred.relogit.manual.data, pcloglog.data)

prob.plot <- ggplot(GLM.pred, aes(x=y, fill= link)) + geom_histogram(binwidth=0.01, alpha=0.7, position="identity") + geom_vline(aes(xintercept = 0.5), size=0.5, linetype="dashed")
print(prob.plot)

########## Preparation for PR and ROC plots #########
Y.train <- data_model.train$y
Y.Test <- data_model.test$y

############ PR curve #########
#redefine Y.Train 
Y.Train.char <- as.data.frame(Y.train)
Y.Train.char$Y.train[Y.train == 1] <- 'Positive'  
Y.Train.char$Y.train[Y.train == 0] <- 'Negative'

plogit.pr.tr <- data.frame('Negative'=1-plogit.tr,'Positive' =plogit.tr, obs=Y.Train.char$Y.train)
pcloglog.pr.tr <- data.frame('Negative'=1-pcloglog.tr ,'Positive' =pcloglog.tr, obs=Y.Train.char$Y.train)
prelogit.pr.tr <- data.frame('Negative'=1-pred.relogit.manual.tr,'Positive' =pred.relogit.manual.tr, obs=Y.Train.char$Y.train)

plogit.pr.tr$Group <- 'Logit'
prelogit.pr.tr$Group <- 'Relogit'
pcloglog.pr.tr$Group <- 'C log-log' 

GLM.pr.tr <- rbind.data.frame(plogit.pr.tr, prelogit.pr.tr, pcloglog.pr.tr)

pr.GLM.tr <- evalm(GLM.pr.tr,fsize=20,dlinecol='black')

#redefine Y.test
Y.test.char <- as.data.frame(Y.Test)
Y.test.char$Y.Test[Y.Test == 1] <- 'Positive'  
Y.test.char$Y.Test[Y.Test == 0] <- 'Negative'

plogit.pr.test <- data.frame('Negative'=1-plogit,'Positive' =plogit, obs=Y.test.char$Y.Test)
pcloglog.test <- data.frame('Negative'=1-pcloglog ,'Positive' =pcloglog, obs=Y.test.char$Y.Test)
prelogit.pr.test <- data.frame('Negative'=1-pred.relogit.manual,'Positive' =pred.relogit.manual, obs=Y.test.char$Y.Test)

plogit.pr.test$Group <- 'Logit'
prelogit.pr.test$Group <- 'Relogit'
pcloglog.test$Group <- 'C log-log' 

GLM.pr.test <- rbind.data.frame(plogit.pr.test, prelogit.pr.test, pcloglog.test)

pr.GLM.test <- evalm(GLM.pr.test,fsize=20,dlinecol='black')


############ Probabilities from SVM ###############
path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\small_b3\\'
# path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\small_b5\\'
# path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\med_b3\\'
# path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\med_b5\\'
# path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\med_b7\\'
# path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\large_b3\\'
# path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\large_b5\\'

standardSVM.train <- read.csv(paste0(path_SVM,'standard_svm_tr.csv'))
standardSVM.test <- read.csv(paste0(path_SVM,'standard_svm_test.csv'))
DEC.SVM.train <- read.csv(paste0(path_SVM,'DEC_svm_tr.csv'))
DEC.SVM.test <- read.csv(paste0(path_SVM,'DEC_svm_test.csv'))

col_order <- c("Negative", "Positive", "obs",
               "Group")

standardSVM.train$obs <- Y.Train.char$Y.train
DEC.SVM.train$obs <- Y.Train.char$Y.train
standardSVM.test$obs <- Y.test.char$Y.Test
DEC.SVM.test$obs <- Y.test.char$Y.Test

standardSVM.train <- standardSVM.train[, col_order]
DEC.SVM.train <- DEC.SVM.train[, col_order]
standardSVM.test <- standardSVM.test[, col_order]
DEC.SVM.test <- DEC.SVM.test[, col_order]

pr.SVM.train <- evalm(standardSVM.train,fsize=20,dlinecol='black',plots = c('pr','cc'))
pr.SVM.test <- evalm(standardSVM.test,fsize=20,dlinecol='black',plots = c('pr','cc'))
pr.DEC.train <- evalm(DEC.SVM.train,fsize=20,dlinecol='black',plots = c('pr','cc'))
pr.DEC.test <- evalm(DEC.SVM.test,fsize=20,dlinecol='black',plots = c('pr','cc'))


########## Generate ROC curves for comparison between GLM and SVM ##########

### Merge probability data frame of GLM with that of SVM 
GLM_SVM.train <- rbind.data.frame(GLM.pr.tr,standardSVM.train) #GLM vs. standard SVM train
GLM_SVM.test <- rbind.data.frame(GLM.pr.test,standardSVM.test) #GLM vs. standard SVM test
GLM_DEC.train <- rbind.data.frame(GLM.pr.tr,DEC.SVM.train) #GLM vs. SVM-DEC train
GLM_DEC.test <- rbind.data.frame(GLM.pr.tr,DEC.SVM.test) #GLM vs. SVM-DEC test

### Plot
roc.GLM_SVM.train <- evalm(GLM_SVM.train,fsize=20,dlinecol='black', plots = 'r') #GLM vs. standard SVM train
roc.GLM_SVM.test <- evalm(GLM_SVM.test,fsize=20,dlinecol='black', plots = 'r') #GLM vs. standard SVM test
roc.GLM_DEC.train <- evalm(GLM_DEC.train,fsize=20,dlinecol='black', plots = 'r') #GLM vs. SVM-DEC train
roc.GLM_DEC.test <- evalm(GLM_DEC.test,fsize=20,dlinecol='black', plots = 'r') #GLM vs. SVM-DEC test



