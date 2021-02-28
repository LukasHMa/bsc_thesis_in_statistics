rm(list=ls())


# install.packages("tidyverse")
# install.packages("devtools")
# devtools::install_github('IQSS/Zelig')
# install.packages('https://cran.r-project.org/src/contrib/Archive/Zelig/Zelig_5.1.5.tar.gz', repos = NULL, type = 'source')
# install.packages('caret')
# install.packages('MLeval')
library(foreign)
library(ggplot2)
library(MASS)
library(stargazer)
library(ROCR) # to construct ROC curve
library(tidyverse)
library(devtools)
library(dplyr)
library(sandwich)
library(Zelig)
library(caret) #confusion matrix
library(reshape2)
library(MLeval)

Sys.setlocale("LC_ALL","English")
# path_dat <- "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\dataverse_files\\SF.dat"
path_dat2 <- "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\dataverse_files\\model.dta"

# data_test <- read.table(file.choose(),header=TRUE)
# data_dat <- read.table(path_dat, skip=3)
# path_dta <- file.choose() 
data_model <- read.dta(path_dat2)

data_model$z1 <- pmax(data_model$x7,data_model$x8)

data_model$z2 <- pmin(data_model$x7,data_model$x8)

data_model$z5 <- pmax(data_model$x9,data_model$x10)

data_model$z6 <- pmin(data_model$x9,data_model$x10)

#Descriptive statistics
#rarity
table(data_model$y)

#summary
summary(data_model)

#Store the unprocessed data
data_model.orig <- data_model

###### Data downsample #####
set.seed(123)#set seed so as to generate the same value each time we run the code

round(table(data_model$y)*0.1,digits=0) #ideal event/non-event quota

source_gist("https://gist.github.com/mrdwab/6424112") #import the stratification function

data_model <- stratified(data_model, "y", 0.1)

table(data_model$y)

# Random sampling (for comparison)
data_model_srs <- slice_sample(data_model.orig, n=round(nrow(data_model.orig)*0.1,digits=0))
  
#rarity
table(data_model$y)

#Comparison
var(data_model.orig$x1)
var(data_model$x1)
var(data_model_srs$x1)

mean(data_model.orig$x1)
mean(data_model$x1)
mean(data_model_srs$x1)

#summary
summary(data_model)

###### Data split #######


#split the dataset into training and test sets randomly 

#70% training set
data_model.train <- stratified(data_model, "y", 0.7, replace = FALSE)

#30% test set
#1: use rownames to extract index
#2: specify the observations in full sample whose index NOT in train data
data_model.test <- data_model[which(!rownames(data_model) %in% rownames(data_model.train)),]

#Select the training set except the DV
Y.train = data_model.train$y
X.train = data_model.train %>% select(-c(y,x3,x7,x8,x9,x10,year))
data_model.train.SVM <- data_model.train %>% select(-c(x3,x7,x8,x9,x10,year))
# Select the test set except the DV
Y.Test = data_model.test$y
X.Test = data_model.test %>% select(-c(y,x3,x7,x8,x9,x10,year))
data_model.test.SVM <- data_model.test %>% select(-c(x3,x7,x8,x9,x10,year))


table(data_model.train$y)

table(data_model.test$y)

r.train <- mean(data_model.train$y)
r.test <- mean(data_model.test$y)

#rarity comparison

table(data_model.orig$y)[2]/sum(table(data_model.orig$y)[1:2])

table(data_model.train$y)[2]/sum(table(data_model.train$y)[1:2])

table(data_model.test$y)[2]/sum(table(data_model.test$y)[1:2])

stargazer(data_model, type='text', title="Descriptive statistics", digits=4, out="table1.txt")

stargazer(data_model.orig, type='text', title="Descriptive statistics", digits=4, out="table2.txt")

#Export splitting results (for Python)
# write.table(data_model.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\replication_train.txt", sep="\t")
# write.table(data_model.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\replication_test.txt", sep="\t")


########## Estimate logistic model ##########


m.logit <- glm(as.factor(y)~x2+x4+x5+x6+z1+z2+z5+z6+x11+x1,
            family = "binomial"(link = logit),data = data_model.train)
summary(m.logit)

########## Calculate robust standard errors ############
q.val <- qnorm(0.975)

robust.se.logit <- sqrt(diag(vcovHC(m.logit, type = "HC0")))

r.est.logit <- cbind(
  Estimate = coef(m.logit)
  , "Robust SE" = robust.se.logit
  , z = (coef(m.logit)/robust.se.logit)
  , "Pr(>|z|) "= 2 * pnorm(abs(coef(m.logit)/robust.se.logit), lower.tail = FALSE)
  , LL = coef(m.logit) - q.val  * robust.se.logit
  , UL = coef(m.logit) + q.val  * robust.se.logit
)

r.est.logit

############# King & Zengs correction ################

m.relogit <- zelig(y~x2+x4+x5+x6+z1+z2+z5+z6+x11+x1, 
                   data = data_model.train, model = "relogit", bias.correct = TRUE, case.control = "weighting")
summary(m.relogit)

############ Estimate the cloglog model ##############

m.cloglog <- glm(as.factor(y)~x2+x4+x5+x6+z1+z2+z5+z6+x11+x1,
                          family = "binomial"(link = cloglog),data = data_model.train)
summary(m.cloglog)

############ Calculate robust standard errors #################

robust.se.cloglog <- sqrt(diag(vcovHC(m.cloglog, type = "HC0")))

q.val <- qnorm(0.975)

r.est.cloglog <- cbind(
  Estimate = coef(m.cloglog)
  , "Robust SE" = robust.se.cloglog
  , z = (coef(m.cloglog)/robust.se.cloglog)
  , "Pr(>|z|) "= 2 * pnorm(abs(coef(m.cloglog)/robust.se.cloglog), lower.tail = FALSE)
  , LL = coef(m.cloglog) - q.val  * robust.se.cloglog
  , UL = coef(m.cloglog) + q.val  * robust.se.cloglog
)
r.est.cloglog

############ Export result ################# 

tab_LR <- stargazer::stargazer(m.logit,from_zelig_model(m.relogit),m.cloglog, title = 'Model for Militarised Interstate Dispute',
                               column.labels = c('Logit','ReLogit','cloglog'), 
                               covariate.labels = c('Contiguous', 
                                                     'Allies', 
                                                     'Foreign policy',
                                                     'Balance of Power',
                                                     'Max. democracy',
                                                     'Min. democracy',
                                                     'Max. trade',
                                                     'Min trade',
                                                     'Years since dispute',
                                                     'Major power'),
                              se=list(robust.se.logit, NULL, robust.se.cloglog))

############# ROC & AUC (in-sample) ##############

#mlogit
plogit.tr <- predict(m.logit,data_model.train, type = 'response')
confusionMatrix(data = as.factor(as.numeric(plogit.tr>0.5)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(plogit.tr>r.train)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")

pred.logit <- prediction(fitted(m.logit), data_model.train$y)  
perf.logit <- performance(pred.logit, "tpr", "fpr") 
plot(perf.logit) #plot ROC
auc.tmp <- performance(pred.logit,"auc"); auc.logit <- as.numeric(auc.tmp@y.values) #calculate AUC

#relogit

# Sigmoid function
sigmoid <- function(x) {
  exp(x)/(1 + exp(x))
}

#confusion matrix
coef.relogit <- coef(m.relogit)
design.relogit.manual.tr <- cbind(1, matrix(c(X.train$x2,X.train$x4,X.train$x5,X.train$x6,X.train$z1,X.train$z2,X.train$z5,X.train$z6,X.train$x11,X.train$x1), ncol=10))
pred.relogit.manual.tr <- design.relogit.manual.tr %*% coef.relogit
pred.relogit.manual.tr <- sigmoid(pred.relogit.manual.tr)

confusionMatrix(data = as.factor(as.numeric(pred.relogit.manual.tr>0.5)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(pred.relogit.manual.tr>r.train)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")

pred.relogit <- prediction(fitted(m.relogit), data_model.train$y)  
perf.relogit <- performance(pred.relogit, "tpr", "fpr") 
plot(perf.relogit) #plot ROC
auc.tmp.relogit <- performance(pred.relogit,"auc"); auc.relogit <- as.numeric(auc.tmp.relogit@y.values) #calculate AUC


#mcloglog
#confusion matrix
pcloglog.tr <- predict(m.cloglog,data_model.train, type = 'response')
confusionMatrix(data = as.factor(as.numeric(pcloglog.tr>0.5)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(pcloglog.tr>r.train)), mode = "prec_recall", reference = as.factor(data_model.train$y),positive="1")

pred.cloglog <- prediction(fitted(m.cloglog), data_model.train$y)  
perf.cloglog <- performance(pred.cloglog, "tpr", "fpr") 
plot(perf.cloglog) #plot ROC
auc.tmp.cloglog <- performance(pred.cloglog,"auc"); auc.cloglog <- as.numeric(auc.tmp.cloglog@y.values) #calculate AUC

########## ROC & AUC (out-of-sample) ###############

#mlogit
plogit <- predict(m.logit,data_model.test, type = 'response')

#confusion matrix
confusionMatrix(data = as.factor(as.numeric(plogit>0.5)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(plogit>r.test)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")

pred.logit.out <- prediction(predict(m.logit,data_model.test), data_model.test$y)  
perf.logit.out <- performance(pred.logit.out, "tpr", "fpr") 
plot(perf.logit.out) #plot ROC
auc.tmp.out <- performance(pred.logit.out,"auc"); auc.logit.out <- as.numeric(auc.tmp.out@y.values) #calculate AUC

#relogit

#confusion matrix
coef.relogit <- coef(m.relogit)
design.relogit.manual <- cbind(1, matrix(c(X.Test$x2,X.Test$x4,X.Test$x5,X.Test$x6,X.Test$z1,X.Test$z2,X.Test$z5,X.Test$z6,X.Test$x11,X.Test$x1), ncol=10))
pred.relogit.manual <- design.relogit.manual %*% coef.relogit

pred.relogit.manual <- sigmoid(pred.relogit.manual)

confusionMatrix(data = as.factor(as.numeric(pred.relogit.manual>0.5)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(pred.relogit.manual>r.test)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")

#ROC & AUC
pred.relogit.out <- prediction(pred.relogit.manual, Y.Test) 
perf.relogit.out<- performance(pred.relogit.out, "tpr", "fpr") 
plot(perf.relogit.out) #plot ROC
auc.tmp.relogit.out<- performance(pred.relogit.out,"auc"); auc.relogit.out<- as.numeric(auc.tmp.relogit.out@y.values) #calculate AUC

#mcloglog

#confusion matrix
pcloglog <- predict(m.cloglog,data_model.test, type = 'response')
confusionMatrix(data = as.factor(as.numeric(pcloglog>0.5)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")
confusionMatrix(data = as.factor(as.numeric(pcloglog>r.test)), mode = "prec_recall", reference = as.factor(data_model.test$y),positive="1")

#RUC & AUC
pred.cloglog.out <- prediction(predict(m.cloglog,data_model.test), data_model.test$y) 
perf.cloglog.out <- performance(pred.cloglog.out, "tpr", "fpr") 
plot(perf.cloglog.out) #plot ROC
auc.tmp.cloglog.out <- performance(pred.cloglog.out,"auc"); auc.cloglog.out <- as.numeric(auc.tmp.cloglog.out@y.values) #calculate AUC


#####Probability ######
#brier score

#logit
br.mlogit.tr <- mean((plogit.tr-Y.train)^2) #train
print(br.mlogit.tr)

br.mlogit <- mean((plogit-Y.Test)^2) #test
print(br.mlogit)

#relogit
br.relogit.tr <- mean((pred.relogit.manual.tr-Y.train)^2) #train
print(br.relogit.tr)

br.relogit <- mean((pred.relogit.manual-Y.Test)^2) #test
print(br.relogit)

#mcloglog
br.cloglog.tr <- mean((pcloglog.tr-Y.train)^2) #train
print(br.cloglog.tr)

br.cloglog <- mean((pcloglog-Y.Test)^2) #test
print(br.cloglog)

#Probability histogram
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

pr.GLM.tr <- evalm(GLM.pr.tr,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 5)

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

pr.GLM.test <- evalm(GLM.pr.test,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 5)


############ Probabilities from SVM ###############
# # uncalibrated probability
# path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\MID_raw\\' 

# calibrated probability
path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\MID\\'



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

# # uncalibrated probabilities
# pr.SVM.train <- evalm(standardSVM.train,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 10)
# pr.SVM.test <- evalm(standardSVM.test,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 10)
# pr.DEC.train <- evalm(DEC.SVM.train,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 10)
# pr.DEC.test <- evalm(DEC.SVM.test,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 10)

# calibrated probabilities 
pr.SVM.train <- evalm(standardSVM.train,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 5)
pr.SVM.test <- evalm(standardSVM.test,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 5)
pr.DEC.train <- evalm(DEC.SVM.train,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 5)
pr.DEC.test <- evalm(DEC.SVM.test,fsize=20,dlinecol='black',plots = c('pr','cc'),bins = 5)


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

# #Save all plots created
# plots.dir.path <- list.files(tempdir(), pattern="rs-graphics", full.names = TRUE); 
# plots.png.paths <- list.files(plots.dir.path, pattern=".png", full.names = TRUE)
# file.copy(from=plots.png.paths, to="C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\Figures")