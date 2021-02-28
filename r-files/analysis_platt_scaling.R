
# Platt scaling reliability curve #####
rm(list = ls())

# install.packages('kernlab')
library(ggplot2)
library(MASS)
library(kernlab)       
library(ISLR)
library(devtools)
library(caret) #confusion matrix
Sys.setlocale("LC_ALL","English")
source_gist("https://gist.github.com/mrdwab/6424112") #import the stratification function
library(MLeval)


##### Data generation #####

# Sigmoid function 

sigmoid <- function(x) {
  exp(x)/(1 + exp(x))
}

#Simulate data
set.seed(123)

num.var <- 5
n1 = 500
mu <- runif(num.var) #specify mean

A <- matrix(runif(num.var^2)*2-1, ncol=num.var) #matrix A with arbitrary values
Sigma <- t(A) %*% A #covariance matrix
X.small <- mvrnorm(n1, mu, Sigma) #design matrix

cor(X.small) #investigate the correlation matrix

b0 <- c(-1.208,-2.315) # define intercept

beta <- runif(num.var,0,num.var)                   # Can be anything (nonzero)!
sigma2 <- beta %*% Sigma %*% beta  
beta <- as.vector(beta) / sqrt(sigma2)    # Assures sum of squares is unity

df_platt_name <- c('y','x1','x2','x3','x4','x5')
for (j in 1:2){
  true_model <- X.small %*% beta + b0[j];
  y <- runif(n1) <= sigmoid(true_model)
  table(y);
  assign(paste0("df_platt_",j),as.data.frame(cbind(y,X.small)));
}

#df_platt_1 is a balanced data set

#df_platt_2 is a inbalanced data set


colnames(df_platt_1) <- df_platt_name
colnames(df_platt_2) <- df_platt_name

table(df_platt_1$y)
mean(df_platt_1$y)

table(df_platt_2$y)
mean(df_platt_2$y)

# train-test split

#split the dataset into training and test sets randomly 

#70% training set
df_platt_1.train <- stratified(df_platt_1, "y", 0.5, replace = FALSE)
df_platt_2.train <- stratified(df_platt_2, "y", 0.5, replace = FALSE)

#30% test set
#1: use rownames to extract index
#2: specify the observations in full sample whose index NOT in train data
df_platt_1.test <- df_platt_1[which(!rownames(df_platt_1) %in% rownames(df_platt_1.train)),]
df_platt_2.test <- df_platt_2[which(!rownames(df_platt_2) %in% rownames(df_platt_2.train)),]


table(df_platt_1.train$y)

table(df_platt_1.test$y)



table(df_platt_2.train$y)

table(df_platt_2.test$y)


# store rarity 
r.train <- mean(df_platt_1.train$y)
r.test <- mean(df_platt_1.test$y)

r.train <- mean(df_platt_2.train$y)
r.test <- mean(df_platt_2.test$y)

#### Data export #### 

# df_platt_1
write.table(df_platt_1.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\df_platt_1_train.txt", sep="\t")
write.table(df_platt_1.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\df_platt_1_test.txt", sep="\t")

# df_platt_2
write.table(df_platt_2.train, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\df_platt_2_train.txt", sep="\t")
write.table(df_platt_2.test, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\df_platt_2_test.txt", sep="\t")


#### Analysis ####

# Preparation

#Balanced case
Y.train <- df_platt_1.train$y
Y.Test <- df_platt_1.test$y

#Imbalanced case
Y.train2 <- df_platt_2.train$y
Y.Test2 <- df_platt_2.test$y


# redefine Y.Train 
Y.Train.char <- as.data.frame(Y.train)
Y.Train.char$Y.train[Y.train == 1] <- 'Positive'  
Y.Train.char$Y.train[Y.train == 0] <- 'Negative'

# redefine Y.test
Y.test.char <- as.data.frame(Y.Test)
Y.test.char$Y.Test[Y.Test == 1] <- 'Positive'  
Y.test.char$Y.Test[Y.Test == 0] <- 'Negative'

# redefine Y.Train2 
Y.Train.char2 <- as.data.frame(Y.train2)
Y.Train.char2$Y.train2[Y.train2 == 1] <- 'Positive'  
Y.Train.char2$Y.train2[Y.train2 == 0] <- 'Negative'

# redefine Y.test2
Y.test.char2 <- as.data.frame(Y.Test2)
Y.test.char2$Y.Test2[Y.Test2 == 1] <- 'Positive'  
Y.test.char2$Y.Test2[Y.Test2 == 0] <- 'Negative'

# Import probabilities from SVM

#balanced case

path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\Platt_balanced\\'

standard.lin.train <- read.csv(paste0(path_SVM,'platt_lin_svm_tr.csv'))
standard.lin.test <- read.csv(paste0(path_SVM,'platt_lin_svm_test.csv'))

col_order <- c("Negative", "Positive", "obs",
               "Group")

standard.lin.train$obs <- Y.Train.char$Y.train
standard.lin.test$obs <- Y.test.char$Y.Test


standard.lin.train <- standard.lin.train[, col_order]
standard.lin.test <- standard.lin.test[, col_order]

cc.lin.train <- evalm(standard.lin.train,fsize=20,dlinecol='black',plots = 'cc', bins=10)
cc.lin.test <- evalm(standard.lin.test,fsize=20,dlinecol='black',plots = 'cc', bins=10)


#Imbalanced case

path_SVM <- 'C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\SVM_prob\\Platt_imbalanced\\'

standard.lin.train <- read.csv(paste0(path_SVM,'platt_lin_svm_tr.csv'))
standard.lin.test <- read.csv(paste0(path_SVM,'platt_lin_svm_test.csv'))

col_order <- c("Negative", "Positive", "obs",
               "Group")

standard.lin.train$obs <- Y.Train.char2$Y.train2
standard.lin.test$obs <- Y.test.char2$Y.Test2

standard.lin.train <- standard.lin.train[, col_order]
standard.lin.test <- standard.lin.test[, col_order]

cc.lin.train <- evalm(standard.lin.train,fsize=20,dlinecol='black',plots = 'cc', bins=10)
cc.lin.test <- evalm(standard.lin.test,fsize=20,dlinecol='black',plots = 'cc', bins=10)
