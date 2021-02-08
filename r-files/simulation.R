rm(list = ls())

library(kernlab)
library(LiblineaR)
library(ggplot2)
library(MASS)
library(pROC) # to construct ROC curve
library(ROCR) 

Sys.setlocale("LC_ALL","English")

############################# simulation #################################

set.seed(123)

# Small sample ------------------------------------------------------------


num.var <- 5
n1 = 150
mu <- sample.int(5, num.var,replace = T) #specify mean

A <- matrix(runif(num.var^2)*2-1, ncol=num.var) #matrix A with arbitrary values
Sigma <- t(A) %*% A #covariance matrix
X.small <- mvrnorm(n1, mu, Sigma) #design matrix

cor(X.small) #investigate the correlation matrix

b0 <- c(-3.6,-6.08)

sigmoid <- function(x) {
  exp(x)/(1 + exp(x))
}

# Bivariate model 
x1 <- rnorm(n1,0,1)

for (i in 1:2){
  true_model <- b0[i]+(x1*2);
  proba <- sigmoid(true_model);
  y <- ifelse(runif(n1,0,1) < proba,1,0);
  table(y);
  assign(paste0("df_small_b",i),data.frame(y=y, x1=x1))
}

# Multivariate model 

beta <- rnorm(num.var)                    # Can be anything (nonzero)!
sigma2 <- beta %*% Sigma %*% beta  
beta <- as.vector(beta) / sqrt(sigma2)    # Assures sum of squares is unity

df_small_name <- c('y','x1','x2','x3','x4','x5')
for (j in 1:2){
  true_model <- X.small %*% beta + b0[j];
  y <- runif(n1) <= sigmoid(true_model)
  table(y);
  assign(paste0("df_small_",j),as.data.frame(cbind(y,X.small)));
}

colnames(df_small_1) <- df_small_name
colnames(df_small_2) <- df_small_name

table(df_small_1$y)
mean(df_small_1$y)
table(df_small_2$y)
mean(df_small_2$y)

write.table(df_small_1, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_small_3.txt", sep="\t")
write.table(df_small_2, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_small_5.txt", sep="\t")

# Data split ------------------------------------------------------------








