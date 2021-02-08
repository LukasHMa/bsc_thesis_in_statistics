rm(list = ls())

library(kernlab)
library(LiblineaR)
library(ggplot2)
library(MASS)
library(pROC) # to construct ROC curve
library(ROCR) 

Sys.setlocale("LC_ALL","English")

############################# simulation #################################

set.seed(100)

# Medium sample -----------------------------------------------------------

num.var <- 10
n1 = 500
mu <- sample.int(10, num.var,replace = T) #specify mean

A <- matrix(runif(num.var^2)*2-1, ncol=num.var) #matrix A with arbitrary values
Sigma <- t(A) %*% A #covariance matrix
X.medium <- mvrnorm(n1, mu, Sigma) #design matrix

cor(X.medium) #investigate the correlation matrix

b0 <- c(-3,-5,-7)

sigmoid <- function(x) {
  exp(x)/(1 + exp(x))
}

# Multivariate model 

beta <- rnorm(num.var)                    # Can be anything (nonzero)!
sigma2 <- beta %*% Sigma %*% beta  
beta <- as.vector(beta) / sqrt(sigma2)    # Assures sum of squares is unity
beta <- as.matrix(beta)
  
df_medium_name <- c('y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10')
for (j in 1:3){
  true_model <- X.medium %*% beta + b0[j];
  y <- runif(n1) <= sigmoid(true_model)
  table(y);
  assign(paste0("df_medium_",j),as.data.frame(cbind(y,X.medium)));
}

colnames(df_medium_1) <- df_medium_name
colnames(df_medium_2) <- df_medium_name
colnames(df_medium_3) <- df_medium_name

table(df_medium_1$y)
mean(df_medium_1$y)
table(df_medium_2$y)
mean(df_medium_2$y)
table(df_medium_3$y)
mean(df_medium_3$y)

# write.table(df_medium_1, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_medium_3.txt", sep="\t")
# write.table(df_medium_2, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_medium_5.txt", sep="\t")
# write.table(df_medium_3, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_medium_7.txt", sep="\t")
