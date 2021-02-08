rm(list = ls())


library(LiblineaR)
library(ggplot2)
library(MASS)
library(pROC) # to construct ROC curve
library(ROCR)

Sys.setlocale("LC_ALL","English")

############################# simulation #################################

set.seed(10)

# Medium sample -----------------------------------------------------------

num.var <- 20
n1 = 1000
mu <- sample.int(15, num.var,replace = T) #specify mean

A <- matrix(runif(num.var^2)*2-1, ncol=num.var) #matrix A with arbitrary values
Sigma <- t(A) %*% A #covariance matrix
X.large <- mvrnorm(n1, mu, Sigma) #design matrix

cor(X.large) #investigate the correlation matrix

b0 <- c(-3.32,-5,-7)

sigmoid <- function(x) {
  exp(x)/(1 + exp(x))
}

# Multivariate model 

beta <- rnorm(num.var)                    # Can be anything (nonzero)!
sigma2 <- beta %*% Sigma %*% beta  
beta <- as.vector(beta) / sqrt(sigma2) # Assures sum of squares is unity 
beta <- as.matrix(beta)  

df_large_name <- c('y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10',
                   'x11','x12','x13','x14','x15','x16','x17','x18','x19','x20')
for (j in 1:3){
  true_model <- X.large %*% beta + b0[j];
  y <- runif(n1) <= sigmoid(true_model)
  table(y);
  assign(paste0("df_large_",j),as.data.frame(cbind(y,X.large)));
}

colnames(df_large_1) <- df_large_name
colnames(df_large_2) <- df_large_name
colnames(df_large_3) <- df_large_name

table(df_large_1$y)
mean(df_large_1$y)
table(df_large_2$y)
mean(df_large_2$y)
table(df_large_3$y)
mean(df_large_3$y)

write.table(df_large_1, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_large_3.txt", sep="\t")
write.table(df_large_2, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_large_5.txt", sep="\t")
# write.table(df_large_3, "C:\\Users\\Work\\OneDrive\\Universitetsstudier\\Kurser\\HT2019\\Kandidatuppsats\\Rare events\\DATA\\simulation_large_7.txt", sep="\t")
