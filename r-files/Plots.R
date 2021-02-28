# install.packages('kernlab')
library(ggplot2)
library(MASS)
library(kernlab)       
library(ISLR)
library(caret)
Sys.setlocale("LC_ALL","English")
####Sigmoid function #####

#define the sigmoid function (cdf to logistic regression)
sigmoid <- function(x) {
  exp(x)/(1 + exp(x))
}

#create a sequence of x 
x1 <- seq(-5,5,0.01)

sigm_data <- data.frame(x=x1, y=sigmoid(x1))

sigm_plot <- ggplot(sigm_data, aes(x1, sigmoid(x1), group=1)) + geom_line(color="darkblue",size = 2,alpha=0.8)


print(sigm_plot + labs(y="Sigmoid(x)", x = "x") + theme(axis.text=element_text(size=20, colour = 'black'),
                                                     axis.title=element_text(size=25,face="bold"),
                                                     axis.line = element_line(colour = 'black', size = 0.8)))


#####Logistic densisty ######

x2 <- seq(-10,10,0.01)

logistic_f <- function(x)
  exp(x)/(1+exp(x))**2

log_den_df <- data.frame(x=x2, y=logistic_f(x2))
logit_plot <- ggplot(log_den_df, aes(x2, logistic_f(x2), group=1)) + geom_ribbon(aes(ymin = pmin(y,0), ymax = pmax(y,0)), 
                                                                                 color="#333333",size = 0.8,fill='#FF9999', alpha=0.6)

print(logit_plot + labs(y="Probability density", x = "x") + theme(axis.text=element_text(size=20, colour = 'black'),
                                                                  axis.title=element_text(size=25,face="bold"),
                                                                  axis.line = element_line(colour = 'black', size = 0.8)))
#####Plots for illustration of logistic decision boundary ######
set.seed(123)
beta0 = -0.5
x <- rnorm(50,0,1) #generate inputs 
true_model <- beta0 + (x*2) 
proba <- sigmoid(true_model)

y <- ifelse(runif(50,0,1) < proba,1,0) #generate y, why uniform distribution?

table(y)

df <- data.frame(y,x)

#estimate model 
res <- glm(y ~ x, family  = binomial(link=logit)) # estimating the model
summary(res)
y_hat <- predict(res, type = "response")

df_desci <- data.frame(x=x, y=y, y_pred=y_hat)

desci_plot <- ggplot(df_desci, aes(x, y)) + geom_point(aes(colour = "Y"), size=3) + 
  geom_point(aes(y = y_hat, colour = "Predicted Y"),size=3) + 
  geom_vline(aes(xintercept = 0.6985/2.1286, size='Decision Boundary'))

print(desci_plot +
        labs(y="Y", x = "X") + 
        scale_color_manual(values = c("red","blue")) + 
        theme(legend.title=element_blank(),
              legend.text = element_text(size = 13),
              axis.text=element_text(size=20, colour = 'black'),
              axis.title=element_text(size=25,face="bold"),
              legend.position="right")) #With legends

print(desci_plot +
        labs(y="Y", x = "X") + 
        scale_color_manual(values = c("red","blue")) + 
        theme(legend.title=element_blank(),
              legend.text = element_text(size = 13),
              axis.text=element_text(size=20, colour = 'black'),
              axis.title=element_text(size=25,face="bold"),
              legend.position='none'))

#######################
x3 <- seq(-10,8,0.01)

log_den_df2 <- data.frame(x=x3, y=dlogis(x3,-2,0.5))
log_den_df3 <- data.frame(x=x3, y=dlogis(x3,-2,1))

log_den_df2$uncertainty <- "no"  
log_den_df3$uncertainty <- "yes"

log_den_uncertain <- rbind.data.frame(log_den_df2, log_den_df3)

logit_plot_2 <- ggplot(log_den_uncertain, aes(x, y, fill=uncertainty)) + geom_ribbon(aes(ymin = pmin(y,0), ymax = pmax(y,0)), 
                                                                                     alpha=0.6, color="#333333")
print(logit_plot_2 
      + labs(y="Probability density", x = "y*")
      + theme(axis.line = element_line(colour = 'black', size = 0.8), 
              axis.text=element_text(size=20, colour = 'black'),
              axis.title=element_text(size=25,face="bold"),
              legend.position="none")) 


###############################################################
# simulation 

set.seed(100)
# small sample: 
n1 = 100

sigmoid <- function(x) {
  exp(x)/(1 + exp(x))
}

b0 <- c(-3.5,-5)

x1 <- rnorm(n1,0,1)

##############################################################
#Bivariate model

for (i in 1:2){
  true_model <- b0[i]+(x1);
  proba <- sigmoid(true_model);
  y <- ifelse(runif(n1,0,1) < proba,1,0);
  table(y);
  assign(paste0("df_small_b",i),data.frame(y=y, x1=x1))
}

#Estimate the model
res_small_b1 <- glm(y ~ x1, family  = binomial(link=logit), data=df_small_b1) # estimating the model
summary(res_small_b1)

df_desci_small_b1 <- data.frame(x=df_small_b1$x1, 
                                y=df_small_b1$y, 
                                y_pred=predict(res_small_b1, type = "response"))

res_small_b2 <- glm(y ~ x1, family  = binomial(link=logit), data=df_small_b2) # estimating the model
summary(res_small_b2)

df_desci_small_b2 <- data.frame(x=df_small_b2$x1, 
                                y=df_small_b2$y, 
                                y_pred=predict(res_small_b2, type = "response"))


desci_plot_2 <- ggplot(df_desci_small_b1, aes(x, y)) + geom_point(aes(colour = "Y"), size=3) + 
  geom_point(aes(y = y_pred, colour = "Predicted Y"),size=2) + 
  geom_vline(aes(xintercept = 6.4325/2.8908, size='Decision Boundary'))

print(desci_plot_2 +
        labs(y="Y", x = "X") + 
        scale_color_manual(values = c("orangered","darkblue")) + 
        theme(legend.title=element_blank(),
              legend.text = element_text(size = 13),
              axis.text=element_text(size=20, colour = 'black'),
              axis.title=element_text(size=25,face="bold"),
              legend.position="none"))

########### comparison c-log-log and sigmoid
#define the inverse Gumbel cdf
cloglog <- function(x) {
  1-exp(-exp(x))
}


# cloglog_data <- data.frame(x=x1, y=cloglog(x1/sqrt(2)))

cloglog_data <- data.frame(x=x1, y=cloglog(x1))

#Combine the two data set (Easier if everything is in the same data frame)
cloglog_data$link <- 'C log-log link' #create a column of identification tag
sigm_data$link <- 'Logit link'
clog_logit <- rbind.data.frame(cloglog_data, sigm_data)

cloglog_plot <- ggplot(clog_logit , aes(x=x, y=y, group=link)) + geom_line(aes(colour = link, linetype = link, size=link))

print(cloglog_plot  
      + labs(y="Probability", x = "x") + theme(axis.text=element_text(size=15),
                                                        axis.title=element_text(size=22,face="bold"),
                                                        axis.line = element_line(colour = 'black', size = 0.8))  
      + scale_linetype_manual(values=c("twodash", "solid"))
      + scale_color_manual(values = c("darkred","darkblue"))
      + scale_size_manual(values = c(1.2,1.2))
      + theme(axis.line = element_line(colour = 'black', size = 0.8), 
             legend.title = element_blank(),
             legend.text = element_text(size = 13),
             axis.text=element_text(size=20, colour = 'black'),
             axis.title=element_text(size=25,face="bold"),
             legend.position=c(0.8, 0.2)))

rm(list = ls())
############################################################
# SVM classification

# Source: https://uc-r.github.io/svm

# Construct sample data set - completely separated
set.seed(123)


x <- matrix(rnorm(50*2), ncol = 2)
y <- c(rep(-1,25), rep(1,25)) 
# x[y==1,] <- x[y==1,] + 6 #perfectly separable case

x[y==1,] <- x[y==1,] + 2
dat <- data.frame(x=x, y=as.factor(y))

kernfit <- ksvm(x, y, type = "C-svc", kernel = 'vanilladot')
plot(kernfit, data = x)



