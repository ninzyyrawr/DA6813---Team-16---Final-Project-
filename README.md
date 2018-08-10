# DA6813---Team-16---Final-Project-

library(sqldf)
library(dplyr)
library(ggplot2)
library(e1071)
library(caret)
library(FNN)
library(Hmisc)
library(randomForest)
library(pROC)
require(MASS)
library (tree)

df <- read.csv("/Documents/Summer 18/Project/Speed Dating Data.csv", head = TRUE, na.strings = c("NA", ""), stringsAsFactors = F)
#str(df)
dim(df)


# Changing the Variables
# Gender variables changed to "F" and "M".
# Same race variables changed to "Yes" and "No".
# Matching variables changed to "Yes" and "No".
df[df$gender == 0,]$gender <- "F"
df[df$gender == 1,]$gender <- "M"
df[df$samerace == 0,]$samerace <- "No"
df[df$samerace == 1,]$samerace <- "Yes"      
df[df$match == 0,]$match <- "No"
df[df$match == 1,]$match <- "Yes" 


# filter not samerace
df_match = df[df$samerace == "No", ]
dim(df_match)

# remove wave 6-9
wave = c(1:5,10:21)
df_match_wave = df_match[df_match$wave %in% wave,]
dim(df_match_wave)

#remove variables with more than 20% NA

f <- function(x) {
    sum(is.na(x)) < length(x) * 0.2
}

df_final = df_match_wave[, vapply(df_match_wave, f, logical(1)), drop = F]

# Find the iid of the only missing id
iid <- df_final[is.na(df_final$id),]$iid
# Assign this iid' id to the missing iid..
df_final[is.na(df_final$id),]$id <- head(df_final[df_final$iid == iid,]$id, 1)


# remove the missing pid rows because in wave 5, there are 9F and 10M but each male was recorded meeting 10 times
df_final <- df_final[complete.cases(df_final$pid), ]


#seem like we can't impute missing age
# remove rows that have missing age and age_o

df_final <- df_final[complete.cases(df_final$age), ]
df_final <- df_final[complete.cases(df_final$age_o), ]



# field_cd, career_c, race, race_o: NA = "0" for "Other"
df_final$field_cd[is.na(df_final$field_cd)] = 0
df_final$career_c[is.na(df_final$career_c)] = 0
df_final$race[is.na(df_final$race)] = 0
df_final$race_o[is.na(df_final$race_o)] = 0


dim(df_final)

sapply(df_final, function(x) sum(is.na(x)))

date_columns = c("iid", "field_cd", "career_c")
date_info = subset(df_final, select = date_columns)

names(date_info) = c("pid", "field_cd_o", "career_c_o")

date_info <- subset(date_info, !duplicated(date_info[,1])) 

df_final = merge(df_final, date_info, by = "pid")

dim(df_final)


# create new indicator variables for when two people have the same field, career, age (grouping age 18-20, 20-25, 25-30, and so on)

df_final$field_same <- df_final$field_cd == df_final$field_cd_o

df_final$career_same <- df_final$career_c == df_final$career_c_o


df_final$agegroup = findInterval(df_final$age, c(20,25,30,35,40,45,50))
df_final$agegroup_o = findInterval(df_final$age_o, c(20,25,30,35,40,45,50))
df_final$age_same <- df_final$agegroup == df_final$agegroup_o

dim(df_final)
#head(df_final)


sapply(df_final, function(x) sum(is.na(x)))

#drop variables not important

drops = c("iid","id","idg","condtn","wave","round","position","order",
                        "partner","pid","samerace","age_o","dec_o","like_o","prob_o","met_o",
                        "age","from","zipcode","goal","career","career_c","career_c_o","field","field_cd","field_cd_o","dec","match_es","satis_2","length","numdat_2")

data = df_final[,!(names(df_final) %in% drops)]

dim(data)

sapply(data, function(x) sum(is.na(x)))

for (i in 1:length(names(data))){
  data[,names(data)[i]] <- with(data, ave(data[,names(data)[i]], gender,
                                    FUN = function(x) replace(x, is.na(x), round(mean(x, na.rm= TRUE)))))
  
}

sapply(data, function(x) sum(is.na(x)))
dim(data)

# convert categorical variables to factor

data$match <- as.factor(data$match)
data$field_same <- as.factor(data$field_same)
data$career_same <- as.factor(data$career_same)
data$age_same <- as.factor(data$age_same)

dim(data)

# Find number of men/women for each race
races <- data %>%
  group_by(gender, race) %>%
  summarise(
    my.n = n()
  )

# plot race
ggplot(races, aes(x = race, y = my.n, fill = factor(gender))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_discrete(name = "Gender") +
  xlab("Race") + ylab("Count") + ggtitle("Race repartition") +
  scale_x_continuous(labels = c("Black", "European", "Latino", "Asian", "Native","Other"), breaks = 1:6)


# Find number of men/women for each age group
age_data <- data %>%
  group_by(gender, agegroup) %>%
  summarise(
    my.n = n()
  )

# plot race
ggplot(age_data, aes(x = agegroup, y = my.n, fill = factor(gender))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_discrete(name = "Gender") +
  xlab("Age Group") + ylab("Count") + ggtitle("Age repartition") +
  scale_x_continuous(labels = c("<20","20", "25", "30", "35", "40","45","<50"), breaks = 0:7)

# drop variables that are unused
drops2 = c("race","race_o","agegroup","agegroup_o")

data = data[,!(names(data) %in% drops2)]

dim(data)

sapply(data, function(x) sum(is.na(x)))

# Isolate the men/female from the dataset
df_M <- data[data$gender == "M",]
df_M = df_M[,-1]
dim(df_M)

df_F <- data[data$gender == "F",]
df_F = df_F[,-1]
dim(df_F)
#head(df_F)

categorical = c("match","field_same","career_same","age_same")
numvar = names(df_F[,!(names(df_F) %in% categorical)])

#Determining Skewness

#seems like there are skewness in the data

se <- function(x) sd(x)/sqrt(length(x))
result = NULL
print("Skewness for Female Data")
for (i in numvar){
 # twose = 
  #skew = 
  if (abs(skewness(df_F[,i])) > 2*se(df_F[,i])){
    result = "Yes"
  } else {result = "No"}
  print(sprintf("%s | 2*se = %s | skewness = %s | %s",i, 2*se(df_F[,i]), skewness(df_F[,i]), result))

}


print("Skewness for Male Data")
for (i in numvar){
 # twose = 
  #skew = 
  if (abs(skewness(df_M[,i])) > 2*se(df_M[,i])){
    result = "Yes"
  } else {result = "No"}
  print(sprintf("%s | 2*se = %s | skewness = %s | %s",i, 2*se(df_M[,i]), skewness(df_M[,i]), result))

}


# Correlations plot for F
correlations_F = cor(df_F[,!(names(df_F) %in% categorical)])
corrplot::corrplot(correlations_F, type = "full")

# Correlations plot for M
correlations_M = cor(df_M[,!(names(df_M) %in% categorical)])
corrplot::corrplot(correlations_M, type = "full")


#FINAL FINAL DATA

final_F = df_F
final_M = df_M

#names(final_F)

dim(final_F)
  
#Data splitting 80% training - 20% test

predictors = final_F[,-1]
classes = final_F[,1]

set.seed(1)
trainingRows <- createDataPartition(classes, p = .80, list= FALSE)
train_set_F = final_F[trainingRows,]
test_set_F = final_F[-trainingRows,]

dim(train_set_F)
dim(test_set_F)

# Class Imbalance
summary(final_F$match)
#  No  Yes 
#1694  308 
summary(train_set_F$match)
#  No  Yes 
#1356  247 
summary(test_set_F$match)
#No Yes 
#338  61

## The simplest approach to counteracting the negative effects of class imbalance is to tune the model to maximize the accuracy of the minority class(es).

barplot(summary(final_M$match))
barplot(summary(final_F$match))

#names(train_set_F)

ggplot(train_set_F, aes(factor(match), attr)) + geom_boxplot()

ggplot(train_set_F, aes(factor(match), fun_o)) + geom_boxplot()

ggplot(train_set_F, aes(factor(match), imprace)) + geom_boxplot()

ggplot() + 
  geom_bar(data = train_set_F,
           aes(x = factor(match),fill = factor(field_same)),
           position = "fill")

ggplot() + 
  geom_bar(data = train_set_F,
           aes(x = factor(match),fill = factor(age_same)),
           position = "fill")

#Looking at some boxplots and barplots, we get a good idea of which variables make a difference and which ones don't. The more attractive the man is and the more fun the woman is, the more likely the date will end in a match. 

# imprace:
#How important is it to you (on a scale of 1-10) that a person you date be of the same racial/ethnic background?
# The important of the same racial does not seem to play a role, meaning that the partner's race is not affected their decision in match.

## Logistic Regression (FEMALE)

set.seed(1)

glm = glm(match~., data = train_set_F, family = "binomial")
summary(glm)

glm_probs = predict(glm, newdata=test_set_F, type = "response")

lab=contrasts(final_F$match)
tn=rownames(lab)

glm_pred_y = rep(tn[1], length(test_set_F$match))

glm_pred_y[glm_probs > 0.5] = tn[2]

## the following command creates the confusion matrix
table(glm_pred_y, test_set_F$match)

#glm_pred_y  No Yes
#       No  323  43
#       Yes  15  18

##compute the accuracy rate
mean(glm_pred_y == test_set_F$match)
# 0.8546366


#plot(glm)

##Logistic Regression (MALE)

set.seed(1)

glmM = glm(match~., data = train_set_M, family = "binomial")
summary(glmM)

glm_probsM = predict(glmM, newdata=test_set_M, type = "response")

labM=contrasts(final_M$match)
tn=rownames(labM)

glm_pred_yM = rep(tn[1], length(test_set_M$match))

glm_pred_yM[glm_probsM > 0.5] = tn[2]

## the following command creates the confusion matrix
table(glm_pred_yM, test_set_M$match)

#glm_pred_yM  No Yes
#        No  316  48
#        Yes  15  20

##compute the accuracy rate
mean(glm_pred_yM == test_set_M$match)
# 0.8421053


#plot(glmM)

# KNN

set.seed(1)

# k is the tuning parameter

for (k_val in 1:50){
  knn_fit <- knn3(match~.,data=train_set_F,k=k_val,prob=FALSE)
  knn_results <- predict(knn_fit,test_set_F,type="class")
  acc <- mean(knn_results == test_set_F$match) 
  error_rate[k_val] = acc
}

## k = 13 is the best k
qplot(1:50, error_rate, xlab = "K",
ylab = "Error Rate",
geom=c("point", "line"))

##  "tuning value = 13 | accuracy = 0.849624060150376"
knn_fit <- knn3(match~.,data=train_set_F,k=13,prob=FALSE)
knn_result <- predict(knn_fit,test_set_F,type="class")
table(knn_result, test_set_F$match)
mean(knn_result == test_set_F$match)

#knn_result  No Yes
#       No  337  59
#       Yes   1   2

#KNN (MALE)
#0.8496241

set.seed(1)

# k is the tuning parameter
error_rateM = NULL

for (k_valM in 1:50){
  knn_fitM <- knn3(match~.,data=train_set_M,k=k_val,prob=FALSE)
  knn_resultsM <- predict(knn_fitM,test_set_M,type="class")
  accM <- mean(knn_resultsM == test_set_M$match) 
  error_rateM[k_valM] = accM
}

## k = 13 is the best k
qplot(1:50, error_rateM, xlab = "K",
ylab = "Error Rate",
geom=c("point", "line"))

##  "tuning value = 13 | accuracy = 0.849624060150376"
knn_fitM <- knn3(match~.,data=train_set_M,k=13,prob=FALSE)
knn_resultM <- predict(knn_fitM,test_set_M,type="class")
table(knn_resultM, test_set_M$match)
mean(knn_resultM == test_set_M$match)

           
#knn_resultM  No Yes
#        No  329  66
#        Yes   2   2
#[1] 0.8295739

# LDA

lda.fit <- lda(match ~ ., data = train_set_F)
#summary(lda.fit)

lda_predict <- predict(lda.fit, test_set_F)

lda_pred_y <- lda_predict$class
table(lda_pred_y,test_set_F$match)
          
#lda_pred_y  No Yes
#       No  329  47
#       Yes   9  14

mean(lda_pred_y == test_set_F$match)
# accuracy = 0.8596491

plot(lda.fit)

#LDA (MALE)

lda.fitM <- lda(match ~ ., data = train_set_M)
#summary(lda.fit)

lda_predictM <- predict(lda.fitM, test_set_M)

lda_pred_yM <- lda_predictM$class
table(lda_pred_yM,test_set_M$match)
          
#lda_pred_y  No Yes
#       No  329  47
#       Yes   9  14

mean(lda_pred_yM == test_set_M$match)
# accuracy = 0.8446115

plot(lda.fitM)

# QDA

qda.fit <- qda(match~ ., data = train_set_F)
qda_pred <- predict(qda.fit,test_set_F)

qda_pred_y = qda_pred$class
table(qda_pred_y, test_set_F$match)
#qda_pred_y  No Yes
#       No  278  30
#       Yes  60  31

mean(qda_pred_y == test_set_F$match)

#R2 = 0.7744361

#QDA (MALE)

qda.fitM <- qda(match~ ., data = train_set_M)
qda_predM <- predict(qda.fitM,test_set_M)

qda_pred_yM = qda_predM$class
table(qda_pred_yM, test_set_M$match)
#qda_pred_y  No Yes
#       No  278  30
#       Yes  60  31

mean(qda_pred_yM == test_set_M$match)

# Decision Tree
set.seed(1)

tree_fit = tree(match~., train_set_F)
summary(tree_fit)
# the training error rate is 11.92 %.

plot(tree_fit)
text(tree_fit ,pretty =0)

tree.pred= predict(tree_fit, test_set_F,type ="class")
table(tree.pred,test_set_F$match)
#tree.pred  No Yes
#      No  321  41
#      Yes  17  20

mean(tree.pred == test_set_F$match)
# 0.8546366


## PRUNING
cv_match = cv.tree(tree_fit, FUN = prune.misclass)
cv_match
# The tree with 4 or 6 terminal nodes results in the lowest cross-validation error rate, with 242 cross-validation errors

par(mfrow=c(1,2))
plot(cv_match$size ,cv_match$dev ,type="b")
plot(cv_match$k ,cv_match$dev ,type="b")

#prune the tree to prune obtain the 6-node tree.
prune.match =prune.misclass (tree_fit ,best =4)
plot(prune.match)
text(prune.match,pretty =0)

#How well does this pruned tree perform on the test data set?
tree.prune.pred=predict (prune.match, test_set_F ,type="class")
table(tree.prune.pred , test_set_F$match)
mean(tree.prune.pred == test_set_F$match)
## 0.8446115 of the test observations are correctly classified
## the classification accuracy is not improved
## --> best = 14

##Decision Trees (MALE)

set.seed(1)

tree_fitM = tree(match~., train_set_M)
summary(tree_fitM)
# the training error rate is 11.92 %.

plot(tree_fitM)
text(tree_fitM ,pretty =0)

tree.predM= predict(tree_fitM, test_set_M,type ="class")
table(tree.predM,test_set_M$match)
#tree.pred  No Yes
#      No  321  41
#      Yes  17  20

mean(tree.predM == test_set_M$match)
# 0.8345865


## PRUNING
cv_matchM = cv.tree(tree_fitM, FUN = prune.misclass)
cv_matchM
# The tree with 4 or 6 terminal nodes results in the lowest cross-validation error rate, with 242 cross-validation errors

par(mfrow=c(1,2))
plot(cv_matchM$size ,cv_matchM$dev ,type="b")
plot(cv_matchM$k ,cv_matchM$dev ,type="b")

#prune the tree to prune obtain the 6-node tree.
prune.matchM =prune.misclass (tree_fitM ,best =4)
plot(prune.matchM)
text(prune.matchM,pretty =0)

#How well does this pruned tree perform on the test data set?
tree.prune.predM=predict (prune.matchM, test_set_M ,type="class")
table(tree.prune.predM , test_set_M$match)
mean(tree.prune.predM == test_set_M$match)
## 0.8370927 of the test observations are correctly classified
## the classification accuracy is not improved
## --> best = 14


# Random Forest
set.seed(1)
rfModel <- randomForest(match~.,train_set_F, ntree = 500, importance = TRUE)

rfTestPred <- predict(rfModel, test_set_F, type = "prob")
head(rfTestPred)

test_set_F$RFprob <- rfTestPred[,"Yes"]
test_set_F$RFclass <- predict(rfModel, test_set_F)

confusionMatrix(data = test_set_F$RFclass,
                reference = test_set_F$match,
                positive = "Yes")
#Prediction  No Yes
#       No  335  43
#       Yes   3  18

# R2 = 0.8847

importance(rfModel)

# Random Forest (MALE)
set.seed(1)
rfModelM <- randomForest(match~.,train_set_M, ntree = 500, importance = TRUE)

rfTestPredM <- predict(rfModelM, test_set_M, type = "prob")
head(rfTestPredM)

test_set_M$RFprobM <- rfTestPredM[,"Yes"]
test_set_M$RFclassM <- predict(rfModelM, test_set_M)

confusionMatrix(data = test_set_M$RFclass,
                reference = test_set_M$match,
                positive = "Yes")
#Reference
#Prediction  No Yes
#       No  326  49
#       Yes   5  19
                                          
#               Accuracy : 0.8647

importance(rfModelM)

#SVM
set.seed(1)
svm.fit = svm(match~., train_set_F)
#Prediction
svm.pred = predict(svm.fit, test_set_F)
table(pred=svm.pred, true=test_set_F[,1])

confusionMatrix(svm.pred, test_set_F$match)

#Reference
#Prediction  No Yes
#       No  337  51
#       Yes   1  10
                                          
#               Accuracy : 0.8697 


#SVM Tuning
#radial
set.seed(1)
svm.tune = tune(svm, match~. , data=train_set_F, kernel="radial", 
              ranges=list(cost=c(0.1, 1, 10, 20, 30, 40, 100), 
                          gamma=c(0.01,0.05,0.1,5)))


summary(svm.tune)
#Best performance is error_rate of 0.1366421 
#Accuracy of 0.8633579


#polynomial
set.seed(1)
svm.tune2=tune(svm, match~. ,data=train_set_F,kernel="polynomial",
              ranges=list(cost=c(0.1, 1, 10, 20, 30, 40, 100),
                          degree=c(1,2,3)))
summary(svm.tune2)
#Best performance is error_rate of 0.1328688 
#Accuracy of 0.8671312

bestmod=svm.tune2$best.model
summary(bestmod)

ypred=predict(bestmod,test_set_F)
table(predict=ypred, truth=test_set_F$match)

mean(ypred!=test_set_F$match)

#Error rate of 0.1303258
#Accuracy rate of 0.8696742

#SVM on Resampling

#ctrl=trainControl(method="cv",number=10) ## 10-fold CV
ctrl=trainControl(method="repeatedcv",repeats = 5, classProbs = TRUE) ## Repeated 10-fold cross-validation

svmFit = train(match~., train_set_F,
               method = "svmRadial",
               preProc = c("center", "scale"),
               tuneLength = 10,
               trControl = ctrl)
svmFit

# predict the outcome on a test set
svm_pred <- predict(svmFit, test_set_F)

# compare predicted outcome and true outcome
confusionMatrix(svm_pred, test_set_F$match)

#Prediction  No Yes
#       No  328  35
#       Yes  10  26

#Accuracy : 0.8872 


#####SVM MALE#####

#SVM
set.seed(1)
svm.fitM = svm(match~., train_set_M)
#Prediction
svm.predM = predict(svm.fitM, test_set_M)
table(pred=svm.predM, true=test_set_F[,1])

confusionMatrix(svm.predM, test_set_M$match)

#Reference
#Prediction  No Yes
#       No  327  64
#       Yes   4   4
                                         
#               Accuracy : 0.8296 


#SVM Tuning
#radial
set.seed(1)
svm.tuneM = tune(svm, match~. , data=train_set_M, kernel="radial", 
              ranges=list(cost=c(0.1, 1, 10, 20, 30, 40, 100), 
                          gamma=c(0.01,0.05,0.1,5)))


summary(svm.tuneM)

#best parameters:
# cost gamma
#    1  0.01

# best performance: 0.1328921
#0.8671079

#


#polynomial
set.seed(1)
svm.tune2M=tune(svm, match~. ,data=train_set_M,kernel="polynomial",
              ranges=list(cost=c(0.1, 1, 10, 20, 30, 40, 100),
                          degree=c(1,2,3)))


#summary(svm.tune2M)
#best parameters:
# cost degree
#   20      1

#best performance: 0.1316654
#0.8683346

bestmodM=svm.tune2M$best.model
summary(bestmodM)

ypredM=predict(bestmodM,test_set_M)
table(predict=ypredM, truth=test_set_M$match)

mean(ypredM!=test_set_M$match)

#truth
#predict  No Yes
#    No  325  53
#    Yes   6  15
#[1] 0.1478697

#SVM on Resampling

#ctrl=trainControl(method="cv",number=10) ## 10-fold CV
ctrl=trainControl(method="repeatedcv",repeats = 5, classProbs = TRUE) ## Repeated 10-fold cross-validation

svmFit = train(match~., train_set_F,
               method = "svmRadial",
               preProc = c("center", "scale"),
               tuneLength = 10,
               trControl = ctrl)
svmFit

# predict the outcome on a test set
svm_pred <- predict(svmFit, test_set_F)

# compare predicted outcome and true outcome
confusionMatrix(svm_pred, test_set_F$match)

#Prediction  No Yes
#       No    
#       Yes  

#Accuracy : 

##Resampling CV (Female) ---------------

#final_F = df_F
#final_M = df_M

#train_set_F = final_F[trainingRows,]
#test_set_F = final_F[-trainingRows,]

#Using 10-fold CV
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(1)
fit.lda <- train(match~., data=train_set_F, method="lda", metric=metric, trControl=control)

#Accuracy   Kappa    
#  0.8615091  0.3270967
#No Parameter


set.seed(1)
fit.qda <- train(match~., data=train_set_F, method="qda", metric=metric, trControl=control)
#Accuracy  Kappa    
#  0.769171  0.2474415

#no parameter

# b) nonlinear algorithms
# CART (Decision Tree)
set.seed(1)
fit.cart <- train(match~., data=train_set_F, method="rpart", metric=metric, trControl=control)

#cp          Accuracy   Kappa    
#  0.01619433  0.8577472  0.3124961

# kNN
set.seed(1)
fit.knn <- train(match~., data=train_set_F, method="knn", metric=metric, trControl=control)

#k  Accuracy   Kappa 
#9  0.8415516  0.01846986

# c) advanced algorithms
# SVM
set.seed(1)
fit.svm <- train(match~., data=train_set_F, method="svmRadial", metric=metric, trControl=control)

# C     Accuracy   Kappa
#1.00  0.8596340  0.180525959

#sigma = 0.007752414 and C = 1


# Random Forest
set.seed(1)
fit.rf <- train(match~., data=train_set_F, method="rf", metric=metric, trControl=control)

#mtry  Accuracy   Kappa 
#38    0.8752435  0.3683191

#Logistic Regression
set.seed(1)
fit.glm <- train(match~., data=train_set_F, method="glm", metric=metric, trControl=control)

#Accuracy   Kappa    
#  0.8621302  0.3674785

##Using 5-fold
control20 <- trainControl(method="cv", number=20)
metric20 <- "Accuracy"

set.seed(1)
fit.glm20 <- train(match~., data=train_set_F, method="glm", metric=metric20, trControl=control20)

predictions.glm20 <- predict(fit.glm20, test_set_F)
confusionMatrix(predictions.glm20, test_set_F$match, positive = "Yes")


resultsF <- resamples(list(lda=fit.lda, dqa=fit.qda, glm=fit.glm, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(resultsF)
dotplot(resultsF)

# summarize Best Model
print(fit.rf)

#FEMALE
```{r}
# estimate skill of RF on the validation dataset
#RF
predictions <- predict(fit.rf, test_set_F)
confusionMatrix(predictions, test_set_F$match)

#Accuracy is: 0.8797   

#LDA
# estimate skill of LDA on the validation dataset
predictions.lda <- predict(fit.lda, test_set_F)
confusionMatrix(predictions.lda, test_set_F$match, positive = "Yes")

#Accuracy : 0.8596 


#QDA
predictions.qda <- predict(fit.qda, test_set_F)
confusionMatrix(predictions.qda, test_set_F$match, positive = "Yes")

#Accuracy : 0.7744

#LOGISCTIC 
predictions.glm <- predict(fit.glm, test_set_F)
confusionMatrix(predictions.glm, test_set_F$match, positive = "Yes")
#Accuracy : 0.8546  

#SVM
predictions.svm <- predict(fit.svm, test_set_F)
confusionMatrix(predictions.svm, test_set_F$match, positive = "Yes")
#Accuracy : 0.8672 

#KNN
predictions.knn <- predict(fit.knn, test_set_F)
confusionMatrix(predictions.knn, test_set_F$match, positive = "Yes")
#Accuracy 0.8421

#CART (Decision Tree)
predictions.cart <- predict(fit.cart, test_set_F)
confusionMatrix(predictions.cart, test_set_F$match, positive = "Yes")
#Accuracy : 0.8471

#RANDOMFOREST FOR FEMALE

##Resampling CV (Male) --


```{r} 
#final_F = df_F
#final_M = df_M

#train_set_F = final_F[trainingRows,]
#test_set_F = final_F[-trainingRows,]

#Using 10-fold CV
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(1)
fit.ldaM <- train(match~., data=train_set_M, method="lda", metric=metric, trControl=control)

#Accuracy   Kappa    
#  0.8627407  0.3368038

set.seed(1)
fit.qdaM <- train(match~., data=train_set_M, method="qda", metric=metric, trControl=control)
#GLM
set.seed(1)
fit.glmM <- train(match~., data=train_set_M, method="glm", metric=metric, trControl=control)

#Accuracy   Kappa    
#  0.8602601  0.3506886

# b) nonlinear algorithms
# CART (Decision Tree)
set.seed(1)
fit.cartM<- train(match~., data=train_set_M, method="rpart", metric=metric, trControl=control)

#cp          Accuracy   Kappa
#0.02777778  0.8496623  0.08428922

#higher Kappa --> 0.01562500  0.8465295  0.24588786

# kNN
set.seed(1)
fit.knnM <- train(match~., data=train_set_M, method="knn", metric=metric, trControl=control)

#k  Accuracy   Kappa
#7  0.8490450  0.11081890

# c) advanced algorithms
# SVM
set.seed(1)
fit.svmM <- train(match~., data=train_set_M, method="svmRadial", metric=metric, trControl=control)

# C     Accuracy   Kappa
#1.00  0.8664907  0.1847549

#sigma = 0.007765363 and C = 1

# Random Forest
set.seed(1)
fit.rfM <- train(match~., data=train_set_M, method="rf", metric=metric, trControl=control)

#mtry  Accuracy   Kappa  
#38    0.8715023  0.3176589

# summarize accuracy of models (MALE)
resultsM <- resamples(list(lda=fit.ldaM, dqa=fit.qdaM, glm=fit.glmM, cart=fit.cartM, knn=fit.knnM, svm=fit.svmM, rf=fit.rfM))
summary(resultsM)
dotplot(resultsM)

#It looks like RF is also the most accurate for Males data

#MALE Prediction (CV)
```{r}
#final_F = df_F
#final_M = df_M

#train_set_F = final_F[trainingRows,]
#test_set_F = final_F[-trainingRows,]

train_set_M = final_M[trainingRows,]
test_set_M = final_M[-trainingRows,]


# estimate skill of RandomForest on the validation dataset
predictionsM <- predict(fit.rfM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")

#Accuracy : 0.8446 

predictionsM <- predict(fit.ldaM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")

#Accuracy: 0.8446

#LDA and RandomForest comes close

predictionsM <- predict(fit.glmM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")

#Accuracy : 0.8421  

predictionsM <- predict(fit.svmM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")
#Accuracy : 0.8371

predictionsM <- predict(fit.knnM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")
#Accuracy : 0.802

predictionsM <- predict(fit.cartM, test_set_M)
confusionMatrix(predictionsM, test_set_M$match, positive = "Yes")
#Accuracy : 0.8296 

#FEMALE UPSAMPLING
```{r}
library(randomForest)

predictions <- predict(fit.rf, test_set_F)
confusionMatrix(predictions, test_set_F$match)

data_balanced_over <- ovun.sample(match ~ ., data = train_set_F, method = "over",N = 2712)$data
table(data_balanced_over$match)

#Using upsampling
set.seed(1)
rfModel.up <- randomForest(match~.,data_balanced_over, ntree = 500, importance = TRUE)

rfTestPred.up <- predict(rfModel.up, test_set_F, type = "prob")
head(rfTestPred.up)

test_set_F$RFclass.up <- predict(rfModel.up, test_set_F)

confusionMatrix(data = test_set_F$RFclass.up,
                reference = test_set_F$match,
                positive = "Yes")

#----
#using 10-fold
set.seed(1)
fit.rf.up <- train(match~., data=data_balanced_over, method="rf", metric=metric, trControl=control)
set.seed(1)
fit.rf.rose <- train(match~., data=data.rose, method="rf", metric=metric, trControl=control)

PredRFModel.up <- predict(fit.rf.up, test_set_F)
confusionMatrix(PredRFModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  326  40
#       Yes  12  21
                                          
#               Accuracy : 0.8697  

#GLM

set.seed(1)
fit.glm.up <- train(match~., data=data_balanced_over, method="glm", metric=metric, trControl=control)
set.seed(1)

PredGLMModel.up <- predict(fit.glm.up, test_set_F)
confusionMatrix(PredGLMModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  263  18
#       Yes  75  43
                                          
#               Accuracy : 0.7669 

#LDA
set.seed(1)
fit.lda.up <- train(match~., data=data_balanced_over, method="lda", metric=metric, trControl=control)
set.seed(1)

PredldaModel.up <- predict(fit.lda.up, test_set_F)
confusionMatrix(PredldaModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  251  16
#       Yes  87  45
                                         
#               Accuracy : 0.7419

#Decision Tree
set.seed(1)
fit.cart.up <- train(match~., data=data_balanced_over, method="rpart", metric=metric, trControl=control)
set.seed(1)

PredcartModel.up <- predict(fit.cart.up, test_set_F)
confusionMatrix(PredcartModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  292  24
#       Yes  46  37
                                          
#               Accuracy : 0.8246


#SVM
set.seed(1)
fit.svm.up <- train(match~., data=data_balanced_over, method="svmRadial", metric=metric, trControl=control)
set.seed(1)

PredsvmModel.up <- predict(fit.svm.up, test_set_F)
confusionMatrix(PredsvmModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  288  20
#       Yes  50  41
                                          
#               Accuracy : 0.8246


set.seed(1)
fit.svmL.up <- train(match~., data=data_balanced_over, method="svmLinear", metric=metric, trControl=control)
set.seed(1)

PredsvmLModel.up <- predict(fit.svmL.up, test_set_F)
confusionMatrix(PredsvmLModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  260  14
#       Yes  78  47
                                          
#               Accuracy : 0.7694 


set.seed(1)
fit.svmP.up <- train(match~., data=data_balanced_over, method="svmPoly", metric=metric, trControl=control)
set.seed(1)

PredsvmPModel.up <- predict(fit.svmP.up, test_set_F)
confusionMatrix(PredsvmPModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  318  31
#       Yes  20  30
                                          
#               Accuracy : 0.8722

#KNN

set.seed(1)
fit.knn.up <- train(match~., data=data_balanced_over, method="knn", metric=metric, trControl=control)
set.seed(1)

PredknnModel.up <- predict(fit.knn.up, test_set_F)
confusionMatrix(PredknnModel.up, test_set_F$match, positive = "Yes")

#Reference
#Prediction  No Yes
#       No  190  16
#       Yes 148  45
                                          
#               Accuracy : 0.589 

#Upsampling is the choice (using Random Forest with CV)

#PredRFModel.rose <- predict(fit.rf.rose, test_set_F)
#confusionMatrix(PredRFModel.rose, test_set_F$match, positive = "Yes")

#result.roc <- roc(test_set_F$match, PredRFModel.up$match[,2]) # Draw ROC curve.
#plot(result.roc, print.thres="best", print.thres.best.method="closest.topleft")

#-------
#varImp(rfModel.up)
#varImpPlot(rfModel.up)
#varImpPlot(rfModel.up, n.var = 10)

varImp(fit.rf.up)
plot(fit.rf.up)

varimpF.up <- varImp(fit.rf.up, scale = FALSE)
plot(varimpF.up, 10)

varimpM.up <- varImp(fit.rf.upM, scale = FALSE)
plot(varimpM.up, 10)

```

#Upsampling on the MALE data

```{r}
#Oversampling

data_balanced_overM <- ovun.sample(match ~ ., data = train_set_M, method = "over",N = 2726)$data
table(data_balanced_overM$match)

set.seed(1)
rfModel.upM <- randomForest(match~.,data_balanced_overM, ntree = 500, importance = TRUE)

rfTestPred.upM <- predict(rfModel.upM, test_set_M, type = "prob")
head(rfTestPred.upM)

test_set_M$RFclass.upM <- predict(rfModel.upM, test_set_M)

confusionMatrix(data = test_set_M$RFclass.upM,
                reference = test_set_M$match,
                positive = "Yes")


#Using 10-fold CV
set.seed(1)
fit.rf.upM <- train(match~., data=data_balanced_overM, method="rf", metric=metric, trControl=control)
set.seed(1)

PredRFModel.upM <- predict(fit.rf.upM, test_set_M)
confusionMatrix(PredRFModel.upM, test_set_M$match, positive = "Yes")


#LDA
set.seed(1)
fit.lda.upM <- train(match~., data=data_balanced_overM, method="lda", metric=metric, trControl=control)
set.seed(1)

PredLDAModel.upM <- predict(fit.lda.upM, test_set_M)
confusionMatrix(PredLDAModel.upM, test_set_M$match, positive = "Yes")


#GLM
set.seed(1)
fit.glm.upM <- train(match~., data=data_balanced_overM, method="glm", metric=metric, trControl=control)
set.seed(1)

PredGLMModel.upM <- predict(fit.glm.upM, test_set_M)
confusionMatrix(PredGLMModel.upM, test_set_M$match, positive = "Yes")



#Decision Tree
set.seed(1)
fit.cart.upM <- train(match~., data=data_balanced_overM, method="rpart", metric=metric, trControl=control)
set.seed(1)

PredCARTModel.upM <- predict(fit.cart.upM, test_set_M)
confusionMatrix(PredCARTModel.upM, test_set_M$match, positive = "Yes")



set.seed(1)
fit.SVM.upM <- train(match~., data=data_balanced_overM, method="svmRadial", metric=metric, trControl=control)
set.seed(1)

PredSVMModel.upM <- predict(fit.SVM.upM, test_set_M)
confusionMatrix(PredSVMModel.upM, test_set_M$match, positive = "Yes")



set.seed(1)
fit.SVML.upM <- train(match~., data=data_balanced_overM, method="svmLinear", metric=metric, trControl=control)
set.seed(1)

PredSVMLModel.upM <- predict(fit.SVML.upM, test_set_M)
confusionMatrix(PredSVMLModel.upM, test_set_M$match, positive = "Yes")



set.seed(1)
fit.SVMP.upM <- train(match~., data=data_balanced_overM, method="svmPoly", metric=metric, trControl=control)
set.seed(1)

PredSVMPModel.upM <- predict(fit.SVMP.upM, test_set_M)
confusionMatrix(PredSVMPModel.upM, test_set_M$match, positive = "Yes")



#KNN
set.seed(1)
fit.knn.upM <- train(match~., data=data_balanced_overM, method="knn", metric=metric, trControl=control)
set.seed(1)

PredknnPModel.upM <- predict(fit.knn.upM, test_set_M)
confusionMatrix(PredknnPModel.upM, test_set_M$match, positive = "Yes")


varimpF.up <- varImp(fit.rf.upM, scale = FALSE)
plot(varimpF.up, 10)

varimpM <- varImp(fit.rfM, scale = FALSE)
plot(varimpM, 10)
```

#FEMALE!!!!
data_balanced_over1 <- ovun.sample(match ~ ., data = final_F, method = "over",N = 3388)$data
table(data_balanced_over1$match)

predictors1 = data_balanced_over1[,-1]
classes1 = data_balanced_over1[,1]

set.seed(1)
trainingRows1 <- createDataPartition(classes1, p = .80, list= FALSE)
train_set_F1 <- data_balanced_over1[trainingRows1,]
test_set_F1 = final_F[-trainingRows1,]

set.seed(1)
fit.rf8 <- train(match~., data=train_set_F1, method="rf", metric=metric, trControl=control)

predUPrf8 <- predict(fit.rf8, test_set_F1)
confusionMatrix(predUPrf8, test_set_F1$match, positive = "Yes")

########
set.seed(1)
fit.lda8 <- train(match~., data=train_set_F1, method="lda", metric=metric, trControl=control)

predUPlda8 <- predict(fit.lda8, test_set_F1)
confusionMatrix(predUPlda8, test_set_F1$match, positive = "Yes")

########
set.seed(1)
fit.glm8 <- train(match~., data=train_set_F1, method="glm", metric=metric, trControl=control)

predUPglm8 <- predict(fit.glm8, test_set_F1)
confusionMatrix(predUPglm8, test_set_F1$match, positive = "Yes")


############
##############
##############MALE!!!##########
data_balanced_over2 <- ovun.sample(match ~ ., data = final_M, method = "over",N = 3388)$data
table(data_balanced_over2$match)

predictors2 = data_balanced_over2[,-1]
classes2 = data_balanced_over2[,1]

set.seed(1)
trainingRows2 <- createDataPartition(classes2, p = .80, list= FALSE)
train_set_M2 <- data_balanced_over2[trainingRows2,]
test_set_M2 = final_M[-trainingRows2,]

set.seed(1)
fit.rf7 <- train(match~., data=train_set_M2, method="rf", metric=metric, trControl=control)

predUPrf7 <- predict(fit.rf7, test_set_M2)
confusionMatrix(predUPrf7, test_set_M2$match, positive = "Yes")

###
set.seed(1)
fit.lda7 <- train(match~., data=train_set_M2, method="lda", metric=metric, trControl=control)

predUPlda7 <- predict(fit.lda7, test_set_M2)
confusionMatrix(predUPlda7, test_set_M2$match, positive = "Yes")

########
set.seed(1)
fit.glm7 <- train(match~., data=train_set_M2, method="glm", metric=metric, trControl=control)

predUPglm7 <- predict(fit.glm7, test_set_M2)
confusionMatrix(predUPglm7, test_set_M2$match, positive = "Yes")

varImp(fit.rf8) #Female
varImp(fit.rf7) #Male

varImpPlot(fit.rf7,type=2)


#varImp(rfModel.up)
#varImpPlot(rfModel.up)
#varImpPlot(rfModel.up, n.var = 10)

#set.seed(1)
#rfModel.up8 <- randomForest(match~.,data_balanced_over1, ntree = 500, importance = TRUE)

varimpF <- varImp(fit.rf8, scale = FALSE)
plot(varimpF)

varimpM <- varImp(fit.rf7, scale = FALSE)
plot(varimpM, 10)



