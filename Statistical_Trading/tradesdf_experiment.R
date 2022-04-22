library(caret)
library(caTools)
library(rpart)
library(rattle)
library(rpart)
library(rpart.plot)
library(InformationValue)
library(ISLR)

setwd("C:/Users/serca/OneDrive/Masaüstü/Python_Files/Statistical_Trading")
data=read.csv("Test_Short_Trades_DF.csv")
print(data)
data$Win.Situation=as.factor(data$Win.Situation)
data$L.T..Win.Situation = as.factor(data$L.T..Win.Situation)

split=sample.split(data$Win.Situation,SplitRatio=0.8) ################
data_train=subset(data,split==TRUE) ##################
data_test=subset(data,split==FALSE) ###############

################### Random Forest Model ###############
rf_model_caret=train(Win.Situation~ Price.Fisher
                     + L.T..Win.Situation 
                     +L.T..Time.in.Trade
                      , data=data_train, method = "rf",
                     trControl = trainControl(method="repeatedcv",number=10, repeats = 3), tuneGrid = expand.grid(.mtry = (5:8)), ntree =(100))
print(rf_model_caret$finalModel)

rf_model_caret_pred=predict(rf_model_caret,newdata=data_test)
print(rf_model_caret_pred)
print(table(data_test$Win.Situation,rf_model_caret_pred))
print(confusionMatrix(rf_model_caret_pred,data_test$Win.Situation))

######################## RPART TREE MODEL ##################
tree_model=rpart(Win.Situation~ Sell.Volumes 
                  
                 +L.T..Time.in.Trade+L.T..Max.Drawdown
                  ,data=data_train)
print(tree_model$finalModel)
prp(tree_model)

tree_model_pred=predict(tree_model,newdata=data_test, type= 'class')
print(tree_model_pred)
print(table(data_test$Win.Situation,tree_model_pred))
print(confusionMatrix(tree_model_pred,data_test$Win.Situation))


##################### Test Predictions Short #################

short_data=read.csv("Test_Short_Trades_DF.csv")
short_data$Win.Situation=as.factor(short_data$Win.Situation)
short_data$L.T..Win.Situation = as.factor(short_data$L.T..Win.Situation)

rf_model_caret_test_pred=predict(rf_model_caret,newdata=short_data)
print(rf_model_caret_test_pred)
print(table(short_data$Win.Situation,rf_model_caret_test_pred))
print(confusionMatrix(rf_model_caret_test_pred,short_data$Win.Situation))


tree_model_test_pred=predict(tree_model,newdata=short_data, type= 'class')
print(tree_model_test_pred)
print(table(short_data$Win.Situation,tree_model_test_pred))
print(confusionMatrix(tree_model_test_pred,short_data$Win.Situation))
#######################################################################
##############################################


log.model <- glm(Win.Situation ~ Price.Fisher+EMA270Fisher.50.
                 +Sell.Volumes+RSI_
                 +L.T..Time.in.Trade+L.T..Win.Situation
                 +L.T..Max.Drawdown +Win.Sequence
                 , data = data_train, family = 'binomial')

summary(log.model)
predicted <- predict(log.model, data, type="response")

#convert defaults from "Yes" and "No" to 1's and 0's
#data_test$Win.Situation <- ifelse(data_test$Win.Situation=="Yes", 1, 0)

#find optimal cutoff probability to use to maximize accuracy
optimal <- optimalCutoff(data$Win.Situation, predicted)[1]

#create confusion matrix
confusionMatrix(data$Win.Situation, predicted)
