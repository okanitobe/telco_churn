
#-----------------------------Load packages------------------------------------
library(readxl)
library(tidyverse)
library(rpart)
library(caret)
library(adabag)
library(ggpubr)
library(pROC)

#------------------------------Import Data--------------------------------------
telco = read_excel("data/Telco_customer_churn.xlsx")

#-----------------------------Dimension Reduction-------------------------------
#variables we dont need:
#--CustomerID
#--Count
#--Country
#--State
#--City
#--Zip Code
#--Lat Long
#--Latitude
#--Longitude
#--Churn Reason
#--Churn Label (redundant)
#--Churn Score
#--CLTV

subdf = telco %>% 
  select(-c(CustomerID, Count, Country, State, City, `Zip Code`, 'Lat Long', Latitude, Longitude, 'Churn Reason', 'Churn Label', 'Churn Score', CLTV))

#--------------------------------Clean subdf------------------------------------
colnames(subdf) = gsub(" ", "", colnames(subdf))

subdf = as.data.frame(lapply(subdf, function(x){ifelse(x=="No phone service", "No", x)}))
subdf = as.data.frame(lapply(subdf, function(x){ifelse(x=="No internet service", "No", x)}))

columns = colnames(subdf)[-c(5,18,19)] #exclude the columns we don't want to factor
subdf = subdf %>% mutate_at(columns, factor)

#------------------------------------EDA----------------------------------------
churnpercent = prop.table(table(subdf$ChurnValue)) *100
pie(churnpercent, labels = paste(round(churnpercent, 2), "%" ,sep = ""))

ggarrange(ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(Gender, position = 'fill')), 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(SeniorCitizen, position = 'fill')),
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(Partner, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(Dependents, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(PhoneService, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(MultipleLines, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(InternetService, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(OnlineSecurity, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(OnlineBackup, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(DeviceProtection, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(TechSupport, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(StreamingTV, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(StreamingMovies, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(Contract, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(PaperlessBilling, position = 'fill')) , 
          ggplot(data = subdf, aes(fill=ChurnValue)) +geom_bar(aes(PaymentMethod, position = 'fill')), 
          ncol = 4, nrow=4)



#---------------Split the dataset into Training and Testing---------------------
#Baseline accuracy: the ZeroR Classfier predicts the majority class
table(subdf$ChurnValue)
prop.table(table(subdf$ChurnValue)) #baseline accuracy is 73.46%. In order for classifier to be useful, it needs to exceed this accuracy


#Create training index to split data between testing and training sets
set.seed(123)
train = sample(1:nrow(subdf), nrow(subdf)*(.70))

#Use train index to split the dataset
trainSplit = subdf[train , ]
testSplit = subdf[-train , ]


#------------------------Logistic Regression-----------------------------------
fit.glm = glm(ChurnValue ~., data = trainSplit, family = "binomial")

summary(fit.glm)

#---Calculate accuracy on trainSplit
glm.pred = ifelse(predict(fit.glm, trainSplit, type = "response") >= 0.5, 1, 0 )
cm.glm = confusionMatrix(as.factor(glm.pred), trainSplit$ChurnValue, positive = "1"); cm.glm

#---Calculate accuracy on testSplit
glm.pred = ifelse(predict(fit.glm, testSplit, type = "response") >= 0.5, 1, 0 )
cm.glm = confusionMatrix(as.factor(glm.pred), testSplit$ChurnValue, positive = "1"); cm.glm


#--------------------------Decision Tree---------------------------------------
#---create fit for tree
#---low minsplit to create large tree
fit.tree = rpart(ChurnValue ~ ., 
            data = trainSplit, 
            method = "class", 
            control = rpart.control(xval=10, minsplit = 2, cp=0),
            parms = list(split="gini"))


#---Prune the tree
#---checking for Complexity Parameter (cp) in the fit's cptable that results in the lowest cross-validation error (xerror)
bestcp = fit.tree$cptable[which.min(fit.tree$cptable[ , "xerror"]) , "CP"]; bestcp

fit.prune = prune.rpart(fit.tree, cp=bestcp)

#---Calculate accuracy on trainSplit
tree.pred = predict(fit.prune, trainSplit, type = "class")
cm.tree = confusionMatrix(tree.pred, trainSplit$ChurnValue, positive = "1"); cm.tree

#---Calculate accuracy on testSplit
tree.pred = predict(fit.prune, testSplit, type = "class")
cm.tree = confusionMatrix(tree.pred, testSplit$ChurnValue, positive = "1"); cm.tree


#-----------------------------Plotting ROC Curves-------------------------------
plot.roc(testSplit$ChurnValue, glm.pred, col="red", print.auc=TRUE, legacy.axes=TRUE)
plot.roc(testSplit$ChurnValue, as.numeric(tree.pred), col="blue",print.auc=TRUE, print.auc.y=0.4, legacy.axes=TRUE, add=TRUE)
legend("bottomright", c("Logistic Regression", "Decision Tree"), col = c("red", "blue"), lwd=4)
