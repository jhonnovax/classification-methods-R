library(caTools)
library(randomForest)
library(rpart)
library(rpart.plot)
library(caret)
library(Metrics)
library(ipred)
library(xgboost)

# Load dataset
churn_data <- read.csv("Special Topics/Assignment 1/telecom_churn.csv")
head(churn_data)

# Partition the Data using a common split (70% train, 30% test)
set.seed(100)
sample_split <- sample.split(Y = churn_data$Churn, SplitRatio = 0.7)
train_set <- subset(x = churn_data, sample_split == TRUE)
test_set <- subset(x = churn_data, sample_split == FALSE)

# Decision Tree
origin_pred = test_set$Churn
dt_model <- rpart(Churn~., data = train_set, method = "class")
dt_pred <- predict(dt_model, test_set[,-1], type = "class")

confusionMatrix(factor(origin_pred), factor(dt_pred))
accuracy(origin_pred, dt_pred)
f1(origin_pred, dt_pred)
ce(origin_pred, dt_pred)
rpart.plot(dt_model)

# Random Forest
origin_pred = test_set$Churn
rf_model <- randomForest(Churn ~ ., data=train_set, ntree=50, ntry=3, importance=TRUE)
rf_pred <- predict(rf_model, newdata = test_set[,-1])
rf_pred<-ifelse(rf_pred> 0.5,1,0)

confusionMatrix(factor(origin_pred), factor(rf_pred))
accuracy(origin_pred, rf_pred)
f1(origin_pred, rf_pred)
ce(origin_pred, rf_pred)
varImpPlot(rf_model, col=3)

# Bagging
origin_pred = test_set$Churn
bagging_model <- bagging(formula=Churn~., data=train_set, nbagg=50, coob=TRUE, control=rpart.control(minsplit=2, cp=0, min_depth=2))
bagging_pred <- predict(bagging_model, test_set[,-1], type = "class")

confusionMatrix(factor(origin_pred), factor(bagging_pred))
accuracy(origin_pred, bagging_pred)
f1(origin_pred, bagging_pred)
ce(origin_pred, bagging_pred)

# XGBoost
origin_pred = test_set$Churn
train_matrix <- as.matrix(train_set[,-1]) # Prepare train data (excluding the target variable for training)

xgb_model <- xgboost(data=train_matrix, label=train_set$Churn, max.depth=3, nrounds=4, eta=0.1, nthread=2, objective="binary:logistic")
test_matrix <- as.matrix(test_set[, -1]) # Prepare test data
xgb_pred_prob <- predict(xgb_model, test_matrix)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 1, 0) # Convert probabilities to binary classes

confusionMatrix(factor(origin_pred), factor(xgb_pred))
accuracy(origin_pred, xgb_pred)
f1(origin_pred, xgb_pred)
ce(origin_pred, xgb_pred)

feature_names <- colnames(train_matrix)
importance_matrix <- xgb.importance(feature_names = feature_names, model = xgb_model)
xgb.plot.importance(importance_matrix)




