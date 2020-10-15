library(caret)
library(randomForest)

data <- read.csv("data/HR-Employee-Attrition.csv")
names(data)[1] <- "Age"

set.seed(2020)

# new features
data$RoleChangeYear <- data$YearsAtCompany - data$YearsInCurrentRole
data$PromChangeYear <- data$YearsAtCompany - data$YearsSinceLastPromotion
data$ManagerChangeYear <- data$YearsAtCompany - data$YearsWithCurrManager
data$JobChangeYear <- data$TotalWorkingYears - data$YearsAtCompany
data$AvgCompYear <- data$TotalWorkingYears / data$NumCompaniesWorked
data$DayMonthRate <- data$DailyRate / data$MonthlyRate
data$StartAge <- data$Age - data$TotalWorkingYears
data$WorkLifePercent <- data$StartAge / data$Age
data$is_firstJob <- ifelse(data$NumCompaniesWorked == 0, 1, 0)
# data$is_trainedLY <- ifelse(data$TrainingTimesLastYear != 0, 1, 0)
data$AnnualSalary <- data$MonthlyIncome * 12 
data$is_promoted <- ifelse(data$YearsInCurrentRole - data$YearsSinceLastPromotion != 0, 1, 0)

data[data == Inf] <- 0

# drop constant columns
data <- data[,-nearZeroVar(data)]

head(data)

splitter <- createDataPartition(data$Attrition, list = F, p =0.8)
  
train <- data[splitter,]
test <- data[-splitter,]

ctrl <- trainControl(method = "cv",
                     number = 5,
                     classProbs = TRUE,
                     summaryFunction = prSummary,
                     search = 'random')

k = 0.7
model_weights <- ifelse(train$Attrition == "Yes",
                        (1/table(train$Attrition)[1]) * k,
                        (1/table(train$Attrition)[2]) * (1-k))


# mtry <- sqrt(ncol(data))
# tunegrid <- expand.grid(.mtry=mtry)


model <- train(Attrition ~ ., method = "rf", 
              data = train,
             trControl = ctrl,
             # tuneGrid = tunegrid,
             verbose = FALSE, 
             ntree = 150,
             weights = model_weights,
             nodesize = 3,
             tuneLength  = 10,
             )

model
confusionMatrix(predict(model, test), reference = as.factor(test$Attrition), mode = "prec_recall")

acc <- confusionMatrix(predict(model, test), reference = as.factor(test$Attrition), mode = "prec_recall")$overall[1]
f1 <- confusionMatrix(predict(model, test), reference = as.factor(test$Attrition), mode = "prec_recall")$byClass[7]
pre <- confusionMatrix(predict(model, test), reference = as.factor(test$Attrition), mode = "prec_recall")$byClass[5]
rec <- confusionMatrix(predict(model, test), reference = as.factor(test$Attrition), mode = "prec_recall")$byClass[6]


png("rf_model.png")
plot(model)
dev.off()

results <- paste0("Accuracy: ", round(acc, 6), "\n", 
                 "F1: ", round(f1, 6),  "\n",
                 "Precision: ", round(pre, 6),  "\n",
                 "Recall: ", round(rec, 6), "\n")
write.table(results, file = 'metrics.txt', col.names = FALSE, row.names = FALSE, quote = FALSE)
