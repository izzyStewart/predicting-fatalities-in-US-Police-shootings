### TO BE RUN SECOND (after running 'preprocess_data-cleaning.R')


#####################

# Load packages

#####################

library(devtools)
library(tidyr)
library(dplyr)
library(bitops)
library(randomForest)
library(gplots)
library(mice)
library(dummies)
library(stringr)
library(dplyr)
library(splitstackshape)
library(caret)
library(pROC)
library(ROCR)
library(AppliedPredictiveModeling)
library(e1071)
library(DMwR)
library(nnet)
library(class)


#####################

# READ IN DATA

#####################

# Read in cleaned train and test dataset.
cleaned_train = 
read.delim("~/A2-MLSDM/code-files/data/cleaned_train.csv", 
header=TRUE,sep = ',', stringsAsFactors = TRUE)

# Read in cleaned test dataset.
cleaned_test = 
  read.delim("~/A2-MLSDM/code-files/data/cleaned_test.csv", 
             header=TRUE,sep = ',', stringsAsFactors = TRUE)

train = cleaned_train
test = cleaned_test

# Look at data.
View(train)

# Update features to correct types.
train$Fatal = factor(train$Fatal)
test$Fatal = factor(test$Fatal)
train$DateYear = factor(train$DateYear)
test$DateYear = factor(test$DateYear)
train[,c(10:126)] = train[,c(10:126)] %>% mutate_if(is.integer,as.factor) 
test[,c(10:126)] = test[,c(10:126)] %>% mutate_if(is.integer,as.factor)
str(train) # Check results.

levels(train$Fatal) = c("NotFatal", "Fatal")
levels(test$Fatal) = c("NotFatal", "Fatal")


#####################

# EXPLORE DATA

#####################

# Numeric column - number of officers
plot = ggplot(train, aes(x=Fatal, y=NumberOfOfficers)) + 
  stat_summary(fun.y="mean", geom="bar")

print(plot + ggtitle("Number of Officers - Fatal"))

# Seperate the factor/ category variables.
cat_cols = subset(train, select = c(SubjectArmed, SubjectRace, SubjectGender,
                                    NumberOfShots, City, DateYear, 
                                    SubjectAgeRange, OfficerRace, 
                                    OfficerGender))

# Look at the number of categories in each feature.
summary(cat_cols)

# Look at ratio of Subject race to Fatal. 
barplot(prop.table(table(train$Fatal, train$SubjectRace)),
        main = "Ratio of Subject Race to Fatal", 
        xlab = "Subject Race", ylab = "Frequency")

# Look at ratio of Subject armed to Fatal. 
barplot(prop.table(table(train$Fatal, train$SubjectArmed)),
        main = "Ratio of Subject Armed to Fatal", 
        xlab = "Subject Armed", ylab = "Frequency")

# Look at ratio of Subject gender to Fatal. 
barplot(prop.table(table(train$Fatal, train$SubjectGender)),
        main = "Ratio of Subject Gender to Fatal", 
        xlab = "Subject Gender", ylab = "Frequency")

# Look at ratio of Subject age range to Fatal. 
barplot(prop.table(table(train$Fatal, train$SubjectAgeRange)),
        main = "Ratio of Subject Age range to Fatal", 
        xlab = "Subject Age range", ylab = "Frequency")

# Class imbalance - Fatal
table(train$Fatal)

# Percentage Not Fatal = 66%
(1573/(1573+810))*100

# Percentage Fatal = 34%
(810/(1573+810))*100


#####################

# DATA SPLITTING

#####################


# 1. SPLIT: TRAIN - VALIDATION

#####################

# Split based on outcome 'Fatal.' Split 80/20 train and validation respectively.
set.seed(2)
t_index = createDataPartition(train$Fatal, p = .8, 
                                 list = FALSE, 
                                 times = 1)

# Training and validation sets.
train = train[ t_index,]
validation  = train[-t_index,]


# 2. SPLIT: TRAIN - NEAR ZERO VARIANCE

#####################

# First check the train for features with zero variance.
near_zero = nearZeroVar(train, saveMetrics = TRUE)
near_zero[near_zero[,"zeroVar"] > 0, ]
# Removing these predictors from the train, test and validation.
train = subset(train, select = -c(OR_0_0_1_0_1_0, OR_0_1_0_1_1_0,
                                  OR_0_1_1_0_0_0, OR_1_1_0_0_1_0,
                                  OG_1_0_1))

test = subset(test, select = -c(OR_0_0_1_0_1_0, OR_0_1_0_1_1_0,
                                  OR_0_1_1_0_0_0, OR_1_1_0_0_1_0,
                                  OG_1_0_1))

validation = subset(validation, select = -c(OR_0_0_1_0_1_0, OR_0_1_0_1_1_0,
                                OR_0_1_1_0_0_0, OR_1_1_0_0_1_0,
                                OG_1_0_1))

# Some models can handle low variance better than others,
# so creating new train set removing nvp.

# Check for features with near zero variance.
nearZeroVar(train)
rem_zvp = nearZeroVar(train)
train_NZV = train[, -rem_zvp]
test_NZV = test[, -rem_zvp]
validation_NZV = validation[, -rem_zvp]
names(train_NZV) # Check remaining results

# Remove one of each feature from remaining full set of dummy variables
# this is to avoid the 'dummy variable trap' / colinearity.
train_NZV = subset(train_NZV, select = -c(Y_2012, SA_Y))
test_NZV = subset(test_NZV, select = -c(Y_2012, SA_Y))
validation_NZV = subset(validation_NZV, select = -c(Y_2012, SA_Y))


# 3. SPLIT: TRAIN - X3 RE-SAMPLES

#####################

set.seed(2)
upSampledTrain = upSample(x = train[,-1],
                          y = train$Fatal,
                          yname = "Fatal")

set.seed(2)
downSampledTrain = downSample(x = train[,-1],
                          y = train$Fatal,
                          yname = "Fatal")

set.seed(2)
SmoteTrain = SMOTE(Fatal ~ ., data = train, perc.over = 100, perc.under=200)

# Now doing the same for sets with near zero variance removed.

set.seed(2)
upSampledTrainNZV = upSample(x = train_NZV[,-1],
                          y = train_NZV$Fatal,
                          yname = "Fatal")

set.seed(2)
downSampledTrainNZV = downSample(x = train_NZV[,-1],
                          y = train_NZV$Fatal,
                          yname = "Fatal")

set.seed(2)
SmoteTrainNZV = SMOTE(Fatal ~ ., data = train_NZV, perc.over = 100, perc.under=200)


dim(train)
table(train$Fatal)
dim(upSampledTrain)
table(upSampledTrain$Fatal)
dim(downSampledTrain)
table(downSampledTrain$Fatal)
dim(SmoteTrain)
table(SmoteTrain$Fatal)


# 4. SPLIT: TRAIN - GROUPED VS INDEPENDANT (CATEGORICAL)

#####################

# Removing dummy variables for 'grouped variables' split.
train_GV = train[,!grepl("*_",names(train))]
validation_GV = validation[,!grepl("*_",names(validation))]
test_GV = test[,!grepl("*_",names(test))]

# Removing original factor variable for random forest test two.
train_IV = subset(train, select = -c(DateYear, SubjectArmed, SubjectRace, SubjectGender,
                                     SubjectAgeRange, NumberOfShots, City, OfficerRace, 
                                     OfficerGender))
                                     
validation_IV = subset(validation, select = -c(DateYear, SubjectArmed, SubjectRace, 
                                               SubjectGender, SubjectAgeRange, 
                                               NumberOfShots, City, OfficerRace,
                                               OfficerGender))

test_IV = subset(test, select = -c(DateYear, SubjectArmed, SubjectRace, SubjectGender,
                                   SubjectAgeRange, NumberOfShots, City, OfficerRace, 
                                   OfficerGender))

# Splitting valX and valY & testX and testY
valX_GV = validation_GV[,-1]
valX_IV = validation_IV[,-1]
valY = validation$Fatal
testX_GV = test_GV[,-1]
testX_IV = test_IV[,-1]
testY = test$Fatal


#####################

# REPORTING STATS

#####################

fiveStats = function(...) c(twoClassSummary(...), 
                            defaultSummary(...))


ctrl = trainControl(method = "cv",
                    number = 10,
                    classProbs = TRUE,
                    summaryFunction = fiveStats,
                    verboseIter = TRUE)


#####################

# RANDOM FOREST - GROUPED VS INDEPENDENT

#####################


# BUILD MODELS 

#####################

set.seed(100)

# Fitting model for train set where categorical data is grouped.
rfFit1 = train(Fatal ~ ., data = train_GV,
               method = "rf",
               trControl = ctrl, # Controls set by 'ctrl' variable above.
               ntree = 1500, # Number of trees 1500.
               tuneLength = 5, # Tune length set to 5.
               metric = "ROC")

set.seed(100)

# Fitting model for train set where categorical data is independent (dummy variables).
rfFit2 = train(Fatal ~ ., data = train_IV,
               method = "rf",
               trControl = ctrl,
               ntree = 1500,
               tuneLength = 5,
               metric = "ROC")


# LOOK AT MODEL RESULTS 

#####################

# Look at grouped data fitted model.
print(rfFit1)
print(rfFit1$finalModel)          

# Look at independent data fitted model.
print(rfFit2)
print(rfFit2$finalModel) 



# PREDICT - VALIDATION SET 

#####################
               
# Creating dataframe to hold the predicted results from the validation set.
evalResults = data.frame(Fatal = valY)
evalResults$RfG = predict(rfFit1, newdata = validation_GV, type = "prob")[,1]
evalResults$RfI = predict(rfFit2, newdata = validation_IV, type = "prob")[,1]

# Building the ROC curve for the grouped data model.
rfROC_G = roc(evalResults$Fatal, evalResults$RfG,
             levels = rev(levels(evalResults$Fatal)))

# Building the ROC curve for the independent data model.
rfROC_I = roc(evalResults$Fatal, evalResults$RfI,
             levels = rev(levels(evalResults$Fatal)))

# Checking results grouped.
rfROC_G
predictions_G = predict(rfFit1, validation_GV)
confusionMatrix(predictions_G, validation_GV$Fatal)

# Correct predictions = 68.48739%
((324+2)/476)*100

# Checking results independent.
rfROC_I
predictions_I = predict(rfFit2, validation_IV)
confusionMatrix(predictions_I, validation_IV$Fatal)

# Correct predictions = 68.48739%
((324+2)/476)*100

# Plotting results.
plot(rfROC_G, legacy.axes = TRUE, main="ROC curve: Grouped vs seperate - validation", 
     col=c("red"))

lines(rfROC_I, col="green")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green"), 
  legend = c("Grouped", "Seperate")
)

###### Results show very small improvement for independent. 
# Grouped model AUC: 0.744
# Independant model AUC: 0.7599


# PREDICT - TEST SET 

#####################

# Fixing unmatched levels.
levels(train_GV$OfficerRace)
levels(test_GV$OfficerRace) # Some factors not in test.
levels(test_GV$OfficerRace) = levels(train_GV$OfficerRace)
               
# Creating dataframe to hold the predicted results from the test set.
testResults = data.frame(Fatal = testY)
testResults$RfG = predict(rfFit1, newdata = test_GV, type = "prob")[,1]
testResults$RfI = predict(rfFit2, newdata = test_IV, type = "prob")[,1]



# Building the ROC curve for the grouped data model.
rfROC_GT = roc(testResults$Fatal, testResults$RfG,
             levels = rev(levels(testResults$Fatal)))

# Building the ROC curve for the independent data model.
rfROC_IT = roc(testResults$Fatal, testResults$RfI,
             levels = rev(levels(testResults$Fatal)))

# Checking results grouped.
rfROC_GT
predictions_GT = predict(rfFit1, test_GV)
confusionMatrix(predictions_GT, test_GV$Fatal)

# Correct predictions = 65.86022%
((490+0)/744)*100

# Checking results independent.
rfROC_IT
predictions_IT = predict(rfFit2, test_IV)
confusionMatrix(predictions_IT, test_IV$Fatal)

# Correct predictions = 65.86022%
((490+0)/744)*100

# Plotting results.
plot(rfROC_GT, legacy.axes = TRUE, main="ROC curve: Grouped vs seperate - test", 
     col=c("red"))

lines(rfROC_IT, col="green")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green"), 
  legend = c("Grouped", "Seperate")
)

###### Results show... 
# Grouped model AUC: 0.7271 
# Independant model AUC: 0.7246


#####################

# RANDOM FOREST - NORMAL VS RESAMPLED

#####################


# REMOVE GROUPED FEATURES

#####################

upSampledTrain = subset(upSampledTrain, select = -c(DateYear, SubjectArmed, 
                                                    SubjectRace, SubjectGender,
                                                    SubjectAgeRange, NumberOfShots, 
                                                    City, OfficerRace, 
                                                    OfficerGender))

downSampledTrain = subset(downSampledTrain, select = -c(DateYear, SubjectArmed, 
                                                    SubjectRace, SubjectGender,
                                                    SubjectAgeRange, NumberOfShots, 
                                                    City, OfficerRace, 
                                                    OfficerGender))

SmoteTrain = subset(SmoteTrain, select = -c(DateYear, SubjectArmed, 
                                                    SubjectRace, SubjectGender,
                                                    SubjectAgeRange, NumberOfShots, 
                                                    City, OfficerRace, 
                                                    OfficerGender))

# BUILD MODELS 

#####################

set.seed(200)

rfFitUp = train(Fatal ~ ., data = upSampledTrain,
               method = "rf",
               trControl = ctrl,
               ntree = 1500,
               tuneLength = 5,
               metric = "ROC")

rfFitDown = train(Fatal ~ ., data = downSampledTrain,
                method = "rf",
                trControl = ctrl,
                ntree = 1500,
                tuneLength = 5,
                metric = "ROC")

rfFitSMOTE = train(Fatal ~ ., data = SmoteTrain,
                  method = "rf",
                  trControl = ctrl,
                  ntree = 1500,
                  tuneLength = 5,
                  metric = "ROC")
                  
# LOOK AT MODEL RESULTS 

#####################

# Look at upsampled fitted model.
print(rfFitUp)
print(rfFitUp$finalModel)

# Look at downsampled fitted model.
print(rfFitDown)
print(rfFitDown$finalModel)

# Look at SMOTE fitted model.
print(rfFitSMOTE)
print(rfFitSMOTE$finalModel)


# PREDICT - VALIDATION SET 

#####################

# Adding predicted results to evaluation dataframe.
evalResults$RFUp = predict(rfFitUp, newdata = validation_IV, type = "prob")[,1]
evalResults$RFDown = predict(rfFitDown, newdata = validation_IV, type = "prob")[,1]
evalResults$RFSmote = predict(rfFitSMOTE, newdata = validation_IV, type = "prob")[,1]

# Building the ROC curve for the upsampled data model.
rfROCUp = roc(evalResults$Fatal, evalResults$RFUp,
             levels = rev(levels(evalResults$Fatal)))

# Building the ROC curve for the downsampled data model.
rfROCDown = roc(evalResults$Fatal, evalResults$RFDown,
              levels = rev(levels(evalResults$Fatal)))

# Building the ROC curve for the SMOTE data model.
rfROCSmote = roc(evalResults$Fatal, evalResults$RFSmote,
                levels = rev(levels(evalResults$Fatal)))

# Checking results for upsample predictions.
rfROCUp
predictions_U = predict(rfFitUp, validation_IV)
confusionMatrix(predictions_U, validation_IV$Fatal)

# Correct predictions = 94.95798%
((310+142)/476)*100

# Checking results for downsample predictions.
rfROCDown
predictions_D = predict(rfFitDown, validation_IV)
confusionMatrix(predictions_D, validation_IV$Fatal)

# Correct predictions = 67.01681%
((222+97)/476)*100

# Checking results for SMOTE predictions.
rfROCSmote
predictions_S = predict(rfFitSMOTE, validation_IV)
confusionMatrix(predictions_S, validation_IV$Fatal)

# Correct predictions = 85.92437%
((275+134)/476)*100

# Plotting results.
plot(rfROC_I, legacy.axes = TRUE, main="ROC curve: Sampling methods - validation", 
     col = "red")

lines(rfROCUp, col="green")
lines(rfROCDown, col="blue")
lines(rfROCSmote, col="purple")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green","blue","purple"), 
  legend = c("Normal", "Upsample", "Downsample", "SMOTE")
)

###### Results show... 
# Normal model AUC: 0.7599
# Upsample model AUC: 0.9801
# Downsample model AUC: 0.7279
# SMOTE model AUC: 0.9269


# PREDICT - TEST SET 

#####################

# Adding predicted results to test dataframe.
testResults$RFUp = predict(rfFitUp, newdata = test_IV, type = "prob")[,1]
testResults$RFDown = predict(rfFitDown, newdata = test_IV, type = "prob")[,1]
testResults$RFSmote = predict(rfFitSMOTE, newdata = test_IV, type = "prob")[,1]

# Building the ROC curve for the upsampled data model.
rfROCUpT = roc(testResults$Fatal, testResults$RFUp,
             levels = rev(levels(testResults$Fatal)))

# Building the ROC curve for the downsampled data model.
rfROCDownT = roc(testResults$Fatal, testResults$RFDown,
              levels = rev(levels(testResults$Fatal)))

# Building the ROC curve for the SMOTE data model.
rfROCSmoteT = roc(testResults$Fatal, testResults$RFSmote,
                levels = rev(levels(testResults$Fatal)))

# Checking results for upsample predictions.
rfROCUpT
predictions_UT = predict(rfFitUp, test_IV)
confusionMatrix(predictions_UT, test_IV$Fatal)

# Correct predictions = 68.41398%
((350+159)/744)*100

# Checking results for downsample predictions.
rfROCDownT
predictions_DT = predict(rfFitDown, test_IV)
confusionMatrix(predictions_DT, test_IV$Fatal)

# Correct predictions = 68.6828%
((341+170)/744)*100

# Checking results for SMOTE predictions.
rfROCSmoteT
predictions_ST = predict(rfFitSMOTE, test_IV)
confusionMatrix(predictions_ST, test_IV$Fatal)

# Correct predictions = 68.14516%
((356+151)/744)*100

# Plotting results.
plot(rfROC_IT, legacy.axes = TRUE, main="ROC curve: Sampling methods - test", 
     col = "red")

lines(rfROCUpT, col="green")
lines(rfROCDownT, col="blue")
lines(rfROCSmoteT, col="purple")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green","blue","purple"), 
  legend = c("Normal", "Upsample", "Downsample", "SMOTE")
)

###### Results show... 
# Normal model AUC: 0.7246  
# Upsample model AUC: 0.7371 
# Downsample model AUC: 0.731
# SMOTE model AUC: 0.7356

# Saving best performing model
save(rfFitUp, file = "RF_best.rda")


# RANDOM FOREST - IMPORTANT VARIABLES 

#####################

set.seed(200)

# Fit random forest model using optimum tuning defined above
train.rf = randomForest(Fatal ~ ., data = upSampledTrain, 
                        # Set 'importance = TRUE'
                        mtry = 56, ntree = 1500, importance = TRUE) 

# View new importance plot.
varImpPlot(train.rf) 

set.seed(200)

# Fit random forest model using optimum tuning defined above, 
# with only important variables.
rfFitUpImp = train(Fatal ~ NumberOfOfficers + SR_W + NOS_M + 
                     Y_2016 + Y_2010 + Y_2013 + Y_2012 + 
                     SA_Y + Y_2015 + C_Chicago + Y_2011 + 
                     NOS_N.A + SA_N + SG_U + SAR_20.29 + 
                     SA_U + OR_0_1_0_0_0_0 + OR_1_0_0_0_0_0 + 
                     Y_2014 + C_MiamiDade + SAR_0.19 + 
                     SAR_40.49 + SR_B + NOS_S + 
                     SR_L + OG_0_0_1 + C_NewYork + 
                     SAR_30.39 + C_LosAngeles + 
                     C_Dallas, data = upSampledTrain,
                   method = "rf",
                   trControl = ctrl,
                   ntree = 1500,
                   tuneLength = 5,
                   metric = "ROC")

# LOOK AT MODEL RESULTS 

#####################

# Look at upsampled fitted model.
print(rfFitUpImp)
print(rfFitUpImp$finalModel)


# PREDICT - VALIDATION SET 

#####################

# Adding predicted results to evaluation dataframe.
evalResults$RFIm = predict(rfFitUpImp, newdata = validation_IV, type = "prob")[,1]

# Building the ROC curve for the upsampled data model.
rfROCImp = roc(evalResults$Fatal, evalResults$RFIm,
             levels = rev(levels(evalResults$Fatal)))

# Checking results for upsample predictions.
rfROCImp
predictions_Im = predict(rfFitUpImp, validation_IV)
confusionMatrix(predictions_Im, validation_IV$Fatal)

# Correct predictions = 92.43697%
((302+138)/476)*100


# Plotting results.
plot(rfROCUp, legacy.axes = TRUE, main="ROC curve: Important features - validation", 
     col = "red")

lines(rfROCImp, col="green")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green","blue","purple"), 
  legend = c("Normal", "Important")
)

###### Results show... 
# Upsample - Normal model AUC: 0.9801
# Upsample - Important model AUC: 0.9532


# PREDICT - TEST SET 

#####################

# Adding predicted results to test dataframe.
testResults$RFIm = predict(rfFitUpImp, newdata = test_IV, type = "prob")[,1]

# Building the ROC curve for the upsampled data model.
rfROCImpT = roc(testResults$Fatal, testResults$RFIm,
             levels = rev(levels(testResults$Fatal)))

# Checking results for upsample predictions.
rfROCImpT
predictions_ImT = predict(rfFitUpImp, test_IV)
confusionMatrix(predictions_ImT, test_IV$Fatal)

# Correct predictions = 67.74194%
((354+150)/744)*100


# Plotting results.
plot(rfROCUpT, legacy.axes = TRUE, main="ROC curve: Important features - test", 
     col = "red")

lines(rfROCImpT, col="green")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green","blue","purple"), 
  legend = c("Normal", "Important")
)

###### Results show... 
# Upsample - Normal model AUC: 0.7371
# Upsample - Important model AUC: 0.7193


#####################

# LOGISTIC REGRESSION - NORMAL VS RESAMPLED

#####################


# REMOVE GROUPED FEATURES

#####################

validation_NZV = subset(validation_NZV, select = -c(DateYear, SubjectArmed, 
                                                    SubjectRace, SubjectGender,
                                                    SubjectAgeRange, 
                                                    NumberOfShots, 
                                                    City, OfficerRace, 
                                                    OfficerGender))     

train_NZV = subset(train_NZV, select = -c(DateYear, SubjectArmed, 
                                        SubjectRace, SubjectGender,
                                        SubjectAgeRange, NumberOfShots, 
                                        City, OfficerRace, 
                                        OfficerGender))

test_NZV = subset(test_NZV, select = -c(DateYear, SubjectArmed, 
                                      SubjectRace, SubjectGender,
                                      SubjectAgeRange, NumberOfShots, 
                                      City, OfficerRace, 
                                      OfficerGender))                                                   

upSampledTrainNZV = subset(upSampledTrainNZV, select = -c(DateYear, SubjectArmed, 
                                                          SubjectRace, 
                                                          SubjectGender,
                                                          SubjectAgeRange, 
                                                          NumberOfShots, 
                                                          City, OfficerRace, 
                                                          OfficerGender))

downSampledTrainNZV = subset(downSampledTrainNZV, select = -c(DateYear, 
                                                              SubjectArmed, 
                                                              SubjectRace, 
                                                              SubjectGender,
                                                              SubjectAgeRange, 
                                                              NumberOfShots, 
                                                              City, OfficerRace, 
                                                              OfficerGender))

SmoteTrainNZV = subset(SmoteTrainNZV, select = -c(DateYear, SubjectArmed, 
                                                  SubjectRace, SubjectGender,
                                                  SubjectAgeRange, NumberOfShots, 
                                                  City, OfficerRace, 
                                                  OfficerGender))



train_NZV$Fatal = relevel(train_NZV$Fatal, ref = "NotFatal")
test_NZV$Fatal = relevel(test_NZV$Fatal, ref = "NotFatal")
validation_NZV$Fatal = relevel(validation_NZV$Fatal, ref = "NotFatal")
upSampledTrainNZV$Fatal = relevel(upSampledTrainNZV$Fatal, ref = "NotFatal")
downSampledTrainNZV$Fatal = relevel(downSampledTrainNZV$Fatal, ref = "NotFatal")
SmoteTrainNZV$Fatal = relevel(SmoteTrainNZV$Fatal, ref = "NotFatal")
levels(factor(train_NZV$Fatal)) # Check results.

                                                    
# BUILD MODELS 

#####################

set.seed(300)

lrFitNorm = train(Fatal ~.,
                  data = train_NZV,
                  method = "glm",
                  family=binomial(link = logit ),
                  trControl = ctrl,
                  metric = "ROC")

lrFitUp = train(Fatal ~.,
                  data = upSampledTrainNZV,
                  method = "glm",
                  family=binomial(link = logit ),
                  trControl = ctrl,
                  metric = "ROC")

lrFitDown = train(Fatal ~.,
                  data = downSampledTrainNZV,
                  method = "glm",
                  family=binomial(link = logit ),
                  trControl = ctrl,
                  metric = "ROC")

lrFitSmote = train(Fatal ~.,
                  data = SmoteTrainNZV,
                  method = "glm",
                  family=binomial(link = logit ),
                  trControl = ctrl,
                  metric = "ROC")

                                                    
# LOOK AT MODEL RESULTS 

#####################

# Look at normal fitted model.
print(lrFitNorm)                                                
summary(lrFitNorm)  

# Look at uosampled fitted model.
print(lrFitUp)                                                
summary(lrFitUp)

# Look at normal fitted model.
print(lrFitDown)                                                
summary(lrFitDown)

# Look at normal fitted model.
print(lrFitSmote)                                                
summary(lrFitSmote)


# PREDICT - VALIDATION SET 

#####################

# Adding predicted results to evaluation dataframe.
evalResults$LR = predict(lrFitNorm, newdata = validation_NZV, type = "prob")[,1]
evalResults$LRUp = predict(lrFitUp, newdata = validation_NZV, type = "prob")[,1]
evalResults$LRDown = predict(lrFitDown, newdata = validation_NZV, type = "prob")[,1]
evalResults$LRSmote = predict(lrFitSmote, newdata = validation_NZV, type = "prob")[,1]

# Building the ROC curve for the normal data model.
lrROCNorm = roc(evalResults$Fatal, evalResults$LR,
              levels = rev(levels(evalResults$Fatal)))

# Building the ROC curve for the upsampled data model.
lrROCUp = roc(evalResults$Fatal, evalResults$LRUp,
              levels = rev(levels(evalResults$Fatal)))

# Building the ROC curve for the downsampled data model.
lrROCDown = roc(evalResults$Fatal, evalResults$LRDown,
                levels = rev(levels(evalResults$Fatal)))

# Building the ROC curve for the SMOTE data model.
lrROCSmote = roc(evalResults$Fatal, evalResults$LRSmote,
                 levels = rev(levels(evalResults$Fatal)))
                                                    

# Checking results for normal predictions.
lrROCNorm
predictions_LN = predict(lrFitNorm, validation_NZV)
confusionMatrix(predictions_LN, validation_NZV$Fatal)                                                    

# Correct predictions = 70.37815%
((288+47)/476)*100
                                                    
# Checking results for upsample predictions.
lrROCUp
predictions_LU = predict(lrFitUp, validation_NZV)
confusionMatrix(predictions_LU, validation_NZV$Fatal) 

# Correct predictions = 64.91597%
((217+92)/476)*100

# Checking results for downsample predictions.
lrROCDown
predictions_LD = predict(lrFitDown, validation_NZV)
confusionMatrix(predictions_LD, validation_NZV$Fatal)

# Correct predictions = 65.54622%
((218+94)/476)*100

# Checking results for SMOTE predictions.
lrROCSmote
predictions_LS = predict(lrFitSmote, validation_NZV)
confusionMatrix(predictions_LS, validation_NZV$Fatal)

# Correct predictions = 63.86555%
((218+86)/476)*100

# Plotting results.
plot(lrROCNorm, legacy.axes = TRUE, main="LR ROC curve: Sampling methods - validation", 
     col = "red")

lines(lrROCUp, col="green")
lines(lrROCDown, col="blue")
lines(lrROCSmote, col="purple")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green","blue","purple"), 
  legend = c("Normal", "Upsample", "Downsample", "SMOTE")
)

###### Results show... 
# Normal model AUC: 0.6916
# Upsample model AUC: 0.6956
# Downsample model AUC: 0.6933
# SMOTE model AUC: 0.6474


# PREDICT - TEST SET 

#####################

# Adding predicted results to test dataframe.
testResults$LR = predict(lrFitNorm, newdata = test_NZV, type = "prob")[,1]
testResults$LRUp = predict(lrFitUp, newdata = test_NZV, type = "prob")[,1]
testResults$LRDown = predict(lrFitDown, newdata = test_NZV, type = "prob")[,1]
testResults$LRSmote = predict(lrFitSmote, newdata = test_NZV, type = "prob")[,1]

# Building the ROC curve for the normal data model.
lrROCNormT = roc(testResults$Fatal, testResults$LR,
              levels = rev(levels(testResults$Fatal)))

# Building the ROC curve for the upsampled data model.
lrROCUpT = roc(testResults$Fatal, testResults$LRUp,
              levels = rev(levels(testResults$Fatal)))

# Building the ROC curve for the downsampled data model.
lrROCDownT = roc(testResults$Fatal, testResults$LRDown,
                levels = rev(levels(testResults$Fatal)))

# Building the ROC curve for the SMOTE data model.
lrROCSmoteT = roc(testResults$Fatal, testResults$LRSmote,
                 levels = rev(levels(testResults$Fatal)))
                                                    

# Checking results for normal predictions.
lrROCNormT
predictions_LNT = predict(lrFitNorm, test_NZV)
confusionMatrix(predictions_LNT, test_NZV$Fatal)                                                    

# Correct predictions = 69.89247%
((432+88)/744)*100
                                                    
# Checking results for upsample predictions.
lrROCUpT
predictions_LUT = predict(lrFitUp, test_NZV)
confusionMatrix(predictions_LUT, test_NZV$Fatal) 

# Correct predictions = 71.50538%
((357+175)/744)*100

# Checking results for downsample predictions.
lrROCDownT
predictions_LDT = predict(lrFitDown, test_NZV)
confusionMatrix(predictions_LDT, test_NZV$Fatal)

# Correct predictions = 70.69892%
((345+181)/744)*100

# Checking results for SMOTE predictions.
lrROCSmoteT
predictions_LST = predict(lrFitSmote, test_NZV)
confusionMatrix(predictions_LST, test_NZV$Fatal)

# Correct predictions = 63.97849%
((333+143)/744)*100

# Plotting results.
plot(lrROCNormT, legacy.axes = TRUE, main="LR ROC curve: Sampling methods - test", 
     col = "red")

lines(lrROCUpT, col="green")
lines(lrROCDownT, col="blue")
lines(lrROCSmoteT, col="purple")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green","blue","purple"), 
  legend = c("Normal", "Upsample", "Downsample", "SMOTE")
)

###### Results show... 
# Normal model AUC: 0.7578
# Upsample model AUC: 0.7596
# Downsample model AUC: 0.7527
# SMOTE model AUC: 0.6748


#####################

# LOGISTIC REGRESSION - SIGNIFICANT FEATURES

#####################


# BUILD MODELS 

#####################

set.seed(300)

lrFitUpSF = train(Fatal ~NumberOfOfficers+SA_N+SA_U+SG_U+SAR_0.19+
                    SAR_20.29+SAR_30.39+SAR_40.49+SAR_U+NOS_M+
                    OR_0_0_0_1_0_0+OR_1_0_0_0_0_0+C_LosAngeles+
                    C_NewYork+C_Philadelphia, 
                  data = upSampledTrainNZV,
                  method = "glm",
                  family=binomial(link = logit ),
                  trControl = ctrl,
                  metric = "ROC")

# Look at updated significant features fitted model.
print(lrFitUpSF)                                                
summary(lrFitUpSF) # Results show decrease from original upsampled model.
                   
 
set.seed(300)

# Build a second model removing only the lowest three significant features
# from the original upsample model.
lrFitUpSF2 = train(Fatal ~ . - SA_N.A - Y_2015 - SR_B,
                   data = upSampledTrainNZV,
                   method = "glm",
                   family=binomial(link = logit ),
                   trControl = ctrl,
                   metric = "ROC")

# Look at updated significant features fitted model.
print(lrFitUpSF2)                                                
summary(lrFitUpSF2) # Results show increase from original upsampled model.

set.seed(300)

# Build a third model exluding two more features - the
# lowest two from the second model.
lrFitUpSF3 = train(Fatal ~ . - SA_N.A - Y_2015 - SR_B - OR_0_1_0_0_0_0 
                   - SR_U, data = upSampledTrainNZV,
                   method = "glm",
                   family=binomial(link = logit ),
                   trControl = ctrl,
                   metric = "ROC")

# Look at updated significant features fitted model.
print(lrFitUpSF3)                                                
summary(lrFitUpSF3) # Results show increase to model 2.

set.seed(300)

# Build a forth model exluding two more features - the
# lowest two from the third model.
lrFitUpSF4 = train(Fatal ~ . - SA_N.A - Y_2015 - SR_B - OR_0_1_0_0_0_0 
                   - SR_U - Y_2013 - OG_0_1_0, 
                   data = upSampledTrainNZV,
                   method = "glm",
                   family=binomial(link = logit ),
                   trControl = ctrl,
                   metric = "ROC")

# Look at updated significant features fitted model.
print(lrFitUpSF4)                                                
summary(lrFitUpSF4) # Results show increase to model 3.

set.seed(300)

# Build a fifth model exluding two more features - the
# lowest two from the forth model.
lrFitUpSF5 = train(Fatal ~ . - SA_N.A - Y_2015 - SR_B - OR_0_1_0_0_0_0 
                   - SR_U - Y_2013 - OG_0_1_0 - Y_2016 - Y_2014, 
                   data = upSampledTrainNZV,
                   method = "glm",
                   family=binomial(link = logit ),
                   trControl = ctrl,
                   metric = "ROC")

# Look at updated significant features fitted model.
print(lrFitUpSF5)                                                
summary(lrFitUpSF5) # Results show increase to model 4.

# Build a sixth model exluding two more features - the
# lowest two from the fifth model.
lrFitUpSF6 = train(Fatal ~ . - SA_N.A - Y_2015 - SR_B - OR_0_1_0_0_0_0 
                   - SR_U - Y_2013 - OG_0_1_0 - Y_2016 - Y_2014 
                   - NOS_N.A - NOS_S, 
                   data = upSampledTrainNZV,
                   method = "glm",
                   family=binomial(link = logit ),
                   trControl = ctrl,
                   metric = "ROC")

# Look at updated significant features fitted model.
print(lrFitUpSF6)                                                
summary(lrFitUpSF6) # Results show increase to model 5.

# Build a seventh model exluding three more features - the
# lowest two from the sixth model.
lrFitUpSF7 = train(Fatal ~ . - SA_N.A - Y_2015 - SR_B - OR_0_1_0_0_0_0 
                   - SR_U - Y_2013 - OG_0_1_0 - Y_2016 - Y_2014 
                   - NOS_N.A - NOS_S - SG_M - Y_2010, 
                   data = upSampledTrainNZV,
                   method = "glm",
                   family=binomial(link = logit ),
                   trControl = ctrl,
                   metric = "ROC")

# Look at updated significant features fitted model.
print(lrFitUpSF7)                                                
summary(lrFitUpSF7) # Results show a decrease to model 6.


# PREDICT - VALIDATION SET 

#####################

# Predicting using lrFitUpSF4, model with best ROC result.

# Adding predicted results to evaluation dataframe.
evalResults$LRSF6 = predict(lrFitUpSF6, newdata = validation_NZV, type = "prob")[,1]

# Building the ROC curve for the upsampled data model.
lrROCSF6 = roc(evalResults$Fatal, evalResults$LRSF6,
              levels = rev(levels(evalResults$Fatal)))

# Checking results for upsample predictions.
lrROCSF6
predictions_LSF6 = predict(lrFitUpSF6, validation_NZV)
confusionMatrix(predictions_LSF6, validation_NZV$Fatal) 

# Correct predictions = 64.07563%
((214+91)/476)*100

# Plotting results.
plot(lrROCUp, legacy.axes = TRUE, main="LR ROC curve: Significant features - validation", 
     col = "red")

lines(lrROCSF6, col="green")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green"), 
  legend = c("All features", "Significant features")
)

###### Results show... 
# All features model AUC: 0.6956
# Significant features model AUC: 0.695


# PREDICT - TEST SET 

#####################

# Adding predicted results to test dataframe.
testResults$LRSF6 = predict(lrFitUpSF6, newdata = test_NZV, type = "prob")[,1]

# Building the ROC curve for the upsampled data model.
lrROCSF6T = roc(testResults$Fatal, testResults$LRSF6,
              levels = rev(levels(testResults$Fatal)))

# Checking results for upsample predictions.
lrROCSF6T
predictions_LSF6T = predict(lrFitUpSF6, test_NZV)
confusionMatrix(predictions_LSF6T, test_NZV$Fatal) 

# Correct predictions = 71.63978%
((358+175)/744)*100

# Plotting results.
plot(lrROCUpT, legacy.axes = TRUE, main="LR ROC curve: Significant features - test", 
     col = "red")

lines(lrROCSF6T, col="green")

legend(
  "bottomright", 
  lty=c(1,1), 
  col=c("red","green"), 
  legend = c("All features", "Significant features")
)

###### Results show... 
# All features model AUC: 0.7596
# Significant features model AUC: 0.7587

# Saving best performing model
save(lrFitNorm, file = "LR_best.rda")


#####################

# NEURAL NETWORKS

#####################


# BUILD MODEL

#####################

nnetGrid = expand.grid(.size = 1:10,
                       .decay = c(0, .1, 1, 2))

maxSize = max(nnetGrid$.size)

numWts = 1*(maxSize * (length(upSampledTrainNZV[,-36]) +1) + maxSize + 1)

set.seed(400)

nnetFit = train(x = upSampledTrainNZV[,-36],
                y = upSampledTrainNZV$Fatal,
                method = "nnet",
                metric = "ROC",
                tuneGrid = nnetGrid,
                trace = FALSE,
                maxit = 2000,
                MaxNWts = numWts,
                trControl = ctrl)

# Look at neural networks fitted model.
print(nnetFit)     

# Using optimal model from above tuning.
nnetMod = nnet(Fatal ~., data = upSampledTrainNZV, size = 10, decay = 0)

source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

# Visualise nnet model
plot.nnet(nnetMod)

# Saving best performing model
save(nnetMod, file = "NN_best.rda")


# PREDICT - VALIDATION SET 

#####################

pred.nnet = predict(nnetMod,validation_NZV,type=("class"))
## MisClassification Confusion Matrix
table(validation_NZV$Fatal,pred.nnet)

# Correct predictions = 80.88235%
((267+118)/476)*100

# PREDICT - TEST SET 

#####################

test.nnet = predict(nnetMod,test_NZV,type=("class"))
## MisClassification Confusion Matrix
table(test_NZV$Fatal,test.nnet)

# Correct predictions = 65.5914%
((138+350)/744)*100


#####################

# KNN

#####################


# BUILD MODEL

#####################

set.seed(500)

# Fit KNN model using upsample train.
knnFit = train(upSampledTrainNZV[,-36], upSampledTrainNZV$Fatal, method = "knn", 
               metric = "ROC", tuneGrid = data.frame(.k = c(4*(0.5)+1,
                                                            20*(1:5)+1,
                                                            50*(2:9)+1)),
               trControl = ctrl)

knnFit # Look at model results. K = 3 produced best model.

# Compute the confusion matrix of the fitted model.
knnCM = confusionMatrix(knnFit, norm = "none")
knnCM

# Plot fitted results.
plot(knnFit, main="KNN - ROC vs Neighbours - Upsampled")


# PREDICT - VALIDATION SET 

#####################

knn_pred = knn(upSampledTrainNZV[,-36], validation_NZV[,-1], 
               upSampledTrainNZV$Fatal, k = 3)
prop.table(table(knn_pred, validation_NZV$Fatal))

# Correct predictions = 74.15966%
(0.48949580 + 0.25210084)*100


# PREDICT - TEST SET 

#####################

knn_test = knn(upSampledTrainNZV[,-36], test_NZV[,-1], 
               upSampledTrainNZV$Fatal, k = 3)
prop.table(table(knn_test, test_NZV$Fatal))

# Correct predictions = 61.55914%
(0.3897849 + 0.2258065)*100

# Saving best performing model
save(knnFit, file = "KNN_best.rda")





