### TO BE RUN FIRST (before running 'predictive_analysis.R')

#####################

# Load packages

#####################

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
library(ROCR)


# Read in shooting data from csv.
shootings_data = 
read.delim("~/A2-MLSDM/code-files/data/Police-shootings-US-Vice-RAW.csv", 
header=TRUE,sep = ',', stringsAsFactors = FALSE)

# Look at data.
#View(shootings_data)


#####################

# INITIAL PRE-PROCCESSING / DATA CLEANING

#####################


# New dataframe for pre-processing. Removing 'NumberOfSubjects' field as just one value.
data = subset(shootings_data, select = -c(NumberOfSubjects))
#View(data)


# 1. OUTCOME FEATURE: FATAL 

#####################

# Checking data types for outcome variable 'Fatal.'
levels(factor(data$Fatal)) 

# Cleaning
data$Fatal[data$Fatal==" F"] = "F"
data$Fatal[data$Fatal==" N"] = "N"

# Changing to 0-1 values (Fatal = 1)
data$Fatal[data$Fatal=="F"] = 1
data$Fatal[data$Fatal=="N"] = 0
# Change U (unknown) to NA
data$Fatal[data$Fatal=="U"] = NA
# Remove NA values.
data = data[!is.na(data$Fatal),]

# Check class type - character change to factor
class(data$Fatal)
data$Fatal = factor(data$Fatal)


# 2. PREDICTOR FEATURE: DATE 

#####################

# Checking data types for 'Date.'
levels(factor(data$Date)) 

# Can see many date types some with missing days & months.
data$Date[!grepl("/", data$Date)]
data$Date[grepl("-", data$Date)]

# Creating new feature 'Year' to just take reference of year.
data$DateYear = data$Date
# Acounting for different year formats (yy or yyyy)
data$DateYear[grep("2010", data$DateYear)] = "2010"
data$DateYear[grep("/10$", data$DateYear)] = "2010"
data$DateYear[grep("2011", data$DateYear)] = "2011"
data$DateYear[grep("/11$", data$DateYear)] = "2011"
data$DateYear[grep("2012", data$DateYear)] = "2012"
data$DateYear[grep("/12$", data$DateYear)] = "2012"
data$DateYear[grep("2013", data$DateYear)] = "2013"
data$DateYear[grep("/13$", data$DateYear)] = "2013"
data$DateYear[grep("2014", data$DateYear)] = "2014"
data$DateYear[grep("/14$", data$DateYear)] = "2014"
data$DateYear[grep("2015", data$DateYear)] = "2015"
data$DateYear[grep("/15$", data$DateYear)] = "2015"
data$DateYear[grep("2016", data$DateYear)] = "2016"
data$DateYear[grep("/16$", data$DateYear)] = "2016"
# Checking Year feature levels
levels(factor(data$DateYear))

# Check class type & change to factor
class(data$DateYear)
data$DateYear = factor(data$DateYear)

# Removing date feature.
data = subset(data, select = -c(Date))


# 3. PREDICTOR FEATURE: SUBJECTARMED 

#####################

# Checking data types for 'SubjectArmed.'
levels(factor(data$SubjectArmed))  

# Cleaning
data$SubjectArmed[data$SubjectArmed=="Y "] = "Y"
data$SubjectArmed[is.na(data$SubjectArmed)] = "N.A"

# Check class type & change to factor
class(data$SubjectArmed)
data$SubjectArmed = factor(data$SubjectArmed)


# 4. PREDICTOR FEATURE: SUBJECTRACE 

#####################

# Checking data types for 'SubjectRace.'
levels(factor(data$SubjectRace))
data$SubjectRace[is.na(data$SubjectRace)] = "N.A"

# Check class type & change to factor
class(data$SubjectRace)
data$SubjectRace = factor(data$SubjectRace)


# 5. PREDICTOR FEATURE: SUBJECTGENDER 

#####################

# Checking data types for 'SubjectGender.'
levels(factor(data$SubjectGender))  
# Checking 'notes' column column for more details of incident & for accuracy. 
# Recoding 'M;U' to 'M' as notes describe one male victim.
data$SubjectGender[data$SubjectGender=="M;U"] = "M"
data$SubjectGender[data$SubjectGender=="N/A"] = "N.A" # Change illegal character '/.'
levels(factor(data$SubjectGender)) # Checking results.

# Check class type & change to factor
class(data$SubjectGender)
data$SubjectGender = factor(data$SubjectGender)


# 6. PREDICTOR FEATURE: SUBJECTAGE 

#####################

# Checking data types for 'SubjectAge.'
levels(factor(data$SubjectAge))

# Can see some values numerical and some in range (0-19). Can regroup these later. 
# 'Juvenile' can fit into the '0-19' category so lets change this.
data$SubjectAge[data$SubjectAge=="Juvenile"] = "0-19"
# 'UNKNOWN' needs to be recoded as 'U.'
data$SubjectAge[data$SubjectAge=="UNKNOWN"] = "U"
# Change '21-23' to 20-29
data$SubjectAge[data$SubjectAge=="21-23"] = "20-29"
levels(factor(data$SubjectAge)) # Checking results.

# Creating new feature 'SubjectAgeRange.'
data$SubjectAgeRange = data$SubjectAge

# For calculations, change all ranges to NA and the feature to numeric.
data$SubjectAge[data$SubjectAge=="0-19"] = "N/A"
data$SubjectAge[data$SubjectAge=="20-29"] = "N/A"
data$SubjectAge[data$SubjectAge=="30-39"] = "N/A"
data$SubjectAge[data$SubjectAge=="40-49"] = "N/A"
data$SubjectAge[data$SubjectAge=="50-59"] = "N/A"
data$SubjectAge[data$SubjectAge=="60-69"] = "N/A"
data$SubjectAge[data$SubjectAge=="U"] = "N/A"
data$SubjectAge[data$SubjectAge=="N/A"] = 0
data$SubjectAge = as.integer(as.character(data$SubjectAge))

# Update numeric values to ranges in 'SubjectAgeRange.'
data[data$SubjectAge>0 & data$SubjectAge<20, "SubjectAgeRange"] = "0-19"
data[data$SubjectAge>19 & data$SubjectAge<30, "SubjectAgeRange"] = "20-29"
data[data$SubjectAge>29 & data$SubjectAge<40, "SubjectAgeRange"] = "30-39"
data[data$SubjectAge>39 & data$SubjectAge<50, "SubjectAgeRange"] = "40-49"
data[data$SubjectAge>49 & data$SubjectAge<60, "SubjectAgeRange"] = "50-59"
data[data$SubjectAge>59 & data$SubjectAge<70, "SubjectAgeRange"] = "60-69"
data[data$SubjectAge>69, "SubjectAgeRange"] = "70-Plus"
levels(factor(data$SubjectAgeRange)) # Checking results.

# Check class type & change to factor
class(data$SubjectAgeRange)
data$SubjectAgeRange = factor(data$SubjectAgeRange)

# Removing SubjectAge feature.
data = subset(data, select = -c(SubjectAge))


# 6. PREDICTOR FEATURE: NUMBEROFSHOTS

#####################

# Checking data types for 'NumberOfShots.'
levels(factor(data$NumberOfShots))
# Changing different 'unknown' values to 'U'
data$NumberOfShots[data$NumberOfShots=="Unknown"] = "U"
data$NumberOfShots[data$NumberOfShots=="not clear"] = "U"
data$NumberOfShots[data$NumberOfShots=="no information"] = "U"
# Checked the 'Notes' column on these cases and no number given so treated as unknown.
data$NumberOfShots[data$NumberOfShots==">/=1"] = "U"
# Checked 'FullNarrative' and 'Notes' field before updating
data$NumberOfShots[data$NumberOfShots=="1*"] = "1"
data$NumberOfShots[data$NumberOfShots=="04-May"] = "1"
data$NumberOfShots[data$NumberOfShots=="03-May"] = "4"
# Removing 'total'
data$NumberOfShots[data$NumberOfShots=="73 total"] = "73"
data$NumberOfShots[data$NumberOfShots=="45 total"] = "45"
data$NumberOfShots[data$NumberOfShots=="45 total "] = "45"
# Change 'Multiple' to 'M.'
data$NumberOfShots[data$NumberOfShots=="Multiple"] = "M"
# Changing ambiguous greater / greater than or equal to values to 'M' (apart from 1)
data$NumberOfShots[data$NumberOfShots==">/=20"] = "M"
data$NumberOfShots[data$NumberOfShots==">/=5"] = "M"
data$NumberOfShots[data$NumberOfShots==">/=12"] = "M"
data$NumberOfShots[data$NumberOfShots==">/=3"] = "M"
data$NumberOfShots[data$NumberOfShots==">/=8"] = "M"
data$NumberOfShots[data$NumberOfShots==">/=2"] = "M"
data$NumberOfShots[data$NumberOfShots==">/=4"] = "M"
data$NumberOfShots[data$NumberOfShots==">1"] = "M"
data$NumberOfShots[data$NumberOfShots==">13"] = "M"
data$NumberOfShots[data$NumberOfShots==">2"] = "M"
data$NumberOfShots[data$NumberOfShots==">3"] = "M"
data$NumberOfShots[data$NumberOfShots==">4"] = "M"

levels(factor(data$NumberOfShots)) # Checking results

# Create new variable for numeric version of number of shots feature.
data$NumberOfShots2 = data$NumberOfShots
# Turn string values to 0.
data$NumberOfShots2[data$NumberOfShots2=="M"] = 0
data$NumberOfShots2[data$NumberOfShots2=="U"] = 0
data$NumberOfShots2[is.na(data$NumberOfShots2)] = 0

# Split feature into multiple by ';' symbol (where each shot of each officer is listed).
data = cSplit(data, "NumberOfShots2", ";")

# All NA values to 0.
data$NumberOfShots2_2[is.na(data$NumberOfShots2_2)] = 0
data$NumberOfShots2_3[is.na(data$NumberOfShots2_3)] = 0
data$NumberOfShots2_4[is.na(data$NumberOfShots2_4)] = 0
data$NumberOfShots2_5[is.na(data$NumberOfShots2_5)] = 0
data$NumberOfShots2_6[is.na(data$NumberOfShots2_6)] = 0
data$NumberOfShots2_7[is.na(data$NumberOfShots2_7)] = 0

# Remove all spaces.
data$NumberOfShots2_1 = sapply(data$NumberOfShots2_1,gsub,pattern=" ",replacement="")
data$NumberOfShots2_2 = sapply(data$NumberOfShots2_2,gsub,pattern=" ",replacement="")
data$NumberOfShots2_3 = sapply(data$NumberOfShots2_3,gsub,pattern=" ",replacement="")
data$NumberOfShots2_4 = sapply(data$NumberOfShots2_4,gsub,pattern=" ",replacement="")
data$NumberOfShots2_5 = sapply(data$NumberOfShots2_5,gsub,pattern=" ",replacement="")
data$NumberOfShots2_6 = sapply(data$NumberOfShots2_6,gsub,pattern=" ",replacement="")
data$NumberOfShots2_7 = sapply(data$NumberOfShots2_7,gsub,pattern=" ",replacement="")

# Change to numeric.
data$NumberOfShots2_1 = as.numeric(as.character(data$NumberOfShots2_1))
data$NumberOfShots2_2 = as.numeric(as.character(data$NumberOfShots2_2))
data$NumberOfShots2_3 = as.numeric(as.character(data$NumberOfShots2_3))
data$NumberOfShots2_4 = as.numeric(as.character(data$NumberOfShots2_4))
data$NumberOfShots2_5 = as.numeric(as.character(data$NumberOfShots2_5))
data$NumberOfShots2_6 = as.numeric(as.character(data$NumberOfShots2_6))
data$NumberOfShots2_7 = as.numeric(as.character(data$NumberOfShots2_7))

# New numeric column from sum of all shots columns.
data$NOS.Num = data$NumberOfShots2_1 + data$NumberOfShots2_2 + data$NumberOfShots2_3 + 
data$NumberOfShots2_4 + data$NumberOfShots2_5 + data$NumberOfShots2_6 + 
data$NumberOfShots2_7

# Change zeros back to NA.
data$NOS.Num[data$NOS.Num==0] = NA

data = subset(data, select = -c(NumberOfShots2_1, NumberOfShots2_2, NumberOfShots2_3,
                                NumberOfShots2_4, NumberOfShots2_5, NumberOfShots2_6,
                                NumberOfShots2_7))

# Recoding number of shots feature to M (multiple), S (single), U & N.A.
data$NumberOfShots[grep(";", data$NumberOfShots)] = "M"
data$NumberOfShots[data$NumberOfShots=="1"] = "S"
data$NumberOfShots[grep("1|2|3|4|5|6|7|8|9", data$NumberOfShots)] = "M"
levels(factor(data$NumberOfShots)) # Check results.
data$NumberOfShots[is.na(data$NumberOfShots)] = "N.A"

# Check class type & change to factor
class(data$NumberOfShots)
data$NumberOfShots = factor(data$NumberOfShots)


# 7. PREDICTOR FEATURE: NUMBEROFOFFICERS

#####################

# Checking data types for 'NumberOfOfficers.'
levels(factor(data$NumberOfOfficers))
# Removing row with zero officers.
data = data[!(data$NumberOfOfficers=="0"),]
# Checked the 'Notes' and 'FullNarrative' columns to update/correct the below case.
# Changing '2 or more' to '1.'
data$NumberOfOfficers[data$NumberOfOfficers=="2 or More"] = "1"
# Checked the number of characters listed in 'OfficerRace' and 'OfficerGender' to 
# update/correct the below cases.
# Changing '>6' to '6'
data$NumberOfOfficers[data$NumberOfOfficers==">6"] = "6"
# Changing '>7' to '7'
data$NumberOfOfficers[data$NumberOfOfficers==">7"] = "7"
# Changing '>3' to '3'
data$NumberOfOfficers[data$NumberOfOfficers==">3"] = "3"
# Changing '>2' to '3'
data$NumberOfOfficers[data$NumberOfOfficers==">2"] = "3"
# Changing '>5' to '5'
data$NumberOfOfficers[data$NumberOfOfficers==">5"] = "5"
# Changing case where '>1' is equal to '1.'
data$NumberOfOfficers[data$NumberOfOfficers==">1" & data$Date=="10/23/2011"] = "1"
# Changing all other cases of '>1' to 2.
data$NumberOfOfficers[data$NumberOfOfficers==">1"] = "2"

# Note enough values to make its own feature, just change to NA.
data$NumberOfOfficers[data$NumberOfOfficers=="U"] = NA

data$NumberOfOfficers = as.numeric(as.character(data$NumberOfOfficers))
levels(factor(data$NumberOfOfficers)) # Checking results.


# 8. PREDICTOR FEATURE: OFFICERRACE

#####################

# Checking data types for 'OfficerRace.'
levels(factor(data$OfficerRace))
# Can seperate values out later. For now lets just correct some values.
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="W/H",replacement="O")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="W/A",replacement="O")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="W/ H",replacement="O")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="W:W",replacement="W;W")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="W:B",replacement="W;B")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="A/PI Unknown",replacement="U")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="Unknown ",replacement="U")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="Unknown",replacement="U")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="Multi-Racial",replacement="O")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="Other",replacement="O")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="A/PI",replacement="A")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="M",replacement="O")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="NA/W",replacement="U")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="m/m",replacement="O")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="H/L",replacement="L")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="AI/AN",replacement="A")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern=": H",replacement=";L")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="A/W",replacement="O")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="H",replacement="L")
data$OfficerRace = sapply(data$OfficerRace,gsub,pattern="I",replacement="O")
data$OfficerRace[data$OfficerRace==""] = "U"

levels(factor(data$OfficerRace)) # Check results.

# Creating new variables to indicate presence of officer race
data$OfficerRaceA = 0
data$OfficerRaceA[grep("A", data$OfficerRace)] = 1
data$OfficerRaceO = 0
data$OfficerRaceO[grep("O", data$OfficerRace)] = 1
data$OfficerRaceW = 0
data$OfficerRaceW[grep("W", data$OfficerRace)] = 1
data$OfficerRaceB = 0
data$OfficerRaceB[grep("B", data$OfficerRace)] = 1
data$OfficerRaceU = 0
data$OfficerRaceU[grep("U", data$OfficerRace)] = 1
data$OfficerRaceL = 0
data$OfficerRaceL[grep("L", data$OfficerRace)] = 1

# Remove original OfficerRace feature.
data = subset(data, select = -c(OfficerRace))
# Merge race features using 0-1 to indicate if an officer of that race was present.
# Eg the result '111111' would mean all race values described above would be present.
data = unite_(data, "OfficerRace", c("OfficerRaceW","OfficerRaceB", "OfficerRaceA", 
"OfficerRaceL", "OfficerRaceO", "OfficerRaceU"))

# Check class type & change to factor
class(data$OfficerRace)
data$OfficerRace = factor(data$OfficerRace)


# 8. PREDICTOR FEATURE: OFFICERGENDER

#####################

# Checking data types for 'OfficerGender.'
levels(factor(data$OfficerGender))

# Fixing data entry errors.
data$OfficerGender = sapply(data$OfficerGender,gsub,pattern="Unknown",replacement="U")
data$OfficerGender = sapply(data$OfficerGender,gsub,pattern="/M",replacement="M")
data$OfficerGender = sapply(data$OfficerGender,gsub,pattern=": M",replacement=";M")
data$OfficerGender = sapply(data$OfficerGender,gsub,pattern="M:",replacement="M;")
data$OfficerGender = sapply(data$OfficerGender,gsub,pattern="Male",replacement="M")
data$OfficerGender = sapply(data$OfficerGender,gsub,pattern="N",replacement="U")
data$OfficerGender = sapply(data$OfficerGender,gsub,pattern="W",replacement="F")

levels(factor(data$OfficerGender)) # Check results.

# Creating new variables to indicate presence of officer gender.
data$OfficerGenderF = 0
data$OfficerGenderF[grep("F", data$OfficerGender)] = 1
data$OfficerGenderM = 0
data$OfficerGenderM[grep("M", data$OfficerGender)] = 1
data$OfficerGenderU = 0
data$OfficerGenderU[grep("U", data$OfficerGender)] = 1

# Remove original OfficerRace feature.
data = subset(data, select = -c(OfficerGender))

# Merge gender features using 0-1 to indicate if an officer of that gender was present.
data = unite_(data, "OfficerGender", c("OfficerGenderF","OfficerGenderM", 
"OfficerGenderU"))

# Check class type & change to factor
class(data$OfficerGender)
data$OfficerGender = factor(data$OfficerGender)


# 9. PREDICTOR FEATURE: CITY

#####################

# Checking data types for 'City.'
levels(factor(data$City))

# Remove all spaces.
data$City = gsub(" ", "", data$City, fixed = TRUE)

levels(factor(data$City)) # Checking results

# Check class type & change to factor
class(data$City)
data$City = factor(data$City)


# 10. PREDICTOR FEATURES: FULL NARRATIVE, NOTES, NATURE OF STOP & DEPARTMENT

#####################

# Drop un-needed features: FullNarrative, Notes & NatureOfStop
data = subset(data, select = -c(FullNarrative, Notes, NatureOfStop, Department))


# 10. CREATE DUMMY VARIABLES

#####################

# Create dummy variables from factors for models that can't handle categorical data.
# The letter codes for 'sep' will help later to easily remove a whole category/feature. 
data = cbind(data, dummy(data$DateYear, sep = "_Y_"))
data = cbind(data, dummy(data$SubjectArmed, sep = "_SA_"))
data = cbind(data, dummy(data$SubjectRace, sep = "_SR_"))
data = cbind(data, dummy(data$SubjectGender, sep = "_SG_"))
data = cbind(data, dummy(data$SubjectAgeRange, sep = "_SAR_"))
data = cbind(data, dummy(data$NumberOfShots, sep = "_NOS_"))
data = cbind(data, dummy(data$OfficerRace, sep = "_OR_"))
data = cbind(data, dummy(data$OfficerGender, sep = "_OG_"))
data = cbind(data, dummy(data$City, sep = "_C_"))

# Clean dummy header names.
names(data) = gsub("data_", "", names(data), fixed = TRUE)


#####################

# SPLIT INTO TRAINING AND TEST DATA

#####################


# Split based on outcome 'Fatal.' Split 80/20 train and test respectively.
set.seed(1)
trainIndex = createDataPartition(data$Fatal, p = .8, 
                                 list = FALSE, 
                                 times = 1)
head(trainIndex)

# Training and test sets.
train = data[trainIndex,]
test  = data[-trainIndex,]


#####################

# PRE-PROCCESSING ON TRAIN

#####################


# PRE-PROCESS: FEATURE TYPES

#####################

str(train) # Check class type of all features.

# Change all dummy vars to factors.
train[,c(11:127)] = train[,c(11:127)] %>% mutate_if(is.integer,as.factor) 
test[,c(11:127)] = test[,c(11:127)] %>% mutate_if(is.integer,as.factor)

str(train) # Check results.


# PRE-PROCESS: IMPUTE MISSING VALUES

#####################

# Features with NA values (numeric).
miss_features = colnames(train)[colSums(is.na(train)) > 0]
miss_features

# Calculate the % of missing values per column.
(sum(is.na(train$NumberOfOfficers))/length(train$NumberOfOfficers))*100
(sum(is.na(train$NOS.Num))/length(train$NOS.Num))*100

# NOS.Num has a very high missing rate, so lets drop this.
train = subset(train, select = -c(NOS.Num))
test = subset(test, select = -c(NOS.Num))

# Now we can impute missing for NumberOfOfficers.
# Get mean of column.
mean = mean(train$NumberOfOfficers, na.rm = TRUE)  
# Replace NAs with mean.
train$NumberOfOfficers[is.na(train$NumberOfOfficers)] = mean
sum(is.na(train)) # Check results.
# Replace test set with mean of train.
test$NumberOfOfficers[is.na(test$NumberOfOfficers)] = mean
sum(is.na(test)) # Check results.


# PRE-PROCESS: SKEWNESS FOR NUMERIC

#####################

skewness(train$NumberOfOfficers)

# Value is above +1 skewness so applying transformation with 'BoxCox.'
NooAreaTrans = BoxCoxTrans(train$NumberOfOfficers)
NooAreaTrans

# Original data...
head(train$NumberOfOfficers)
# After transformation...
predict(NooAreaTrans, head(train$NumberOfOfficers))

# Apply to whole feature (train).
train$NumberOfOfficers = predict(NooAreaTrans, train$NumberOfOfficers)
# Apply transformation of train on test.
test$NumberOfOfficers = predict(NooAreaTrans, test$NumberOfOfficers) 
skewness(train$NumberOfOfficers) # Checking results. Skewness is reduced.


#####################

# WRITE TO CSV

#####################


# Write files to CSV
write.csv(train, file = "data/cleaned_train.csv", row.names = FALSE)
write.csv(test, file = "data/cleaned_test.csv", row.names = FALSE)

