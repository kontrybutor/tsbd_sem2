list.of.packages <- c("ggplot2", "class", "lattice", "caret", "gmodels", "rattle")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(caret)
library(ggplot2)
library(lattice)
library(gmodels)
library(class)
library(rattle)

# Read CSV into R
prc <- read.csv(file="/Users/rafal/Documents/Studia/4GL/Prostate_Cancer.csv", header=TRUE, sep=",")
hrt <- read.csv(file="/Users/rafal/Documents/Studia/4GL/heart_tidy.csv", header=FALSE, sep=",")
prc <- prc[-1] # removing the ID variable from dataset, it doesn't provide useful info..

# print(str(prc))  # structure of dataset
summary(prc) # print min, max, median, mean of attributes
summary(hrt)
# print(table(prc$diagnosis_result))  # print type of cancer, just for quick check
hrt$diagnosis <- factor(hrt$V14, levels = c(0,1), labels = c("Absence", "Present"))
prc$diagnosis <- factor(prc$diagnosis_result, levels = c("B", "M"), labels = c("Benign", "Malignant"))  # change B and M into more meaningful values, store it in new a new attribute

# print(str(prc))  # structure of dataset, new column added
# print(table(prc$diagnosis)) # diagnosis is the same as diagnosis_result

normalize <- function(x) # values has different type and range, let's normalize them...
{
    return ((x - min(x)) / (max(x) - min(x)))
}

prc_n <- as.data.frame(lapply(prc[2:9], normalize)) # apply normalize function on data frame (which is containter for vectors)
summary(prc_n)
hrt_n <- as.data.frame(lapply(hrt[1:13], normalize))
summary(hrt_n)
str(hrt_n)
summary(hrt)
# print(summary(prc_n)) # print normalized values


### TRZEBA ZNORMALIZOWAC!!!!!!

set <- prc # it can be any column from analyzed data_set
#set <- prc
##############################                    #################################

##############################         KNN         #################################

intrain <- createDataPartition(y = set$diagnosis, p=0.6, list = FALSE)
training <- set[intrain,]
testing <- set[-intrain,]
# train_control <- trainControl(method="LOOCV") # define training method -> LOOCV
 train_control <- trainControl(method="cv", number=10) # define training method -> Cross Validation
# train_control <- trainControl(method="boot", number=100) # define training method -> Bootstrap

 knn <- train(diagnosis~., data=training, trControl=train_control, method="knn", preProcess = c("range")) # train the model
 
 test_pred <- predict(knn, newdata = testing)
 
 confusionMatrix(test_pred, testing$diagnosis)



####################################### NAIVE BAYES ##############################


 intrain <- createDataPartition(y = prc$diagnosis, p=0.5, list = FALSE)
 training <- prc[intrain,]
 testing <- prc[-intrain,]

 train_control <- trainControl(method="cv", number=10) # define training control
# train_control <- trainControl(method="boot", number=100) 

 naive_bayes <- train(diagnosis~., data=training, trControl=train_control, method="nb", preProcess = c("center", "scale")) # train the model
 
 test_pred <- predict(naive_bayes, newdata = testing)
 
 confusionMatrix(test_pred, testing$diagnosis)
 
# print(model) # summarize results

#####################################    SVM      #######################

set.seed(200)
intrain <- createDataPartition(y = prc$diagnosis, p=0.5, list = FALSE)
training <- prc[intrain,]
testing <- prc[-intrain,]

#trctrl <- trainControl(method="cv", number=10)
trctrl <- trainControl(method="LOOCV")

# trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

svm_Linear <- train(diagnosis~., data=training, method = "svmRadial",   #could be also radial
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)


test_pred <- predict(svm_Linear, newdata = testing)

confusionMatrix(test_pred, testing$diagnosis)

#####################################    DECISION TREES       #######################


intrain <- createDataPartition(y = prc$diagnosis, p=0.3, list = FALSE)
training <- prc[intrain,]
testing <- prc[-intrain,]

train_control <- trainControl(method="cv", number=5) # define training control
#train_control <-  trainControl(method="boot", number=100) 


r_part <- train(diagnosis~., data=training, trControl=train_control, method="rpart", preProcess = c("center", "scale")) # train the model

test_pred <- predict(r_part, newdata = testing)

confusionMatrix(test_pred, testing$diagnosis)

fancyRpartPlot(r_part$finalModel)


