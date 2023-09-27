#-------------------------------#
#Group 21 - Car price prediction
#-------------------------------#

#set-up
library(dplyr)
library(stringr)
library(ggplot2)
library(caret)
library(corrplot)
library(randomForest)
library(gbm)
library(e1071)
library(reshape2)
library(xgboost)
library(Metrics)

#import dataset
t <- file.choose()
car <- read.csv(t, header=T)
data <- car
head(car)
summary(car) #found some outliers in car price (Price = 1)

#remove duplicate column
duplicate <- car[duplicated(car$ID), ]
car_distinct <- car[!duplicated(car), ]
car_distinct <- car_distinct[,-1]

#detect, view and remove unusual value in Manufacturer and Model column
non_ascii <- grepl("[^\x01-\x7F]", car_distinct)
non_ascii_manufacturer <- subset(car_distinct, grepl("[^\x01-\x7F]", car_distinct$Manufacturer))
non_ascii_model <- subset(car_distinct, grepl("[^\x01-\x7F]", car_distinct$Model))

car_distinct <- car_distinct %>% mutate(Model = str_replace_all(Model, "[^a-zA-Z0-9 ]", "")) #remove unusual symbol in Model column
car_distinct <- car_distinct %>% mutate(Manufacturer = str_replace_all(Manufacturer, "[^a-zA-Z0-9 ]", "")) ##remove unusual symbol in Manufacturer column

#clean car_distinct$Doors, Wheel and Levy
car_distinct$Doors <- gsub('04-May', '2', car_distinct$Doors)
car_distinct$Doors <- gsub('02-Mar', '1', car_distinct$Doors)
car_distinct$Doors <- gsub('>5', '3', car_distinct$Doors)
car_distinct$Wheel <- gsub('Left wheel', '1', car_distinct$Wheel)
car_distinct$Wheel <- gsub('Right-hand drive', '0', car_distinct$Wheel)
car_distinct$Levy <- as.numeric(gsub("-", "0", car_distinct$Levy))

#reset rows names
rownames(car_distinct) <- NULL

#Mileage, Leather.interior and Engine.volume column
car_distinct$Mileage <- as.numeric(sub(" km", "", car_distinct$Mileage)) #remove "km" in Mileage column
car_distinct$Leather.interior <- as.numeric(factor(car_distinct$Leather.interior, levels=c('No', 'Yes'))) - 1 #Yes into 1 and No to 0

car_distinct$Turbo <- ifelse(grepl(" Turbo", car_distinct$Engine.volume), 1, 0) #add Turbo column to know whether the engine has turbo boost or not (1/0)
car_distinct$Engine.volume <- sub(" Turbo", " ",car_distinct$Engine.volume) #subtract the "turbo" part in engine.volume column

#cut down outliers
car_distinct <- car_distinct[car_distinct$Price >= 500,]

# Calculate the IQR for Price
Q1_price <- quantile(car_distinct$Price, 0.25, na.rm = TRUE)
Q3_price <- quantile(car_distinct$Price, 0.75, na.rm = TRUE)
IQR_price <- Q3_price - Q1_price

# Define the upper and lower bounds for Price outliers
upper_bound_price <- Q3_price + 1.5 * IQR_price
lower_bound_price <- Q1_price - 1.5 * IQR_price

# Calculate the IQR for Mileage
Q1_mileage <- quantile(car_distinct$Mileage, 0.25, na.rm = TRUE)
Q3_mileage <- quantile(car_distinct$Mileage, 0.75, na.rm = TRUE)
IQR_mileage <- Q3_mileage - Q1_mileage

# Define the upper and lower bounds for Mileage outliers
upper_bound_mileage <- Q3_mileage + 1.5 * IQR_mileage
lower_bound_mileage <- Q1_mileage - 1.5 * IQR_mileage

# Filter the values where Price and Mileage are within the bounds
car_distinct <- car_distinct[car_distinct$Price >= lower_bound_price & car_distinct$Price <= upper_bound_price &
                               car_distinct$Mileage >= lower_bound_mileage & car_distinct$Mileage <= upper_bound_mileage, ]


#EDA
str(car_distinct)
summary(car_distinct)

#Top 10 most popular manufacter
top_manufacturers <- car_distinct %>%
  group_by(Manufacturer) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  head(10)
ggplot(top_manufacturers, aes(x = reorder(Manufacturer, Count), y = Count)) +
  geom_bar(stat = "identity", fill = "forest green", color = "black") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Top 10 Car Manufacturers",
       x = "Manufacturer",
       y = "Count")

#Car price distribution
ggplot(car_distinct, aes(x = Price))+
  geom_histogram(fill = "forest green", color = "black", bins = 50) + 
  labs(title = "Price distribution",
       y = "Car price") +
  geom_vline(xintercept = mean(car_distinct$Price), color = "red", linetype = "dashed")
  

#Category distribution
category_counts <- car_distinct %>%
  group_by(Category) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count))

ggplot(category_counts, aes(x = reorder(Category, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = 'forest green', color = "black") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Distribution of Category",
       x = "Category",
       y = "Count")

#year produced
ggplot(car_distinct, aes(x = Prod..year)) +
  geom_histogram(fill = "forest green", color = "black", bins = 50) +
  labs(title = "Distribution of Production Year",
       x = "Production Year",
       y = "Count") +
  geom_vline(xintercept = mean(car_distinct$Prod..year), 
             color = "red", linetype = "dashed")

#milleage
boxplot(car_distinct$Mileage)
summary(car_distinct$Mileage)

#heatmap among numeric variables
car_distinct_numeric <- car_distinct[, sapply(car_distinct, is.numeric)]
correlation_matrix <- cor(car_distinct_numeric)
corrplot(correlation_matrix, method = "color", type = "upper",
         addCoef.col = "black", # Add correlation coefficient on the plot
         tl.col = "black", tl.srt = 45)  # Text label color and rotation


#convert chr columns into numeric 
for (i in 1:ncol(car_distinct)){
  if (is.character(car_distinct[,i])==TRUE){
    for(j in 1:nrow(car_distinct)){
      asics <- as.numeric(charToRaw(car_distinct[j,i]))
      car_distinct[j,i] <- sum(asics)
    }
  }
  car_distinct[,i] <- as.numeric(car_distinct[,i])
}

#data partitioning sequence 
#creating sequence for data partitioning
set.seed(123)
training_data_percentage <- seq(from =0.1, to =0.9, length.out = 9)
car_distinct$Price <- log(car_distinct$Price)
car_distinct$Mileage <- log(car_distinct$Mileage)

#random forest model
result_rf <- data.frame()

for (t in training_data_percentage){
  indx_Partition <- createDataPartition(car_distinct$Price, p=t, list = FALSE) #proportion
  training_data <- car_distinct[indx_Partition,]
  testing_data <- car_distinct[-indx_Partition,]
  
  #create regression model
  TrainedRegressionModel_rf <- randomForest(Price~ ., data= training_data,tree=2000, nodesize=10, mtry=10)
  
  #predict on testing data
  Predicted_outcomes_rf <- predict(TrainedRegressionModel_rf, newdata = testing_data)
  
  #evaluating model performance
  par(mfrow=c(5,1))
  corln <- cor(Predicted_outcomes_rf, testing_data$Price)
  rmse_value <- rmse(testing_data$Price, Predicted_outcomes_rf)
  
  result_rf <- rbind(result_rf, data.frame(partition_percentage = t, correlation_rf = corln, rf_rmse = rmse_value))
}

#gradient boosting 
result_gbm <- data.frame()

for (t in training_data_percentage){
  indx_Partition <- createDataPartition(car_distinct$Price, p=t, list = FALSE) 
  training_data <- car_distinct[indx_Partition,]
  testing_data <- car_distinct[-indx_Partition,]
  
  #create regression model
  TrainedRegressionModel <- gbm(Price~ ., data= training_data)
  
  #predict on testing data
  Predicted_outcomes <- predict(TrainedRegressionModel, newdata = testing_data)
  
  #evaluating model performance
  par(mfrow=c(5,1))
  corln <- cor(Predicted_outcomes, testing_data$Price)
  rmse_value <- rmse(testing_data$Price, Predicted_outcomes)
  
  result_gbm <- rbind(result_gbm, data.frame(partition_percentage = t, correlation_gbm = corln, gbm_rmse = rmse_value))
}

#support vector regression 
result_svm <- data.frame()

for (t in training_data_percentage){
  indx_Partition <- createDataPartition(car_distinct$Price, p=t, list = FALSE) #proportion
  training_data <- car_distinct[indx_Partition,]
  testing_data <- car_distinct[-indx_Partition,]
  
  #create regression model
  TrainedRegressionModel <- svm(Price~ ., data= training_data)
  
  #predict on testing data
  Predicted_outcomes <- predict(TrainedRegressionModel, newdata = testing_data)
  
  #evaluating model performance
  par(mfrow=c(5,1))
  corln <- cor(Predicted_outcomes, testing_data$Price)
  rmse_value <- rmse(testing_data$Price, Predicted_outcomes)
  
  result_svm <- rbind(result_svm, data.frame(partition_percentage = t, correlation_svm = corln, svm_rmse = rmse_value))
}

# Xgboost 
# Define the predictors and target variable
predictors <- names(car_distinct)[!names(car_distinct) %in% "Price"]
target <- "Price"

# Create an empty data frame to store the results
result_xgb <- data.frame()

# Loop over the training data percentages
for (t in training_data_percentage) {
  indx_Partition <- createDataPartition(car_distinct$Price, p = t, list = FALSE) #proportion
  training_data <- car_distinct[indx_Partition,]
  testing_data <- car_distinct[-indx_Partition,]
  
  # Create the regression model with xgboost
  TrainedRegressionModel <- xgboost(data = as.matrix(training_data[, predictors]),
                                    label = training_data$Price,
                                    nrounds = 100,
                                    objective = "reg:squarederror")
  
  # Predict on testing data
  Predicted_outcomes <- predict(TrainedRegressionModel, newdata = as.matrix(testing_data[, predictors]))
  
  # Evaluating model performance
  corln <- cor(Predicted_outcomes, testing_data$Price)
  mse <- mean((Predicted_outcomes - testing_data$Price)^2)
  rmse <- sqrt(mse)
  
  # Store the results in the data frame
  result_xgb <- rbind(result_xgb, data.frame(partition_percentage = t, correlation_xgb = corln, xgboost_rmse = rmse))
}

#Merge all the result
Correlation <- data.frame(
  partition_percentage = result_svm$partition_percentage,
  rf_cor = result_rf$correlation_rf,
  gbm_cor = result_gbm$correlation_gbm,
  svm_cor = result_svm$correlation_svm,
  xgb_cor = result_xgb$correlation_xgb
)
long_data <- melt(Correlation, id.vars = "partition_percentage", variable.name = "model", value.name = "correlation")
ggplot(long_data, aes(x = factor(partition_percentage), y = correlation, fill = model)) +
  geom_bar(stat = "identity", position = "dodge",  color = "black") +
  labs(x = "Partition Percentage", y = "Correlation",
       title = "Model Performances", 
       subtitle = "Higher correlation values indicate better performance") +
  theme_minimal()

RMSE <- data.frame(
  partition_percentage = result_gbm$partition_percentage,
  rf_rmse = result_rf$rf_rmse,
  gbm_rmse = result_gbm$gbm_rmse,
  svm_rmse = result_svm$svm_rmse,
  xgb_rmse = result_gbm$gbm_rmse
)
rmse <- melt(RMSE, id.vars = "partition_percentage", variable.name = "model", value.name = "RMSE")
ggplot(rmse, aes(x = factor(partition_percentage), y = RMSE, fill = model)) +
  geom_bar(stat = "identity", position = "dodge",  color = "black") +
  labs(x = "Partition Percentage", y = "RMSE",
       title = "Model Performances", 
       subtitle = "Lower RMSE values indicate better performance") +
  theme_minimal()

#Features importance
varImpPlot(TrainedRegressionModel_rf)
