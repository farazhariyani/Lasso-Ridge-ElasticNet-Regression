# Load the Data set
data <- read.csv(file.choose())

summary(data)
library(dplyr)

data <- data[,c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

# Rename columns
data <- data %>% rename(price = Price, age = Age_08_04, km = KM, hp = HP, cc= cc, doors = Doors, gears = Gears, qtax = Quarterly_Tax, weight = Weight)
attach(data)

library(glmnet)

# x = converting data into matrix without output data
x <- model.matrix(price ~ ., data = data)[,-1]
y <- data$price

#grid-search technique | 10^10 to 10^2
grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

#cross-validation technique
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)

#extracting minimum lambda value
optimumlambda <- cv_fit$lambda.min
y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared
# 0.85

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)

# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)
cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min
y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared
# 0.86

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)