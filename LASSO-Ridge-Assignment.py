# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:49:46 2020

@author: Faraz
"""
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
data = pd.read_csv("D:/Faraz/Courses/dataeerEx/Data Science Certification/WEEK-5/Lasso-Ridge Regression/ToyotaCorolla.csv")

data = data.filter(['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'], axis = 1)
data.columns = "Price","Age","KM","HP","cc","Doors","Gears","QuarterlyTax","Weight"

data.describe()
data.columns

# Correlation matrix 
a = data.corr()
a

# EDA
a1 = data.describe()
# Scatter plot and histogram between variables
sns.pairplot(data) 
# weight-QuarterlyTax = high colinearity

# Preparing the model on train data 
model_train = smf.ols("Price ~ Age + KM + HP + cc + Doors + Gears + QuarterlyTax + Weight", data = data).fit()
model_train.summary()
# Prediction
pred = model_train.predict(data)
# Error
resid  = pred - data.Price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
# 1338.25

# To overcome the issues, LASSO and RIDGE regression are used

# LASSO MODEL
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.13, normalize = True)
lasso.fit(data.iloc[:, 1:], data.Price)
# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_
plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(data.columns[1:]))
lasso.alpha
pred_lasso = lasso.predict(data.iloc[:, 1:])
# Adjusted r-square
lasso.score(data.iloc[:, 1:], data.Price)
# 0.86

# RMSE
np.sqrt(np.mean((pred_lasso - data.Price)**2))
# 1338.31

# RIDGE REGRESSION 
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.4, normalize = True)
rm.fit(data.iloc[:, 1:], data.Price)
# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_
plt.bar(height = pd.Series(rm.coef_), x = pd.Series(data.columns[1:]))
rm.alpha
pred_rm = rm.predict(data.iloc[:, 1:])
# Adjusted r-square
rm.score(data.iloc[:, 1:], data.Price)
# 0.83

# RMSE
np.sqrt(np.mean((pred_rm - data.Price)**2))
# 1429.27

# ELASTIC NET REGRESSION 
from sklearn.linear_model import ElasticNet 
enet = ElasticNet(alpha = 0.4)
enet.fit(data.iloc[:, 1:], data.Price) 
# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_
plt.bar(height = pd.Series(enet.coef_), x = pd.Series(data.columns[1:]))
enet.alpha
pred_enet = enet.predict(data.iloc[:, 1:])
# Adjusted r-square
enet.score(data.iloc[:, 1:], data.Price)
# 0.86

# RMSE
np.sqrt(np.mean((pred_enet - data.Price)**2))
# 1341.42

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(data.iloc[:, 1:], data.Price)
lasso_reg.best_params_
lasso_reg.best_score_
lasso_pred = lasso_reg.predict(data.iloc[:, 1:])
# Adjusted r-square#
lasso_reg.score(data.iloc[:, 1:], data.Price)
# 0.86

# RMSE
np.sqrt(np.mean((lasso_pred - data.Price)**2))
# 1342.67

# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(data.iloc[:, 1:], data.Price)
ridge_reg.best_params_
ridge_reg.best_score_
ridge_pred = ridge_reg.predict(data.iloc[:, 1:])
# Adjusted r-square#
ridge_reg.score(data.iloc[:, 1:], data.Price)
# 0.86

# RMSE
np.sqrt(np.mean((ridge_pred - data.Price)**2))
# 1338.64

# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
enet = ElasticNet()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(data.iloc[:, 1:], data.Price)
enet_reg.best_params_
enet_reg.best_score_
enet_pred = enet_reg.predict(data.iloc[:, 1:])
# Adjusted r-square
enet_reg.score(data.iloc[:, 1:], data.Price)
# -1801108.34

# RMSE
np.sqrt(np.mean((enet_pred - data.Price)**2))
# 1342.05