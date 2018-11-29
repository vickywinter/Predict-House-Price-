#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:17:21 2018

@author: vickywinter
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import time

def Stacking(real_train_tar):
    predictions_train = pd.DataFrame([np.expm1(y_lasso_predict), np.expm1(y_ridge_predict), np.expm1(y_rf_predict), np.expm1(y_xgb_predict)]).T
    sns.pairplot(predictions_train)
    
    learning_rate = [round(float(x), 2) for x in np.linspace(start = .1, stop = .2, num = 11)]
        # Minimum for sum of weights for observations in a node
    min_child_weight = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # Maximum nodes in each tree
    max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
    n_estimators=[int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
    subsample=[0.3, 0.4,0.5,0.6, 0.7]
    stack_model = xgb.XGBRegressor()
    random_grid = {'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'min_child_weight': min_child_weight,
                    'subsample': subsample,
                    'n_estimators':n_estimators
                    }
    
        # Make a RandomizedSearchCV object with correct model and specified hyperparams
    xgb_stack = RandomizedSearchCV(estimator=stack_model, param_distributions=random_grid, n_iter=1000, cv=5, verbose=2, random_state=42, n_jobs=-1)
    start = time.time()
        # Fit models
    xgb_stack.fit(predictions_train, real_train_tar)
    xgb_stack.best_params_
    write_pkl(xgb_stack.best_params_, '/Users/vickywinter/Documents/NYC/Machine Learning Proj/Pickle/stack_params.pkl')
    
    model_stacking = XGBRegressor(**xgb_stack.best_params_)
    #model_xgb = XGBRegressor(**best_params_)
    start=time.time()
    model_stacking.fit(predictions_train,real_train_tar)
    end=time.time()
    print("MSE for train data is: %f" % mean_squared_error(np.log1p(real_train_tar),np.log1p( model_stacking.predict(predictions_train))))
    print('Time elapsed: %.4f seconds' % (end-start))
    
    
    y_stack_predict=model_stacking.predict(predictions_train)
    x_line = np.arange(700000)
    y_line=x_line
    plt.scatter(real_train_tar,y_stack_predict)
    plt.plot(x_line, y_line, color='r')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predict Sle Price')