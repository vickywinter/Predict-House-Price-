#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:11:07 2018

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

def Model(train_linear, test_linear):
    train_linear_fea=train_linear.drop(columns=['SalePrice'])
    train_linear_tar=train_linear.SalePrice
    x_train, x_test, y_train, y_test = train_test_split(train_linear_fea, train_linear_tar,test_size=0.2, random_state=0)
    def evaluate(model, test_features, test_labels,train_features, train_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))    
        print("MSE for train data is: %f" % mean_squared_error(y_train, model.predict(x_train)))
        print("MSE for validation data is: %f" % mean_squared_error(y_test, model.predict(x_test)))
        return accuracy
    real_train_tar=np.expm1(train_linear_tar)
    """
        . Lasso model
    """
    
    lassocv = LassoCV(alphas = np.logspace(-5, 4, 400), )
    lassocv.fit(train_linear_fea, train_linear_tar)
    lassocv_score = lassocv.score(train_linear_fea, train_linear_tar)
    lassocv_alpha = lassocv.alpha_
    print("Best alpha : ", lassocv_alpha, "Score: ",lassocv_score)
    
    start=time.time()
    lasso =Lasso(normalize = True)
    lasso.set_params(alpha=lassocv_alpha,max_iter = 10000)
    lasso.fit(x_train, y_train)
    end=time.time()
    mean_squared_error(y_test, lasso.predict(x_test))
    coef_lasso=pd.Series(lassocv.coef_, index=x_train.columns).sort_values(ascending =False)
    evaluate(lasso,x_test,y_test,x_train,y_train)
    print('Time elapsed: %.4f seconds' % (end-start))
    
    y_lasso_predict=lasso.predict(train_linear_fea)
    x_line = np.arange(700000)
    y_line=x_line
    plt.scatter(real_train_tar,np.expm1(y_lasso_predict))
    plt.plot(x_line, y_line, color='r')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predict Sle Price')
    
    test_prediction_lasso=np.expm1(lasso.predict(test_linear))
    
    
    """
        . Ridge model
    """
    
    ridgecv = RidgeCV(alphas = np.logspace(-5, 4, 400))
    ridgecv.fit(x_train, y_train)
    ridgecv_score = ridgecv.score(x_train, y_train)
    ridgecv_alpha = ridgecv.alpha_
    print("Best alpha : ", ridgecv_alpha, "Score: ",ridgecv_score)
    coef=pd.Series(ridgecv.coef_, index=x_train.columns).sort_values(ascending =False)
    
    start=time.time()
    ridge =Ridge(normalize = True)
    ridge.set_params(alpha=ridgecv_alpha,max_iter = 10000)
    ridge.fit(x_train, y_train)
    end=time.time()
    mean_squared_error(y_test, ridge.predict(x_test))
    coef_ridge=pd.Series(ridgecv.coef_, index=x_train.columns).sort_values(ascending =False)
    evaluate(ridge,x_test,y_test,x_train,y_train)
    print('Time elapsed: %.4f seconds' % (end-start))
    
    y_ridge_predict=ridge.predict(train_linear_fea)
    x_line = np.arange(700000)
    y_line=x_line
    plt.scatter(real_train_tar,np.expm1(y_ridge_predict))
    plt.plot(x_line, y_line, color='r')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predict Sle Price')
    
    test_prediction_ridge=np.expm1(ridge.predict(test_linear))
    
    
    """
        . Random Forest
    """
    #train=train.drop(columns=['DateSold'])
    #test=test.drop(columns=['DateSold'])
    #X_train=train.drop(columns=['SalePrice'])
    #Y_train=train['SalePrice']
    X_train=train_linear_fea
    Y_train=train_linear_tar
    x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(X_train, Y_train,test_size=0.2, random_state=0)
    
    
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    #
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train, Y_train)
    #rf_random.fit(x_train_rf, y_train_rf)
    rf_random.best_params_
    
    #Random search allowed us to narrow down the range for each hyperparameter. Now that we know where to concentrate our search,
    # we can explicitly specify every combination of settings to try. 
    param_grid = {
        'bootstrap': [False],
        'max_depth': [80, 90, 100, 110,120,130],
        'max_features': [2, 3],
        'min_samples_leaf': [1,2,3, 4],
        'min_samples_split': [2,4,6,8, 10, 12],
        'n_estimators': [600,700, 800, 900, 1000]
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
    #grid_search.fit(x_train, y_train)
    grid_search.fit(X_train, Y_train)
    grid_search.best_params_
    
    best_random = grid_search.best_estimator_
    start=time.time()
    best_random.fit(x_train_rf,y_train_rf)
    end=time.time()
    evaluate(best_random, x_test_rf, y_test_rf,x_train_rf,y_train_rf)
    print('Time elapsed: %.4f seconds' % (end-start))
    
    y_rf_predict=best_random.predict(train_linear_fea)
    x_line = np.arange(700000)
    y_line=x_line
    plt.scatter(real_train_tar,np.expm1(y_rf_predict))
    plt.plot(x_line, y_line, color='r')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predict Sle Price')
    importance_rf = pd.DataFrame({'features':train_linear_fea.columns, 'imp':best_random.feature_importances_}).\
                            sort_values('imp',ascending=False)
    
    importance_top20_rf = importance_rf.iloc[:20,]
    
    plt.barh(importance_top20_rf.features, importance_top20_rf.imp)
    plt.xlabel('Feature Importance')
    
    test_prediction_rf=np.expm1(best_random.predict(test_linear))
    
    """
        . Xgboost
    """
    
    learning_rate = [round(float(x), 2) for x in np.linspace(start = .1, stop = .2, num = 11)]
        # Minimum for sum of weights for observations in a node
    min_child_weight = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # Maximum nodes in each tree
    max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
    n_estimators=[int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
    subsample=[0.3, 0.4,0.5,0.6, 0.7]
    model = xgb.XGBRegressor()
    random_grid = {'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'min_child_weight': min_child_weight,
                    'subsample': subsample,
                    'n_estimators':n_estimators
                    }
    
        # Make a RandomizedSearchCV object with correct model and specified hyperparams
    xgb_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=1000, cv=5, verbose=2, random_state=42, n_jobs=-1)
    start = time.time()
        # Fit models
    xgb_random.fit(X_train, Y_train)
    xgb_random.best_params_
    
    
    """
    best_params_={'learning_rate': 0.1,
     'max_depth': 2,
     'min_child_weight': 4,
     'n_estimators': 900,
     'subsample': 0.5}
    """
    model_xgb = XGBRegressor(**xgb_random.best_params_)
    #model_xgb = XGBRegressor(**best_params_)
    start=time.time()
    model_xgb.fit(x_train_rf,y_train_rf)
    end=time.time()
    evaluate(model_xgb, x_test_rf, y_test_rf,x_train_rf,y_train_rf)
    print('Time elapsed: %.4f seconds' % (end-start))
    
    
    
    y_xgb_predict=model_xgb.predict(train_linear_fea)
    x_line = np.arange(700000)
    y_line=x_line
    plt.scatter(real_train_tar,np.expm1(y_xgb_predict))
    plt.plot(x_line, y_line, color='r')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predict Sle Price')
    importance_xgb = pd.DataFrame({'features':train_linear_fea.columns, 'imp':model_xgb.feature_importances_}).\
                            sort_values('imp',ascending=False)
    
    importance_top20_xgb = importance_xgb.iloc[:20,]
    
    plt.barh(importance_top20_xgb.features, importance_top20_xgb.imp)
    plt.xlabel('Feature Importance')
    
    test_prediction_xgb=np.expm1(model_xgb.predict(test_linear))
    
    return(test_prediction_lasso, test_prediction_ridge, test_prediction_rf, test_prediction_xgb,y_lasso_predict, y_ridge_predict, y_rf_predict, y_xgb_predict)