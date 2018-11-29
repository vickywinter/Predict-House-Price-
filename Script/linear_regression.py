#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:19:06 2018

@author: vickywinter
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
import time

def evaluate(model, test_features, test_labels,train_features, train_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))    
    print("MSE for train data is: %f" % mean_squared_error(train_labels, model.predict(train_features)))
    print("MSE for validation data is: %f" % mean_squared_error(test_labels, model.predict(test_features)))
    return accuracy



def write_pkl(my_dict, opt_path):
    """Writes a dictionary to a pickle
    Keyword arguments
    -----------------
    my_dict -- A dictionary
    opt_path -- Name to save the pickle as
    """
    # Save dictionary to pickle
    with open(opt_path, 'wb') as f:
        pickle.dump(my_dict, f)

def Lasso_model(train_linear, test_linear):
    train_linear_fea=train_linear.drop(columns=['SalePrice'])
    train_linear_tar=train_linear.SalePrice
    real_train_tar=np.expm1(train_linear_tar)
    x_train, x_test, y_train, y_test = train_test_split(train_linear_fea, train_linear_tar,test_size=0.2, random_state=0)
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
    write_pkl(lassocv_alpha, '/Users/vickywinter/Documents/NYC/Machine Learning Proj/Pickle/lasso_params.pkl')
    return test_prediction_lasso
    
def Ridge_model(train_linear, test_linear):
    ridgecv = RidgeCV(alphas = np.logspace(-5, 4, 400))
    ridgecv.fit(train_linear_fea, train_linear_tar)
    ridgecv_score = ridgecv.score(train_linear_fea, train_linear_tar)
    ridgecv_alpha = ridgecv.alpha_
    print("Best alpha : ", ridgecv_alpha, "Score: ",ridgecv_score)
    coef=pd.Series(ridgecv.coef_, index=x_train.columns).sort_values(ascending =False)
    
    start=time.time()
    ridge =Ridge(normalize = True)
    ridge.set_params(alpha=ridgecv_alpha,max_iter = 10000)
    #ridge.set_params(alpha=6,max_iter = 10000)
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
    write_pkl(ridgecv_alpha, '/Users/vickywinter/Documents/NYC/Machine Learning Proj/Pickle/ridge_params.pkl')
    return test_prediction_ridge
    
    