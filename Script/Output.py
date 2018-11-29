#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:46:49 2018

@author: vickywinter
"""

def output():
    predictions_test = pd.DataFrame([test_prediction_lasso, test_prediction_ridge, test_prediction_rf, test_prediction_xgb]).T
    predictions_test.columns = ['0', '1', '2', '3']
    test_prediction_stacking=model_stacking.predict(predictions_test)
    
    test_prediction_lasso=pd.DataFrame(test_prediction_lasso)
    test_prediction_lasso.to_csv('/Users/vickywinter/Documents/NYC/Machine Learning Proj/Output/Lasso.csv', sep=',',index='False')
    
    test_prediction_ridge=pd.DataFrame(test_prediction_ridge)
    test_prediction_ridge.to_csv('/Users/vickywinter/Documents/NYC/Machine Learning Proj/Output/Ridge.csv', sep=',',index='False')
    
    test_prediction_rf=pd.DataFrame(test_prediction_rf)
    test_prediction_rf.to_csv('/Users/vickywinter/Documents/NYC/Machine Learning Proj/Output/RandomForest.csv', sep=',',index='False')
    
    test_prediction_xgb=pd.DataFrame(test_prediction_xgb)
    test_prediction_xgb.to_csv('/Users/vickywinter/Documents/NYC/Machine Learning Proj/Output/XgBoost.csv', sep=',',index='False')
    
    test_prediction_stacking=pd.DataFrame(test_prediction_stacking)
    test_prediction_stacking.to_csv('/Users/vickywinter/Documents/NYC/Machine Learning Proj/Output/Stacking.csv', sep=',',index='False')