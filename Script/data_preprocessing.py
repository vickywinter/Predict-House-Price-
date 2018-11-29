#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:53:53 2018

@author: vickywinter
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import sklearn.model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from datetime import date
import time


test=pd.read_csv('/Users/vickywinter/Documents/NYC/Machine Learning Proj/Data/all/test.csv')
train=pd.read_csv('/Users/vickywinter/Documents/NYC/Machine Learning Proj/Data/all/train.csv')
all_data=pd.concat([train,test])
all_data=all_data.drop(columns=['Id'])

id_num=len(train)

def data_engineering(all_data,id_num):
    all_data['Bath']=all_data['FullBath']+all_data['HalfBath']*0.25
    all_data['BsmtBath']=all_data['BsmtFullBath']+all_data['BsmtHalfBath']*0.25
    all_data=all_data.drop(columns=['FullBath', 'HalfBath','BsmtFullBath','BsmtHalfBath'])
    all_data["DateSold"]=all_data.apply(lambda row: date(row['YrSold'], row['MoSold'],1), axis=1)
    all_data=all_data.drop(columns=['YrSold', 'MoSold'])
    
    """
       2. Cleaning high collinearity variables
    """
    train=all_data[all_data["SalePrice"]>0]
    test=all_data[all_data["SalePrice"].isnull()]
    corrmat = train.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True);
    
    corr=train.corr().abs()
    s = corr.unstack()
    so = s.sort_values(kind="quicksort")
    all_data=pd.concat([train,test])
    
    all_data=all_data.drop(columns=['GarageArea'])
    all_data['HasGarage']=np.where(all_data['GarageYrBlt']>0,1,0)
    all_data=all_data.drop(columns=['GarageYrBlt'])
    all_data=all_data.drop(columns=['TotalBsmtSF'])

    """
       3. Cleaning missing value
    """
    missing_value=pd.DataFrame({'miss':all_data.isnull().sum(),'ratio':(all_data.isnull().sum() / len(all_data)) * 100})
    missing_value=missing_value.sort_values(by=['miss'],ascending=False)
    # missing_value shows the number and percentage of missing value for each variables
    
    for var in ("MasVnrType",'BsmtFinType1','BsmtFinType2','MasVnrType','BsmtFinType1','BsmtFinType2','BsmtQual','BsmtExposure','BsmtCond','GarageType','GarageQual','GarageCond','GarageFinish','Fence','FireplaceQu','Fence','Alley','MiscFeature','PoolQC'):
        all_data[var]=all_data[var].fillna("None")
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    
    for var in ("MasVnrArea","BsmtBath",'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','Utilities'):
        all_data[var]=all_data[var].fillna(0)
    
    for var in ('MSZoning','KitchenQual',"Exterior1st","Exterior2nd",'SaleType','Electrical'):
        all_data[var] = all_data[var].fillna(all_data[var].mode()[0])
    
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    
    all_data["GarageCars"] = all_data["GarageCars"].fillna(lambda row: 0 if row['GarageType']=='None' else row["GarageCars"])
    all_data["GarageCars"] = all_data["GarageCars"].convert_objects(convert_numeric=True)
    
    missing_value=pd.DataFrame({'miss':all_data.isnull().sum(),'ratio':(all_data.isnull().sum() / len(all_data)) * 100})
    missing_value=missing_value.sort_values(by=['miss'],ascending=False)
    all_data["GarageCars"] = all_data["GarageCars"].fillna(0)
    
    train=all_data[all_data["SalePrice"]>0]
    test=all_data[all_data["SalePrice"].isnull()]
    
    """
       4.Cleaning outlier
    """
    corrmat = train.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True);
    
    corr=train.corr().abs()
    s = corr.unstack()
    so = s.sort_values(kind="quicksort")
    
    k = 10 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    
    # Find outlier use most correlated variables and delete them 
    sns.set()
    cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF','Bath', 'TotRmsAbvGrd','YearBuilt','YearRemodAdd','MasVnrArea']
    sns.pairplot(train[cols], size = 2.5)
    plt.show();
    
    plt.scatter(train['GrLivArea'],train['SalePrice'])
    plt.xlabel('Ground Living Area')
    plt.ylabel('Sale Price')
    
    plt.scatter(train['OverallQual'],train['SalePrice'])
    plt.xlabel('Overall Quality')
    plt.ylabel('Sale Price')
    
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
    sns.set()
    cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF','Bath', 'TotRmsAbvGrd','YearBuilt','YearRemodAdd','MasVnrArea']
    sns.pairplot(train[cols], size = 2.5)
    plt.show();
    
    all_data=pd.concat([train,test])
    return(all_data,train,test)
    
def encode_log(all_data):
    
    """
        Encode categorical data and create dummy data
    """
    all_data=all_data.drop(columns=['DateSold'])
    all_data = pd.get_dummies(all_data, dummy_na=True)
    all_data_linear=all_data
    train_linear=all_data_linear[all_data_linear["SalePrice"]>0]
    test_linear=all_data_linear[all_data_linear["SalePrice"].isnull()]
    train=all_data[all_data_linear["SalePrice"]>0]
    test=all_data[all_data_linear["SalePrice"].isnull()]
    
    """
       Data transfer and preparation
    """
    sns.distplot(train_linear['SalePrice'],fit=norm)
    fig=plt.figure()
    stats.probplot(np.sqrt(train_linear['SalePrice']), plot=plt)
    
    all_data_linear["SalePrice"] = np.log1p(all_data_linear["SalePrice"])
    numeric_feats = all_data_linear.dtypes[all_data_linear.dtypes != "object"].index
    skewed=all_data[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
    
    #change to log if right skew and power if left skew
    all_data_linear["GrLivArea"] = np.log1p(all_data_linear["GrLivArea"])
    all_data_linear["LotArea"] = np.log1p(all_data_linear["LotArea"])
    
    all_data_linear["MiscVal"] = np.log1p(all_data_linear["MiscVal"])
    all_data_linear["BsmtFinSF2"] = np.log1p(all_data_linear["BsmtFinSF2"])
    all_data_linear["EnclosedPorch"] = np.log1p(all_data_linear["EnclosedPorch"])
    all_data_linear["MasVnrArea"] = np.log1p(all_data_linear["MasVnrArea"])
    
    all_data_linear["OpenPorchSF"] = np.log1p(all_data_linear["OpenPorchSF"])
    all_data_linear["WoodDeckSF"] = np.log1p(all_data_linear["WoodDeckSF"])
    all_data_linear["1stFlrSF"] = np.log1p(all_data_linear["1stFlrSF"])
    all_data_linear["GrLivArea"] = np.log1p(all_data_linear["GrLivArea"])
    all_data_linear["WoodDeckSF"] = np.log1p(all_data_linear["WoodDeckSF"])
    all_data_linear["1stFlrSF"] = np.log1p(all_data_linear["1stFlrSF"])
    
    train_linear=all_data_linear[all_data_linear["SalePrice"]>0]
    test_linear=all_data_linear[all_data_linear["SalePrice"].isnull()]
    test_linear=test_linear.drop(columns=['SalePrice'])