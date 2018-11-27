#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:04:20 2018

@author: vickywinter
"""

def diagram(train):
    feats_continu=['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch', 'GrLivArea',  \
                   'LotArea', 'LotFrontage', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd']
    feats_dis=[ 'Alley', 'BedroomAbvGr', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',  'MSSubClass','Bath','BsmtBath',\
               'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'Electrical', \
               'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation', \
               'Functional', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC', \
               'HouseStyle', 'KitchenAbvGr', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MSZoning', \
               'MasVnrType', 'MiscFeature', 'Neighborhood', 'OverallCond', 'OverallQual', 'PavedDrive', 'PoolQC', 'RoofMatl', \
               'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'TotRmsAbvGrd', 'Utilities','DateSold']
    #function to plot 
    """ plot diagram in group
    Parameter
    frame: data
    fests: features to plot with SalePrice
    kind: scatter,boxplot, histogrm plot
    cols: number of column for the plot
    
    return : none
    """
    def plotfeats(frame,feats,kind,cols=4):
        rows = int(np.ceil((len(feats))/cols))
        if rows==1 and len(feats)<cols:
            cols = len(feats)
        if kind == 'hs': #hs:hist and scatter
            fig, axes = plt.subplots(nrows=rows*2,ncols=cols,figsize=(cols*5,rows*10))
        else:
            fig, axes = plt.subplots(nrows=rows,ncols=cols,figsize=(cols*5,rows*5))
            if rows==1 and cols==1:
                axes = np.array([axes])
            axes = axes.reshape(rows,cols) # 当 rows=1 时，axes.shape:(cols,)，需要reshape一下
        i=0
        for f in feats:
            #print(int(i/cols),i%cols)
            if kind == 'hist':
                frame.plot.hist(y=f,bins=100,ax=axes[int(i/cols),i%cols])
            elif kind == 'scatter':
                frame.plot.scatter(x=f,y='SalePrice',ylim=(0,800000), ax=axes[int(i/cols),i%cols])
            elif kind == 'hs':
                frame.plot.hist(y=f,bins=100,ax=axes[int(i/cols)*2,i%cols])
                frame.plot.scatter(x=f,y='SalePrice',ylim=(0,800000), ax=axes[int(i/cols)*2+1,i%cols])
            elif kind == 'box':
                frame.plot.box(y=f,ax=axes[int(i/cols),i%cols])
            elif kind == 'boxp':
                sns.boxplot(x=f,y='SalePrice', data=frame, ax=axes[int(i/cols),i%cols])
            i += 1
        plt.show()
    
    
    plotfeats(train,feats_continu,kind='scatter',cols=6)
    plotfeats(train,feats_dis,kind='boxp',cols=6)