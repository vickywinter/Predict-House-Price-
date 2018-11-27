#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:28:01 2018

@author: vickywinter
"""

def RandomForest(train_linear, test_linear):
    #train=train.drop(columns=['DateSold'])
    #test=test.drop(columns=['DateSold'])
    #X_train=train.drop(columns=['SalePrice'])
    #Y_train=train['SalePrice']
    train_linear_fea=train_linear.drop(columns=['SalePrice'])
    train_linear_tar=train_linear.SalePrice
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
    

def XgBoost(train_linear, test_linear):
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