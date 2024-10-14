param_grids = {
    "xgb" : {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'min_child_weight': [1, 3, 5],
        'eval_metric': ['logloss', 'auc']
    },
    "lr" : {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'max_iter': [100, 500, 1000],
        'fit_intercept': [True, False],
        'random_state': [42]
    },
    "rf" : {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'random_state': [42]
    },
    "mlp" : {
        'hidden_layer_sizes': [(50,50), (100,50), (100,100,50), (200,100)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000, 2000],
        'random_state': [42]
    },
    "svc" : {
        'C': [0.1, 1.0, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
        'max_iter': [1000, 5000],
        'random_state': [42]
    },
    "mnb" : {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'fit_prior': [True, False]
    },
}