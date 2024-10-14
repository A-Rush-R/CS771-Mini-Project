from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from param_grid import param_grids

def predict_xgboost(x_train, y_train, x_val, use_label_encoder=False, eval_metric='logloss') :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    
    xgb_model = XGBClassifier(use_label_encoder=use_label_encoder, eval_metric=eval_metric)
    xgb_model.fit(x_train, y_train)
    
    y_pred = xgb_model.predict(x_val)

    return y_pred


def predict_logistic_regression(x_train, y_train, x_val, max_iter=1000, random_state=42) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    
    lr_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    lr_model.fit(x_train, y_train)
    
    y_pred = lr_model.predict(x_val)

    return y_pred

def predict_random_forest(x_train, y_train, x_val, n_estimators=100, random_state=42, min_samples_leaf=1, max_features='sqrt') :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, min_samples_leaf=min_samples_leaf, max_features=max_features)
    rf_model.fit(x_train, y_train)
    
    y_pred = rf_model.predict(x_val)

    return y_pred

def predict_mlp(x_train, y_train, x_val, max_iter=1000, random_state=42, hidden_layer_sizes=(100,50)) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    
    mlp_model = MLPClassifier(max_iter=max_iter, random_state=random_state, hidden_layer_sizes=hidden_layer_sizes)
    mlp_model.fit(x_train, y_train)
    
    y_pred = mlp_model.predict(x_val)

    return y_pred

def predict_svc(x_train, y_train, x_val, random_state=42, kernel='rbf', c=1.0, gamma='scale', max_iter=1000) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    
    svm_model = SVC(random_state=random_state, kernel=kernel, C=c, gamma=gamma, max_iter=max_iter)
    svm_model.fit(x_train, y_train)
    
    y_pred = svm_model.predict(x_val)

    return y_pred

def predict_mnb(x_train, y_train, x_val) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    
    mnb_model = MultinomialNB()
    mnb_model.fit(x_train, y_train)
    
    y_pred = mnb_model.predict(x_val)

    return y_pred

def grid_search_(model, X_train, y_train, param_grid = None,model_name : str = None, cv = 5, scoring = 'accuracy') :
    '''
    inputs : model, associated parameter grid, cross validation, scoring metric, seed
    outputs : best parameters and the associated score (accuracy by default)
    '''
    if param_grid is None :
        if model_name is None :
            raise ValueError("Either param_grid or model_name should be provided")
        elif model_name not in param_grids :
            raise ValueError("Model name not found in param_grids")
        param_grid = param_grids[model_name]
        
    grid_search = GridSearchCV(model, param_grid = param_grid, cv = cv, scoring = scoring)
    grid_search.fit(X_train, y_train)
    # write code so that it breaks if the param_grid does not match the associated model parameters

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    