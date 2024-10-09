from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

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
