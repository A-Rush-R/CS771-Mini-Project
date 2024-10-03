from xgboost import XGBClassifier

def learnAndMakePredictions(X_train, y_train, X_val, y_val) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    # X_train = X_train.reshape(X_train.shape[0], -1)
    # X_val = X_val.reshape(X_val.shape[0], -1)
    
    print(X_train.shape, y_train.shape)
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_val)

    return y_val, y_pred