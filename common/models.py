from xgboost import XGBClassifier

def predict_xgboost(x_train, y_train, x_val) :
    '''
    inputs : training data and validation data in dataframe
    outputs : validation predictions and truth values
    '''
    print(x_train.shape, y_train.shape)
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(x_train, y_train)
    
    y_pred = xgb_model.predict(x_val)

    return y_pred