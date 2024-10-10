import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def getdfs():
    '''
    preprocess and return train_df and val_df
    '''
    train_df = pd.read_csv("../datasets/train/train_emoticon.csv")
    val_df = pd.read_csv("../datasets/valid/valid_emoticon.csv")


    def get_char_columns(df):
        for i in range(len(df['input_emoticon'][0])):
            df[f'c_{i+1}'] = df['input_emoticon'].apply(lambda x, _i=i: x[_i])
        
        return df[df.columns.to_list()[2:] + ['label']]

    train_df = get_char_columns(train_df)
    val_df = get_char_columns(val_df)

    return train_df, val_df

def one_hot_encode(train_df, val_df):
    '''
    one hot encode the character columns
    '''
    y_train = train_df['label']
    y_val = val_df['label']
    train_df.drop('label', axis=1,inplace=True)
    val_df.drop('label', axis = 1, inplace = True)
    oh_encoder = OneHotEncoder(handle_unknown = 'ignore')
    oh_encoder.fit(train_df)
    train_df = pd.DataFrame(oh_encoder.transform(train_df).toarray())
    val_df = pd.DataFrame(oh_encoder.transform(val_df).toarray())
    return train_df, val_df, y_train, y_val