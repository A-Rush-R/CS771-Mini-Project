import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Literal

def getdfs(data : str, train_size : float = 1):
    '''
    preprocess and return train_df and val_df
    '''
    if data not in ['text_seq', 'feature' , 'emoticon']:
        raise ValueError("Invalid data type")
    
    train_df = pd.read_csv(f"../datasets/train/train_{data}.{'npz' if data == 'feature' else 'csv'}")
    val_df = pd.read_csv(f"../datasets/valid/valid_{data}.{'npz' if data == 'feature' else 'csv'}")
    
    train_df = train_df[:int(len(train_df)*train_size)]

    def get_char_columns(df):
        for i in range(len(df['input_emoticon'][0])):
            df[f'c_{i+1}'] = df['input_emoticon'].apply(lambda x, _i=i: x[_i])
        
        return df[df.columns.to_list()[2:] + ['label']]

    if data == 'emoticon':
        train_df = get_char_columns(train_df)
        val_df = get_char_columns(val_df)

    return train_df, val_df

def one_hot_encode(train_df, val_df):
    '''
    one hot encode the character columns for emoticons
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