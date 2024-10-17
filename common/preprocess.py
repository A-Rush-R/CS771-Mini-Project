import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Literal
import numpy as np

def get_char_columns(df):
    for i in range(len(df['input_emoticon'][0])):
        df[f'c_{i+1}'] = df['input_emoticon'].apply(lambda x, _i=i: x[_i])
    
    return df[df.columns.to_list()[2:] + ['label']]

def getdfs(data : str, train_size : float = 1):
    '''
    preprocess and return train_df and val_df
    '''
    if data not in ['text_seq', 'feature' , 'emoticon']:
        raise ValueError("Invalid data type")
    
    if data == 'feature' : 
        train_df = np.load(f"../datasets/train/train_{data}.npz")
        valid_df = np.load(f"../datasets/valid/valid_{data}.npz")
    else :
        train_df = pd.read_csv(f"../datasets/train/train_{data}.csv")
        valid_df = pd.read_csv(f"../datasets/valid/valid_{data}.csv")
    
    train_df = train_df.iloc[:int(len(train_df)*train_size)]

    # if data == 'emoticon':
    #     train_df = get_char_columns(train_df)
    #     valid_df = get_char_columns(valid_df)

    return train_df, valid_df

def one_hot_encode(train_df, valid_df):
    '''
    one hot encode the character columns for emoticons
    '''
    y_train = train_df['label']
    y_val = valid_df['label']
    
    new_train_df = train_df.drop('label', axis=1)
    new_valid_df = valid_df.drop('label', axis = 1)
    
    oh_encoder = OneHotEncoder(handle_unknown = 'ignore')
    oh_encoder.fit(new_train_df)
    
    
    new_train_df = pd.DataFrame(oh_encoder.transform(new_train_df).toarray())
    new_valid_df = pd.DataFrame(oh_encoder.transform(new_valid_df).toarray())
    
    print(new_train_df.shape, new_valid_df.shape)
    
    return new_train_df, new_valid_df, y_train, y_val
