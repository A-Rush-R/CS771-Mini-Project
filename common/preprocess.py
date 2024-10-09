import pandas as pd

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