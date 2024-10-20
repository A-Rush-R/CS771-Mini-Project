import numpy as np
import pandas as pd
from utils import print_accuracy, remove_common_characters, get_char_columns

def generate_submission_txt(model, x_test, file_name):
    '''
    Generate submission file for the model
    '''
    
    preds = model.predict(x_test)
    with open(file_name, 'w') as f:
        for pred in preds:
            f.write(str(pred) + '\n')
    print(f"Submission file generated at {file_name}")
    
    return

def save_emoticons() :
    train_df = pd.read_csv('datasets/train/train_emoticon.csv')
    valid_df = pd.read_csv('datasets/valid/valid_emoticon.csv')
    test_df = pd.read_csv('datasets/test/test_emoticon.csv')
    
    train_df['input_emoticon'] = remove_common_characters(train_df['input_emoticon'])
    valid_df['input_emoticon'] = remove_common_characters(valid_df['input_emoticon'])
    test_df['input_emoticon'] = remove_common_characters(test_df['input_emoticon'])

    train_df = get_char_columns(train_df)
    valid_df = get_char_columns(valid_df)
    test_df = get_char_columns(test_df)
    
    y_train = train_df['label']
    y_valid = valid_df['label']
    
    train_df = train_df.drop('label', axis=1)
    valid_df = valid_df.drop('label', axis = 1)
    
    oh_encoder = OneHotEncoder(handle_unknown = 'ignore')
    oh_encoder.fit(train_df)

    
    train_df = pd.DataFrame(oh_encoder.transform(train_df).toarray())
    valid_df = pd.DataFrame(oh_encoder.transform(valid_df).toarray())
    test_df = pd.DataFrame(oh_encoder.transform(test_df).toarray())
    
    train_df = get_char_columns(train_df)
    valid_df = get_char_columns(valid_df)
    test_df = get_char_columns(test_df)
    
    # model = 
    model.fit 
    y_pred = model.predict
    
    print_accuracy(y_valid, y_pred)
    
    generate_submission_txt(model, x_test, file_name='pred_emoticon.txt')

    
def save_features() :
    train_df = np.load('datasets/train/train_feature.npz', allow_pickle=True)
    valid_df = np.load('datasets/valid/valid_feature.npz', allow_pickle=True)
    test_df = np.load('datasets/test/test_feature.npz', allow_pickle=True)
    
    # model = 
    model.fit 
    y_pred = model.predict
    
    print_accuracy(y_valid, y_pred)
    
    generate_submission_txt(model, x_test, file_name='pred_deepfeat.txt')
    
def save_text_seq() :
    
    train_df = pd.read_csv('datasets/train/train_text_seq.csv')
    valid_df = pd.read_csv('datasets/valid/valid_text_seq.csv')
    test_df = pd.read_csv('datasets/test/test_text_seq.csv')
    
    # model = 
    model.fit 
    y_pred = model.predict
    
    print_accuracy(y_valid, y_pred)
    
    generate_submission_txt(model, x_test, file_name='pred_text_seq.txt')



if __name__ == "__main__" :
    
    save_emoticons()
    save_features()
    save_text_seq()