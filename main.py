import numpy as np
import pandas as pd
from utils import print_accuracy, remove_common_characters, get_char_columns, find_common_characters, process_strings, get_columns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

def generate_submission_txt(model, x_test, file_name):
    '''
    Generate submission file for the model
    '''
    
    preds = model.predict(x_test)
    with open(file_name, 'w') as f:
        for pred in preds:
            f.write(str(pred) + '\n')
    print(f"Submission file generated at {file_name}")

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

    
    x_train = pd.DataFrame(oh_encoder.transform(train_df).toarray())
    x_valid = pd.DataFrame(oh_encoder.transform(valid_df).toarray())
    x_test = pd.DataFrame(oh_encoder.transform(test_df).toarray())
    
    params = {'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}

    model = LogisticRegression(**params, max_iter= 10000)
    model.fit(x_train, y_train)
    y_valid_pred = model.predict(x_valid)
    
    print_accuracy(y_valid, y_valid_pred)
    
    generate_submission_txt(model, x_test, file_name='pred_emoticon.txt')


def save_features() :
    train_df = np.load('datasets/train/train_feature.npz', allow_pickle=True)
    valid_df = np.load('datasets/valid/valid_feature.npz', allow_pickle=True)
    test_df = np.load('datasets/test/test_feature.npz', allow_pickle=True)
    
    x_train = train_df['features']
    y_train = train_df['label']
    x_valid = valid_df['features']
    x_valid = valid_df['features']
    y_valid = valid_df['label']
    
    x_test = test_df['features']
    
    # model = 
    model.fit(x_train, y_train)
    y_valid_pred = model.predict(x_valid)
    
    print_accuracy(y_valid, y_valid_pred)
    
    generate_submission_txt(model, x_test, file_name='pred_deepfeat.txt')
    
def save_text_seq() :
    emo_train_df = pd.read_csv('datasets/train/train_emoticon.csv')
    repeat_emos = find_common_characters(emo_train_df['input_emoticon'])
    print(repeat_emos)
    repeat_emo_code = {
        "🙼": "284",
        "🛐": "464",
        "🙯": "262",
        "😛": "15436",
        "😣": "614",
        "😑": "1596",
        "🚼": "422",
    }

    train_df = pd.read_csv('datasets/train/train_text_seq.csv')
    valid_df = pd.read_csv('datasets/valid/valid_text_seq.csv')
    test_df = pd.read_csv('datasets/test/test_text_seq.csv')
    
    train_df["input_str"] = process_strings(train_df["input_str"])
    valid_df["input_str"] = process_strings(valid_df["input_str"])
    test_df["input_str"] = process_strings(test_df["input_str"])

    num_feat = 15  # size of the max length among all strings. the rest of the strings are padded to this length
    
    train_df = get_columns(train_df)
    valid_df = get_columns(valid_df)
    test_df = get_columns(test_df)

    oh_encoder = OneHotEncoder(handle_unknown = 'ignore')
    oh_encoder.fit(train_df)

    x_train = train_df.values
    x_valid = valid_df.values
    x_test = test_df.values
    # model = 
    model.fit(x_train, y_train)
    y_valid_pred = model.predict(x_valid)
    
    print_accuracy(y_valid, y_valid_pred)
    
    generate_submission_txt(model, x_test, file_name='pred_text_seq.txt')



if __name__ == "__main__" :
    
    save_emoticons()
    # save_features()
    save_text_seq()