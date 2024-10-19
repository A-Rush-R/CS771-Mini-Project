import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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
    
    model = ??
    model.fit 
    y_pred = model.predict
    
    print("Accuracy Score for validaition set :", accuracy_score(y_valid, y_pred))
    
    generate_submission_txt(model, x_test, file_name='pred_emoticon.txt')
    
def save_features() :
    train_df = np.load('datasets/train/train_feature.npz', allow_pickle=True)
    valid_df = np.load('datasets/valid/valid_feature.npz', allow_pickle=True))
    test_df = np.load('datasets/test/test_feature.npz', allow_pickle=True))
    
    model = ??
    model.fit 
    y_pred = model.predict
    
    print("Accuracy Score for validaition set :", accuracy_score(y_valid, y_pred))
    
    generate_submission_txt(model, x_test, file_name='pred_deepfeat.txt')
    
def save_text_seq() :
    
    train_df = pd.read_csv('datasets/train/train_text_seq.csv')
    valid_df = pd.read_csv('datasets/valid/valid_text_seq.csv')
    test_df = pd.read_csv('datasets/test/test_text_seq.csv')
    
    model = ??
    model.fit 
    y_pred = model.predict
    
    print("Accuracy Score for validaition set :", accuracy_score(y_valid, y_pred))
    
    generate_submission_txt(model, x_test, file_name='pred_text_seq.txt')



if __name__ == "__main__" :
    
    save_emoticons()
    save_features()
    save_text_seq()