from sklearn.metrics import accuracy_score
import typing

def find_common_characters(strings):
    # Convert the first string to a set of characters
    common_chars = set(strings[0])

    # Intersect with characters from all other strings
    for string in strings[1:]:
        common_chars &= set(string)
    
    return common_chars

def remove_common_characters(strings):
    new_strings = []
    
    common_chars = find_common_characters(strings)
    
    for string in strings:
        # Remove all common characters from the string
        new_string = ''.join(char for char in string if char not in common_chars)
        new_strings.append(new_string)

    return new_strings
        
def print_accuracy(y_valid, y_pred, title : str = 'model') :
    print(f"Accuracy for {title} is {accuracy_score(y_valid, y_pred)}")

def get_char_columns(df):
    for i in range(3):
        df[f'c_{i+1}'] = df['input_emoticon'].apply(lambda x, _i=i: x[_i])
        
    columns = df.columns.to_list()
    if 'label' in columns:
        columns.remove('label')
    columns.remove('input_emoticon')
    
    return df[columns + (['label'] if 'label' in df.columns else []) ]

def process_strings(strs: typing.List[str]) -> typing.List[str]:
    strs = [x.lstrip("0") for x in strs]

    # Example usage
    substrings = repeat_emo_code.values()

    # Remove the substrings
    strs = [remove_substrings(input_string, substrings) for input_string in strs]

    padded_strs = []

    for s in strs:
        if len(s) < 15:
            s = s + "0" * (15 - len(s))
        padded_strs.append(s)

    return padded_strs


def remove_substrings(input_string, substrings):
    """
    Removes all occurrences of substrings from the input string.

    Parameters:
    input_string (str): The string to remove substrings from.
    substrings (list): List of substrings to remove from the input string.

    Returns:
    str: The input string with substrings removed.
    """
    for substring in substrings:
        input_string = input_string.replace(substring, "")
    return input_string

def get_columns(df, num_feat):
    for i in range(num_feat):
        df[f"c_{i}"] = df["input_str"].apply(lambda x: x[i])
    return df.drop(columns=["input_str"])

