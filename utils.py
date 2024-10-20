

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
