def write_to_file(content, target):
    """ Write string content to a file """
    with open(target, 'w') as target_file:
        target_file.write(content)

def to_float(x):
    if isinstance(x, list):
        return [single_value_to_float(i) for i in x]
    else:
        return single_value_to_float(x)

def single_value_to_float(x):
    try:
        return float(x)
    except Exception: 
        return x

def remove_comment_elements(line_string, comment_token='#'):
    line_list = line_string.split('\n')
    return [ line for line in line_list if not line.startswith(comment_token)]
