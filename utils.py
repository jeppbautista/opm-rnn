def pre_process(text):
    text = text.lower()
    text = "\n".join(list(set(text.split("\n"))))
    return text

def map_chars_to_int(text):
    chars = sorted(list(set(text)))
    return {ch : i for i, ch in enumerate(chars)}

def map_int_to_char(text):
    chars = sorted(list(set(text)))
    return {i : ch for i, ch in enumerate(chars)}

def prepare_dset(text, char_map, params):
    _X = []
    _y = []
    for i in range(0, params["n_chars"] - params["seq_length"]):
        _X.append([char_map[c] for c in text[i:i + params["seq_length"]]])
        _y.append(char_map[text[i + params["seq_length"]]])

    return _X, _y

