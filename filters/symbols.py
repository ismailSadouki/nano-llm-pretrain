import string

SYMBOLS = set(string.punctuation)
def symbol_ratio_filter(doc, config):
    text = doc["text"]
    chars = len(text)
    if chars == 0:
        return {
            "symbol_ratio": 1.0,
            "passed": False
        }
    symbols = sum(c in SYMBOLS for c in text)
    ratio = symbols /chars
    
    return {
        "name": "high_symbol_ratio",
        "symbol_ratio": ratio,
        "passed": ratio <= config["max_symbol_ratio"]
    }