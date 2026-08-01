def length_filter(doc, config):
    text = doc["text"]
    chars = len(text)

    return {
        "name": "too_short",
        "chars": chars,
        "passed": (
            chars >= config["min_chars"] and 
            chars <= config["max_chars"]
        )
    }