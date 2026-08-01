def token_count_filter(doc, config):
    tokens = doc["token_count"]

    return {
        "name": "token_count",
        "value": tokens,
        "passed": (
            config["min_token_count"]
            <= tokens
            <= config["max_token_count"]
        ),
    }