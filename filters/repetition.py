import re
def repeated_char_filter(doc, config):
    text = doc["text"]
    longest = max(
        (len(m.group()) for m in re.finditer(r"(.)\1+", text)),
        default=1,
    )
    return {
        "name": "repeated_chars",
        "longest_run": longest,
        "passed": longest <= config["max_repeat_char"],
    }

from collections import Counter

def repeated_lines_filter(doc, config):
    text = doc["text"]
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return {
            "name": "repeated_lines",
            "value": 0.0,
            "passed": False,
        }

    counts = Counter(lines)

    repeated = sum(
        count
        for count in counts.values()
        if count > 1
    )

    ratio = repeated / len(lines)

    return {
        "name": "repeated_lines",
        "value": ratio,
        "passed": ratio <= config["max_repeated_line_ratio"],
    }
