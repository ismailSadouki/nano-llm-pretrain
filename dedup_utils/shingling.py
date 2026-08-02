def shingles(text: str, k: int) -> set[str]:
    words = text.lower().split()


    if len(words) < k:
        return {" ".join(words)}

    return {
        " ".join(words[i:i+k]) for i in range(len(words) - k+1)
    }