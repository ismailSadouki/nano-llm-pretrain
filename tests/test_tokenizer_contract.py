tokenizer = load_tokenizer(...)

assert tokenizer.token_to_id("<pad>") == 0
assert tokenizer.token_to_id("<bos>") == 1
assert tokenizer.token_to_id("<eos>") == 2
assert tokenizer.token_to_id("<unk>") == 3

assert tokenizer.get_vocab_size() == 16000