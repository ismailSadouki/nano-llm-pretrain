# Tokenizer v0


## Model

- Type: Byte-Level BPE
- Library: Hugging Face tokenizers
- Vocabulary size: 16000
- Minimum frequency: 2

## Version
tokenizer_v0

## Training data

- Source: data/dedup/
- Data version: data-v0
- Corpus: Filtered and deduplicated FineWeb subset

## Special tokens

| Token | ID |
|------|----|
| [PAD] | 0 |
| [UNK] | 1 |
| [BOS] | 2 |
| [EOS] | 3 |

(Replace IDs with the actual values after training.)

## Files

- tokenizer.json

## Training command

```bash
python scripts/train_tokenizer.py
```

## Notes

- Uses Byte-Level BPE.
- Case is preserved (no lowercasing).
- Packed datasets must be regenerated whenever the tokenizer changes.