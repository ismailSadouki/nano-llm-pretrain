# Tokenizer Fertility Report (v0)

Date: 2026-08-03

## Purpose

This report evaluates the efficiency of the custom Byte-Level BPE tokenizer trained on the cleaned FineWeb corpus.

Fertility measures how many tokens are required to represent text. Lower fertility means more words fit into the model context window, improving compute efficiency.

Diagnostics were performed on a held-out subset of the corpus (10% of shards) that was **not** used for tokenizer training.

---

# Held-out dataset

- Documents: 99
- Source: `data/dedup/`
- Held-out split: last 10% of shards

---

# Results

| Metric | Mean | Median | P90 |
|---------|-----:|-------:|----:|
| Tokens per word | **1.416** | **1.398** | **1.576** |
| Tokens per character | **0.239** | **0.240** | **0.262** |

Totals:

- Total tokens: **68,146**
- Total words: **48,353**
- Total characters: **286,284**

---

# Interpretation

The tokenizer averages approximately **1.42 tokens per word** on unseen English text.

This indicates that most English words are represented using one or two subword tokens, which is expected for a Byte-Level BPE tokenizer trained on an English corpus.

The 90th percentile (1.58 tokens/word) shows that even relatively difficult text remains reasonably compact.

---

# Effective Context

For a context length of **1024 tokens**:

Approximate words that fit:

1024 / 1.416 ≈ **723 words**

Reducing fertility increases the amount of text the model can process within a fixed context window.

---

# Manual Inspection

Twenty examples with the highest tokens-per-word ratio were saved to:

```
reports/tokenizer_diagnostics.json
```

These examples should be inspected for:

- unusual Unicode characters
- URLs
- code
- repeated punctuation
- fragmented words
- unexpected byte-level splits

---

# Conclusion

The tokenizer demonstrates good compression on held-out English text.

No major pathological segmentations were observed during the initial inspection.

The tokenizer is accepted for subsequent experiments.