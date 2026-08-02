# Data Statistics

## Pipeline Summary

| Stage | Output |
|--------|---------|
| Ingestion | data/raw |
| Filtering | data/filtered |
| Deduplication | data/dedup |
| Packing | data/packed |

---

# Filtering

Source:

```
data/filtered/filter_report.json
```


Record:

- Documents processed
- Documents kept
- Documents removed
- Removal reasons
- Character length statistics
- Token statistics
- Symbol ratio statistics
- Example removed documents

---

# Deduplication

Source:

```
data/dedup/dedup_report.json
```


Record:

- Documents processed
- Documents removed
- Removed tokens
- Similarity threshold
- MinHash parameters
- Example duplicate pairs

---

# Packing

Source:

```
data/packed/packing_report.json
```



Record:

- Documents
- Tokens
- Train tokens
- Validation tokens
- Block size
- EOS count
- Train blocks
- Validation blocks
- Padding percentage
- Token utilization

---

# Final Dataset

| Metric | Value |
|---------|------:|
| Documents | *(from packing_report)* |
| Tokens | *(from packing_report)* |
| Train tokens | *(from packing_report)* |
| Validation tokens | *(from packing_report)* |
| Utilization | *(from packing_report)* |
| Padding | *(from packing_report)* |

---

# Biases and Limitations

- English-only corpus.
- Web data may contain factual inaccuracies and duplicated content.
- Current tokenizer is GPT-2 and will be replaced in M2.
- Deduplication uses MinHash LSH and may not detect all semantic duplicates.
