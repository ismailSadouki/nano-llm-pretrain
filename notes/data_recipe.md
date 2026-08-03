# Data Recipe

## Corpus

### Primary corpus

- FineWeb (English)

### Reason for selection

- Publicly available
- Modern web corpus
- Large enough for LLM pretraining
- Supports streaming through Hugging Face Datasets
- Suitable for implementing a complete preprocessing pipeline

### Purpose

- Build a complete decoder-only LLM pretraining pipeline.
- Learn corpus preprocessing and quality filtering.
- Implement MinHash-LSH deduplication.
- Train a custom Byte-Level BPE tokenizer.
- Produce packed training data for language model pretraining.

---

# Pipeline Overview

```
FineWeb
    ↓
ingest.py
    ↓
raw Parquet shards
    ↓
filter.py
    ↓
filtered Parquet shards
    ↓
dedup.py
    ↓
deduplicated Parquet shards
    ↓
train_tokenizer.py
    ↓
tokenizer/tokenizer.json
    ↓
pack.py
    ↓
packed train/validation arrays
```

---

# Data Source

Dataset

- HuggingFaceFW/FineWeb
- English subset
- Accessed through the Hugging Face Datasets streaming API

Reason

- Large-scale open web corpus
- Rich document metadata
- Commonly used for LLM pretraining experiments

---

# Stage 1 — Ingestion

### Input

- FineWeb streaming dataset

### Output

```
data/raw/
```

### Stored fields

- doc_id
- text
- id
- dump
- url
- date
- file_path
- language
- language_score
- token_count

### Configuration

```
configs/ingest_v0.yaml
```

### Implementation

```bash
python scripts/ingest.py
```

---

# Stage 2 — Quality Filtering

### Input

```
data/raw/
```

### Output

```
data/filtered/
```

### Applied filters

| Filter | Threshold |
|---------|----------:|
| Minimum characters | `min_chars` |
| Maximum characters | `max_chars` |
| Maximum symbol ratio | `max_symbol_ratio` |
| Minimum token count | `min_token_count` |
| Maximum token count | `max_token_count` |
| Maximum repeated character run | `max_repeat_char` |
| Maximum repeated line ratio | `max_repeated_line_ratio` |

### Configuration

```
configs/data_filter_v0.yaml
```

### Implementation

```bash
python scripts/filter.py
```

---

# Stage 3 — Near-Duplicate Removal

### Input

```
data/filtered/
```

### Output

```
data/dedup/
```

### Method

- MinHash
- Locality Sensitive Hashing (LSH)

### Configuration

| Parameter | Value |
|-----------|------:|
| num_perm | See config |
| shingle_size | See config |
| similarity_threshold | See config |

Configuration file

```
configs/dedup_v0.yaml
```

### Implementation

```bash
python scripts/dedup.py
```

---

# Stage 4 — Tokenizer Training

### Input

```
data/dedup/
```

### Output

```
tokenizer/tokenizer.json
```

### Tokenizer

- Byte-Level BPE
- Vocabulary size: 16,000
- Trained only on the cleaned and deduplicated corpus

### Special tokens

- `<pad>`
- `<bos>`
- `<eos>`
- `<unk>`

### Configuration

```
configs/tokenizer_v0.yaml
```

### Implementation

```bash
python scripts/train_tokenizer.py
```

---

# Stage 5 — Sequence Packing

### Input

```
data/dedup/
```

### Output

```
data/packed/
```

### Procedure

1. Encode each document using the custom tokenizer.
2. Append an `<eos>` token.
3. Concatenate all tokenized documents.
4. Split into fixed-length sequences.
5. Generate next-token labels.
6. Create loss masks to ignore padding tokens.

### Configuration

```
configs/packing_v0.yaml
```

### Implementation

```bash
python scripts/pack.py
```

---

# Final Outputs

```
data/packed/

train_input_ids.npy
train_labels.npy
train_loss_mask.npy

val_input_ids.npy
val_labels.npy
val_loss_mask.npy

packing_report.json
golden_test.json
```

---

# Reproducibility

Run the complete pipeline from raw data to packed sequences:

```bash
python scripts/ingest.py
python scripts/filter.py
python scripts/dedup.py
python scripts/train_tokenizer.py
python scripts/pack.py
```

All preprocessing parameters are stored under:

```
configs/
```

Each pipeline stage generates a JSON report containing the corresponding statistics.

---

# Current Dataset Statistics

| Metric | Value |
|-------|------:|
| Documents | 996 |
| Total tokens | 733,960 |
| Training tokens | 726,620 |
| Validation tokens | 7,340 |
| Context length | 1024 |
| Training blocks | 710 |
| Validation blocks | 8 |
| Packing utilization | 99.83% |
| Padding percentage | 0.17% |

---

# Current Limitations

- The tokenizer is trained only on a small FineWeb subset.
- Packed datasets must be regenerated whenever the tokenizer changes.
- Deduplication thresholds may require retuning for larger corpora.
- PII removal is not implemented.
- Language quality classification is not implemented.

---

# Initial Scale

Dataset

- 996 documents
- 733,960 tokens after preprocessing

Tokenizer

- Byte-Level BPE
- Vocabulary size: 16,000

Model

- Target size: 20–30M parameters

Training

- Target training corpus: ~200M tokens
- Context length: 1024 tokens

---

# Future Work

- Scale preprocessing to a substantially larger FineWeb subset.
- Evaluate tokenizer fertility on held-out text.
- Replace the English corpus with an Algerian Darija corpus while preserving the same preprocessing pipeline.
- Add language identification and PII removal.
- Explore document-aware packing and attention masking.