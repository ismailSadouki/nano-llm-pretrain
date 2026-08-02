# Data Recipe

## Corpus

Primary corpus:
- FineWeb (English)

Reason for selection:

- Publicly available
- Modern web corpus
- Large enough for LLM pretraining
- Easy to stream with Hugging Face Datasets
- Suitable for implementing a complete preprocessing pipeline

Purpose:
- Build a complete LLM pretraining pipeline.
- Learn filtering and deduplication.
- Train a custom tokenizer.
- Train a small decoder-only transformer.

---

## Planned Pipeline



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
pack.py
    ↓
packed train/validation arrays
```

---

## Data source

Dataset:
- HuggingFaceFW/FineWeb
- Accessed through Hugging Face Datasets streaming API
- English subset

Reason:
- Large open web corpus
- Rich metadata
- Widely used for LLM pretraining

---
## Stage 1 — Ingestion

Input

- FineWeb streaming dataset

Output

```
data/raw/
```

Stored fields

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

Implementation

```
python scripts/ingest.py
```
---

## Stage 2 — Quality filtering

Input

```
data/raw/
```

Output

```
data/filtered/
```


Applied filters

| Filter | Threshold |
|---------|----------:|
| Minimum characters | `min_chars` |
| Maximum characters | `max_chars` |
| Maximum symbol ratio | `max_symbol_ratio` |
| Minimum token count | `min_token_count` |
| Maximum token count | `max_token_count` |
| Maximum repeated character run | `max_repeat_char` |
| Maximum repeated line ratio | `max_repeated_line_ratio` |

Configuration

```
configs/data_filter_v0.yaml
```

Implementation

```
python scripts/filter.py
```


---

## Stage 3 — Near-duplicate removal

Input

```
data/filtered/
```

Output

```
data/dedup/
```

Method

- MinHash
- Locality Sensitive Hashing (LSH)

Configuration

| Parameter | Value |
|-----------|------:|
| num_perm | see config |
| shingle_size | see config |
| similarity_threshold | see config |

Configuration


```
configs/dedup_v0.yaml
```

Implementation

```
python scripts/dedup.py
```

---

## Stage 4 — Sequence packing

Input

```
data/dedup/
```

Output

```
data/packed/
```

Tokenizer

- GPT-2 tokenizer (temporary)

Procedure

1. Tokenize every document.
2. Append EOS token.
3. Concatenate all documents.
4. Split into fixed-length blocks.
5. Generate shifted labels.
6. Create loss masks for padded positions.

Configuration

```
configs/packing_v0.yaml
```

Implementation

```
python scripts/pack.py
```

---

## Final outputs

```
train_input_ids.npy
train_labels.npy
train_loss_mask.npy

val_input_ids.npy
val_labels.npy
val_loss_mask.npy
```

---

## Reproducibility

Run the complete pipeline:

```bash
python scripts/ingest.py
python scripts/filter.py
python scripts/dedup.py
python scripts/pack.py
```

All preprocessing parameters are stored under:

```
configs/
```

Each stage produces a JSON report documenting its statistics.

---

## Current limitations

- Uses the GPT-2 tokenizer for M1.
- A custom BPE tokenizer will replace it in M2.
- Deduplication parameters were selected for experimentation and may require retuning on larger corpora.
- PII removal and language-quality classification are not yet included.

---

## Initial Scale



Dataset:
- ≈10000 tokens

Target model:
- ~20–30M parameters

Target training tokens:
- ~200M


Context length:
- 1024

Tokenizer:
- BPE
- Vocabulary = 32k

---

Future work

Replace FineWeb with an Algerian Darija corpus while keeping the same pipeline.