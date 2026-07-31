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

Raw FineWeb
↓
Streaming download
↓
Language filtering
↓
Document quality filtering
↓
Unicode normalization (NFC)
↓
Exact duplicate removal
↓
MinHash near-duplicate removal
↓
Tokenizer training
↓
Sequence packing
↓
Training

---

## Initial Scale



Dataset:
- ≈200M tokens

Target model:
- ~20–30M parameters

Target training tokens:
- ~400M


Context length:
- 1024

Tokenizer:
- BPE
- Vocabulary = 32k

---

Future work

Replace FineWeb with an Algerian Darija corpus while keeping the same pipeline.