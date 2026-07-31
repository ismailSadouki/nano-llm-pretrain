# Data Statement (v0)

## Purpose

This repository demonstrates end-to-end decoder-only LLM pretraining.

The objective is educational and engineering-focused rather than producing a state-of-the-art model.

---

## Corpus

Primary source:

**Dataset:** FineWeb

FineWeb is a large-scale English web corpus created by Hugging Face from Common Crawl web crawls collected between 2013 and 2024. It was processed using the Datatrove library through text extraction, language filtering, quality filtering, MinHash deduplication, and limited PII anonymization.



---

## Provenance

Original source:
- Common Crawl

Dataset provider:
- Hugging Face: https://huggingface.co/datasets/HuggingFaceFW/fineweb

Time span:
- Summer 2013 – April 2024

Language:
- English

---

## Planned Corpus

For this project only a subset of FineWeb will be used.

Target dataset size:
- ~200M tokens after preprocessing

Target model:
- 20–30M parameters

---

## Collection

Documents are streamed using Hugging Face Datasets.

No manual document selection is performed.

---


---

## Planned Preprocessing

Although FineWeb has already been processed, this project intentionally reimplements several preprocessing stages for educational purposes.

The planned pipeline consists of:

- Unicode normalization
- Document filtering
- Exact duplicate removal
- MinHash near-duplicate detection
- Custom BPE tokenizer training
- Sequence packing



Vocabulary:

- #####?

Target Size

- 100–300M training tokens


Filtering parameters are documented in `configs/data_filter_v0.yaml`.





---

## Known Risks

- Bias inherited from web data
- Toxic or offensive content may remain
- Possible factual inaccuracies
- English-only corpus
- Possible benchmark contamination


---

## Intended Use

Educational implementation of a modern LLM pretraining pipeline.

Not intended for production or commercial deployment.



---

### Note

FineWeb has already undergone filtering and deduplication by its creators.

This project reimplements several preprocessing stages to understand and validate the engineering behind modern LLM data pipelines rather than to improve the original dataset.

---

## Training Budget

The target model size is approximately 20–30 million parameters.

Following the **Chinchilla** scaling heuristic, compute-optimal training would require roughly 400–600 million training tokens (≈20 tokens per parameter). Due to the project's hardware and time constraints, this implementation targets approximately 100–300 million training tokens. This represents a practical engineering compromise rather than a compute-optimal training regime.


## Future revisions

- [ ] Report document count
- [ ] Report token count
- [ ] Report filtering statistics
- [ ] Report duplicate removal rate
- [ ] Report language distribution





---

## License

This project uses the publicly released FineWeb dataset and follows its documented licensing and usage terms.
