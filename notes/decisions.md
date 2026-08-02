# Engineering Decisions

| Date | Decision | Evidence | Consequence |
|------|----------|----------|-------------|
| 2026-08-01 | Selected HuggingFaceFW/FineWeb as the pretraining corpus. | Large open English corpus (~15T GPT-2 tokens), permissive research use, includes metadata (URL, date, language, token_count). | Build ingestion around FineWeb streaming. |
| 2026-08-01 | Stream data instead of downloading the full dataset. | FineWeb is ~15T tokens and too large for local storage. Hugging Face streaming allows sequential processing. | Ingestion is sequential and resumable. |
| 2026-08-01 | Store ingestion output as Parquet shards. | Columnar format, efficient IO, easy filtering and metadata access with PyArrow. | Later pipeline stages read Parquet instead of FineWeb directly. |
| 2026-08-01 | Keep ingestion separate from filtering. | Simpler pipeline and easier debugging. | `filter.py` consumes Parquet shards rather than the original dataset. |
| 2026-08-02 | Apply quality filtering before deduplication. | Removing low-quality documents first prevents noisy or malformed text from influencing similarity comparisons. | `dedup.py` operates on cleaned Parquet shards from `data/filtered`. |
| 2026-08-02 | Implement near-duplicate removal using MinHash + LSH with configurable shingle size, signature length, and similarity threshold. | Exact hashing removes only identical documents, whereas MinHash approximates Jaccard similarity and scales efficiently to large web corpora. | The deduplication stage removes highly similar documents before token packing and produces a reproducible report of removed documents and tokens. |