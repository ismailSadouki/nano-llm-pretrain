# Engineering Decisions

| Date | Decision | Evidence | Consequence |
|------|----------|----------|-------------|
| 2026-08-01 | Selected HuggingFaceFW/FineWeb as the pretraining corpus. | Large open English corpus (~15T GPT-2 tokens), permissive research use, includes metadata (URL, date, language, token_count). | Build ingestion around FineWeb streaming. |
| 2026-08-01 | Stream data instead of downloading the full dataset. | FineWeb is ~15T tokens and too large for local storage. Hugging Face streaming allows sequential processing. | Ingestion is sequential and resumable. |
| 2026-08-01 | Store ingestion output as Parquet shards. | Columnar format, efficient IO, easy filtering and metadata access with PyArrow. | Later pipeline stages read Parquet instead of FineWeb directly. |
| 2026-08-01 | Keep ingestion separate from filtering. | Simpler pipeline and easier debugging. | `filter.py` consumes Parquet shards rather than the original dataset. |