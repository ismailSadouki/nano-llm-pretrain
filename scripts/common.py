import pyarrow as pa

SCHEMA = pa.schema([
        ("doc_id", pa.string()),
        ("text", pa.string()),
        ("id", pa.string()),
        ("dump", pa.string()),
        ("url", pa.string()),
        ("date", pa.string()),              
        ("file_path", pa.string()),
        ("language", pa.string()),
        ("language_score", pa.float32()),
        ("token_count", pa.int32()),
    ])