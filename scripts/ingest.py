import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm


OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# SHARD_SIZE = 10
# TARGET_TOKENS = None
# TARGET_DOCS = 1000

TARGET_TOKENS = 200_000_000
TARGET_DOCS = None
SHARD_SIZE = 10_000

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


def write_shard(buffer, output_path):
    """Write one buffered batch of documents to a Parquet shard."""
    table = pa.Table.from_pylist(buffer, schema=SCHEMA)
    pq.write_table(table, output_path)

def get_directory_size(path: Path) -> int:
    """Return total size of all files in bytes."""
    return sum(
        f.stat().st_size
        for f in path.glob("*")
        if f.is_file()
    )

def format_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def write_manifest(
    output_dir,
    dataset_name,
    split,
    docs_seen,
    tokens_seen,
    shard_count,
    shard_size,
    size_bytes,
    config_path,
    schema_version
):
    """Write metadata describing the generated dataset."""

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "streaming": True,
        "documents": docs_seen,
        "tokens": tokens_seen,
        "shards": shard_count,
        "target_tokens": TARGET_TOKENS,
        "target_docs": TARGET_DOCS,
        "bytes": get_directory_size(output_dir),
        "size": format_bytes(size_bytes),
        "config": config_path,
        "output_dir": str(output_dir),
        "shard_size": shard_size,
        "schema_version": schema_version,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    """Stream FineWeb, write Parquet shards, and generate a manifest."""

    buffer = []
    shard_idx = 0


    tokens_seen = 0
    docs_seen = 0





  
    dataset = load_dataset('HuggingFaceFW/fineweb',  split='train', streaming=True)

    if TARGET_DOCS is not None:
        total = TARGET_DOCS
        unit = "docs"
    else:
        total = TARGET_TOKENS
        unit = "tok"

    pbar = tqdm(
        total=total,
        unit=unit,
        unit_scale=True,
        desc="Ingesting",
    )

    for doc in dataset:
        record = {
            "doc_id": f"fineweb-{docs_seen:012d}",
            **doc,
        }
        buffer.append(record)
        if TARGET_DOCS is not None:
            pbar.update(1)
        else:
            pbar.update(doc["token_count"])
        tokens_seen += doc["token_count"]
        docs_seen += 1
        


        if len(buffer) >= SHARD_SIZE:
            output_path = OUTPUT_DIR / f"shard_{shard_idx:05d}.parquet"
            write_shard(
                buffer,
                output_path,
                
            )
            pbar.set_postfix(
                shard=f"{shard_idx:05d}",
                docs=docs_seen,
            )
            buffer.clear()
            
            
            shard_idx += 1


        if TARGET_DOCS is not None and docs_seen >= TARGET_DOCS:
            break

        if TARGET_TOKENS is not None and tokens_seen >= TARGET_TOKENS:
            break

    pbar.write(
            f"Wrote {output_path.name} | "
            f"docs={docs_seen:,} | "
            f"tokens={tokens_seen:,}"
        )
        

    if buffer:
        output_path = OUTPUT_DIR / f"shard_{shard_idx:05d}.parquet"
        write_shard(
            buffer,
            output_path,
            
        )
        pbar.write(
                f"Wrote {output_path.name} | "
                f"docs={docs_seen:,} | "
                f"tokens={tokens_seen:,}"
            )

 
    pbar.close()



    size_bytes = get_directory_size(OUTPUT_DIR)
    write_manifest(
        OUTPUT_DIR,
        "HuggingFaceFW/fineweb",
        "train",
        docs_seen,
        tokens_seen,
        shard_idx + bool(buffer),
        SHARD_SIZE,
        size_bytes,
        "configs/ingest.yaml",
        1
    )

if __name__ == "__main__":
    main()