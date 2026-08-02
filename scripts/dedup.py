from common import SCHEMA

import json
from pathlib import Path

from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasketch import MinHashLSH


import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dedup_utils import (
    build_minhash,
    shingles
)


with open("configs/dedup_v0.yaml", "r") as f:
    config = yaml.safe_load(f)




INPUT_DIR = Path("data/filtered")
OUTPUT_DIR = Path("data/dedup")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

lsh = MinHashLSH(threshold=config["similarity_threshold"], num_perm=config["num_perm"]) # MinHashLSH stores many fingerprints and lets you efficiently find similar ones without comparing against every document.

stats = {
    "seen": 0,
    "kept": 0,
    "removed": 0,
    "removed_tokens": 0,
}
removed_examples = []


raw_shards = sorted(INPUT_DIR.glob("shard_*.parquet"))
pbar = tqdm(
    raw_shards,
    desc="Deduplicating shards",
    unit="shard",
)
for shard_path in pbar:
    output_path = OUTPUT_DIR / shard_path.name

    table = pq.read_table(shard_path)
    docs = table.to_pylist()

    kept = []
    for doc in docs:

        stats["seen"] += 1
        mh = build_minhash(doc["text"], config["num_perm"], config["shingle_size"])
        matches = lsh.query(mh)

        if matches:
            stats['removed'] += 1
            stats["removed_tokens"] += doc["token_count"]

            removed_examples.append({
                "removed_doc": doc["doc_id"],
                "matched_docs": matches,
                "cluster_size": len(matches),
            })

        else:


            lsh.insert(doc['doc_id'], mh)

            kept.append(doc)
            stats['kept'] += 1


    table = pa.Table.from_pylist(kept, schema=SCHEMA)

    pq.write_table(table, output_path)
    pbar.set_postfix(
        docs=stats["seen"],
        kept=stats["kept"],
        removed=stats["removed"],
    )


pbar.close()

duplicate_rate = (
    stats["removed"] / stats["seen"]
    if stats["seen"] else 0.0
)
report = {
    "stats": stats,
    "duplicate_rate": duplicate_rate,
    "config": config,
    "num_perm": config["num_perm"],
    "shingle_size": config["shingle_size"],
    "similarity_threshold": config["similarity_threshold"],
    "examples": removed_examples[:20],
}

with open(OUTPUT_DIR / "dedup_report.json", "w") as f:
    json.dump(report, f, indent=2)


print(
    f"Processed {stats['seen']:,} docs | "
    f"Kept {stats['kept']:,} | "
    f"Removed {stats['removed']:,}"
)