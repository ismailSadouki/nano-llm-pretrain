import json

import pyarrow.parquet as pq
import pyarrow as pa
import yaml
from collections import defaultdict
from statistics import mean, median
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from filters import (
    length_filter,
    token_count_filter,
    symbol_ratio_filter,
    repeated_char_filter,
    repeated_lines_filter,
)


with open("configs/data_filter_v0.yaml", "r") as f:
    config = yaml.safe_load(f)

FILTERS = [
            length_filter,
            symbol_ratio_filter,
            repeated_char_filter,
            repeated_lines_filter,
            token_count_filter
        ]

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

OUTPUT_DIR = Path("data/filtered")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



stats = {
    "seen": 0,
    "kept": 0,
    "dropped": 0,
    "reasons": defaultdict(int)
}
metrics = defaultdict(list)
dropped_examples = []

    

raw_shards = sorted(Path("data/raw").glob("shard_*.parquet"))

for shard_path in tqdm(raw_shards, desc="Filtering shards"):
    filtered_output_path = OUTPUT_DIR / shard_path.name

    table = pq.read_table(shard_path)
    docs = table.to_pylist()
    filtered = []

    for doc in docs:
        stats["seen"] += 1
        results = [f(doc, config) for f in FILTERS]
        passed = all(r['passed'] for r in results)


        if passed:
            filtered.append(doc)
            stats["kept"] += 1
        else:
            stats["dropped"] += 1

            failed = [r for r in results if not r["passed"]]

            for r in failed:
                stats["reasons"][r["name"]] += 1

            if len(dropped_examples) < 20:
                dropped_examples.append({
                    "doc_id": doc["doc_id"],
                    "reasons": [r["name"] for r in failed],
                    "text": doc["text"][:300],
                })

        for r in results:
            if "chars" in r:
                metrics["char_length"].append(r["chars"])

            if "symbol_ratio" in r:
                metrics["symbol_ratio"].append(r["symbol_ratio"])

            if "value" in r:
                metrics[r["name"]].append(r["value"])

            if "longest_run" in r:
                metrics["longest_repeat"].append(r["longest_run"])

    # Write it back.
    filtered_table = pa.Table.from_pylist(filtered, schema=SCHEMA)

    pq.write_table(
        filtered_table,
        filtered_output_path,
    )




metric_summary = {}

for name, values in metrics.items():
    if not values:
        continue

    metric_summary[name] = {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "p95": float(np.percentile(values, 95)),
    }

report = {
    "stats": {
        **stats,
        "keep_rate": stats["kept"] / stats["seen"],
        "drop_rate": stats["dropped"] / stats["seen"],
    },
    "metrics": metric_summary,
    "dropped_examples": dropped_examples[:20],
    "config": config,
}

with open(OUTPUT_DIR / "filter_report.json", "w") as f:
    json.dump(report, f, indent=2)



