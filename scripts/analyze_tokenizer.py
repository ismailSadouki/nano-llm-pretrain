from pathlib import Path

import pyarrow.parquet as pq
import yaml
from tokenizers import Tokenizer
import pyarrow.parquet as pq
import numpy as np

import json

with open("configs/tokenizer_v0.yaml") as f:
    config = yaml.safe_load(f)

tokenizer = Tokenizer.from_file(config["tokenizer_dir"] + "/tokenizer.json")

INPUT_DIR = Path("data/dedup")

OUTPUT_DIR = Path("reports/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Hold out the last 10% of shards
raw_shards = sorted(INPUT_DIR.glob("*.parquet"))


split = int(len(raw_shards) * (1 - config["heldout_split"]))

train_shards = raw_shards[:split]
heldout_shards = raw_shards[split:]


print(f"Held-out shards: {len(heldout_shards)}")










def heldout_iterator(shards):
    for shard in shards:
        table = pq.read_table(shard)
        for text in table["text"]:
            yield text.as_py()


tokens_per_word = []
tokens_per_char = []

examples = []

total_tokens = 0
total_words = 0
total_chars = 0


for text in heldout_iterator(heldout_shards):
    encoding = tokenizer.encode(text)

    words = len(text.split())
    chars = len(text)
    tokens = len(encoding.ids)

    if words == 0 or chars == 0:
        continue

    total_tokens += tokens
    total_words += words
    total_chars += chars

    tpw = tokens / words
    tpc = tokens / chars

    tokens_per_word.append(tpw)
    tokens_per_char.append(tpc)

    examples.append(
        {
            "text": text[:500],          # keep JSON reasonably small
            "decoded": tokenizer.decode(encoding.ids),
            "tokens": encoding.tokens,
            "ids": encoding.ids,
            "num_tokens": tokens,
            "num_words": words,
            "num_characters": chars,
            "tokens_per_word": tpw,
            "tokens_per_char": tpc,
        }
    )


stats = {
    "documents": len(tokens_per_word),
    "total_tokens": total_tokens,
    "total_words": total_words,
    "total_characters": total_chars,
    "tokens_per_word": {
        "mean": float(np.mean(tokens_per_word)),
        "median": float(np.median(tokens_per_word)),
        "p90": float(np.percentile(tokens_per_word, 90)),
    },
    "tokens_per_char": {
        "mean": float(np.mean(tokens_per_char)),
        "median": float(np.median(tokens_per_char)),
        "p90": float(np.percentile(tokens_per_char, 90)),
    },
}


worst_examples = sorted(
    examples,
    key=lambda x: x["tokens_per_word"],
    reverse=True,
)[:20]


with open(OUTPUT_DIR / "tokenizer_diagnostics.json", "w") as f:
    json.dump(
        {
            "statistics": stats,
            "worst_examples": worst_examples,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )


print(json.dumps(stats, indent=2))
print(f"Saved diagnostics to {OUTPUT_DIR / 'tokenizer_diagnostics.json'}")