import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from tqdm import tqdm
import numpy as np


import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.pack_utils import load_tokenizer, pack_tokens


# IMPORTANT:
# Packed datasets depend on tokenizer IDs.
# If tokenizer.json changes, rerun:
#
# python scripts/pack.py


with open("configs/packing_v0.yaml", "r") as f:
    config = yaml.safe_load(f)


INPUT_DIR = Path("data/dedup")
OUTPUT_DIR = Path("data/packed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw_shards = sorted(INPUT_DIR.glob("*.parquet"))



tokenizer = load_tokenizer(config["tokenizer_path"])
EOS_ID = tokenizer.token_to_id("<eos>")
PAD_ID = tokenizer.token_to_id("<pad>")
BOS_ID = tokenizer.token_to_id("<bos>")
UNK_ID = tokenizer.token_to_id("<unk>")
assert EOS_ID is not None, "[EOS] token missing."
assert PAD_ID is not None, "[PAD] token missing."
assert BOS_ID is not None, "[BOS] token missing."
assert UNK_ID is not None, "[UNK] token missing."

assert PAD_ID != EOS_ID, "[PAD] and [EOS] tokens are the same."
assert PAD_ID != BOS_ID, "[PAD] and [BOS] tokens are the same."
assert EOS_ID != BOS_ID, "[EOS] and [BOS] tokens are the same."

all_tokens = []
docs_seen = 0
eos_count = 0

pbar = tqdm(raw_shards, desc="Packing shards")

for shard_path in pbar:
    table = pq.read_table(shard_path)
    docs = table.to_pylist()
    for doc in docs:

        ids = tokenizer.encode(doc["text"]).ids + [EOS_ID]
        all_tokens.extend(ids)
        docs_seen += 1
        eos_count += 1
    pbar.set_postfix(
        docs=docs_seen,
        tokens=len(all_tokens),
    )
pbar.close()



split = int(len(all_tokens) * config["train_split"])
train_tokens = all_tokens[:split]
val_tokens = all_tokens[split:]

# Pack train and validate


BLOCK_SIZE = config["block_size"]

VOCAB_SIZE = tokenizer.get_vocab_size()

train_ids, train_labels, train_mask = pack_tokens(train_tokens, BLOCK_SIZE, PAD_ID)

val_ids, val_labels, val_mask = pack_tokens(val_tokens, BLOCK_SIZE, PAD_ID)

assert train_ids.max() < VOCAB_SIZE
assert val_ids.max() < VOCAB_SIZE


np.save(OUTPUT_DIR / "train_input_ids.npy", train_ids)
np.save(OUTPUT_DIR / "train_labels.npy", train_labels)
np.save(OUTPUT_DIR / "train_loss_mask.npy", train_mask)

np.save(OUTPUT_DIR / "val_input_ids.npy", val_ids)
np.save(OUTPUT_DIR / "val_labels.npy", val_labels)
np.save(OUTPUT_DIR / "val_loss_mask.npy", val_mask)

report = {
    "documents": docs_seen,
    "tokens": len(all_tokens),
    "eos_count": eos_count,
    "block_size": BLOCK_SIZE,
    "train_blocks": len(train_ids),
    "val_blocks": len(val_ids),
    "train_tokens": len(train_tokens),
    "val_tokens": len(val_tokens),
    "pad_tokens": int((train_mask == 0).sum() + (val_mask == 0).sum()),
}
report["utilization_percent"] = (
    100
    * (report["train_tokens"] + report["val_tokens"])
    / (
        (report["train_blocks"] + report["val_blocks"])
        * BLOCK_SIZE
    )
)
report["vocab_size"] = VOCAB_SIZE
report["tokenizer"] = str(Path(config["tokenizer_path"]).resolve())
report["eos_token_id"] = EOS_ID
report["pad_token_id"] = PAD_ID
report["bos_token_id"] = BOS_ID
report["train_split"] = config["train_split"]
report["pad_percentage"] = (
    report["pad_tokens"]
    /
    (
        (report["train_blocks"] + report["val_blocks"])
        * BLOCK_SIZE
    )
    * 100
)
report["avg_tokens_per_doc"] = len(all_tokens) / docs_seen
report["avg_tokens_per_block"] = (
    report["train_tokens"] + report["val_tokens"]
) / (
    report["train_blocks"] + report["val_blocks"]
)

with open(OUTPUT_DIR / "packing_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))



def golden_test(eos_id, pad_id):
    tokens = [1, 2, eos_id, 3, eos_id]

    ids, labels, mask = pack_tokens(
        tokens,
        block_size=6,
        pad_id=pad_id,
    )

    golden = {
        "input_ids": ids.tolist(),
        "labels": labels.tolist(),
        "loss_mask": mask.tolist(),
    }

    with open(OUTPUT_DIR / "golden_test.json", "w") as f:
        json.dump(golden, f, indent=2)
    print(ids)
    print(labels)
    print(mask)

golden_test(EOS_ID, PAD_ID)