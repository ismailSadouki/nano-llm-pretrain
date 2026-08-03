import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
import yaml
from pathlib import Path

from tqdm import tqdm

def corpus_iterator(shards):
    # total_docs = sum(
    #     pq.read_table(shard, columns=["text"]).num_rows
    #     for shard in shards
    # )

    for shard in shards:
        table = pq.read_table(shard, columns=["text"])
        for text in table["text"]: #tqdm(table["text"], total=table.num_rows, desc=shard.name):
            yield text.as_py()





with open("configs/tokenizer_v0.yaml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = Path(config["input_dir"])
OUTPUT_DIR = Path(config["tokenizer_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw_shards = sorted(INPUT_DIR.glob("*.parquet"))


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevelPreTokenizer()
tokenizer.decoder = ByteLevelDecoder()
trainer = BpeTrainer(
    vocab_size=config["vocab_size"],
    min_frequency=config["min_frequency"],
    special_tokens=config["special_tokens"],
    show_progress=True,
        max_token_length=32,
)



tokenizer.train_from_iterator(corpus_iterator(raw_shards), trainer=trainer)

tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))


print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
print(f"Saved tokenizer to {OUTPUT_DIR/'tokenizer.json'}")