import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from models.model import GPTModel, GPTConfig
from models.generation import generate
from utils.checkpoint import load_checkpoint
from utils.device import get_device_and_dtype
from utils.tokenizer_utils import load_tokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    device, _, _ = get_device_and_dtype("fp32")

    # Build model from checkpoint config
    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model_config = ckpt["model_config"]

    model = GPTModel(model_config).to(device)

    load_checkpoint(
        path=args.checkpoint,
        model=model,
    )

    model.eval()

    tokenizer_path = ckpt["train_config"]["tokenizer_path"]
    tokenizer = load_tokenizer(tokenizer_path)


    encoded = tokenizer.encode(args.prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    output = generate(
        model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_k=50,
    )

    text = tokenizer.decode(
        output[0].tolist()
    )
    print(text)


if __name__ == "__main__":
    main()