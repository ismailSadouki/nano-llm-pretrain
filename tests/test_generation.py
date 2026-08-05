import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.model import GPTConfig, GPTModel
from models.generation import generate


def test_generate_runs():
    config = GPTConfig(
        vocab_size=100,
        block_size=32,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
    )

    model = GPTModel(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    output = generate(
        model,
        input_ids,
        max_new_tokens=5,
        seed=42,
    )

    assert output.shape == (2, 13)