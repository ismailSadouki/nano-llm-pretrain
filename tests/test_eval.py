import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import GPTConfig, GPTModel
from utils.data import PackedDataset
from utils.eval import estimate_loss


def build_model():
    config = GPTConfig(
        vocab_size=16000,
        block_size=1024,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=4,
    )
    return GPTModel(config)


def test_estimate_loss():

    model = build_model()

    train_dataset = PackedDataset("train")
    val_dataset = PackedDataset("val")

    losses = estimate_loss(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_iters=2,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert "train" in losses
    assert "val" in losses

    assert isinstance(losses["train"], float)
    assert isinstance(losses["val"], float)

    assert torch.isfinite(torch.tensor(losses["train"]))
    assert torch.isfinite(torch.tensor(losses["val"]))

    # estimate_loss() should restore training mode
    assert model.training


if __name__ == "__main__":
    test_estimate_loss()
    print("✓ estimate_loss test passed")