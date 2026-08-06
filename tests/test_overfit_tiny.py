import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import GPTConfig, GPTModel



def build_model():
    config = GPTConfig(
        vocab_size=20,
        block_size=8,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=4,
        attn_impl="sdpa", 
    )

    return GPTModel(config)


def test_overfit_tiny():

    input_ids = torch.tensor([
        [1,2,3,4,5,6,7,8]
    ])

    targets = torch.tensor([
        [2,3,4,5,6,7,8,9]
    ])

    model = build_model()
    model.train()

    optimizer = model.configure_optimizers(
        weight_decay=0.0,
        learning_rate=1e-2,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    initial_loss = None

    for step in range(300):

        optimizer.zero_grad()

        _, loss = model(
            input_ids,
            targets,
        )

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()

        optimizer.step()

    # After training
    assert loss.item() < initial_loss
    assert loss.item() < 0.1

    model.eval()


    with torch.no_grad():
        logits = model(input_ids)
        pred = logits.argmax(dim=-1)

    assert torch.equal(pred, targets)


if __name__ == "__main__":

    test_overfit_tiny()

    print("✓ Tiny overfit test passed")