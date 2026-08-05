import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import GPTConfig, GPTModel


def build_model():
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=4,
        tie_embeddings=True,
        bias=False,
        attn_impl="naive",
    )
    return GPTModel(config)


def test_forward_shape():

    model = build_model()

    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (2, 16),
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        16,
        model.config.vocab_size,
    )


def test_forward_with_loss():

    model = build_model()

    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (2, 16),
    )

    targets = torch.randint(
        0,
        model.config.vocab_size,
        (2, 16),
    )

    logits, loss = model(
        input_ids,
        targets,
    )

    assert logits.shape == (
        2,
        16,
        model.config.vocab_size,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward():

    model = build_model()

    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (2, 16),
    )

    targets = torch.randint(
        0,
        model.config.vocab_size,
        (2, 16),
    )

    _, loss = model(
        input_ids,
        targets,
    )

    loss.backward()

    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad
    ]

    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_weight_tying():

    model = build_model()

    assert (
        model.tok_embeddings.weight
        is
        model.lm_head.weight
    )


def test_parameter_count():

    model = build_model()

    n_params = model.get_num_parameters()

    assert n_params > 0


def test_optimizer_creation():

    model = build_model()

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=3e-4,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    assert len(optimizer.param_groups) == 2

    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_sequence_length_assertion():

    model = build_model()

    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (2, model.config.block_size + 1),
    )

    try:
        model(input_ids)
        assert False
    except AssertionError:
        pass

def test_model_forward_backward():
    config = GPTConfig(
        vocab_size=1000,
        block_size=32,
        n_layers=2,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=4,
    )

    model = GPTModel(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 16, config.vocab_size)
    assert loss.ndim == 0

    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)

if __name__ == "__main__":

    test_forward_shape()
    test_forward_with_loss()
    test_backward()
    test_weight_tying()
    test_parameter_count()
    test_optimizer_creation()
    test_sequence_length_assertion()
    test_model_forward_backward()

    print("✓ All model tests passed.")