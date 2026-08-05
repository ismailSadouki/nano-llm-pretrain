import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.decoder import DecoderBlock
from models.model import GPTConfig


def build_decoder():

    config = GPTConfig(
        vocab_size=16000,
        block_size=32,
        n_layers=2,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        ffn_mult=4,
        bias=False,
        attn_impl="naive",
    )

    return DecoderBlock(config), config


def test_decoder_output_shape():

    model, config = build_decoder()

    x = torch.randn(
        2,
        16,
        config.d_model,
    )

    out = model(x)

    assert out.shape == x.shape


def test_decoder_dtype():

    model, config = build_decoder()

    x = torch.randn(
        2,
        16,
        config.d_model,
    )

    out = model(x)

    assert out.dtype == x.dtype


def test_decoder_backward():

    model, config = build_decoder()

    x = torch.randn(
        2,
        16,
        config.d_model,
        requires_grad=True,
    )

    out = model(x)

    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_decoder_contains_expected_modules():

    model, _ = build_decoder()

    assert isinstance(model.attn_norm, torch.nn.Module)
    assert isinstance(model.attn, torch.nn.Module)
    assert isinstance(model.ffn_norm, torch.nn.Module)
    assert isinstance(model.ffn, torch.nn.Module)


def test_decoder_residual_connection():

    model, config = build_decoder()

    x = torch.randn(
        2,
        16,
        config.d_model,
    )

    out = model(x)

    # residual block should modify the activations
    assert not torch.allclose(out, x)


def test_decoder_train_eval():

    model, config = build_decoder()

    x = torch.randn(
        2,
        16,
        config.d_model,
    )

    model.train()
    out_train = model(x)

    model.eval()
    out_eval = model(x)

    assert out_train.shape == out_eval.shape


if __name__ == "__main__":

    test_decoder_output_shape()
    test_decoder_dtype()
    test_decoder_backward()
    test_decoder_contains_expected_modules()
    test_decoder_residual_connection()
    test_decoder_train_eval()

    print("✓ Decoder tests passed")