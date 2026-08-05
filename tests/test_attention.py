import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.attention import Attention, repeat_kv


def build_attention(attn_impl: str = "naive"):
    return Attention(
        dim=512,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=32,
        attn_impl=attn_impl,
    )


def test_attention_output_shape():

    model = build_attention()

    x = torch.randn(2, 16, 512)

    out = model(x)

    assert out.shape == x.shape


def test_repeat_kv_shape():

    model = build_attention()

    x = torch.randn(
        2,
        model.n_kv_heads,
        16,
        model.head_dim,
    )

    out = repeat_kv(
        x,
        model.n_rep,
    )

    assert out.shape == (
        2,
        model.n_heads,
        16,
        model.head_dim,
    )


def test_repeat_kv_groups():

    model = build_attention()

    x = torch.randn(
        2,
        model.n_kv_heads,
        16,
        model.head_dim,
    )

    out = repeat_kv(
        x,
        model.n_rep,
    )

    for kv_head in range(model.n_kv_heads):

        start = kv_head * model.n_rep
        end = start + model.n_rep

        for h in range(start, end):

            assert torch.allclose(
                out[:, h],
                x[:, kv_head],
            )


def test_attention_backward():

    model = build_attention()

    x = torch.randn(
        2,
        16,
        512,
        requires_grad=True,
    )

    out = model(x)

    loss = out.mean()

    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_naive_and_sdpa_match():

    torch.manual_seed(42)

    naive = build_attention(attn_impl="naive")
    sdpa = build_attention(attn_impl="sdpa")

    sdpa.load_state_dict(
        naive.state_dict()
    )

    x = torch.randn(
        2,
        16,
        512,
    )

    y_naive = naive(x)
    y_sdpa = sdpa(x)

    assert torch.allclose(
        y_naive,
        y_sdpa,
        atol=1e-5,
        rtol=1e-5,
    )


def test_causal_mask():

    torch.manual_seed(42)

    model = build_attention()

    x1 = torch.randn(
        1,
        8,
        512,
    )

    x2 = x1.clone()

    # Modify only the final token.
    x2[:, -1] += 1000

    y1 = model(x1)
    y2 = model(x2)

    # Earlier outputs must be unchanged.
    assert torch.allclose(
        y1[:, :-1],
        y2[:, :-1],
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == "__main__":

    test_attention_output_shape()
    test_repeat_kv_shape()
    test_repeat_kv_groups()
    test_attention_backward()
    test_naive_and_sdpa_match()
    test_causal_mask()

    print("✓ Attention tests passed")