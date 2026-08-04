import torch


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.rope import (
    precompute_rope_cos_sin,
    precompute_freqs_cis,
    apply_rotary_pos_emb,
    apply_rotary_emb_complex,
)


def test_rope_shapes():

    B = 2
    H = 8
    S = 16
    D = 64

    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)


    cos, sin = precompute_rope_cos_sin(
        head_dim=D,
        max_seq_len=S,
    )


    q_rot, k_rot = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
    )


    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape



def test_rope_no_nan():

    q = torch.randn(2,8,16,64)
    k = torch.randn(2,8,16,64)

    cos, sin = precompute_rope_cos_sin(
        64,
        16,
    )

    q_rot, k_rot = apply_rotary_pos_emb(
        q,k,cos,sin
    )

    assert not torch.isnan(q_rot).any()
    assert not torch.isnan(k_rot).any()

def test_rope_preserves_norm():

    B = 2
    H = 8
    S = 16
    D = 64

    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)

    cos, sin = precompute_rope_cos_sin(
        head_dim=D,
        max_seq_len=S,
    )

    q_rot, k_rot = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
    )

    q_norm = torch.linalg.norm(q, dim=-1)
    q_rot_norm = torch.linalg.norm(q_rot, dim=-1)

    assert torch.allclose(
        q_norm,
        q_rot_norm,
        atol=1e-5,
    )

    k_norm = torch.linalg.norm(k, dim=-1)
    k_rot_norm = torch.linalg.norm(k_rot, dim=-1)

    assert torch.allclose(
        k_norm,
        k_rot_norm,
        atol=1e-5,
    )


def test_hf_and_meta_rope_equivalence():

    B = 2
    H = 8
    S = 16
    D = 64

    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)

    cos, sin = precompute_rope_cos_sin(
        head_dim=D,
        max_seq_len=S,
    )

    freqs_cis = precompute_freqs_cis(
        head_dim=D,
        max_seq_len=S,
    )

    q_hf, k_hf = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
    )

    q_meta, k_meta = apply_rotary_emb_complex(
        q,
        k,
        freqs_cis,
    )

    assert torch.allclose(
        q_hf,
        q_meta,
        atol=1e-5,
    )

    assert torch.allclose(
        k_hf,
        k_meta,
        atol=1e-5,
    )


if __name__ == "__main__":
    test_rope_shapes()
    test_rope_no_nan()
    test_rope_preserves_norm()
    test_hf_and_meta_rope_equivalence()
    print("✓ Rope tests passed")
    