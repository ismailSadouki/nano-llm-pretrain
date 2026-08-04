import torch
import torch.nn as nn
import torch.nn.functional as F








def precompute_rope_cos_sin(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    position_offset: int = 0,
):
    assert head_dim % 2 == 0, "head_dim must be even."

    # ω_i = θ^{-2i/D}
    # Shape: (D/2,)
    inv_freq = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        )
    )

    # RoPE is position dependent, Θ_{p,i} = pω_i its shape is (S, D/2). where p is token position
    positions = (
        torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )
        + position_offset
    ) # shape is (S,)
    # Note: Θ_{p,i} = pω_i is the outer product. every row corresponds to one position, and every column corresponds to one frequency.
    # freqs = positions[:, None] * inv_freq[None, :]
    freqs = torch.outer(positions, inv_freq) # This is Θ with shape (S, D/2)

    # we have [S,D/2] we want [S, D] matching the last dim of q in [B, H, S, D]
    emb = torch.cat((freqs, freqs), dim=-1) # shape is (S, D)

    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def rotate_half(x):
    """
    - R(θ)x=xcosθ+rotate(x)sinθ.
    - this is the rotate(x) part (x_a, x_b) -> (-x_b, x_a)
    - takes [B, H, S, D] and returns another tensor of exactly the same shape.
    - Only the last dimension participates in the rotation.
    """
    assert x.shape[-1] % 2 == 0
    D = x.shape[-1]

    x1 = x[..., : D//2]
    x2 = x[..., D//2 :]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
):
    assert q.shape == k.shape
    assert cos.shape == sin.shape
    assert q.shape[-1] == cos.shape[-1]
    assert q.shape[-2] == cos.shape[-2]

    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)

    cos = cos.unsqueeze(0).unsqueeze(0) # [1,1,S,D]
    sin = sin.unsqueeze(0).unsqueeze(0) # [1,1,S,D]

    q_embed = (
        q * cos
        +
        rotate_half(q) * sin
    )

    k_embed = (
        k * cos
        +
        rotate_half(k) * sin
    )

    return q_embed, k_embed


# ============================================================================
# Meta Llama complex-number implementation
# ============================================================================


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    position_offset: int = 0,
):
    assert head_dim % 2 == 0, "head_dim must be even."

    # ω_i = θ^{-2i/D}
    # Shape: (D/2,)
    inv_freq = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        )
    )

    # RoPE is position dependent, Θ_{p,i} = pω_i its shape is (S, D/2). where p is token position
    positions = (
    torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )
        + position_offset
    ) # shape is (S,)

    # Note: Θ_{p,i} = pω_i is the outer product. every row corresponds to one position, and every column corresponds to one frequency.
    freqs = torch.outer(positions, inv_freq) # This is Θ with shape (S, D/2)

    # Convert Θ into complex exponentials:
    # e^{iΘ} = cos(Θ) + i sin(Θ)
    freqs_cis = torch.polar(
        torch.ones_like(freqs),
        freqs,
    )

    return freqs_cis


def reshape_for_broadcast(
    freqs_cis: torch.Tensor,
    x: torch.Tensor,
):
    """
    Reshape freqs_cis for broadcasting.

    freqs_cis:
        [S, D/2]

    x:
        [B, S, H, D/2]

    output:
        [1, S, 1, D/2]
    """

    assert freqs_cis.shape == (x.shape[2], x.shape[-1])

    shape = [
        d if i == 2 or i == x.ndim - 1 else 1
        for i, d in enumerate(x.shape)
    ]

    return freqs_cis.view(*shape)

def apply_rotary_emb_complex(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    assert q.shape == k.shape
    assert q.shape[-1] % 2 == 0

    D = q.shape[-1]

    # Convert from HF layout:
    # [x0 x1 ... x_{D/2-1} | x_{D/2} ... x_{D-1}]
    #
    # into interleaved layout:
    # [x0 x_{D/2} x1 x_{D/2+1} ...]
    #
    # so that Meta's complex multiplication rotates the same pairs
    # as rotate_half().

    def interleave(x):
        x1 = x[..., : D // 2]
        x2 = x[..., D // 2 :]

        return torch.stack((x1, x2), dim=-1).flatten(-2)

    q = interleave(q)
    k = interleave(k)

    q_complex = torch.view_as_complex(
        q.float().reshape(*q.shape[:-1], -1, 2)
    )

    k_complex = torch.view_as_complex(
        k.float().reshape(*k.shape[:-1], -1, 2)
    )

    freqs_cis = freqs_cis.to(
        device=q.device,
        dtype=q_complex.dtype,
    )

    freqs_cis = reshape_for_broadcast(
        freqs_cis,
        q_complex,
    )

    q_out = torch.view_as_real(
        q_complex * freqs_cis
    ).flatten(3)

    k_out = torch.view_as_real(
        k_complex * freqs_cis
    ).flatten(3)

    # Undo the interleaving so output matches HF layout

    def deinterleave(x):
        x = x.reshape(*x.shape[:-1], D // 2, 2)

        x1 = x[..., 0]
        x2 = x[..., 1]

        return torch.cat((x1, x2), dim=-1)

    q_out = deinterleave(q_out)
    k_out = deinterleave(k_out)

    return (
        q_out.type_as(q),
        k_out.type_as(k),
    )