import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import apply_rotary_pos_emb, precompute_rope_cos_sin




class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        bias: bool = False,
        attn_impl: str = "naive",
        
    ):
        super().__init__()

        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.head_dim = dim // n_heads #  D=C/H
        self.n_rep = n_heads // n_kv_heads # H/Hkv

        self.scale = self.head_dim ** -0.5
        self.attn_impl = attn_impl

        self.q_proj = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=bias
        )
        self.k_proj = nn.Linear(
            dim,
            n_kv_heads * self.head_dim,
            bias=bias
        )
        self.v_proj = nn.Linear(
            dim,
            n_kv_heads * self.head_dim,
            bias=bias
        )

        self.out_proj = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=bias
        )

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        cos, sin = precompute_rope_cos_sin( # [max_seq_len, D], # [max_seq_len, D]
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
        )
        self.register_buffer(
            "cos_cached",
            cos,
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            sin,
            persistent=False,
        )

    def forward(self, x: torch.Tensor): # x : [B, S, C]
        B, S, C = x.shape

        assert C == self.dim


        q = self.q_proj(x) # [B, S, HxD]
        k = self.k_proj(x) # [B, S, HkvxD]
        v = self.v_proj(x) # [B, S, HkvxD]

        q = q.view(B, S, self.n_heads, self.head_dim) # [B, S, H, D]
        k = k.view( B, S, self.n_kv_heads, self.head_dim ) # [B, S, Hkv, D]
        v = v.view( B, S, self.n_kv_heads,self.head_dim) # [B, S, Hkv, D]

        q = q.transpose(1, 2) # [B, H, S, D]
        k = k.transpose(1, 2) # [B, Hkv, S, D]
        v = v.transpose(1, 2) # [B, Hkv, S, D]


        # RoPE
        cos = self.cos_cached[:S]
        sin = self.sin_cached[:S]
        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
        )


        k = repeat_kv(k, self.n_rep) # [B, H, S, D]
        v = repeat_kv(v, self.n_rep) # [B, H, S, D]


        if self.attn_impl == "naive":
            out = self.naive_attention(q, k, v)
        elif self.attn_impl == "sdpa":
            out = self.sdpa_attention(q, k, v)
        else:
            raise ValueError(
                f"Unknown attention implementation: {self.attn_impl}"
            )


        return out

    def naive_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        B, _, S, _ = q.shape
        scores = ( # [B, H, S, S]
            q @ k.transpose(-2, -1)
        ) * self.scale

        mask = self.causal_mask[:S, :S]
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(q)

        out = attn_weights @ v # [B, H, S, D]
        out = out.transpose(1, 2) # [B, S, H, D]
        out = out.reshape(B, S, self.dim) # [B, S, C]
        out = self.out_proj(out)


        return out

    def sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        B, _, S, _ = q.shape
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            # dropout_p = self.dropout if self.training else 0.0
            is_causal=True,
        )

        out = out.transpose(1, 2)
        out = out.reshape(B, S, self.dim)
        out = self.out_proj(out)

        return out
        



def repeat_kv(x: torch.Tensor, n_rep: int = 1): 
    """
    x: [B, Hkv, S, D]

    returns:
    x: [B, H, S, D]
    """
    B, Hkv, S, D = x.shape

    if n_rep == 1:
        return x

    x = x[:, :, None, :, :] # [B, Hkv, 1, S, D]
    x = x.expand(B, Hkv, n_rep, S, D) # [B, Hkv, n_rep, S, D]
    x = x.reshape(B, Hkv*n_rep, S, D) # [B, Hkv*n_rep, S, D]


    return x



        