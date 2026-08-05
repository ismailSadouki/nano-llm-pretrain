import torch
import torch.nn as nn
import torch.nn.functional as F



class KVCache:
    def __init__(
            self,
            batch_size: int,
            max_seq_len: int,
            n_kv_heads: int,
            head_dim: int,
            device=None,
            dtype=torch.float32,
    ):
        self.k = torch.zeros( # [B, Hkv, T, D]
            batch_size,
            n_kv_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v = torch.zeros_like(self.k)
        