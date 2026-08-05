import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention
from models.layers import FeedForward, RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(
            dim=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.block_size,
            bias=config.bias,
            attn_impl=config.attn_impl,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = FeedForward(
            dim=config.d_model,
            hidden_dim=config.d_model * config.ffn_mult,
            bias=config.bias,
        )
        # self.dropout = nn.Dropout(config.dropout)
    def forward(self, x, cache=None, start_pos = 0):

        x = x + self.attn(
            self.attn_norm(x),
            cache=cache,
            start_pos=start_pos,
        )
        x = x + self.ffn(self.ffn_norm(x))

        return x