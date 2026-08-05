from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class GPTConfig:

    vocab_size: int = 16000
    block_size: int = 1024 # context_length / max_seq_len

    n_layers: int = 12

    n_heads: int = 12
    n_kv_heads: int

    d_model: int = 768 # n_embd

    ffn_mult: int

    dropout: float = 0.0

    tie_embeddings: bool

    bias: bool


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config


    def forward(self, input_ids, targets=None):
        pass

    def get_num_parameters(self, non_embedding=False):
        """
            Return the number of parameters in the model.
            For non-embedding count (default), the position embeddings get subtracted.
            The token embeddings would too, except due to the parameter sharing these
            params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.toke_emb.weight.numel() # check nanoGPT for this line?

        return n_params
            





