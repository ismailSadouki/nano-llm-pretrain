from dataclasses import dataclass
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import DecoderBlock
from models.layers import RMSNorm

@dataclass
class GPTConfig:

    vocab_size: int = 16000
    block_size: int = 1024

    n_layers: int = 12

    d_model: int = 768

    n_heads: int = 12
    n_kv_heads: int = 4

    ffn_mult: int = 4

    dropout: float = 0.0

    tie_embeddings: bool = True
    bias: bool = False

    attn_impl: str = "naive"


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(config)
            for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight



        # Weight initialization
        self.apply(self._init_weights)
        if config.tie_embeddings:
            assert self.lm_head.weight is self.tok_embeddings.weight
        # report number of parameters
        # print(f"Number of parameters: {self.get_num_parameters():,}")
        n_params = self.get_num_parameters()

        print(
            f"""
        Architecture
        ------------
        Layers:        {config.n_layers}
        Hidden dim:    {config.d_model}
        Heads:         {config.n_heads}
        KV heads:      {config.n_kv_heads}
        FFN mult:      {config.ffn_mult}

        Parameters:
        {n_params:,}
        """
        )

        



    def forward(self, input_ids, targets=None, loss_mask=None, caches=None, start_pos=0):
        B, S = input_ids.shape
        assert S <= self.config.block_size

        x = self.tok_embeddings(input_ids) # [B,S] to [B,S,C]

        for i, block in enumerate(self.layers):
            cache = None if caches is None else caches[i]

            x = block(
                x,
                cache=cache,
                start_pos=start_pos
            )


        x = self.norm(x)
        logits = self.lm_head(x) # [B,S,C] to [B,S,V]

        if targets is None: # inference
            return logits # should add None for loss??
        
        # loss = F.cross_entropy(
        #     logits.reshape(-1, self.config.vocab_size), # [B,S,V] -> [B*S, V]
        #     targets.reshape(-1) # [B,S] -> [B*S]
        # )

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            reduction="none",
        )

        if loss_mask is not None:
            loss = loss[loss_mask.reshape(-1)]

        assert loss.numel() > 0, "All tokens are masked."

        loss = loss.mean()

        return logits, loss 


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_( # GPT-2 initialization.
                module.weight,
                mean=0.0,
                std=0.02
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )
        
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str = "cuda",
    ):
        param_dict = {
            name: p
            for name, p in self.named_parameters()
            if p.requires_grad
        }

        decay_params = [
            p 
            for p in param_dict.values()
            if p.dim() >= 2
        ]
        no_decay_params = [
            p
            for p in param_dict.values()
            if p.dim() < 2
        ]


        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0
            }
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)

        print(
            f"Decay params: {len(decay_params)} tensors, {num_decay:,} parameters"
        )

        print(
            f"No decay params: {len(no_decay_params)} tensors, {num_no_decay:,} parameters"
        )

        # Check if AdamW supports fused implementation
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters

        use_fused = fused_available and device_type == "cuda"

        extra_args = {"fused": True} if use_fused else {}

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )

        print(f"Using fused AdamW: {use_fused}")
        return optimizer


    def get_num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )
            





