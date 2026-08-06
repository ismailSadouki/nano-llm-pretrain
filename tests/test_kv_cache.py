import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.kv_cache import KVCache
from models.model import GPTConfig, GPTModel


def build_model():

    config = GPTConfig(
        vocab_size=100,
        block_size=32,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
    )

    model = GPTModel(config)
    model.eval()

    return model, config


def build_caches(model, config, batch_size=1):

    return [
        KVCache(
            batch_size=batch_size,
            max_seq_len=config.block_size,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.d_model // config.n_heads,
            device=model.tok_embeddings.weight.device,
            dtype=model.tok_embeddings.weight.dtype,
        )
        for _ in range(config.n_layers)
    ]


@torch.inference_mode()
def test_cached_vs_uncached_logits_equivalence():

    torch.manual_seed(0)

    model, config = build_model()

    #
    # Random prompt
    #

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 8),
    )

    #
    # ----------------------------------------------------
    # Full forward
    # ----------------------------------------------------
    #

    full_logits = model(input_ids)

    #
    # ----------------------------------------------------
    # Cached forward
    # ----------------------------------------------------
    #

    caches = build_caches(model, config)

    # Fill the KV cache with the first T-1 tokens
    model(
        input_ids[:, :-1],
        caches=caches,
        start_pos=0,
    )

    # Decode only the final token using the cache
    cached_logits = model(
        input_ids[:, -1:],
        caches=caches,
        start_pos=input_ids.size(1) - 1,
    )

    #
    # ----------------------------------------------------
    # The logits of the final token must be identical
    # ----------------------------------------------------
    #

    assert torch.allclose(
        full_logits[:, -1, :],
        cached_logits[:, -1, :],
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == "__main__":

    test_cached_vs_uncached_logits_equivalence()

    print("✓ Cached vs uncached logits equivalence test passed")