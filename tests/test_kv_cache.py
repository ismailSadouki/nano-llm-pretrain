import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.kv_cache import KVCache
from models.model import GPTConfig, GPTModel


def test_cached_logits_match_uncached():
    torch.manual_seed(0)

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

    # Sequence length = 8
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 8),
    )

    #
    # -------- Full Forward --------
    #

    full_logits = model(input_ids)

    #
    # -------- Cached Forward --------
    #

    caches = [
        KVCache(
            batch_size=1,
            max_seq_len=config.block_size,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.d_model // config.n_heads,
            device=input_ids.device,
            dtype=model.tok_embeddings.weight.dtype,
        )
        for _ in range(config.n_layers)
    ]

    #
    # Feed first 7 tokens
    #

    _ = model(
        input_ids[:, :-1],
        caches=caches,
        start_pos=0,
    )

    #
    # Feed only last token
    #

    cached_logits = model(
        input_ids[:, -1:],
        caches=caches,
        start_pos=input_ids.size(1) - 1,
    )

    #
    # Compare logits for the last token
    #

    assert torch.allclose(
        full_logits[:, -1],
        cached_logits[:, -1],
        atol=1e-5,
    )