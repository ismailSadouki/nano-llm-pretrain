import torch
from models.model import GPTModel, GPTConfig

def test_logits_shape():
    config = GPTConfig(
        vocab_size = 16000,
        block_size = 128,
        n_emb = 256
    )
    model = GPTModel(config)

    x = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(x)

    assert logits.shape == (2, 16, config.vocab_size)


def test_parameter_count():
    config = GPTConfig(
        vocab_size=16000,
        block_size=128,
        n_embd=256,
    )

    model = GPTModel(config)

    n = model.get_num_params()

    # Later you'll tighten it: assert 18_000_000 < n < 25_000_000
    # assert model.config.block_size == 128
    assert n > 0