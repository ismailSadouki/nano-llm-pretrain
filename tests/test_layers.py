import pytest

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.layers import RMSNorm, FeedForward


def test_rmsnorm():
    B, S, C = 2, 8, 512

    x = torch.randn(B, S, C)

    norm = RMSNorm(C)

    y = norm(x)

    # Shape is preserved
    assert y.shape == x.shape

    # Dtype is preserved
    assert y.dtype == x.dtype

    # No NaNs
    assert not torch.isnan(y).any()

    # Output should differ from input
    assert not torch.equal(x, y)


def test_feedforward():
    B, S, C = 2, 8, 512

    x = torch.randn(B, S, C)

    ffn = FeedForward(
        dim=C,
        hidden_dim=4 * C,
    )

    y = ffn(x)

    # Shape is preserved
    assert y.shape == x.shape

    # Dtype is preserved
    assert y.dtype == x.dtype

    # No NaNs
    assert not torch.isnan(y).any()

    # Output should differ from input
    assert not torch.equal(x, y)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_feedforward_autocast():
    model = FeedForward(dim=512, hidden_dim=4 * 512).cuda()

    x = torch.randn(2, 8, 512, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)

    assert y.shape == x.shape
    assert not torch.isnan(y).any()

if __name__ == "__main__":
    test_rmsnorm()
    test_feedforward()
    test_feedforward_autocast()
    print("✓ RMSNorm tests passed")
    print("✓ FeedForward tests passed")
    print("✓ FeedForward autocast tests passed")