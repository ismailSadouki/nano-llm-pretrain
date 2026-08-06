import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.lr_scheduler import get_lr


def test_warmup():

    lr = get_lr(
        0,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_iters=100,
        lr_decay_iters=1000,
    )

    assert lr > 0
    assert lr < 1e-3


def test_end_of_warmup():

    lr = get_lr(
        100,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_iters=100,
        lr_decay_iters=1000,
    )

    assert abs(lr - 1e-3) < 1e-9


def test_decay():

    lr = get_lr(
        500,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_iters=100,
        lr_decay_iters=1000,
    )

    assert 1e-4 < lr < 1e-3


def test_after_decay():

    lr = get_lr(
        2000,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_iters=100,
        lr_decay_iters=1000,
    )

    assert lr == 1e-4


if __name__ == "__main__":

    test_warmup()
    test_end_of_warmup()
    test_decay()
    test_after_decay()

    print("✓ LR scheduler tests passed")