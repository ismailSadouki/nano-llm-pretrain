import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from models.model import GPTModel, GPTConfig
from utils.data import PackedDataset
from utils.eval import estimate_loss
from utils.lr_scheduler import get_lr


def build_model():
    config = GPTConfig(
        vocab_size=16000,
        block_size=1024,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=4,
    )
    return GPTModel(config)


def test_single_training_step():

    device = torch.device("cpu")

    model = build_model().to(device)

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type=device.type,
    )

    dataset = PackedDataset("train")

    optimizer.zero_grad(set_to_none=True)

    x, y, loss_mask = dataset.get_batch(
        batch_size=2,
        device=device,
    )

    _, loss = model(
        x,
        targets=y,
        loss_mask=loss_mask,
    )

    assert torch.isfinite(loss)

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    assert torch.isfinite(grad_norm)

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)


def test_lr_scheduler():

    lr = get_lr(
        step=0,
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup_iters=100,
        lr_decay_iters=1000,
    )

    assert lr > 0

    lr_end = get_lr(
        step=2000,
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup_iters=100,
        lr_decay_iters=1000,
    )

    assert lr_end == 3e-5


def test_estimate_loss():

    device = torch.device("cpu")

    model = build_model().to(device)

    train_dataset = PackedDataset("train")
    val_dataset = PackedDataset("val")

    losses = estimate_loss(
        model,
        train_dataset,
        val_dataset,
        eval_iters=2,
        batch_size=2,
        device=device,
    )

    assert "train" in losses
    assert "val" in losses

    assert isinstance(losses["train"], float)
    assert isinstance(losses["val"], float)


if __name__ == "__main__":

    test_single_training_step()
    test_lr_scheduler()
    test_estimate_loss()

    print("✓ Training tests passed.")