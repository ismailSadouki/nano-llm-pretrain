import sys
from pathlib import Path

import torch

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

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=False,      # CPU -> no fp16 scaling
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

    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    assert torch.isfinite(grad_norm)

    scaler.step(optimizer)
    scaler.update()

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
        amp_dtype=None,      # CPU -> fp32
    )

    assert isinstance(losses["train"], float)
    assert isinstance(losses["val"], float)

def test_optimizer_lr_update():

    model = build_model()

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    new_lr = 5e-4

    for group in optimizer.param_groups:
        group["lr"] = new_lr

    assert optimizer.param_groups[0]["lr"] == new_lr
    assert optimizer.param_groups[1]["lr"] == new_lr

if __name__ == "__main__":
    test_single_training_step()
    test_lr_scheduler()
    test_estimate_loss()
    test_optimizer_lr_update()

    print("✓ Training tests passed.")