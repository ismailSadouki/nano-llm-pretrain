import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import yaml
import torch
import argparse

from models.model import GPTModel, GPTConfig
from utils.data import PackedDataset
from contextlib import nullcontext
from utils.eval import estimate_loss
from utils.lr_scheduler import get_lr


from utils.device import get_device_and_dtype


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # Allow TensorFloat-32 on Ampere+ GPUs (e.g. Kaggle T4)
        # Faster matrix multiplications with negligible accuracy loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def build_model(config, device):
    model_config = GPTConfig(**config["model"])

    model = GPTModel(model_config).to(device)

    return model


def build_dataset():
    train_dataset = PackedDataset("train")

    val_dataset = PackedDataset("val")

    return train_dataset, val_dataset




def train(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        config,
        device,
        amp_dtype,
        scaler
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step = 0
    best_val_loss = float("inf")


    while step < config['max_iters']: # one optimizer step corresponds to one effective batch.



        # Gradient accumulation
        for micro_step in range(config["gradient_accumulation_steps"]):
            x, y, loss_mask = train_dataset.get_batch(
                batch_size=config["batch_size"],
                device=device
            )



            if device.type == "cuda" and amp_dtype is not None:
                ctx = torch.autocast(
                            device_type="cuda",
                            dtype=amp_dtype,
                        )
            else:
                ctx = nullcontext()

            with ctx:
                _, loss = model(
                    x,
                    targets=y,
                    loss_mask=loss_mask
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")

            
            loss = loss / config["gradient_accumulation_steps"]

            # train_loss = loss.item() * config["gradient_accumulation_steps"]

            scaler.scale(loss).backward()


        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["grad_clip"]
        )
        # Optimizer step
        lr = get_lr(
            step,
            learning_rate=config["learning_rate"],
            min_lr=config["min_lr"],
            warmup_iters=config["warmup_iters"],
            lr_decay_iters=config["lr_decay_iters"],
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr




        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)


        # Logging / evaluation
        if step % config["eval_interval"] == 0:
            losses = estimate_loss(
                model,
                train_dataset,
                val_dataset,
                eval_iters=config["eval_iters"],
                batch_size=config["batch_size"],
                device=device,
                amp_dtype=amp_dtype
            )
            print(
                f"step {step:6d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm:.3f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "model_config": model.config,
                    "train_config": config,
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                }

                torch.save(
                    checkpoint,
                    "checkpoints/best.pt",
                )
                print(f"Saved checkpoint at step {step}")

                if device.type == "cuda":
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

                    print(
                        f"Peak memory: {peak_memory:.1f} MB"
                    )
                    # torch.cuda.reset_peak_memory_stats()

        step += 1




def main():
    Path("checkpoints").mkdir(parents=True, exist_ok=True)


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config["seed"])

    device, amp_dtype, use_scaler = get_device_and_dtype(
        requested_dtype=config["dtype"]
    )

    print(f"Device      : {device}")

    print(
        "AMP dtype   :",
        amp_dtype if amp_dtype is not None else "fp32"
    )

    print(
        "GradScaler  :",
        "enabled" if use_scaler else "disabled"
    )


    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_scaler
    )


    model = build_model(
        config=config,
        device=device,
    )

    optimizer = model.configure_optimizers(
        weight_decay=config["weight_decay"],
        learning_rate=config["learning_rate"],
        betas=config["betas"],
        device_type=device.type
    )

    train_dataset, val_dataset = build_dataset()


    try:
        train(
                model=model,
                optimizer=optimizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=config,
                device=device,
                amp_dtype=amp_dtype,
                scaler=scaler
            )

    except KeyboardInterrupt:
        print("Training interrupted.")



if __name__ == "__main__":
    main()










