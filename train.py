import yaml
import torch
import argparse

from models.model import GPTModel, GPTConfig
from utils.data import PackedDataset
from utils.eval import estimate_loss
from utils.lr_scheduler import get_lr
from pathlib import Path



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
        device
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0
    best_val_loss = float("inf")

    while step < config['max_iters']: # one optimizer step corresponds to one effective batch.

        # Gradient accumulation
        for micro_step in range(config["gradient_accumulation_steps"]):
            x, y, loss_mask = train_dataset.get_batch(
                batch_size=config["batch_size"],
                device=device
            )

            _, loss = model(
                x,
                targets=y,
                loss_mask=loss_mask
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")

            
            loss = loss / config["gradient_accumulation_steps"]

            # train_loss = loss.item() * config["gradient_accumulation_steps"]

            loss.backward()

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



        optimizer.step()
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
                }

                torch.save(
                    checkpoint,
                    "checkpoints/best.pt",
                )
                print(f"Saved checkpoint at step {step}")


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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
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
                device=device
            )

    except KeyboardInterrupt:
        print("Training interrupted.")



if __name__ == "__main__":
    main()










