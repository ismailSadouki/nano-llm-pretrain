import shutil
import sys
from pathlib import Path

from utils.logger import JSONLLogger
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.checkpoint import (save_checkpoint, load_checkpoint)


import yaml
import torch
import argparse
from datetime import datetime

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
        scaler,
        start_step,
        best_val_loss,
        logger,
        run_dir
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step = start_step


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




            save_checkpoint(
                path=run_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                best_val_loss=best_val_loss,
                model_config=model.config,
                train_config=config,
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                save_checkpoint(
                    path=run_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    best_val_loss=best_val_loss,
                    model_config=model.config,
                    train_config=config

                )


                print(f"Saved best checkpoint at step {step}")
            peak_memory = None


            if device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2

                print(
                    f"Peak memory: {peak_memory:.1f} MB"
                    )



            logger.log(
                step=step,
                train_loss=losses["train"],
                val_loss=losses["val"],
                learning_rate=lr,
                grad_norm=float(grad_norm),
                best_val_loss=best_val_loss,
                peak_memory_mb=peak_memory,
            )                

        step += 1




def main():


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )

    args = parser.parse_args()


    # Create run directory
    if args.resume is not None:
        run_dir = Path(args.resume).parent
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / run_name

    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is None:
        shutil.copy(
            args.config,
            run_dir / "train.yaml",
        )
        
    logger = JSONLLogger(
        run_dir / "log.jsonl"
    )

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

    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler
        )

        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["best_val_loss"]

        print(
            f"Resumed training from step {start_step} "
            f"(best val = {best_val_loss:.4f})"
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
                scaler=scaler,
                start_step=start_step,
                best_val_loss=best_val_loss,
                logger=logger,
                run_dir=run_dir,
            )

    except KeyboardInterrupt:
        print("Training interrupted.")



if __name__ == "__main__":
    main()










