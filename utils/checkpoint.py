import os
import torch




def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    scaler,
    step,
    best_val_loss,
    model_config,
    train_config,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (
                scaler.state_dict()
                if scaler is not None and scaler.is_enabled()
                else None
            ),  

        "step": step,
        "best_val_loss": best_val_loss,

        "model_config": model_config,
        "train_config": train_config,

        "torch_rng": torch.get_rng_state(), # CPU reproducibility

        "cuda_rng": ( # GPU reproducibility
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
                ),
        }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)




def load_checkpoint(
    path,
    *,
    model,
    optimizer=None,
    scaler=None,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
         weights_only=False,
    )
    model.load_state_dict(
        checkpoint["model"]
    )
    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    if (
        scaler is not None
        and checkpoint["scaler"] is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    if "torch_rng" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_rng"]
        )

    
    if (
        torch.cuda.is_available()
        and checkpoint.get("cuda_rng") is not None
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng"]
        )


    return {
        "step": checkpoint["step"],
        "best_val_loss": checkpoint["best_val_loss"],
        "model_config": checkpoint["model_config"],
        "train_config": checkpoint["train_config"],
    }