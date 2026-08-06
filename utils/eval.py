import torch


@torch.no_grad()
def estimate_loss(
        model,
        train_dataset,
        val_dataset,
        eval_iters,
        batch_size,
        device
):
    """
    Estimate average train and validation loss.

    Returns
    -------
    {
        "train": float,
        "val": float,
    }
    """

    losses = {}

    model.eval()


    for split, dataset in (
                            ("train", train_dataset),
                            ("val", val_dataset)
                        ):
        split_losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            x, y, loss_mask = dataset.get_batch(
                batch_size=batch_size,
                device=device
            )

            _, loss = model(
                x,
                targets=y,
                loss_mask=loss_mask
            )


            split_losses[k] = loss.item()

        losses[split] = split_losses.mean().item()
    model.train()

    return losses
        

