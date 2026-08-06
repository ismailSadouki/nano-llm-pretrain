import math


def get_lr(
        step: int,
        *,
        learning_rate: float,
        min_lr: float,
        warmup_iters: int,
        lr_decay_iters: int,
):
    """
    Warmup + cosine decay

    Returns the learning rate for the current optimizer step.
    """

    # linear warmup
    if step < warmup_iters:
        # This increases the learning rate linearly from a very small value to the target learning rate.
        return learning_rate * (step + 1) / warmup_iters

    # Finished cosine decay
    if step >= lr_decay_iters:
        return min_lr

    # cosine decay
    decay_ratio = ( # Compute how far you are through the decay, So it always goes from 0 to 1.
         step - warmup_iters
     ) / (
         lr_decay_iters - warmup_iters
     )

    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * ( # smoothly decreases from 1 to 0.
         1.0 + math.cos(math.pi * decay_ratio)
     )

    return min_lr + coeff * (
        learning_rate - min_lr
     )