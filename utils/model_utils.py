def count_parameters(module, trainable_only=False):
    if trainable_only:
        return sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )

    return sum(
        p.numel()
        for p in module.parameters()
    )