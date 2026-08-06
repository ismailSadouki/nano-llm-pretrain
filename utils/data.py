import numpy as np
import torch 

class PackedDataset:
    """
    Loads packed training examples using NumPy memmaps.

    input_ids : [N, S]
    labels    : [N, S]
    loss_mask : [N, S]

    get_batch returns:

        x         [B, S]
        y         [B, S]
        loss_mask [B, S]
    """
    def __init__(self, split="train"):
        self.input_ids = np.load(
            f"data/packed/{split}_input_ids.npy",
            mmap_mode="r"
        )

        self.labels = np.load(
            f"data/packed/{split}_labels.npy",
            mmap_mode="r"
        )

        self.loss_mask = np.load(
            f"data/packed/{split}_loss_mask.npy",
            mmap_mode="r"
        )


        assert len(self.input_ids) == len(self.labels), "input_ids and labels should have the same length"
        assert len(self.labels) == len(self.loss_mask), "labels and loss_mask should have the same length"

    def __len__(self):
        return len(self.input_ids)
    
    def get_batch(
            self,
            batch_size,
            device
    ):
        idx = np.random.randint(
            0,
            len(self),
            size=batch_size,
        )
        x = torch.from_numpy(
            np.asarray(self.input_ids[idx])
        ).long()

        y = torch.from_numpy(
            np.asarray(self.labels[idx])
        ).long()

        loss_mask = torch.from_numpy(
            np.asarray(self.loss_mask[idx])
        ).bool()

        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
            loss_mask = loss_mask.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
            loss_mask = loss_mask.to(device)

        assert x.shape == y.shape
        assert x.shape == loss_mask.shape
        assert x.dtype == torch.long
        assert y.dtype == torch.long
        assert loss_mask.dtype == torch.bool

        return x, y, loss_mask
    