import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import PackedDataset


def build_dataset(split="train"):
    return PackedDataset(split)


def test_dataset_lengths():

    dataset = build_dataset()

    assert len(dataset.input_ids) == len(dataset.labels)
    assert len(dataset.labels) == len(dataset.loss_mask)


def test_len():

    dataset = build_dataset()

    assert len(dataset) == len(dataset.input_ids)


def test_batch_shapes():

    dataset = build_dataset()

    x, y, loss_mask = dataset.get_batch(
        batch_size=8,
        device="cpu",
    )

    assert x.shape == y.shape
    assert y.shape == loss_mask.shape

    assert x.ndim == 2
    assert y.ndim == 2
    assert loss_mask.ndim == 2


def test_batch_dtypes():

    dataset = build_dataset()

    x, y, loss_mask = dataset.get_batch(
        batch_size=8,
        device="cpu",
    )

    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert loss_mask.dtype == torch.bool


def test_batch_device():

    dataset = build_dataset()

    x, y, loss_mask = dataset.get_batch(
        batch_size=4,
        device="cpu",
    )

    assert x.device.type == "cpu"
    assert y.device.type == "cpu"
    assert loss_mask.device.type == "cpu"


def test_memmap_used():

    dataset = build_dataset()

    assert isinstance(dataset.input_ids, np.memmap)
    assert isinstance(dataset.labels, np.memmap)
    assert isinstance(dataset.loss_mask, np.memmap)


if __name__ == "__main__":

    test_dataset_lengths()
    test_len()
    test_batch_shapes()
    test_batch_dtypes()
    test_batch_device()
    test_memmap_used()

    print("✓ PackedDataset tests passed")