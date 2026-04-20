"""Dataset loading for the mini-Transformer benchmark.

The provided CSV files (train.csv, validation.csv, test.csv) contain
one row per sequence with the following relevant columns:

    seq_len              - true (non-padding) length of the sequence
    label                - binary classification target (0 or 1)
    token_01 .. token_20 - padded token IDs (PAD=0, A=1, B=2, C=3, D=4)
    mask_01  .. mask_20  - attention mask (1 = real token, 0 = PAD)

This module exposes:

* ``SequenceDataset`` - a thin ``torch.utils.data.Dataset`` wrapper that
  returns ``(tokens, mask, label)`` tensors.
* ``load_dataloaders`` - convenience builder that returns the three
  PyTorch ``DataLoader`` objects used in training / evaluation.

The label-generation rule is documented in the assignment brief:

    Label = 1 if the first non-padding token of the sequence
    appears again in the second half of the non-padding portion.

The CSV files already contain the label, so we simply read it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Vocabulary constants (kept here so the rest of the project can import them)
PAD_ID: int = 0
VOCAB = {"PAD": 0, "A": 1, "B": 2, "C": 3, "D": 4}
VOCAB_SIZE: int = len(VOCAB)  # 5
MAX_LEN: int = 20             # as specified in the brief

TOKEN_COLS = [f"token_{i:02d}" for i in range(1, MAX_LEN + 1)]
MASK_COLS = [f"mask_{i:02d}" for i in range(1, MAX_LEN + 1)]


class SequenceDataset(Dataset):
    """PyTorch ``Dataset`` built from one of the provided CSV files."""

    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path)
        self.tokens = torch.tensor(df[TOKEN_COLS].to_numpy(), dtype=torch.long)
        self.mask = torch.tensor(df[MASK_COLS].to_numpy(), dtype=torch.long)
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.mask[idx], self.labels[idx]


def load_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test ``DataLoader`` objects from the CSV files."""
    data_dir = Path(data_dir)

    train_ds = SequenceDataset(data_dir / "train.csv")
    val_ds = SequenceDataset(data_dir / "validation.csv")
    test_ds = SequenceDataset(data_dir / "test.csv")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check when run directly.
    tr, va, te = load_dataloaders("data", batch_size=4)
    x, m, y = next(iter(tr))
    print("tokens :", x.shape, x.dtype)
    print("mask   :", m.shape, m.dtype)
    print("labels :", y.shape, y.dtype)
    print("sample :", x[0].tolist())
    print("mask   :", m[0].tolist())
    print("label  :", y[0].item())
