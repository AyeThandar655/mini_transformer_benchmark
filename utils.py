"""Utility helpers: reproducibility, timing, and metric helpers.

This module centralises small helper functions used across the project
so that `train.py` and `benchmark.py` stay focused on the actual
learning / evaluation logic.
"""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make the run as reproducible as possible.

    Python random, NumPy, and PyTorch (CPU + CUDA) are all seeded.
    Deterministic cuDNN flags are also set, but note that some ops
    on GPU can still be non-deterministic.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@contextmanager
def timer() -> Iterator[dict]:
    """Simple context manager that records elapsed wall time in seconds."""
    state: dict = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - start


def format_time(seconds: float) -> str:
    """Format seconds as `<m> min <s> s` for human-readable output."""
    minutes = int(seconds // 60)
    secs = seconds - 60 * minutes
    return f"{minutes} min {secs:.1f} s"


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy from raw binary-classification logits (shape [N, 2])."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float().sum().item()
    return correct / max(labels.numel(), 1)
