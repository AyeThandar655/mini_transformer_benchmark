"""Training loop for the mini-Transformer classifier.

The ``train_model`` function is written as a library function so it
can also be called by ``benchmark.py`` to train many variants in a loop.
Running ``python train.py`` directly trains a single configuration
and prints the results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import load_dataloaders
from model import MiniTransformerClassifier
from utils import binary_accuracy, count_parameters, set_seed, timer


@dataclass
class TrainConfig:
    """Hyper-parameters for a single training run."""

    alias: str = "A"
    use_positional_encoding: bool = True
    num_heads: int = 4
    num_layers: int = 1
    d_model: int = 64
    d_ff: int = 128
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 15
    seed: int = 42
    device: str = "cpu"


@dataclass
class TrainResult:
    """Outputs of a single training run."""

    alias: str
    config: TrainConfig
    param_count: int
    train_time_sec: float
    best_val_acc: float
    test_acc: float
    train_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    final_epoch: int = 0


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Run one pass over ``loader``; updates weights if optimizer is given."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for tokens, mask, labels in loader:
        tokens = tokens.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(tokens, attention_mask=mask)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_seen += labels.size(0)

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Compute (loss, accuracy) on a data loader without training."""
    return _run_epoch(model, loader, optimizer=None, criterion=criterion, device=device)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train_model(
    config: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    verbose: bool = True,
) -> TrainResult:
    """Train one model configuration and return its metrics."""
    set_seed(config.seed)
    device = config.device

    model = MiniTransformerClassifier(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_positional_encoding=config.use_positional_encoding,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses: List[float] = []
    val_accs: List[float] = []
    best_val = 0.0
    best_state: Dict[str, torch.Tensor] = {
        k: v.detach().clone() for k, v in model.state_dict().items()
    }

    with timer() as t:
        for epoch in range(1, config.epochs + 1):
            train_loss, train_acc = _run_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_accs.append(val_acc)

            if val_acc > best_val:
                best_val = val_acc
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }

            if verbose:
                print(
                    f"  epoch {epoch:02d}/{config.epochs}  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                )

    train_time = t["elapsed"]

    # Restore best-on-val weights for the test-set evaluation
    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, test_loader, criterion, device)

    result = TrainResult(
        alias=config.alias,
        config=config,
        param_count=count_parameters(model),
        train_time_sec=train_time,
        best_val_acc=best_val,
        test_acc=test_acc,
        train_losses=train_losses,
        val_accs=val_accs,
        final_epoch=config.epochs,
    )

    if verbose:
        print(
            f"  -> alias={config.alias}  best_val={best_val:.4f}  "
            f"test_acc={test_acc:.4f}  params={result.param_count}  "
            f"time={train_time:.1f}s"
        )

    return result


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    cfg = TrainConfig(alias="demo")
    train_loader, val_loader, test_loader = load_dataloaders(
        data_dir=data_dir, batch_size=cfg.batch_size
    )
    train_model(cfg, train_loader, val_loader, test_loader, verbose=True)
