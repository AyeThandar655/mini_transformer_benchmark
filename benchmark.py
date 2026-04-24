"""Run the small benchmark required by the assignment.

This script trains several model variants sequentially and writes:

    results/benchmark.csv        - one row per variant with all metrics
    results/benchmark.md         - human-readable Markdown table
    figures/train_loss.png       - training loss curves
    figures/val_accuracy.png     - validation accuracy curves

Variants (aliases match the ones used in the report):

    A - positional encoding ON,  1 head,  1 layer   (baseline)
    B - positional encoding ON,  4 heads, 1 layer   (more heads)
    C - positional encoding OFF, 4 heads, 1 layer   (no PE ablation)
    D - positional encoding ON,  4 heads, 2 layers  (deeper)
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from data import load_dataloaders
from train import TrainConfig, TrainResult, train_model
from utils import format_time, set_seed


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def build_configs() -> List[TrainConfig]:
    """Return the four variants"""
    base = dict(
        d_model=64,
        d_ff=128,
        dropout=0.1,
        batch_size=32,
        learning_rate=1e-3,
        epochs=15,
        seed=42,
        device="cpu",
    )
    return [
        TrainConfig(
            alias="A",
            use_positional_encoding=True,
            num_heads=1,
            num_layers=1,
            **base,
        ),
        TrainConfig(
            alias="B",
            use_positional_encoding=True,
            num_heads=4,
            num_layers=1,
            **base,
        ),
        TrainConfig(
            alias="C",
            use_positional_encoding=False,
            num_heads=4,
            num_layers=1,
            **base,
        ),
        TrainConfig(
            alias="D",
            use_positional_encoding=True,
            num_heads=4,
            num_layers=2,
            **base,
        ),
    ]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _write_csv(results: List[TrainResult], path: Path) -> None:
    fieldnames = [
        "alias",
        "positional_encoding",
        "num_heads",
        "num_layers",
        "d_model",
        "d_ff",
        "dropout",
        "epochs",
        "param_count",
        "best_val_acc",
        "test_acc",
        "train_time_sec",
        "train_time_human",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            cfg = r.config
            writer.writerow(
                {
                    "alias": r.alias,
                    "positional_encoding": "Yes" if cfg.use_positional_encoding else "No",
                    "num_heads": cfg.num_heads,
                    "num_layers": cfg.num_layers,
                    "d_model": cfg.d_model,
                    "d_ff": cfg.d_ff,
                    "dropout": cfg.dropout,
                    "epochs": cfg.epochs,
                    "param_count": r.param_count,
                    "best_val_acc": round(r.best_val_acc, 4),
                    "test_acc": round(r.test_acc, 4),
                    "train_time_sec": round(r.train_time_sec, 2),
                    "train_time_human": format_time(r.train_time_sec),
                }
            )


def _write_markdown(results: List[TrainResult], path: Path) -> None:
    lines = [
        "# Benchmark Results",
        "",
        "| Model | Positional Encoding | Heads | Layer | Val Acc | Test Acc | Train Time | Params |",
        "|-------|---------------------|-------|-------|---------|----------|------------|--------|",
    ]
    for r in results:
        cfg = r.config
        lines.append(
            f"| {r.alias} | "
            f"{'Yes' if cfg.use_positional_encoding else 'No'} | "
            f"{cfg.num_heads} | "
            f"{cfg.num_layers} | "
            f"{r.best_val_acc:.4f} | "
            f"{r.test_acc:.4f} | "
            f"{format_time(r.train_time_sec)} | "
            f"{r.param_count:,} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_json(results: List[TrainResult], path: Path) -> None:
    payload = []
    for r in results:
        payload.append(
            {
                "alias": r.alias,
                "config": asdict(r.config),
                "param_count": r.param_count,
                "best_val_acc": r.best_val_acc,
                "test_acc": r.test_acc,
                "train_time_sec": r.train_time_sec,
                "train_losses": r.train_losses,
                "val_accs": r.val_accs,
            }
        )
    path.write_text(json.dumps(payload, indent=2))


def _plot_training_curves(results: List[TrainResult]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Training loss curves
    plt.figure(figsize=(7, 4.5))
    for r in results:
        epochs = range(1, len(r.train_losses) + 1)
        plt.plot(epochs, r.train_losses, marker="o", label=f"Model {r.alias}")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (cross-entropy)")
    plt.title("Training loss per epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "train_loss.png", dpi=150)
    plt.close()

    # Validation accuracy curves
    plt.figure(figsize=(7, 4.5))
    for r in results:
        epochs = range(1, len(r.val_accs) + 1)
        plt.plot(epochs, r.val_accs, marker="o", label=f"Model {r.alias}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy per epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "val_accuracy.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> List[TrainResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    configs = build_configs()

    # Build loaders once (they are cheap but let's avoid repeating disk I/O).
    train_loader, val_loader, test_loader = load_dataloaders(
        data_dir=DATA_DIR, batch_size=configs[0].batch_size
    )

    results: List[TrainResult] = []
    for cfg in configs:
        print(f"\n=== Training variant {cfg.alias} ===")
        print(
            f"  PE={cfg.use_positional_encoding}  heads={cfg.num_heads}  "
            f"layers={cfg.num_layers}  d_model={cfg.d_model}  "
            f"d_ff={cfg.d_ff}  epochs={cfg.epochs}"
        )
        result = train_model(
            cfg,
            train_loader,
            val_loader,
            test_loader,
            verbose=True,
        )
        results.append(result)

    _write_csv(results, RESULTS_DIR / "benchmark.csv")
    _write_markdown(results, RESULTS_DIR / "benchmark.md")
    _write_json(results, RESULTS_DIR / "benchmark.json")
    _plot_training_curves(results)

    print("\n=== Summary ===")
    for r in results:
        print(
            f"  {r.alias}: val={r.best_val_acc:.4f}  test={r.test_acc:.4f}  "
            f"params={r.param_count}  time={format_time(r.train_time_sec)}"
        )
    print(f"\nArtefacts written to:\n  {RESULTS_DIR}\n  {FIGURES_DIR}")

    return results


if __name__ == "__main__":
    main()
