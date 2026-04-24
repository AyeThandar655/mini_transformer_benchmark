# Mini Transformer Benchmark

A from-scratch PyTorch implementation of a small Transformer encoder for a
synthetic sequence-classification task, with a four-variant benchmark.

## Task

Given a sequence of tokens from the vocabulary
`{PAD=0, A=1, B=2, C=3, D=4}`, padded to length 20, predict whether the
**first non-padding token appears again in the second half** of the
non-padding portion of the sequence.

Split definition (from the assignment tips): if the non-padding length is
`L`, then `mid = L // 2`, `first_half = seq[:mid]`, `second_half = seq[mid:]`.

The label is 1 if `seq[0]` occurs in `second_half`, otherwise 0.

## Restrictions (honoured in `model.py`)

I may use `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, and
general PyTorch autograd/optimisation primitives.

## Project layout

```
mini_transformer_benchmark/
├── README.md              # this file
├── data.py                # CSV loaders and Dataset
├── model.py               # mini Transformer encoder (from scratch)
├── train.py               # single-run training loop
├── benchmark.py           # trains all 4 variants, writes tables + plots
├── utils.py               # seeding, timing, metric helpers
├── report.pdf             # short report
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── results/
│   ├── benchmark.csv
│   ├── benchmark.md
│   ├── benchmark.json
│   └── run_log.txt
└── figures/
    ├── train_loss.png
    └── val_accuracy.png
```

## Quick start

```bash
pip install -r requirements.txt
python benchmark.py
```

The benchmark runs end-to-end on CPU in well under a minute.

## Variants

| Alias | Positional Encoding | Heads | Layers | Purpose                       |
|-------|---------------------|-------|--------|-------------------------------|
| A     | Yes                 | 1     | 1      | Baseline                      |
| B     | Yes                 | 4     | 1      | Effect of more attention heads|
| C     | No                  | 4     | 1      | Ablation of positional encoding|
| D     | Yes                 | 4     | 2      | Effect of depth               |

## Hyperparameters (fixed across variants)

| Parameter        | Value |
|------------------|-------|
| `d_model`        | 64    |
| `d_ff`           | 128   |
| `dropout`        | 0.1   |
| `batch_size`     | 32    |
| `learning_rate`  | 1e-3  |
| `epochs`         | 15    |
| `optimizer`      | Adam  |
| `seed`           | 42    |

## Results (see `results/benchmark.md` for the generated file)

| Model | PE  | Heads | Layers | Val Acc | Test Acc | Train Time | Params |
|-------|-----|-------|--------|---------|----------|------------|--------|
| A     | Yes | 1     | 1      | 0.9050  | 0.8800   | 5.9 s      | 34,050 |
| B     | Yes | 4     | 1      | 0.9890  | 0.9730   | 8.5 s      | 34,050 |
| C     | No  | 4     | 1      | 0.8140  | 0.8230   | 8.5 s      | 34,050 |
| D     | Yes | 4     | 2      | 0.9870  | 0.9800   | 15.9 s     | 67,522 |

Training curves are saved to `figures/train_loss.png` and
`figures/val_accuracy.png`.

**Note on reproducibility across machines.** PyTorch guarantees
bit-for-bit reproducibility only on the same machine, BLAS library,
and PyTorch version. With seed = 42 fixed, numbers on Linux/x86 can
differ by up to \~1–2% from numbers on macOS/Apple-Silicon because the
underlying BLAS (MKL/OpenBLAS vs Accelerate) rounds differently. The
qualitative conclusions — C collapses to the majority baseline, B and
D are the strongest — are stable across platforms.

## Reproducibility

`utils.set_seed(42)` is called at the start of every run. It seeds
Python's `random`, NumPy, and PyTorch (CPU and CUDA), and sets
deterministic cuDNN flags. All hyperparameters are listed above and in
`benchmark.py::build_configs`.

## Academic integrity

All model code is written in `model.py`. No prebuilt Transformer
modules are used.
