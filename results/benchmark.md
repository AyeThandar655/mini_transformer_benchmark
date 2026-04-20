# Benchmark Results

| Model | Positional Encoding | Heads | Layer | Val Acc | Test Acc | Train Time | Params |
|-------|---------------------|-------|-------|---------|----------|------------|--------|
| A | Yes | 1 | 1 | 0.9050 | 0.8800 | 0 min 5.9 s | 34,050 |
| B | Yes | 4 | 1 | 0.9890 | 0.9730 | 0 min 8.5 s | 34,050 |
| C | No | 4 | 1 | 0.8140 | 0.8230 | 0 min 8.5 s | 34,050 |
| D | Yes | 4 | 2 | 0.9870 | 0.9800 | 0 min 15.9 s | 67,522 |
