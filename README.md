# Linear Regression Benchmark

This benchmark compares linear regression training across C, BLAS, PyTorch, and naive Python.

## Model

```text
y_pred = Xw + b
loss = mean((y_pred - y)^2)
```

## Benchmark Config

```text
N = 100000
D = 100
EPOCHS = 5000
RUNS = 10
CPU only
```

For fair CPU comparison:

```python
torch.set_num_threads(1)
```

For BLAS:

```bash
OPENBLAS_NUM_THREADS=1 ./linreg_blas
```

## Implementations

| File | Description |
|---|---|
| `linreg_blas.c` | C implementation using BLAS/OpenBLAS for `Xw` and `X^T error` |
| `linreg_naive.c` | Naive C implementation using manual loops |
| `linreg_pytorch.py` | PyTorch implementation using `torch.nn.Linear` |
| `linreg_manual_pytorch.py` | PyTorch implementation using manual tensors/autograd |
| `linreg_naive.py` | Pure Python loops; too slow at this scale |

## CPU Results

| Rank | Implementation | Avg Time (s) | Best (s) | Worst (s) | Time/Epoch (ms) | Samples/sec |
|---:|---|---:|---:|---:|---:|---:|
| 1 | C + BLAS | 13.9480 | 13.6906 | 14.8688 | 2.7896 | 35,847,332 |
| 2 | Naive C | 15.4578 | 15.1607 | 15.9135 | 3.0916 | 32,346,178 |
| 3 | PyTorch `nn.Linear` | 18.3733 | 16.9909 | 22.2292 | 3.6747 | 27,213,402 |
| 4 | Manual PyTorch | 19.2326 | 17.1299 | 20.4122 | 3.8465 | 25,997,567 |
| 5 | Naive Python | Too slow | — | — | — | — |

## Ranking

1. C + BLAS
2. Naive C
3. PyTorch `nn.Linear`
4. Manual PyTorch
5. Naive Python
