import os
os.environ["OMP_NUM_THREADS"] = "1"
import random, statistics, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.set_num_threads(1)
RESULTS = Path(__file__).resolve().parents[1] / "results"


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def median_ms(fn, x, warmup=10, repeats=100):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


lines = []
def log(s=""):
    print(s); lines.append(s)

model = SmallMLP().eval()

# 1) graph inspection
log("== FX graph (operations PyTorch records for this model) ==")
gm = torch.fx.symbolic_trace(model)
log(str(gm.graph))
log("== Autograd graph: grad_fn chain from the output back to the input ==")
out = model(torch.randn(4, 64)).sum()
fn, depth = out.grad_fn, 0
while fn is not None and depth < 12:
    log("  " * depth + type(fn).__name__)
    nxt = [f for f, _ in fn.next_functions if f is not None and "AccumulateGrad" not in type(f).__name__]
    fn, depth = (nxt[0] if nxt else None), depth + 1

# 2) eager vs traced vs compiled
rows = []
for batch in [1, 256]:
    x = torch.randn(batch, 64)
    variants = {"eager": model}
    with torch.no_grad():
        variants["jit.trace"] = torch.jit.trace(model, x)
    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled(x)                       # triggers compilation
        variants["torch.compile"] = compiled
    except Exception as e:
        log(f"torch.compile unsupported here: {type(e).__name__}: {str(e)[:120]}")
    base = None
    for name, fn in variants.items():
        ms = median_ms(fn, x)
        base = base or ms
        rows.append({"batch": batch, "mode": name, "median_ms": round(ms, 4),
                     "speedup_vs_eager": round(base / ms, 2)})
modes = pd.DataFrame(rows)
modes.to_csv(RESULTS / "eager_vs_compiled.csv", index=False)
log("\n== Eager vs compiled (median of 100 runs after warm-up) ==")
log(modes.to_string(index=False))

# 3) tensor representation and 4) contiguity cost
x = torch.randn(2048, 2048)
xt = x.t()                                    # a view: same memory, swapped strides
log("\n== Tensor properties ==")
for name, t in [("x", x), ("x.t()", xt)]:
    log(f"{name:6} shape={tuple(t.shape)} dtype={t.dtype} stride={t.stride()} "
        f"contiguous={t.is_contiguous()} shares_memory_with_x={t.data_ptr() == x.data_ptr()}")

ops = {"elementwise x * 2 + 1": lambda t: t * 2 + 1,
       "row sum (dim=1)": lambda t: t.sum(dim=1),
       "matmul t @ x": lambda t: t @ x}
crow = []
for op_name, op in ops.items():
    reps = 20 if "matmul" in op_name else 100
    c = median_ms(op, x, warmup=3, repeats=reps)
    n = median_ms(op, xt, warmup=3, repeats=reps)
    crow.append({"operation": op_name, "contiguous_ms": round(c, 3),
                 "non_contiguous_ms": round(n, 3), "slowdown": round(n / c, 2)})
copy_ms = median_ms(lambda t: t.contiguous(), xt, warmup=3, repeats=50)
contig = pd.DataFrame(crow)
contig.to_csv(RESULTS / "contiguity.csv", index=False)
log("\n== Contiguous vs non-contiguous (2048 x 2048 float32) ==")
log(contig.to_string(index=False))
log(f"cost of making x.t() contiguous (copy): {copy_ms:.3f} ms")
(RESULTS / "execution_modes.txt").write_text("\n".join(lines) + "\n")
