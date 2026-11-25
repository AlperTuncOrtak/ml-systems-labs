# Lab 5: Frameworks and Automatic Differentiation

## 1. Goal

Build a scalar automatic-differentiation engine that records a computation graph during the forward pass and backpropagates through it, check it against PyTorch autograd and numerical gradients, use it to train a tiny model on make_moons and compare its speed with PyTorch, then use PyTorch itself to inspect a graph, compare eager with compiled or traced execution, and measure the cost of non-contiguous tensors.

## 2. Method

- **Environment:** same pinned environment as Lab 1 (`results/versions.txt`), single-threaded CPU (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), Apple Silicon MacBook Pro. All four seeds set to 42.
- **Value node** (`src/engine.py`): stores `data`, `grad`, `_parents` (the nodes it was computed from) and `_backward` (its local derivative rule). Every operation creates a new node linked to its inputs, so running the forward pass builds the graph.
- **Operations and local derivatives:** addition (∂/∂a = 1, ∂/∂b = 1), multiplication (∂/∂a = b, ∂/∂b = a), constant power (k·aᵏ⁻¹), ReLU (1 if a > 0 else 0). Sigmoid (s(1 − s)) and log (1/a) were added for the BCE loss; subtraction, negation and division are built from the operations above.
- **Backward pass:** depth-first topological sort of the graph, all gradients reset to 0, output gradient set to 1, then each node's rule is applied in reverse topological order. Rules use `+=`, so a node used several times receives the sum of all contributions.
- **Gradient check** (`src/check_grads.py`): `((a*b + c)**2).relu().sum()` with a, b, c random 3×4 matrices (36 inputs). Compared with PyTorch autograd in float64 and with central-difference numerical gradients (ε = 1e-5). Extra check: d(x·x)/dx at x = 3 must be 6, which only holds if gradients accumulate.
- **make_moons** (`src/train_moons.py`): `make_moons(n_samples=1000, noise=0.1, random_state=42)`, 70/30 stratified split. Model 2 → 8 (ReLU) → 1 (sigmoid), BCE loss, full-batch gradient descent, lr 0.5, 30 epochs. Our engine and PyTorch (float64) start from the same weights and run the same algorithm. Timing: 1 warm-up, then median of 5 full training runs.
- **Execution modes** (`src/execution_modes.py`): a 64 → 128 → 128 → 10 MLP. Graph inspected with `torch.fx.symbolic_trace` and the output's `grad_fn` chain. Eager vs `torch.compile` vs `torch.jit.trace` at batch 1 and 256, under `torch.no_grad()`, 10 warm-up calls, median of 100 runs.
- **Contiguity:** a 2048×2048 float32 tensor and its transpose view `x.t()`. Shape, dtype, stride and contiguity printed; three operations timed on both (median of 100 runs, 20 for matmul).

## 3. Results

### Gradient check (`results/gradient_check.txt`)

| Input | Max abs. difference vs PyTorch | Max abs. difference vs numerical | Pass (< 1e-6) |
|---|---|---|---|
| a | 0 | 9.9e-11 | Yes |
| b | 0 | 1.2e-10 | Yes |
| c | 0 | 7.8e-11 | Yes |

Forward value 13.5184535429 in both. The graph for 12 elements has 97 nodes. Accumulation check: d(x·x)/dx = 6.0 at x = 3.

### make_moons training (`results/moons_speed.csv`)

| Implementation | Median training time (5 runs) | Final train loss | Test accuracy |
|---|---|---|---|
| Our engine | 10.28 s | 0.340795 | 0.8567 |
| PyTorch | 0.0023 s | 0.340795 | 0.8567 |

PyTorch is about 4,500× faster for the same result.

### Graph inspection (`results/execution_modes.txt`)

The FX graph lists the forward operations in order: `fc1 → relu → fc2 → relu_1 → fc3`. The autograd graph read from the output backwards is `SumBackward0 → AddmmBackward0 → ReluBackward0 → AddmmBackward0 → ReluBackward0 → AddmmBackward0 → TBackward0`: the same structure as our engine, one node per operation with its own derivative rule, but each node works on a whole tensor.

### Eager vs traced (`results/eager_vs_compiled.csv`)

`torch.compile` failed on this machine: the inductor backend could not compile its generated C++ code with the system `clang++` (PyTorch 2.3.1 with a newer macOS toolchain). As the lab allows, `torch.jit.trace` was used instead.

| Batch | Eager (ms) | jit.trace (ms) | Speed-up |
|---|---|---|---|
| 1 | 0.0105 | 0.0063 | 1.68× |
| 256 | 0.0391 | 0.0342 | 1.14× |

### Tensor representation and contiguity (`results/contiguity.csv`)

| Tensor | Shape | dtype | Stride | Contiguous | Shares memory with x |
|---|---|---|---|---|---|
| x | (2048, 2048) | float32 | (2048, 1) | True | — |
| x.t() | (2048, 2048) | float32 | (1, 2048) | False | Yes |

| Operation | Contiguous (ms) | Non-contiguous (ms) | Slowdown |
|---|---|---|---|
| Elementwise x * 2 + 1 | 0.575 | 0.578 | 1.01× |
| Row sum (dim=1) | 0.175 | 0.327 | 1.87× |
| Matmul t @ x | 8.827 | 12.510 | 1.42× |

Making `x.t()` contiguous (a full copy) takes 1.886 ms.

## 4. Conclusions

1. **Automatic differentiation is only local rules plus graph order.** Each operation knows one simple derivative, and the topological sort makes sure a node's gradient is complete before it is passed to its parents. With these two pieces our engine matches PyTorch exactly (difference 0) and the numerical estimate to about 1e-10, and gives the same training result on make_moons. Accumulating with `+=` is essential: any value used more than once, such as a weight shared by 700 training samples, needs the sum of all its gradient contributions.
2. **The framework's value is in the system, not the math.** Both implementations compute the same function and gradients, but ours creates a Python object and a function call for every scalar operation, tens of thousands per epoch. PyTorch represents the whole batch as tensors and runs a few optimised C++ kernels. That difference alone makes it about 4,500× faster on a 33-parameter model.
3. **When compilation helps.** Tracing or compiling removes per-operation Python overhead: the graph is captured once and replayed without the interpreter, and a compiler can also fuse operations. This helps most when the work per operation is small, such as small models, batch size 1 or many cheap element-wise ops: here jit.trace gave 1.68× at batch 1 but only 1.14× at batch 256, where matrix multiplication dominates. It helps little for large models whose time is already in big optimised kernels, costs time up front, and needs a working toolchain: `torch.compile` failed here because of a compiler incompatibility. Memory layout matters too: a transposed view is free to create but made the row sum 1.87× and the matmul 1.42× slower, while the element-wise op was unaffected because PyTorch kept the input's memory order. Calling `.contiguous()` once (1.9 ms) is worth it when a tensor is read many times.
