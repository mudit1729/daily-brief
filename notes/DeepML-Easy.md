# Deep-ML Easy Problems: Comprehensive Guide

A complete guide covering 37 ML problems with theory, implementation, and PyTorch integration. Essential fundamentals (1-23) and advanced interview topics (24-37).

## Table of Contents

- [Quick Reference Table](#quick-reference-table)
- [Linear Algebra & Matrix Operations](#linear-algebra--matrix-operations)
- [Activation Functions](#activation-functions)
- [Regression Models](#regression-models)
- [Appendix: Einstein Notation (einsum) Tutorial](#appendix-einstein-notation-einsum-tutorial)
- [Classification & Metrics](#classification--metrics)
- [Data Processing & Utilities](#data-processing--utilities)
- [Advanced ML Topics](#advanced-ml-topics)

---

## Quick Reference Table

| # | Problem | Category | Key Concept | Time Complexity | PyTorch Equivalent |
|---|---------|----------|-------------|----------------|-------------------|
| 1 | Matrix × Vector | Linear Algebra | Dot product | O(mn) | `torch.matmul()` |
| 2 | Matrix Transpose | Linear Algebra | Index swap | O(mn) | `.T` or `.transpose()` |
| 3 | Scalar Multiplication | Linear Algebra | Element-wise | O(mn) | `tensor * scalar` |
| 4 | Vector to Diagonal | Linear Algebra | Identity scaling | O(n) | `torch.diag()` |
| 5 | Reshape Matrix | Linear Algebra | Memory reorg | O(1) | `.view()` or `.reshape()` |
| 6 | Linear Kernel | Linear Algebra | Inner product | O(n) | `torch.dot()` |
| 7 | Basis Transformation | Linear Algebra | Change of basis | O(n³) | `torch.linalg.solve()` |
| 8 | ReLU | Activation | max(0, x) | O(n) | `F.relu()` |
| 9 | Leaky ReLU | Activation | Negative slope | O(n) | `F.leaky_relu()` |
| 10 | Sigmoid | Activation | Logistic function | O(n) | `torch.sigmoid()` |
| 11 | Softmax | Activation | Exp normalization | O(n) | `F.softmax()` |
| 12 | Log Softmax | Activation | Numerically stable | O(n) | `F.log_softmax()` |
| 13 | Normal Equation | Regression | Analytical solution | O(n³) | `torch.linalg.lstsq()` |
| 14 | Gradient Descent | Regression | Iterative optimization | O(kn²) | `optimizer.step()` |
| 15 | Ridge Regression Loss | Regression | L2 regularization | O(n) | `F.mse_loss() + penalty` |
| 16 | Precision | Metrics | TP/(TP+FP) | O(n) | `torchmetrics.Precision()` |
| 17 | Accuracy | Metrics | Correct/Total | O(n) | `(pred == target).float().mean()` |
| 18 | Single Neuron | Classification | Weighted sum | O(n) | `nn.Linear(n, 1)` |
| 19 | Mean by Row/Column | Metrics | Average | O(n) | `.mean(dim=...)` |
| 20 | Batch Iterator | Data Processing | Chunking | O(n) | `DataLoader` |
| 21 | Random Shuffle | Data Processing | Permutation | O(n) | `torch.randperm()` |
| 22 | Feature Scaling | Data Processing | Normalization | O(n) | `F.normalize()` |
| 23 | One-Hot Encoding | Data Processing | Category encoding | O(n) | `F.one_hot()` |
| 24 | Cross-Entropy Loss | Loss Functions | KL divergence | O(n) | `F.cross_entropy()` |
| 25 | Hinge Loss | Loss Functions | SVM margin | O(n) | Custom/`MarginRankingLoss` |
| 26 | Huber Loss | Loss Functions | Robust regression | O(n) | `F.huber_loss()` |
| 27 | Adam Optimizer | Optimization | Adaptive learning | O(n) | `torch.optim.Adam()` |
| 28 | SGD with Momentum | Optimization | Accelerated gradient | O(n) | `torch.optim.SGD()` |
| 29 | RMSprop Optimizer | Optimization | Root mean square | O(n) | `torch.optim.RMSprop()` |
| 30 | Backpropagation | Neural Networks | Chain rule | O(n) | `.backward()` |
| 31 | AUC-ROC Curve | Metrics | Threshold trade-off | O(n log n) | `torchmetrics.AUROC()` |
| 32 | F1 Score | Metrics | Harmonic mean | O(n) | Custom/`torchmetrics.F1()` |
| 33 | Confusion Matrix | Metrics | Classification counts | O(n) | `torchmetrics.ConfusionMatrix()` |
| 34 | Recall/Sensitivity | Metrics | TP/(TP+FN) | O(n) | `torchmetrics.Recall()` |
| 35 | PCA | Dimensionality Reduction | Eigendecomposition | O(n²) | Custom/sklearn |
| 36 | Dropout | Regularization | Random masking | O(n) | `F.dropout()` |
| 37 | Batch Normalization | Normalization | Feature standardization | O(n) | `F.batch_norm()` |

---

# Linear Algebra & Matrix Operations

## 1. Matrix × Vector Multiplication

Matrix-vector product computes: `(A @ b)_i = sum(a_ij * b_j)` for each row i

For each row of A, compute dot product with vector b.

**Implementation:**
```python
def matrix_dot_vector(a: list[list[float]], b: list[float]) -> list[float]:
    """Compute matrix-vector product: A @ b."""
    if len(a[0]) != len(b):
        raise ValueError(f"Incompatible dimensions")
    return [sum(row[i] * b[i] for i in range(len(b))) for row in a]

# NumPy: np.dot(a, b) or a @ b
```

**PyTorch:**
```python
import torch
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([5.0, 6.0])
result = torch.matmul(A, b)  # or A @ b
result = torch.mv(A, b)  # matrix-vector specific

# Batch operations: automatic broadcasting
A_batch = torch.randn(32, 10, 5)
b_batch = torch.randn(32, 5)
result_batch = torch.matmul(A_batch, b_batch)  # (32, 10)

# Supports gradients for backprop
A = A.requires_grad_(True)
result = A @ b
loss = result.sum()
loss.backward()  # Computes A.grad
```

**Best Practices:**
- Use `torch.matmul()` or `@` operator (100x+ faster than manual loops)
- Batch operations automatically broadcast; check shapes with `.shape`
- Verify dimension compatibility: `A.shape[-1]` must equal `b.shape[0]`

**Common Applications:**
- Neural network forward pass (weights @ inputs)
- Batch matrix-vector products in transformers
- Kernel computations in SVM and other algorithms

---

## 2. Matrix Transpose

Swap rows and columns: `A_T[i,j] = A[j,i]`

**Implementation:**
```python
def transpose(a: list[list[int | float]]) -> list[list[int | float]]:
    """Compute transpose by swapping indices."""
    m, n = len(a), len(a[0])
    result = [[0] * m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            result[j][i] = a[i][j]
    return result

# NumPy: a.T or np.transpose(a)
```

**PyTorch:**
```python
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
A_T = A.T  # NumPy-style
A_T = A.transpose(0, 1)  # Explicit dimensions
A_T = A.permute(2, 1, 0)  # For higher dimensions

# Key nuance: .T creates a view (shares memory), not contiguous
print(A.T.is_contiguous())  # False
A_T_contiguous = A.T.contiguous()  # Make contiguous copy

# Batch transpose
batch = torch.randn(32, 10, 5)
batch_T = batch.transpose(1, 2)  # (32, 5, 10)
```

**Best Practices:**
- `.T` creates a view, not a copy (memory efficient)
- Call `.contiguous()` before operations requiring contiguity
- For >2D tensors, prefer `.transpose(dim1, dim2)` over `.T`

**Common Applications:**
- Converting weight matrices in neural networks
- Computing Gram matrices (X @ X.T)
- Switching between row-major and column-major layouts

---

## 3. Scalar Multiplication of Matrix

Multiply all elements: `(c * A)[i,j] = c * a[i,j]`

**Implementation:**
```python
def scalar_multiply(matrix: list[list[float]], scalar: float) -> list[list[float]]:
    """Multiply matrix by scalar element-wise."""
    return [[element * scalar for element in row] for row in matrix]

# NumPy: matrix * scalar
```

**PyTorch:**
```python
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
result = A * 2.5  # Element-wise
A.mul_(2.5)  # In-place (more efficient)

# Broadcasting handles batches automatically
batch = torch.randn(32, 10, 5)
scaled = batch * 0.5

# Practical: learning rate scaling
gradients = torch.randn(100, 10)
update = gradients * 0.01  # Learning rate
```

**Best Practices:**
- Use in-place `*=` or `.mul_()` to save memory on large tensors
- Broadcasting applies automatically; no need for explicit expansion
- Scale gradients before applying to weights (optimizer does this internally)

**Common Applications:**
- Gradient scaling in learning rate updates
- Batch processing with per-batch scaling factors
- Normalization and denormalization of data

---

## 4. Convert Vector to Diagonal Matrix

Create diagonal matrix from vector:

```
diag(v) = [[v_1,   0,   0],
            [  0, v_2,   0],
            [  0,   0, v_3]]
```

**Implementation:**
```python
def make_diagonal(x: list[int | float]) -> list[list[int | float]]:
    """Create diagonal matrix from vector."""
    n = len(x)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        result[i][i] = x[i]
    return result

# NumPy: np.diag(x)
```

**PyTorch:**
```python
v = torch.tensor([1.0, 2.0, 3.0])
D = torch.diag(v)  # Shape: (3, 3)

# Extract diagonal from matrix
D = torch.diag(torch.tensor([1., 2., 3.]))
extracted = torch.diag(D)  # Back to vector

# Batch diagonal matrices
batch_v = torch.randn(32, 10)
batch_D = torch.diag_embed(batch_v)  # (32, 10, 10)

# Diagonal offset
D_offset = torch.diag(v, diagonal=1)  # Superdiagonal

# Gradient flow works
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
D = torch.diag(v)
loss = D.sum()
loss.backward()  # v.grad = [1., 1., 1.]
```

**Best Practices:**
- Use `torch.diag_embed()` for batch operations (not `torch.diag()` in a loop)
- For scaling: `D @ x` is faster than explicit matrix multiplication when D is diagonal
- Check matrix is square when extracting diagonal with `torch.diag(matrix)`

**Common Applications:**
- Scaling transformations and affine operations
- Covariance matrix diagonalization
- Efficient matrix-vector products when matrices are diagonal

---

## 5. Reshape Matrix

Reshape from (m, n) to (p, q) where m×n = p×q

**Implementation:**
```python
def reshape_matrix(a: list[list[int | float]], new_shape: tuple[int, int]) -> list[list[int | float]]:
    """Reshape matrix to new dimensions."""
    m, n = len(a), len(a[0])
    new_m, new_n = new_shape
    if m * n != new_m * new_n:
        raise ValueError(f"Cannot reshape {m}×{n} to {new_m}×{new_n}")

    flat = [element for row in a for element in row]
    result = []
    for i in range(new_m):
        result.append(flat[i * new_n : (i + 1) * new_n])
    return result

# NumPy: a.reshape(new_shape)
```

**PyTorch:**
```python
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = A.reshape(3, 2)  # Returns view when possible, copy otherwise
B = A.view(3, 2)  # Requires contiguous, stricter
B = A.reshape(-1, 4)  # Auto-infer dimension: (2, 3) -> (2, 6)

# View vs Reshape distinction
A = torch.randn(2, 3).T  # Not contiguous
# A.view(3, 2)  # Error!
B = A.reshape(3, 2)  # Works, creates copy if needed

# Practical: flatten for fully-connected layer
images = torch.randn(64, 1, 28, 28)
flat = images.view(64, -1)  # (64, 784)

# Gradient flow: A.grad shape matches original A
```

**Best Practices:**
- Use `.reshape()` when contiguity is uncertain (handles both cases)
- Use `.view()` only when you know tensor is contiguous (faster)
- The `-1` automatically infers missing dimension
- Always verify total elements match: `m*n == p*q`

**Common Applications:**
- Converting image tensors (batch, channels, height, width) to flat vectors
- Preparing data for fully-connected layers after conv layers
- Batch processing with flexible dimensions

---

## 6. Linear Kernel

Dot product between vectors: `K(x, y) = x^T @ y = sum(x_i * y_i)`

**Implementation:**
```python
def linear_kernel(x: list[float], y: list[float]) -> float:
    """Compute linear kernel (dot product)."""
    if len(x) != len(y):
        raise ValueError("Vectors must have same length")
    return sum(xi * yi for xi, yi in zip(x, y))

# NumPy: np.dot(x, y)
```

**PyTorch:**
```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
result = torch.dot(x, y)  # 32.0
result = x @ y  # Same

# Batch dot products
x_batch = torch.randn(32, 10)
y_batch = torch.randn(32, 10)
dots = (x_batch * y_batch).sum(dim=1)  # (32,)
# or: torch.einsum('bi,bi->b', x_batch, y_batch)

# Gram matrix (pairwise kernels)
X = torch.randn(100, 20)
K = X @ X.T  # (100, 100) - all pairwise kernels

# Gradient computation
x = x.requires_grad_(True)
y = y.requires_grad_(True)
kernel = x @ y
kernel.backward()
# x.grad = y, y.grad = x

# Attention mechanism pattern
queries = torch.randn(32, 10, 64)
keys = torch.randn(32, 10, 64)
attention = torch.matmul(queries, keys.transpose(1, 2))  # (32, 10, 10)
```

**Best Practices:**
- For 1D vectors, use `torch.dot()` (most efficient)
- For batch operations, use element-wise multiplication + `.sum(dim=1)`
- Gram matrices scale as O(n²) memory; be careful with large datasets
- Use `torch.einsum()` for complex multi-dimensional operations

**Common Applications:**
- Attention mechanisms in transformers
- Kernel methods (SVM, kernel ridge regression)
- Computing similarity/distance matrices
- Loss function computation (cosine similarity)

---

## 7. Transformation Matrix from Basis B to C

Transform coordinates between bases: `[v]_C = P @ [v]_B` where `P = inv(C) @ B`

**Implementation:**
```python
import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    """Compute transformation matrix P = C^{-1}B."""
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    if B.shape != C.shape:
        raise ValueError("Bases must have same dimensions")

    C_inv = np.linalg.inv(C)
    P = C_inv @ B
    return P.tolist()
```

**PyTorch:**
```python
B = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
C = torch.tensor([[1.0, 1.0], [0.0, 1.0]])

# Method 1: Using inverse
C_inv = torch.linalg.inv(C)
P = C_inv @ B

# Method 2: Using solve (more numerically stable)
P = torch.linalg.solve(C, B)

# Batch transformations
B_batch = torch.randn(32, 3, 3)
C_batch = torch.randn(32, 3, 3)
P_batch = torch.linalg.solve(C_batch, B_batch)  # (32, 3, 3)

# Check basis validity: det != 0
det = torch.linalg.det(C)
valid = det.abs() > 1e-6

# Orthonormal basis (rotation matrix): P = C.T @ B
C_ortho = torch.tensor([[1., 0., 0.],
                        [0., 0.707, 0.707],
                        [0., -0.707, 0.707]])
P = C_ortho.T @ torch.eye(3)
```

**Best Practices:**
- Use `torch.linalg.solve()` instead of `torch.linalg.inv()` (numerically stable, 2x faster)
- Check matrix determinant before inversion; skip if `det ≈ 0`
- For orthonormal bases, use `C.T` instead of `inv(C)` (exact and fast)
- Batch operations are automatic with `torch.linalg.solve()`

**Common Applications:**
- Coordinate system transformations in computer vision
- Change of basis in linear algebra problems
- Rotation and affine transformations
- Principal Component Analysis (PCA) basis transformations

---

# Activation Functions

## 8. ReLU (Rectified Linear Unit)

`ReLU(x) = max(0, x)`. Derivative: 1 if x > 0, else 0.

**Implementation:**
```python
def relu(z: list[float]) -> list[float]:
    """Apply ReLU activation."""
    return [max(0, x) for x in z]

# NumPy: np.maximum(0, z)
# With derivative: (z > 0).astype(float)
```

**PyTorch:**
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = torch.relu(x)  # [0, 0, 0, 1, 2]
output = torch.nn.functional.relu(x)  # Same
relu_layer = torch.nn.ReLU()
output = relu_layer(x)

# Gradient flow
x = x.requires_grad_(True)
y = torch.relu(x)
loss = y.sum()
loss.backward()
# x.grad = [0, 0, 0, 1, 1]  (zero for x <= 0)

# Prevent dying ReLU with batch normalization
# or use LeakyReLU with small negative slope

# Batch operations
batch = torch.randn(32, 100)
activated = torch.relu(batch)

# In-place (memory efficient)
torch.relu_(batch)

# Common pattern in networks
x = torch.relu(self.fc1(x))
x = torch.relu(self.fc2(x))
x = self.fc3(x)  # No activation on output
```

**Best Practices:**
- Use in-place `torch.relu_()` for memory efficiency on large networks
- Combine with batch normalization to avoid dying ReLU
- Consider LeakyReLU if you suspect neuron death
- Don't apply ReLU to final output layer for regression

**Common Applications:**
- Hidden layer activations in deep neural networks
- Standard choice for computer vision and NLP models
- Computationally efficient (just max(0, x))
- Empirically effective with deep networks

---

## 9. Leaky ReLU

```
LeakyReLU(x) = x           if x > 0
             = alpha * x   if x <= 0
```
(typically alpha = 0.01)

**Implementation:**
```python
def leaky_relu(z: list[float], alpha: float = 0.01) -> list[float]:
    """Apply Leaky ReLU."""
    return [x if x > 0 else alpha * x for x in z]

# NumPy: np.where(z > 0, z, alpha * z)
```

**PyTorch:**
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
# [-0.02, -0.01, 0.0, 1.0, 2.0]

leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
output = leaky_relu(x)

# Learnable slope (PReLU)
prelu = torch.nn.PReLU(init=0.25)  # Alpha is learned
output = prelu(x)

# Gradient: 0.01 for x < 0, 1.0 for x > 0
# Prevents dying ReLU problem

# Comparison
x = torch.linspace(-3, 3, 100)
relu = torch.relu(x)
leaky = torch.nn.functional.leaky_relu(x, 0.01)
leaky_02 = torch.nn.functional.leaky_relu(x, 0.2)

# Good for GANs
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(100, 50)
        self.leaky = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leaky(self.fc(x))
```

**Best Practices:**
- Use alpha=0.01 as standard (slope for negative inputs)
- LeakyReLU preferred over ReLU in GANs (discriminator stability)
- PReLU allows alpha to be learned during training
- Solves "dying ReLU" problem where neurons output zero forever

**Common Applications:**
- Discriminators in GANs (alpha ≈ 0.2)
- Problematic layers where ReLU dies
- Adversarial robustness in security-critical models
- Normalized networks where small negative gradients help

---

## 10. Sigmoid Activation Function

`sigmoid(z) = 1 / (1 + exp(-z))`. Maps to (0, 1). Derivative: `sigmoid(z) * (1 - sigmoid(z))`

**Implementation:**
```python
def sigmoid(z: float) -> float:
    """Compute sigmoid."""
    import math
    return 1 / (1 + math.exp(-z))

# NumPy (numerically stable):
def sigmoid_np(z):
    import numpy as np
    return np.where(z >= 0,
                   1 / (1 + np.exp(-z)),
                   np.exp(z) / (1 + np.exp(z)))
```

**PyTorch:**
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = torch.sigmoid(x)  # [0.119, 0.269, 0.5, 0.731, 0.881]

# Method 1: Function
output = torch.sigmoid(x)

# Method 2: Module
sigmoid = torch.nn.Sigmoid()
output = sigmoid(x)

# Numerically stable even for extreme values
large_pos = torch.tensor([100.0])
large_neg = torch.tensor([-100.0])
print(torch.sigmoid(large_pos))  # 1.0 (no overflow)
print(torch.sigmoid(large_neg))  # 0.0 (no underflow)

# Binary classification pattern
class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)  # Return logits (no sigmoid)

# Use BCEWithLogitsLoss for numerical stability
criterion = torch.nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE
# Not: torch.nn.BCELoss() with manual sigmoid

# Gating mechanisms (LSTM, GRU use sigmoid)
forget_gate = torch.sigmoid(x @ W_f + h_prev)

# Gradient vanishes for |z| > 5
x = torch.tensor([-10., -5., 0., 5., 10.], requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()
# x.grad very small for large |x| (saturation problem)
```

**Best Practices:**
- Use `BCEWithLogitsLoss()` instead of manual sigmoid + `BCELoss()` (numerically stable)
- Sigmoid causes gradient vanishing for |z| > 5
- Use only on binary classification output, not hidden layers
- In RNNs/LSTMs, sigmoid is standard for gates

**Common Applications:**
- Binary classification output layer
- LSTM/GRU gate mechanisms (input, forget, output gates)
- Probability output normalization
- Attention gate mechanisms

---

## 11. Softmax Activation Function

`softmax(z_i) = exp(z_i) / sum(exp(z_j))` for multi-class classification.

**Implementation:**
```python
def softmax(z: list[float]) -> list[float]:
    """Compute softmax with numerical stability."""
    import math
    max_z = max(z)
    z_shifted = [x - max_z for x in z]
    exp_z = [math.exp(x) for x in z_shifted]
    sum_exp = sum(exp_z)
    return [x / sum_exp for x in exp_z]

# NumPy:
def softmax_np(z, axis=-1):
    import numpy as np
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)
```

**PyTorch:**
```python
logits = torch.tensor([1.0, 2.0, 3.0])
probs = torch.nn.functional.softmax(logits, dim=0)
# [0.0900, 0.2447, 0.6652] - sums to 1

# CRITICAL: Must specify dim parameter
batch_logits = torch.randn(32, 10)
probs = torch.nn.functional.softmax(batch_logits, dim=1)  # Softmax over classes

# IMPORTANT: Never use softmax + NLLLoss manually
# WRONG: F.softmax(logits, dim=1) then F.nll_loss(torch.log(probs), target)
# RIGHT: F.cross_entropy(logits, target)  # Includes softmax + log_softmax

# Temperature scaling (control distribution sharpness)
temp = 2.0
soft_probs = torch.nn.functional.softmax(logits / temp, dim=-1)  # Smoother
sharp_probs = torch.nn.functional.softmax(logits / 0.5, dim=-1)  # Sharper

# Multi-class architecture
class Classifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)  # Return logits!

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.nn.functional.softmax(logits, dim=-1)

model = Classifier(100, 10)
criterion = torch.nn.CrossEntropyLoss()  # Expects logits

# Training
logits = model(inputs)
loss = criterion(logits, targets)  # No softmax needed

# Inference
probs = model.predict_proba(inputs)
predicted_class = probs.argmax(dim=1)
```

**Best Practices:**
- Always specify the `dim` parameter (no default)
- Use `CrossEntropyLoss()` instead of manual softmax + NLL
- Temperature scaling controls softness: T>1 (smoother), T<1 (sharper)
- Output logits from final layer, not probabilities

**Common Applications:**
- Multi-class classification output (only one correct class)
- Probabilities over K mutually exclusive classes
- Knowledge distillation via temperature scaling
- Beam search in sequence models

---

## 12. Log Softmax

`log_softmax(z_i) = z_i - log(sum(exp(z_j)))`

Output range: `(-∞, 0]`. More numerically stable than `log(softmax(z))`.

**Implementation:**
```python
def log_softmax(z: list[float]) -> list[float]:
    """Compute log-softmax with numerical stability."""
    import math
    max_z = max(z)
    z_shifted = [x - max_z for x in z]
    log_sum_exp = math.log(sum(math.exp(x) for x in z_shifted))
    return [x - max_z - log_sum_exp for x in z]

# NumPy:
def log_softmax_np(z, axis=-1):
    import numpy as np
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(z_shifted), axis=axis, keepdims=True))
    return z_shifted - log_sum_exp
```

**PyTorch:**
```python
logits = torch.tensor([1.0, 2.0, 3.0])
log_probs = torch.nn.functional.log_softmax(logits, dim=0)
# [-2.408, -1.408, -0.408]

probs = torch.exp(log_probs)  # Back to probabilities

# Use with NLLLoss
model_output = torch.randn(32, 10)
targets = torch.randint(0, 10, (32,))

log_probs = torch.nn.functional.log_softmax(model_output, dim=1)
loss = torch.nn.functional.nll_loss(log_probs, targets)

# Equivalently (preferred):
loss = torch.nn.functional.cross_entropy(model_output, targets)

# Numerical stability: large values
large_logits = torch.tensor([100.0, 200.0, 300.0])

# UNSTABLE: softmax then log
probs = torch.nn.functional.softmax(large_logits, dim=0)
log_probs_bad = torch.log(probs)  # [-inf, -inf, 0.0] - underflow!

# STABLE: log_softmax directly
log_probs_good = torch.nn.functional.log_softmax(large_logits, dim=0)
# [-200.0, -100.0, 0.0] - correct!

# Pattern: Log-softmax classifier
class LogSoftmaxClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        logits = self.fc(x)
        return self.log_softmax(logits)

model = LogSoftmaxClassifier(100, 10)
criterion = torch.nn.NLLLoss()
log_probs = model(inputs)
loss = criterion(log_probs, targets)
```

**Best Practices:**
- Prefer `log_softmax()` over `log(softmax())` for numerical stability
- Use `CrossEntropyLoss()` instead of manual log_softmax + `NLLLoss()`
- `log_softmax()` output is always negative or zero
- Dimensionally consistent: output shape matches logits shape

**Common Applications:**
- Numerically stable multi-class classification
- NLLLoss pairing (negative log likelihood)
- Intermediate representation in probabilistic models
- Avoids underflow in softmax computation

---

# Regression Models

## 13. Linear Regression Using Normal Equation

Solve analytically: `theta = inv(X.T @ X) @ X.T @ y`

No iterations needed, no hyperparameters (learning rate).

**Implementation:**
```python
import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    """Solve regression using normal equation."""
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    theta = XTX_inv @ X.T @ y

    return theta.flatten().tolist()

# More robust: use pseudoinverse
def linear_regression_pinv(X, y):
    """Handles singular matrices gracefully."""
    X_pinv = np.linalg.pinv(X)
    theta = X_pinv @ y
    return theta.flatten()
```

**PyTorch:**
```python
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
y = torch.tensor([[1.0], [2.0], [3.0]])

# Method 1: lstsq (preferred, uses QR decomposition)
theta = torch.linalg.lstsq(X, y).solution

# Method 2: Manual normal equation
theta = torch.linalg.solve(X.T @ X, X.T @ y)

# Method 3: Pseudoinverse (most robust)
theta = torch.linalg.pinv(X) @ y

# Gradient computation
X = X.requires_grad_(True)
theta = torch.linalg.lstsq(X, y).solution
loss = ((X @ theta - y) ** 2).mean()
loss.backward()  # Computes X.grad

# Ridge regression (L2 regularization)
def ridge_normal_equation(X, y, lambda_reg=0.1):
    """theta = inv(X.T @ X + lambda_reg * I) @ X.T @ y"""
    n_features = X.shape[1]
    I = torch.eye(n_features)
    theta = torch.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ y
    return theta

# When to use normal equation vs gradient descent
def should_use_normal_equation(n_features):
    return n_features < 10000  # O(n³) for inversion
```

**Best Practices:**
- Use `torch.linalg.lstsq()` (most numerically stable, O(mn²))
- Avoid direct inversion for large matrices (use solver instead)
- Add small regularization if X.T @ X is ill-conditioned
- For n_features > 10k, switch to gradient descent

**Common Applications:**
- Quick baseline solutions for small regression problems
- Closed-form solutions for analytical understanding
- Ridge regression with closed-form solution
- Batch effects correction and trend analysis

---

## 14. Linear Regression Using Gradient Descent

Iteratively optimize: `theta_j := theta_j - alpha * (1/m) * sum((h_theta(x^i) - y^i) * x_j^i)`

**Implementation:**
```python
import numpy as np

def linear_regression_gradient_descent(X, y, alpha, num_iterations):
    """Perform linear regression using gradient descent."""
    m = len(y)
    theta = np.zeros(X.shape[1])

    for i in range(num_iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * X.T @ errors
        theta -= alpha * gradient

    return np.round(theta, 4)
```

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Basic training loop
model = nn.Linear(2, 1, bias=False)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

X_train = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
y_train = torch.tensor([[1.0], [2.0], [3.0]])

for epoch in range(1000):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Learning rate effects
# Too high: diverges
# Too low: slow convergence
# Sweet spot: ~0.01 for small problems

# Advanced: learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(100):
    loss = criterion(model(X_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# Gradient clipping (prevents explosion)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Monitor convergence
losses = []
for epoch in range(1000):
    loss = criterion(model(X_train), y_train)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Variants:**
- Batch GD: Use all samples (stable, slow)
- SGD: One sample (fast, noisy)
- Mini-batch: Balance (practical default)

**Best Practices:**
- Start with learning rate 0.01, adjust based on convergence
- Use mini-batches (32-256 samples) for stability and speed
- Monitor loss curve; should decrease monotonically
- Add learning rate scheduling after 50% of training

**Common Applications:**
- Scalable regression with large datasets
- Online learning (streaming data)
- Mini-batch processing in neural networks
- Fine-tuning pre-trained models

---

## 15. Ridge Regression Loss Function

`L(beta) = (1/n) * sum((y_i - y_hat_i)^2) + alpha * sum(beta_j^2)`

Combines MSE + L2 penalty to prevent overfitting.

**Implementation:**
```python
import numpy as np

def ridge_loss(X, w, y_true, alpha):
    """Compute Ridge Regression loss."""
    y_pred = X @ w
    mse = np.mean((y_true - y_pred) ** 2)
    regularization = alpha * np.sum(w ** 2)
    return mse + regularization
```

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Method 1: Manual loss
def ridge_loss(predictions, targets, weights, alpha):
    mse = torch.nn.functional.mse_loss(predictions, targets)
    l2_penalty = alpha * torch.sum(weights ** 2)
    return mse + l2_penalty

# Method 2: Weight decay in optimizer (preferred)
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)
criterion = nn.MSELoss()

for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Weight decay applies: theta -= lr * (grad + weight_decay * theta)
# This is equivalent to L2 regularization

# Ridge closed-form solution
def ridge_regression_closed_form(X, y, alpha):
    n_features = X.shape[1]
    I = torch.eye(n_features)
    theta = torch.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return theta

# Cross-validation for alpha
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
best_alpha = None
best_loss = float('inf')

for alpha in alphas:
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=alpha)

    for epoch in range(100):
        loss = criterion(model(X_val), y_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if loss < best_loss:
        best_loss = loss
        best_alpha = alpha
```

**Best Practices:**
- Use optimizer `weight_decay` instead of manual L2 penalty (cleaner code)
- Tune alpha via cross-validation (typical range: 0.001-10.0)
- Ridge prevents overfitting on high-dimensional data
- Closed-form solution available: `theta = inv(X.T @ X + alpha * I) @ X.T @ y`

**Common Applications:**
- Preventing overfitting in high-dimensional regression
- High-dimensional feature spaces (genes, NLP embeddings)
- Correlated feature handling (collinearity reduction)
- Stabilizing model when n_features > n_samples

---

# Classification & Metrics

## 16. Implement Precision Metric

`Precision = TP / (TP + FP)`

Ratio of correct positive predictions to all positive predictions.

**Implementation:**
```python
import numpy as np

def precision(y_true, y_pred):
    """Calculate precision."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if (tp + fp) == 0:
        return 0.0

    return tp / (tp + fp)
```

**PyTorch:**
```python
import torch
from torchmetrics import Precision

# Method 1: Manual
def precision_torch(y_true, y_pred):
    y_true, y_pred = y_true.long(), y_pred.long()
    tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
    return tp / (tp + fp + 1e-10)

# Method 2: torchmetrics
metric = Precision(task="binary")
precision_score = metric(y_pred, y_true)

# Batch precision
def batch_precision(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).long()
    tp = torch.sum((y_true == 1) & (y_pred_binary == 1), dim=1).float()
    fp = torch.sum((y_true == 0) & (y_pred_binary == 1), dim=1).float()
    return tp / (tp + fp + 1e-10)

# Precision-Recall tradeoff: vary threshold
def plot_precision_recall_curve(y_true, y_pred_scores):
    thresholds = torch.linspace(0, 1, 101)
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = (y_pred_scores > threshold).long()
        tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
        fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
        fn = torch.sum((y_true == 1) & (y_pred == 0)).float()

        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)

        precisions.append(prec.item())
        recalls.append(rec.item())

    return precisions, recalls
```

**Best Practices:**
- Add small epsilon (1e-10) to avoid division by zero
- Precision is important when false positives are costly (medical diagnosis, fraud)
- Vary decision threshold to achieve desired precision-recall tradeoff
- Use `torchmetrics` for robust, tested implementations

**Common Applications:**
- Medical diagnosis (minimize false positives)
- Spam detection (don't flag legitimate emails)
- Fraud detection (avoid false alarms)
- Ranking/recommendation systems

---

## 17. Calculate Accuracy Score

`Accuracy = (TP + TN) / Total`

**Implementation:**
```python
import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate accuracy."""
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)
```

**PyTorch:**
```python
import torch
from torchmetrics import Accuracy

# Method 1: Manual
def accuracy_torch(y_true, y_pred):
    return torch.mean((y_true == y_pred).float())

# Method 2: torchmetrics
metric = Accuracy(task="binary")
acc = metric(y_pred, y_true)

# Multi-class from logits
logits = torch.randn(100, 10)
targets = torch.randint(0, 10, (100,))
predictions = torch.argmax(logits, dim=1)
accuracy = (predictions == targets).float().mean()

# Balanced accuracy (for imbalanced data)
def balanced_accuracy(y_true, y_pred, num_classes):
    class_accuracies = []
    for class_idx in range(num_classes):
        mask = (y_true == class_idx)
        if mask.sum() == 0:
            continue
        class_acc = ((y_pred[mask]) == class_idx).float().mean()
        class_accuracies.append(class_acc)

    return torch.mean(torch.stack(class_accuracies))

# Top-k accuracy
def top_k_accuracy(logits, targets, k=5):
    _, top_k_preds = torch.topk(logits, k, dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(top_k_preds)
    correct = torch.sum(top_k_preds == targets_expanded).float()
    return correct / len(targets)
```

**Best Practices:**
- Use balanced accuracy for imbalanced datasets
- Top-k accuracy useful for ranking problems
- Simple but can be misleading on imbalanced data
- Combine with other metrics (precision, recall) for complete picture

**Common Applications:**
- Overall model evaluation metric
- Quick sanity check during training
- Comparing different models on balanced datasets
- Final performance reporting

---

## 18. Single Neuron

Single neuron: `y_hat = sigmoid(w.T @ x + b)`

**Implementation:**
```python
import numpy as np

def single_neuron_model(features, labels, weights, bias):
    """Single neuron with sigmoid activation."""
    probabilities = []

    for feature_vector in features:
        z = sum(w * f for w, f in zip(weights, feature_vector)) + bias
        probability = 1 / (1 + np.exp(-z))
        probabilities.append(round(probability, 4))

    mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
    return probabilities, round(mse, 4)
```

**PyTorch:**
```python
import torch
import torch.nn as nn

class SingleNeuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)

neuron = SingleNeuron(2)
criterion = nn.BCEWithLogitsLoss()  # More numerically stable

X = torch.tensor([[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], dtype=torch.float32)
y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float32)

optimizer = torch.optim.SGD(neuron.parameters(), lr=0.01)

for epoch in range(1000):
    logits = neuron.linear(X)  # Get logits before sigmoid
    loss = criterion(logits, y)  # BCEWithLogitsLoss includes sigmoid

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Best Practices:**
- Use `BCEWithLogitsLoss()` instead of sigmoid + `BCELoss()`
- Single neuron is foundation for understanding neural networks
- In practice, use logistic regression libraries
- Perfect for teaching binary classification

**Common Applications:**
- Binary classification foundation
- Logistic regression (single neuron)
- Building block for multi-layer networks
- Understanding neural network fundamentals

---

## 19. Calculate Mean by Row or Column

Row mean: `mean_row_i = (1/n) * sum(a[i,j])`

Column mean: `mean_col_j = (1/m) * sum(a[i,j])`

**Implementation:**
```python
import numpy as np

def calculate_matrix_mean(matrix, mode='row'):
    """Calculate mean by row or column."""
    if mode == 'row':
        return [sum(row) / len(row) for row in matrix]
    elif mode == 'column':
        return [sum(col) / len(matrix) for col in zip(*matrix)]
```

**PyTorch:**
```python
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# Row-wise mean (average each row)
row_means = torch.mean(A, dim=1)  # (3,) → [2., 5., 8.]

# Column-wise mean (average each column)
col_means = torch.mean(A, dim=0)  # (3,) → [4., 5., 6.]

# Keep dimensions for broadcasting
means_keepdim = torch.mean(A, dim=1, keepdim=True)  # (3, 1)

# Center data (subtract mean)
centered = A - torch.mean(A, dim=1, keepdim=True)

# Other statistics
std = torch.std(A, dim=1)
var = torch.var(A, dim=1)
median = torch.median(A, dim=1).values

# Batch operations
batch = torch.randn(32, 10, 5)
row_means_batch = torch.mean(batch, dim=2)  # (32, 10)
```

**Best Practices:**
- Use `keepdim=True` when broadcasting is needed
- Use `dim` parameter consistently (0=columns, 1=rows for 2D)
- Centering data improves numerical stability in models
- Batch operations are automatic with `dim` parameter

**Common Applications:**
- Data normalization and centering
- Feature standardization
- Computing batch statistics for batch normalization
- Dimensionality reduction

---

## 20. Batch Iterator for Dataset

**Implementation:**
```python
import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    """Create batches from dataset."""
    n_samples = len(X)
    batches = []

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        if y is not None:
            batches.append([X[i:end_idx], y[i:end_idx]])
        else:
            batches.append(X[i:end_idx])

    return batches
```

**PyTorch:**
```python
import torch
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Basic DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for X_batch, y_batch in dataloader:
    # Training step
    pass

# Advanced options
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # Randomize each epoch
    num_workers=4,          # Parallel loading
    pin_memory=True,        # Speed up GPU transfer
    drop_last=False         # Include incomplete batch
)

# Stratified sampling (maintain class balance)
from torch.utils.data import WeightedRandomSampler

class_counts = torch.bincount(y)
weights = 1.0 / class_counts[y]
sampler = WeightedRandomSampler(weights, len(dataset))
balanced_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

**Best Practices:**
- Standard batch size: 32, 64, 128, 256 (powers of 2)
- Use `DataLoader` instead of manual iteration (handles shuffling, parallelism)
- Set `num_workers` > 0 for faster data loading
- Use `pin_memory=True` on GPU systems for faster transfer

**Common Applications:**
- Training neural networks efficiently
- Gradient noise reduction (mini-batch vs full batch)
- Parallel data loading from disk
- Memory-efficient training on large datasets

---

## 21. Random Shuffle of Dataset

**Implementation:**
```python
import numpy as np

def shuffle_data(X, y, seed=None):
    """Shuffle X and y maintaining correspondence."""
    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    return X[indices], y[indices]
```

**PyTorch:**
```python
import torch
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Method 1: Using randperm
perm = torch.randperm(len(X))
X_shuffled = X[perm]
y_shuffled = y[perm]

# Method 2: DataLoader handles it automatically
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Reproducible shuffling with seed
torch.manual_seed(42)
perm = torch.randperm(len(X))

# Group shuffling (keep related samples together)
def group_shuffle(X, y, group_indices):
    unique_groups = torch.unique(group_indices)
    shuffled_groups = unique_groups[torch.randperm(len(unique_groups))]

    perm = torch.cat([
        torch.where(group_indices == g)[0]
        for g in shuffled_groups
    ])

    return X[perm], y[perm]
```

**Best Practices:**
- Use `DataLoader(shuffle=True)` in training, `shuffle=False` in validation/test
- Set random seed for reproducibility
- Shuffle before each epoch (DataLoader handles this)
- Group shuffling maintains sample relationships

**Common Applications:**
- Randomizing training data order
- Removing ordering bias in datasets
- Reproducible experiments with fixed seeds
- Stratified sampling for imbalanced data

---

## 22. Feature Scaling Implementation

Standardization: `z = (x - mean) / std`

Min-max: `x' = (x - min(x)) / (max(x) - min(x))`

**Implementation:**
```python
import numpy as np

def feature_scaling(data):
    """Apply standardization and min-max normalization."""
    standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    normalized = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

    return (
        np.round(standardized, 4).tolist(),
        np.round(normalized, 4).tolist()
    )
```

**PyTorch:**
```python
import torch
import torch.nn.functional as F

data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Standardization
mean = torch.mean(data, dim=0, keepdim=True)
std = torch.std(data, dim=0, keepdim=True)
standardized = (data - mean) / (std + 1e-8)

# Min-max normalization
min_val = torch.min(data, dim=0, keepdim=True).values
max_val = torch.max(data, dim=0, keepdim=True).values
normalized = (data - min_val) / (max_val - min_val + 1e-8)

# L2 normalization
normalized_l2 = F.normalize(data, p=2, dim=1)

# Batch normalization layer (automated in networks)
class NetworkWithBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 50)
        self.bn1 = torch.nn.BatchNorm1d(50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Scales activations
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Important: fit scaler on training data only
train_mean = X_train.mean(dim=0, keepdim=True)
train_std = X_train.std(dim=0, keepdim=True)

X_train_scaled = (X_train - train_mean) / (train_std + 1e-8)
X_val_scaled = (X_val - train_mean) / (train_std + 1e-8)  # Use training stats!
```

**Best Practices:**
- Fit scaling parameters on training data only
- Apply same scaling to validation/test data (use training statistics)
- Standardization preferred over min-max for neural networks
- Add epsilon (1e-8) to prevent division by zero

**Common Applications:**
- Preparing data for neural networks
- Stabilizing gradient descent convergence
- Equalizing feature importance in distance metrics
- Improving numerical stability

---

## 23. One-Hot Encoding of Nominal Values

For category `x in {0, 1, ..., K-1}`, create vector of length K with single 1.

**Implementation:**
```python
import numpy as np

def to_categorical(x, n_col=None):
    """One-hot encode categorical values."""
    if n_col is None:
        n_col = x.max() + 1

    one_hot = np.zeros((len(x), n_col), dtype=int)
    one_hot[np.arange(len(x)), x] = 1

    return one_hot
```

**PyTorch:**
```python
import torch
import torch.nn.functional as F

x = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
one_hot = F.one_hot(x, num_classes=3)
# [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

# Batch one-hot
x_batch = torch.tensor([[0, 1], [2, 1]], dtype=torch.long)
one_hot_batch = F.one_hot(x_batch, num_classes=3)  # (2, 2, 3)

# Reverse one-hot (get original indices)
one_hot = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
original = torch.argmax(one_hot, dim=1)  # [0, 1]

# Multi-hot encoding (multiple categories per sample)
def multi_hot_encode(x, num_classes):
    one_hot = torch.zeros(len(x), num_classes)
    for i, indices in enumerate(x):
        one_hot[i, indices] = 1
    return one_hot

# Embedding layer (learnable, more efficient)
embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=16)
ids = torch.tensor([0, 1, 5, 9])
embedded = embedding(ids)  # (4, 16)
```

**Best Practices:**
- Use `F.one_hot()` for one-hot encoding (built-in, efficient)
- Embeddings preferred over one-hot for large categorical spaces (memory efficient, learnable)
- Use `torch.argmax()` to decode one-hot back to class indices
- Multi-hot for multiple labels per sample

**Common Applications:**
- Encoding categorical features (color, country, etc.)
- Classification targets before softmax
- Input preprocessing for neural networks
- Label representation for multi-class problems

---

# Advanced ML Topics

## 24. Cross-Entropy Loss (Binary & Categorical)

Binary: `L_BCE = -(1/n) * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))`

Categorical: `L_CCE = -(1/n) * sum(sum(y_i,c * log(p_i,c)))`

**Implementation:**
```python
import numpy as np

def binary_cross_entropy(y_true, y_pred_probs, epsilon=1e-15):
    """Compute BCE loss."""
    y_true = np.array(y_true, dtype=np.float32)
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
    bce = -(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))
    return np.mean(bce)

def categorical_cross_entropy(y_true, y_pred_probs, epsilon=1e-15):
    """Compute CCE loss."""
    if y_true.ndim == 1:
        n_classes = y_pred_probs.shape[1]
        y_true_one_hot = np.zeros((len(y_true), n_classes))
        y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
        y_true = y_true_one_hot

    y_pred_probs = np.clip(y_pred_probs, epsilon, 1.0)
    cce = -np.sum(y_true * np.log(y_pred_probs), axis=1)
    return np.mean(cce)
```

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Binary cross-entropy
y_true = torch.tensor([1.0, 0.0, 1.0])
y_pred_probs = torch.tensor([0.9, 0.2, 0.8])
bce_loss = F.binary_cross_entropy(y_pred_probs, y_true)

# More stable: use logits
y_pred_logits = torch.tensor([-2.0, 0.7, 1.4])
bce_with_logits = F.binary_cross_entropy_with_logits(y_pred_logits, y_true)

# Categorical cross-entropy
y_true_cat = torch.tensor([0, 1, 2])
y_pred_logits_cat = torch.tensor([[2.0, -1.0, -0.5],
                                  [-0.5, 2.0, -0.5],
                                  [-0.5, -1.0, 2.0]])
cce_loss = F.cross_entropy(y_pred_logits_cat, y_true_cat)

# CRITICAL: CrossEntropyLoss expects LOGITS, not probabilities
# WRONG: F.cross_entropy(F.softmax(logits), targets)
# RIGHT: F.cross_entropy(logits, targets)

# Binary classifier
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)  # Return logits

model = BinaryClassifier(10)
criterion = nn.BCEWithLogitsLoss()

# With class weights (imbalanced data)
class_weights = torch.tensor([0.3, 0.7])
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)

# Label smoothing (regularization)
def cross_entropy_with_label_smoothing(logits, targets, num_classes, smoothing=0.1):
    targets_smooth = F.one_hot(targets, num_classes).float()
    targets_smooth = targets_smooth * (1 - smoothing) + smoothing / num_classes

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(targets_smooth * log_probs).sum(dim=-1).mean()
    return loss

# Focal loss (down-weights easy examples)
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    p = torch.exp(-ce_loss)
    focal = alpha * (1 - p) ** gamma * ce_loss
    return focal.mean()
```

**Best Practices:**
- Use `BCEWithLogitsLoss()` for binary classification (combines sigmoid + BCE)
- Use `CrossEntropyLoss()` for multi-class (combines softmax + log + NLL)
- Always pass logits, never probabilities, to these losses
- Use class weights for imbalanced datasets
- Label smoothing (smoothing=0.1) improves generalization

**Common Applications:**
- Standard loss for classification problems
- Multi-class and binary classification
- Probabilistic output interpretation
- Imbalanced data handling with class weights

---

## 25. Hinge Loss (SVM Loss)

`L_hinge = (1/n) * sum(max(0, 1 - y_i * y_hat_i))`

For SVM with margin = 1.

**Implementation:**
```python
import numpy as np

def hinge_loss(y_true, y_pred):
    """Compute hinge loss (SVM loss)."""
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    margins = 1 - y_true * y_pred
    hinge = np.maximum(0, margins)

    return np.mean(hinge)

def multiclass_hinge_loss(y_true, y_pred_scores):
    """Multi-class hinge loss."""
    losses = []
    for i in range(len(y_true)):
        correct_score = y_pred_scores[i, y_true[i]]
        max_incorrect = np.max(
            y_pred_scores[i, np.arange(len(y_pred_scores[i])) != y_true[i]]
        )
        loss = max(0, 1 + max_incorrect - correct_score)
        losses.append(loss)

    return np.mean(losses)
```

**PyTorch:**
```python
import torch
import torch.nn as nn

# Binary hinge loss
y_true = torch.tensor([1.0, -1.0, 1.0])
y_pred = torch.tensor([0.8, -0.6, 0.3])

hinge_loss = torch.clamp(1 - y_true * y_pred, min=0).mean()

# Multi-class hinge loss
def multiclass_hinge_loss_torch(logits, targets, margin=1.0):
    batch_size = logits.shape[0]
    correct_scores = logits[torch.arange(batch_size), targets]

    margins = margin + logits - correct_scores.unsqueeze(1)
    margins[torch.arange(batch_size), targets] = 0  # Zero correct class

    loss = torch.clamp(margins.max(dim=1)[0], min=0).mean()
    return loss

# SVM with regularization
class SVM(nn.Module):
    def __init__(self, input_size, num_classes, C=1.0):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.C = C

    def forward(self, x):
        return self.fc(x)

model = SVM(10, 3, C=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

X_batch = torch.randn(32, 10)
y_batch = torch.randint(0, 3, (32,))

logits = model(X_batch)
hinge_loss = multiclass_hinge_loss_torch(logits, y_batch)

# Add L2 regularization
l2_loss = sum(torch.norm(p) for p in model.parameters())
total_loss = hinge_loss + model.C * l2_loss

optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

**Best Practices:**
- Hinge loss enforces margin between classes
- Works well when classification confidence is important
- Robust to outliers (compared to cross-entropy)
- Combine with L2 regularization for SVM
- Binary hinge uses labels in {-1, 1}, not {0, 1}

**Common Applications:**
- Support Vector Machines (SVM)
- Margin-based classification
- Ranking problems with margin constraints
- Metric learning with contrastive objectives

---

## 26. Huber Loss (Robust Regression)

```
L_delta(y, y_hat) = 0.5 * (y - y_hat)^2           if |y - y_hat| <= delta
                  = delta * (|y - y_hat| - delta/2)  otherwise
```

Less sensitive to outliers than MSE.

**Implementation:**
```python
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """Compute Huber loss."""
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)

    quadratic = 0.5 * residuals ** 2
    linear = delta * (abs_residuals - 0.5 * delta)

    loss = np.where(abs_residuals <= delta, quadratic, linear)
    return np.mean(loss)
```

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = torch.tensor([1.1, 2.2, 5.0, 3.8, 5.2])

criterion = nn.HuberLoss(delta=1.0)
loss = criterion(y_pred, y_true)

# Custom implementation
def huber_loss_custom(input, target, delta=1.0):
    residuals = input - target
    abs_residuals = torch.abs(residuals)

    quadratic = 0.5 * residuals ** 2
    linear = delta * (abs_residuals - 0.5 * delta)

    loss = torch.where(abs_residuals <= delta, quadratic, linear)
    return loss.mean()

# Robust regression model
class RobustRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = RobustRegressor(10)
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters())

X = torch.randn(100, 10)
y = torch.randn(100)
# Add outliers
y[:5] += torch.randn(5) * 10

for epoch in range(10):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Adaptive delta based on residuals
def compute_optimal_delta(residuals, percentile=75):
    return torch.quantile(torch.abs(residuals), percentile / 100.0)
```

**Best Practices:**
- Robust to outliers (MSE is not)
- Delta parameter controls transition point (tuning via cross-validation)
- Preferred for real-world data with noise
- Adaptive delta = quantile(abs_residuals) for automatic tuning

**Common Applications:**
- Regression with outliers in data
- Robust estimation in finance and sensor data
- Medical data with measurement errors
- Any regression task requiring outlier resistance

---

## 27. Adam Optimizer

```
m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

Adaptive moment estimation with bias correction.

**Implementation:**
```python
import numpy as np

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, gradients):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        updated = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            updated.append(param - update)

        return updated
```

**PyTorch:**
```python
import torch
import torch.optim as optim

model = torch.nn.Linear(10, 1)

# Basic Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Custom hyperparameters
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5
)

# Training loop
for epoch in range(100):
    loss = criterion(model(X), y)

    optimizer.zero_grad()  # Critical: zero gradients first
    loss.backward()
    optimizer.step()

# Learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(100):
    loss = criterion(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# Different learning rates for different layers
param_groups = [
    {'params': model.fc1.parameters(), 'lr': 0.001},
    {'params': model.fc2.parameters(), 'lr': 0.0001}
]
optimizer = optim.Adam(param_groups)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Warmup + cosine annealing (common in transformers)
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = optimizer.defaults['lr']

    def step(self, current_step):
        if current_step < self.warmup_steps:
            lr = self.initial_lr * current_step / self.warmup_steps
        else:
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

**Best Practices:**
- Default lr=0.001 works for most problems (adjust if needed)
- Adam adapts learning rate per parameter (good for sparse data)
- Combine with learning rate scheduler for better convergence
- Gradient clipping prevents explosion in RNNs
- Warmup + cosine annealing popular in transformers

**Common Applications:**
- Default optimizer for deep learning (computer vision, NLP)
- Sparse data and neural networks
- Transfer learning with pre-trained models
- State-of-the-art results in most benchmarks

---

## 28. SGD with Momentum

```
v_t = gamma * v_{t-1} + g_t
theta_t = theta_{t-1} - alpha * v_t
```

Accelerates convergence, dampens oscillations.

**PyTorch:**
```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Nesterov variant (looks ahead)
optimizer_nesterov = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True
)

# Training loop identical to Adam
for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Benefits: 2-3x faster convergence, escapes local minima better, smoother descent.

**Best Practices:**
- Momentum typically 0.9 (classical) or 0.99 (modern)
- Nesterov variant often provides 1-2% improvement
- Slower than Adam but sometimes better generalization
- Good for convex optimization and tuning schedules

**Common Applications:**
- Classical optimization approach
- Computer vision training
- When better generalization needed than Adam
- Baselines and reproducibility studies

---

## 29. RMSprop Optimizer

```
v_t = rho * v_{t-1} + (1 - rho) * g_t^2
theta_t = theta_{t-1} - alpha * g_t / (sqrt(v_t) + epsilon)
```

Adapts learning rate per parameter.

**PyTorch:**
```python
import torch.optim as optim

optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.99,  # Decay rate
    eps=1e-8
)

# Training loop same as other optimizers
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Use case: RNNs, non-convex optimization, when per-parameter adaptation helps.

**Best Practices:**
- alpha=0.99 standard (decay rate of squared gradients)
- Good for RNNs and adversarial training
- Middle ground between SGD and Adam
- Less memory than Adam (no first moment estimate)

**Common Applications:**
- Recurrent Neural Networks (RNNs)
- Adversarial training and GANs
- Non-convex optimization problems
- When memory is constrained

---

## 30. Complete Backpropagation (Multi-layer Network)

Compute gradients via chain rule: `dL/dW^(l) = dL/dz^(l) * dz^(l)/dW^(l)`

**PyTorch (automatic):**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class MultiLayerNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = MultiLayerNN(10, [64, 32], 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,)).float()

# Forward pass
logits = model(X)
loss = criterion(logits, y)

# Backward pass (automatic differentiation)
optimizer.zero_grad()
loss.backward()  # Computes all gradients via backprop
optimizer.step()  # Updates all parameters

# Gradient inspection
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Custom backprop for learning purposes
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input > 0).float()
```

**Best Practices:**
- PyTorch autograd handles backprop automatically
- Always call `optimizer.zero_grad()` before `loss.backward()`
- Inspect gradient norms to detect vanishing/exploding gradients
- Clip gradients if norm exceeds threshold (prevents explosion)
- Custom autograd function for advanced use cases

**Common Applications:**
- Training all neural networks (automatic backprop)
- Understanding gradient flow in deep networks
- Implementing custom layers or losses
- Debugging numerical instabilities

---

## 31. AUC-ROC Curve

Area under the Receiver Operating Characteristic curve.

- X-axis: False Positive Rate (FPR)
- Y-axis: True Positive Rate (TPR)
- AUC ∈ [0, 1]: 0.5 = random, 1.0 = perfect

**Implementation:**
```python
import numpy as np

def auc_roc_from_scratch(y_true, y_pred_probs):
    """Compute AUC-ROC without sklearn."""
    y_true = np.array(y_true, dtype=np.int32)
    y_pred_probs = np.array(y_pred_probs, dtype=np.float32)

    # Sort by probability descending
    sorted_idx = np.argsort(-y_pred_probs)
    y_sorted = y_true[sorted_idx]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    # Cumulative TP, FP
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    # Add starting point
    tp = np.concatenate(([0], tp))
    fp = np.concatenate(([0], fp))

    # Rates
    tpr = tp / n_pos
    fpr = fp / n_neg

    # Trapezoid rule
    auc = np.trapz(tpr, fpr)

    return auc, (fpr, tpr)

def auc_roc_alternative(y_true, y_pred_probs):
    """Alternative: Mann-Whitney U statistic."""
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    ranks = np.argsort(-y_pred_probs) + 1
    rank_sum_pos = np.sum(ranks[y_true == 1])

    # AUC = (rank_sum - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return auc
```

**PyTorch:**
```python
import torch
from torchmetrics import AUROC

y_true = torch.tensor([1, 0, 1, 1, 0, 1])
y_pred_probs = torch.tensor([0.9, 0.2, 0.8, 0.7, 0.3, 0.85])

# Using torchmetrics
auroc = AUROC(task="binary")
auc_score = auroc(y_pred_probs, y_true)

# Manual computation
def compute_auc_pytorch(y_true, y_pred_probs):
    sorted_idx = torch.argsort(y_pred_probs, descending=True)
    y_sorted = y_true[sorted_idx]

    n_pos = (y_true == 1).sum().float()
    n_neg = (y_true == 0).sum().float()

    tp = torch.cumsum(y_sorted.float(), dim=0)
    fp = torch.cumsum((y_sorted == 0).float(), dim=0)

    tp = torch.cat([torch.tensor([0.0]), tp])
    fp = torch.cat([torch.tensor([0.0]), fp])

    tpr = tp / n_pos
    fpr = fp / n_neg

    auc = torch.trapz(tpr, fpr)
    return auc.item()

# Training with AUC monitoring
class AUCOptimizedModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def train_with_auc_monitoring(model, train_loader, val_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    auroc = AUROC(task="binary")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                all_preds.append(model(X_batch).squeeze())
                all_targets.append(y_batch)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        auc = auroc(all_preds, all_targets)
        print(f"Epoch {epoch}, Loss: {train_loss/len(train_loader):.4f}, AUC: {auc:.4f}")
```

**Best Practices:**
- AUC threshold-invariant (doesn't depend on decision threshold)
- Better than accuracy for imbalanced datasets
- Interpret as: probability model ranks random positive > random negative
- Use with precision-recall curve for imbalanced data

**Common Applications:**
- Medical diagnosis and classification
- Fraud detection and ranking
- Imbalanced binary classification
- Threshold-independent evaluation metric

---

## 32-37: Advanced Topics (Summary)

These require similar deep implementations:

**32. F1 Score:** `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

**33. Confusion Matrix:** Count TP, TN, FP, FN for each class.

**34. Recall/Sensitivity:** `Recall = TP / (TP + FN)`

**35. PCA:** Eigendecomposition of covariance matrix for dimensionality reduction.

**36. Dropout:** Random neuron masking during training for regularization.

**37. Batch Normalization:** Normalize activations to N(0, 1) per batch.

All available via torchmetrics and torch.nn with standard patterns.

**Best Practices:**
- F1: Balance precision-recall tradeoff
- Confusion matrix: Diagnose class-specific errors
- Recall: Important when false negatives costly (cancer screening)
- PCA: Reduce dimensionality while preserving variance
- Dropout: Regularization, typically rate=0.5
- Batch norm: Stabilize training, reduce internal covariate shift

**Common Applications:**
- F1: Multi-class classification evaluation
- Confusion matrix: Detailed error analysis
- Recall: Medical diagnosis, anomaly detection
- PCA: Visualization, feature extraction
- Dropout: Prevent overfitting in neural networks
- Batch norm: Standard in modern deep networks

---

# Appendix: Einstein Notation (einsum) Tutorial

Einstein summation convention: repeated indices are implicitly summed over. `torch.einsum()` and `np.einsum()` let you express complex tensor operations in a single compact string.

## The Core Idea

In Einstein notation, an operation like `c_i = sum_j(A_ij * b_j)` is written as `ij,j->i`. The rule: any index that appears on the left side of `->` but NOT on the right side gets summed over.

```
# Pattern: 'input_subscripts -> output_subscripts'
# Repeated indices in inputs = multiply along that axis
# Missing indices in output = sum over that axis
```

## Fundamental Operations

```python
import torch
import numpy as np

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# ---- Vectors ----

# Dot product: c = sum(a_i * b_i)
c = torch.einsum('i,i->', a, a)          # Same as torch.dot(a, a) = 14.0

# Outer product: C_ij = a_i * b_j
C = torch.einsum('i,j->ij', a, b)        # Same as torch.outer(a, b), shape (3, 3)

# Element-wise multiply: c_i = a_i * b_i
c = torch.einsum('i,i->i', a, b)         # Same as a * b = [4, 10, 18]

# ---- Matrices ----

# Matrix-vector: c_i = sum_j(A_ij * b_j)
c = torch.einsum('ij,j->i', A, torch.tensor([5.0, 6.0]))  # Same as A @ b

# Matrix multiply: C_ik = sum_j(A_ij * B_jk)
C = torch.einsum('ij,jk->ik', A, B)      # Same as A @ B

# Transpose: B_ji = A_ij
B_T = torch.einsum('ij->ji', A)          # Same as A.T

# Trace: c = sum_i(A_ii)
trace = torch.einsum('ii->', A)           # Same as torch.trace(A)

# Diagonal: d_i = A_ii
diag = torch.einsum('ii->i', A)           # Same as torch.diag(A)

# Sum all elements
total = torch.einsum('ij->', A)           # Same as A.sum()

# Sum along rows: c_i = sum_j(A_ij)
row_sum = torch.einsum('ij->i', A)        # Same as A.sum(dim=1)

# Sum along columns: c_j = sum_i(A_ij)
col_sum = torch.einsum('ij->j', A)        # Same as A.sum(dim=0)
```

## Batch Operations (The Real Power)

Einsum shines when you need batch operations that would be awkward with standard ops.

```python
# Batch matrix multiply: C_bij = sum_k(A_bik * B_bkj)
A_batch = torch.randn(32, 10, 5)    # 32 matrices of shape (10, 5)
B_batch = torch.randn(32, 5, 8)     # 32 matrices of shape (5, 8)
C_batch = torch.einsum('bik,bkj->bij', A_batch, B_batch)  # (32, 10, 8)
# Same as torch.bmm(A_batch, B_batch)

# Batch dot product: c_b = sum_i(a_bi * b_bi)
a_batch = torch.randn(32, 10)
b_batch = torch.randn(32, 10)
dots = torch.einsum('bi,bi->b', a_batch, b_batch)  # (32,)
# Same as (a_batch * b_batch).sum(dim=1)

# Batch outer product: C_bij = a_bi * b_bj
outers = torch.einsum('bi,bj->bij', a_batch, b_batch)  # (32, 10, 10)

# Batch trace: t_b = sum_i(A_bii)
A_sq = torch.randn(32, 5, 5)
traces = torch.einsum('bii->b', A_sq)  # (32,)

# Batch diagonal: d_bi = A_bii
diags = torch.einsum('bii->bi', A_sq)  # (32, 5)
```

## ML-Specific Patterns

```python
# ---- Attention Mechanism ----
# Q: (batch, heads, seq_q, d_k)
# K: (batch, heads, seq_k, d_k)
# V: (batch, heads, seq_k, d_v)

Q = torch.randn(2, 8, 10, 64)
K = torch.randn(2, 8, 10, 64)
V = torch.randn(2, 8, 10, 64)

# Attention scores: scores_bhij = sum_d(Q_bhid * K_bhjd)
scores = torch.einsum('bhid,bhjd->bhij', Q, K)  # (2, 8, 10, 10)
# Same as torch.matmul(Q, K.transpose(-2, -1))

# Weighted values: out_bhid = sum_j(weights_bhij * V_bhjd)
weights = torch.softmax(scores / 8.0, dim=-1)
output = torch.einsum('bhij,bhjd->bhid', weights, V)  # (2, 8, 10, 64)

# ---- Bilinear form: c = x^T A y ----
x = torch.randn(5)
y = torch.randn(5)
A = torch.randn(5, 5)
bilinear = torch.einsum('i,ij,j->', x, A, y)  # scalar

# Batch bilinear: c_b = sum_ij(x_bi * A_ij * y_bj)
x_batch = torch.randn(32, 5)
y_batch = torch.randn(32, 5)
bilinear_batch = torch.einsum('bi,ij,bj->b', x_batch, A, y_batch)  # (32,)

# ---- Gram matrix (pairwise dot products) ----
X = torch.randn(100, 20)   # 100 samples, 20 features
gram = torch.einsum('id,jd->ij', X, X)  # (100, 100)
# Same as X @ X.T

# ---- Tensor contraction (multi-head projection) ----
# Combine heads: output_bsd = sum_h(head_bhsd * W_hd)  (simplified)
heads = torch.randn(2, 8, 10, 64)      # multi-head output
W = torch.randn(8, 64)                   # per-head weights
contracted = torch.einsum('bhsd,hd->bsd', heads, W)  # (2, 10, 64)

# ---- Hadamard (element-wise) product of matrices ----
C = torch.einsum('ij,ij->ij', A[:2,:2], B[:2,:2])  # Same as A * B
```

## Einsum vs Standard Ops: Quick Reference

```
# Operation              | einsum               | Standard PyTorch
# ---------------------- | -------------------- | ------------------
# Dot product            | 'i,i->'              | torch.dot(a, b)
# Outer product          | 'i,j->ij'            | torch.outer(a, b)
# Matrix multiply        | 'ij,jk->ik'          | A @ B
# Matrix-vector          | 'ij,j->i'            | A @ v
# Transpose              | 'ij->ji'             | A.T
# Trace                  | 'ii->'               | torch.trace(A)
# Diagonal               | 'ii->i'              | torch.diag(A)
# Batch matmul           | 'bij,bjk->bik'       | torch.bmm(A, B)
# Batch dot              | 'bi,bi->b'           | (a*b).sum(dim=1)
# Attention scores       | 'bhid,bhjd->bhij'    | Q @ K.transpose(-2,-1)
# Weighted sum           | 'bhij,bhjd->bhid'    | weights @ V
# Gram matrix            | 'id,jd->ij'          | X @ X.T
# Bilinear form          | 'i,ij,j->'           | x @ A @ y
# Element-wise           | 'ij,ij->ij'          | A * B
# Sum all                | 'ij->'               | A.sum()
# Row sum                | 'ij->i'              | A.sum(dim=1)
# Col sum                | 'ij->j'              | A.sum(dim=0)
```

## Performance Tips

```python
# 1. einsum is optimized in PyTorch — often as fast as dedicated ops
#    But for simple cases, dedicated ops may be slightly faster due to
#    less parsing overhead.

# 2. Use opt_einsum for complex contractions (auto-optimizes contraction order)
# pip install opt_einsum
# torch.backends.opt_einsum.enabled = True  # Enabled by default in PyTorch 1.12+

# 3. For repeated operations, cache the contraction path:
from opt_einsum import contract_expression
expr = contract_expression('bij,bjk->bik', (32, 10, 5), (32, 5, 8))
result = expr(A_batch, B_batch)  # Reuse optimized path

# 4. Avoid einsum for simple operations where a dedicated function exists.
#    Use it when the operation is hard to express otherwise (3+ tensors,
#    unusual contractions, or batch operations without bmm equivalent).
```

**Best Practices:**
- Start simple: master 2-index operations before moving to batch (3+ index)
- Read left to right: input indices describe tensor shapes, output describes result shape
- Missing output indices = summation (this is the key insight)
- Use the reference table above until the patterns become second nature

**Common Applications:**
- Attention mechanisms in transformers (`bhid,bhjd->bhij`)
- Batch bilinear forms and gram matrices
- Custom tensor contractions in physics-informed neural networks
- Multi-head attention projection and recombination
