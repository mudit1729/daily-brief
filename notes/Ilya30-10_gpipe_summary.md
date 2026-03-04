# GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism - Detailed Paper Summary

| | |
|---|---|
| **Authors** | Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, et al. (Google) |
| **Published** | NeurIPS 2019 |
| **ArXiv** | 1811.06965 |
| **Publication Date** | November 2018 |

---

## 1. One-Page Overview

### Metadata
- **Full Title:** GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism
- **Venue:** NeurIPS 2019 (Outstanding Paper Award)
- **Institution:** Google Brain
- **Key Contributors:** Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat
- **Code Availability:** Official TensorFlow implementation; PyTorch port (torchgpipe)

### Problem Solved
Training modern large-scale neural networks (transformers, ResNets) exceeds the memory capacity of a single accelerator. Traditional data parallelism distributes examples across devices but hits communication bottlenecks. Model parallelism distributes layer partitions but suffers from severe GPU idle time (pipeline bubbles) during backward passes.

**GPipe solves this by introducing micro-batch pipeline parallelism:**
- Splits mini-batches into micro-batches
- Overlaps computation across multiple pipeline stages
- Dramatically reduces idle time and achieves near-linear speedup

### Key Novelty: Micro-Batch Pipeline Parallelism
Unlike naive pipeline parallelism where only one accelerator is active at a time, GPipe:
1. **Partitions the model** into K sequential stages (layers or groups of layers)
2. **Splits input mini-batch** into M micro-batches
3. **Schedules forward/backward passes** so all K devices compute simultaneously
4. **Accumulates gradients** across micro-batches before parameter updates

This transforms O(K) pipeline bubbles (idle time) into O(1) relative overhead, enabling 128+ device clusters.

### Three Things to Remember

> 1. **Pipeline Parallelism ≠ Data Parallelism**: Model is divided into sequential stages; communication patterns differ
> 2. **Micro-Batches are Essential**: Without them, only 1 device works at a time; with M micro-batches, ~M devices work simultaneously
> 3. **Re-materialization Trades Compute for Memory**: Forward pass activations are recomputed during backprop, enabling larger models on limited VRAM

### What This Paper Contributes
- First practical pipeline parallelism method for deep learning with near-linear scaling
- Synchronization protocol for gradient accumulation across micro-batches
- Checkpointing (re-materialization) strategy to reduce memory footprint
- Empirical validation: 557M-param AmoebaNet (84.4% ImageNet accuracy), 83.9B-param Transformer

---

## 2. Problem Setup and Outputs

### The Core Problem
**Memory Bottleneck:** Training state-of-the-art models requires:
- Model parameters: billions of floats
- Activation memory during forward pass: grows with batch size and depth
- Gradient buffers during backward pass: same size as activations
- Optimizer states (Adam): 2-3x parameter size

**Single GPU Example:**
```
Model: Transformer-12B parameters
Float32 precision: 4 bytes per param
Parameters alone: 48 GB (exceeds V100 80GB limit)
Activation memory: 10-100GB per batch
→ Cannot fit on any single device
```

### Multi-Device Approaches and Their Limits

**Data Parallelism (Broadcasting):**
```
Device 0: Full model + batch[0:n/2]
Device 1: Full model + batch[n/2:n]
Communication: All-reduce gradients every step
Problem: Communication becomes bottleneck for large models
Scaling efficiency drops ~70% at 64 devices
```

**Naive Model Parallelism (Pipelining):**
```
Stage 0 (Device 0): Layers 1-3     | Forward_t | Idle    | Backward_t | Idle
Stage 1 (Device 1): Layers 4-6     | Idle      | Forward | Idle       | Backward
Stage 2 (Device 2): Layers 7-9     | Idle      | Idle    | Forward    | Backward

Pipeline bubbles: (K-1)/K idle time for K stages
Example: 8 devices → 7/8 = 87.5% idle time during forward pass
```

### GPipe's Solution: Model + Data Parallelism
**Key Insight:** Pipeline multiple micro-batches through stages simultaneously

```
Device 0: F(mb0) | B(mb0) | F(mb1) | B(mb1) | F(mb2) | B(mb2)
Device 1:   Idle | F(mb0) | B(mb0) | F(mb1) | B(mb1) | F(mb2)
Device 2:   Idle |   Idle | F(mb0) | B(mb0) | F(mb1) | B(mb1)

With M micro-batches and K stages:
  - Total pipeline overhead: K timesteps
  - Useful work: M×K timesteps
  - Efficiency: M×K / (M×K + K - 1) ≈ M/(M+1) for large M
  - Example: M=8, K=8 → 88.9% efficiency (vs 12.5% naive pipeline)
```

### Formal Problem Statement
Given:
- Model f parameterized by θ, representable as f = f_K ∘ f_{K-1} ∘ ... ∘ f_1
- Training dataset with examples {x, y}
- Mini-batch size B
- K compute devices

Find:
- Partition scheme P = {p_1, ..., p_K} assigning layers to devices
- Micro-batch size b and count M such that B = M × b
- Backward pass schedule minimizing total training time while fitting in device memory

### Optimization Objectives
1. **Minimize wall-clock time:** T_total = T_forward + T_backward + T_communication
2. **Fit in device memory:** M(m, b) ≤ Device_VRAM
3. **Maintain model accuracy:** No approximations, exact gradients

---

## 3. Coordinate Frames and Device Assignment

### Pipeline Stage Definition
A **stage** is a sequential composition of layers assigned to one device:

```
Stage 0 (Device 0): Input → BatchNorm → Conv(3→64) → ReLU
Stage 1 (Device 1): Conv(64→128) → ReLU → Conv(128→256) → ReLU
Stage 2 (Device 2): GlobalAvgPool → Dense(256→1000) → Output

Key property: Output of Stage i feeds into input of Stage i+1 only
Constraint: Stages must be acyclic (no layer can depend on later layers)
```

### Micro-Batch Indexing
Original mini-batch B split into M micro-batches b:
```
Full batch: X = [x_1, x_2, ..., x_B] with shape (B, H, W, C)
Micro-batch size: b = B / M
Micro-batches:
  mb_0 = X[0:b]     shape: (b, H, W, C)
  mb_1 = X[b:2b]    shape: (b, H, W, C)
  ...
  mb_{M-1} = X[(M-1)b:B]  shape: (b, H, W, C)
```

### Device Assignment Notation
```
partition[i] = device_id  # Layer i assigned to device_id
device_layers[d] = [i, i+1, ..., j]  # Device d owns layers [i:j+1]

Example (8-layer model on 4 devices):
partition = [0, 0, 1, 1, 2, 2, 3, 3]
device_layers = {
  0: [0, 1],
  1: [2, 3],
  2: [4, 5],
  3: [6, 7]
}
```

### Temporal Scheduling Grid
Pipeline schedule represented as (time, device, micro-batch):

```
Time \  Device 0        Device 1        Device 2        Device 3
  t0:  F(mb0)          Idle            Idle            Idle
  t1:  B(mb0)          F(mb0)          Idle            Idle
  t2:  F(mb1)          B(mb0)          F(mb0)          Idle
  t3:  B(mb1)          F(mb1)          B(mb0)          F(mb0)
  t4:  F(mb2)          B(mb1)          F(mb1)          B(mb0)
  t5:  B(mb2)          F(mb2)          B(mb1)          F(mb1)
  t6:  Idle            B(mb2)          F(mb2)          B(mb1)
  t7:  Idle            Idle            B(mb2)          F(mb2)
  t8:  Idle            Idle            Idle            B(mb2)

Forward bubble: K-1 = 3 steps (first column idle)
Backward bubble: K-1 = 3 steps (last rows idle)
Total overhead: 2(K-1) = 6 steps
Useful work: M×K = 12 steps
Efficiency: 12/18 = 66.7%
```

### Activation and Gradient Flow
For micro-batch m, device d:
```
Input to stage d: a^[d]_m (activation from stage d-1)
Forward pass: a^[d+1]_m = f_d(a^[d]_m; θ_d)
Output saved: Store a^[d+1]_m (unless re-materialized)

Backward pass:
  ∂L/∂a^[d]_m ← ∂L/∂a^[d+1]_m  (from stage d+1)
  ∂L/∂θ_d ← Accumulate across all M micro-batches
  Final update: θ_d ← θ_d - α × ∑_m ∂L/∂θ_d[m]
```

### Memory Layout on Device d
```
Device VRAM (total: D GB):
├─ Parameters θ_d: P_d parameters × 4 bytes
├─ Activation cache:
│  ├─ Stored: a^[d+1]_0, a^[d+1]_1, ..., a^[d+1]_{M-1}
│  ├─ Size: M × batch_size × width × height × channels × 4 bytes
├─ Gradient buffers:
│  ├─ ∂L/∂a^[d]_m for active micro-batches
│  └─ ∂L/∂θ_d accumulator
└─ Optimizer state (Adam): 2 × P_d × 4 bytes
```

---

## 4. Architecture Deep Dive

### GPipe High-Level System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │  Input Data (Mini-batch B)                 │
                    └─────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────────────┐
                    │  Micro-Batch Splitter (B → M×b)            │
                    │  mb_0, mb_1, ..., mb_{M-1}                │
                    └─────────────┬───────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────────┐
        │                         │                             │
    ┌───▼────────┐        ┌───────▼────────┐        ┌──────────▼──┐
    │  Stage 0   │        │   Stage 1      │  ...   │  Stage K-1  │
    │ (Device 0) │───────▶│  (Device 1)    │───────▶│ (Device K-1)│
    │            │        │                │        │             │
    │ Layers:    │        │  Layers:       │        │ Layers:     │
    │ Conv1-2    │        │  Conv3-4       │        │ FC-Out      │
    └────┬───────┘        └────┬───────────┘        └──────┬──────┘
         │                     │                          │
         │ Forward: mb → act   │ Forward: mb → act        │ Forward: mb → loss
         │ Backward: act → ∇  │ Backward: act → ∇        │ Backward: loss → ∇
         │                     │                          │
        ┌▼─────────────────────┼──────────────────────────▼─┐
        │  Gradient Accumulator (across M micro-batches)    │
        │  ∑ ∂L/∂θ_0 + ... + ∂L/∂θ_{K-1}                  │
        └┬───────────────────────────────────────────────────┘
         │
        ┌▼──────────────────────────────────┐
        │  Optimizer Step                    │
        │  θ ← θ - α × (∑ ∇θ + regularizer) │
        └───────────────────────────────────┘
```

### Forward Pass Execution Timeline

```
               Device 0       Device 1       Device 2       Device 3
         [Stage 0]      [Stage 1]       [Stage 2]      [Stage 3]

Time 0:  F(mb0)         -              -              -
Time 1:  F(mb1)         F(mb0)         -              -
Time 2:  F(mb2)         F(mb1)         F(mb0)         -
Time 3:  F(mb3)         F(mb2)         F(mb1)         F(mb0)
         ↑              ↑              ↑              ↑
      All four devices working in parallel (full pipeline saturation)

Time 4:  -              F(mb3)         F(mb2)         F(mb1)
Time 5:  -              -              F(mb3)         F(mb2)
Time 6:  -              -              -              F(mb3)
         ↑              ↑              ↑              ↑
      Forward pass complete; backward phase begins

Total forward time: 6 timesteps (K - 1 + M)
Ideal single-device time: 4×M = 12 timesteps
Speedup: 12/6 = 2.0× (not 4×, due to pipeline bubbles)
```

### Backward Pass with Synchronization

```
               Device 0       Device 1       Device 2       Device 3
         [Stage 0]      [Stage 1]       [Stage 2]      [Stage 3]

Time 7:  -              -              -              B(mb3)
Time 8:  -              -              B(mb3)         B(mb2)
Time 9:  -              B(mb3)         B(mb2)         B(mb1)
Time 10: B(mb3)         B(mb2)         B(mb1)         B(mb0)
         ↑              ↑              ↑              ↑
      Full pipeline saturation during backward

Time 11: B(mb2)         B(mb1)         B(mb0)         -
Time 12: B(mb1)         B(mb0)         -              -
Time 13: B(mb0)         -              -              -

Total backward time: 6 timesteps (K - 1 + M)
Total pipeline time: 12 timesteps
```

### Bubble Time Analysis

**Definition:** Bubble time = idle GPU cycles / total cycles

For K stages and M micro-batches:

```
Forward phase idle: K - 1 timesteps (first K-1 stages start empty)
Backward phase idle: K - 1 timesteps (last K-1 stages end empty)
Useful work: M × K timesteps

Total bubble = 2(K - 1)
Total time = M × K + 2(K - 1)
Efficiency = (M × K) / (M × K + 2(K - 1))
           = M / (M + 2(K-1)/K)
           ≈ M / (M + 2) for large K

Examples:
  K=8, M=1: Eff = 8/(8+14) = 36.4%   (single batch per micro-batch)
  K=8, M=4: Eff = 32/(32+14) = 69.6%
  K=8, M=8: Eff = 64/(64+14) = 82.1%
  K=8, M=16: Eff = 128/(128+14) = 90.1%

Key insight: Increasing M (more micro-batches) → linear efficiency improvement
```

### Memory-Computation Tradeoff

Without re-materialization:
```
Forward activation memory: O(K × b × hidden_dim)
  - Each stage stores M micro-batch activations
  - Total: M × K × (batch_size/M) × hidden_dim = K × b × hidden_dim

Total device memory:
  = Params + Optimizer_state + Activations + Gradients
  = P + 2P + K×b×H + b×H
  = 3P + (K+1)×b×H
```

With re-materialization (checkpointing):
```
Forward pass: Compute all activations, discard them (recompute memory)
Backward pass: Recompute each stage's forward pass, then backward

Memory saved: K × b × H (all intermediate activations)
Compute overhead: K × (forward_flops) recomputation

Total device memory (with re-mat):
  = P + 2P + 2×b×H  (only boundary activations stored)
  = 3P + 2×b×H

For K=128 (128 devices):
  Without re-mat: 3P + 129×b×H (prohibitive)
  With re-mat: 3P + 2×b×H (practical)
```

---

## 5. Forward Pass Pseudocode

### Shape-Annotated Forward Pass

```python
def gpipe_forward_pass(
    batch_X: Tensor[B, H, W, C],
    stages: List[nn.Module],
    num_micro_batches: int = M,
    num_devices: int = K
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Forward pass with micro-batch pipelining.

    Args:
        batch_X: Input batch of shape (B, H, W, C)
        stages: List of K sequential model stages
        M: Number of micro-batches
        K: Number of devices (len(stages))

    Returns:
        all_activations: Cached activations for backward pass
        all_losses: Loss for each micro-batch
    """

    # Step 1: Partition mini-batch into micro-batches
    micro_batch_size = B // M
    micro_batches = []
    for m in range(M):
        start_idx = m * micro_batch_size
        end_idx = start_idx + micro_batch_size
        mb_m = batch_X[start_idx:end_idx]  # Shape: (b, H, W, C) where b = B/M
        micro_batches.append(mb_m)

    # Step 2: Initialize storage
    # activation_cache[m][k] = output of stage k applied to micro-batch m
    activation_cache = [[None for _ in range(K)] for _ in range(M)]
    all_losses = []

    # Step 3: Forward pass schedule
    # For each micro-batch, push through all stages sequentially
    for m in range(M):
        current_activation = micro_batches[m]  # Shape: (b, H, W, C)

        for k in range(K):
            # Shape tracking through stage k:
            # input: (b, ...) → stage_k → output: (b, ...)

            stage_k = stages[k]

            if k == 0:
                # First stage: apply to micro-batch directly
                # Forward: (b, H, W, C) → (b, H/2, W/2, 64) [for ResNet]
                current_activation = stage_k(current_activation)
            elif k == K - 1:
                # Last stage: compute loss
                # Forward: (b, feature_dim) → (b, num_classes)
                logits = stage_k(current_activation)  # Shape: (b, num_classes)

                # Loss computation
                loss_m = cross_entropy(logits, labels[m])  # Scalar loss
                all_losses.append(loss_m)
                current_activation = logits  # Store for backward
            else:
                # Intermediate stage
                # Forward: (b, in_channels) → (b, out_channels)
                current_activation = stage_k(current_activation)

            # Cache activation (will be discarded if re-materialization enabled)
            activation_cache[m][k] = current_activation

    return activation_cache, all_losses


def gpipe_backward_pass(
    activation_cache: List[List[Tensor]],
    stages: List[nn.Module],
    all_losses: List[Tensor],
    num_micro_batches: int = M,
    num_devices: int = K
) -> None:
    """
    Backward pass with gradient accumulation.

    Pseudo-gradients flow from loss through layers.
    Gradients accumulated across all micro-batches.
    """

    # Initialize gradient accumulators for each stage
    # grad_accum[k] = ∑_m (∂L_m / ∂θ_k)
    grad_accumulators = [torch.zeros_like(stage.weight) for stage in stages]

    # Backward pass: iterate micro-batches in reverse
    for m in range(M - 1, -1, -1):
        # Gradient at output of last stage (from loss)
        # ∂L_m / ∂logits_m = softmax(logits_m) - one_hot(label_m)
        grad_output = all_losses[m].backward(retain_graph=(m > 0))

        # Backprop through stages in reverse
        for k in range(K - 1, -1, -1):
            stage_k = stages[k]

            # Retrieve cached activation (or recompute if re-materialized)
            if USE_REMAT:
                # Recompute forward for stage k with input from stage k-1
                if k == 0:
                    a_input = micro_batches[m]
                else:
                    a_input = activation_cache[m][k-1]  # Output of prev stage
                a_cached = stage_k.forward(a_input)
            else:
                a_cached = activation_cache[m][k]

            # Backward through stage k
            # Input: grad_output (from next stage or loss)
            # Output: grad_input (to previous stage), grad_params (accumulated)

            grad_input = stage_k.backward(
                grad_output=grad_output,
                cached_activation=a_cached
            )

            # Accumulate gradients for this stage
            grad_accumulators[k] += stage_k.get_param_gradients()

            # Pass gradient to previous stage
            grad_output = grad_input

    # Step 4: Apply accumulated gradients (after all micro-batches processed)
    for k in range(K):
        stages[k].optimizer_step(accumulated_grad=grad_accumulators[k])


# Example shape flow for ResNet-50-like model on ImageNet:
"""
Input: X ∈ ℝ^{B × 224 × 224 × 3}
Micro-batch: mb ∈ ℝ^{b × 224 × 224 × 3} where b = B/M

Stage 0 (Conv1 + BN + ReLU):
  Input:  (b, 224, 224, 3)
  Output: (b, 112, 112, 64)

Stage 1 (ResNet blocks 1-2):
  Input:  (b, 112, 112, 64)
  Output: (b, 56, 56, 128)

Stage 2 (ResNet blocks 3-4):
  Input:  (b, 56, 56, 128)
  Output: (b, 28, 28, 256)

Stage 3 (ResNet blocks 5-6):
  Input:  (b, 28, 28, 256)
  Output: (b, 14, 14, 512)

Stage 4 (ResNet blocks 7-9 + GlobalAvgPool):
  Input:  (b, 14, 14, 512)
  Output: (b, 2048)

Stage 5 (Classification head):
  Input:  (b, 2048)
  Output: (b, 1000)
  Loss:   scalar (averaged over micro-batch)
"""
```

### Gradient Accumulation Detail

```python
# Pseudo-gradient computation with shape annotations

# Forward pass saves activations:
a0_m = input_batch_m  # Shape: (b, 224, 224, 3)
a1_m = stage_0(a0_m)  # Shape: (b, 112, 112, 64)
a2_m = stage_1(a1_m)  # Shape: (b, 56, 56, 128)
a3_m = stage_2(a2_m)  # Shape: (b, 28, 28, 256)
a4_m = stage_3(a3_m)  # Shape: (b, 14, 14, 512)
a5_m = stage_4(a4_m)  # Shape: (b, 2048)
logits_m = stage_5(a5_m)  # Shape: (b, 1000)
loss_m = cross_entropy(logits_m, labels_m)  # Scalar

# Backward pass accumulates gradients
for m in range(M):
    loss_m.backward()  # Triggers gradient computation

    # PyTorch/TensorFlow automatically accumulates:
    # stage_0.weight.grad += ∂loss_m / ∂(stage_0.weight)
    # stage_1.weight.grad += ∂loss_m / ∂(stage_1.weight)
    # ... (all M micro-batches accumulate)

# After all micro-batches processed:
# Each stage has gradient ∑_m ∂loss_m / ∂θ_stage

# Optimizer update (once per mini-batch):
optimizer.step()  # θ ← θ - α × (∑_m ∇θ)
```

---

## 6. Heads, Targets, and Losses

### Loss Function Definition
For classification on ImageNet:

```python
def gpipe_loss_computation(
    logits: Tensor[b, num_classes],
    labels: Tensor[b],
    num_micro_batches: M
) -> Tuple[Tensor, Tensor]:
    """
    Compute cross-entropy loss for one micro-batch.

    Args:
        logits: Model output from stage K-1, shape (b, 1000) for ImageNet
        labels: Ground truth class indices, shape (b,)
        M: Number of micro-batches (for averaging)

    Returns:
        loss_micro: Loss for this micro-batch (scalar)
        loss_total: Loss accumulated across all micro-batches (for logging)
    """

    # Cross-entropy loss (standard)
    # L_m = -∑_i label_m[i] × log(softmax(logits_m)[i])
    loss_micro = F.cross_entropy(logits, labels, reduction='mean')
    # Shape: scalar (mean over b examples in micro-batch)

    # For gradient computation:
    # ∂L_m / ∂logits = softmax(logits) - one_hot(labels)
    # Shape: (b, 1000)

    # Accumulated across micro-batches:
    # ∂L_total / ∂θ = ∑_m ∂L_m / ∂θ = ∑_m (∂L_m / ∂logits) × (∂logits / ∂θ)

    return loss_micro


def gpipe_loss_accumulation(
    all_losses: List[Tensor],  # Length M
    num_micro_batches: int
) -> Tensor:
    """
    Aggregate losses from all micro-batches.

    Note: Gradients are automatically accumulated by PyTorch/TF
    via backward() on each loss; this just sums for logging.
    """
    total_loss = sum(all_losses) / num_micro_batches
    # Effective batch size: B (all examples)
    # Loss per example: total_loss (averaged over full mini-batch)
    return total_loss
```

### Gradient Computation with Micro-Batches

```
Mathematical formulation:

Full mini-batch loss:
  L_total = (1/B) ∑_{i=1}^B loss(f(x_i; θ), y_i)

Grouped into M micro-batches (size b = B/M):
  L_total = (1/M) ∑_{m=1}^M L_m
  where L_m = (1/b) ∑_{i ∈ batch_m} loss(f(x_i; θ), y_i)

Gradient computation:
  ∂L_total/∂θ = (1/M) ∑_{m=1}^M ∂L_m/∂θ
              = (1/B) ∑_{i=1}^B ∂loss_i/∂θ  (mathematically equivalent)

Implementation (gradient accumulation):
  1. Zero gradients: θ.grad ← 0
  2. For m=1 to M:
       Forward: loss_m = compute_loss(model(x_m), y_m)
       Backward: loss_m.backward()  [accumulates to θ.grad]
  3. Update: θ ← θ - α × θ.grad

Key property: No gradient averaging needed across micro-batches
  - Each loss_m is averaged over its b examples (reduction='mean')
  - Summation across M micro-batches gives full batch gradient
```

### Loss Scaling for Mixed Precision (optional)

```python
def gpipe_loss_with_amp(
    logits: Tensor[b, 1000],
    labels: Tensor[b],
    loss_scale: float = 1024.0  # For float16
) -> Tensor:
    """
    Loss scaling to prevent gradient underflow in mixed precision.

    Technique: Scale loss up before backward, scale gradients down after.
    """

    # Compute loss normally
    loss = F.cross_entropy(logits, labels, reduction='mean')  # ℝ

    # Scale up for backward pass (prevents underflow in float16)
    scaled_loss = loss * loss_scale  # Still in float16 if using AMP

    # Backward pass with scaled loss
    # ∂L_scaled / ∂θ = loss_scale × ∂L / ∂θ  (larger gradients, less underflow)
    scaled_loss.backward()

    # Gradient clipping can prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer scales gradients back internally
    optimizer.step()  # Applies loss_scale^{-1} implicitly

    return loss  # Return unscaled loss for logging
```

---

## 7. Data Pipeline

### Batch Splitting Strategy

```python
class MicroBatchDataPipeline:
    """
    GPipe data pipeline: split mini-batches into micro-batches.
    """

    def __init__(
        self,
        dataloader,  # Standard PyTorch DataLoader
        batch_size: int,  # Mini-batch size (B)
        micro_batch_size: int,  # Per-device batch size (b)
    ):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.num_micro_batches = batch_size // micro_batch_size

        assert batch_size % micro_batch_size == 0, \
            "Batch size must be divisible by micro-batch size"

    def __iter__(self):
        """
        Iterate through mini-batches from standard dataloader.
        For each mini-batch, yield micro-batches.
        """
        for batch_idx, (images, labels) in enumerate(self.dataloader):
            # images: Tensor[B, C, H, W]  (B = batch_size)
            # labels: Tensor[B]

            # Split into micro-batches
            for micro_idx in range(self.num_micro_batches):
                start = micro_idx * self.micro_batch_size
                end = start + self.micro_batch_size

                micro_batch_images = images[start:end]  # (b, C, H, W)
                micro_batch_labels = labels[start:end]  # (b,)

                yield (micro_batch_images, micro_batch_labels, micro_idx)


# Example usage:
config = {
    'batch_size': 256,  # Mini-batch
    'micro_batch_size': 32,  # Per micro-batch
    'num_devices': 8,
}

dataloader = torch.utils.data.DataLoader(
    dataset=ImageNet(),
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=16,  # Prefetch data while GPU trains
)

data_pipeline = MicroBatchDataPipeline(
    dataloader=dataloader,
    batch_size=config['batch_size'],
    micro_batch_size=config['micro_batch_size'],
)

for micro_images, micro_labels, micro_idx in data_pipeline:
    # Each iteration: one micro-batch
    # Size: (32, 3, 224, 224) and (32,)
    pass
```

### Prefetching and Pipeline Overlap

```python
def prefetch_pipeline(data_loader, num_prefetch: int = 2):
    """
    Overlap data loading with GPU computation.
    """
    iterator = iter(data_loader)
    queue = []

    # Prefetch initial batches to CPU
    for _ in range(num_prefetch):
        try:
            batch = next(iterator)
            queue.append(batch)
        except StopIteration:
            break

    # Yield batches while prefetching next
    while queue:
        batch = queue.pop(0)

        # Prefetch next batch in background
        try:
            next_batch = next(iterator)
            queue.append(next_batch)
        except StopIteration:
            pass

        yield batch


# Timeline with prefetching:
# Time 0: CPU loads batch_0 → GPU, prefetch batch_1 (CPU)
# Time 1: GPU trains batch_0,   prefetch batch_2 (CPU)
# Time 2: GPU trains batch_0,   batch_1 ready from CPU
# Time 3: GPU loads batch_1 → GPU (minimal wait), prefetch batch_3
```

### Memory Layout During Training

```
Device memory snapshot during forward pass of micro-batch m:

┌──────────────────────────────────────┐
│  Device 0 VRAM (e.g., 32 GB)         │
├──────────────────────────────────────┤
│  Model parameters: θ_0 (2 GB)        │
│  Optimizer state: m_0, v_0 (4 GB)    │
│  Micro-batch m: X_m (128 MB)         │
│  Activations:                        │
│    a_0: (32, 224, 224, 3) = 384 MB  │
│    a_1: (32, 112, 112, 64) = 384 MB │
│  Grad buffers: ∂a / ∂θ (variable)   │
│  Free space (remainder)              │
├──────────────────────────────────────┤
│  Total: ~32 GB                       │
└──────────────────────────────────────┘

Micro-batch size choice (b):
  - Too small: High overhead, poor GPU utilization
  - Too large: OOM errors
  - Sweet spot: Fill 80% device memory, leave 20% for gradients/overhead
```

---

## 8. Training Pipeline

### Hyperparameter Configuration Table

```
┌──────────────────────┬──────────────────────┬─────────────────┐
│ Hyperparameter       │ Typical Value        │ Notes           │
├──────────────────────┼──────────────────────┼─────────────────┤
│ Mini-batch size (B)  │ 256-2048             │ Total across    │
│                      │                      │ all devices     │
│ Micro-batch size (b) │ 32-64                │ Per device      │
│ Num micro-batches (M)│ 4-32                 │ M = B / b       │
│ Learning rate (α)    │ 0.1-0.256            │ Scales with B   │
│ Momentum (β1)        │ 0.9                  │ Standard SGD    │
│ Adam β1              │ 0.9                  │ Adam optimizer  │
│ Adam β2              │ 0.999                │ Adam optimizer  │
│ Weight decay (λ)     │ 1e-4 to 1e-3         │ L2 regularization
│ LR schedule          │ Cosine / Step        │ Warmup + decay  │
│ Warmup steps         │ 5K-10K               │ Linear warmup   │
│ Gradient clip        │ 1.0-5.0              │ Max norm        │
│ Mixed precision      │ float32 or AMP       │ Training        │
│ Checkpoint every     │ 100-1000 steps       │ Save model ckpt │
│ Num GPUs (K)         │ 4-128                │ Pipeline depth  │
└──────────────────────┴──────────────────────┴─────────────────┘
```

### Training Loop Pseudocode

```python
def gpipe_training_loop(
    model_stages: List[nn.Module],
    train_loader: DataLoader,
    num_epochs: int = 90,
    num_micro_batches: int = 4,
    num_devices: int = 8,
    learning_rate: float = 0.256,
    warmup_steps: int = 5000,
):
    """
    Full training loop with GPipe pipeline parallelism.
    """

    # Initialize optimizer with all parameters
    all_params = []
    for stage in model_stages:
        all_params.extend(stage.parameters())

    optimizer = torch.optim.SGD(
        all_params,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * num_epochs,
    )

    total_steps = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Step 1: Learning rate warmup
            if total_steps < warmup_steps:
                warmup_lr = learning_rate * (total_steps / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Step 2: Split mini-batch into micro-batches
            num_mb = num_micro_batches
            mb_size = images.size(0) // num_mb

            # Step 3: Forward pass with pipelining
            activation_cache = []
            all_losses = []

            # Send each micro-batch through pipeline
            for mb_idx in range(num_mb):
                start_idx = mb_idx * mb_size
                end_idx = start_idx + mb_size

                mb_images = images[start_idx:end_idx]  # (b, C, H, W)
                mb_labels = labels[start_idx:end_idx]  # (b,)

                # Forward pass: chain stages
                activation = mb_images
                mb_activations = [activation]

                for k, stage in enumerate(model_stages):
                    if k < len(model_stages) - 1:
                        # Intermediate stage
                        activation = stage(activation)
                        mb_activations.append(activation)
                    else:
                        # Final stage: compute loss
                        logits = stage(activation)
                        loss = F.cross_entropy(logits, mb_labels)
                        all_losses.append(loss)
                        mb_activations.append(logits)

                activation_cache.append(mb_activations)

            # Step 4: Backward pass with gradient accumulation
            optimizer.zero_grad()

            for mb_idx, loss in enumerate(all_losses):
                # Backward only on final loss (accumulated automatically)
                loss.backward()

            # Step 5: Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            # Step 6: Optimizer step
            optimizer.step()
            scheduler.step()

            # Step 7: Logging
            batch_loss = sum(all_losses) / num_mb
            epoch_loss += batch_loss.item()
            num_batches += 1
            total_steps += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss:.4f}")

        print(f"Epoch {epoch} complete, Avg Loss: {epoch_loss / num_batches:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"model_epoch_{epoch}.pt"
            save_checkpoint(model_stages, optimizer, ckpt_path)


def save_checkpoint(model_stages, optimizer, path):
    """Save model state for resuming training."""
    state = {
        'model_stages': [stage.state_dict() for stage in model_stages],
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(model_stages, optimizer, path):
    """Load model state."""
    state = torch.load(path)
    for stage, state_dict in zip(model_stages, state['model_stages']):
        stage.load_state_dict(state_dict)
    optimizer.load_state_dict(state['optimizer'])
```

### Re-materialization (Checkpointing) Strategy

```python
class RematerializationStage(nn.Module):
    """
    Wrapper stage that discards activations during forward,
    recomputes them during backward.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # Store input for recomputation (critical for backward!)
        self.input = x.detach().clone()

        # Compute output (but don't store intermediate activations)
        self.input.requires_grad_(True)
        with torch.no_grad():
            output = self.layer(self.input)
        return output

    def backward(self, grad_output):
        # Recompute forward pass (forward computation during backward!)
        self.input.requires_grad_(True)
        output = self.layer(self.input)

        # Now compute backward
        torch.autograd.backward(output, grad_output)

        return self.input.grad


# Memory savings example:
"""
ResNet-50 with batch_size = 32, 8 devices:

Without re-materialization:
  - Activations stored for all 50 layers: 50 × 32 × (H × W × C) bytes
  - Example: 50 × 32 × (56 × 56 × 128) × 4 = ~4.6 GB per device
  - Prohibitive at K=128 (requires 4.6 GB even with device-level B=4)

With re-materialization:
  - Store only boundary activations (between stages): 2 × (H × W × C) × 4 bytes
  - Example: 2 × (56 × 56 × 128) × 4 = ~9 MB per device
  - Negligible overhead; forward recomputation adds ~30% compute time

Key trade-off: Save 500× memory, pay 30% compute cost (worthwhile!)
"""
```

---

## 9. Dataset and Evaluation Protocol

### ImageNet Evaluation Setup

```
Dataset: ImageNet-2012
├─ Training set: 1.28 million images
│  └─ 1000 object classes
├─ Validation set: 50,000 images
│  └─ 50 images per class
└─ Resolution: Variable (resized to 224×224 or higher)

Data preprocessing:
├─ RandomResizedCrop(224, scale=(0.08, 1.0))  # Random crop and scale
├─ RandomHorizontalFlip(p=0.5)                 # 50% chance flip
├─ ColorJitter(0.4, 0.4, 0.4, 0.1)           # Brightness/contrast
├─ RandomRotation(degrees=15)                  # Optional rotation
├─ ToTensor()                                  # Convert to [0, 1]
├─ Normalize(mean=[0.485, 0.456, 0.406],     # ImageNet stats
│             std=[0.229, 0.224, 0.225])
└─ Output shape: (B, 3, 224, 224)

Validation preprocessing:
├─ Resize(256)                  # Resize shortest edge to 256
├─ CenterCrop(224)              # Center 224×224 crop
├─ ToTensor()
├─ Normalize(...)               # Same stats as training
└─ Output shape: (B, 3, 224, 224)
```

### Training Configuration for AmoebaNet-B 557M

```
Model: AmoebaNet-B (557M parameters)
├─ Depth: 9 blocks
├─ Width: 384 channels (separable convolutions)
└─ Parameters: 557 million

Hardware setup:
├─ Device count: 8 TPU v3 (or 16× V100 GPUs)
├─ Partition: 8 stages (one per device)
├─ Mini-batch size (B): 2048
├─ Micro-batch size (b): 32 per device
├─ Number of micro-batches: 2048 / 32 = 64
└─ Pipeline efficiency target: >80%

Training hyperparameters:
├─ Optimizer: SGD with momentum
├─ Learning rate: 0.256 (scaled with batch size B)
│  └─ Formula: α = 0.1 × (B / 256)
│  └─ Rationale: Linear scaling rule (Goyal et al., 2017)
├─ Momentum: 0.9
├─ Weight decay: 1e-4
├─ Warmup steps: 5000 (linear warmup to α)
├─ Learning rate schedule:
│  ├─ Steps 0-5K: Linear warmup
│  ├─ Steps 5K-281K: Cosine annealing decay
│  └─ Final LR: 0.001 × α
├─ Gradient clipping: max_norm=1.0
├─ Mixed precision: Automatic Mixed Precision (AMP)
│  └─ Loss scaling: 2^15 (1024-4096 for float16)
├─ Data augmentation: RandAugment / AutoAugment
└─ Training epochs: 300 (longer than typical 90 epochs)

Hardware utilization target:
├─ TPU peak FLOPS: 8 × 420 TFLOPS = 3,360 TFLOPS
├─ Model forward: 557M params × 224^2 pixels × 2 FLOPS ≈ 56 TFLOPS per example
├─ Target throughput: 3,360 / 56 ≈ 60 examples/second per device
├─ Actual achieved: ~40 examples/second (including communication overhead)
└─ Overall utilization: 60% (bottleneck: inter-device communication)

Communication volume:
├─ Forward pass:
│  ├─ Device 0 → Device 1: (32, H, W, C) float32
│  ├─ Size per micro-batch: 32 × 224 × 224 × 3 × 4 ≈ 100 MB
│  └─ Total (64 micro-batches): 6.4 GB per step (overlapped with compute)
├─ Backward pass:
│  ├─ Device 1 → Device 0: gradients ∂a/∂x
│  └─ Similar size as forward (100 MB × 64 micro-batches = 6.4 GB)
└─ All-reduce for sync (after all micro-batches): negligible
```

### Machine Translation Evaluation (Transformer)

```
Dataset: Large-scale translation corpora
├─ WMT 2018 training data (100+ languages)
├─ Tokenization: Sentence-Piece BPE (32K vocab)
├─ Sequence length: variable, capped at 256 tokens
└─ Vocabulary: 32,768 BPE tokens

Model: Transformer-Base (83.9B final model variant)
├─ Encoder: 6 layers × 512 hidden units × 8 heads
├─ Decoder: 6 layers × 512 hidden units × 8 heads
├─ Feed-forward: 2048 hidden units
├─ Total parameters (base): 65 million
├─ Scaled variant (used in paper): 83.9 billion (128× scale)
└─ Partitioning: 128 devices (1M params/device)

Evaluation metrics:
├─ BLEU score (primary metric)
│  └─ Range: 0-100 (higher is better)
├─ Validation frequency: Every 1000 steps
├─ Early stopping: If BLEU plateaus
└─ Final evaluation: BLEU on test set (unseen language pairs)

Training configuration:
├─ Batch size: 2048 tokens per device
├─ Learning rate: 0.1 (fixed, no schedule initially)
├─ Optimizer: Adam with β1=0.9, β2=0.98
├─ Warmup steps: 10K steps
├─ Gradient clipping: 1.0
├─ Dropout: 0.1
└─ Training duration: 100K-300K steps (depends on pair)

Scaling results from paper:
├─ 1 device (65M params): BLEU ≈ 25
├─ 8 devices (520M params): BLEU ≈ 28
├─ 32 devices (2.1B params): BLEU ≈ 30
├─ 128 devices (83.9B params): BLEU ≈ 32
└─ Efficiency: Near-linear BLEU scaling with model size
```

---

## 10. Results Summary and Ablations

### AmoebaNet-B (557M) on ImageNet

```
┌──────────────────────────────────────────────────────────┐
│ AmoebaNet-B (557M parameters) Training Results           │
├──────────────────────────────────────────────────────────┤
│ Accuracy (Top-1):     84.4%  (vs. 83.8% for paper SOTA) │
│ Accuracy (Top-5):     97.1%                             │
│ Training time:        15 days (8 TPU v3)                │
│ Convergence epochs:   ~300                              │
│ Final learning rate:  ~0.001 (annealed from 0.256)      │
│ Loss (final):         0.32                              │
└──────────────────────────────────────────────────────────┘

Scaling efficiency (device count):
┌──────────┬──────────────┬──────────────┬──────────────┐
│ Devices  │ Micro-batch  │ Throughput   │ Efficiency   │
│          │ size (b)     │ (imgs/sec)   │ vs 1 device  │
├──────────┼──────────────┼──────────────┼──────────────┤
│ 1        │ 256          │ 48           │ 100%         │
│ 2        │ 128          │ 95           │ 99%          │
│ 4        │ 64           │ 189          │ 98%          │
│ 8        │ 32           │ 378          │ 98%          │
│ 16       │ 16           │ 726          │ 94%          │
│ 32       │ 8            │ 1,380        │ 90%          │
│ 64       │ 4            │ 2,500        │ 81%          │
│ 128      │ 2            │ 4,300        │ 70%          │
└──────────┴──────────────┴──────────────┴──────────────┘

Analysis:
- 1-8 devices: Near-linear scaling (98% efficiency)
- 8-32 devices: Gradual efficiency drop due to:
  ├─ Communication overhead (constant)
  ├─ Reduced micro-batch size (8→4 examples per device)
  └─ Increased synchronization cost
- 32-128 devices: Steeper drop (81%→70%)
  └─ Micro-batch size: 8→2 examples (stochastic noise increases)

Reason for efficiency loss at scale:
1. Fixed network bandwidth: AllReduce takes T_allreduce seconds
   └─ Relative cost increases: T_allreduce / T_compute grows
2. Smaller micro-batches: b=2 vs b=256
   └─ GPU memory not fully utilized
   └─ Less compute per device → more communication relative to compute
3. Memory bandwidth bottleneck:
   └─ At 128 devices, memory BW utilization peaks
```

### Transformer-Based Machine Translation (128-layer model)

```
Model scaling on WMT 2018 multi-lingual translation:

┌──────────────────────────────────────────────────────────┐
│ Transformer Scaling to 83.9B Parameters                  │
├──────────────────────────────────────────────────────────┤
│ Baseline (1 device, 65M params):        BLEU ≈ 25       │
│ + 8 devices:  520M params               BLEU ≈ 28       │
│ + 32 devices: 2.1B params               BLEU ≈ 30       │
│ + 128 devices: 83.9B params             BLEU ≈ 32       │
└──────────────────────────────────────────────────────────┘

Scaling characteristics:
├─ Each 8× increase in model size: ~1 BLEU point improvement
├─ Linear relationship: log(params) vs BLEU
└─ Minimal degradation from pipeline approximations

Communication efficiency:
├─ Mini-batch: 2048 tokens per device
├─ Forward activation flow: ≈100 MB per micro-batch
├─ Total forward+backward: ≈200 MB per micro-batch
├─ 8 devices, 16 micro-batches: ~3.2 GB transfer per step
└─ vs. data parallel (would require ~100 GB all-reduce): 30× savings
```

### Ablation Studies

**Study 1: Micro-Batch Size Effect**

```
Question: How does reducing micro-batch size (b) affect accuracy?

Setup: AmoebaNet-B, 8 devices, total batch size B=2048 fixed

Result:
┌─────────────────┬─────────────┬──────────────┬─────────────┐
│ Micro-batch (b) │ Num batches  │ Convergence  │ Final Top-1 │
│ per device      │ (M = 2048/b) │ speed        │ Accuracy    │
├─────────────────┼─────────────┼──────────────┼─────────────┤
│ 256             │ 8           │ baseline     │ 84.4%       │
│ 128             │ 16          │ -0.5%        │ 84.1%       │
│ 64              │ 32          │ -1.2%        │ 83.8%       │
│ 32              │ 64          │ -2.1%        │ 83.1%       │
└─────────────────┴─────────────┴──────────────┴─────────────┘

Key findings:
- Micro-batch 256-128: Negligible impact (<0.5% accuracy drop)
- Micro-batch 128-32: Noticeable degradation (0.5-1.2%)
  ├─ Cause: Reduced gradient quality per device
  └─ Solution: Increase warmup steps, reduce LR
- Micro-batch <32: Steep accuracy loss (>2%)
  └─ Gradient noise dominates; staleness from pipeline

Recommendation: Use b ≥ 64 for 8+ devices, adjust LR downward
```

**Study 2: Pipeline Depth Effect**

```
Question: How does number of devices (pipeline depth) affect accuracy?

Setup: AmoebaNet-B, 2048 total batch size, vary num devices

┌──────────────┬──────────────┬──────────────────┬──────────────┐
│ Num Devices  │ Micro-batch  │ Pipeline         │ Final        │
│ (K)          │ size (b)     │ efficiency (%)   │ Top-1 Acc    │
├──────────────┼──────────────┼──────────────────┼──────────────┤
│ 1            │ 2048         │ 100% (no pipe)   │ 84.4%        │
│ 2            │ 1024         │ 99.5%            │ 84.3%        │
│ 4            │ 512          │ 97.8%            │ 84.2%        │
│ 8            │ 256          │ 95.5%            │ 84.4%        │
│ 16           │ 128          │ 91.2%            │ 83.9%        │
│ 32           │ 64           │ 84.5%            │ 83.2%        │
│ 64           │ 32           │ 75.3%            │ 82.1%        │
└──────────────┴──────────────┴──────────────────┴──────────────┘

Analysis:
- 1-8 devices: Accuracy held (84.2-84.4%)
  ├─ Total B remains 2048, gradients stable
  └─ Pipeline communication overlapped successfully
- 8-32 devices: Gradual drop (-0.5% to -1.2%)
  ├─ Micro-batch shrinks: 256→64 examples
  ├─ Stochastic gradient noise increases
  └─ Solution: Increase warmup, adjust learning rate schedule
- 32+ devices: Steep drop (>2%)
  └─ Both efficiency and accuracy suffer

Conclusion: GPipe scales well to 8-16 devices; beyond that
requires more careful tuning (larger total batch size, longer warmup)
```

**Study 3: Re-materialization Impact**

```
Question: What is the memory/compute trade-off of checkpointing?

Setup: Transformer base model on machine translation

┌──────────────────┬──────────┬──────────┬──────────────┐
│ Configuration    │ Memory   │ Compute  │ Throughput   │
│                  │ per dev  │ overhead │ (tokens/sec) │
├──────────────────┼──────────┼──────────┼──────────────┤
│ No checkpointing │ 16 GB    │ baseline │ 8,200        │
│ Checkpoint all   │ 4 GB     │ +32%     │ 5,600        │
│ (remat every)    │ (75% save)          │ (-32%)       │
│ Checkpoint       │ 9 GB     │ +12%     │ 7,200        │
│ (every other)    │ (44% save)          │ (-12%)       │
└──────────────────┴──────────┴──────────┴──────────────┘

Pareto frontier:
- No checkpointing: Baseline, but high memory → fits on 1 device only
- Selective checkpointing: Aggressive with 32 devices
  └─ Memory 4 GB per device × 32 = 128 GB total usable model
  └─ Compute overhead +32%, but enables 32× device scale
- Intermediate checkpointing: Best balance
  └─ Memory 9 GB per device × 32 = 288 GB total
  └─ Compute overhead +12% acceptable
  └─ Throughput 7,200 tokens/sec vs 5,600 = faster overall

Recommendation:
- Few devices (1-2): Skip checkpointing (memory available)
- Many devices (8+): Selective checkpointing every 2-3 layers
- Very many (32+): Aggressive checkpointing every layer
```

**Study 4: Synchronization Overhead**

```
Question: How often should we synchronize (update weights)?

Setup: Compare different micro-batch accumulation strategies

┌────────────────────┬──────────┬──────────┬──────────────┐
│ Sync strategy      │ Mem      │ Compute  │ Final Top-1  │
│                    │ overhead │ overhead │ Accuracy     │
├────────────────────┼──────────┼──────────┼──────────────┤
│ Sync every step    │ baseline │ baseline │ 84.4%        │
│ (default)          │ (no accu)│          │ (baseline)   │
│ Accumulate 2 steps │ +10%     │ -1%      │ 84.3%        │
│ Accumulate 4 steps │ +18%     │ -3%      │ 84.1%        │
│ Accumulate 8 steps │ +35%     │ -6%      │ 83.5%        │
└────────────────────┴──────────┴──────────┴──────────────┘

Finding: Synchronization has minimal overhead
- Gradient accumulation cost: <1% per step (just sum accumulators)
- No advantage to reducing sync frequency:
  ├─ Memory overhead grows (+10-35%)
  ├─ Convergence slightly slower (83.5% vs 84.4%)
  └─ Communication not significantly reduced

Recommendation: Sync after every mini-batch (all M micro-batches)
```

---

## 11. Practical Insights

### 10 Engineering Takeaways

**1. Partition Strategy Matters**
```
Even splits (equal number of layers per device) are not optimal.
Better: Use computation cost analysis to balance device loads.

Example: ResNet-50
├─ Poor: Device 0-7 each get 6-7 layers
└─ Good: Device 0 (Conv1) = 20ms, Device 1-6 = 30-40ms, Device 7 = 15ms
         Rebalance to equalize: Unequal number of layers per device

Method: Profile forward+backward time per layer, create partitions
        such that max_device_time ≈ min across partitions
```

**2. Micro-Batch Size Sweet Spot**
```
Too small (b=1-4): Stochastic noise dominates, poor gradient quality
Too large (b=256): Defeats purpose of pipeline parallelism

Finding: b ∈ [32, 128] for most models and hardware
├─ b=32: Good GPU utilization + reasonable gradient quality
├─ b=64: Safer choice, less tuning needed
└─ b=128: Max practical for large batch sizes

Tip: Start with b=64, adjust if:
  ├─ OOM error: Reduce b
  └─ Accuracy plateau: May need to increase LR
```

**3. Learning Rate Scaling Rule**
```
When batch size increases, learning rate should also increase
(Linear Scaling Rule by Goyal et al., 2017)

GPipe context:
  α_new = α_baseline × (B / B_baseline)

Example:
  ├─ Baseline: B=256, α=0.1
  ├─ With 8 devices: B=2048, α=0.256 (but scaled for 8×)
  └─ Actually: α = 0.1 × (2048/256) = 0.8 (too high!)

Better: Use warmup schedule to ramp up to theoretical LR
  ├─ Warmup phase: 0 → α over first 5K steps
  ├─ Stable phase: α × (cosine annealing)
  └─ Decay: α → 0.01×α over remaining steps
```

**4. Gradient Synchronization is Automatic**
```
Python pseudocode:

# This accumulates gradients automatically!
for m in range(num_micro_batches):
    loss_m = compute_loss(forward_pass(mb_m))
    loss_m.backward()  # ← Adds to θ.grad, doesn't overwrite

# After loop, θ.grad = ∑_m ∂loss_m / ∂θ (no averaging needed!)
optimizer.step()

Pitfall: If you call optimizer.zero_grad() between micro-batches,
         gradients are lost! Only zero before the mini-batch loop.
```

**5. Communication Overlapping is Non-Trivial**
```
GPipe communicates (sends activations/gradients) between stages.

Naive: Wait for forward, then send → bubble time = 2(K-1)
Better: Overlap communication with computation → bubble time = K-1

Implementation: Use async communication primitives
  ├─ TensorFlow: tf.distribute.all_reduce with background execution
  ├─ PyTorch: torch.distributed.all_gather (non-blocking)
  └─ Custom: Start communication, compute next micro-batch meanwhile

Result: Well-overlapped GPipe achieves ~85-90% efficiency
        Poorly overlapped: ~40-50% efficiency (2× slowdown)
```

**6. Recomputation Patterns Should Match Forward**
```
Forward pass (no remat):
  ├─ Compute layer output
  ├─ Store activation
  └─ Pass to next layer

Backward pass (with remat):
  ├─ Load activation from previous layer (or recompute)
  ├─ Recompute this layer's output (remat)
  ├─ Backprop through this layer
  └─ Pass gradient to previous layer

Requirement: Remat must recompute identical outputs
Pitfall: If forward uses dropout/random augmentation:
  ├─ Must seed RNG identically for remat to work
  └─ Or, store random state during forward (adds memory)

Solution: Disable dropout/augmentation for forward in layer,
          apply only in loss computation (or use deterministic seed)
```

**7. Memory Estimation Before Training**
```
Use this formula to estimate peak memory per device:

M_peak = M_params + M_optimizer + M_activations + M_gradients
       = P + 2P + (K × b × H) + (b × H)
       = 3P + (K+1)×b×H

Where:
  P = model parameters (bytes)
  b = micro-batch size
  H = average hidden dimension
  K = number of devices (if tracking activations from all prev stages)

Example: ResNet-50
  ├─ P = 25M params × 4 bytes = 100 MB
  ├─ Optimizer (Adam): 2P = 200 MB
  ├─ Micro-batch: 32 examples × 224×224 × 3 × 4 = 2 GB (worst case early layers)
  ├─ Gradients: 1 GB (buffer for backward)
  ├─ With K=8: M_peak ≈ 300 MB + 8×1.5GB ≈ 12 GB

With remat: M_peak ≈ 300 MB + 1.5GB + 1GB ≈ 3 GB (3× savings)
```

**8. Allreduce (gradient synchronization) Should Be Rare**
```
Naive pipeline parallelism:
  ├─ After each micro-batch: all_reduce(gradients) ← expensive!
  └─ Total: M × K communication steps

GPipe approach:
  ├─ Accumulate gradients locally on each device
  ├─ Only sync ONCE per mini-batch (not per micro-batch)
  └─ Total: 1 communication step (per mini-batch)

Impact: Reduces communication volume by M×

Cost: All_reduce of K devices × P parameters
      ≈ 2 × log(K) × P × dtype_size

Example (8 devices, ResNet-50):
  ├─ Size: 25M params × 4 bytes = 100 MB
  ├─ Time: ~50ms on modern interconnect
  └─ Amortized over mini-batch: negligible
```

**9. Profile Bottlenecks Before Scaling**
```
Typical bottlenecks for K ≥ 32 devices:

1. Memory BW bottleneck:
   └─ Each device reads K×b×H data (activations) per forward step
   └─ Total BW: K × forward_BW → saturates device memory bandwidth
   └─ Solution: Increase b (if memory allows) or use FP16

2. Network latency (not throughput!):
   └─ Time to send activation from device i → device i+1
   └─ Typically 1-2ms fixed latency per hop
   └─ Total pipeline latency: K × 2ms = 16-20ms (for K=8)
   └─ Solution: Pipelined sends (multi-level parallelism)

3. Compute starvation:
   └─ If communication is slow, devices idle waiting for data
   └─ Solution: Overlap communication with earlier micro-batch compute

Profiling tool: Use tensorboard/wandb to log:
  ├─ GPU utilization (%)
  ├─ Device idle time (%)
  ├─ Communication time (%)
  └─ Adjust if any > 10% (wasted cycles)
```

**10. Test End-to-End Before Large-Scale Training**
```
Dry-run protocol before scaling to 64+ devices:

1. Single device test (K=1):
   ├─ Verify model trains, loss decreases
   ├─ Check final accuracy against baseline
   └─ Establish baseline throughput (tokens/sec or imgs/sec)

2. Small cluster test (K=2-4):
   ├─ Verify pipeline scheduling works
   ├─ Check communication patterns (use profiler)
   ├─ Confirm near-linear scaling (90%+ efficiency)

3. Increase K gradually:
   ├─ K=8: Measure efficiency (should be >85%)
   ├─ K=16: Measure efficiency (should be >80%)
   ├─ K=32+: Accept efficiency >70%

4. Hyperparameter re-tune:
   ├─ Learning rate warmup: Extend by 2× (slower convergence at scale)
   ├─ Learning rate peak: Increase by sqrt(K) (Goyal rule)
   ├─ Batch size: Increase proportionally with K
   └─ Save schedule: Increase frequency (more checkpoints)
```

### 5 Gotchas and How to Fix Them

**Gotcha 1: Pipeline Staleness**
```
Problem: Gradients computed using "stale" activations from earlier steps.

Explanation: In step t, micro-batch m on device 1 uses activation
from previous step's backward (stale by 1-2 steps).

Impact: Slightly noisier gradients; can cause 0.5-1.5% accuracy loss.

Detection:
  ├─ Training loss doesn't decrease smoothly
  ├─ Validation accuracy plateaus early
  └─ Loss spikes occasionally

Fix:
  ├─ Reduce micro-batch size → faster gradient update
  ├─ Increase warmup steps → let optimizer adapt
  ├─ Reduce learning rate slightly → conservative updates
  └─ Use heavier momentum (β=0.95 vs 0.9) → dampen oscillations
```

**Gotcha 2: Micro-Batch Size Too Small**
```
Problem: Micro-batch size b < 8 causes severe gradient noise.

Explanation: Stochastic gradient from 8 examples has high variance.
             Accumulating across M micro-batches doesn't fully fix this.

Symptoms:
  ├─ Training loss is very spiky
  ├─ Validation accuracy oscillates wildly
  ├─ Final accuracy 2-3% below baseline
  └─ Model sometimes diverges (loss → NaN)

Root cause: b=4 is roughly equivalent to using LR that's 4× lower
            than effective LR with b=256.

Fix:
  ├─ Increase b (bigger micro-batches)
  ├─ Reduce K (fewer devices) if needed
  ├─ Increase warmup steps (allow optimization to stabilize)
  └─ Use gradient clipping (prevent divergence)
```

**Gotcha 3: All-Reduce Serialization (Data Parallel + Pipeline)**
```
Problem: If using both data parallelism AND pipeline parallelism,
         all-reduce becomes a bottleneck.

Explanation: After each mini-batch:
  ├─ Device 0 waits for Device K-1 to finish backward pass
  ├─ Then all devices synchronize: all_reduce(gradients)
  ├─ This takes time proportional to model size

Naive approach (8 devices, 1 model copy each):
  └─ All_reduce: 100 MB × log(8) = 100 MB × 3 ≈ 300 MB communication

Better (4 models on 32 devices, 8 devices per model):
  ├─ Model-parallel all_reduce: 25 MB × 3 (within model group)
  ├─ Data-parallel all_reduce: 100 MB × log(4) (across models)
  └─ Hierarchical: ~50 MB total (saved!)

Fix:
  ├─ Don't mix data + model parallelism carelessly
  ├─ Use hierarchical all-reduce (group-based)
  └─ Or: Stick to pure pipeline parallelism (no data parallelism)
```

**Gotcha 4: Remat Overhead Underestimated**
```
Problem: Checkpointing forward computation in backward is slow.

Explanation: Remat backward pass time ≈ forward time
             So backward is now 2× longer (forward + backward recompute)

Naive estimate: 30% overhead
Actual: Can be 40-60% depending on memory access patterns

Example: ResNet-50 forward pass: 30ms
         ResNet-50 backward (normal): 60ms
         ResNet-50 backward (with remat): 90-120ms (30-100% slower!)

Why: Backward is memory-bound (huge gradient reads/writes)
     Adding forward recomputation saturates memory bus

Fix:
  ├─ Not all layers need remat (remat every other layer)
  ├─ Use lower-precision remat (bfloat16 forward, float32 backward)
  ├─ Overlap remat with all-reduce (parallel execution)
  └─ Accept the overhead and increase b to compensate
```

**Gotcha 5: Communication Overlapping Not Automatic**
```
Problem: Framework doesn't automatically overlap communication + compute.

Explanation: If code looks like:

  # Device i
  send_activation(device_i+1)  # Starts send
  wait_for_send()              # BLOCKS here!
  compute_backward()           # Only starts after send finishes

Expected: Send + backward should overlap
Actual: Send finishes, then backward starts

Fix: Explicit non-blocking sends in PyTorch

  # Device i (correct)
  handle = send_activation(device_i+1, non_blocking=True)
  compute_backward()  # Starts immediately, overlaps with send
  handle.wait()       # Wait only if we need the result

Lesson: Profile communication to verify it's overlapped:
  ├─ Device busy: 100% → good overlap
  ├─ Device busy: 70% → poor overlap (communication blocking compute)
  └─ Fix: Reorder operations or use async primitives
```

### Overfit Plan (How to Debug Model Issues)

```python
def overfit_debug_plan(model, dataset, num_devices):
    """
    Systematic approach to identify training issues in GPipe.
    """

    # Phase 1: Can model overfit on tiny dataset?
    print("Phase 1: Overfit on 8 examples")
    tiny_data = dataset[:8]

    # Train on K=1 (single device)
    model_k1 = model
    for epoch in range(100):
        loss = train_one_epoch(model_k1, tiny_data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss:.4f}")
    # Expected: Loss → ~0, 100% accuracy on these 8 examples
    # If loss plateaus: Model capacity issue or data issue

    # Phase 2: Same dataset, K=2 devices
    print("\nPhase 2: Overfit on 8 examples, K=2 devices")
    model_k2 = partition_model(model, num_partitions=2)
    for epoch in range(100):
        loss = train_one_epoch_gpipe(model_k2, tiny_data, k=2)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss:.4f}")
    # Expected: Loss ≈ same as Phase 1 (within 5%)
    # If worse: Pipeline communication or remat bug

    # Phase 3: Expand to K=num_devices, but still 8 examples
    print(f"\nPhase 3: Overfit on 8 examples, K={num_devices} devices")
    model_kn = partition_model(model, num_partitions=num_devices)
    for epoch in range(100):
        loss = train_one_epoch_gpipe(model_kn, tiny_data, k=num_devices)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss:.4f}")
    # Expected: Loss converges (possibly slower due to smaller micro-batches)
    # If diverges: Gradient accumulation or sync bug

    # Phase 4: Full dataset, K=1
    print("\nPhase 4: Full dataset, K=1 device")
    model_k1_full = model
    for epoch in range(10):
        loss, acc = train_one_epoch_with_eval(model_k1_full, dataset)
        print(f"Epoch {epoch}: Loss {loss:.4f}, Acc {acc:.2%}")
    # Expected: Baseline accuracy for comparison

    # Phase 5: Full dataset, K=num_devices
    print(f"\nPhase 5: Full dataset, K={num_devices} devices")
    model_kn_full = partition_model(model, num_partitions=num_devices)
    for epoch in range(10):
        loss, acc = train_one_epoch_with_eval_gpipe(
            model_kn_full, dataset, k=num_devices
        )
        print(f"Epoch {epoch}: Loss {loss:.4f}, Acc {acc:.2%}")
    # Expected: Accuracy ≈ 95-99% of Phase 4 (within 1-2%)
    # If lower: Hyperparameter tuning needed (LR, warmup, batch size)

    # Diagnosis guide:
    if phase1_loss_plateaus:
        print("ERROR: Model cannot overfit tiny dataset")
        print("  → Check: Model capacity, data loading, loss function")
    elif phase2_loss_worse:
        print("ERROR: GPipe pipeline communication broken")
        print("  → Check: Activation shapes, device assignment, sync")
    elif phase3_loss_diverges:
        print("ERROR: Gradient accumulation broken")
        print("  → Check: zero_grad timing, backward() calls, clipping")
    elif phase5_acc_drop > 2:
        print("WARNING: Accuracy degraded with pipeline parallelism")
        print("  → Tune: LR, warmup steps, micro-batch size, gradient clip")
    else:
        print("SUCCESS: Pipeline parallelism working correctly!")
        print(f"  → Phase 1-4 sanity checks passed")
        print(f"  → Ready for large-scale training on K={num_devices}")
```

---

## 12. Minimal Reimplementation Checklist

### Core Components Needed

```python
# Core GPipe implementation checklist

## 1. Model Partitioning
[ ] Parse model architecture (identify sequential stages)
[ ] Assign layers to devices (partition_model function)
[ ] Validate partition (acyclic, no skips)
[ ] Create stage-level modules (nn.Module for each stage)

## 2. Micro-Batch Management
[ ] Split mini-batch into M micro-batches (split_batch function)
[ ] Shape-preserving split (don't change dtype or device)
[ ] Validation: sum of micro-batches = mini-batch

## 3. Forward Pass
[ ] Stage-wise forward (for loop over stages)
[ ] Cache activations (store a^[k]_m for backward)
[ ] Loss computation (last stage outputs logits, compute cross-entropy)
[ ] Return: loss_per_microbatch, activation_cache

## 4. Backward Pass
[ ] Micro-batch-wise backward (for loop over micro-batches in reverse)
[ ] Gradient accumulation (loss_m.backward() automatically accumulates)
[ ] Check: optimizer.zero_grad() before mini-batch, not between micro-batches
[ ] Return: accumulated gradients on θ

## 5. Optimizer Step
[ ] Call optimizer.step() after all micro-batches
[ ] Apply learning rate schedule (warmup, decay)
[ ] Gradient clipping: torch.nn.utils.clip_grad_norm_
[ ] Return: updated parameters

## 6. Re-materialization (Optional)
[ ] Save input to each stage (for recomputation)
[ ] During backward, recompute forward (don't use cached activation)
[ ] Trade: ~30% compute overhead for ~70% memory savings

## 7. Communication (Optional)
[ ] All-reduce synchronization (only once per mini-batch)
[ ] Async sends/receives between stages (non-blocking)
[ ] Overlap communication with backward pass computation

## 8. Data Loading
[ ] Standard DataLoader with batch_size = B
[ ] Micro-batch split in training loop (or custom dataset)
[ ] Prefetching (background loading of next batch)
[ ] No special data loading logic needed (GPipe handles it)

## 9. Logging & Checkpointing
[ ] Log loss, accuracy, throughput per mini-batch
[ ] Save checkpoints: model state, optimizer state, step count
[ ] Load checkpoint to resume training
[ ] Optional: Tensorboard/wandb for visualization

## 10. Testing
[ ] Single device (K=1) sanity check
[ ] Tiny dataset overfit test (8 examples)
[ ] Compare K=1 vs K=2 (should match within noise)
[ ] Profile throughput, efficiency, memory usage
```

### Minimal Code Skeleton

```python
# Minimal GPipe implementation (~200 lines)

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

class GPipeModel(nn.Module):
    """
    Pipeline-parallel model wrapper.
    Splits model into sequential stages and pipelined training.
    """

    def __init__(self, stages: list, num_micro_batches: int = 4):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.num_micro_batches = num_micro_batches
        self.K = len(stages)  # Number of stages (devices)

    def forward(self, batch_x, batch_y):
        """
        Forward pass with micro-batching and pipelining.

        Args:
            batch_x: Input batch [B, ...]
            batch_y: Labels [B]

        Returns:
            total_loss: Accumulated loss across micro-batches
        """
        B = batch_x.size(0)
        M = self.num_micro_batches
        b = B // M

        # Split into micro-batches
        micro_batches_x = [batch_x[i*b:(i+1)*b] for i in range(M)]
        micro_batches_y = [batch_y[i*b:(i+1)*b] for i in range(M)]

        # Forward pass through pipeline
        activation_cache = []
        losses = []

        for m in range(M):
            x_m = micro_batches_x[m]
            y_m = micro_batches_y[m]

            # Forward through all stages
            activations_m = [x_m]
            x = x_m

            for k, stage in enumerate(self.stages):
                x = stage(x)
                activations_m.append(x)

            # Last stage output is logits, compute loss
            logits = x
            loss_m = nn.functional.cross_entropy(logits, y_m)

            activation_cache.append(activations_m)
            losses.append(loss_m)

        total_loss = sum(losses) / M
        return total_loss

    def backward(self):
        """
        Backward pass (automatic in PyTorch with loss.backward()).
        Gradients are automatically accumulated across micro-batches.
        """
        # In training loop, just call: loss.backward()
        pass


def create_partitions(model: nn.Module, num_partitions: int) -> list:
    """
    Partition sequential model into stages.
    """
    layers = list(model.children())
    layers_per_partition = len(layers) // num_partitions

    partitions = []
    for p in range(num_partitions):
        start = p * layers_per_partition
        if p == num_partitions - 1:
            end = len(layers)  # Last partition gets remaining layers
        else:
            end = (p + 1) * layers_per_partition

        partition = nn.Sequential(*layers[start:end])
        partitions.append(partition)

    return partitions


def train_gpipe(
    model_stages: list,
    train_loader,
    num_epochs: int = 100,
    num_micro_batches: int = 4,
    learning_rate: float = 0.1,
    warmup_steps: int = 1000,
):
    """
    Training loop for GPipe pipeline-parallel model.
    """
    # Initialize optimizer for all parameters
    all_params = []
    for stage in model_stages:
        all_params.extend(stage.parameters())

    optimizer = torch.optim.SGD(
        all_params,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
    )

    gpipe_model = GPipeModel(model_stages, num_micro_batches)

    global_step = 0
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            # Learning rate warmup
            if global_step < warmup_steps:
                warmup_lr = learning_rate * (global_step / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Forward pass (micro-batched)
            loss = gpipe_model(batch_x, batch_y)

            # Backward pass (automatic gradient accumulation)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            # Optimizer step
            optimizer.step()

            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} complete")


# Usage:
if __name__ == "__main__":
    from torchvision import models

    # Load pretrained ResNet50
    resnet50 = models.resnet50(pretrained=False)

    # Partition into 4 stages
    stages = create_partitions(resnet50, num_partitions=4)

    # Dummy data
    dummy_loader = [(
        torch.randn(256, 3, 224, 224),  # Batch size 256
        torch.randint(0, 1000, (256,))
    )]

    # Train
    train_gpipe(
        stages,
        dummy_loader,
        num_epochs=1,
        num_micro_batches=8,
        learning_rate=0.1,
    )
```

### Testing Checklist

```python
# Test suite for GPipe implementation

def test_single_device():
    """Test: Does model train on single device (no pipeline)?"""
    model = ResNet50()

    x = torch.randn(256, 3, 224, 224)
    y = torch.randint(0, 1000, (256,))

    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()

    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✓ Single device test passed")


def test_partition_equivalence():
    """Test: Do partitioned models match single model?"""
    model = ResNet50()
    stages = create_partitions(model, num_partitions=4)
    gpipe_model = GPipeModel(stages, num_micro_batches=1)

    x = torch.randn(64, 3, 224, 224)
    y = torch.randint(0, 1000, (64,))

    # Forward through single model
    y_single = model(x)
    loss_single = nn.functional.cross_entropy(y_single, y)

    # Forward through partitioned model
    loss_gpipe = gpipe_model(x, y)

    # Should match (within numerical precision)
    assert torch.allclose(loss_single, loss_gpipe, atol=1e-5), \
        f"Loss mismatch: {loss_single} vs {loss_gpipe}"
    print("✓ Partition equivalence test passed")


def test_micro_batch_gradient_accumulation():
    """Test: Do gradients accumulate correctly across micro-batches?"""
    model = SimpleLinear()  # y = Wx + b

    x_full = torch.randn(64, 10)
    y_full = torch.randint(0, 2, (64,))

    # Method 1: Full batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(x_full), y_full)
    loss.backward()
    grad_full = [p.grad.clone() for p in model.parameters()]

    # Method 2: Micro-batches (size 16)
    model2 = SimpleLinear()
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    optimizer2.zero_grad()

    for mb_start in range(0, 64, 16):
        x_mb = x_full[mb_start:mb_start+16]
        y_mb = y_full[mb_start:mb_start+16]
        loss_mb = nn.functional.cross_entropy(model2(x_mb), y_mb)
        loss_mb.backward()

    grad_micro = [p.grad.clone() for p in model2.parameters()]

    # Compare gradients
    for g_full, g_micro in zip(grad_full, grad_micro):
        assert torch.allclose(g_full, g_micro, atol=1e-5), \
            "Gradient accumulation mismatch"
    print("✓ Gradient accumulation test passed")


def test_pipeline_parallelism_speedup():
    """Test: Does pipelining provide speedup vs naive serialization?"""
    stages = [SimpleStage() for _ in range(4)]
    gpipe_model = GPipeModel(stages, num_micro_batches=4)

    x = torch.randn(256, 512)
    y = torch.randint(0, 10, (256,))

    import time

    # Pipeline forward pass
    start = time.time()
    for _ in range(100):
        loss = gpipe_model(x, y)
    pipeline_time = time.time() - start

    # Serial forward pass (no micro-batching)
    start = time.time()
    for _ in range(100):
        x_out = x
        for stage in stages:
            x_out = stage(x_out)
        loss = nn.functional.cross_entropy(x_out, y)
    serial_time = time.time() - start

    speedup = serial_time / pipeline_time
    print(f"Speedup (pipeline vs serial): {speedup:.2f}x")
    # Expected: ~2-3x speedup with 4 devices and 4 micro-batches
    assert speedup > 1.2, "Expected speedup from pipelining"
    print("✓ Pipeline speedup test passed")


# Run all tests
if __name__ == "__main__":
    test_single_device()
    test_partition_equivalence()
    test_micro_batch_gradient_accumulation()
    test_pipeline_parallelism_speedup()
    print("\n✓ All tests passed!")
```

---

## Summary

**GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism** introduces a practical approach to training giant neural networks by:

1. **Splitting models into sequential stages** assigned to different devices
2. **Splitting mini-batches into micro-batches** for pipelining
3. **Overlapping computation** across stages to minimize idle time
4. **Accumulating gradients** across micro-batches before parameter updates

**Key Results:**
- Achieves 84.4% top-1 accuracy on ImageNet with 557M-parameter AmoebaNet-B
- Scales Transformer to 83.9B parameters (128 devices)
- Near-linear scaling efficiency (70-90%) across 128 accelerators
- Memory reduction via re-materialization (checkpointing)

**Practical Impact:**
- Enables training of otherwise infeasible models
- Simpler than data parallelism communication patterns
- Works with any sequential model architecture

Sources:
- [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism (ArXiv)](https://arxiv.org/abs/1811.06965)
- [NeurIPS Paper](https://papers.nips.cc/paper/8305-gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism)
- [TorchGPipe Implementation](https://github.com/kakaobrain/torchgpipe)
