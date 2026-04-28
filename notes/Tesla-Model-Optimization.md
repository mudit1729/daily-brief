# Model Optimization & Latency — Tesla ML Engineer Technical Screen

Generated: April 27, 2026

Purpose: a focused guide for the model optimization and latency portion of the Tesla ML Engineer screen (~8% of total weight, 10–20 min). Tesla's AI page explicitly calls out throughput, latency, correctness, determinism, quantization, pruning, and real-time constraints. This guide maps directly to those themes.

Screen weight note: optimization questions are often follow-ups to a model design question. Expect "Your model is accurate but 3x too slow on FSD hardware. What do you do?"

---

## Interview Mindset

The single most important thing you can say before anything else:

```text
"Before I touch the model, I'd profile to find where time is actually going.
 Optimizations applied to the wrong bottleneck waste time and can silently
 regress accuracy."
```

Standard answer scaffold for any "model is too slow" question:

```text
Step 1 — Profile first.
  Identify: is time in data loading, CPU-GPU transfer, kernel compute,
  or memory bandwidth? Never assume.

Step 2 — Identify the actual bottleneck.
  Data loading → fix DataLoader, prefetch, pin memory.
  CPU-GPU transfer → async transfer, reduce small transfers.
  Compute-bound kernel → quantization, architecture changes, fusion.
  Memory-bandwidth-bound → reduce model size, mixed precision.

Step 3 — Smallest change first.
  Try inference_mode, static shapes, batch tuning before pruning/quantizing.

Step 4 — Validate accuracy by slice after each change.
  At Tesla: regression on night driving, rain, rare objects, edge cases.
  Never report only top-line accuracy — check safety-critical slices.

Step 5 — Iterate. Combine techniques only after each is validated alone.
```

Red flags interviewers watch for:

```text
- "I would quantize the model" said before profiling.
- Optimizing the training loop when the question is about inference latency.
- Reporting overall mAP without checking accuracy on rare or safety slices.
- Treating INT8 as free — calibration data, per-layer sensitivity, outlier activations all matter.
```

---

## Profiling

### PyTorch Profiler

Minimal usage to find the expensive ops:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model.eval()
inputs = torch.randn(1, 3, 640, 640, device="cuda")  # [B, C, H, W]

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,  # set True for full callstack, slower
) as prof:
    with record_function("model_inference"):  # label shows in trace
        with torch.inference_mode():
            out = model(inputs)  # [B, num_classes] or [B, anchors, 6]

# Sort by total CUDA time to find the real bottleneck
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20,
))

# Export Chrome trace for visual inspection
prof.export_chrome_trace("trace.json")  # open in chrome://tracing
```

What to read in the output:

```text
Self CPU time:   time spent in this op's Python/C++ code (not children).
CPU total time:  includes all children.
CUDA time:       time on GPU — this is what matters for inference latency.
# Calls:         high call count on a tiny op signals Python loop overhead.
```

Wrap a training step to profile the full pipeline:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=3, active=5),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profiler"),
) as prof:
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device)
        with record_function("forward"):
            logits = model(x)  # [B, C]
        with record_function("loss"):
            loss = criterion(logits, y)  # scalar
        with record_function("backward"):
            loss.backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        prof.step()  # advance profiler schedule
        if step >= 9:
            break
```

### cProfile for Python Overhead

Use when PyTorch profiler shows surprisingly low CUDA time — the bottleneck may be Python:

```python
import cProfile, pstats, io

pr = cProfile.Profile()
pr.enable()
for x, y in loader:  # one epoch worth
    logits = model(x.to(device))
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)  # top 30 Python functions
print(s.getvalue())
```

Signals that Python is the bottleneck:

```text
- DataLoader workers are 0 (default) and GPU sits idle between batches.
- Collate function is slow (complex augmentation in Python).
- Model has conditional Python control flow inside forward().
```

### GPU Profiling — Nsight Systems

For kernel-level analysis beyond PyTorch profiler:

```bash
# profile one run, 10 seconds of capture
nsys profile \
    --trace=cuda,cudnn,nvtx \
    --output=run_profile \
    python inference.py

# open in Nsight Systems GUI or analyze CLI
nsys stats run_profile.nsys-rep
```

Key things to look for in Nsight:

```text
- Long gaps between kernels: GPU is idle waiting for CPU (pipeline stall).
- Kernel occupancy: low occupancy means the kernel under-uses the hardware.
- Memory transfer bars: large H2D or D2H transfers between compute kernels.
- Repeated small kernels: many tiny ops instead of one fused op.
```

### When to Suspect Data Loading vs Compute

```text
GPU utilization < 70% and CUDA time << wall time  → data loading bottleneck.
GPU utilization ≈ 100% and model is large          → compute bottleneck.
Large CPU-GPU transfer time in profiler            → pinned memory / async transfer fix.
Many Python calls per batch, low CUDA time         → Python overhead / TorchScript candidate.
```

---

## Quantization

The most impactful inference optimization. Also the most likely to silently break accuracy if misapplied.

### Precision Format Trade-offs

| Format | Bits | Range | Notes |
|--------|------|-------|-------|
| FP32   | 32   | ±3.4×10³⁸ | Training default. Full precision. |
| FP16   | 16   | ±65504 | Fast on Volta+. Overflows on large logits/loss. Needs `GradScaler`. |
| BF16   | 16   | ±3.4×10³⁸ | Same range as FP32 (8 exponent bits). Less precision. No overflow risk. Preferred for training on Ampere+. |
| INT8   | 8    | –128…127 | ~2–4× inference speedup. Needs calibration or QAT. Loses precision in activations. |
| INT4   | 4    | –8…7 | Aggressive. Used in LLM weight compression. High accuracy risk for dense vision models. |

When each format breaks:

```text
FP16 training: loss diverges when gradients are very small (underflow to zero)
               or activations overflow (>65504). Fix: GradScaler.
FP16 inference: softmax and log-softmax with large logit spread → NaN.
               Fix: cast those layers back to FP32.
BF16 inference: low mantissa precision hurts models with many small weight differences.
INT8: activations with outliers (common in transformers) cause large quantization error.
      Fix: per-channel quantization, SmoothQuant, or QAT.
```

### Quantization Math — Scale and Zero-Point

INT8 quantization maps a real value `x` to an 8-bit integer `q` via an affine transform:

```text
q = round(x / scale) + zero_point         # quantize
x = scale * (q - zero_point)              # dequantize

# Asymmetric (unsigned, typical for activations after ReLU):
#   q ∈ [0, 255]
#   scale      = (x_max - x_min) / 255
#   zero_point = round(-x_min / scale)              # clipped to [0, 255]
#
# Symmetric (signed, typical for weights):
#   q ∈ [-127, 127]   (we use 127 not 128 to keep the range symmetric)
#   scale      = max(|x_max|, |x_min|) / 127
#   zero_point = 0
```

Per-tensor vs per-channel: per-tensor uses one (scale, zero_point) for the whole weight; per-channel uses one per output channel. Per-channel is the default for conv/linear weights — outlier channels would otherwise blow up the per-tensor scale and saturate the rest.

A matmul `y = W·x` becomes (with symmetric W, asymmetric x):

```text
y ≈ scale_w * scale_x * Σ q_w * (q_x - zp_x)
  = scale_w * scale_x * (q_w · q_x  -  zp_x * Σ q_w)   # the second term is precomputable
```

This is why INT8 inference is fast: the inner sum `q_w · q_x` runs as int8 GEMM on tensor cores, and the `Σ q_w` correction is precomputed offline.

### Mixed Precision Training — `torch.amp`

```python
import torch
from torch.cuda.amp import GradScaler
from torch import autocast

model = model.to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # loss scaler prevents FP16 gradient underflow

for x, y in loader:
    x, y = x.to("cuda"), y.to("cuda")  # x: [B, C, H, W], y: [B]
    optimizer.zero_grad(set_to_none=True)

    with autocast(device_type="cuda", dtype=torch.float16):
        # inside autocast: eligible ops run in FP16
        logits = model(x)  # [B, num_classes], computed in FP16
        loss = criterion(logits, y)  # scalar, FP32 accumulation inside CELoss

    # scale the loss, backprop in scaled space, unscale before step
    scaler.scale(loss).backward()   # compute gradients
    scaler.unscale_(optimizer)      # unscale for gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)          # update parameters
    scaler.update()                 # adjust scale factor for next iter
```

What `GradScaler` does:

```text
1. Multiplies loss by a large scale factor S before backward().
2. Gradients are computed in the scaled space — avoids FP16 underflow.
3. Before optimizer.step(), divides gradients by S (unscale_).
4. If unscaled gradients contain inf/nan: skip the step, halve S.
5. If N consecutive clean steps: double S (S grows back over time).
```

BF16 does not need `GradScaler` (same range as FP32):

```python
with autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)  # BF16 compute, no overflow risk
    loss = criterion(logits, y)
loss.backward()  # no scaler needed
optimizer.step()
```

### Post-Training Quantization (PTQ)

Fast, no retraining. Requires a representative calibration dataset (min 200–1000 samples) to measure activation ranges.

```python
import torch
from torch.quantization import quantize_dynamic, prepare, convert
from torch.quantization import get_default_qconfig

# --- Option A: Dynamic quantization (weights INT8, activations FP32) ---
# Good for linear-heavy models (LSTM, Transformer). Minimal accuracy loss.
model_dynamic = quantize_dynamic(
    model.cpu(),
    qconfig_spec={torch.nn.Linear},  # only quantize Linear layers
    dtype=torch.qint8,
)
# No calibration data needed. Weights are quantized statically, activations at runtime.

# --- Option B: Static PTQ (weights + activations INT8) ---
# More speedup. Requires calibration.
model.eval()
model.qconfig = get_default_qconfig("x86")  # or "qnnpack" for ARM
prepare(model, inplace=True)  # insert observer hooks to collect stats

# Calibration pass — run representative inputs through the model
with torch.inference_mode():
    for x, _ in calibration_loader:  # 200-1000 representative samples
        model(x)                     # observers record activation ranges

convert(model, inplace=True)  # replace float ops with INT8 quantized ops
```

Calibration pitfall: if calibration data doesn't cover the activation range of rare inputs (night scenes, glare, rain at Tesla), INT8 will saturate and silently lose accuracy on exactly those safety-critical cases.

### Quantization-Aware Training (QAT)

When PTQ accuracy is unacceptable. Simulates quantization during training so the model learns to be robust to the precision loss.

```python
from torch.quantization import prepare_qat, convert, get_default_qat_qconfig

model.train()
model.qconfig = get_default_qat_qconfig("x86")
prepare_qat(model, inplace=True)  # insert fake-quantize nodes

# fine-tune for a few epochs with fake quantization active
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)  # lower LR for QAT
for epoch in range(5):
    for x, y in train_loader:
        logits = model(x)   # [B, C] — quantization error is simulated here
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

model.eval()
convert(model, inplace=True)  # replace fake-quant with real INT8 ops
```

PTQ vs QAT — when to choose:

```text
PTQ: model recovers accuracy well with good calibration data.
     Use first. Much faster than QAT.
QAT: PTQ drops >1% accuracy on critical slices, or model has unusual activation
     distributions (transformer attention, depthwise conv). Budget 3–10 extra
     training epochs.
```

---

## Pruning

Removing weights to reduce model size and potentially inference cost. Key insight: **unstructured pruning rarely gives real speedup** without special sparse kernels. Structured pruning gives real speedup.

### Structured vs Unstructured

| Type | What is removed | Real speedup? | Accuracy risk |
|------|----------------|---------------|---------------|
| Unstructured (weight) | Individual weights (irregular) | No — hardware runs dense ops | Low at moderate sparsity |
| Structured (channel) | Entire channels / filters | Yes — smaller tensor shapes | Higher — removes full feature maps |
| Structured (head) | Attention heads | Yes — fewer matmuls | Medium — some heads redundant |

Why unstructured often doesn't give speedup: modern GPU tensor cores process dense blocks. A weight tensor with 50% zeros takes the same time as a dense tensor unless you use sparse CUDA kernels (e.g., `torch.sparse`), which have overhead of their own at typical sparsity levels.

### PyTorch Pruning API

```python
import torch.nn.utils.prune as prune

# --- Unstructured magnitude pruning: remove 30% of smallest-magnitude weights ---
prune.l1_unstructured(
    module=model.layer1[0].conv1,
    name="weight",
    amount=0.30,  # fraction of weights to zero out
)
# This adds a 'weight_mask' buffer and 'weight_orig' parameter.
# The effective weight = weight_orig * weight_mask (applied on forward).

# Make pruning permanent (remove mask, store sparse weights)
prune.remove(model.layer1[0].conv1, "weight")

# --- Global magnitude pruning across entire model ---
parameters_to_prune = [
    (module, "weight")
    for module in model.modules()
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.40,  # 40% sparsity globally
)

# --- Structured pruning: remove 20% of channels by L1 norm ---
prune.ln_structured(
    module=model.layer1[0].conv1,
    name="weight",
    amount=0.20,
    n=1,    # L1 norm
    dim=0,  # dim 0 = output channels
)
```

Iterative magnitude pruning workflow:

```text
1. Train model to convergence.
2. Prune a small fraction (e.g., 10–20%) of weights by lowest magnitude.
3. Fine-tune for a few epochs to recover accuracy.
4. Evaluate accuracy on full held-out set AND critical slices.
5. If accuracy holds, repeat from step 2 with higher total sparsity target.
6. Stop when accuracy on safety slices drops below threshold.
```

Lottery ticket hypothesis (one-line): a large trained network contains a small sparse subnetwork ("winning ticket") that — if trained from scratch with the same initialization — matches the full network's accuracy. Used to motivate iterative pruning + retraining.

---

## Knowledge Distillation

Train a small student model to mimic a large teacher model's output distribution, not just its hard labels. Often more effective than pruning for getting a compact model.

### Teacher-Student Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# assume teacher and student are both trained classification models
# teacher: [B, C] logits  |  student: [B, C] logits

def distillation_loss(
    student_logits: torch.Tensor,  # [B, C]
    teacher_logits: torch.Tensor,  # [B, C]
    true_labels: torch.Tensor,     # [B], int64
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Combined hard-label + soft-label distillation loss.
    temperature: flattens teacher distribution to expose dark knowledge.
    alpha: weight on soft (distillation) loss; (1-alpha) on hard loss.
    """
    T = temperature

    # Soft targets: teacher distribution at temperature T
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)  # [B, C]
    log_soft_student = F.log_softmax(student_logits / T, dim=-1)  # [B, C]

    # KL divergence: how different is student from teacher?
    # Multiply by T^2 to compensate for gradient scaling from temperature
    kl_loss = F.kl_div(
        input=log_soft_student,
        target=soft_teacher,
        reduction="batchmean",
    ) * (T ** 2)  # [scalar]

    # Hard label loss: standard cross-entropy on true labels
    ce_loss = F.cross_entropy(student_logits, true_labels)  # [scalar]

    return alpha * kl_loss + (1.0 - alpha) * ce_loss  # [scalar]
```

When distillation helps:

```text
- Large teacher exists and inference cost is the constraint (common at Tesla edge).
- Student is architecturally much smaller than teacher.
- Dataset is small — soft targets from teacher act as label smoothing.
- Replacing teacher ensemble with single student.
```

When distillation doesn't help:

```text
- Teacher and student have similar capacity — not much extra knowledge to transfer.
- Teacher accuracy is low — student learns from a poor teacher.
- Task has hard, discrete labels with no ambiguity (binary detection without confidence).
- No compute budget for running teacher during student training.
```

---

## Architecture-Level Optimization

The highest-leverage category. Changing the architecture often beats post-hoc techniques.

### Input Resolution Reduction

Cutting input resolution from 1280×720 to 640×360 reduces FLOPs by ~4× for conv layers.

```text
Check: does accuracy hold at lower resolution? Run ablation on full eval set.
Tesla note: near-field objects (pedestrians at crosswalk) may degrade — check those slices.
```

### Backbone Downgrade

```text
ResNet-50 → MobileNetV3-Large: ~10× fewer FLOPs, ~3× fewer parameters.
ViT-Large  → ViT-Base:         ~4× fewer FLOPs, same patch size.
EfficientNet-B7 → B0:          ~9× fewer FLOPs.
```

Verify with distillation from larger backbone — often recovers accuracy.

### Depthwise Separable Convolution

Replaces standard Conv2d with depthwise + pointwise, reducing FLOPs significantly:

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels,  # one filter per input channel
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,  # 1×1 conv to mix channels
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, H, W]
        x = self.depthwise(x)   # [B, in_channels, H', W']
        x = self.pointwise(x)   # [B, out_channels, H', W']
        return self.act(self.bn(x))
```

FLOPs reduction vs standard Conv2d with kernel size k, in channels Cin, out channels Cout:

```text
Standard:   FLOPs = k² × Cin × Cout × H × W
Depthwise:  FLOPs = k² × Cin × H × W  +  Cin × Cout × H × W
Ratio:      1/Cout + 1/k²   (typically ~8–9× reduction for k=3)
```

### Early Exit / Cascade

Run a small, fast model first. Only pass uncertain examples to the heavy model.

```text
Stage 1 (cheap): small classifier → confident detections exit here (~70% of frames).
Stage 2 (heavy): full model runs only on ambiguous or high-priority regions.

Tesla relevance: common pattern in the occupancy + object detection pipeline.
                 Saves compute on highway driving where the scene is simple.
```

### ROI Processing

Rather than running the full heavy model on every patch, run a cheap model first to propose regions of interest, then run the expensive model only on those regions.

```text
Applicable to: pedestrian detection, traffic sign reading, lane change decisions.
Trade-off: adds latency for the ROI proposal step — profile total pipeline time.
```

### Operator Fusion

Conv + BN + ReLU are commonly fused at inference time to remove redundant reads/writes:

```python
# PyTorch 2.x torch.compile fuses ops automatically on supported backends
model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune"
# mode="reduce-overhead": minimizes kernel launch overhead (good for small batches)
# mode="max-autotune":    tries many kernel configs, slow to compile, fast at runtime

# Manual fusion: fold BN into preceding Conv weights after training
torch.nn.utils.fusion.fuse_conv_bn_eval(conv_layer, bn_layer)  # returns fused conv
```

Fusing Conv+BN removes BN's separate forward pass: BN parameters (weight, bias, running mean/var) are folded into the Conv weight and bias at export time.

### FlashAttention (one-liner you should know)

FlashAttention is an IO-aware exact attention kernel: it tiles Q/K/V into SRAM-sized blocks, computes softmax-weighted attention block-by-block, and never materializes the full N×N attention matrix in HBM. Result: O(N) memory instead of O(N²) and a 2–4× wall-clock speedup on long sequences. Available via `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.x picks FlashAttention automatically on supported GPUs) or `flash_attn` package. Relevant at Tesla for transformer-based perception/planning stacks where long context (lots of tokens per scene) is the bottleneck.

---

## Runtime & Deployment

### ONNX Export and TensorRT

```python
import torch

dummy_input = torch.randn(1, 3, 640, 640, device="cuda")  # [B, C, H, W]

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=["images"],
    output_names=["logits"],
    dynamic_axes=None,  # None = static shapes (faster TensorRT compilation)
)
```

Then build a TensorRT engine from the ONNX file:

```bash
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.trt \
    --fp16 \                    # enable FP16 precision
    --workspace=4096            # MB of GPU workspace
```

Runtimes at a glance:

```text
ONNX Runtime:   easy to use, portable, supports CPU/CUDA/TensorRT EP.
TensorRT:       NVIDIA-only, highest throughput, kernel auto-tuning, layer fusion.
OpenVINO:       Intel CPU/VPU inference, good for non-NVIDIA edge hardware.
torch.compile:  native PyTorch 2.x, good for research, less optimal than TensorRT for production.
```

### Static vs Dynamic Shapes

```text
Static shapes:  input dimensions are fixed at engine build time.
                TensorRT fuses and tunes kernels specifically for that shape.
                Significantly faster than dynamic (often 10–30% throughput gain).

Dynamic shapes: input height/width can vary at runtime.
                Useful for multi-scale inference or variable-length sequences.
                TensorRT builds optimization profiles for a range of shapes.
                Prefer static when possible for edge deployment.
```

### Batch Size Tuning

```text
Batch 1 (streaming):  lowest latency per request, lowest GPU utilization.
Batch N (offline):    highest throughput. Find sweet spot with profiler.
Rule of thumb:        double batch size until latency doubles — that is the
                      memory-bandwidth knee point.
```

### Determinism vs Performance

```python
import torch

# Enable deterministic ops — every run produces the same result
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# Disable cuDNN benchmark mode — benchmark mode picks fastest non-deterministic kernel
torch.backends.cudnn.benchmark = False  # set True for max throughput (non-deterministic)
```

Trade-off at Tesla:

```text
Determinism: required for regression testing, debugging NaN/inf, safety validation.
             Needed when comparing two model versions — must be same run-to-run.
benchmark=True: ~5–15% throughput gain on fixed input sizes. Use in production
                inference when reproducibility is not the goal.
```

### `torch.inference_mode()` vs `torch.no_grad()`

```python
# no_grad: disables gradient accumulation. Tensors still participate in autograd.
with torch.no_grad():
    out = model(x)  # no gradient graph built, but version counter still tracked

# inference_mode: stricter. Tensors created inside cannot be used in autograd later.
#                 Lower overhead than no_grad. Preferred for pure inference.
with torch.inference_mode():
    out = model(x)  # fastest option for eval-only code
```

```text
Use inference_mode for: validation loops, benchmark runs, production inference.
Use no_grad for:        cases where you might re-enter autograd with the output.
Neither replaces model.eval() — BatchNorm and Dropout still need eval mode.
```

### Pinned Memory and Async Data Transfer

```python
# DataLoader: pin_memory=True allocates CPU tensors in pinned (page-locked) memory.
# This allows non-blocking GPU transfer — CPU and GPU overlap.
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,          # parallel CPU workers for loading
    pin_memory=True,        # pinned memory for fast H2D transfers
    persistent_workers=True,  # workers stay alive between epochs
)

# non_blocking=True: H2D transfer happens asynchronously (CPU does not wait)
for x, y in loader:
    x = x.to(device, non_blocking=True)  # [B, C, H, W] transferred async
    y = y.to(device, non_blocking=True)  # [B] transferred async
    # GPU will wait for transfer automatically before using x in kernels
    logits = model(x)  # [B, C]
```

Without pinned memory: each batch requires a system call to page-lock memory before GPU transfer, stalling the pipeline.

---

## Decision Framework — "It's 3x Too Slow. What Do You Do?"

The step-by-step answer Tesla expects:

```text
Step 1 — Profile. Do not guess.
  Run torch.profiler, check CUDA time vs wall time.
  Is GPU utilization high? Is data loading the bottleneck?

Step 2 — Check the easy wins first.
  - Add torch.inference_mode() and model.eval() if missing.
  - Increase DataLoader num_workers and enable pin_memory.
  - Try static input shapes (disable dynamic shapes in ONNX/TRT).
  - Try torch.compile(model, mode="reduce-overhead").

Step 3 — Check batch size.
  Is there idle GPU time between batches? Increase batch size.
  Is latency measured per-sample? Batch=1 is often the constraint at edge.

Step 4 — Mixed precision.
  Try autocast with FP16 or BF16 for compute-bound layers.
  Profile again — check actual speedup vs expected.

Step 5 — Architecture changes.
  Reduce input resolution (fastest, biggest impact on conv layers).
  Downgrade backbone (MobileNet, EfficientNet-Lite).
  Add depthwise separable conv.

Step 6 — Quantization (PTQ first).
  Collect calibration data covering rare scenarios.
  Apply static INT8 PTQ. Profile speedup.
  Validate accuracy on full set AND safety-critical slices.

Step 7 — QAT if PTQ hurts accuracy.
  3–5 epochs of QAT fine-tuning.
  Re-evaluate all slices.

Step 8 — Structured pruning + fine-tune.
  Prune 20% of channels, fine-tune, evaluate.
  Repeat iteratively.

Step 9 — Knowledge distillation.
  If a larger/slower model exists with higher accuracy, distill into
  the current architecture or a smaller one.

Step 10 — Export to TensorRT.
  Build a TensorRT engine with static shapes.
  Run trtexec benchmark, compare vs target latency.

Tesla-specific gates at each step:
  - Accuracy must not regress on night/rain/rare-object slices.
  - Determinism must be verifiable for the regression test suite.
  - Latency measured on the actual target hardware (HW3/HW4), not cloud GPU.
```

---

## Top Asked Questions

**Q: Explain AMP and what GradScaler does.**

```text
AMP (Automatic Mixed Precision) runs forward pass ops in FP16 (or BF16) to gain
compute speed while accumulating losses and updating weights in FP32 for stability.
GradScaler addresses the FP16 gradient underflow problem: it multiplies the loss by
a large scale factor before backward(), computes gradients in the scaled space
(avoiding underflow to zero), then divides gradients back before the optimizer step.
If a gradient contains inf or nan (overflow), GradScaler skips that optimizer step
and reduces the scale. Over time it adaptively finds a stable scale factor.
```

**Q: Why does FP16 sometimes diverge during training?**

```text
Two root causes:
1. Underflow: very small gradients (common in early training or deep networks)
   fall below FP16's minimum representable value (~6×10^-5) and become zero.
   GradScaler fixes this.
2. Overflow: large activations or logits exceed FP16's max (65504) and become inf,
   which then propagates NaN. Fix: clip gradients, use BF16 instead (same range
   as FP32), or cast overflow-prone layers (loss, softmax) to FP32.
```

**Q: PTQ vs QAT — when do you need QAT?**

```text
Start with PTQ. Use QAT when:
- PTQ drops >0.5–1% accuracy on your primary metric, or
- Accuracy on rare/safety slices drops measurably, or
- Model has large activation outliers (common in transformers — attention scores
  can have extreme magnitudes that saturate INT8 scale), or
- The target hardware requires INT8 and the model is sensitive (e.g., small models
  where each layer's precision loss compounds more).
QAT costs 3–10 extra training epochs but usually recovers PTQ accuracy loss.
```

**Q: Why does pruning often not give the expected speedup?**

```text
Unstructured pruning sets individual weights to zero but does not change tensor
shapes. Modern GPU kernels (cuBLAS, cuDNN tensor cores) operate on dense matrices —
a matrix with 50% zeros takes the same time as a fully dense matrix unless sparse
kernels are used. Sparse CUDA kernels have their own overhead and typically only
win at very high sparsity (>70–80%). Structured pruning (channel/filter removal)
actually changes tensor dimensions, so standard dense kernels run faster on smaller
tensors. If speedup is the goal, prefer structured pruning.
```

**Q: How do you maintain determinism across runs?**

```python
import torch, random, numpy as np

def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)   # raises if non-deterministic op used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     # disable kernel auto-selection
    # CUBLAS_WORKSPACE_CONFIG must be set in environment for full determinism:
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

```text
Caveats: some ops (e.g., atomicAdd-based reductions in PyTorch scatter) have no
deterministic implementation. torch.use_deterministic_algorithms(True) will raise
an error at the offending op — you then choose between algorithm substitution
or accepting non-determinism for that specific op.
```

**Q: What's the difference between TorchScript and ONNX?**

```text
TorchScript:
  - PyTorch-native IR. Supports torch.jit.script (tracing control flow) and
    torch.jit.trace (records one execution path).
  - Portable across C++ PyTorch runtime without Python dependency.
  - Limited to ops in the TorchScript op set.
  - Good for: edge deployment with LibTorch, mobile (iOS/Android via PyTorch Mobile).

ONNX:
  - Open standard IR. Exportable from PyTorch, TensorFlow, JAX, etc.
  - Wide backend support: ONNX Runtime, TensorRT, OpenVINO, CoreML.
  - Does not support all PyTorch ops — complex control flow requires tracing a
    specific path.
  - Good for: cross-framework deployment, TensorRT integration (ONNX → TRT).

At Tesla: ONNX → TensorRT is the common production path for NVIDIA hardware.
TorchScript may be used for early prototyping or non-NVIDIA targets.
```

**Q: What is `torch.compile` and when should you use it?**

```text
torch.compile (PyTorch 2.x) uses TorchDynamo to capture the model's computation
graph and passes it to a backend compiler (default: Inductor → Triton kernels).
It performs op fusion, kernel selection, and graph optimization automatically.

When to use:
- Training speedup: typically 10–40% for transformer/CNN workloads.
- Inference speedup: competitive with TorchScript, simpler to apply.
- mode="reduce-overhead": reduce Python/kernel launch overhead (good for small batches).
- mode="max-autotune": exhaustive kernel search (slow compile, fastest runtime).

Limitations:
- First call is slow (compilation). Use warmup iterations before benchmarking.
- Recompiles if input shapes change (prefer static shapes or mark dynamic dims).
- Graph breaks on unsupported Python patterns (fallback to eager, lower speedup).
- Not a replacement for TensorRT in production NVIDIA deployment.
```

**Q: How do you validate that an optimization didn't regress the model?**

```text
Tesla-oriented answer:
1. Run full eval set — check top-level metric (mAP, accuracy) is within threshold.
2. Slice evaluation: separately check performance on rare classes, night/rain/fog,
   long-tail objects (cyclists, construction workers), edge-case distances.
3. Numerical regression gate: compare model outputs on a fixed set of test inputs
   between baseline and optimized model. Flag any output diff > tolerance.
4. Determinism check: run optimized model twice with the same seed — outputs must match.
5. Latency check: measure on target hardware (not cloud GPU), multiple runs, report P50/P99.
A/B gate: optimized model ships only if metric delta on safety slices is within SLA.
```

---

## Common Pitfalls

- **Optimizing before profiling.** The most common mistake. Quantization applied to a data-loading bottleneck delivers zero speedup while risking accuracy.

- **Calibration set too small or not representative.** INT8 ranges computed on 50 clean highway images will saturate on night, rain, or construction scenes. Calibration must cover the tail of the activation distribution.

- **Accuracy regression on rare slices.** INT8 or pruning may preserve overall mAP while quietly degrading precision on rare classes (cyclists, road debris). Always slice-evaluate after every optimization step.

- **Reporting speedup on cloud GPU instead of target hardware.** A 2× speedup on an A100 may be a 1.1× speedup on the edge inference chip. Always benchmark on the actual deployment target.

- **Dropping to FP16 globally without checking loss-sensitive layers.** Softmax, log-softmax, and cross-entropy loss accumulate errors in FP16. Cast those specific layers to FP32 manually when using mixed precision for inference.

- **Forgetting `model.eval()` during inference benchmarking.** BatchNorm uses batch statistics in train mode (adds noise to latency) and Dropout randomly zeros activations (changes output). Always call `model.eval()` before benchmarking.

- **Assuming structured pruning is free.** After channel pruning, the next layer's input dimension changes — you must update all downstream layers. Weight-only pruning APIs in PyTorch do not do this automatically; you need to rebuild the model or use a framework that handles connectivity (e.g., torch-pruning library).

- **torch.compile compilation time counted in benchmark.** The first forward pass after `torch.compile` triggers compilation and takes seconds. Always run 5–10 warmup iterations before measuring latency.

- **Train/inference numerical mismatch.** Using `autocast` during training but not during inference (or vice versa) changes the numeric path. Models with BatchNorm may behave differently if running stats are collected under FP16 but inference is FP32. Keep precision consistent or explicitly validate.

- **Ignoring P99 latency.** Optimizations that reduce mean latency may increase tail latency (P99, P999) due to memory pressure or kernel launch variability. Tesla's real-time constraints care about worst-case, not average.
