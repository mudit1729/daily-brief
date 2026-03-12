# PyTorch Refresher & Interview Cheatsheet

Complete revision guide — basics, internals, best practices, and top interview questions.

---

# 1. Tensor Fundamentals

## Creation

```python
import torch

x = torch.tensor([1, 2, 3])                    # from list, infers dtype=int64
x = torch.tensor([1.0, 2.0], dtype=torch.float32)  # explicit dtype
x = torch.zeros(3, 4)                          # 3×4 zeros, float32
x = torch.ones(2, 3, device='cuda')            # directly on GPU
x = torch.empty(5, 5)                          # uninitialized (fast, garbage values)
x = torch.randn(3, 4)                          # standard normal N(0,1)
x = torch.rand(3, 4)                           # uniform [0,1)
x = torch.arange(0, 10, 2)                     # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, steps=5)              # [0.0, 0.25, 0.5, 0.75, 1.0]
x = torch.eye(3)                               # 3×3 identity matrix
x = torch.full((2, 3), fill_value=7)           # 2×3 filled with 7
x = torch.zeros_like(other)                    # same shape/dtype/device as other
x = torch.from_numpy(np_array)                 # shares memory with numpy array
```

## Dtypes You Must Know

```python
torch.float32   # (torch.float)  — default for weights, activations
torch.float16   # (torch.half)   — mixed precision, saves VRAM, can overflow
torch.bfloat16  #                — wider range than fp16, preferred on A100+
torch.float64   # (torch.double) — rarely needed, double precision
torch.int64     # (torch.long)   — required for class labels, indices
torch.int32     # (torch.int)    — rarely used directly
torch.bool      #                — masks, comparisons
```

## Shape Operations

```python
x = torch.randn(2, 3, 4)

x.shape                       # torch.Size([2, 3, 4]) — same as x.size()
x.ndim                        # 3 — number of dimensions
x.numel()                     # 24 — total elements (2*3*4)

# Reshape — must preserve numel, may copy
x.reshape(6, 4)               # [6, 4] — contiguous-safe reshape
x.view(6, 4)                  # [6, 4] — requires contiguous memory, no copy
x.view(-1, 4)                 # [6, 4] — infer first dim automatically

# Add/remove dims
x.unsqueeze(0)                # [1, 2, 3, 4] — add dim at position 0
x.unsqueeze(-1)               # [2, 3, 4, 1] — add dim at end
x.squeeze()                   # removes ALL dims of size 1
x.squeeze(0)                  # removes dim 0 only if size 1

# Transpose and permute
x.T                           # transpose (only for 2D tensors)
x.transpose(0, 1)             # swap dim 0 and dim 1 → [3, 2, 4]
x.permute(2, 0, 1)            # reorder all dims → [4, 2, 3]

# Flatten
x.flatten()                   # [24] — flatten all dims
x.flatten(1)                  # [2, 12] — flatten from dim 1 onward

# Contiguous
x.is_contiguous()             # True if memory layout matches logical layout
x.contiguous()                # returns contiguous copy if needed
```

> **Interview Note:** `view()` vs `reshape()` — `view` requires contiguous memory and never copies. `reshape` works on any tensor and copies only when necessary. After `transpose`/`permute`, tensor is non-contiguous, so `view` will fail but `reshape` won't.

## Indexing and Slicing

```python
x = torch.randn(4, 5)

x[0]                          # first row → shape [5]
x[:, 0]                       # first column → shape [4]
x[1:3, 2:4]                   # rows 1-2, cols 2-3 → shape [2, 2]
x[x > 0]                      # boolean mask → 1D tensor of positives
x[[0, 2, 3]]                  # fancy indexing → rows 0, 2, 3

# Advanced indexing
idx = torch.tensor([0, 2])
x[idx]                        # select rows 0 and 2
x[:, idx]                     # select columns 0 and 2

# Scatter and gather (critical for embeddings, NMS)
torch.gather(x, dim=1, index=idx.unsqueeze(0).expand(4, -1))  # gather along cols
```

---

# 2. Tensor Operations

## Math

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

a + b                         # element-wise add (same as torch.add(a, b))
a * b                         # element-wise multiply (Hadamard product)
a / b                         # element-wise divide
a ** 2                        # element-wise square
a @ b.T                       # matrix multiply [3,4] @ [4,3] → [3,3]
torch.matmul(a, b.T)          # same as @ operator
torch.mm(a, b.T)              # strict 2D matrix multiply (no broadcasting)
torch.bmm(batch_a, batch_b)   # batched matmul: [B,N,M] @ [B,M,K] → [B,N,K]
torch.einsum('ij,jk->ik', a, b.T)  # Einstein notation — flexible, readable
```

## Reductions

```python
x = torch.randn(3, 4)

x.sum()                       # scalar sum of all elements
x.sum(dim=0)                  # sum across rows → shape [4]
x.sum(dim=1)                  # sum across cols → shape [3]
x.sum(dim=1, keepdim=True)    # keep dim → shape [3, 1] (useful for broadcasting)
x.mean(dim=-1)                # mean along last dim
x.max(dim=1)                  # returns (values, indices) tuple
x.argmax(dim=1)               # indices of max along dim 1 → shape [3]
x.min(), x.std(), x.var()     # other reductions
x.clamp(min=0)                # ReLU equivalent — clamp below 0
x.clamp(min=-1, max=1)        # clip to [-1, 1]
```

## Broadcasting Rules

```python
# Rule: align dims from the RIGHT. Dims must be equal or one of them is 1.
a = torch.randn(3, 1)        # [3, 1]
b = torch.randn(1, 4)        # [1, 4]
c = a + b                    # [3, 4] — a broadcast across cols, b across rows

# Common pattern: subtract mean per row
x = torch.randn(5, 10)
x_centered = x - x.mean(dim=1, keepdim=True)  # [5,10] - [5,1] → [5,10]

# Broadcasting for pairwise distances
p1 = torch.randn(100, 2).unsqueeze(1)   # [100, 1, 2]
p2 = torch.randn(50, 2).unsqueeze(0)    # [1, 50, 2]
dists = ((p1 - p2) ** 2).sum(-1).sqrt() # [100, 50] pairwise distances
```

> **Interview Note:** Broadcasting is how you vectorize pairwise operations (IoU, distance matrices). Always think: can I use `unsqueeze` + broadcasting instead of a Python loop?

---

# 3. Autograd — Automatic Differentiation

```python
x = torch.tensor(3.0, requires_grad=True)   # track gradients
y = x ** 2 + 2 * x + 1                      # forward pass builds computation graph
y.backward()                                 # compute dy/dx
print(x.grad)                                # tensor(8.0) — dy/dx = 2x + 2 = 8

# Multi-variable
w = torch.randn(3, requires_grad=True)
b = torch.randn(1, requires_grad=True)
x = torch.randn(3)
y = (w * x).sum() + b
y.backward()
print(w.grad.shape)                          # [3] — dy/dw for each weight

# CRITICAL: gradients accumulate by default
w.grad.zero_()                               # must zero before next backward()

# Detaching from graph
z = y.detach()                               # z has same value, no grad tracking
z = y.clone().detach()                       # safe independent copy

# No-grad context (inference, metrics)
with torch.no_grad():                        # disables grad tracking globally
    pred = model(x)                          # faster, less memory
    acc = (pred.argmax(1) == labels).float().mean()

# Gradient clipping (prevents exploding gradients in RNNs/transformers)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # L2 norm clip
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)  # value clip
```

> **Interview Note:** Why `zero_grad()`? PyTorch accumulates gradients into `.grad` on each `backward()` call. Without zeroing, gradients from previous iterations add up → wrong updates. This design enables gradient accumulation over multiple mini-batches (useful when GPU memory is limited).

---

# 4. nn.Module — Building Models

## Basic Module

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()                            # MUST call super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)          # registered as parameter
        self.bn = nn.BatchNorm1d(hidden)              # has running stats
        self.relu = nn.ReLU()                         # stateless, could also use F.relu
        self.drop = nn.Dropout(p=0.3)                 # drops 30% of neurons at train time
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):                             # defines computation graph
        x = self.fc1(x)                               # [B, hidden]
        x = self.bn(x)                                # normalize per feature
        x = self.relu(x)                              # non-linearity
        x = self.drop(x)                              # regularization (disabled at eval)
        x = self.fc2(x)                               # [B, out_dim]
        return x

model = MyModel(784, 256, 10)
print(sum(p.numel() for p in model.parameters()))     # total parameter count
```

## nn.Sequential — Quick Stacking

```python
model = nn.Sequential(
    nn.Linear(784, 256),          # layer 0
    nn.BatchNorm1d(256),          # layer 1
    nn.ReLU(),                    # layer 2
    nn.Dropout(0.3),              # layer 3
    nn.Linear(256, 10),           # layer 4
)
out = model(x)                    # chains forward through all layers
```

## Parameter Inspection

```python
for name, param in model.named_parameters():
    print(f"{name:30s} {param.shape} requires_grad={param.requires_grad}")

# Freeze layers (transfer learning)
for param in model.fc1.parameters():
    param.requires_grad = False                # freeze fc1 weights

# Only optimize trainable params
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
```

---

# 5. Common Layers Reference

```python
# Linear / Dense
nn.Linear(in_features, out_features, bias=True)      # y = xW^T + b

# Conv2d
nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)  # same padding
nn.Conv2d(64, 128, 3, stride=2, padding=1)            # downsample by 2x

# Transpose Conv (upsampling / decoder)
nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # upsample by 2x

# Normalization
nn.BatchNorm2d(num_features)       # per-channel, needs batch > 1 at train
nn.LayerNorm(normalized_shape)     # per-sample, works with any batch size
nn.GroupNorm(num_groups, num_ch)    # compromise: group channels
nn.InstanceNorm2d(num_features)    # per-sample per-channel (style transfer)

# Activation
nn.ReLU(inplace=True)              # max(0, x), inplace saves memory
nn.LeakyReLU(0.01)                 # allows small negative gradient
nn.GELU()                          # used in transformers
nn.SiLU()                          # x * sigmoid(x), aka Swish

# Pooling
nn.MaxPool2d(kernel_size=2, stride=2)    # downsample, keep max
nn.AvgPool2d(kernel_size=2, stride=2)    # downsample, keep average
nn.AdaptiveAvgPool2d((1, 1))             # global average pool → [B, C, 1, 1]

# Recurrent
nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
nn.GRU(input_size, hidden_size, batch_first=True)

# Attention
nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

# Embedding
nn.Embedding(vocab_size, embed_dim)       # lookup table: index → vector
nn.Embedding(1000, 256, padding_idx=0)    # zero vector for padding token
```

> **Interview Note:** `BatchNorm` vs `LayerNorm` — BatchNorm normalizes across the batch dimension (great for CNNs, but breaks with small batches or inference on single samples). LayerNorm normalizes across features within each sample (used in Transformers because sequence lengths vary and batch independence is needed).

---

# 6. Loss Functions

```python
import torch.nn.functional as F

# Classification
F.cross_entropy(logits, targets)             # logits [B,C], targets [B] (Long)
                                              # combines LogSoftmax + NLLLoss
nn.CrossEntropyLoss(weight=class_weights)    # weighted for class imbalance
nn.CrossEntropyLoss(ignore_index=-100)       # ignore padded positions
nn.CrossEntropyLoss(label_smoothing=0.1)     # smooth one-hot → prevents overconfidence

# Binary
F.binary_cross_entropy_with_logits(logits, targets.float())  # sigmoid + BCE
                                              # numerically stable, preferred over BCE

# Regression
F.mse_loss(pred, target)                     # L2 loss — penalizes large errors heavily
F.l1_loss(pred, target)                      # L1 loss — robust to outliers
F.smooth_l1_loss(pred, target, beta=1.0)     # Huber loss — L1 outside beta, L2 inside
                                              # used in object detection box regression

# Detection-specific
# Focal loss — down-weight easy examples for extreme class imbalance
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * bce).mean()

# Contrastive / metric learning
F.cosine_similarity(a, b, dim=-1)            # similarity in [-1, 1]
# InfoNCE, triplet loss — see specialized libraries
```

> **Interview Note:** Why `cross_entropy` takes raw logits, not softmax? Combining log + softmax in one op avoids numerical instability (log of very small softmax outputs → -inf). Never apply softmax before `cross_entropy`.

---

# 7. Optimizers & Schedulers

```python
# SGD — the baseline
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam — adaptive learning rates per parameter
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# AdamW — decoupled weight decay (correct L2 regularization for Adam)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Per-parameter groups (different LR for backbone vs head)
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},     # fine-tune slowly
    {'params': model.head.parameters(), 'lr': 1e-3},         # train head faster
], weight_decay=0.01)

# Learning rate schedulers
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=len(loader)*epochs
)

# In training loop:
for epoch in range(epochs):
    train(...)
    scheduler.step()                          # update LR after each epoch
    print(f"LR: {scheduler.get_last_lr()}")   # check current LR
```

> **Interview Note:** `Adam` vs `AdamW` — vanilla Adam applies weight decay inside the gradient update, coupling it with the adaptive learning rate. AdamW decouples weight decay, applying it directly to weights. This matters: AdamW gives proper L2 regularization and is standard for training transformers.

---

# 8. Data Loading

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data                      # e.g., list of file paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)                 # REQUIRED: total samples

    def __getitem__(self, idx):               # REQUIRED: return one sample
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,                             # shuffle every epoch (train only)
    num_workers=4,                            # parallel data loading processes
    pin_memory=True,                          # faster CPU→GPU transfer
    drop_last=True,                           # drop incomplete last batch (for BN)
    persistent_workers=True,                  # keep workers alive between epochs
)

# Custom collate for variable-length data (detection, NLP)
def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])  # [B, C, H, W] — same size assumed
    targets = [b[1] for b in batch]               # list of dicts (variable box count)
    return images, targets

loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
```

> **Interview Note:** Why `num_workers > 0`? Data loading (disk I/O, augmentation) runs in parallel with GPU compute. Without workers, GPU waits idle while CPU loads data. Rule of thumb: `num_workers = 4 * num_gpus`. Too many workers → excessive RAM usage.

---

# 9. Training Loop — The Complete Pattern

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel(in_dim, hidden, out_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # ── Training ──
    model.train()                             # enable dropout, BN uses batch stats
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)            # move to GPU
        labels = labels.to(device)

        optimizer.zero_grad()                 # clear old gradients
        outputs = model(images)               # forward pass
        loss = loss_fn(outputs, labels)       # compute loss
        loss.backward()                       # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional
        optimizer.step()                      # update weights

        train_loss += loss.item()             # .item() → Python float, frees graph

    # ── Validation ──
    model.eval()                              # disable dropout, BN uses running stats
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():                     # no grad tracking → faster, less memory
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += loss_fn(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    scheduler.step()                          # update learning rate

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Val Loss: {val_loss/len(val_loader):.4f} "
          f"Val Acc: {correct/total:.4f}")
```

### The Order Matters

```
optimizer.zero_grad()    # 1. clear gradients
output = model(input)    # 2. forward pass
loss = loss_fn(output, target)  # 3. compute loss
loss.backward()          # 4. compute gradients
optimizer.step()         # 5. update weights
```

> **Interview Note:** What happens if you swap `backward()` and `step()`? If you call `step()` before `backward()`, weights update using stale gradients from the previous iteration (or zeros on the first iteration). Always: zero → forward → loss → backward → step.

---

# 10. GPU & Device Management

```python
# Check availability
torch.cuda.is_available()                    # True if CUDA GPU available
torch.cuda.device_count()                    # number of GPUs
torch.cuda.current_device()                  # current GPU index
torch.cuda.get_device_name(0)                # e.g., "NVIDIA A100-SXM4-80GB"

# Move data
device = torch.device('cuda:0')              # specific GPU
tensor = tensor.to(device)                   # move tensor
model = model.to(device)                     # move all parameters + buffers

# Memory management
torch.cuda.memory_allocated()                # bytes currently allocated
torch.cuda.max_memory_allocated()            # peak allocation
torch.cuda.empty_cache()                     # free unused cached memory (not allocated)

# Multi-GPU — DataParallel (simple, but slower)
model = nn.DataParallel(model)               # wraps model, splits batch across GPUs

# Multi-GPU — DistributedDataParallel (production, faster)
# Requires torch.distributed.launch or torchrun
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

---

# 11. Mixed Precision Training (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()                        # scales loss to prevent fp16 underflow

for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()

    with autocast():                         # auto-casts ops to fp16 where safe
        outputs = model(images)              # forward in fp16 (faster, less VRAM)
        loss = loss_fn(outputs, labels)      # loss still fp32

    scaler.scale(loss).backward()            # scaled backward in fp16
    scaler.step(optimizer)                   # unscale gradients, then step
    scaler.update()                          # adjust scale factor
```

> **Interview Note:** Why mixed precision? FP16 uses half the memory and is 2-8x faster on modern GPUs (Tensor Cores). But FP16 has limited range (max ~65504) — GradScaler multiplies loss by a large factor before backward to prevent small gradients from becoming zero, then unscales before optimizer step.

---

# 12. Saving & Loading

```python
# Save model weights only (recommended)
torch.save(model.state_dict(), 'model.pth')

# Load weights
model = MyModel(...)
model.load_state_dict(torch.load('model.pth', map_location=device))  # map_location handles CPU↔GPU
model.eval()                                 # set to eval before inference

# Save full checkpoint (for resuming training)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Resume training
ckpt = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
scheduler.load_state_dict(ckpt['scheduler_state_dict'])
start_epoch = ckpt['epoch'] + 1

# ONNX export (deployment)
dummy = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(model, dummy, 'model.onnx', opset_version=11)
```

> **Interview Note:** Why `state_dict()` instead of saving the whole model? `torch.save(model)` uses pickle, which binds to the exact class definition and file path. If you rename or move the class, loading breaks. `state_dict()` is just a dict of tensor name → tensor, portable across code changes.

---

# 13. Transforms & Augmentation

```python
import torchvision.transforms as T

# Standard ImageNet preprocessing
transform_train = T.Compose([
    T.RandomResizedCrop(224),                # random crop and resize
    T.RandomHorizontalFlip(p=0.5),           # 50% chance horizontal flip
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),                            # PIL/numpy → [C,H,W] float [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]),   # ImageNet std
])

transform_val = T.Compose([
    T.Resize(256),                           # resize shorter side to 256
    T.CenterCrop(224),                       # center crop to 224
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# For detection: use Albumentations (applies same transform to image + boxes)
# import albumentations as A
# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.3),
# ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
```

---

# 14. Common CNN Architectures — Quick Reference

```python
import torchvision.models as models

# ResNet (residual connections — solved vanishing gradients in deep nets)
resnet = models.resnet50(weights='IMAGENET1K_V2')   # pretrained
features = nn.Sequential(*list(resnet.children())[:-1])  # remove classifier
# Output: [B, 2048, 1, 1] after global avg pool

# Use as backbone
class Detector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # keep spatial
        self.head = nn.Conv2d(2048, num_classes, 1)   # 1×1 conv classification head

    def forward(self, x):
        feat = self.features(x)               # [B, 2048, H/32, W/32]
        return self.head(feat)                 # [B, num_classes, H/32, W/32]
```

---

# INTERVIEW QUESTIONS & ANSWERS

---

## Q1: What is the difference between `model.train()` and `model.eval()`?

**Answer:**
- `model.train()` enables training-specific behaviors:
  - **Dropout** randomly zeros neurons (regularization)
  - **BatchNorm** computes mean/var from the current batch and updates running statistics
- `model.eval()` disables these:
  - **Dropout** is turned off (all neurons active)
  - **BatchNorm** uses stored running mean/var (accumulated during training) instead of batch stats
- Neither affects gradient computation — you still need `torch.no_grad()` to disable that
- **Common bug:** Forgetting `model.eval()` at inference → dropout is active → inconsistent, lower predictions

---

## Q2: Explain `backward()` and the computation graph.

**Answer:**
- PyTorch builds a **dynamic computation graph** (DAG) during the forward pass — each operation records its inputs and the function applied
- `loss.backward()` traverses this graph in reverse (backpropagation), computing `∂loss/∂param` for every parameter with `requires_grad=True`
- The graph is **destroyed after `backward()`** by default (set `retain_graph=True` to keep it, e.g., for multiple backward passes)
- Gradients **accumulate** in `.grad` — you must call `optimizer.zero_grad()` to clear them
- **Dynamic graph** means the graph can change every forward pass (unlike TensorFlow 1.x static graphs) — enables control flow (if/else, loops) inside forward

---

## Q3: What is gradient accumulation and when do you use it?

**Answer:**
```python
accumulation_steps = 4                        # effective batch = 4 × batch_size
optimizer.zero_grad()
for i, (images, labels) in enumerate(loader):
    outputs = model(images)
    loss = loss_fn(outputs, labels) / accumulation_steps  # scale loss
    loss.backward()                           # gradients accumulate in .grad
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()                      # update weights with accumulated grads
        optimizer.zero_grad()                 # reset for next accumulation
```
- **Use case:** GPU memory can't fit large batch sizes. Gradient accumulation simulates larger batches by summing gradients over multiple forward-backward passes before stepping
- **Important:** Divide loss by `accumulation_steps` so the total gradient magnitude matches a real large batch

---

## Q4: How does `DataParallel` vs `DistributedDataParallel` work?

**Answer:**
- **DataParallel (DP):**
  - Single process, GIL-limited
  - Replicates model to all GPUs each forward pass
  - Splits batch, runs forward on each GPU, gathers outputs to GPU:0 for loss
  - GPU:0 does backward, then scatters gradients → GPU:0 bottleneck
- **DistributedDataParallel (DDP):**
  - One process per GPU (avoids GIL)
  - Each process has its own model replica and data partition (`DistributedSampler`)
  - After backward, gradients are **all-reduced** (averaged) across processes via NCCL
  - No single-GPU bottleneck → near-linear scaling
- **Always use DDP in production.** DP is only for quick prototyping.

---

## Q5: What is `torch.no_grad()` vs `torch.inference_mode()`?

**Answer:**
- `torch.no_grad()` — disables gradient tracking. Tensors created inside won't have `grad_fn`. Saves memory and compute. Tensors can still be used with autograd outside the context.
- `torch.inference_mode()` (PyTorch 1.9+) — stricter version. Also disables autograd, but additionally marks tensors as inference-only. Slightly faster because PyTorch can skip more bookkeeping. Tensors created inside **cannot** be used in autograd afterward.
- **Use `inference_mode()`** for pure inference (deployment). Use `no_grad()` when you might need the tensors for autograd later (e.g., computing validation metrics that feed into a custom backward).

---

## Q6: Explain weight initialization. Why does it matter?

**Answer:**
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)                    # recursively applies to all submodules
```
- **Why it matters:** Bad initialization → vanishing/exploding gradients → network doesn't train
- **Xavier/Glorot** — for sigmoid/tanh activations. Maintains variance across layers: `Var(output) ≈ Var(input)`
- **Kaiming/He** — for ReLU activations. Accounts for ReLU zeroing half the outputs: scales by √(2/fan_in)
- PyTorch defaults are reasonable (Kaiming uniform for Linear/Conv), but custom init helps for specific architectures

---

## Q7: How do you handle class imbalance in PyTorch?

**Answer:**
```python
# Method 1: Weighted loss
class_counts = torch.tensor([5000, 500, 100])       # samples per class
weights = 1.0 / class_counts.float()                  # inverse frequency
weights = weights / weights.sum()                      # normalize
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

# Method 2: Weighted sampler (oversample minority class)
from torch.utils.data import WeightedRandomSampler
sample_weights = [weights[label] for label in dataset.labels]  # per-sample weight
sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)   # no shuffle with sampler

# Method 3: Focal loss (detection — see Section 6)
```

---

## Q8: What is the difference between `nn.Module` layers and `torch.nn.functional`?

**Answer:**
- `nn.Module` layers (e.g., `nn.Linear`, `nn.Dropout`) are **stateful** — they hold parameters and/or state (running mean in BN). Must be defined in `__init__` to be registered.
- `torch.nn.functional` (e.g., `F.relu`, `F.dropout`) are **stateless functions** — just math operations. No stored parameters.
- **Rule:** Use `nn.Module` for layers with learnable parameters or state. Use `F.*` for pure operations like activation functions.
- **Common mistake:** Using `F.dropout(x, p=0.3)` without passing `training=self.training` → dropout stays active at eval time.

```python
# WRONG — dropout always active
def forward(self, x):
    return F.dropout(F.relu(self.fc(x)), p=0.3)

# CORRECT — respects train/eval mode
def forward(self, x):
    return F.dropout(F.relu(self.fc(x)), p=0.3, training=self.training)
```

---

## Q9: How does `torch.jit` (TorchScript) work and when do you use it?

**Answer:**
```python
# Tracing — records operations on example input
traced = torch.jit.trace(model, example_input)    # captures fixed control flow
traced.save('model_traced.pt')                     # save for C++ deployment

# Scripting — compiles Python code to TorchScript IR
scripted = torch.jit.script(model)                 # handles if/else, loops
scripted.save('model_scripted.pt')

# Load in C++ or Python without original class definition
loaded = torch.jit.load('model_traced.pt')
```
- **Tracing:** Records operations with a concrete input. Fast, but ignores data-dependent control flow (if/else based on tensor values).
- **Scripting:** Parses Python code into TorchScript IR. Supports control flow but has syntax restrictions.
- **Use cases:** Production deployment (C++ inference), mobile (PyTorch Mobile), removing Python dependency.

---

## Q10: Explain the hooks system in PyTorch.

**Answer:**
```python
# Forward hook — inspect intermediate activations
activations = {}
def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()   # save output of this layer
    return hook

model.layer3.register_forward_hook(save_activation('layer3'))
out = model(x)                                # hook fires during forward
print(activations['layer3'].shape)            # [B, C, H, W]

# Backward hook — inspect or modify gradients
def grad_hook(module, grad_input, grad_output):
    print(f"Gradient norm: {grad_output[0].norm():.4f}")

model.fc.register_full_backward_hook(grad_hook)

# Tensor hook — modify gradient during backward
x = torch.randn(3, requires_grad=True)
x.register_hook(lambda grad: grad * 2)       # double the gradient for x
```
- **Use cases:** Feature extraction (Grad-CAM), debugging gradient flow, gradient modification (gradient reversal layer for domain adaptation)

---

## Q11: What is `torch.compile()` and how does it speed things up?

**Answer:**
```python
# PyTorch 2.0+
model = torch.compile(model, mode='reduce-overhead')  # compiles forward graph
```
- `torch.compile` traces the model and optimizes the computation graph using **TorchDynamo** (captures Python bytecode) + **TorchInductor** (generates optimized GPU kernels)
- Fuses operations (e.g., conv + BN + ReLU into one kernel), eliminates unnecessary memory reads/writes
- Modes: `'default'` (balanced), `'reduce-overhead'` (minimize launch overhead), `'max-autotune'` (try many kernel variants, slow compile, fast run)
- **1.5-2x speedup** on typical training/inference without code changes

---

## Q12: How do you debug shape mismatches?

**Answer:**
```python
# Strategy 1: Print shapes at each step
def forward(self, x):
    print(f"Input: {x.shape}")                # [B, 3, 224, 224]
    x = self.conv1(x)
    print(f"After conv1: {x.shape}")          # [B, 64, 112, 112]
    x = self.pool(x)
    print(f"After pool: {x.shape}")           # [B, 64, 56, 56]
    x = x.flatten(1)
    print(f"After flatten: {x.shape}")        # [B, 64*56*56] = [B, 200704]
    x = self.fc(x)                            # if fc expects different in_features → CRASH
    return x

# Strategy 2: Use torchinfo (pip install torchinfo)
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))   # shows layer-by-layer shapes

# Common causes:
# 1. Forgetting to flatten before Linear
# 2. Wrong in_features for Linear after conv/pool
# 3. Missing batch dimension (model expects [B,...], you pass [...])
# 4. Transpose/permute putting dims in wrong order
```

---

## Q13: What are `register_buffer` and when do you use it?

**Answer:**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        # Buffer: saved in state_dict, moved with .to(device), but NOT a parameter
        self.register_buffer('running_mean', torch.zeros(5))
        # vs regular attribute: NOT saved, NOT moved
        self.some_constant = torch.tensor(3.14)  # stays on CPU even after .cuda()
```
- **Parameters** (`nn.Parameter`): learned via optimizer, `requires_grad=True`, in `state_dict`
- **Buffers** (`register_buffer`): not learned, `requires_grad=False`, but saved in `state_dict` and moved with `.to(device)`. Example: BatchNorm's running mean/var, positional encodings in transformers
- **Regular attributes**: not tracked at all — won't be saved or moved to GPU

---

## Q14: Explain `torch.utils.checkpoint` (gradient checkpointing).

**Answer:**
```python
from torch.utils.checkpoint import checkpoint

class BigModel(nn.Module):
    def forward(self, x):
        # Instead of storing all intermediate activations:
        x = checkpoint(self.layer1, x, use_reentrant=False)  # recompute during backward
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = self.layer3(x)                    # normal forward (activations stored)
        return x
```
- **Problem:** Deep models store all intermediate activations for backward → huge memory
- **Solution:** Don't store activations for checkpointed layers. During backward, re-run the forward pass for those layers to recompute activations on the fly
- **Tradeoff:** ~30% slower training, but ~50-70% less memory → enables training much larger models or bigger batches

---

## Q15: How do you profile PyTorch performance?

**Answer:**
```python
# PyTorch profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    with record_function("model_inference"):
        model(x)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Quick CUDA timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = model(x)
end.record()
torch.cuda.synchronize()                     # wait for GPU to finish
print(f"{start.elapsed_time(end):.2f} ms")   # accurate GPU time

# DON'T use time.time() for GPU ops — GPU is async, time.time() only measures launch time
```

---

## Q16: What is `pin_memory` in DataLoader?

**Answer:**
- `pin_memory=True` allocates data in **page-locked (pinned) host memory**
- Pinned memory enables **async CPU→GPU transfer** via `tensor.to(device, non_blocking=True)`
- Without pinning: CPU→GPU transfer is synchronous and slower (must first copy to pinned staging buffer)
- **Use when:** Training on GPU with CPU-based data loading (almost always)
- **Don't use when:** Running on CPU only, or system has limited RAM (pinned memory can't be swapped to disk)

---

## Q17: How do you handle variable-length sequences in PyTorch?

**Answer:**
```python
# Method 1: Padding + mask
from torch.nn.utils.rnn import pad_sequence
seqs = [torch.tensor([1,2,3]), torch.tensor([4,5]), torch.tensor([6])]
padded = pad_sequence(seqs, batch_first=True, padding_value=0)  # [3, 3]
# tensor([[1, 2, 3],
#          [4, 5, 0],
#          [6, 0, 0]])
lengths = torch.tensor([3, 2, 1])
mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)  # [3, 3] bool

# Method 2: PackedSequence for RNNs (avoids computing on padding)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
output_packed, (h_n, c_n) = lstm(packed)
output_padded, out_lengths = pad_packed_sequence(output_packed, batch_first=True)
```

---

## Q18: Explain `torch.einsum` and give practical examples.

**Answer:**
```python
# Einstein summation — flexible, readable tensor operations
a = torch.randn(3, 4)
b = torch.randn(4, 5)

torch.einsum('ij,jk->ik', a, b)             # matrix multiply: [3,5]
torch.einsum('ij->ji', a)                    # transpose: [4,3]
torch.einsum('ij->i', a)                     # row sum: [3]
torch.einsum('ij->', a)                      # total sum: scalar
torch.einsum('ii->', a[:3,:3])               # trace: sum of diagonal

# Batched matmul
A = torch.randn(8, 3, 4)                     # batch of matrices
B = torch.randn(8, 4, 5)
torch.einsum('bnm,bmk->bnk', A, B)          # [8, 3, 5]

# Attention: Q @ K^T / sqrt(d)
Q = torch.randn(2, 8, 32, 64)               # [B, heads, seq, dim]
K = torch.randn(2, 8, 32, 64)
attn = torch.einsum('bhsd,bhtd->bhst', Q, K) / 64**0.5  # [B, heads, seq, seq]

# Outer product
a = torch.randn(3)
b = torch.randn(4)
torch.einsum('i,j->ij', a, b)               # [3, 4]
```

---

## Q19: What are common causes of CUDA out of memory (OOM)?

**Answer:**
1. **Batch size too large** → reduce batch size or use gradient accumulation
2. **Storing tensors that aren't detached** → `loss_history.append(loss.item())` not `loss_history.append(loss)` (the latter keeps the entire computation graph alive)
3. **Not using `torch.no_grad()` at validation** → grad tracking allocates memory for intermediate activations
4. **Large model** → use mixed precision (AMP), gradient checkpointing, or model parallelism
5. **Memory leak in custom Dataset** → opening files/images without closing, growing lists

```python
# Quick fixes
torch.cuda.empty_cache()                     # free cached (not allocated) memory
# Use nvidia-smi or torch.cuda.memory_summary() to diagnose
print(torch.cuda.memory_summary(abbreviated=True))
```

---

## Q20: How do you implement early stopping?

**Answer:**
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience              # epochs to wait before stopping
        self.min_delta = min_delta            # minimum improvement to count
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict().copy()  # save best weights
            return False                      # don't stop
        self.counter += 1
        return self.counter >= self.patience  # True → stop training

# Usage
stopper = EarlyStopping(patience=5)
for epoch in range(100):
    train(...)
    val_loss = validate(...)
    if stopper.step(val_loss, model):
        model.load_state_dict(stopper.best_state)  # restore best weights
        print(f"Early stop at epoch {epoch}")
        break
```

---

# Quick Reference Card

| Operation | Code |
|---|---|
| Create tensor | `torch.randn(B, C, H, W)` |
| Move to GPU | `x.to('cuda')` |
| Matrix multiply | `a @ b` or `torch.matmul(a, b)` |
| Reshape | `x.reshape(B, -1)` or `x.view(B, -1)` |
| Add batch dim | `x.unsqueeze(0)` |
| Remove batch dim | `x.squeeze(0)` |
| Concatenate | `torch.cat([a, b], dim=0)` |
| Stack | `torch.stack([a, b], dim=0)` |
| Boolean mask | `x[x > 0]` |
| Argmax | `x.argmax(dim=1)` |
| Gradient clip | `clip_grad_norm_(model.parameters(), 1.0)` |
| Freeze layer | `param.requires_grad = False` |
| Count params | `sum(p.numel() for p in model.parameters())` |
| Save model | `torch.save(model.state_dict(), path)` |
| Load model | `model.load_state_dict(torch.load(path))` |

---

*Revision tip: Read this twice — once for concepts, once while typing out the code from memory. If you can write the training loop, vectorized IoU, and custom Dataset without looking, you're ready.*
