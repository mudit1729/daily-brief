# ML Coding Cheatsheet — Scale Robotics Interview

> Ready-to-write code snippets: PyTorch essentials, backprop from scratch, transformers, training loops, and common interview implementations.

---

## 0. INTERVIEW CODING CHECKLIST

```text
Before writing code:
1. State input/output shapes and dtype
2. Clarify train vs inference behavior
3. Start with correct brute-force logic, then vectorize
4. Test on tiny tensors you can inspect by hand
5. Mention time/space complexity if algorithmic
6. Watch device, dtype, batch dimension, and masking bugs
```

### Easy Points To Score Out Loud
- Say the tensor shapes as you code.
- Add one tiny sanity test before optimizing.
- For training code: remember `model.train()`, `model.eval()`, `zero_grad()`, and `no_grad()`/`inference_mode()`.
- For attention: say whether the mask is padding, causal, or both.

---

## 1. PYTORCH ESSENTIALS

### Tensor Basics
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Creation
x = torch.zeros(3, 4)                  # shape (3, 4)
x = torch.randn(3, 4)                  # normal distribution
x = torch.tensor([1.0, 2.0, 3.0])      # from list
x = torch.arange(0, 10, 2)             # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, 5)            # [0, 0.25, 0.5, 0.75, 1]
x = torch.eye(3)                       # identity matrix

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

# Reshaping
x.view(2, 6)          # reshape (must be contiguous)
x.reshape(2, 6)       # reshape (always works)
x.permute(1, 0, 2)    # transpose dimensions
x.unsqueeze(0)         # add dim at position 0: (3,4) → (1,3,4)
x.squeeze(0)           # remove dim of size 1
x.flatten(1)           # flatten from dim 1: (B,C,H,W) → (B, C*H*W)

# Indexing
x[:, 1:3]             # slice
x[x > 0]              # boolean indexing
torch.where(cond, a, b)  # element-wise conditional

# Common ops
torch.cat([a, b], dim=0)     # concatenate
torch.stack([a, b], dim=0)   # stack (adds new dim)
torch.einsum('ij,jk->ik', A, B)  # matrix multiply
x @ y                        # matrix multiply (shorthand)
```

### Autograd
```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()           # compute gradients
print(x.grad)          # dy/dx = 2x

# Disable gradient tracking
with torch.no_grad():
    y = model(x)       # inference mode

with torch.inference_mode():
    y = model(x)       # slightly faster, stricter inference context

# Detach from computation graph
z = x.detach()         # shares data, no grad
z = x.clone().detach() # safe copy
```

---

## 2. nn.Module — BUILDING MODELS

### Basic Module Pattern
```python
class MyModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MyModel(784, 256, 10).to(device)
print(sum(p.numel() for p in model.parameters()))  # count params
```

### CNN
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # (B,3,H,W) → (B,32,H,W)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # (B,32,H/2,W/2)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # (B,64,H/4,W/4)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # (B,64,1,1)
            nn.Flatten(),                        # (B,64)
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### Residual Block
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual       # skip connection
        return F.relu(out)
```

### Bottleneck Block (ResNet-50+)
```python
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)
```

---

## 3. TRANSFORMER FROM SCRATCH

### Scaled Dot-Product Attention
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_k)
    mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (B, H, Sq, Sk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)           # (B, H, Sq, Sk)
    output = attn_weights @ V                           # (B, H, Sq, d_v)
    return output, attn_weights
```

### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        # Linear projections and reshape to (B, H, seq, d_k)
        Q = self.W_q(Q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        out, attn = scaled_dot_product_attention(Q, K, V, mask)

        # Concat heads and project
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out)
```

### Positional Encoding
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### Causal Mask
```python
def make_causal_mask(seq_len, device):
    # (1, 1, T, T), True where attention is allowed
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    ).unsqueeze(0).unsqueeze(0)
```

### Transformer Encoder Block
```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm variant (more stable)
        x_norm = self.norm1(x)
        x = x + self.dropout(self.attn(x_norm, x_norm, x_norm, mask))
        x = x + self.ffn(self.norm2(x))
        return x
```

### Full Encoder
```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, num_classes, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x = self.pos_enc(self.embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.head(x[:, 0])  # CLS token
```

---

## 4. BACKPROPAGATION FROM SCRATCH

### Two-Layer Network (NumPy)
```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    # y_true: one-hot (N, C)
    N = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / N

# Data: X (N, D), Y (N, C) one-hot
N, D, H, C = 64, 784, 128, 10
lr = 0.01

# Init
W1 = np.random.randn(D, H) * np.sqrt(2.0 / D)   # He init
b1 = np.zeros((1, H))
W2 = np.random.randn(H, C) * np.sqrt(2.0 / H)
b2 = np.zeros((1, C))

# Forward
z1 = X @ W1 + b1          # (N, H)
a1 = relu(z1)              # (N, H)
z2 = a1 @ W2 + b2          # (N, C)
y_pred = softmax(z2)        # (N, C)
loss = cross_entropy(y_pred, Y)

# Backward
dz2 = (y_pred - Y) / N     # (N, C) — softmax + CE gradient
dW2 = a1.T @ dz2           # (H, C)
db2 = dz2.sum(axis=0, keepdims=True)  # (1, C)

da1 = dz2 @ W2.T           # (N, H)
dz1 = da1 * relu_grad(z1)  # (N, H)
dW1 = X.T @ dz1            # (D, H)
db1 = dz1.sum(axis=0, keepdims=True)  # (1, H)

# Update
W2 -= lr * dW2
b2 -= lr * db2
W1 -= lr * dW1
b1 -= lr * db1
```

---

## 5. TRAINING LOOP (PRODUCTION-QUALITY)

```python
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

def train(model, train_loader, val_loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled
            ):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Mixed precision backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = 100. * correct / total

        # Validation
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch}: Loss={total_loss/len(train_loader):.4f} '
              f'TrainAcc={train_acc:.2f}% ValAcc={val_acc:.2f}%')

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100. * correct / total
```

---

## 6. CUSTOM DATASET & DATALOADER

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        tfms = [transforms.Resize(256)]
        if split == 'train':
            tfms += [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        else:
            tfms += [transforms.CenterCrop(224)]
        tfms += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(tfms)
        self.samples = []  # list of (path, label)
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

# Usage
train_loader = DataLoader(
    CustomImageDataset('data/train', 'train'),
    batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
```

---

## 7. LOSS FUNCTION IMPLEMENTATIONS

```python
# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)   # p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Dice Loss (for segmentation)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + self.smooth) / \
               (pred_flat.sum() + target_flat.sum() + self.smooth)

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()

# InfoNCE / NT-Xent (Contrastive)
def info_nce_loss(features, temperature=0.07):
    """features: (2N, d) — first N are view1, second N are view2"""
    N = features.shape[0] // 2
    features = F.normalize(features, dim=1)
    sim = features @ features.T / temperature         # (2N, 2N)

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(features.device)

    # Mask out self-similarity
    mask = torch.eye(2*N, dtype=torch.bool, device=features.device)
    sim.masked_fill_(mask, float('-inf'))

    return F.cross_entropy(sim, labels)
```

---

## 8. IoU & NMS IMPLEMENTATION

```python
def compute_iou(box1, box2):
    """
    box1, box2: (N, 4) and (M, 4) in format [x1, y1, x2, y2]
    Returns: (N, M) IoU matrix
    """
    x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / (union + 1e-6)


def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: (N, 4), scores: (N,)
    Returns: indices of kept boxes
    """
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        ious = compute_iou(
            boxes[i].unsqueeze(0),
            boxes[order[1:]]
        ).squeeze(0)

        mask = ious <= iou_threshold
        order = order[1:][mask]

    return boxes.new_tensor(keep, dtype=torch.long)
```

---

## 9. CONV OUTPUT SIZE FORMULA

```python
# Output size formula
# H_out = floor((H_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)

# Quick reference (common cases):
# Conv2d(in, out, 3, padding=1, stride=1) → same size
# Conv2d(in, out, 3, padding=1, stride=2) → half size
# Conv2d(in, out, 1, stride=1)            → same size (1x1 conv)
# MaxPool2d(2, stride=2)                   → half size

# Transposed convolution (upsampling):
# H_out = (H_in - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
# ConvTranspose2d(in, out, 4, stride=2, padding=1) → double size
```

---

## 10. COMMON INTERVIEW IMPLEMENTATIONS

### K-Means
```python
def kmeans(X, K, max_iters=100):
    N, D = X.shape
    # Random init
    indices = torch.randperm(N)[:K]
    centroids = X[indices].clone()

    for _ in range(max_iters):
        # Assign clusters
        dists = torch.cdist(X, centroids)  # (N, K)
        labels = dists.argmin(dim=1)        # (N,)

        # Update centroids
        new_centroids = torch.stack([
            X[labels == k].mean(dim=0) if (labels == k).any()
            else centroids[k]
            for k in range(K)
        ])

        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, centroids
```

### Linear Regression (Closed Form)
```python
# Normal equation: w = (XᵀX)⁻¹Xᵀy
def linear_regression(X, y):
    # Add bias term
    ones = torch.ones(X.shape[0], 1)
    X_b = torch.cat([ones, X], dim=1)
    w = torch.linalg.lstsq(X_b, y).solution
    return w
```

### Batch Normalization
```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            # Update running stats
            self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### Self-Attention (Compact)
```python
# Minimal self-attention in ~10 lines
def self_attention(x, W_q, W_k, W_v):
    """x: (B, T, D)"""
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    d_k = Q.size(-1)
    attn = F.softmax(Q @ K.transpose(-2, -1) / d_k**0.5, dim=-1)
    return attn @ V
```

---

## 11. MODEL EXPORT & OPTIMIZATION

### ONNX Export
```python
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17
)
```

### Quantization (Post-Training)
```python
# Dynamic quantization (weights only, good for RNNs/Transformers)
quantized_model = torch.ao.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Static quantization (activations too, needs calibration)
model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
model_prepared = torch.ao.quantization.prepare(model)
# Run calibration data through model_prepared
model_quantized = torch.ao.quantization.convert(model_prepared)
```

### TorchScript
```python
# Tracing
traced = torch.jit.trace(model, dummy_input)
traced.save("model_traced.pt")

# Scripting (handles control flow)
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")
```

---

## 12. POINT CLOUD OPERATIONS

### Farthest Point Sampling (FPS)
```python
def farthest_point_sampling(points, n_samples):
    """points: (N, 3), returns indices of n_samples points"""
    N = points.shape[0]
    selected = [torch.randint(N, (1,)).item()]
    distances = torch.full((N,), float('inf'), device=points.device)

    for _ in range(n_samples - 1):
        last = points[selected[-1]].unsqueeze(0)
        dist = torch.cdist(points, last).squeeze(-1)
        distances = torch.min(distances, dist)
        selected.append(distances.argmax().item())

    return torch.tensor(selected, device=points.device)
```

### Ball Query
```python
def ball_query(points, centroids, radius, max_neighbors):
    """
    points: (N, 3), centroids: (M, 3)
    Returns: (M, max_neighbors) indices
    """
    dists = torch.cdist(centroids, points)  # (M, N)
    mask = dists < radius
    result = torch.full(
        (centroids.shape[0], max_neighbors),
        -1,
        dtype=torch.long,
        device=points.device
    )

    for i in range(centroids.shape[0]):
        indices = torch.where(mask[i])[0]
        k = min(len(indices), max_neighbors)
        result[i, :k] = indices[:k]

    return result
```

---

## 13. USEFUL PYTORCH PATTERNS

### Gradient Accumulation
```python
accumulation_steps = 4
for i, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        loss = criterion(model(inputs), targets) / accumulation_steps
    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

### EMA (Exponential Moving Average)
```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: p.clone().detach()
                       for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p, alpha=1 - self.decay)

    def apply(self):
        """Swap model params with EMA params for evaluation"""
        self.backup = {name: p.clone() for name, p in self.model.named_parameters()}
        for name, p in self.model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self):
        for name, p in self.model.named_parameters():
            p.data.copy_(self.backup[name])
```

### Freeze/Unfreeze Layers
```python
# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Only train specific layers
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# Unfreeze after N epochs (fine-tuning)
for param in model.backbone.parameters():
    param.requires_grad = True
```

### Learning Rate Finder (quick version)
```python
def lr_finder(model, loader, criterion, start_lr=1e-7, end_lr=10, num_steps=100):
    lrs, losses = [], []
    lr = start_lr
    factor = (end_lr / start_lr) ** (1 / num_steps)
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        if i >= num_steps:
            break
        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())
        lr *= factor

    return lrs, losses  # Plot to find steepest descent point
```

---

## 14. QUICK REFERENCE: COMMON SHAPES

```markdown
Image classification:
  Input:  (B, 3, H, W)     e.g., (32, 3, 224, 224)
  Output: (B, num_classes)  e.g., (32, 1000)

Object detection (DETR-style):
  Input:  (B, 3, H, W)
  Output: (B, num_queries, 4+num_classes)  e.g., (B, 100, 4+80)

Segmentation:
  Input:  (B, 3, H, W)
  Output: (B, num_classes, H, W)

Sequence model:
  Input:  (B, T)            token IDs
  Embed:  (B, T, D)         after embedding
  Output: (B, T, vocab)     for language modeling

Point cloud:
  Input:  (B, N, 3)         or (B, N, 3+F) with features
  Output: (B, num_classes)  classification
          (B, N, K)         per-point segmentation
```
