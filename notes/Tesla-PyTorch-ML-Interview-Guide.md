# PyTorch ML Interview Guide

Generated: April 27, 2026

Purpose: a practical PyTorch guide for ML Engineer interviews. It starts with a cheatsheet, then gives high-frequency PyTorch questions with answers, code, debugging cues, and follow-ups.

Source basis:

- Official PyTorch docs for technical behavior: tensors, autograd, `nn.Module`, `DataLoader`, `CrossEntropyLoss`, `Conv2d`, scaled dot-product attention, and AMP.
- Your local DeepML/ML coding notes for problem style: training loops, CNNs, attention, ViT blocks, losses, optimizers, and debugging.
- Public interview-prep search as weak signal for question themes: autograd, training loop, `nn.Module`, `train()`/`eval()`, data pipeline bugs, Conv2D shapes, and attention.

Note: PyTorch is not installed in this local environment, so snippets were written against official PyTorch APIs and reviewed statically. NumPy snippets in the companion guide were executable locally.

## PyTorch Cheatsheet

### Interview Mindset

Say tensor shapes, dtype, and device as you code.

```text
x:      [B, C, H, W], float32, device
logits: [B, num_classes], raw logits
target: [B], int64 class indices
loss:   scalar
```

Default training skeleton:

```python
model.train()  # enable training mode
for x, y in loader:  # iterate over items
    x, y = x.to(device), y.to(device)  # move tensor to device
    optimizer.zero_grad(set_to_none=True)  # clear old gradients
    logits = model(x)  # run forward pass
    loss = criterion(logits, y)  # compute loss
    loss.backward()  # backpropagate loss
    optimizer.step()  # update parameters
```

Default evaluation skeleton:

```python
model.eval()  # enable eval mode
with torch.inference_mode():  # disable autograd for inference
    for x, y in val_loader:  # iterate over items
        logits = model(x.to(device))  # move tensor to device
```

### Imports

```python
import math  # import dependency
import torch  # import dependency
import torch.nn as nn  # import dependency
import torch.nn.functional as F  # import dependency
from torch.utils.data import Dataset, DataLoader  # import specific APIs
```

### Tensor Creation

```python
x = torch.tensor([1.0, 2.0, 3.0])  # create tensor
x = torch.zeros(3, 4)  # allocate zeros
x = torch.ones(3, 4)  # allocate ones
x = torch.empty(3, 4)       # uninitialized
x = torch.randn(3, 4)  # sample normal tensor
x = torch.arange(0, 10, 2)  # create range tensor
x = torch.linspace(0, 1, 5)  # create evenly spaced tensor
x = torch.eye(3)  # create identity matrix
```

Device and dtype:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select compute device
x = x.to(device)  # move tensor to device
x = x.float()  # cast to float32
x = x.long()  # cast to int64 labels
x = x.to(dtype=torch.float32, device=device)  # move or cast tensor
```

Common dtypes:

```text
model inputs: float32 or float16/bfloat16 under AMP
class labels for CrossEntropyLoss: int64/long
binary labels for BCEWithLogitsLoss: float32
masks: bool preferred
```

### Shape Operations

```python
x.shape  # inspect shape
x.ndim  # inspect rank
x.numel()  # count elements

x.reshape(2, 3)  # reshape tensor
x.view(2, 3)             # requires compatible contiguous layout
x.contiguous().view(2, 3)  # view contiguous tensor
x.flatten(start_dim=1)  # flatten dimensions
x.unsqueeze(0)  # insert size-one axis
x.squeeze(0)  # remove size-one axis
x.permute(0, 2, 3, 1)    # NCHW -> NHWC
x.transpose(1, 2)  # swap axes
x.movedim(-1, 1)  # move axis
```

Rule:

```text
After permute/transpose, use reshape or contiguous().view(...).
```

### Indexing, Gather, Scatter

```python
x[:, 1:3]  # slice tensor values
x[x > 0]  # slice tensor values
torch.where(mask, a, b)  # select by condition

idx = logits.argmax(dim=1)  # pick max index
target_scores = logits.gather(dim=1, index=y[:, None]).squeeze(1)  # gather target logits
```

Top-k:

```python
values, indices = torch.topk(scores, k, dim=-1)  # select top-k values
```

### Broadcasting

PyTorch broadcasting follows NumPy-style right-aligned rules.

```python
X = torch.randn(5, 1, 3)  # sample normal tensor
Y = torch.randn(1, 7, 3)  # sample normal tensor
Z = X - Y                 # [5, 7, 3]
```

Use `keepdim=True`:

```python
x = x - x.mean(dim=0, keepdim=True)  # compute mean
x = x / (x.norm(dim=1, keepdim=True) + 1e-12)  # compute norm
```

### Autograd Basics

```python
x = torch.randn(3, requires_grad=True)  # sample normal tensor
y = (x ** 2).sum()  # build scalar loss
y.backward()  # compute gradients
x.grad  # inspect gradients
```

Key terms:

```text
requires_grad=True: track ops for gradient computation.
grad_fn: operation that produced a non-leaf tensor.
leaf tensor: tensor created by user, often parameters.
backward(): computes gradients into .grad fields of leaves.
```

Clear gradients:

```python
optimizer.zero_grad(set_to_none=True)  # clear old gradients
```

Detach:

```python
z = x.detach()  # detach from graph
z = x.detach().clone()  # detach and copy
```

No gradient contexts:

```python
with torch.no_grad():  # disable gradient tracking
    y = model(x)  # run forward pass

with torch.inference_mode():  # disable autograd for inference
    y = model(x)  # run forward pass
```

Difference:

- `no_grad` disables graph recording.
- `inference_mode` is stricter and can be faster for pure inference.
- Neither replaces `model.eval()`.

### `train()` vs `eval()`

```python
model.train()  # enable training mode
model.eval()  # enable eval mode
```

Affected modules:

- Dropout: random dropping in train, disabled in eval.
- BatchNorm: batch statistics in train, running stats in eval.

Important:

```text
model.eval() does not disable gradients.
torch.no_grad()/inference_mode() does not switch dropout/batchnorm behavior.
Use both during validation/inference.
```

### `nn.Module` Pattern

```python
class MLP(nn.Module):  # define MLP
    def __init__(self, in_dim, hidden_dim, num_classes):  # define __init__
        super().__init__()  # initialize base module
        self.net = nn.Sequential(  # define layer stack
            nn.Linear(in_dim, hidden_dim),  # create linear layer
            nn.ReLU(),  # create ReLU activation
            nn.Linear(hidden_dim, num_classes),  # create linear layer
        )  # close layer stack

    def forward(self, x):  # define forward
        return self.net(x)  # return result
```

Parameters:

```python
sum(p.numel() for p in model.parameters())  # count all parameters
sum(p.numel() for p in model.parameters() if p.requires_grad)  # count trainable parameters
```

Buffers:

```python
self.register_buffer("mean", torch.zeros(3))  # allocate zeros
```

Use buffers for non-trainable tensors that should move with `.to(device)` and save in `state_dict`.

### Loss Functions

Multi-class classification:

```python
criterion = nn.CrossEntropyLoss()  # create multiclass loss
loss = criterion(logits, target)  # compute loss
```

Shapes:

```text
logits: [B, C] raw logits
target: [B] int64 class indices
```

Do not apply softmax before `CrossEntropyLoss`.

Binary/multi-label:

```python
criterion = nn.BCEWithLogitsLoss()  # create binary loss
loss = criterion(logits, targets.float())  # cast to float32
```

Shapes:

```text
logits: [B] or [B, C]
targets: same shape, float 0/1
```

Regression:

```python
nn.MSELoss()  # create regression loss
nn.L1Loss()  # create L1 loss
nn.SmoothL1Loss()  # create Huber-style loss
```

### Optimizers and Schedulers

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)  # create AdamW optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # create LR scheduler
```

Training step order:

```text
zero_grad -> forward -> loss -> backward -> optional clip -> optimizer.step -> scheduler.step
```

Gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradient norm
```

### Dataset and DataLoader

```python
class MyDataset(Dataset):  # define MyDataset
    def __init__(self, X, y):  # define __init__
        self.X = X  # store features
        self.y = y  # store labels

    def __len__(self):  # define __len__
        return len(self.y)  # return result

    def __getitem__(self, idx):  # define __getitem__
        return self.X[idx], self.y[idx]  # return result


loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)  # build batch loader
```

Custom collate:

```python
def collate_fn(batch):  # define collate_fn
    xs, ys = zip(*batch)  # split samples and labels
    return torch.stack(xs), torch.tensor(ys)  # return result
```

### CNN Essentials

Conv2D output shape:

```text
H_out = floor((H_in + 2p_h - d_h*(k_h - 1) - 1) / s_h + 1)
W_out = floor((W_in + 2p_w - d_w*(k_w - 1) - 1) / s_w + 1)
```

Parameter count:

```text
C_out * (C_in / groups) * K_h * K_w + optional C_out bias
```

PyTorch:

```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)  # create convolution layer
```

### Attention Essentials

Scaled dot-product attention:

```python
def scaled_dot_product_attention(q, k, v, mask=None):  # define scaled_dot_product_attention
    d = q.size(-1)  # read head dimension
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # compute scaled scores
    if mask is not None:  # branch on condition
        scores = scores.masked_fill(mask == 0, float("-inf"))  # mask invalid positions
    weights = torch.softmax(scores, dim=-1)  # normalize scores
    return weights @ v, weights  # return result
```

Shapes:

```text
q: [B, H, Lq, D]
k: [B, H, Lk, D]
v: [B, H, Lk, Dv]
scores: [B, H, Lq, Lk]
out: [B, H, Lq, Dv]
```

### Mixed Precision

Modern AMP pattern:

```python
scaler = torch.amp.GradScaler("cuda")  # scale mixed-precision loss

for x, y in loader:  # iterate over items
    optimizer.zero_grad(set_to_none=True)  # clear old gradients
    with torch.amp.autocast("cuda"):  # enter context manager
        logits = model(x)  # run forward pass
        loss = criterion(logits, y)  # compute loss
    scaler.scale(loss).backward()  # compute gradients
    scaler.step(optimizer)  # step optimizer safely
    scaler.update()  # update scale factor
```

Interview note:

- AMP reduces memory and can speed up training on supported hardware.
- Validate numerical behavior and loss scaling.

### Saving and Loading

Recommended:

```python
torch.save(model.state_dict(), "model.pt")  # save checkpoint

model = MyModel(...)  # instantiate model
state_dict = torch.load("model.pt", map_location=device, weights_only=True)  # load checkpoint
model.load_state_dict(state_dict)  # restore state dict
model.to(device)  # move tensor to device
model.eval()  # enable eval mode
```

Checkpoint:

```python
torch.save({  # start mapping
    "model": model.state_dict(),  # store model weights
    "optimizer": optimizer.state_dict(),  # store optimizer state
    "epoch": epoch,  # store current epoch
}, "checkpoint.pt")  # write checkpoint file
```

### Debugging Checklist

If training fails:

- Print shapes at every boundary.
- Overfit one tiny batch.
- Check labels and target dtype.
- Check logits vs probabilities.
- Check train/eval mode.
- Check LR and optimizer.
- Check loss goes down on a small subset.
- Check gradients are not `None`, NaN, or exploding.
- Disable augmentations.
- Disable AMP.
- Compare against a simple baseline.

## Top Asked PyTorch Questions

### 1. Write a Training Loop From Memory

Question:

```text
Write a basic PyTorch training loop.
```

Answer:

```python
def train_one_epoch(model, loader, optimizer, criterion, device):  # define train_one_epoch
    model.train()  # enable training mode
    total_loss = 0.0  # accumulate loss
    total = 0  # count examples

    for x, y in loader:  # iterate over items
        x = x.to(device)  # move tensor to device
        y = y.to(device)  # move tensor to device

        optimizer.zero_grad(set_to_none=True)  # clear old gradients
        logits = model(x)  # run forward pass
        loss = criterion(logits, y)  # compute loss
        loss.backward()  # backpropagate loss
        optimizer.step()  # update parameters

        batch = x.size(0)  # read batch size
        total_loss += loss.item() * batch  # add batch loss
        total += batch  # add batch count

    return total_loss / max(total, 1)  # return result
```

Follow-up:

- Add gradient clipping after `loss.backward()` and before `optimizer.step()`.
- Use `model.eval()` plus `torch.inference_mode()` for validation.

### 2. Write an Evaluation Loop

Question:

```text
Write validation code for classification accuracy.
```

Answer:

```python
def evaluate(model, loader, criterion, device):  # define evaluate
    model.eval()  # enable eval mode
    total_loss = 0.0  # accumulate loss
    total_correct = 0  # count correct predictions
    total = 0  # count examples

    with torch.inference_mode():  # disable autograd for inference
        for x, y in loader:  # iterate over items
            x = x.to(device)  # move tensor to device
            y = y.to(device)  # move tensor to device
            logits = model(x)  # run forward pass
            loss = criterion(logits, y)  # compute loss
            pred = logits.argmax(dim=1)  # pick max index

            batch = x.size(0)  # read batch size
            total_loss += loss.item() * batch  # add batch loss
            total_correct += (pred == y).sum().item()  # add correct predictions
            total += batch  # add batch count

    return total_loss / max(total, 1), total_correct / max(total, 1)  # return result
```

Common mistake:

- Calling `model.eval()` without `inference_mode()` still tracks gradients.

### 3. Explain Autograd

Answer:

```text
PyTorch records operations on tensors with requires_grad=True into a dynamic computation graph.
When backward() is called on a scalar loss, PyTorch applies reverse-mode automatic differentiation and accumulates gradients in leaf tensors' .grad fields.
```

Code:

```python
x = torch.randn(5, requires_grad=True)  # sample normal tensor
loss = (x ** 2).mean()  # build scalar loss
loss.backward()  # backpropagate loss
print(x.grad)  # 2*x/5
```

Follow-up:

- Gradients accumulate by default, so call `zero_grad()`.

### 4. `no_grad` vs `inference_mode` vs `eval`

Answer:

```text
model.eval() changes module behavior, especially dropout and batchnorm.
torch.no_grad() disables autograd recording.
torch.inference_mode() also disables autograd recording and can be faster, but is stricter.
For validation/inference, use model.eval() plus no_grad or inference_mode.
```

### 5. `CrossEntropyLoss` Inputs

Question:

```text
What should you pass to nn.CrossEntropyLoss?
```

Answer:

```text
Pass raw logits of shape [B, C] and integer class labels of shape [B] with dtype torch.long.
Do not apply softmax first.
```

Code:

```python
criterion = nn.CrossEntropyLoss()  # create multiclass loss
logits = model(x)        # [B, C]
loss = criterion(logits, y.long())  # cast to int64 labels
```

Why:

- Cross entropy internally combines log-softmax and negative log-likelihood.

### 6. `BCEWithLogitsLoss` vs `CrossEntropyLoss`

Answer:

```text
CrossEntropyLoss is for mutually exclusive classes: one correct class among C.
BCEWithLogitsLoss is for binary or multi-label classification: each class independently on/off.
Both expect logits, not probabilities.
```

Example:

```python
multi_class_loss = nn.CrossEntropyLoss()(logits_class, target_class)  # create multiclass loss
multi_label_loss = nn.BCEWithLogitsLoss()(logits_labels, targets_float)  # create binary loss
```

### 7. `view` vs `reshape` vs `permute`

Answer:

```text
view changes shape but requires compatible contiguous memory.
reshape returns a view when possible, otherwise a copy.
permute reorders dimensions and often creates a non-contiguous view.
After permute, use reshape or contiguous().view().
```

Code:

```python
x = torch.randn(2, 3, 4)  # sample normal tensor
y = x.permute(0, 2, 1)  # reorder axes
z = y.reshape(2, 12)  # reshape tensor
```

### 8. Build a Simple `nn.Module`

Question:

```text
Write an MLP classifier.
```

Answer:

```python
class Classifier(nn.Module):  # define Classifier
    def __init__(self, in_dim, hidden_dim, num_classes):  # define __init__
        super().__init__()  # initialize base module
        self.fc1 = nn.Linear(in_dim, hidden_dim)  # create linear layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # create linear layer

    def forward(self, x):  # define forward
        x = F.relu(self.fc1(x))  # apply hidden layer
        return self.fc2(x)  # return result
```

Follow-up:

- Parameters assigned as submodules are registered automatically.

### 9. Custom Dataset and DataLoader

Question:

```text
Create a dataset wrapper for arrays X and y.
```

Answer:

```python
class ArrayDataset(Dataset):  # define ArrayDataset
    def __init__(self, X, y):  # define __init__
        self.X = torch.as_tensor(X, dtype=torch.float32)  # store feature tensor
        self.y = torch.as_tensor(y, dtype=torch.long)  # store label tensor

    def __len__(self):  # define __len__
        return self.y.numel()  # return result

    def __getitem__(self, idx):  # define __getitem__
        return self.X[idx], self.y[idx]  # return result


loader = DataLoader(ArrayDataset(X, y), batch_size=32, shuffle=True)  # inherit dataset API
```

Follow-up:

- For variable-length data, write a custom `collate_fn`.

### 10. Freeze a Backbone

Question:

```text
How do you freeze part of a model?
```

Answer:

```python
for p in model.backbone.parameters():  # iterate over items
    p.requires_grad = False  # freeze parameter

optimizer = torch.optim.AdamW(  # build optimizer
    filter(lambda p: p.requires_grad, model.parameters()),  # pass trainable params
    lr=1e-3,  # set learning rate
)  # finish optimizer call
```

Follow-up:

- BatchNorm can still update running stats in train mode. Consider setting frozen backbone to eval mode.

### 11. Overfit One Tiny Batch

Question:

```text
Your model is not learning. What is the first debugging experiment?
```

Answer:

```python
x_small, y_small = next(iter(loader))  # fetch one tiny batch
x_small, y_small = x_small.to(device), y_small.to(device)  # move tensor to device

model.train()  # enable training mode
for step in range(200):  # iterate over items
    optimizer.zero_grad(set_to_none=True)  # clear old gradients
    loss = criterion(model(x_small), y_small)  # compute loss
    loss.backward()  # backpropagate loss
    optimizer.step()  # update parameters
```

Expected:

- Loss should approach near zero for a small batch if model/loss/optimizer are wired correctly.

If not:

- Check labels, target dtype, LR, loss, final layer shape, and preprocessing.

### 12. Conv2D Output Shape and Params

Question:

```text
For input [B, C_in, H, W], compute Conv2D output shape.
```

Answer:

```text
H_out = floor((H + 2p - dilation*(kernel - 1) - 1) / stride + 1)
W_out = floor((W + 2p - dilation*(kernel - 1) - 1) / stride + 1)
params = C_out * (C_in / groups) * K_h * K_w + optional C_out bias
```

Example:

```text
Conv2d(3, 64, kernel_size=7, stride=2, padding=3), bias=True
params = 64 * 3 * 7 * 7 + 64 = 9472
```

### 13. Implement a Simple CNN

Question:

```text
Write a CNN classifier for images.
```

Answer:

```python
class SimpleCNN(nn.Module):  # define SimpleCNN
    def __init__(self, num_classes):  # define __init__
        super().__init__()  # initialize base module
        self.features = nn.Sequential(  # define feature extractor
            nn.Conv2d(3, 32, 3, padding=1),  # create convolution layer
            nn.BatchNorm2d(32),  # create batch norm
            nn.ReLU(inplace=True),  # create ReLU activation
            nn.MaxPool2d(2),  # downsample feature map
            nn.Conv2d(32, 64, 3, padding=1),  # create convolution layer
            nn.BatchNorm2d(64),  # create batch norm
            nn.ReLU(inplace=True),  # create ReLU activation
            nn.MaxPool2d(2),  # downsample feature map
        )  # close layer stack
        self.classifier = nn.Sequential(  # define classifier head
            nn.AdaptiveAvgPool2d(1),  # pool to fixed size
            nn.Flatten(),  # flatten pooled features
            nn.Linear(64, num_classes),  # create linear layer
        )  # close layer stack

    def forward(self, x):  # define forward
        return self.classifier(self.features(x))  # return result
```

Why `AdaptiveAvgPool2d(1)`:

- Makes classifier independent of input spatial size.

### 14. Implement Scaled Dot-Product Attention

Question:

```text
Implement attention with optional mask.
```

Answer:

```python
def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False):  # define scaled_dot_product_attention
    d = q.size(-1)  # read head dimension
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # compute scaled scores
    if mask is not None:  # branch on condition
        scores = scores.masked_fill(mask == 0, float("-inf"))  # mask invalid positions
    weights = torch.softmax(scores, dim=-1)  # normalize scores
    weights = F.dropout(weights, p=dropout_p, training=training)  # apply dropout
    return weights @ v, weights  # return result
```

Follow-up:

- Mask before softmax.
- Check mask convention. Some APIs use `True` to keep, others use `True` to mask.

### 15. Multi-Head Attention From Scratch

Question:

```text
Implement multi-head self-attention.
```

Answer:

```python
class MultiHeadSelfAttention(nn.Module):  # define MultiHeadSelfAttention
    def __init__(self, d_model, num_heads, dropout=0.0):  # define __init__
        super().__init__()  # initialize base module
        assert d_model % num_heads == 0  # validate assumption
        self.num_heads = num_heads  # store head count
        self.head_dim = d_model // num_heads  # compute head width
        self.qkv = nn.Linear(d_model, 3 * d_model)  # create linear layer
        self.out = nn.Linear(d_model, d_model)  # create linear layer
        self.dropout = dropout  # store dropout rate

    def forward(self, x, mask=None):  # define forward
        B, T, D = x.shape  # inspect shape
        qkv = self.qkv(x)  # project Q/K/V together
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)  # reshape tensor
        qkv = qkv.permute(2, 0, 3, 1, 4)  # reorder axes
        q, k, v = qkv[0], qkv[1], qkv[2]  # split Q, K, and V
        y, _ = scaled_dot_product_attention(  # apply attention
            q, k, v, mask=mask, dropout_p=self.dropout, training=self.training  # pass attention inputs
        )  # finish attention call
        y = y.transpose(1, 2).reshape(B, T, D)  # reshape tensor
        return self.out(y)  # return result
```

Follow-up:

- Attention cost is O(T^2 * D).
- For long sequences, consider local/sparse attention or efficient kernels.

### 16. Custom Loss: Focal Loss

Question:

```text
Implement focal loss for class imbalance.
```

Answer:

```python
def focal_loss(logits, targets, gamma=2.0, alpha=None):  # define focal_loss
    ce = F.cross_entropy(logits, targets, reduction="none", weight=alpha)  # compute CE loss
    pt = torch.exp(-ce)  # exponentiate tensor
    loss = ((1 - pt) ** gamma) * ce  # compute loss
    return loss.mean()  # return result
```

Explain:

- Down-weights easy examples.
- Useful in detection/class imbalance.
- Can hurt calibration, so evaluate calibration after training.

### 17. Gradient Clipping and Accumulation

Question:

```text
How do you handle exploding gradients or simulate a larger batch?
```

Answer:

```python
accum_steps = 4  # choose accumulation window
optimizer.zero_grad(set_to_none=True)  # clear old gradients

for step, (x, y) in enumerate(loader):  # iterate over items
    logits = model(x.to(device))  # move tensor to device
    loss = criterion(logits, y.to(device)) / accum_steps  # move tensor to device
    loss.backward()  # backpropagate loss

    if (step + 1) % accum_steps == 0:  # branch on condition
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradient norm
        optimizer.step()  # update parameters
        optimizer.zero_grad(set_to_none=True)  # clear old gradients
```

Notes:

- Divide loss by accumulation steps.
- Clip after gradients are accumulated, before optimizer step.

### 18. Save and Resume Training

Question:

```text
How do you save a checkpoint and resume?
```

Answer:

```python
torch.save({  # start mapping
    "epoch": epoch,  # store current epoch
    "model": model.state_dict(),  # store model weights
    "optimizer": optimizer.state_dict(),  # store optimizer state
    "scheduler": scheduler.state_dict() if scheduler else None,  # store scheduler state
}, "ckpt.pt")  # write checkpoint file

ckpt = torch.load("ckpt.pt", map_location=device, weights_only=True)  # load checkpoint
model.load_state_dict(ckpt["model"])  # restore state dict
optimizer.load_state_dict(ckpt["optimizer"])  # restore state dict
if scheduler and ckpt["scheduler"] is not None:  # branch on condition
    scheduler.load_state_dict(ckpt["scheduler"])  # restore state dict
start_epoch = ckpt["epoch"] + 1  # resume after saved epoch
```

Best practice:

- Save `state_dict`, not the whole model object.
- Use `weights_only=True` for state-dict style loads when supported; it avoids unpickling arbitrary Python objects.

### 19. Handle Class Imbalance

Question:

```text
Your classifier ignores rare classes. What do you do in PyTorch?
```

Answer:

Options:

```python
class_weights = torch.tensor([...], dtype=torch.float32, device=device)  # create tensor
criterion = nn.CrossEntropyLoss(weight=class_weights)  # create multiclass loss
```

Other tools:

- Weighted sampler.
- Focal loss.
- Oversampling rare classes.
- Hard negative mining.
- Threshold tuning.
- Slice metrics.

Warning:

- Weighting can affect calibration; evaluate calibrated probabilities separately.

### 20. Device or Dtype Mismatch Debugging

Question:

```text
You get a runtime error about tensor device or dtype mismatch. What do you check?
```

Answer:

```python
print(x.device, x.dtype)  # print debug value
print(next(model.parameters()).device, next(model.parameters()).dtype)  # print debug value
print(y.device, y.dtype)  # print debug value
```

Common fixes:

```python
x = x.to(device=device, dtype=torch.float32)  # move or cast tensor
y = y.to(device=device, dtype=torch.long)  # move or cast tensor
model = model.to(device)  # move tensor to device
```

Loss-specific dtype:

- `CrossEntropyLoss`: target must be `long`.
- `BCEWithLogitsLoss`: target should be float.

### 21. BatchNorm and Dropout Bug

Question:

```text
Validation metrics are noisy or worse than expected. What PyTorch mode bug might cause this?
```

Answer:

```text
Forgetting model.eval() leaves dropout active and BatchNorm using/updating batch statistics.
Use model.eval() for validation and model.train() for training.
```

Correct validation:

```python
model.eval()  # enable eval mode
with torch.inference_mode():  # disable autograd for inference
    logits = model(x)  # run forward pass
```

### 22. Implement Early Stopping

Question:

```text
Implement basic early stopping on validation loss.
```

Answer:

```python
best = float("inf")  # initialize best metric
patience = 5  # set patience limit
bad_epochs = 0  # reset stale counter

for epoch in range(num_epochs):  # iterate over items
    train_one_epoch(...)  # run training epoch
    val_loss, _ = evaluate(...)  # validate model

    if val_loss < best:  # branch on condition
        best = val_loss  # update best metric
        bad_epochs = 0  # reset stale counter
        torch.save(model.state_dict(), "best.pt")  # save checkpoint
    else:  # handle remaining case
        bad_epochs += 1  # increment stale counter
        if bad_epochs >= patience:  # branch on condition
            break  # stop early
```

Follow-up:

- Use a fixed validation set and monitor critical slices, not only aggregate loss.

### 23. Write a Custom Collate Function

Question:

```text
Batch variable-length sequences.
```

Answer:

```python
def pad_collate(batch):  # define pad_collate
    sequences, labels = zip(*batch)  # split sequences and labels
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)  # create tensor
    max_len = lengths.max().item()  # find padded length
    feat_dim = sequences[0].shape[-1]  # inspect shape
    x = torch.zeros(len(sequences), max_len, feat_dim)  # allocate zeros
    for i, s in enumerate(sequences):  # iterate over items
        x[i, :len(s)] = s  # slice tensor values
    y = torch.tensor(labels, dtype=torch.long)  # create tensor
    mask = torch.arange(max_len)[None, :] < lengths[:, None]  # create range tensor
    return x, y, mask  # return result
```

Follow-up:

- Mask padded tokens in attention/loss.
- Move the returned mask to the same device as the batch before using it in GPU attention or loss code.

### 24. Mixed Precision Training

Question:

```text
How do you train with AMP?
```

Answer:

```python
scaler = torch.amp.GradScaler("cuda")  # scale mixed-precision loss

for x, y in loader:  # iterate over items
    x, y = x.to(device), y.to(device)  # move tensor to device
    optimizer.zero_grad(set_to_none=True)  # clear old gradients
    with torch.amp.autocast("cuda"):  # enter context manager
        logits = model(x)  # run forward pass
        loss = criterion(logits, y)  # compute loss
    scaler.scale(loss).backward()  # compute gradients
    scaler.step(optimizer)  # step optimizer safely
    scaler.update()  # update scale factor
```

Follow-up:

- Disable AMP while debugging NaNs.

### 25. PyTorch vs NumPy

Question:

```text
How is PyTorch different from NumPy?
```

Answer:

```text
Both provide n-dimensional arrays/tensors and vectorized ops.
PyTorch adds GPU acceleration, autograd, neural network modules, optimizers, and training utilities.
NumPy is the core CPU numerical array library; PyTorch is built for differentiable tensor programs.
```

Bridge:

```python
t = torch.from_numpy(arr)   # shares memory when possible on CPU
arr = t.detach().cpu().numpy()  # detach from graph
```

Warning:

- Shared memory means mutation can affect both.
- Move CUDA tensor to CPU before `.numpy()`.

## PyTorch Interview Red Flags

- Applying softmax before `CrossEntropyLoss`.
- Using float targets with `CrossEntropyLoss`.
- Forgetting `optimizer.zero_grad()`.
- Forgetting `model.eval()` during validation.
- Thinking `model.eval()` disables gradients.
- Calling `.view()` after `permute()` without `.contiguous()`.
- Creating tensors on CPU inside `forward()` while model is on GPU.
- Not moving labels to device.
- Using `.data` instead of `detach()` or `no_grad()`.
- Not checking for NaNs or gradient norms.
- Ignoring masks for padded sequences.
- Saving whole model object instead of `state_dict`.

## PyTorch Debugging Playbook

### Shape Mismatch

Do:

```python
print("x", x.shape)  # inspect shape
print("logits", logits.shape)  # inspect shape
print("y", y.shape, y.dtype)  # inspect shape
```

Check:

- Batch dimension.
- Channel order: NCHW vs NHWC.
- Flatten size.
- Sequence length axis.
- Class dimension.

### Loss Does Not Decrease

Do:

- Overfit one tiny batch.
- Print initial loss and post-update loss.
- Check `requires_grad`.
- Check optimizer receives trainable params.
- Check LR.
- Check labels.
- Disable augmentation.
- Disable AMP.

### Gradients Are None

Likely causes:

- Used `.detach()`.
- Wrapped forward in `no_grad`.
- Parameter not used in forward.
- Parameter has `requires_grad=False`.
- Loss not connected to model output.

Debug:

```python
for name, p in model.named_parameters():  # iterate over items
    print(name, p.requires_grad, p.grad is None)  # inspect gradients
```

### NaNs

Check:

- Learning rate too high.
- `log(0)` or division by zero.
- Exploding gradients.
- AMP overflow.
- Bad labels or invalid inputs.

Tools:

```python
torch.autograd.set_detect_anomaly(True)  # trace autograd anomalies
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradient norm
```

## 3-Day PyTorch Drill Plan

Day 1:

- Tensor shape ops.
- `nn.Module`.
- Training and evaluation loops.
- CrossEntropy/BCEWithLogits target shapes.

Day 2:

- Dataset/DataLoader.
- CNN output shapes.
- Simple CNN.
- Freeze backbone and class imbalance.

Day 3:

- Attention.
- Gradient clipping/accumulation.
- Checkpointing.
- Debugging non-convergence and NaNs.
- One full timed PyTorch coding mock.

## Source Notes

Official PyTorch:

- [torch.Tensor reference](https://docs.pytorch.org/docs/stable/tensors.html)
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [torch.utils.data](https://docs.pytorch.org/docs/stable/data.html)
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.conv.Conv2d.html)
- [scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [torch.amp](https://docs.pytorch.org/docs/stable/amp.html)
- [Serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)

Local notes used:

- `/Users/muditj/Desktop/Projects/daily-brief/notes/DeepML-Hard.md`
- `/Users/muditj/Desktop/Projects/daily-brief/notes/DeepML-Vision-Transformer.md`
- `/Users/muditj/Desktop/Projects/daily-brief/notes/Cheatsheet-02-Machine-Learning.md`
- `/Users/muditj/Desktop/Projects/daily-brief/notes/Cheatsheet-03-ML-Coding.md`

Interview-theme search signals:

- Public prep sources repeatedly emphasize autograd, `nn.Module`, training loops, train/eval mode, loss input shapes, Conv2D, attention, DataLoader, and debugging broken training pipelines. Treat these as question-theme signals, not technical authorities.
