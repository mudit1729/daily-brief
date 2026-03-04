# ViT: An Image is Worth 16x16 Words -- Transformers for Image Recognition at Scale

## Paper Overview

| | |
|---|---|
| **Authors** | Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. (Google Research) |
| **Published** | 2020 (ICLR 2021) |
| **Paper Link** | https://arxiv.org/abs/2010.11929 |

---

## Detailed Description

**Vision Transformer** (ViT) demonstrates that a pure transformer architecture, applied directly to sequences of image patches, can perform excellently on image classification tasks when pre-trained on large amounts of data. This paper challenged the dominance of convolutional neural networks (CNNs) in computer vision and showed that the transformer architecture, originally designed for NLP, could be adapted for vision tasks with remarkable success.

### Key Architecture Components

1. **Patch Embedding:**
   - Input image is divided into fixed-size patches (typically 16x16 pixels)
   - Each patch is linearly embedded into a vector
   - For a 224x224 image with 16x16 patches, you get 196 patches
   - Patches are flattened and projected to embedding dimension

2. **Position Embeddings:**
   - Since transformers don't have built-in notion of position, learnable position embeddings are added
   - Each patch position gets a unique embedding
   - Position embeddings are learned during training (not fixed like in original Transformer)

3. **[CLS] Token:**
   - Special classification token prepended to the sequence
   - Similar to BERT's [CLS] token
   - The representation of this token is used for classification
   - Aggregates information from all patches through self-attention

4. **Transformer Encoder:**
   - Standard transformer encoder (same as BERT)
   - Multi-head self-attention layers
   - MLP blocks with GELU activation
   - Layer normalization (*pre-norm* architecture)
   - Residual connections

5. **Classification Head:**
   - MLP head attached to [CLS] token representation
   - During pre-training: larger MLP
   - During fine-tuning: single linear layer

### Model Variants

| Variant | Layers | Hidden Dim | Attention Heads | Parameters |
|---------|--------|-----------|-----------------|------------|
| **ViT-Base** | 12 | 768 | 12 | ~86M |
| **ViT-Large** | 24 | 1024 | 16 | ~307M |
| **ViT-Huge** | 32 | 1280 | 16 | ~632M |

### Training Strategy

1. **Pre-training:**
   - Trained on large datasets (ImageNet-21K, JFT-300M)
   - Standard supervised learning with classification objective
   - Large batch sizes (e.g., 4096)
   - Long training schedules

2. **Fine-tuning:**
   - Higher resolution images often used
   - Position embeddings interpolated for different resolutions
   - Shorter training compared to pre-training

### Mathematical Formulation

**Patch Embedding:**
```math
Given image x in R^(H x W x C)
Divide into N patches: x_p in R^(N x P^2 x C) where P is patch size
Linear projection: z_0 = [x_class; E * x_p^1; E * x_p^2; ...; E * x_p^N] + E_pos
```

**Transformer Encoder:**
```math
z'_l = MSA(LN(z_{l-1})) + z_{l-1}        (Multi-Head Self-Attention)
z_l  = MLP(LN(z'_l)) + z'_l              (Feed-Forward)
```

**Classification:**
```math
y = LN(z_L^0)                             (Extract [CLS] token)
output = MLP_head(y)                       (Classification head)
```

---

## Pain Point Addressed

### Limitations of Convolutional Networks

1. **Inductive Biases:**
   - CNNs have strong inductive biases (locality, translation equivariance)
   - These biases are helpful for small data but may limit learning from large datasets
   - Transformers have fewer inductive biases, allowing them to learn patterns directly from data

2. **Limited Receptive Fields:**
   - CNNs require deep stacks to achieve global receptive fields
   - Even with deep networks, global context is limited
   - Transformers have global receptive field from the first layer via self-attention

3. **Fixed Kernel Sizes:**
   - Convolutional kernels have fixed sizes (3x3, 5x5, etc.)
   - Multi-scale processing requires explicit architecture design (pyramids, FPN)
   - Transformers naturally capture multi-scale dependencies through attention

4. **Scalability:**
   - CNN performance plateaus with model size
   - Unclear how to scale CNNs to hundreds of billions of parameters
   - Transformers have clear scaling laws from NLP

5. **Transfer Learning Gap:**
   - NLP had highly successful transfer learning with transformers (BERT, GPT)
   - Vision still relied on CNNs with less effective transfer learning
   - Needed unified architecture for both modalities

6. **Computational Efficiency at Scale:**
   - While CNNs are efficient for small images, large high-resolution images become expensive
   - Self-attention scales quadratically but parallelizes well
   - On large datasets with big models, ViT can be more efficient

---

## Novelty of the Paper

### Key Innovations

1. **Minimal Vision-Specific Modifications:**
   - Applied standard transformer architecture with minimal changes
   - No specialized components for vision
   - Demonstrated that general-purpose transformers work for vision

2. **Patch-Based Approach:**
   - Treating image patches as "tokens" in a sequence
   - Elegant and simple: no convolutions needed
   - Direct application of NLP transformer architecture

3. **Scaling Results:**
   - Demonstrated that ViT performance improves consistently with:
     - Larger models
     - More data
     - Longer training
   - Showed scaling laws similar to NLP transformers

4. **Superior Performance with Large Data:**
   - When pre-trained on large datasets (JFT-300M), ViT surpasses state-of-the-art CNNs
   - Achieved 88.55% top-1 accuracy on ImageNet with ViT-Huge
   - Competitive with best CNNs while being more computationally efficient to train

5. **Data Efficiency Analysis:**
   - Showed that ViT requires large datasets to outperform CNNs
   - With small datasets (ImageNet-1K only), CNNs perform better
   - Provided insights into the role of inductive biases

6. **Transferability:**
   - Excellent transfer learning performance
   - Pre-trained ViT models transfer better than CNN models
   - Strong performance across diverse downstream tasks

7. **Computational Efficiency:**
   - Despite quadratic complexity of attention, ViT is competitive in compute
   - More efficient than ResNet-152 and EfficientNet in terms of compute vs accuracy
   - Better scaling with model size

### Impact on Computer Vision

> ViT sparked the "transformer revolution" in computer vision, leading to numerous variants (DeiT, Swin Transformer, BEiT, MAE) and enabling unified architectures for multi-modal models (CLIP, DALL-E).

---

## Implementation

Below is a complete PyTorch implementation of Vision Transformer from the repository:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

# DATA LOADING & VISUALIZATION
to_tensor = Compose([Resize((144, 144)), ToTensor()])
dataset = OxfordIIITPet(root=".", download=True, transform=to_tensor,
                        target_types="category")
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

def show_images(dataloader, num_samples=20, cols=4):
    plt.figure(figsize=(15, 15))
    to_pil = ToPILImage()
    images, _ = next(iter(dataloader))
    for i in range(min(num_samples, len(images))):
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(to_pil(images[i]))
        plt.axis("off")
    plt.show()

# MODEL COMPONENTS
class PatchEmbedding(nn.Module):
    """
    Converts image into sequence of patches and projects them to embedding dimension
    """
    def __init__(self, in_channels=3, patch_size=8, emb_size=128):
        super().__init__()
        self.patch_size = patch_size

        # Rearrange: splits image into patches and flattens them
        # Then projects to embedding dimension
        self.projection = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

class Attention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,
                                        dropout=dropout)

    def forward(self, x):
        # nn.MultiheadAttention expects (seq_len, batch, dim)
        x = x.transpose(0, 1)
        attn_output, _ = self.att(x, x, x)  # Self-attention: Q=K=V=x
        return attn_output.transpose(0, 1)

class PreNorm(nn.Module):
    """
    Layer normalization before the function (Pre-LN architecture)
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Sequential):
    """
    Position-wise feed-forward network
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # GELU activation (used in ViT paper)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

class Residual(nn.Module):
    """
    Residual connection wrapper
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res  # Residual connection
        return x

# VISION TRANSFORMER MODEL
class ViT(nn.Module):
    """
    Vision Transformer model

    Args:
        ch: Number of input channels
        img_size: Input image size (assumes square images)
        patch_size: Size of each patch
        emb_dim: Embedding dimension
        n_layers: Number of transformer encoder layers
        out_dim: Number of output classes
        dropout: Dropout rate
        heads: Number of attention heads
    """
    def __init__(self, ch=3, img_size=144, patch_size=4, emb_dim=32,
                 n_layers=6, out_dim=37, dropout=0.1, heads=2):
        super(ViT, self).__init__()

        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                             patch_size=patch_size,
                                             emb_size=emb_dim)

        # Positional embeddings
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1,
                                                      emb_dim))

        # [CLS] token
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Transformer encoder layers
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                # Multi-head self-attention with residual and pre-norm
                Residual(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads,
                                                    dropout=dropout))),
                # Feed-forward with residual and pre-norm
                Residual(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim * 4,
                                                     dropout=dropout)))
            )
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, img):
        # Patch embedding
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Prepend [CLS] token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x += self.pos_embedding[:, :(n + 1)]

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Classification using [CLS] token
        return self.head(x[:, 0, :])  # Take only [CLS] token output

# TRAINING SETUP
train_split = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset,
                                          [train_split,
                                           len(dataset) - train_split])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# TRAINING LOOP
num_epochs = 50
train_loss_history = []
test_loss_history = []

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    train_loss_history.append(avg_train_loss)

    if epoch % 5 == 0:
        print(f">>> Epoch {epoch} train loss: {avg_train_loss:.4f}")

    # Evaluation
    model.eval()
    test_losses = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

    avg_test_loss = np.mean(test_losses)
    test_loss_history.append(avg_test_loss)
    print(f">>> Epoch {epoch} test loss: {avg_test_loss:.4f}")

# VISUALIZATION & EVALUATION
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_loss_history, label="Train Loss",
         marker="o")
plt.plot(range(num_epochs), test_loss_history, label="Test Loss",
         marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Test Loss")
plt.legend()
plt.grid()
plt.show()

# Test predictions
inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)

print("Predicted classes:", outputs.argmax(-1))
print("Actual classes:", labels)
```

---

## Key Results from the Paper

| Metric | Result |
|--------|--------|
| **ViT-H/14 on ImageNet** | 88.55% top-1 accuracy (pre-trained on JFT-300M) |
| **vs BiT-L (ResNet-152x4)** | Outperformed: 88.55% vs 87.54% |
| **Compute efficiency** | ~2.5x less compute to pre-train |
| **Transfer learning** | Excellent on 19 downstream tasks |
| **ViT-L/16 vs EfficientNet-L2** | ~5x less compute for same performance |

---

## Key Takeaways

> 1. **Transformers work for vision:** Pure transformer architecture is viable for computer vision
> 2. **Scale matters:** ViT shines with large datasets and models
> 3. **Inductive biases trade-off:** Fewer biases require more data but enable better scaling
> 4. **Unified architecture:** Same architecture works for NLP and vision
> 5. **Patch-based approach:** Treating images as sequences of patches is effective
> 6. **Transfer learning:** Pre-trained ViTs transfer excellently to downstream tasks

---

## Repository Reference

Implementation source: https://github.com/atullchaurasia/problem-solving/tree/main/ViT
