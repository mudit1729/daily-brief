# Vision Transformer (ViT) - Complete Solutions

**Paper**: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
**URL**: https://arxiv.org/abs/2010.11929

---

## Task 01: Patch Embedding Forward

Image (H,W,C) → sequence of flattened patches (N, P²·C) where N = HW/P²

```python
import torch

def patch_embedding_forward(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    H, W, C = image.shape
    P = patch_size
    nH, nW = H // P, W // P
    # reshape → (nH, P, nW, P, C), transpose → (nH, nW, P, P, C), flatten → (N, P²C)
    patches = image.reshape(nH, P, nW, P, C).permute(0, 2, 1, 3, 4).reshape(nH * nW, P * P * C)
    return patches.to(torch.float32)
```

**Theory**: ViT treats an image as a sequence of tokens, just like words in NLP. Each non-overlapping P×P patch is flattened into a vector of size P²C. For a 224×224 image with P=16: N = 14×14 = 196 tokens, each of dimension 16×16×3 = 768. This is equivalent to a Conv2d with kernel_size=P and stride=P.

**Interview Q&A**:
- **Why not use overlapping patches?** Non-overlapping patches = every pixel appears exactly once. Overlapping would increase sequence length and compute. Some variants (Swin) use overlapping for hierarchical features.
- **reshape-permute-reshape trick?** This is the key operation: reshape isolates patch boundaries, permute reorders to row-major, final reshape flattens. Alternative: `F.unfold(image, P, stride=P)` or `nn.Conv2d(C, D, kernel_size=P, stride=P)`.
- **Conv2d vs manual patching?** In practice, ViT implementations use `nn.Conv2d(3, d_model, kernel_size=P, stride=P)` which combines patch extraction + linear projection in one efficient operation. The manual version here separates them for clarity.
- **Patch size tradeoffs?** Smaller P → more tokens → more compute (O(N²) attention), but finer spatial resolution. P=16 (ViT-B) balances compute and resolution. P=14 (ViT-L) or P=8 are used for high-res tasks.

---

## Task 02: Position Embedding Add

Z_0 = E + E_pos — element-wise add learned position embeddings to patch embeddings

```python
def position_embedding_add(patch_embeddings: torch.Tensor, position_embeddings: torch.Tensor) -> torch.Tensor:
    return (patch_embeddings + position_embeddings).to(torch.float32) # (N, D) + (N, D)
```

**Theory**: Unlike the original Transformer which uses sinusoidal PE, ViT uses learned position embeddings. Each of the N patch positions gets a learnable vector. After training, nearby positions have similar embeddings — the model learns 2D spatial structure despite receiving a 1D sequence.

**Interview Q&A**:
- **Why learned instead of sinusoidal?** ViT paper found no significant difference. Learned PE can capture 2D grid structure; sinusoidal PE is designed for 1D sequences. 2D-aware PE (separate row + column embeddings) also showed no improvement.
- **How does ViT handle different resolutions?** Position embeddings are trained for a fixed grid (e.g., 14×14). For different resolutions, interpolate the PE using bicubic interpolation on the 2D grid. This is why ViT can fine-tune on higher resolutions than pre-training.
- **Position embeddings for CLS token?** The CLS token gets its own position embedding (index 0). Patch positions are indices 1..N. So the full PE has shape (N+1, D).

---

## Task 03: Class Token Prepend

Z_0' = [x_class; Z_0] — prepend learnable CLS token to patch sequence

```python
def class_token_prepend(patch_embeddings: torch.Tensor, class_token: torch.Tensor) -> torch.Tensor:
    return torch.cat([class_token, patch_embeddings], dim=0).to(torch.float32) # (N+1, D)
```

**Theory**: Borrowed from BERT. The CLS token is a learnable vector prepended to the patch sequence. Through self-attention, it aggregates information from all patches. After the encoder, only the CLS token's output is used for classification — it becomes a global image representation.

**Interview Q&A**:
- **CLS token vs global average pooling (GAP)?** ViT paper found CLS and GAP perform similarly. DeiT and later work often prefer GAP (average all patch representations) as it's simpler and equally effective. CLIP uses CLS token.
- **Why prepend and not append?** Convention from BERT. Position doesn't matter much since attention is permutation-equivariant (position info comes from PE, not order).
- **What happens to CLS during attention?** CLS attends to all patches and all patches attend to CLS. After L layers, CLS has progressively aggregated information from all spatial locations. It functions as a learned pooling mechanism.

---

## Task 04: ViT MLP Block Forward

MLP(x) = GELU(xW1 + b1) W2 + b2 — two-layer FFN with GELU (not ReLU like original Transformer)

```python
import torch.nn.functional as F

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate='tanh') # 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x³)))

def vit_mlp_block_forward(x, W1, b1, W2, b2):
    hidden = gelu(x @ W1 + b1) # (N, D) → (N, D_ff) — expand + activate
    return (hidden @ W2 + b2).to(torch.float32) # (N, D_ff) → (N, D) — project back
```

**Theory**: The FFN processes each token independently (unlike attention which mixes tokens). It expands to d_ff = 4×d_model, applies non-linearity, then projects back. This is where the model stores "knowledge" — attention routes information, FFN transforms it. GELU is smoother than ReLU, avoiding the "dead neuron" problem.

**Interview Q&A**:
- **Why GELU instead of ReLU?** GELU = x · Φ(x), where Φ is the Gaussian CDF. It smoothly gates activations — small negative values are partially passed, unlike ReLU's hard cutoff. Empirically better for transformers. GPT and BERT both use GELU.
- **What is SwiGLU?** SwiGLU(x) = Swish(xW1) ⊙ (xW3) — a gated variant used in LLaMA/PaLM. Uses 3 weight matrices instead of 2, with d_ff = 8/3 * d_model to keep param count equal. Better quality than GELU FFN.
- **Why expand then project?** The bottleneck architecture (D → 4D → D) lets the model learn non-linear transformations in a higher-dimensional space. Similar intuition to autoencoders: the expansion creates a rich intermediate representation.
- **What fraction of params are in FFN?** In a standard transformer layer: MHA = 4d² params, FFN = 8d² params. So FFN has 2/3 of each layer's parameters. This is why FFN is often called the "memory" of the transformer.

---

## Task 05: ViT Encoder Layer Forward

Pre-LN: x' = x + MHA(LN(x)), x'' = x' + MLP(LN(x')) — residual around each sub-block

```python
import math

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = torch.mean(x, dim=-1, keepdim=True) # per-token mean
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False) # population variance
    return (gamma * (x - mean) / torch.sqrt(var + eps) + beta).to(torch.float32)

def vit_encoder_layer_forward(x, attn_params, mlp_params, ln1_params, ln2_params):
    # 1. Pre-LN → Multi-head attention → residual
    x_norm = layer_norm(x, ln1_params['gamma'], ln1_params['beta'])
    attn_out = multi_head_attention_forward(x_norm, attn_params['W_q'], attn_params['W_k'],
                                             attn_params['W_v'], attn_params['W_o'], attn_params['num_heads'])
    x = x + attn_out # residual connection
    # 2. Pre-LN → MLP → residual
    x_norm = layer_norm(x, ln2_params['gamma'], ln2_params['beta'])
    x = x + vit_mlp_block_forward(x_norm, mlp_params['W1'], mlp_params['b1'],
                                   mlp_params['W2'], mlp_params['b2'])
    return x.to(torch.float32)
```

**Theory**: ViT uses Pre-LN (normalize before each sub-layer), which is more stable than the original Transformer's Post-LN. Residual connections ensure gradients flow directly through the identity path — without them, deep transformers are untrainable. The two sub-blocks serve complementary roles: attention mixes information across tokens, FFN transforms each token's representation.

**Interview Q&A**:
- **Why Pre-LN > Post-LN for training stability?** Pre-LN keeps the residual path "clean" — gradients flow through x + f(LN(x)) without being distorted by LN. Post-LN applies LN to the sum, which can amplify or suppress the residual signal. Pre-LN models converge with larger learning rates.
- **What's the gradient flow through a residual block?** ∂L/∂x = ∂L/∂out · (1 + ∂f/∂x). The "1" term means gradients always flow, even if the sub-layer's gradient vanishes. This is why ResNets and Transformers can be hundreds of layers deep.
- **Dropout in transformers?** Applied to: (1) attention weights after softmax, (2) residual connections (after sub-layer, before addition), (3) embedding layer. ViT uses 0.1 dropout typically.

---

## Task 06: Scaled Dot-Product Attention

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k) # (N, M) — similarity scores
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9) # True positions → -inf before softmax
    attn_weights = torch.softmax(scores, dim=-1) # normalize over keys
    return (attn_weights @ V).to(torch.float32) # (N, d_v) — weighted sum of values
```

**Note**: This is the unbatched ViT version (2D inputs). See Attention-Is-All-You-Need Task 02 for the batched 3D version.

---

## Task 07: Multi-Head Attention Forward

MultiHead(x) = Concat(head_1,...,head_h) W_O — split D into h heads of d_k = D/h

```python
def multi_head_attention_forward(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    N, D = x.shape
    d_k = D // num_heads
    # project: (N, D) @ (D, D) → (N, D), then split heads: (N, D) → (h, N, d_k)
    Q = (x @ W_q).reshape(N, num_heads, d_k).permute(1, 0, 2) # (h, N, d_k)
    K = (x @ W_k).reshape(N, num_heads, d_k).permute(1, 0, 2)
    V = (x @ W_v).reshape(N, num_heads, d_k).permute(1, 0, 2)
    # per-head attention
    heads = []
    for i in range(num_heads):
        scores = (Q[i] @ K[i].transpose(-2, -1)) / math.sqrt(d_k) # (N, N)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        heads.append(torch.softmax(scores, dim=-1) @ V[i]) # (N, d_k)
    # concat heads → output projection: (N, D) @ (D, D)
    return (torch.cat(heads, dim=-1) @ W_o).to(torch.float32) # (N, D)
```

**Note**: This uses a per-head loop (matching the DeepML functional API). The batched version in the Attention file merges heads into the batch dimension for a single matmul call — much more efficient for GPU.

**Interview Q&A**:
- **What do ViT attention heads learn?** Early layers: local spatial patterns (nearby patches). Middle layers: object-part relationships. Deep layers: global scene understanding. Some heads develop behavior similar to conv filters; others attend to all patches uniformly.
- **ViT vs CNN attention?** CNN has local receptive fields that grow with depth. ViT has global attention from layer 1 — every patch can attend to every other patch. This is powerful but data-hungry (ViT needs ~300M images to match CNN quality from scratch). With enough data or pretraining, ViT wins.

---

## Task 08: ViT Encoder Forward

Z_l = EncoderLayer_l(Z_{l-1}) for l = 1..L — sequential stack of identical encoder layers

```python
def vit_encoder_forward(x, layers_params):
    for lp in layers_params: # apply each encoder layer sequentially
        x = vit_encoder_layer_forward(x, lp['attn_params'], lp['mlp_params'],
                                       lp['ln1_params'], lp['ln2_params'])
    return x.to(torch.float32) # (N, D) — same shape in/out
```

**Theory**: Each encoder layer refines the token representations. Early layers capture low-level features; deeper layers build semantic understanding. ViT-Base has 12 layers, ViT-Large has 24, ViT-Huge has 32.

**Interview Q&A**:
- **ViT model sizes?** ViT-B/16: 12 layers, d=768, h=12, 86M params. ViT-L/16: 24 layers, d=1024, h=16, 307M params. ViT-H/14: 32 layers, d=1280, h=16, 632M params. The /16 or /14 denotes patch size.
- **Why does ViT need more data than CNN?** CNNs have strong inductive biases: locality (conv kernels), translation equivariance (weight sharing), hierarchical features (pooling). ViT has almost none — it must learn these properties from data. This is why ViT underperforms CNNs on ImageNet-1K but surpasses them with JFT-300M pretraining.

---

## Task 09: ViT Classification Head

y = Z_L[0] @ W_cls + b_cls — only the CLS token (index 0) is used for classification

```python
def vit_classification_head(encoder_output, W_cls, b_cls):
    cls_token = encoder_output[0] # (D,) — extract CLS token representation
    return (cls_token @ W_cls + b_cls).to(torch.float32) # (num_classes,) — logits
```

**Theory**: After L layers of self-attention, the CLS token has aggregated information from all patches. A single linear layer projects this d-dimensional vector to num_classes logits. This is simpler than CNN classifiers which often use global average pooling + FC layer.

**Interview Q&A**:
- **Why no hidden layer in the classification head?** During pretraining, ViT uses a 1-hidden-layer MLP head. During fine-tuning, a single linear layer suffices because the encoder already produces a rich representation.
- **For detection/segmentation?** Drop the CLS token, use all N patch outputs as spatial feature maps. ViTDet (FAIR) and SegViT do this. The patch outputs form a 14×14 or similar spatial grid that can be upsampled.

---

## Task 10: ViT Forward Pipeline

Image → Patches → +PosEmbed → +CLS → Encoder → Classification Head

```python
def vit_forward_pipeline(image, patch_size, position_embeddings, class_token,
                         encoder_params, W_cls, b_cls):
    patches = patch_embedding_forward(image, patch_size) # (N, P²C) — image to patch tokens
    embedded = position_embedding_add(patches, position_embeddings) # (N, D) — add position info
    sequence = class_token_prepend(embedded, class_token) # (N+1, D) — prepend CLS
    encoded = vit_encoder_forward(sequence, encoder_params) # (N+1, D) — transformer processing
    return vit_classification_head(encoded, W_cls, b_cls) # (num_classes,) — classify from CLS
```

**Theory**: The full ViT pipeline is remarkably simple: patch → embed → transform → classify. No pooling layers, no normalization between stages, no feature pyramids. The transformer encoder does all the heavy lifting. This simplicity is ViT's advantage — it scales predictably with compute and data.

**Interview Q&A**:
- **What's missing from this implementation?** Linear projection after patching (Conv2d), dropout, final LayerNorm before classification head, proper weight initialization (truncated normal with std=0.02).
- **How is ViT trained?** Pretrain on large dataset (ImageNet-21K or JFT-300M) with cross-entropy loss. Fine-tune on target task at higher resolution (384×384 vs 224×224 pretraining) with interpolated position embeddings. Use strong data augmentation (RandAugment, Mixup, CutMix) to compensate for lack of inductive bias.

---

## Key Interview Concepts

**ViT vs CNN comparison**:
- CNNs: local receptive fields, translation equivariance, hierarchical features, parameter efficient, strong with limited data
- ViT: global attention from layer 1, no built-in spatial bias, needs more data, scales better with compute, simpler architecture

**Hybrid approaches**:
- **DeiT** (Data-efficient Image Transformers): Uses distillation from CNN teacher + strong augmentation to train ViT on ImageNet-1K alone
- **Swin Transformer**: Hierarchical ViT with shifted windows — combines CNN's multi-scale design with transformer attention. O(n) instead of O(n²).
- **ConvNeXt**: "Modernized" ResNet that adopts transformer training recipes — shows CNNs can match ViT with the right tricks

**Scaling laws**: ViT performance improves log-linearly with compute, data, and model size. Doubling compute → consistent accuracy gain. This is why Google/Meta invest in scaling ViT rather than designing new architectures.

**Inference efficiency**: ViT's main bottleneck is O(N²) attention where N = (H/P)². For 224×224 with P=16: N=196 (fast). For 1024×1024 with P=16: N=4096 (slow). Solutions: FlashAttention, window attention, token pruning.

**MAE (Masked Autoencoder)**: Self-supervised pretraining for ViT. Mask 75% of patches, encode visible patches, decode to reconstruct masked patches. Learns strong representations without labels. Analogous to BERT's masked language modeling.
