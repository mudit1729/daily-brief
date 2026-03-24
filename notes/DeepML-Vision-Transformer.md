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

---

## Task 02: Position Embedding Add

Z_0 = E + E_pos — element-wise add learned position embeddings to patch embeddings

```python
def position_embedding_add(patch_embeddings: torch.Tensor, position_embeddings: torch.Tensor) -> torch.Tensor:
    return (patch_embeddings + position_embeddings).to(torch.float32) # (N, D) + (N, D)
```

---

## Task 03: Class Token Prepend

Z_0' = [x_class; Z_0] — prepend learnable CLS token to patch sequence

```python
def class_token_prepend(patch_embeddings: torch.Tensor, class_token: torch.Tensor) -> torch.Tensor:
    return torch.cat([class_token, patch_embeddings], dim=0).to(torch.float32) # (N+1, D)
```

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

---

## Task 09: ViT Classification Head

y = Z_L[0] @ W_cls + b_cls — only the CLS token (index 0) is used for classification

```python
def vit_classification_head(encoder_output, W_cls, b_cls):
    cls_token = encoder_output[0] # (D,) — extract CLS token representation
    return (cls_token @ W_cls + b_cls).to(torch.float32) # (num_classes,) — logits
```

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
