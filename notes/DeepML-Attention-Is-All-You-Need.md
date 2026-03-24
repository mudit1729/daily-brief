# Attention Is All You Need - Complete Solutions

**Paper**: Attention Is All You Need (Vaswani et al., 2017)
**URL**: https://arxiv.org/abs/1706.03762

---

## Task 01: Positional Encoding

PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

```python
import torch, math

def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float32).reshape(-1, 1) # (seq_len, 1)
    dim_indices = torch.arange(d_model, dtype=torch.float32).reshape(1, -1) # (1, d_model)
    angles = positions / (10000 ** (2 * (dim_indices // 2) / d_model)) # (seq_len, d_model) — broadcast pos/freq
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles[:, 0::2]) # even dims: sin
    pe[:, 1::2] = torch.cos(angles[:, 1::2]) # odd dims: cos
    return pe.unsqueeze(0) # (1, seq_len, d_model)
```

---

## Task 02: Scaled Dot-Product Attention

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: (B, seq_q, d_k), K: (B, seq_k, d_k), V: (B, seq_k, d_v)
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k) # (B, seq_q, seq_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf')) # True positions → -inf before softmax
    attn_weights = torch.softmax(scores, dim=-1) # (B, seq_q, seq_k) — normalize over keys
    output = attn_weights @ V # (B, seq_q, d_v) — weighted sum of values
    return output, attn_weights
```

---

## Task 03: Single Attention Head

Q = XW_Q, K = XW_K, V = XW_V, out = Attention(Q,K,V) @ W_O

```python
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Parameter(torch.randn(d_model, d_k)) # project to query space
        self.W_k = nn.Parameter(torch.randn(d_model, d_k)) # project to key space
        self.W_v = nn.Parameter(torch.randn(d_model, d_k)) # project to value space
        self.W_o = nn.Parameter(torch.randn(d_k, d_model)) # project back to model dim

    def forward(self, x, mask=None):
        Q, K, V = x @ self.W_q, x @ self.W_k, x @ self.W_v # (B, S, d_k) each
        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        return output @ self.W_o, attn_weights # (B, S, d_model), (B, S, S)
```

---

## Task 04: Multi-Head Attention

MultiHead(X) = Concat(head_1, ..., head_h) W_O — each head operates on d_k = d_model/h

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads, self.d_k = num_heads, d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model) # single matrix, split into heads after
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        h, d_k = self.num_heads, self.d_k
        # project & reshape: (B, S, d_model) → (B*h, S, d_k) — merge batch+heads for parallel attn
        Q = self.W_Q(x).reshape(B, S, h, d_k).transpose(1, 2).reshape(B * h, S, d_k)
        K = self.W_K(x).reshape(B, S, h, d_k).transpose(1, 2).reshape(B * h, S, d_k)
        V = self.W_V(x).reshape(B, S, h, d_k).transpose(1, 2).reshape(B * h, S, d_k)
        if mask is not None: # expand mask for each head: (B, S, S) → (B*h, S, S)
            mask = mask.unsqueeze(1).expand(B, h, S, S).contiguous().reshape(B * h, S, S)
        output, _ = scaled_dot_product_attention(Q, K, V, mask=mask) # (B*h, S, d_k)
        # concat heads: (B*h, S, d_k) → (B, S, d_model)
        output = output.reshape(B, h, S, d_k).transpose(1, 2).reshape(B, S, -1)
        return self.W_O(output) # (B, S, d_model)
```

---

## Task 05: Layer Normalization

LayerNorm(x) = gamma * (x - mu) / sqrt(var + eps) + beta — normalize over last dim

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model)) # learnable scale
        self.beta = nn.Parameter(torch.zeros(d_model)) # learnable shift

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True) # per-token mean
        var = torch.var(x, dim=-1, keepdim=True, correction=0) # population variance (no Bessel)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta
```

---

## Task 06: Tiny Transformer Forward Pass

Decoder-only: Embed + PE → N x [LN → MHA(causal) + residual → LN → FFN + residual] → LN → Linear

```python
import torch.nn.functional as F

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.d_model, self.max_seq_len = d_model, max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pe", positional_encoding(max_seq_len, d_model)) # fixed sinusoidal PE
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': LayerNorm(d_model), 'mha': MultiHeadAttention(d_model, num_heads),
                'ln2': LayerNorm(d_model),
                'ff1': nn.Linear(d_model, d_ff), 'ff2': nn.Linear(d_ff, d_model),
            }) for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        B, S = token_ids.shape
        x = self.embedding(token_ids) + self.pe[:, :S, :] # (B, S, d_model) — embed + positional
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=token_ids.device), diagonal=1).unsqueeze(0) # causal: True=mask future
        for layer in self.layers:
            x = x + layer['mha'](layer['ln1'](x), mask=mask) # pre-norm attention + residual
            x = x + layer['ff2'](F.relu(layer['ff1'](layer['ln2'](x)))) # pre-norm FFN + residual
        return self.lm_head(self.final_norm(x)) # (B, S, vocab_size)
```
