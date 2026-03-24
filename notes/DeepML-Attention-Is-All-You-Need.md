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

**Theory**: Transformers have no recurrence or convolution, so they need explicit position information. Sinusoidal PE uses different frequencies across dimensions — low dims oscillate fast (local position), high dims oscillate slowly (global position). The key property: PE(pos+k) can be represented as a linear function of PE(pos), so the model can learn to attend to relative positions.

**Interview Q&A**:
- **Why sin/cos instead of learned embeddings?** Sinusoidal generalizes to longer sequences than seen during training (extrapolation). Learned PE must be trained for each position. In practice, both work similarly for fixed-length tasks.
- **Why 10000 as the base?** It sets the wavelength range from 2π to 10000·2π. Low dims change every token; high dims change every ~10000 tokens. Any large base works — 10000 was empirically chosen.
- **What are alternatives?** RoPE (rotary) encodes relative position by rotating Q/K vectors — used in LLaMA/GPT-NeoX. ALiBi adds position-dependent bias to attention scores (no PE parameters at all).
- **Why `correction=0` matters for variance?** Population variance (divides by N) matches the LayerNorm formula. Bessel correction (N-1) is for unbiased sample estimation, not wanted here.

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

**Theory**: Attention computes a weighted average of values, where weights come from query-key similarity. The scaling factor 1/sqrt(d_k) prevents softmax saturation: without it, dot products grow as O(d_k) for random unit-variance vectors, pushing softmax into near-one-hot regions with vanishing gradients.

**Interview Q&A**:
- **Why divide by sqrt(d_k)?** If Q, K entries are iid with mean 0, variance 1, then QK^T has variance d_k. Dividing by sqrt(d_k) restores unit variance, keeping softmax in a region with healthy gradients.
- **Why softmax and not other normalizations?** Softmax gives a valid probability distribution (non-negative, sums to 1). It's also differentiable and amplifies the largest scores (winner-take-more behavior). Alternatives: sigmoid attention (used in some linear attention variants), sparsemax.
- **What's the complexity?** O(n² · d) for sequence length n. This is the main bottleneck — FlashAttention, linear attention, and sliding window attention are all strategies to reduce this.
- **Causal mask vs padding mask?** Causal mask prevents attending to future tokens (autoregressive). Padding mask prevents attending to [PAD] tokens in variable-length batches. Both are applied the same way (set scores to -inf before softmax).
- **What is FlashAttention?** It computes exact attention but tiles the computation to stay in GPU SRAM, avoiding O(n²) HBM reads. Fuses softmax, masking, and matmul into a single kernel. 2-4x faster, exact same math.

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

**Theory**: The projections let the model learn *what to query for*, *what to advertise as key*, and *what information to return as value* — all from the same input. Without projections, attention would only measure raw similarity between input vectors.

**Interview Q&A**:
- **Why separate Q, K, V projections?** They decouple "what I'm looking for" (Q) from "what I contain" (K) from "what I provide when matched" (V). A token's key and value can encode different things.
- **nn.Parameter vs nn.Linear?** `nn.Parameter(randn)` creates a raw weight matrix (no bias, no init scheme). `nn.Linear` adds bias by default and initializes with Kaiming uniform. For interview problems, raw parameters show understanding; in production, use `nn.Linear`.
- **Why is W_O needed?** It mixes information after attention. Without it, the output is stuck in the d_k subspace and can't interact with the full d_model representation.

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

**Theory**: A single attention head can only attend to one pattern at a time. Multi-head attention runs h parallel attention operations in d_k-dimensional subspaces, then concatenates and projects. This lets different heads learn different relationships — some may attend to syntax, others to semantics, positional patterns, etc. Total compute is the same as a single head with full d_model (h heads × d_k cost = d_model cost).

**Interview Q&A**:
- **Why reshape to (B*h, S, d_k) instead of keeping 4D?** Merging batch and heads into one dimension lets us reuse the same 3D `scaled_dot_product_attention` function. Alternatively, keep (B, h, S, d_k) and use `einsum` or 4D matmul.
- **What do different heads learn?** In practice: some heads attend to adjacent tokens (local), some attend to [SEP]/punctuation (delimiter), some attend to the first token (global), some form syntactic dependency patterns. This emerges without supervision.
- **How many heads is optimal?** The original Transformer uses h=8 with d_model=512 (d_k=64). GPT-3 uses h=96 with d_model=12288. Generally, more heads = more diverse patterns, but diminishing returns. Each head needs enough d_k to represent meaningful patterns (d_k < 32 tends to hurt).
- **MHA vs MQA vs GQA?** Multi-Query Attention (MQA): all heads share one K, V projection — saves KV-cache memory at inference. Grouped-Query Attention (GQA, used in LLaMA 2): groups of heads share K, V — compromise between MHA quality and MQA efficiency.

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

**Theory**: Layer norm normalizes each token's feature vector independently (across the d_model dimension). This stabilizes activations and enables residual learning. Unlike batch norm, it doesn't depend on batch statistics — works identically in training and inference, and with batch size 1.

**Interview Q&A**:
- **LayerNorm vs BatchNorm?** BatchNorm normalizes across the batch dimension (per-feature statistics). LayerNorm normalizes across the feature dimension (per-token statistics). For transformers, LayerNorm is preferred because: (1) sequence lengths vary, (2) causal models can't use future statistics, (3) no running mean/var needed at inference.
- **Pre-norm vs post-norm?** Original Transformer: post-norm (LN after residual). GPT-2+: pre-norm (LN before sub-layer). Pre-norm is more stable for deep models — gradients flow directly through residual path without passing through LN. Post-norm can achieve slightly higher quality but is harder to train.
- **RMSNorm?** Used in LLaMA. Drops the mean subtraction: RMSNorm(x) = x / RMS(x) * gamma. ~15% faster since you skip computing the mean. Empirically equivalent quality.
- **Why `correction=0`?** This gives population variance (divide by N). `correction=1` (default) gives sample/Bessel variance (divide by N-1). LayerNorm uses population variance — we're normalizing a fixed vector, not estimating from samples.

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

**Theory**: This is a GPT-style decoder-only transformer. The causal mask (upper-triangular) ensures token i can only attend to tokens 0..i, enabling autoregressive generation. The FFN (expand → ReLU → project) adds non-linearity — attention alone is a linear operation (weighted sum). Residual connections ensure gradients flow directly through the network. The final LN + linear head maps to vocabulary logits for next-token prediction.

**Interview Q&A**:
- **Why d_ff = 4 * d_model?** Empirical convention from the original paper. The FFN expands to 4x, applies non-linearity, then projects back. This is where most of the model's parameters live (~2/3 of total params). SwiGLU (used in LLaMA) uses d_ff = 8/3 * d_model with a gating mechanism.
- **Encoder vs decoder vs encoder-decoder?** Encoder-only (BERT): bidirectional attention, good for classification/NLU. Decoder-only (GPT): causal mask, autoregressive generation. Encoder-decoder (T5, original Transformer): encoder sees full input, decoder generates output autoregressively with cross-attention to encoder. Modern trend is decoder-only for everything.
- **How is this trained?** Next-token prediction (causal LM): given tokens [0..t], predict token t+1. Loss = cross-entropy between logits and shifted targets. Teacher forcing: always use ground truth tokens as input during training.
- **Weight tying?** Common optimization: share weights between embedding layer and lm_head (both are vocab_size × d_model matrices). Reduces parameters by ~30% with no quality loss.
- **register_buffer vs nn.Parameter?** `register_buffer` saves the tensor with the model state but doesn't compute gradients. Perfect for fixed PE, masks, or constants. `nn.Parameter` participates in gradient computation.
- **KV-cache for inference?** During autoregressive generation, we only need the new token's Q but all previous K, V. KV-cache stores previously computed K, V to avoid recomputation — reduces inference from O(n²) to O(n) per step.

---

## Key Interview Concepts

**Transformer complexity**: Self-attention is O(n² · d), FFN is O(n · d²). For short sequences, FFN dominates; for long sequences, attention dominates.

**Parameter count** (per layer): MHA = 4 · d_model² (W_Q, W_K, W_V, W_O). FFN = 2 · d_model · d_ff = 8 · d_model². Total per layer ≈ 12 · d_model². For GPT-3 (96 layers, d=12288): ~175B parameters.

**Training tricks**: Learning rate warmup (linear warmup → cosine/inverse sqrt decay), gradient clipping, mixed precision (fp16/bf16 compute, fp32 master weights), dropout on attention weights and residual connections.

**Modern improvements over original Transformer**:
- RoPE instead of sinusoidal PE (better extrapolation, relative position)
- SwiGLU instead of ReLU FFN (better quality, ~same compute)
- RMSNorm instead of LayerNorm (faster, same quality)
- GQA instead of MHA (saves KV-cache memory)
- FlashAttention (faster, memory-efficient, exact)
- Pre-norm instead of post-norm (training stability)
