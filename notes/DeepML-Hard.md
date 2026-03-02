# Deep-ML Hard Problems: Advanced Concepts Guide

A comprehensive guide to advanced deep learning concepts, PyTorch patterns, and challenging problems.

## Table of Contents

- [Backpropagation Through Time (BPTT)](#1-backpropagation-through-time-bptt)
- [Attention Mechanisms](#2-attention-mechanisms)
- [Variational Autoencoders (VAE)](#3-variational-autoencoders-vae)

---

# Advanced Deep Learning Concepts

## 1. Backpropagation Through Time (BPTT)

### Problem Statement

Implement backpropagation for recurrent neural networks that process sequential data.

### Mathematical Theory

**Forward Pass:**
```
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
y_t = W_hy @ h_t + b_y
```

**Loss:**
```
L = sum_{t=1}^{T} loss(y_t, y_hat_t)
```

**Backward Pass - Gradient through time:**
```
dL/dW_hh = sum_{t=1}^{T} sum_{k=1}^{t} dL_t/dh_t * dh_t/dh_k * dh_k/dW_hh
```

**Key Challenges:**
1. Vanishing gradients: dh_t/dh_k -> 0 as t - k grows
2. Exploding gradients: Multiplicative chain can grow exponentially
3. Memory: Full BPTT requires storing all hidden states

### Implementation

```python
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    """Vanilla RNN with manual BPTT for gradient flow visualization."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.b_y = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, h_prev=None):
        batch_size, seq_len, _ = x.shape
        h = h_prev if h_prev is not None else torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs, hidden_states = [], [h]

        for t in range(seq_len):
            h = torch.tanh(h @ self.W_hh + x[:, t, :] @ self.W_xh + self.b_h)
            y = h @ self.W_hy + self.b_y
            outputs.append(y)
            hidden_states.append(h)

        return torch.stack(outputs, dim=1), hidden_states
```

**Truncated BPTT** - Backpropagate only k timesteps to reduce memory:

```python
class TruncatedBPTT:
    """Gradient flows through only truncation_length steps."""
    def __init__(self, model, truncation_length=20):
        self.model = model
        self.truncation_length = truncation_length

    def train_step(self, sequence, targets, optimizer):
        batch_size, total_len, _ = sequence.shape
        h = None
        total_loss = 0
        num_chunks = (total_len + self.truncation_length - 1) // self.truncation_length

        for i in range(num_chunks):
            start_idx = i * self.truncation_length
            end_idx = min((i + 1) * self.truncation_length, total_len)
            seq_chunk = sequence[:, start_idx:end_idx, :]
            target_chunk = targets[:, start_idx:end_idx, :]

            # Detach hidden state to truncate gradient flow
            if h is not None:
                h = h.detach()

            outputs, hidden_states = self.model(seq_chunk, h)
            h = hidden_states[-1]

            loss = ((outputs - target_chunk) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / num_chunks
```

**Gradient Clipping** - Prevent exploding gradients:

```python
def train_rnn_with_clipping(model, sequence, targets, optimizer, clip_value=1.0):
    outputs, _ = model(sequence)
    loss = ((outputs - targets) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
    return loss.item()
```

**LSTM** - Solves vanishing gradients with multiplicative gates:

```python
class LSTMCell(nn.Module):
    """LSTM cell: input gate, forget gate, cell gate, output gate control info flow."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # All gates computed efficiently in one pass
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.W(combined)
        i, f, g, o = gates.chunk(4, dim=1)

        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c_prev + i * g  # Cell state preserves gradient flow
        h = o * torch.tanh(c)
        return h, (h, c)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):
        lstm_out, state = self.lstm(x, state)
        return self.fc(lstm_out), state
```

### PyTorch Best Practices

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 1. PackedSequence for variable-length sequences
def process_variable_length(model, sequences, lengths):
    lengths, sort_idx = lengths.sort(descending=True)
    sequences = sequences[sort_idx]
    packed = pack_padded_sequence(sequences, lengths, batch_first=True)
    output_packed, state = model.lstm(packed)
    output, _ = pad_packed_sequence(output_packed, batch_first=True)
    _, unsort_idx = sort_idx.sort()
    return output[unsort_idx]

# 2. Bidirectional LSTM for full context
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

# 3. Teacher Forcing for seq2seq training
def train_seq2seq_teacher_forcing(encoder, decoder, src, tgt, teacher_forcing_ratio=0.5):
    """Use ground truth target with probability teacher_forcing_ratio."""
    batch_size, tgt_len, _ = tgt.shape
    encoder_outputs, encoder_state = encoder(src)
    decoder_state = encoder_state
    decoder_input = tgt[:, 0:1, :]
    outputs = []

    for t in range(1, tgt_len):
        output, decoder_state = decoder(decoder_input, decoder_state)
        outputs.append(output)
        decoder_input = tgt[:, t:t+1, :] if torch.rand(1).item() < teacher_forcing_ratio else output

    return torch.cat(outputs, dim=1)
```

**Best Practices:**
- Use truncated BPTT for sequences longer than 100 steps to manage memory and training time
- Always apply gradient clipping with RNNs (max_norm=1.0) to prevent exploding gradients
- Prefer LSTM/GRU over vanilla RNN to avoid vanishing gradients in long sequences
- Use PackedSequence for variable-length inputs to avoid unnecessary computation on padding

**Common Applications:**
- Language modeling, machine translation, speech recognition
- Time series forecasting, stock price prediction
- Named entity recognition, sentiment analysis, text classification

---

## 2. Attention Mechanisms

### Problem Statement

Implement attention to allow models to focus on relevant input parts.

### Mathematical Theory

**Scaled Dot-Product Attention:**

Given Query Q, Key K, Value V:
```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

The scaling by sqrt(d_k) prevents gradient vanishing when d_k is large.

**Multi-Head Attention:** Multiple representation subspaces in parallel:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
```

where `head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)`

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """Core attention mechanism in Transformers."""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, value), attn_weights


class MultiHeadAttention(nn.Module):
    """Projects to multiple subspaces, then concatenates results."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = self.attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        output = attn_output.view(batch_size, -1, self.d_model)
        return self.W_o(output), attn_weights


class SelfAttention(nn.Module):
    """Query, Key, Value all from same source."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, x, mask=None):
        return self.mha(x, x, x, mask)


class CrossAttention(nn.Module):
    """Query from decoder, Key/Value from encoder."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, query, context, mask=None):
        return self.mha(query, context, context, mask)


class TransformerBlock(nn.Module):
    """Self-attention + LayerNorm + FFN with residual connections."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        return x
```

**Efficient Attention Variants:**

```python
class LinearAttention(nn.Module):
    """O(n) complexity via kernel trick: φ(Q) @ (φ(K).T @ V) instead of softmax(Q @ K.T) @ V."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Feature map: φ(x) = elu(x) + 1
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # Compute K^T V first to avoid O(n²) attention matrix
        KV = torch.matmul(K.transpose(-2, -1), V)
        output = torch.matmul(Q, KV)

        # Normalize by sum of keys
        normalizer = torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1))
        output = output / (normalizer + 1e-6)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)


class LocalAttention(nn.Module):
    """O(n*w) complexity for long sequences via local window."""
    def __init__(self, d_model, n_heads, window_size=128):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.window_size = window_size

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        w = self.window_size

        # Create local attention mask
        mask = torch.ones(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            start = max(0, i - w // 2)
            end = min(seq_len, i + w // 2 + 1)
            mask[i, start:end] = 0

        mask = mask.unsqueeze(0).unsqueeze(0)
        output, _ = self.mha(x, x, x, mask)
        return output


def create_sparse_attention_mask(seq_len, stride=32):
    """Combines local (causal) and strided attention patterns."""
    local = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        local[i, max(0, i-stride):i+1] = 1

    strided = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        strided[i, ::stride] = 1

    return (local + strided).clamp(0, 1)
```

**Best Practices:**
- Use multi-head attention (8-16 heads) to capture different representation subspaces
- Apply dropout after attention weights to prevent overfitting
- Use efficient variants (LinearAttention, LocalAttention) for sequences longer than 1024 tokens
- Mask future tokens in decoder self-attention (causal masking) for autoregressive generation

**Common Applications:**
- Machine translation, text summarization, question answering
- Image classification (Vision Transformer), object detection
- Speech recognition, multimodal learning, recommendation systems

---

## 3. Variational Autoencoders (VAE)

### Problem Statement

Implement a generative model that learns a probabilistic latent representation.

### Mathematical Theory

**Encoder (recognition network):**
```
q(z|x) = Normal(mu(x), sigma²(x))
```

**Decoder (generative network):**
```
p(x|z) = Normal(mu(z), sigma²(z))
```

**ELBO (Evidence Lower Bound) - what we maximize:**
```
L(theta, phi; x) = E_{q(z|x)}[log p(x|z)] - KL(q(z|x) || N(0, I))
```

Reconstruction loss drives accurate decoding; KL divergence regularizes the latent distribution to N(0, I).

**Reparameterization Trick** - Enable gradients through sampling:
```
z = mu + sigma * epsilon,  where epsilon ~ N(0, I)
```

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Encodes to latent distribution, decodes back to data space."""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """z = mu + sigma * epsilon for gradient flow through sampling."""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, num_samples, device='cpu'):
        """Generate from prior: z ~ N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """Reconstruction loss + beta * KL divergence."""
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + log(sigma²) - mu² - sigma²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae_epoch(model, dataloader, optimizer, device, beta=1.0):
    """Train VAE for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.view(data.size(0), -1).to(device)
        x_recon, mu, logvar = model(data)
        loss, _, _ = vae_loss(x_recon, data, mu, logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader.dataset)


class ConditionalVAE(nn.Module):
    """VAE conditioned on class labels for controlled generation."""
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        label_embed = self.label_embedding(labels)
        x_cond = torch.cat([x, label_embed], dim=1)
        h = self.encoder(x_cond)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, labels):
        label_embed = self.label_embedding(labels)
        z_cond = torch.cat([z, label_embed], dim=1)
        return self.decoder(z_cond)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    def sample(self, num_samples, label, device='cpu'):
        """Generate samples of a specific class."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        labels = torch.full((num_samples,), label, dtype=torch.long, device=device)
        return self.decode(z, labels)
```

**Best Practices:**
- Use beta-VAE (beta > 1) to strengthen KL regularization and disentangle latent factors
- Apply log-scale outputs for variance to ensure numerical stability and positivity
- Start training with low beta and gradually increase to balance reconstruction vs. regularization
- Monitor both reconstruction and KL loss separately to detect posterior collapse

**Common Applications:**
- Image generation, data augmentation, anomaly detection
- Disentangled representation learning, semi-supervised learning
- Dimensionality reduction, data compression, missing data imputation

---

## Key Takeaways

**BPTT:** Truncate gradient flow for long sequences; use LSTM gates to preserve information through time.

**Attention:** Query-Key-Value mechanism allows selective focus; multi-head captures diverse patterns; efficient variants (Linear, Local, Sparse) handle long sequences.

**VAE:** Reparameterization trick enables gradient flow through sampling; KL term regularizes latent space; ELBO balances reconstruction and distribution matching.

### Resources

- **Deep-ML**: [www.deep-ml.com](https://www.deep-ml.com)
- **Papers**: Transformer (Vaswani et al.), VAE (Kingma & Welling), Attention Is All You Need
- **Books**: "Dive into Deep Learning" (d2l.ai), "Understanding Deep Learning" by Simon J.D. Prince
