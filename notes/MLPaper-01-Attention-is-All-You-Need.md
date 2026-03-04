# Attention is All You Need (Transformers)

## Paper Overview

| | |
|---|---|
| **Authors** | Vaswani et al. (Google Brain/Google Research) |
| **Published** | 2017 (NeurIPS) |
| **Paper Link** | https://arxiv.org/abs/1706.03762 |

---

## Detailed Description

"Attention is All You Need" introduced the **Transformer** architecture, a groundbreaking neural network model that relies entirely on attention mechanisms, dispensing with recurrence and convolutions altogether. The paper revolutionized sequence-to-sequence modeling and became the foundation for modern NLP models like BERT, GPT, and T5.

### How It Works (Intuitive Summary)

Traditional sequence models (like RNNs) process words one at a time, left to right, with the hidden state carrying all prior context. This is slow and makes capturing distant dependencies difficult. The Transformer processes all words simultaneously and uses attention to directly compare any word with any other word, regardless of distance. Instead of a single hidden state bottleneck, each word can "look at" all other words and gather relevant information in parallel. This parallelization is what makes Transformers so fast to train and so effective at capturing long-range dependencies.

### Architecture Overview

The Transformer consists of several key components working together:

**Self-Attention Mechanism** is the core innovation. When processing each word/token, the model learns to weigh the importance of all other words in the sequence. This is computed using three components: *Queries* (Q), *Keys* (K), and *Values* (V). Q represents the current token's question "what should I attend to?", K represents what each token *offers* (can be matched against), and V contains the actual information to extract.

**Scaled Dot-Product Attention** implements this mathematically:

```math
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

The key scaling by `sqrt(d_k)` prevents attention scores from becoming too large.

**Multi-Head Attention** runs multiple attention mechanisms in parallel (typically 8-12 heads). Each head focuses on different aspects and representation subspaces simultaneously. This is more powerful than a single attention head and allows the model to capture different types of relationships.

**Positional Encoding** provides sequence order information. Since attention is order-agnostic, sinusoidal positional encodings are added to embeddings to tell the model where each token appears in the sequence. These encodings use sine and cosine functions that generalize to longer sequences than seen during training.

**Encoder-Decoder Architecture:** The encoder processes the input sequence and creates contextualized representations. The decoder generates the output sequence while attending to both its own previous outputs and the encoder's representations.

**Position-wise Feed-Forward Networks** apply two linear transformations with ReLU activation to each position separately and identically. These add non-linearity and model capacity.

**Layer Normalization and Residual Connections** stabilize training by normalizing layer inputs and allowing gradients to flow directly through skip connections, enabling much deeper networks.

---

## Pain Point Addressed

### Problems with Previous Approaches

1. **Sequential Processing in RNNs:**
   - RNNs (including LSTMs and GRUs) process sequences sequentially, making parallelization difficult
   - Training is slow due to sequential dependency
   - Gradient vanishing/exploding issues over long sequences

2. **Limited Long-Range Dependencies:**
   - RNNs struggle to capture dependencies between distant tokens
   - Information can be lost as sequences get longer
   - Computational cost grows with sequence length

3. **Memory Constraints:**
   - Hidden states need to compress all previous information
   - Bottleneck for long sequences

4. **Inefficient Use of Hardware:**
   - Sequential nature prevents effective use of modern GPU parallelization
   - Training on long sequences is time-consuming

---

## Novelty of the Paper

### Key Innovations

1. **Fully Attention-Based Architecture:**
   - First model to rely entirely on attention mechanisms
   - Eliminated recurrence and convolutions completely
   - Revolutionary shift from sequential to parallel processing

2. **Multi-Head Self-Attention:**
   - Multiple attention heads capture different types of relationships simultaneously
   - Each head learns different aspects of the input
   - Allows model to focus on different positions

3. **Positional Encoding:**
   - Sinusoidal functions to inject position information
   - Allows model to learn relative positions
   - Works for sequences longer than those seen during training

4. **Scalability and Parallelization:**
   - All tokens processed simultaneously rather than sequentially
   - Dramatically faster training on modern hardware
   - Enables training on much larger datasets

5. **Constant Path Length:**
   - Direct connections between any two positions in the sequence
   - Maximum path length is O(1) vs O(n) for RNNs
   - Better gradient flow for long-range dependencies

6. **Superior Performance:**
   - Achieved state-of-the-art results on machine translation tasks
   - Better BLEU scores with significantly less training time
   - More parameter-efficient than previous approaches

---

## Impact

> The Transformer architecture became the foundation for virtually all modern deep learning, creating the era of large language models.

- **BERT** (Bidirectional Encoder Representations from Transformers)
- **GPT** series (Generative Pre-trained Transformers)
- **T5** (Text-to-Text Transfer Transformer)
- **Vision Transformers** (ViT)
- **DALL-E, Stable Diffusion** and other multimodal models

---

## Implementation

Below is a complete PyTorch implementation of the Transformer architecture from the repository:

```python
import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    """Get the best available device (CUDA if available, CPU otherwise)."""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        q, k, v: Query, Key, Value tensors of shape (batch, heads, seq_len, head_dim)
        mask: Optional mask tensor to prevent attending to certain positions

    Returns:
        values: Weighted values from attention
        attention: Attention weights
    """
    d_k = q.size()[-1]
    # Scale dot product by sqrt(d_k) to prevent gradient vanishing
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        # Permute to apply mask correctly across batch and head dimensions
        # This is necessary because mask is (batch, 1, seq_len, seq_len) and needs
        # to broadcast properly with scaled which is (batch, heads, seq_len, seq_len)
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)

    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    """
    Generates positional encodings using sinusoidal functions.

    Design Note: forward() takes no arguments and returns a static positional encoding
    matrix for all sequence positions up to max_sequence_length. The actual sequence
    embeddings are added to these encodings in SentenceEmbedding.forward().
    """
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        """
        Generate sinusoidal positional encodings.
        Uses sine for even dimensions and cosine for odd dimensions.

        Returns:
            PE: Positional encoding matrix of shape (max_sequence_length, d_model)
        """
        # Create position indices and dimension indices
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                    .reshape(self.max_sequence_length, 1))

        # Compute sin/cos encodings for even/odd dimensions
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)

        # Interleave sin and cos to create final encoding
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    """
    Converts sentences to embeddings with positional encoding.

    This module handles tokenization, embedding lookup, and adds positional
    information to the word embeddings before passing to the transformer.
    """
    def __init__(self, max_sequence_length, d_model, language_to_index,
                 START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, start_token, end_token):
        """Convert a batch of sentences to token indices with padding."""
        def tokenize(sentence, start_token, end_token):
            # Convert each character/token to its index
            sentence_word_indices = [self.language_to_index[token]
                                     for token in list(sentence)]
            if start_token:
                sentence_word_indices.insert(0,
                    self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indices.append(
                    self.language_to_index[self.END_TOKEN])
            # Pad sequence to fixed length
            for _ in range(len(sentence_word_indices), self.max_sequence_length):
                sentence_word_indices.append(
                    self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indices)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())

    def forward(self, x, start_token, end_token):
        """
        Embed sentences and add positional encoding.

        Args:
            x: Batch of sentences (list of strings)
            start_token: Whether to add START token
            end_token: Whether to add END token

        Returns:
            Embedded and position-encoded sentences
        """
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Splits the model dimension into multiple heads, allowing the model to attend
    to different representation subspaces simultaneously. Each head performs
    scaled dot-product attention independently, then results are concatenated.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Project input to Q, K, V (all at once for efficiency)
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        # Project concatenated heads back to d_model
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        """
        Apply multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            mask: Optional attention mask

        Returns:
            Output of shape (batch_size, sequence_length, d_model)
        """
        batch_size, sequence_length, d_model = x.size()

        # Project to Q, K, V
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads,
                         3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch, heads, seq_len, 3*head_dim)

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask)

        # Concatenate heads and project back to d_model
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length,
                                                     self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out

class LayerNormalization(nn.Module):
    """
    Layer normalization with learnable scale and shift parameters.

    Normalizes the input across feature dimensions (not batch), making training
    more stable and allowing the model to learn at higher learning rates.
    """
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        # Learnable scale (gamma) and shift (beta) parameters
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        """Normalize inputs and apply learnable scale and shift."""
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applies two linear transformations with ReLU activation to each position
    independently and identically. This adds non-linearity and increases model
    capacity without increasing attention complexity.
    """
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Expand to hidden dimension, then project back to d_model
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """Apply feed-forward transformation: linear -> ReLU -> dropout -> linear."""
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """
    Single encoder layer consisting of multi-head attention and feed-forward networks.

    Uses residual connections and layer normalization to stabilize training and
    enable deeper architectures.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden,
                                          drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        """Apply attention -> norm -> FFN with residual connections."""
        # Self-attention block with residual connection
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)

        # Feed-forward block with residual connection
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class SequentialEncoder(nn.Sequential):
    """Sequential container for encoder layers that handles mask passing."""
    def forward(self, *inputs):
        """Apply each encoder layer in sequence."""
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    """
    Full transformer encoder.

    Stacks multiple encoder layers on top of the sentence embedding layer.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, language_to_index, START_TOKEN,
                 END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model,
            language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        # Stack num_layers encoder layers
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden,
            num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        """Embed input and apply encoder layers."""
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention for decoder-encoder attention.

    Queries come from the decoder while keys and values come from the encoder,
    allowing the decoder to attend to encoder outputs.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Project encoder outputs to K, V
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        # Project decoder state to Q
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask):
        """
        Apply cross-attention.

        Args:
            x: Encoder output (source for K, V)
            y: Decoder state (source for Q)
            mask: Cross-attention mask
        """
        batch_size, sequence_length, d_model = x.size()

        # Project to K, V from encoder output
        kv = self.kv_layer(x)
        # Project to Q from decoder state
        q = self.q_layer(y)

        # Reshape for multi-head
        kv = kv.reshape(batch_size, sequence_length, self.num_heads,
                       2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)

        # Split K and V
        k, v = kv.chunk(2, dim=-1)

        # Apply scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask)

        # Concatenate heads and project
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length,
                                                     d_model)
        out = self.linear_layer(values)
        return out

class DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention, cross-attention, and feed-forward.

    The decoder attends to both its own previous outputs and the encoder outputs,
    with residual connections and layer normalization throughout.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        # Self-attention on decoder outputs
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        # Cross-attention to encoder outputs
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model,
                                                                 num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        # Feed-forward network
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden,
                                          drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        """Apply self-attention -> cross-attention -> FFN with residual connections."""
        # Self-attention block
        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        # Cross-attention block (attend to encoder outputs)
        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        # Feed-forward block
        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y

class SequentialDecoder(nn.Sequential):
    """Sequential container for decoder layers that handles mask passing."""
    def forward(self, *inputs):
        """Apply each decoder layer in sequence."""
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    """
    Full transformer decoder.

    Stacks multiple decoder layers on top of the sentence embedding layer.
    The decoder generates output sequences while attending to encoder outputs.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, language_to_index, START_TOKEN,
                 END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model,
            language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        # Stack num_layers decoder layers
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden,
            num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask,
               start_token, end_token):
        """
        Decode with cross-attention to encoder outputs.

        Args:
            x: Encoder output
            y: Decoder input sequence
            self_attention_mask, cross_attention_mask: Attention masks
            start_token, end_token: Tokenization flags
        """
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer(nn.Module):
    """
    Complete transformer encoder-decoder model.

    Combines encoder (processes input) and decoder (generates output) with a
    final linear projection to vocabulary size.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, kn_vocab_size, english_to_index,
                 kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob,
            num_layers, max_sequence_length, english_to_index, START_TOKEN,
            END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob,
            num_layers, max_sequence_length, kannada_to_index, START_TOKEN,
            END_TOKEN, PADDING_TOKEN)
        # Project decoder output to target vocabulary logits
        self.linear = nn.Linear(d_model, kn_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() \
                     else torch.device('cpu')

    def forward(self, x, y, encoder_self_attention_mask=None,
               decoder_self_attention_mask=None,
               decoder_cross_attention_mask=None, enc_start_token=False,
               enc_end_token=False, dec_start_token=False, dec_end_token=False):
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input sequence (source language)
            y: Target sequence (target language)
            *_mask: Optional attention masks for preventing attention to certain positions
            enc_start_token, enc_end_token: Add special tokens to encoder input
            dec_start_token, dec_end_token: Add special tokens to decoder input

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
        """
        # Encode input
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token,
                        end_token=enc_end_token)
        # Decode with cross-attention to encoder
        out = self.decoder(x, y, decoder_self_attention_mask,
                          decoder_cross_attention_mask, start_token=dec_start_token,
                          end_token=dec_end_token)
        # Project to vocabulary logits
        out = self.linear(out)
        return out
```

---

## Key Takeaways

> 1. The Transformer eliminates sequential processing, enabling massive parallelization
> 2. Self-attention mechanisms allow the model to capture long-range dependencies efficiently
> 3. The architecture is highly scalable and forms the basis of modern LLMs
> 4. Multi-head attention enables the model to focus on different aspects simultaneously
> 5. The paper fundamentally changed the landscape of NLP and deep learning

---

## Repository Reference

Implementation source: https://github.com/atullchaurasia/problem-solving/tree/main/Transformers
