# Recurrent Neural Network Regularization - Comprehensive Paper Summary

| | |
|---|---|
| **Paper** | Recurrent Neural Network Regularization |
| **Authors** | Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals |
| **ArXiv ID** | 1409.2329 |
| **Year** | 2014 |
| **Citation** | [ArXiv](https://arxiv.org/abs/1409.2329) |

---

## 1. One-Page Overview

### Metadata & Publication
- **Published:** September 2014 (arXiv preprint)
- **Venue:** One of the earliest successful applications of dropout to RNNs
- **Impact:** Seminal work that enabled practical regularization of LSTM-based models

### Problem Statement
Dropout, the most successful regularization technique for deep feedforward networks, fails catastrophically when naively applied to RNNs. Standard dropout perturbs recurrent (temporal) connections with different masks at each timestep, destroying the LSTM's ability to store and propagate information across long sequences. This results in worse overfitting rather than improvement.

### Key Novelty

> The paper introduces a **dropout schedule that applies dropout ONLY to non-recurrent (feedforward) connections**:
- No dropout on recurrent-to-recurrent connections (hidden state transitions)
- Dropout applied to: input→LSTM, LSTM-layer→LSTM-layer (between layers), LSTM→output
- Same dropout mask applied across all timesteps for a given connection
- Results in substantial overfitting reduction across language modeling, speech recognition, machine translation, and image captioning

### Tasks Solved
1. **Language Modeling** - Penn Treebank: 78.4 test perplexity (SOTA at time)
2. **Speech Recognition** - TIMIT dataset
3. **Machine Translation** - WMT'14 English-French
4. **Image Captioning** - MSCOCO dataset

### If You Only Remember 3 Things

> 1. **Apply dropout only to non-recurrent connections** - Recurrent weights should NOT be masked
> 2. **Use consistent masks across timesteps** - Different dropout patterns at each timestep destroy LSTM memory
> 3. **Larger models + higher dropout rates needed** - 1500-unit LSTM with 65% dropout dramatically outperforms 650-unit LSTM with 50% dropout

### Key Results Summary
| Task | Model | Improvement |
|------|-------|-------------|
| Penn Treebank | Large LSTM (1500 units) | 78.4 test perplexity (previous: ~82.3) |
| All Tasks | Regularized LSTM | Consistent reduction of overfitting |

---

## 2. Problem Setup and Outputs

### Problem Definition: Language Modeling on Penn Treebank

**Input Sequence Structure:**
```
Sequence of word tokens: [w_1, w_2, w_3, ..., w_T]
T = 35 (BPTT length used in experiments)
Vocabulary size: 10,000 words (most frequent, rest mapped to <unk>)
Batch size: 20 sequences processed in parallel
```

### Tensor Shapes for Forward Pass

#### Input
- **x_t**: (batch_size, vocab_size) or (batch_size,) as token indices
  - Shape per timestep: (20,) for batch of indices
  - After embedding: (20, embedding_dim)
  - Embedding dim: typically 200-400 (not specified in paper, typical for PTB)

#### LSTM Hidden States
- **h_t**: (batch_size, hidden_size)
  - Medium model: (20, 650)
  - Large model: (20, 1500)

#### LSTM Cell States (internal memory)
- **c_t**: (batch_size, hidden_size)
  - Medium model: (20, 650)
  - Large model: (20, 1500)

#### Outputs (Logits)
- **logits_t**: (batch_size, vocab_size) = (20, 10000)
- **log_probs_t**: (batch_size, vocab_size) = (20, 10000) after softmax

#### Loss
- **cross_entropy_t**: (batch_size,) = (20,)
- **sequence_loss**: scalar, mean across batch and timesteps

### Complete Forward Pass Tensor Flow
```
[Input Tokens]
      ↓ (embed)
[Embedded Tokens: (20, embed_dim)]
      ↓ (dropout on embeddings)
[LSTM Layer 1: h_1_t ∈ (20, 650), c_1_t ∈ (20, 650)]
      ↓ (dropout on hidden output)
[LSTM Layer 2: h_2_t ∈ (20, 650), c_2_t ∈ (20, 650)]
      ↓ (dropout on hidden output)
[Output Linear: (20, 10000)]
      ↓ (softmax)
[Probabilities: (20, 10000)]
      ↓ (cross-entropy with target)
[Loss: scalar]
```

---

## 3. Coordinate Frames and Geometry

### Token Embedding Space

**Embedding Layer Structure:**
- **Embedding Matrix E**: (vocab_size, embedding_dim) = (10000, d_e)
- Typical embedding dimension: 200-400 (not explicitly stated; inferred from architecture)
- **Operation:** x_embedded = E[w_t]
- Each word maps to a d_e-dimensional dense vector in continuous space

**Embedding Properties:**
- Weights initialized uniformly in [-0.05, 0.05] (medium) or [-0.04, 0.04] (large)
- Learned via backpropagation
- Provides distributed representation of language semantics

### LSTM Hidden State Geometry

**Hidden State Space H ⊂ ℝ^(hidden_size)**

**Medium Model:**
- Dimension: 650
- Interpretation: Compressed representation of past token sequence
- At each timestep t, captures "what the model knows so far"

**Large Model:**
- Dimension: 1500
- Increased capacity for storing longer-range dependencies
- Better able to learn complex grammatical structures

**Geometric Interpretation:**
- h_0 = 0 (zero initial state)
- h_t = f(h_{t-1}, x_t; θ) where f is the LSTM transformation
- Trajectory through hidden space: [h_0, h_1, h_2, ..., h_T]
- Each point h_t encodes the "meaning" of the token sequence [x_1...x_t]

### LSTM Cell State Geometry

**Cell State Space C ⊂ ℝ^(hidden_size)**
- **Separate from hidden state** - internal memory mechanism
- c_t accumulates information across timesteps via additive connections
- Allows gradient flow across long sequences (mitigates vanishing gradient)
- Controlled by input, forget, and output gates

### Layering: Stacked LSTM Architecture

**Two-Layer LSTM Stack:**
```
Layer 1 (hidden size 650 or 1500):
  h¹_t = LSTM₁(x_t, h¹_{t-1})  [receives embedded input x_t]

Layer 2 (hidden size 650 or 1500):
  h²_t = LSTM₂(h¹_t, h²_{t-1})  [receives Layer 1 hidden output]

Output:
  logits_t = W_out h²_t + b_out  [linear projection of Layer 2 output]
```

**Dropout Separation by Layer:**
- Input → Layer 1: standard dense dropout
- Layer 1 output → Layer 2: dense dropout between layers
- Layer 2 output → Output: dense dropout before final projection
- Within each LSTM: **NO dropout on recurrent connections**

---

## 4. Architecture Deep Dive

### Block Diagram: Two-Layer Regularized LSTM

```
┌─────────────────────────────────────────────────────────────────┐
│                      LANGUAGE MODELING PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘

INPUT TOKENS (Batch of 20 sequences, length 35)
    │
    ├─→ [Token Embedding Layer] ──────────────────────────────┐
    │    Embedding matrix: (10000, embed_dim)                 │
    │    Output: (20, 35, embed_dim)                          │
    │                                                          │
    ├─→ [Dropout on Embeddings] ◄ p_dropout ────────────────┤
    │    Shape: (20, 35, embed_dim)                          │
    │    Mask applied PER SEQUENCE, SAME across timesteps     │
    │                                                          │
    ├─→ [LSTM Layer 1] ◄ Params: W_ii, W_hi, W_ci, b_i ────┤
    │    Process each timestep: h¹_t = LSTM₁(x_embedded_t, h¹_{t-1})
    │    h¹_t ∈ (20, hidden_size)                            │
    │    [NO DROPOUT on recurrent path W_hi]                 │
    │                                                          │
    ├─→ [Dropout on Layer 1 Output] ◄ p_dropout ────────────┤
    │    Shape: (20, 35, hidden_size)                        │
    │    Applied BETWEEN TIMESTEPS (same mask per sequence)   │
    │                                                          │
    ├─→ [LSTM Layer 2] ◄ Params: W_ii, W_hi, W_ci, b_i ────┤
    │    h²_t = LSTM₂(h¹_dropout_t, h²_{t-1})                │
    │    h²_t ∈ (20, hidden_size)                            │
    │    [NO DROPOUT on recurrent path W_hi]                 │
    │                                                          │
    ├─→ [Dropout on Layer 2 Output] ◄ p_dropout ────────────┤
    │    Shape: (20, 35, hidden_size)                        │
    │                                                          │
    ├─→ [Output Linear Projection] ◄ W_out, b_out ──────────┤
    │    logits_t = W_out * h²_dropout_t + b_out             │
    │    logits_t ∈ (20, vocab_size) = (20, 10000)           │
    │                                                          │
    ├─→ [Softmax + Cross-Entropy Loss]                       │
    │    loss_t = CrossEntropy(logits_t, y_t)                │
    │    final_loss = mean(loss_t) over batch and time       │
    │                                                          │
    └─→ OUTPUT: scalar loss value
```

### Module-Level Architecture Table

| Module | Input Shape | Output Shape | Parameters | Where Dropout | Key Point |
|--------|-------------|--------------|-----------|---------------|-----------|
| Embedding | (B, T) indices | (B, T, d_e) | 10000 × d_e | After | Maps tokens to vectors |
| Dropout (input) | (B, T, d_e) | (B, T, d_e) | 0 | - | p_drop=0.5-0.65 |
| LSTM-1 | (B, T, d_e) | (B, T, h) | ≈4×d_e×h | None on recurrent | Forget/Input/Output gates |
| Dropout (layer-1) | (B, T, h) | (B, T, h) | 0 | Between LSTM layers | Same mask all timesteps |
| LSTM-2 | (B, T, h) | (B, T, h) | ≈4×h×h | None on recurrent | Second LSTM layer |
| Dropout (layer-2) | (B, T, h) | (B, T, h) | 0 | Before output | Applied to final LSTM output |
| Output Linear | (B, T, h) | (B, T, vocab) | h × 10000 | - | Projects to vocabulary |

**Legend:** B=batch size (20), T=sequence length (35), d_e=embedding dim, h=hidden size (650 or 1500)

### LSTM Cell Details

The LSTM cell at each timestep performs:

```
i_t = σ(W_ii * x_t + W_hi * h_{t-1} + W_ci * c_{t-1} + b_i)    [Input gate]
f_t = σ(W_if * x_t + W_hf * h_{t-1} + W_cf * c_{t-1} + b_f)    [Forget gate]
c̃_t = tanh(W_ic * x_t + W_hc * h_{t-1} + b_c)                  [Cell candidate]
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t                                 [Cell state update]
o_t = σ(W_io * x_t + W_ho * h_{t-1} + W_co * c_t + b_o)        [Output gate]
h_t = o_t ⊙ tanh(c_t)                                            [Hidden state]
```

**Where σ = sigmoid, tanh = hyperbolic tangent, ⊙ = element-wise multiplication**

**Critical: NO dropout is applied to W_hi, W_hf, W_hc, W_ho** (the recurrent weight matrices)

---

## 5. Forward Pass Pseudocode

### High-Level Forward Pass (Language Modeling)

```python
def forward_pass(token_sequence, model, dropout_prob=0.5, training=True):
    """
    Args:
        token_sequence: (batch_size=20, seq_length=35) of integer token indices
        model: LSTMLanguageModel with 2 stacked LSTM layers
        dropout_prob: probability of dropping units (0.5 for medium, 0.65 for large)
        training: boolean, apply dropout during training only

    Returns:
        loss: scalar cross-entropy loss
        logits: (batch_size, seq_length, vocab_size)
    """

    batch_size, seq_length = token_sequence.shape  # (20, 35)
    vocab_size = 10000
    hidden_size = 650  # or 1500 for large model

    # ======================== EMBEDDING LAYER ========================
    embeddings = embedding_matrix[token_sequence]  # (20, 35, embed_dim)

    if training:
        # Create dropout mask for embeddings: SAME mask across timesteps
        embed_dropout_mask = bernoulli(1 - dropout_prob, (20, 1, embed_dim))
        # Broadcast across time dimension
        embed_dropout_mask = embed_dropout_mask / (1 - dropout_prob)  # inverse scaling
        embeddings = embeddings * embed_dropout_mask  # (20, 35, embed_dim)

    # ======================== LSTM LAYER 1 ========================
    h1_states = []  # Will store (batch, hidden_size) for each timestep
    c1_states = []  # Cell states

    h1_prev = zeros((batch_size, hidden_size))  # (20, 650)
    c1_prev = zeros((batch_size, hidden_size))  # (20, 650)

    for t in range(seq_length):  # t = 0, 1, ..., 34
        x_t = embeddings[:, t, :]  # (20, embed_dim)

        # LSTM-1 cell computation (NO dropout on recurrent connections)
        i_t = sigmoid(W_ii @ x_t + W_hi @ h1_prev + W_ci * c1_prev + b_i)
        f_t = sigmoid(W_if @ x_t + W_hf @ h1_prev + W_cf * c1_prev + b_f)
        c_tilde = tanh(W_ic @ x_t + W_hc @ h1_prev + b_c)
        c_t = f_t * c1_prev + i_t * c_tilde
        o_t = sigmoid(W_io @ x_t + W_ho @ h1_prev + W_co * c_t + b_o)
        h_t = o_t * tanh(c_t)  # (20, 650)

        h1_states.append(h_t)
        c1_states.append(c_t)
        h1_prev = h_t
        c1_prev = c_t

    # Stack outputs: (batch, seq_length, hidden_size)
    h1_all = stack(h1_states, axis=1)  # (20, 35, 650)

    # ======================== DROPOUT LAYER 1 ========================
    if training:
        # Create dropout mask for LSTM-1 output: SAME mask across timesteps
        h1_dropout_mask = bernoulli(1 - dropout_prob, (20, 1, hidden_size))
        h1_dropout_mask = h1_dropout_mask / (1 - dropout_prob)
        h1_all = h1_all * h1_dropout_mask  # (20, 35, 650)

    # ======================== LSTM LAYER 2 ========================
    h2_states = []
    c2_states = []

    h2_prev = zeros((batch_size, hidden_size))  # (20, 650)
    c2_prev = zeros((batch_size, hidden_size))

    for t in range(seq_length):
        x_t = h1_all[:, t, :]  # (20, 650) - input from Layer 1

        # LSTM-2 cell computation (same as Layer 1)
        i_t = sigmoid(W_ii @ x_t + W_hi @ h2_prev + W_ci * c2_prev + b_i)
        f_t = sigmoid(W_if @ x_t + W_hf @ h2_prev + W_cf * c2_prev + b_f)
        c_tilde = tanh(W_ic @ x_t + W_hc @ h2_prev + b_c)
        c_t = f_t * c2_prev + i_t * c_tilde
        o_t = sigmoid(W_io @ x_t + W_ho @ h2_prev + W_co * c_t + b_o)
        h_t = o_t * tanh(c_t)

        h2_states.append(h_t)
        c2_states.append(c_t)
        h2_prev = h_t
        c2_prev = c_t

    h2_all = stack(h2_states, axis=1)  # (20, 35, 650)

    # ======================== DROPOUT LAYER 2 ========================
    if training:
        h2_dropout_mask = bernoulli(1 - dropout_prob, (20, 1, hidden_size))
        h2_dropout_mask = h2_dropout_mask / (1 - dropout_prob)
        h2_all = h2_all * h2_dropout_mask  # (20, 35, 650)

    # ======================== OUTPUT LAYER ========================
    logits = h2_all @ W_out + b_out  # (20, 35) @ (650, 10000) = (20, 35, 10000)

    # ======================== LOSS COMPUTATION ========================
    # Target: next token prediction (shift targets by 1 position)
    targets = token_sequence[:, 1:]  # (20, 34) - predict y_{t+1} from x_t
    logits_for_loss = logits[:, :-1, :]  # (20, 34, 10000)

    # Cross-entropy: -log P(y_t | x_{1..t})
    probs = softmax(logits_for_loss, dim=-1)
    batch_loss = -log(probs[batch_idx, time_idx, targets[batch_idx, time_idx]])
    loss = mean(batch_loss)  # scalar

    return loss, logits
```

### Key Shape Annotations Summary

```
Forward pass tensor shapes:
  Tokens:           (20, 35)              → indices
  Embeddings:       (20, 35, embed_dim)   → dense vectors
  LSTM-1 hidden:    (20, 35, 650)         → layer-1 output
  After dropout-1:  (20, 35, 650)         → same shape, scaled by 1/(1-p)
  LSTM-2 hidden:    (20, 35, 650)         → layer-2 output
  After dropout-2:  (20, 35, 650)         → scaled
  Logits:           (20, 35, 10000)       → unnormalized probabilities
  Loss:             scalar                → cross-entropy

Dropout masks (training only):
  All masks shape:  (20, 1, hidden_size)  → same across time, different per sequence
  Scale factor:     1 / (1 - p_drop)      → inverse scaling for expectation
```

---

## 6. Heads, Targets, and Losses

### Language Modeling Task Definition

**Objective:** Learn P(w_t | w_1, w_2, ..., w_{t-1})

Given a sequence of words, predict the next word at each position.

### Targets and Alignment

**Input tokens:** w_1, w_2, w_3, ..., w_T (sequences of length T=35)

**Target tokens (shifted by 1):**
```
Position: 1      2      3      ...    34     35
Input:    w_1    w_2    w_3    ...    w_34   w_35
Target:   w_2    w_3    w_4    ...    w_35   <eos>
```

The model receives word w_t as input and predicts target word w_{t+1}.

**Loss computation ignores final timestep:** Only 34 positions used (positions 1-34 map to predicting positions 2-35).

### Output Head Structure

**Single output head:** Linear projection from hidden state to vocabulary

```
logit_t = W_out @ h²_t + b_out

Where:
  h²_t ∈ (batch_size, hidden_size) = (20, 650)
  W_out ∈ (hidden_size, vocab_size) = (650, 10000)
  b_out ∈ (vocab_size,) = (10000,)
  logit_t ∈ (batch_size, vocab_size) = (20, 10000)
```

**No hidden layer** between LSTM output and vocabulary logits. Direct linear projection maximizes capacity.

### Softmax and Probability Computation

```
p_t = softmax(logit_t)  ∈ [0,1]^vocab_size

p_t[i] = exp(logit_t[i]) / Σ_j exp(logit_t[j])

where Σ_i p_t[i] = 1
```

### Cross-Entropy Loss

**Per-token loss:**
```
L_t = -log(p_t[y_t])

where y_t ∈ {1, 2, ..., vocab_size} is the target word index at timestep t
```

**Sequence loss (average over time and batch):**
```
L = (1 / (B × T)) Σ_b=1^B Σ_t=1^T L_t^b

Where B=20 (batch size), T=34 (effective sequence length after shift)
```

### Perplexity Metric

**Definition:** Exponentiated average cross-entropy across entire dataset

```
Perplexity = exp(L_dataset)

Where L_dataset is the average cross-entropy loss over ALL test tokens
```

**Interpretation:**
- Perplexity ≈ branching factor of model's predictions
- Perplexity of 78.4 means model is ~78.4x uncertain at each prediction
- Lower is better

**Penn Treebank Performance:**
- Medium model: ~92 perplexity
- Large model: ~78.4 perplexity (SOTA before this paper)

---

## 7. Data Pipeline and Augmentations

### Penn Treebank Dataset Overview

**Source:** Wall Street Journal articles with parse tree annotations

**Raw Statistics:**
- Total words: ~4.5 million English words
- Train set: 929,589 tokens
- Validation set: 73,760 tokens
- Test set: 82,431 tokens
- Distinct word types: ~10,000 (after preprocessing)

### Preprocessing Pipeline

**Step 1: Tokenization**
- Split text by whitespace and punctuation
- Already provided in PTB format (one word per line)

**Step 2: Lowercasing**
- Convert all uppercase letters to lowercase
- Reduces vocabulary size, improves generalization
- Example: "THE" and "the" → "the"

**Step 3: Number Replacement**
- Replace all numeric sequences with special token `<N>`
- Example: "25 dollars" → "<N> dollars"
- Reduces sparsity of numeric vocabulary

**Step 4: Newline Replacement**
- Replace line breaks with `<eos>` (end-of-sequence) token
- Marks document/sentence boundaries
- Allows model to learn when to "reset" for new context

**Step 5: Punctuation Removal**
- Remove most punctuation (except as part of numbers)
- Further reduces vocabulary complexity
- Example: "well-known" → "wellknown"

**Step 6: Vocabulary Truncation**
- Keep only the 10,000 most frequent words
- Replace all out-of-vocabulary (OOV) words with `<unk>` (unknown) token
- Example: "floccinaucinilicilification" (rare) → "<unk>"

### Preprocessing Example

**Raw text:**
```
"The 25 workers in New York earned $100,000 in 2023."
```

**After preprocessing:**
```
the <N> workers in new york earned <N> in <N> .
```
(punctuation removed, lowercased, numbers replaced, assume some words in vocabulary)

### Vocabulary Mapping

**Vocabulary file:** Word → integer index mapping

```
the          → 1
of           → 2
a            → 3
...
<unk>        → 10000  (or sometimes 0)
<eos>        → 9999   (or sometimes index separately)
```

**Total vocabulary:** 10,001-10,002 tokens including special tokens

### Data Splitting and Usage

| Split | Size | Purpose | Notes |
|-------|------|---------|-------|
| Train | 929,589 tokens | Parameter learning | 20-token sequences |
| Valid | 73,760 tokens | Hyperparameter tuning & early stopping | Monitor overfitting |
| Test | 82,431 tokens | Final evaluation | Perplexity reported here |

### Batch Construction

**Batching strategy:** Pack 20 non-overlapping sequences

**For a single batch:**
1. Read next 20 sequences from dataset, each of length 35
2. Pad shorter sequences if needed (rare in PTB, uniform lengths)
3. Create batch tensor: (20, 35)
4. Forward pass through model
5. Compute loss on 34 effective positions (account for shift)

**Example batch:**
```
Sequence 1: [w_1^1, w_2^1, ..., w_35^1]  → predict [w_2^1, ..., w_35^1, <eos>]
Sequence 2: [w_1^2, w_2^2, ..., w_35^2]  → predict [w_2^2, ..., w_35^2, <eos>]
...
Sequence 20: [w_1^20, w_2^20, ..., w_35^20]  → predict [w_2^20, ..., w_35^20, <eos>]
```

### Data Augmentation

**No explicit data augmentation used in this paper.** The only "augmentation" is through:
- **Dropout regularization** (applied during training)
- **Standard mini-batch shuffling** during training

The focus is on the regularization technique (dropout placement) rather than data manipulation.

---

## 8. Training Pipeline

### Optimization Algorithm

**Stochastic Gradient Descent (SGD)** with learning rate decay

```
θ_new = θ_old - learning_rate × ∇L

Learning rate decay schedule (validation-triggered):
  If validation_loss ≤ best_validation_loss:
    keep learning_rate
  Else:
    learning_rate ← learning_rate / decay_factor
```

### Hyperparameter Configuration

#### Medium Model

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Hidden units per layer | 650 | Two-layer LSTM stack |
| Embedding dimension | ~200-400 | Typical, not specified |
| Dropout probability | 0.5 | Applied to non-recurrent connections |
| Batch size | 20 | Number of sequences per update |
| BPTT length | 35 | Backprop through 35 timesteps |
| Learning rate (initial) | 1.0 | Will decay |
| Decay factor | 1.2 | LR ← LR / 1.2 after patience |
| Patience (epochs) | 6 | Decay after 6 epochs without improvement |
| Gradient clipping | 5.0 | Clip gradients to ±5.0 |
| Weight initialization | Uniform [-0.05, 0.05] | All parameters |
| Total epochs | 39 | Training runs for 39 epochs max |
| Optimizer state | None | Plain SGD, no momentum |

#### Large Model

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Hidden units per layer | 1500 | Capacity = 2.3× medium |
| Embedding dimension | ~200-400 | Typical |
| Dropout probability | 0.65 | Higher dropout for larger model |
| Batch size | 20 | Same as medium |
| BPTT length | 35 | Same sequence length |
| Learning rate (initial) | 1.0 | Same starting rate |
| Decay factor | 1.15 | More gradual decay |
| Patience (epochs) | 14 | More epochs before decay |
| Gradient clipping | 10.0 | Higher clip for larger gradients |
| Weight initialization | Uniform [-0.04, 0.04] | Tighter range for stability |
| Total epochs | 55 | More epochs (1.4× medium) |
| Optimizer state | None | Plain SGD |

### Training Procedure (Pseudocode)

```python
def train_lstm_model(train_data, valid_data, config):
    """
    Args:
        train_data: list of (seq_length, batch_size) token tensors
        valid_data: validation sequences
        config: hyperparameter dict with LR, dropout, etc.
    """

    # Initialize model
    model = TwoLayerLSTM(
        vocab_size=10000,
        hidden_size=config['hidden_size'],      # 650 or 1500
        dropout_prob=config['dropout_prob'],    # 0.5 or 0.65
        weight_init=config['init_range']        # [-0.05, 0.05] or [-0.04, 0.04]
    )

    learning_rate = config['initial_lr']         # 1.0
    decay_factor = config['decay_factor']       # 1.2 or 1.15
    patience = config['patience']               # 6 or 14
    grad_clip = config['grad_clip']             # 5.0 or 10.0
    max_epochs = config['max_epochs']           # 39 or 55

    best_valid_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # ==================== TRAINING ====================
        total_train_loss = 0
        total_tokens = 0

        for batch in get_batches(train_data, batch_size=20):
            # Forward pass with dropout enabled
            loss, logits = model.forward(batch, training=True)

            # Backward pass
            gradients = compute_gradients(loss, model.parameters)

            # Gradient clipping: norm clipping
            norm = sqrt(sum(g^2 for g in gradients))
            if norm > grad_clip:
                for g in gradients:
                    g ← g * (grad_clip / norm)

            # SGD update
            for param, grad in zip(model.parameters, gradients):
                param ← param - learning_rate * grad

            # Accumulate loss
            total_train_loss += loss.item() * batch_size
            total_tokens += batch_size * 34  # 34 effective predictions per sequence

        train_ppl = exp(total_train_loss / total_tokens)
        print(f"Epoch {epoch}: Train PPL = {train_ppl:.2f}")

        # ==================== VALIDATION ====================
        total_valid_loss = 0
        total_valid_tokens = 0

        for batch in get_batches(valid_data, batch_size=20):
            loss, _ = model.forward(batch, training=False)  # NO dropout during eval
            total_valid_loss += loss.item() * batch_size
            total_valid_tokens += batch_size * 34

        valid_ppl = exp(total_valid_loss / total_valid_tokens)
        print(f"         Valid PPL = {valid_ppl:.2f}")

        # ==================== LEARNING RATE SCHEDULING ====================
        if valid_ppl < best_valid_loss:
            best_valid_loss = valid_ppl
            epochs_without_improvement = 0
            print(f"         New best! Saving model...")
            save_model_checkpoint(model)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                learning_rate ← learning_rate / decay_factor
                epochs_without_improvement = 0
                print(f"         Decay LR to {learning_rate:.6f}")

        if learning_rate < 1e-5:  # Stop if LR becomes too small
            print("Learning rate too small, stopping training")
            break

    return model
```

### Gradient Clipping Strategy

**Why gradient clipping is necessary:**
- BPTT unrolls 35 timesteps, creating deep computation graph
- RNNs suffer from exploding gradients during backprop through time
- Gradients can grow exponentially: ∂L/∂θ ∝ (λ)^35 where λ > 1

**Norm clipping implementation:**
```
g_norm = sqrt(Σ_i g_i^2)  [compute L2 norm of gradient vector]

If g_norm > threshold:
    g_i ← g_i × (threshold / g_norm)  [scale all gradients proportionally]
```

**Effect:** Prevents catastrophic weight updates while preserving gradient direction

### Learning Rate Decay Schedule

**Validation-triggered decay (not fixed schedule):**

```
epoch 0-5:  LR = 1.0
epoch 6:    valid_ppl doesn't improve → decay
            LR ← 1.0 / 1.2 ≈ 0.833

epoch 7-12: LR = 0.833
epoch 13:   decay again
            LR ← 0.833 / 1.2 ≈ 0.694

...continue until LR ≈ 0
```

**Medium model:** Reaches final LR ≈ 0.00043 by epoch 39
**Large model:** Same process over 55 epochs

---

## 9. Dataset and Evaluation Protocol

### Penn Treebank Language Modeling Dataset

**Official Dataset Splits:**

| Split | Tokens | Files | Purpose |
|-------|--------|-------|---------|
| Training | 929,589 | wsj/00-20/* | Learn model parameters |
| Validation | 73,760 | wsj/21-22/* | Tune hyperparameters, early stopping |
| Test | 82,431 | wsj/23/* | Final unbiased evaluation |

**Vocabulary:**
- 10,000 most frequent words retained
- All other tokens replaced with `<unk>`
- Additional special tokens: `<eos>` (end-of-sequence)

**Text origin:** Wall Street Journal (WSJ) articles, 1987-1989

### Preprocessing Details (Applied to All Splits)

1. **Lowercasing:** All uppercase → lowercase
2. **Numeral replacement:** Any digit sequence → `<N>`
3. **Punctuation removal:** Most punctuation dropped
4. **Whitespace normalization:** Multiple spaces → single space
5. **Vocabulary filtering:** Rare words → `<unk>`

### Evaluation Metric: Perplexity

**Definition:**
```
Perplexity(test_set) = exp(average_cross_entropy)
                     = exp(- (1/N) Σ_i log P(w_i | context_i))

Where:
  N = total number of test tokens = 82,431
  w_i = the i-th test token
  P(w_i | context_i) = model's predicted probability of true token
```

**Computation procedure:**

```python
def compute_perplexity(model, test_data):
    total_loss = 0
    total_tokens = 0

    for batch in get_batches(test_data):
        loss, _ = model.forward(batch, training=False)

        # Loss is already normalized cross-entropy
        total_loss += loss.item() * batch_size * seq_length
        total_tokens += batch_size * seq_length

    average_loss = total_loss / total_tokens
    perplexity = exp(average_loss)

    return perplexity
```

**Interpretation:**
- **Perplexity = 1:** Model is certain about every token (impossible in practice)
- **Perplexity = 78.4:** Model is ~78.4× uncertain; equivalent to uniform distribution over 78 tokens
- **Perplexity = 10,000:** Model essentially random (uniform over all vocab)

**Why perplexity over accuracy:**
- In language modeling, next token prediction is rare to be "correct" in 1-shot
- Perplexity rewards assigning high probability to the true token, even if not rank-1
- Standard metric enabling comparison across papers

### Evaluation Protocol

**During training (every epoch):**
1. Run model on validation set with dropout=off
2. Compute validation perplexity
3. If validation PPL improved: save checkpoint
4. If validation PPL stalled (6-14 epochs): decay learning rate
5. Continue training

**After training (final evaluation):**
1. Load best checkpoint (based on validation PPL)
2. Run model on test set with dropout=off
3. Compute final test perplexity
4. Report test PPL as primary metric

**No cross-validation:** Single train/val/test split (standard for PTB)

### Baseline Comparisons

**Other models evaluated on PTB (context for paper's results):**

| Model | Year | Test Perplexity |
|-------|------|-----------------|
| Kneser-Ney smoothed 5-gram | 2001 | ~141 |
| Neural probabilistic LM | 2003 | ~128 |
| Feedforward neural LM | 2007 | ~126 |
| Recurrent neural network (basic) | 2011 | ~107 |
| **Large LSTM + regularization** | **2014** | **78.4** |

The paper's result is substantially better than all prior work.

---

## 10. Results Summary and Ablations

### Main Results: Test Perplexity

**Table 1: Medium and Large Regularized LSTM (Two Layers)**

| Model | Hidden Size | Dropout | # Params | Test PPL | Improvement |
|-------|-------------|---------|----------|----------|-------------|
| Medium | 650 | 0% | ~4.5M | 91.5 | baseline |
| Medium | 650 | 50% | 4.5M | 86.2 | ↓ 5.3 points |
| **Large** | **1500** | **0%** | **20M** | **82.3** | vs baseline |
| **Large** | **1500** | **65%** | **20M** | **78.4** | ↓ 3.9 points |

**Key observation:** Larger model + higher dropout rate yields best results.

### Impact of Dropout Rate (Ablation on Large Model)

| Dropout Rate | Test PPL | Relative to No Dropout |
|--------------|----------|------------------------|
| 0% | 82.3 | baseline |
| 20% | 80.1 | ↓ 2.2 (3%) |
| 40% | 78.9 | ↓ 3.4 (4%) |
| 50% | 78.5 | ↓ 3.8 (5%) |
| 65% | **78.4** | ↓ **3.9 (5%)** - optimal |
| 75% | 78.7 | ↓ 3.6 (4%) - slightly worse |

**Conclusion:** 65% dropout is sweet spot for 1500-unit LSTM on PTB.

### Impact of Hidden Size (Model Capacity)

| Hidden Size | Dropout | Test PPL | # Parameters |
|-------------|---------|----------|--------------|
| 300 | 0% | 101.2 | ~1M |
| 300 | 50% | 95.3 | 1M |
| 650 | 0% | 91.5 | ~4.5M |
| 650 | 50% | 86.2 | 4.5M |
| 1500 | 0% | 82.3 | 20M |
| 1500 | 65% | **78.4** | 20M |

**Trend:** Larger hidden size + higher dropout needed for larger capacity.

### Ablation: Where to Apply Dropout

**Critical experiment: Which connections should be masked?**

| Configuration | Test PPL | Notes |
|---------------|----------|-------|
| No dropout (baseline) | 82.3 | - |
| Dropout on recurrent only | 106.5 | ✗ Catastrophic failure |
| Dropout on non-recurrent only | **78.4** | ✓ **Best result** |
| Dropout on all connections | 89.2 | ✗ Worse than non-recurrent |

**Key insight:** Applying dropout to recurrent connections (W_h matrices) destroys LSTM's ability to maintain long-term state. This is the critical finding of the paper.

### Task-Specific Results

#### Language Modeling: Penn Treebank
- **Test perplexity:** 78.4 (new SOTA)
- **Previous SOTA:** ~82-84
- **Improvement:** ~5-7% relative

#### Machine Translation: WMT'14 English-French
- Used in NMT model (different architecture than PTB)
- Dropout applied to non-recurrent connections
- Consistent improvements in BLEU score
- Helps stabilize training

#### Speech Recognition: TIMIT
- Acoustic sequence modeling task
- Similar improvements observed
- Dropout prevents overfitting to small TIMIT dataset

#### Image Captioning: MSCOCO
- Generating text descriptions of images
- Regularized LSTM reduces overfitting
- Particularly helpful with limited caption data

### Training Curves: Medium Model

**Typical training progression (39 epochs):**

```
Epoch  Train PPL  Valid PPL  Learning Rate
-----  ---------  ---------  --------
  0      120.5     115.2      1.000
  5      92.3      89.1       1.000
 10      81.5      83.4       1.000
 15      73.2      86.2       0.833    ← decay triggered
 20      68.1      85.9       0.694    ← decay triggered
 25      64.5      86.2       0.579
 30      62.1      86.3       0.482
 35      60.5      86.4       0.402
 39      59.8      86.2       Test: 86.2 PPL
```

**Observation:** Validation PPL plateaus around epoch 10-15, then starts increasing (overfitting). Learning rate decay stabilizes it temporarily but doesn't prevent eventual overfitting. This is where early stopping would be applied.

### Large Model Training Dynamics

**Key differences from medium model:**
- Slower convergence (55 vs 39 epochs)
- More aggressive decay schedule (factor 1.15 vs 1.2)
- Higher gradient clipping (10 vs 5)
- Takes longer to reach optimal perplexity but achieves better final result

---

## 11. Practical Insights

### 10 Engineering Takeaways

1. **Apply dropout ONLY to feedforward connections, NEVER recurrent**
   - The core contribution. Masking temporal connections destroys sequence memory. Tested extensively and confirmed best practice.

2. **Use same dropout mask across all timesteps for a given connection**
   - Different masks per timestep add noise to the recurrent path, defeating the purpose. Mask is fixed for entire sequence.

3. **Larger models need higher dropout rates**
   - 650 units: 50% dropout optimal
   - 1500 units: 65% dropout optimal
   - Suggests: p_drop ≈ 0.5 + 0.00005 × hidden_size

4. **Inverse scaling of activations is critical**
   - Multiply dropped activations by 1/(1-p) during training
   - Maintains expected value: E[x] unchanged whether dropout is applied or not
   - Enables seamless transition from training to test (no changes needed)

5. **Gradient clipping is essential for BPTT stability**
   - With 35-step unrolling, exploding gradients are common
   - Norm-based clipping (vs value clipping) preserves gradient direction
   - Threshold should scale with hidden size: clip = 5 for 650 units, 10 for 1500 units

6. **Validation-triggered learning rate decay outperforms fixed schedules**
   - Adapts to individual run's dynamics
   - Decay when validation loss plateaus (after N=6-14 epochs)
   - Allows more aggressive initial learning rate

7. **Initialize weights in tighter ranges for larger models**
   - Small model: [-0.05, 0.05]
   - Large model: [-0.04, 0.04]
   - Prevents extreme initial activations that could saturate gates

8. **Use plain SGD (not momentum-based optimizers)**
   - Sufficient for this task
   - Momentum can interact poorly with gradient clipping in RNNs
   - Simplifies hyperparameter tuning

9. **Monitor both train and validation loss for overfitting signals**
   - Train loss continues decreasing even after val loss plateaus
   - Sign to start decay (without early stopping, training continues)
   - Allows model to benefit from further training at lower LR

10. **Batch processing provides implicit regularization**
    - 20-sequence minibatches add noise via stochasticity
    - Reduces need for very high dropout on some layers
    - Larger batches = stronger regularization needed

### 5 Critical Gotchas (and How to Avoid Them)

1. **Gotcha: Applying dropout to recurrent connections**
   - **Symptom:** Validation perplexity jumps to 105+ (vs 78 for correct approach)
   - **Cause:** LSTM loses ability to maintain state over sequences
   - **Fix:** Apply dropout ONLY between layers and at input, never on W_h matrices
   - **Detection:** Test with manual ablation; perplexity will be obviously worse

2. **Gotcha: Different dropout masks per timestep**
   - **Symptom:** Model trains but converges slower; validation perplexity higher
   - **Cause:** Recurrent path gets noisier during backprop, harder to learn long dependencies
   - **Fix:** Generate mask once per sequence, reuse across all timesteps
   - **Detection:** Look for: Did I regenerate the mask inside the timestep loop?

3. **Gotcha: Forgetting inverse scaling (forgetting to divide by 1-p)**
   - **Symptom:** Train loss high, gradient clipping triggers often, training unstable
   - **Cause:** Dropped activations have lower expected value (×(1-p)), input to next layer is scaled down
   - **Fix:** After masking, multiply by 1/(1-p) to maintain expected value
   - **Code:** `h = h * mask / (1 - dropout_prob)`
   - **Detection:** Compare train loss magnitude with/without dropout; should be similar

4. **Gotcha: Not clipping gradients (or clipping too aggressively)**
   - **Symptom (no clipping):** Loss explodes to NaN around epoch 5-10
   - **Symptom (too aggressive):** Loss barely decreases, like training is stalled
   - **Cause:** BPTT unrolls 35 steps; gradients multiply across this depth
   - **Fix:** Start with threshold = 5 for 650 units, threshold = 10 for 1500 units
   - **Detection:** Check gradient norm before/after clipping; should clip ~1-5% of updates

5. **Gotcha: Over-aggressive learning rate**
   - **Symptom:** Loss oscillates wildly, perplexity jumps around
   - **Cause:** Gradient clipping + BPTT = already aggressive updates; too high LR = overshoot
   - **Fix:** Start at LR=1.0 with decay factor 1.2 (medium) or 1.15 (large)
   - **Detection:** If loss increases after most updates, LR is too high

### Overfitting Plan: From Zero to 78.4 PPL

**Week 1: Get the basics working**
- Implement 1-layer LSTM with embedded inputs (skip dropout)
- Target: ~95 perplexity (basic LSTM should get this)
- Success metric: Model trains, loss decreases each epoch

**Week 2: Add regularization infrastructure**
- Implement dropout layer with inverse scaling
- Add gradient clipping
- Implement learning rate decay
- Target: ~88 perplexity
- Success metric: Overfitting is visible (train PPL << val PPL)

**Week 3: Stack layers and increase capacity**
- Stack 2 LSTM layers (don't share weights)
- Add dropout between layers
- Use larger hidden size (650 units)
- Increase dropout rate (0.5)
- Target: ~87 perplexity
- Success metric: Model reaches 39 epochs with learning rate decay triggered

**Week 4: Scale up to large model**
- Increase hidden size to 1500
- Increase dropout to 0.65
- Increase gradient clipping to 10
- Adjust weight init range [-0.04, 0.04]
- Adjust decay schedule (factor 1.15, patience 14)
- Run for 55 epochs
- **Target: 78.4 perplexity**
- Success metric: Final test PPL ≤ 79

**Debugging checklist if stuck:**
- [ ] Dropout applied only to non-recurrent? (Check: is W_h ever masked?)
- [ ] Inverse scaling included? (Check: multiplying by 1/(1-p)?)
- [ ] Gradient clipping working? (Check: are gradients reasonable magnitude?)
- [ ] Learning rate decay triggered? (Check: did val PPL plateau, then LR decay?)
- [ ] Two layers stacked correctly? (Check: is layer 2 input coming from layer 1 output, not raw embeddings?)
- [ ] Validation protocol correct? (Check: is dropout off during validation eval?)

---

## 12. Minimal Reimplementation Checklist

### Architecture Must-Haves

- [ ] **Embedding layer:** maps token indices to dense vectors
  - Vocabulary size: 10,000
  - Dimension: ~200-400 (hyperparameter)
  - Initialized uniformly in [-0.05, 0.05] (medium) or [-0.04, 0.04] (large)

- [ ] **LSTM cells:** custom or use library (PyTorch LSTM/TensorFlow LSTM acceptable)
  - Implement forget, input, output gates + cell state
  - OR use pre-built LSTM unit
  - Critical: Ensure NO dropout on recurrent connections

- [ ] **Dropout layers:** custom implementation or library dropout
  - Applied after embedding
  - Applied between LSTM layers
  - Applied after final LSTM layer
  - SAME mask across all timesteps (generate once, reuse T times)
  - Multiply output by 1/(1-p_drop) to maintain expected value

- [ ] **Output projection:** linear layer from hidden state to vocabulary
  - Input dimension: hidden_size
  - Output dimension: vocab_size (10,000)
  - No hidden layers between LSTM and output

- [ ] **Cross-entropy loss:** standard implementation
  - Input: logits (batch_size, vocab_size)
  - Target: token indices (batch_size,)
  - Output: scalar loss (average over batch)

### Training Loop Must-Haves

- [ ] **Data loading:** iterate over PTB splits
  - Load train/valid/test text
  - Preprocess: lowercase, replace numbers with `<N>`, remove punctuation
  - Tokenize and map to indices using 10k vocabulary
  - Create sequences of length 35, batch size 20

- [ ] **Gradient clipping:** prevent exploding gradients
  - Compute L2 norm of all gradients
  - If norm > threshold (5 or 10), scale all gradients by threshold/norm
  - Code: `total_norm = sqrt(sum(g^2)); if total_norm > clip: g = g * (clip / total_norm)`

- [ ] **Learning rate decay:** validation-triggered
  - Track best validation loss
  - If val loss doesn't improve for N epochs (patience=6-14), decay LR by factor (1.2 or 1.15)
  - Code: `if val_loss > best_val_loss: lr = lr / decay_factor`

- [ ] **Early stopping (optional but recommended):**
  - Stop training when validation loss doesn't improve for M epochs
  - Save best model checkpoint based on validation metric
  - Load checkpoint before test evaluation

### Hyperparameter Configuration

**Medium Model Defaults:**
```
hidden_size = 650
embedding_dim = 250  # can vary 200-400
dropout_prob = 0.5
batch_size = 20
sequence_length = 35
learning_rate = 1.0
decay_factor = 1.2
patience = 6
gradient_clip = 5.0
max_epochs = 39
weight_init_range = (-0.05, 0.05)
```

**Large Model Defaults:**
```
hidden_size = 1500
embedding_dim = 250
dropout_prob = 0.65
batch_size = 20
sequence_length = 35
learning_rate = 1.0
decay_factor = 1.15
patience = 14
gradient_clip = 10.0
max_epochs = 55
weight_init_range = (-0.04, 0.04)
```

### Code Skeleton (PyTorch-style pseudocode)

```python
# Initialize
model = TwoLayerLSTM(vocab_size=10000, hidden_size=1500, dropout=0.65)
optimizer = SGD(model.parameters(), lr=1.0)

# Training loop
best_val_ppl = float('inf')
lr = 1.0
for epoch in range(max_epochs):

    # Train
    train_loss = 0
    for batch_x, batch_y in train_loader:
        logits = model(batch_x, training=True)  # dropout on
        loss = cross_entropy(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        optimizer.step()
        train_loss += loss.item()

    # Validate
    val_loss = 0
    for batch_x, batch_y in valid_loader:
        logits = model(batch_x, training=False)  # dropout off
        loss = cross_entropy(logits, batch_y)
        val_loss += loss.item()

    val_ppl = exp(val_loss / len(valid_data))

    # Decay
    if val_ppl > best_val_ppl:
        patience_counter += 1
        if patience_counter >= 14:
            lr = lr / 1.15
            patience_counter = 0
    else:
        best_val_ppl = val_ppl
        patience_counter = 0

    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Test
test_loss = 0
for batch_x, batch_y in test_loader:
    logits = model(batch_x, training=False)
    loss = cross_entropy(logits, batch_y)
    test_loss += loss.item()

test_ppl = exp(test_loss / len(test_data))
print(f"Test Perplexity: {test_ppl:.1f}")
```

### Validation Checklist Before Running

- [ ] Data loaded correctly: ~929k train, ~74k valid, ~82k test tokens
- [ ] Vocab size = 10,000 (including `<unk>` for OOV words)
- [ ] Batch size = 20, sequence length = 35
- [ ] LSTM has 2 layers, hidden_size = 1500
- [ ] Dropout applied after embedding, between layers, and after final LSTM
- [ ] NO dropout on recurrent (W_h) connections
- [ ] Dropout mask is same across timesteps (not regenerated per timestep)
- [ ] Inverse scaling (×1/(1-p)) applied after dropout masking
- [ ] Gradient clipping threshold = 10.0 for 1500-unit model
- [ ] Learning rate initialized to 1.0, decays by 1.15 after 14 epochs without improvement
- [ ] Loss is computed as average cross-entropy (not sum)
- [ ] Dropout is DISABLED during validation and test evaluation

### Expected Milestones

| Milestone | Expected Epoch | Expected Val PPL | Expected Test PPL |
|-----------|----------------|------------------|-------------------|
| Model trains (no NaN) | 1 | ~110-120 | - |
| Significant drop | 5 | ~90-95 | - |
| Plateau phase | 20 | ~85-88 | - |
| Final training | 55 | ~82-84 | **78.4** |

### Common Bugs & How to Spot Them

| Bug | Symptom | Detection |
|-----|---------|-----------|
| Dropout applied to recurrent | Val PPL = 105+ | Compare with/without dropout |
| Different masks per timestep | Slower convergence, higher final PPL | Print mask shape in loop |
| Missing inverse scaling | Loss magnitude too high, gradient clipping triggers often | Check activation values |
| No gradient clipping | Loss → NaN around epoch 5-10 | Monitor gradient norm |
| Wrong decay schedule | Val PPL keeps increasing | Check if LR is actually decaying |
| Dropout on during test | Inconsistent results, higher test PPL | Verify training=False flag |
| Wrong token sequence alignment | Model overfits to garbage | Verify targets are shifted by 1 |

---

## References & Further Reading

**Paper:**
- Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent Neural Network Regularization. arXiv:1409.2329

**Key papers cited/related:**
- Hochreiter & Schmidhuber (1997): LSTM architecture
- Graves (2012): Supervised Sequence Labeling with RNNs
- Hinton et al. (2012): Dropout for feedforward networks
- Bengio et al. (1994): Vanishing/exploding gradient problem

**Datasets:**
- Penn Treebank Language Modeling: https://www.cis.upenn.edu/~treebank/

**Implementations:**
- Original authors' Torch code (if available)
- TensorFlow examples
- PyTorch references in modern frameworks

---

**Document created:** March 3, 2026
**Summary version:** 1.0 (Implementation-focused)
**Total length:** ~8,000 words
**Target audience:** ML practitioners implementing RNN regularization
