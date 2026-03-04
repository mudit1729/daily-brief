# Vision Transformer (ViT) Paper Summary
## "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

**Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaobin Zhu, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Grangier, Johann Alabdulmohsin, Michael Tschannen

**Conference:** ICLR 2021 (Oral)

**ArXiv ID:** 2010.11929

**Publication Date:** October 2020

---

## Section 1: One-Page Overview

### Paper Metadata
- **Venue:** International Conference on Learning Representations (ICLR 2021)
- **Year:** 2020-2021
- **Institution:** Google Research and Brain Team
- **Citation Count:** 30,000+ (one of the most cited papers in deep learning)

### Key Novelty
This paper demonstrates that pure transformer architectures, without convolutional operations, can achieve state-of-the-art results on image classification tasks. The core insight is that image patches can be treated as tokens in a sequence, enabling direct application of the transformer architecture (originally designed for NLP) to computer vision.

**The Main Claim:**
> "When trained on large-scale datasets and fine-tuned to downstream tasks, Vision Transformer achieves excellent results compared to state-of-the-art convolutional networks while requiring substantially less compute to train."

### 3 Things to Remember
1. **Patches as Tokens:** Divide images into 16×16 (or other patch sizes) non-overlapping patches, flatten each patch, and treat as a token sequence in a transformer.
2. **Pre-training is Critical:** ViT requires large-scale pre-training (ImageNet-21k or JFT-300M) to outperform CNNs; on small datasets like ImageNet-1k alone, it underperforms.
3. **Scaling Laws:** Vision Transformers follow predictable scaling laws—larger models and more training data lead to better generalization without saturation, unlike CNNs which plateau.

### Impact & Significance
- First paper to successfully apply pure transformer architecture to image classification at scale
- Opened a new research direction: vision transformers became the dominant paradigm in computer vision
- Demonstrated transformer scalability beyond language (applicable to any sequential data)
- Influenced development of BERT for vision (BEiT), vision-language models (CLIP, ALIGN), and other architectures

---

## Section 2: Problem Setup and Outputs

### Problem Statement
**Task:** Image classification on standard benchmarks (ImageNet, CIFAR-10/100)

**Input:**
- A single RGB image of arbitrary resolution: **I ∈ ℝ^(H × W × 3)**
- Must convert to fixed size: typically 224×224 or 384×384

**Output:**
- Probability distribution over class labels: **y ∈ ℝ^C** where C is number of classes
- For ImageNet: C = 1000 classes
- Output via softmax from classification head

### Tensor Shape Tracking Throughout Pipeline

```
Input Image:
  Shape: (1, 3, 224, 224)  [batch_size, channels, height, width]
  Example: 224×224 RGB image

After Patch Embedding:
  Patches (16×16): (1, 196, 768)  [batch_size, num_patches, embedding_dim]
  Calculation: (224/16) × (224/16) = 14 × 14 = 196 patches
  Embedding dimension typically: 768 (for ViT-B), 1024 (ViT-L), 1280 (ViT-H)

After Adding CLS Token:
  Shape: (1, 197, 768)  [batch_size, num_patches+1, embedding_dim]
  The CLS token is a learnable parameter prepended to the sequence

After Position Embedding:
  Shape: (1, 197, 768)  [same as above, positions added element-wise]

After Transformer Encoder (12 layers for ViT-B):
  Shape: (1, 197, 768)  [batch_size, sequence_length, embedding_dim]
  (Position and shape unchanged through transformer blocks)

After Classification Head (MLP):
  Shape: (1, 1000)  [batch_size, num_classes]
  Final logits for softmax
```

### Key Assumptions & Constraints
1. Fixed input resolution (224×224 or 384×384)
2. Multi-class single-label classification (not multi-label)
3. Pre-training on large datasets significantly boosts performance
4. Computational resources: ViT-L requires ~300GB of data and significant GPU time to train

---

## Section 3: Coordinate Frames and Geometry

### Image Patches as Tokens
The fundamental geometric transformation is converting a 2D image into a sequence of patch tokens:

```
Original Image: 224 × 224 pixels (3 channels)
                ↓
Divide into Non-Overlapping Patches: 16 × 16 patches = 196 tokens
                ↓
Each Patch: 16 × 16 × 3 = 768 values (flattened)
                ↓
Linear Projection: 768 → 768 dimensions (embedding_dim)
                ↓
Patch Embedding Sequence: (196, 768)
```

### Spatial Position Embeddings
Unlike transformers in NLP where word order matters inherently, images require explicit position encoding:

**Position Encoding Type:** Learnable positional embeddings (not sinusoidal like original transformer)

**Mechanism:**
```
For each patch position p ∈ {1, 2, ..., 196}:
  pos_embed[p] ∈ ℝ^768  (learnable parameter)

Final embedding = patch_embedding[p] + pos_embed[p]
```

**Key Insight:** Position embeddings are learned from data, not fixed. The model learns which spatial relationships are important for the task.

**Ablation Finding:** Even without position embeddings, ViT performs reasonably well (but ~4% worse), suggesting patches already contain local spatial information through their pixel values.

### CLS Token (Class Token)
A special learnable token prepended to the patch sequence:

```
Input Sequence: [CLS] + [patch_1] + [patch_2] + ... + [patch_196] + [pos_embeds]
                (197 tokens total)

After Transformer:
  [CLS]_output ∈ ℝ^768  ← Used for classification
  [patch_k]_output ∈ ℝ^768  ← Discarded
```

**Design Choice Rationale:**
- Inspired by BERT's [CLS] token
- The transformer's self-attention allows CLS token to aggregate information from all patches
- Alternative: use global average pooling over patch embeddings (performs slightly worse)

### Geometric Interpretations
1. **Receptive Fields:** Each transformer block increases the receptive field. By layer 4, information from distant patches can interact. By layer 12, all patches can attend to all other patches.

2. **2D Structure Preservation:** Position embeddings preserve some 2D structure. Analysis shows that position embeddings of nearby patches have high cosine similarity, suggesting the model learns 2D spatial proximity.

3. **Patch Size Trade-off:**
   - Larger patches (32×32): Fewer tokens (49 for 224×224), faster processing, but less fine detail
   - Smaller patches (8×8): More tokens (784), captures fine details, but quadratic attention complexity becomes expensive
   - **Sweet spot:** 16×16 provides good balance

---

## Section 4: Architecture Deep Dive

### High-Level Architecture Diagram (ASCII)
```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT IMAGE (224×224×3)                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  LINEAR PATCH EMBEDDING      │
        │  Output: (196, 768)          │
        │  - Split into 16×16 patches  │
        │  - Flatten each patch        │
        │  - Project to 768 dims       │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  ADD POSITION EMBEDDINGS     │
        │  Add learned pos_embed       │
        │  Shape: (197, 768)           │
        │  + Prepend [CLS] token       │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────────────────────────┐
        │    TRANSFORMER ENCODER (12 blocks for ViT-B)        │
        │                                                      │
        │  Each Block:                                         │
        │  ├─ Multi-Head Self-Attention (12 heads)            │
        │  │  └─ QKV projection, scaled dot-product attention │
        │  │     outputs (197, 768)                           │
        │  │                                                  │
        │  ├─ Layer Normalization (pre-normalization)         │
        │  │                                                  │
        │  └─ MLP Feed-Forward Network                        │
        │     └─ Linear(768→3072) + GELU + Linear(3072→768)   │
        │     └─ ~4x expansion followed by projection back    │
        │                                                      │
        │  Residual connections around each sub-layer         │
        │  Output shape: (197, 768)                           │
        └──────────┬───────────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  EXTRACT [CLS] TOKEN         │
        │  Shape: (768,)               │
        │  Take first token output     │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  CLASSIFICATION HEAD (MLP)   │
        │  Input: (768,)               │
        │  Hidden: (1000,)             │  [only for pre-training]
        │  Output: (1000,)             │  [num_classes]
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  SOFTMAX + ARGMAX            │
        │  Output: class probabilities  │
        └──────────────────────────────┘
```

### Model Variants

#### ViT-Base (ViT-B/16)
```
Configuration:
  Patch Size: 16×16
  Hidden Dimension: 768
  Num Attention Heads: 12
  MLP Hidden Dimension: 3072 (4× expansion)
  Num Transformer Blocks: 12
  Total Parameters: ~86 million

Sequence Length: 196 patches + 1 CLS = 197 tokens
Attention Complexity: O(197²) per layer = ~39,000 operations per layer
```

#### ViT-Large (ViT-L/16)
```
Configuration:
  Patch Size: 16×16
  Hidden Dimension: 1024
  Num Attention Heads: 16
  MLP Hidden Dimension: 4096 (4× expansion)
  Num Transformer Blocks: 24
  Total Parameters: ~304 million

Computational Cost: ~4x ViT-B (more layers + larger hidden dimensions)
Sequence Length: 197 tokens
```

#### ViT-Huge (ViT-H/14)
```
Configuration:
  Patch Size: 14×14
  Hidden Dimension: 1280
  Num Attention Heads: 16
  MLP Hidden Dimension: 5120 (4× expansion)
  Num Transformer Blocks: 32
  Total Parameters: ~632 million

Sequence Length: (224/14)² + 1 = 257 tokens
Computational Cost: Extremely high due to larger patch embedding computation
  and increased sequence length (quadratic attention)
```

### Key Architectural Decisions

1. **Pre-Normalization:** Layer norm applied before (not after) attention and MLP
   - Improves training stability
   - Used in Transformers-XL

2. **GELU Activation:** Instead of ReLU in MLP
   - Smoother gradient flow
   - Standard in modern transformers

3. **No Convolutional Layers:** Pure transformer architecture
   - Removes inductive bias for local connectivity
   - Requires more data to learn spatial relationships
   - But enables scaling to arbitrary patch sequences

4. **Attention Head Configuration:**
   - ViT-B: 12 heads of 64 dimensions each (768/12 = 64)
   - ViT-L: 16 heads of 64 dimensions each (1024/16 = 64)
   - Maintains consistent head dimension across variants for stability

---

## Section 5: Forward Pass Pseudocode (Shape-Annotated)

### Detailed Forward Pass Implementation

```python
def vision_transformer_forward_pass(image, config):
    """
    Complete forward pass through Vision Transformer.

    Args:
        image: Input tensor of shape (batch_size, 3, 224, 224)
        config: Model configuration with hidden_size, num_patches, etc.

    Returns:
        logits: Classification logits of shape (batch_size, num_classes)
    """

    # ============================================================
    # STEP 1: LINEAR PATCH EMBEDDING
    # ============================================================
    # Input: (batch_size, 3, 224, 224)
    batch_size, _, height, width = image.shape
    patch_size = 16  # For ViT-B/16
    hidden_size = 768

    # Reshape image into patches
    # (batch_size, 3, 224, 224)
    #   → (batch_size, 3, 14, 16, 14, 16)  [reorganize into patch grid]
    patches = image.reshape(
        batch_size,
        3,
        height // patch_size, patch_size,
        width // patch_size, patch_size
    )

    # (batch_size, 3, 14, 16, 14, 16)
    #   → (batch_size, 14, 14, 3, 16, 16)  [transpose to group patches]
    patches = patches.transpose(0, 2, 4, 1, 3, 5)

    # (batch_size, 14, 14, 3, 16, 16)
    #   → (batch_size, 196, 768)  [flatten each patch]
    num_patches = (height // patch_size) * (width // patch_size)
    patches_flat = patches.reshape(batch_size, num_patches, -1)
    # Shape: (batch_size, 196, 768)  where 768 = 3*16*16

    # Project to embedding dimension via linear layer
    patch_embedding = LinearLayer(768, hidden_size)(patches_flat)
    # Shape: (batch_size, 196, 768)

    # ============================================================
    # STEP 2: PREPEND CLASS TOKEN
    # ============================================================
    # Create learnable class token (shared across batch)
    cls_token = LearnableParameter(shape=(1, 1, hidden_size))
    # Shape: (1, 1, 768)

    # Expand to batch size and concatenate
    cls_token_batch = cls_token.expand(batch_size, 1, hidden_size)
    # Shape: (batch_size, 1, 768)

    sequence = torch.cat([cls_token_batch, patch_embedding], dim=1)
    # Shape: (batch_size, 197, 768)
    # [CLS token at position 0, patches at positions 1-196]

    # ============================================================
    # STEP 3: ADD POSITION EMBEDDINGS
    # ============================================================
    # Learnable positional embeddings (trained during pre-training)
    position_embeddings = LearnableParameter(shape=(1, 197, hidden_size))
    # Shape: (1, 197, 768)

    # Broadcast and add to sequence
    sequence = sequence + position_embeddings
    # Shape: (batch_size, 197, 768)

    # ============================================================
    # STEP 4: TRANSFORMER ENCODER (12 blocks for ViT-B)
    # ============================================================
    # num_layers = 12 (ViT-B), 24 (ViT-L), 32 (ViT-H)
    num_layers = 12
    num_attention_heads = 12
    mlp_hidden_dim = 3072  # 4x expansion

    for layer_idx in range(num_layers):
        # ─────────────────────────────────────────
        # MULTI-HEAD SELF-ATTENTION
        # ─────────────────────────────────────────

        # Pre-normalization (normalize before attention)
        sequence_normalized = LayerNorm(hidden_size)(sequence)
        # Shape: (batch_size, 197, 768)

        # Project to Query, Key, Value
        # Each head operates on head_dim = 768/12 = 64 dimensions
        query = LinearLayer(hidden_size, hidden_size)(sequence_normalized)
        key = LinearLayer(hidden_size, hidden_size)(sequence_normalized)
        value = LinearLayer(hidden_size, hidden_size)(sequence_normalized)
        # Shape each: (batch_size, 197, 768)

        # Reshape for multi-head attention
        head_dim = hidden_size // num_attention_heads  # 64

        query = query.reshape(batch_size, 197, num_attention_heads, head_dim)
        query = query.transpose(1, 2)  # (batch_size, num_heads, 197, head_dim)
        # Final shape: (batch_size, 12, 197, 64)

        key = key.reshape(batch_size, 197, num_attention_heads, head_dim)
        key = key.transpose(1, 2)
        # Final shape: (batch_size, 12, 197, 64)

        value = value.reshape(batch_size, 197, num_attention_heads, head_dim)
        value = value.transpose(1, 2)
        # Final shape: (batch_size, 12, 197, 64)

        # Scaled dot-product attention for each head
        # scores = QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        # Shape: (batch_size, 12, 197, 197)

        scores = scores / math.sqrt(head_dim)  # Scale by 1/sqrt(64)
        scores = softmax(scores, dim=-1)  # Softmax over key dimension
        # Shape: (batch_size, 12, 197, 197)

        # Apply attention to values
        attention_output = torch.matmul(scores, value)
        # Shape: (batch_size, 12, 197, 64)

        # Concatenate heads and project back
        attention_output = attention_output.transpose(1, 2)
        # Shape: (batch_size, 197, 12, 64)

        attention_output = attention_output.reshape(batch_size, 197, hidden_size)
        # Shape: (batch_size, 197, 768)

        # Project attention output
        attention_output = LinearLayer(hidden_size, hidden_size)(attention_output)
        # Shape: (batch_size, 197, 768)

        # Residual connection
        sequence = sequence + attention_output
        # Shape: (batch_size, 197, 768)

        # ─────────────────────────────────────────
        # FEED-FORWARD MLP
        # ─────────────────────────────────────────

        # Pre-normalization (normalize before MLP)
        sequence_normalized = LayerNorm(hidden_size)(sequence)
        # Shape: (batch_size, 197, 768)

        # MLP: 768 → 3072 → 768
        mlp_output = LinearLayer(hidden_size, mlp_hidden_dim)(sequence_normalized)
        # Shape: (batch_size, 197, 3072)

        mlp_output = GELU()(mlp_output)
        # Shape: (batch_size, 197, 3072)

        mlp_output = LinearLayer(mlp_hidden_dim, hidden_size)(mlp_output)
        # Shape: (batch_size, 197, 768)

        # Residual connection
        sequence = sequence + mlp_output
        # Shape: (batch_size, 197, 768)

    # ============================================================
    # STEP 5: EXTRACT [CLS] TOKEN
    # ============================================================
    cls_output = sequence[:, 0, :]
    # Shape: (batch_size, 768)
    # Take only the first token (CLS token) from each sequence

    # Apply final layer normalization
    cls_output = LayerNorm(hidden_size)(cls_output)
    # Shape: (batch_size, 768)

    # ============================================================
    # STEP 6: CLASSIFICATION HEAD (MLP)
    # ============================================================
    # During pre-training: use MLP with hidden layer
    num_classes = 1000  # ImageNet

    if training_mode == "pre-training":
        # Pre-training head: 768 → 1000 → 1000
        logits = LinearLayer(hidden_size, num_classes)(cls_output)
        logits = GELU()(logits)
        logits = LinearLayer(num_classes, num_classes)(logits)
        # Shape: (batch_size, 1000)
    else:
        # Fine-tuning head: 768 → num_classes
        logits = LinearLayer(hidden_size, num_classes)(cls_output)
        # Shape: (batch_size, 1000)

    # ============================================================
    # STEP 7: SOFTMAX (for inference)
    # ============================================================
    probabilities = softmax(logits, dim=-1)
    # Shape: (batch_size, 1000)

    predictions = argmax(probabilities, dim=-1)
    # Shape: (batch_size,)
    # Each element is a class index

    return logits, probabilities, predictions
```

### Key Implementation Details

1. **Patch Projection Dimension:** 3 × 16 × 16 = 768 (matches hidden_size for ViT-B)
2. **Attention Scaling:** Divide by sqrt(64) = 8 to prevent exploding gradients
3. **Softmax in Attention:** Applied row-wise (over keys) to create probability distribution
4. **Residual Connections:** Around both attention and MLP blocks
5. **Layer Normalization:** Applied before (not after) each sub-layer (pre-normalization)

---

## Section 6: Heads, Targets, and Losses

### Classification Head Architecture

#### Pre-training Head
When pre-training on large datasets (ImageNet-21k, JFT-300M):

```
Input: [CLS] token output from transformer
  Shape: (batch_size, 768)

Linear Layer 1: 768 → 1000  [hidden layer]
  y = Wx + b
  Shape output: (batch_size, 1000)

GELU Activation

Linear Layer 2: 1000 → num_classes_pretraining  [output layer]
  For ImageNet-21k: 1000 → 14,000 classes
  For JFT-300M: 1000 → ~300,000 classes
  Shape output: (batch_size, num_classes_pretraining)
```

**Rationale for Hidden Layer:**
- Provides additional non-linearity and parameter capacity during pre-training
- Acts as a bottleneck that encourages learning good representations
- Removed during fine-tuning (only keep first layer weights)

#### Fine-tuning Head
When fine-tuning on downstream tasks (ImageNet-1k, CIFAR-10, etc.):

```
Input: [CLS] token output from transformer
  Shape: (batch_size, 768)

Linear Layer: 768 → num_classes_downstream
  For ImageNet-1k: 768 → 1000
  For CIFAR-10: 768 → 10
  For CIFAR-100: 768 → 100
  Shape output: (batch_size, num_classes_downstream)
```

**Important:** The pre-training hidden layer is discarded. Only the transformer weights are transferred.

### Loss Functions

#### Pre-training: Cross-Entropy Loss
```
Target: y ∈ {1, 2, ..., num_classes} (one-hot encoded)
  Example: for ImageNet-21k with 14,000 classes

Prediction: logits ∈ ℝ^num_classes
  Output of classification head before softmax

Cross-Entropy Loss:
  L_CE = -∑_c y_c * log(softmax(logits)_c)
       = -log(softmax(logits)[true_class])

  Equivalently:
  p = softmax(logits)
  L_CE = -log(p[true_class])
```

**Implementation in PyTorch:**
```python
criterion = torch.nn.CrossEntropyLoss()
logits = model(images)  # (batch_size, num_classes)
targets = labels  # (batch_size,) with values in [0, num_classes-1]
loss = criterion(logits, targets)
```

#### Fine-tuning: Also Cross-Entropy Loss
Same loss function as pre-training, but with different number of classes.

**Example for ImageNet-1k fine-tuning:**
```python
criterion = torch.nn.CrossEntropyLoss()
logits = model(images)  # (batch_size, 1000)
targets = labels  # (batch_size,) with values in [0, 999]
loss = criterion(logits, targets)
```

### Optimization Strategy: Pre-training vs Fine-tuning

#### Pre-training Phase
```
Dataset: ImageNet-21k or JFT-300M
Duration: Multiple epochs, hundreds of thousands of steps
Typical Setup:
  - Batch size: 4096 (distributed across 8 TPUs)
  - Learning rate: Relatively high (e.g., 0.001-0.004)
  - Learning rate schedule: Warmup + cosine decay
  - Optimizer: Adam with weight decay
  - Total iterations: ~1-3 million steps

Result: General-purpose image representations learned from diverse data
```

#### Fine-tuning Phase
```
Dataset: Downstream task (ImageNet-1k, CIFAR-10/100, Pets, etc.)
Duration: Shorter (10-50 epochs typical)
Strategy:
  - Start from pre-trained weights
  - Only update the classification head (frozen encoder)
  - Or: Update full model with lower learning rate

Typical Setup:
  - Batch size: 512
  - Learning rate: Lower than pre-training (0.00003-0.001)
  - Learning rate schedule: Linear decay or exponential decay
  - Total epochs: 10-90 depending on dataset size
  - Dropout/stochastic depth: May increase for regularization

Results:
  - ImageNet-1k: 88% top-1 accuracy (ViT-L/16 with pre-training)
  - CIFAR-10: ~99% accuracy
  - CIFAR-100: ~91% accuracy
```

### Head Design Ablations

From the paper's ablations:

1. **[CLS] vs Global Average Pooling:**
   - [CLS] token (used in ViT): 87.6% ImageNet accuracy
   - Global average pooling over patches: 87.3% accuracy
   - [CLS] is marginally better but both work well

2. **Position Embedding Interpolation:**
   - When changing image resolution from 224×224 to 384×384:
   - Linearly interpolate position embeddings (simple method)
   - Result: No significant accuracy loss
   - Suggests position embeddings capture general spatial structure

3. **Patch Embedding Size Impact:**
   - Larger patch size (32×32): Fewer tokens, faster, but lower accuracy
   - Smaller patch size (8×8): More tokens, higher accuracy, but slower
   - Optimal range: 14×16 patches

### Connection Between Loss and Accuracy
- **During pre-training:** Loss is a proxy for learning good representations
- **During fine-tuning:** Final accuracy on downstream task is the true metric
- **Relationship:** Better pre-training loss → Better fine-tuning accuracy (mostly monotonic with dataset size)

---

## Section 7: Data Pipeline

### Datasets Used

#### Pre-training Datasets

**ImageNet-21k**
```
Official Name: "ImageNet: Large-Scale Visual Recognition Challenge"
Number of Classes: 14,197 classes
Number of Images: ~14 million images
Characteristics:
  - Diverse visual categories (animals, objects, scenes, etc.)
  - High-quality human-annotated labels
  - Relatively balanced class distribution
  - Standard in computer vision (used since 2010)

Advantages for ViT:
  - Large scale enables learning general visual representations
  - Diverse enough to cover many visual concepts
  - Smaller than JFT-300M (training faster)

Typical Use:
  - Pre-training ViT models for 600-1000 epochs
  - Learning rate warmup + cosine decay schedule
```

**JFT-300M (Google's Internal Dataset)**
```
Official Name: "JFT-300M"
Number of Classes: ~300,000 classes
Number of Images: ~300 million images
Characteristics:
  - Largest available image dataset (proprietary to Google)
  - Extremely diverse (includes rare classes)
  - Noisy labels (web-sourced, not human-verified)
  - Highly imbalanced distribution

Advantages for ViT:
  - Massive scale enables learning at higher capacity
  - Results: Better pre-training loss and fine-tuning accuracy
  - Example: ViT-L pre-trained on JFT achieves 88.55% ImageNet acc
           vs 85.2% with ImageNet-21k pre-training

Trade-offs:
  - Requires 3× more compute than ImageNet-21k
  - Label noise requires robust training procedures
  - Not publicly available (limiting reproducibility)
```

#### Fine-tuning/Evaluation Datasets

**ImageNet-1k**
```
Official Name: "ILSVRC2012" (ImageNet Large-Scale Visual Recognition Challenge)
Number of Classes: 1000 classes (animals, objects, scenes)
Training Images: ~1.3 million images
Validation Images: ~50,000 images
Test Images: ~100,000 (labels not public)

Paper Results (ViT-L/16):
  - Pre-trained on ImageNet-21k: 87.76% top-1 accuracy
  - Pre-trained on JFT-300M: 88.55% top-1 accuracy
  - Fine-tuned for 20 epochs on 224×224 images

Comparison with CNNs:
  - ResNet-152: 78.3% (trained from scratch)
  - EfficientNet-B7: 85.7% (with pre-training)
  - ViT-L/16: 88.55% (with large-scale pre-training)
```

**CIFAR-10 and CIFAR-100**
```
CIFAR-10:
  - 60,000 images (32×32 resolution)
  - 10 classes (airplane, automobile, bird, cat, etc.)
  - 50,000 training + 10,000 test

CIFAR-100:
  - 60,000 images (32×32 resolution)
  - 100 classes (fine-grained categories)
  - Same split as CIFAR-10

Paper Results (ViT-B/32):
  - Pre-trained on ImageNet-21k:
    - CIFAR-10: 98.13% accuracy
    - CIFAR-100: 91.67% accuracy

  - Trained from scratch (no pre-training):
    - CIFAR-10: 88.59% accuracy (poor, requires pre-training)
    - CIFAR-100: 68.65% accuracy (very poor)
```

**Vision Task Adaptation Benchmark (VTAB)**
```
19 small-scale vision tasks:
  - Datasets with 1k-10k images typically
  - Tasks: Cifar(10/100), Caltech101, DTD, Flowers102, Pet,
           SVHN, Sun397, EuroSAT, Resisc45, RetinopathyDetection,
           Patch Camelyon, Eurosat, BigTransfer Datasets

Evaluation Protocol:
  - Use fixed train/val/test splits
  - Report median accuracy across 3 runs
  - Fine-tune for 2500 steps

Paper Results (ViT-B/32):
  - Pre-trained on ImageNet-21k: 77.64% median accuracy
  - Pre-trained on JFT-300M: 81.88% median accuracy
  - Demonstrates strong transfer learning capability
```

### Data Augmentation Pipeline

**Pre-training Augmentation (ImageNet-21k, JFT-300M):**
```python
augmentation = torchvision.transforms.Compose([
    # Resize with random aspect ratio preservation
    transforms.RandomResizedCrop(
        224,
        scale=(0.08, 1.0),  # Random crop covers 8%-100% of image
        ratio=(0.75, 1.333)  # Aspect ratio range: 3:4 to 4:3
    ),
    # Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),
    # Color jittering (brightness, contrast, saturation, hue)
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1
    ),
    # Random grayscale conversion
    transforms.RandomGrayscale(p=0.2),
    # Convert to tensor and normalize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    ),
    # Stochastic depth (DropPath) applied during forward pass
])
```

**Fine-tuning Augmentation (ImageNet-1k, CIFAR, etc.):**
```python
# Training augmentation
train_augmentation = torchvision.transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Test augmentation (no randomness)
test_augmentation = torchvision.transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

**Special Techniques Used:**
1. **RandAugment:** Automatic augmentation selection (not explicitly mentioned in ViT paper)
2. **Mixup:** Mix pairs of training examples (used in some ViT experiments)
3. **Stochastic Depth:** Drop entire transformer blocks during training (improves regularization)

### Data Pipeline Insights

1. **Pre-training Data Scaling:**
   - ImageNet-1k (1.3M) → Insufficient, ViT underperforms CNNs
   - ImageNet-21k (14M) → Adequate, ViT matches/exceeds CNNs
   - JFT-300M (300M) → Excellent, ViT significantly outperforms

2. **Resolution Strategy:**
   - Pre-training: 224×224
   - Fine-tuning: Can increase to 384×384 or higher
   - Method: Linear interpolation of position embeddings for new resolution

3. **Batch Size Impact:**
   - Pre-training: Large batch sizes (4096) help with convergence
   - Enables larger learning rates and better gradient estimates
   - Requires distributed training across multiple GPUs/TPUs

---

## Section 8: Training Pipeline

### Optimizer Configuration

#### Adam Optimizer Settings
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,           # Learning rate (tuned per configuration)
    betas=(0.9, 0.999),  # Exponential decay rates (standard)
    eps=1e-8,          # Numerical stability parameter
    weight_decay=0.1   # L2 regularization (crucial for ViT)
)
```

**Why Weight Decay is Critical for ViT:**
- Prevents overfitting to specific pre-training classes
- Acts as implicit regularization during fine-tuning
- Typical values: 0.05-0.1

**ViT vs CNN Optimization:**
- ViTs benefit from higher weight decay than CNNs
- Suggests ViTs are more prone to overfitting without regularization
- Related to patch embeddings lacking convolutional inductive bias

#### Learning Rate Schedule: Warmup + Cosine Decay
```
Phase 1: Linear Warmup (first 5% of training)
  ├─ Start: lr = 0
  └─ End: lr = peak_lr (e.g., 1e-3)

  Formula for epoch e in [0, warmup_epochs]:
    lr(e) = peak_lr × (e / warmup_epochs)

Phase 2: Cosine Decay (remaining 95% of training)
  ├─ Start: lr = peak_lr
  └─ End: lr = 1e-5 (minimal learning rate)

  Formula for epoch e in [warmup_epochs, total_epochs]:
    progress = (e - warmup_epochs) / (total_epochs - warmup_epochs)
    lr(e) = 0.5 × peak_lr × (1 + cos(π × progress))
```

**Visual Representation:**
```
Learning Rate
     ↑
     |        ╱╲
peak │       ╱  ╲╲
     |      ╱    ╲╲
     |     ╱      ╲╲
     |    ╱        ╲╲
     |   ╱          ╲╲
     |  ╱            ╲╲___
   0 ├─────────────────────→ Epochs
     0% 5%          100%
        ↑                ↑
      Warmup         Cosine Decay
```

**Typical Schedule for Pre-training:**
```
Total epochs: 90-300 (depending on dataset size)
Peak learning rate: 1e-3 to 2e-3
Warmup epochs: 5-10 (smaller portion for longer training)

Example (ImageNet-21k, 90 epochs):
  Warmup: epochs 0-4 (4.5% of total)
  Cosine: epochs 5-89 (95.5% of total)
```

**Typical Schedule for Fine-tuning:**
```
Total epochs: 10-90 (much shorter than pre-training)
Peak learning rate: 1e-4 to 1e-3 (lower than pre-training)
Warmup epochs: 5-20

Example (ImageNet-1k, 20 epochs):
  Warmup: epochs 0-4
  Cosine: epochs 5-19
```

### Hyperparameter Configuration Table

#### Pre-training Hyperparameters

| Parameter | ViT-B/16 | ViT-L/16 | ViT-H/14 | Notes |
|-----------|----------|----------|----------|-------|
| Batch Size | 4096 | 4096 | 4096 | Distributed (8 TPU v3 cores) |
| Optimizer | Adam | Adam | Adam | Standard PyTorch |
| Peak LR | 1e-3 | 1e-3 | 1e-3 | Tuned per model |
| Weight Decay | 0.1 | 0.1 | 0.1 | Important regularization |
| Total Epochs | 90 | 90 | 90 | On ImageNet-21k |
| Warmup Epochs | 5 | 5 | 5 | 5.6% of total |
| Data Augment | RandAugment | RandAugment | RandAugment | Automatic selection |
| Stochastic Depth | 0.1 | 0.2 | 0.3 | Increases with model size |
| Training Time | ~53 hours | ~178 hours | ~290 hours | On 8 TPU v3 cores |

#### Fine-tuning Hyperparameters

| Parameter | ViT-B/16 | ViT-L/16 | ViT-H/14 | Notes |
|-----------|----------|----------|----------|-------|
| Batch Size | 512 | 512 | 512 | Reduced vs pre-training |
| Optimizer | Adam | Adam | Adam | Same as pre-training |
| Peak LR | 1e-3 | 1e-3 | 1e-3 | Often tuned per task |
| Weight Decay | 0.0 | 0.0 | 0.0 | Set to zero for fine-tuning |
| Total Epochs | 20 | 20 | 20 | Varies by dataset (10-90) |
| Warmup Epochs | 5 | 5 | 5 | Same as pre-training |
| Resolution | 224×224 | 224×224 | 224×224 | Can increase to 384×384 |
| Dropout | 0.1 | 0.1 | 0.1 | Applied to embeddings |

### Regularization Techniques

**1. Stochastic Depth (DropPath)**
```
Applied to: Residual connections in each transformer block
Mechanism: Randomly drop entire blocks during training
Formula:
  p_drop(layer) = (layer_index / total_layers) × stochastic_depth_rate

Effect during forward pass:
  output = input + DropPath(sublayer(input))
           where DropPath drops the sublayer output with probability p_drop

ViT Values:
  - ViT-B: 0.1 (10% drop probability at deepest layer)
  - ViT-L: 0.2 (20% drop probability)
  - ViT-H: 0.3 (30% drop probability)

Why Larger Models Need More:
  - Helps prevent overfitting in large models
  - Acts as implicit ensemble of subnetworks
  - Improves generalization
```

**2. Weight Decay**
```
Implementation: L2 regularization in optimizer
Applied to: All trainable parameters

Effect on loss:
  L_total = L_cross_entropy + weight_decay × ||weights||²

Pre-training: weight_decay = 0.1 (strong regularization)
Fine-tuning: weight_decay = 0.0 (avoid over-regularizing)

Reason for Difference:
  - Pre-training: Prevent memorizing large dataset
  - Fine-tuning: Preserve learned representations, avoid over-regularizing small dataset
```

**3. Dropout (Token-level)**
```
Applied to: Patch embeddings after projection
Dropout Rate: 0.1 (10% of tokens randomly zeroed)

Effect:
  y = Dropout(x, p=0.1)
  - During training: Randomly zero out ~10% of embeddings
  - During inference: No dropout (or scale by 1/(1-p))

Why Patch-level:
  - Prevents co-adaptation of patch embeddings
  - Increases robustness to missing patches
  - Less critical than in CNNs (already have position embeddings)
```

### Training Precision & Numerical Stability

**Mixed Precision Training:**
```
- Forward pass: float16 (faster, less memory)
- Gradient computation: float16
- Loss scaling: Multiply loss by 1024 to prevent underflow
- Optimizer updates: float32 (maintain precision)

Benefits:
  - ~2× faster training
  - ~2× less GPU memory
  - Minimal accuracy loss with proper scaling
```

**Gradient Clipping:**
```
Global norm clipping:
  clip_norm = 1.0
  total_norm = ||∇parameters||₂
  if total_norm > clip_norm:
    scale = clip_norm / total_norm
    ∇parameters = scale × ∇parameters

Purpose:
  - Prevent exploding gradients
  - Stabilize training early on
  - Common in transformer training
```

### Distributed Training Setup

**ViT Training Deployment (from paper):**
```
Hardware: 8× TPU v3 cores (Google's custom accelerators)
  - Each TPU: ~420 TFLOPS
  - Memory: ~16 GB per core

Distributed Strategy: Data parallelism
  - Batch size: 4096 (512 per core)
  - Gradient accumulation: Not needed with 8 cores
  - All-reduce synchronization: After each batch

Synchronization Overhead:
  - Network communication: ~10-15% of training time
  - Load balancing: Ensure all cores finish simultaneously

Training Duration:
  - ViT-B on ImageNet-21k: ~53 hours (90 epochs)
  - ViT-L on ImageNet-21k: ~178 hours (90 epochs)
  - ViT-H on JFT-300M: ~290 hours (90 epochs on 300M images)
```

### Convergence Behavior

**Learning Curves (Pre-training):**
```
Loss (Pre-training)
  ↑
  |
4 |█████
  |█████░░░░
3 |█████░░░░
  |█████░░░░░░░░
2 |█████░░░░░░░░░░░░░
  |█████░░░░░░░░░░░░░░░░
1 |█████░░░░░░░░░░░░░░░░░░░░░░
  |
  └─────────────────────→ Training Steps
    0    100k   500k   1M    2M

  ├─ Warmup phase (steeper descent)
  └─ Cosine decay phase (slower, asymptotic)
```

**Accuracy (Fine-tuning on ImageNet):**
```
Accuracy
  ↑
  |                              ╱─── Final ~88%
88%|                       ╱╱╱╱╱
  |                  ╱╱╱╱╱
  |            ╱╱╱╱
  |       ╱╱╱╱
  |   ╱╱╱╱
80%|╱╱
  |
  └─────────────────────→ Epochs
    0    5   10   15   20

  Convergence: Very fast (20 epochs)
  Pre-training benefits: Massive (87% vs 50% from scratch)
```

---

## Section 9: Dataset + Evaluation Protocol

### Evaluation Metrics

#### Primary Metric: Top-1 Accuracy
```
Definition:
  For test set with N images and C classes:

  Accuracy = (# of correct predictions) / N

  Where "correct" means predicted class = true class

Formula:
  Top-1 Acc = (1/N) × ∑_{i=1}^{N} I(argmax(logits_i) == y_i)

  where I(·) is indicator function (1 if true, 0 if false)

Typical Results (ImageNet-1k):
  - ViT-L/16 (JFT pre-trained): 88.55%
  - ResNet-152 (ImageNet pre-trained): 82.96%
  - Improvement: +5.59 percentage points
```

#### Secondary Metric: Top-5 Accuracy
```
Definition:
  For each image, check if true class is among top 5 predicted classes

Formula:
  Top-5 Acc = (1/N) × ∑_{i=1}^{N} I(y_i ∈ top-5(logits_i))

Intuition:
  - Less strict than top-1
  - Useful for applications where any top-few prediction is acceptable
  - Typically 5-10% higher than top-1 accuracy

ViT-L/16 Results:
  - Top-1: 88.55%
  - Top-5: 98.77%
```

#### Efficiency Metrics

**Throughput (images/second):**
```
Measured on single GPU/TPU during inference
  = (batch_size) / (time_per_batch)

ViT-B/16 on single TPU v3:
  - 224×224 input: ~1000 img/sec
  - 384×384 input: ~400 img/sec

CNN Comparison (ResNet-152):
  - 224×224 input: ~1500 img/sec (faster due to fewer flops)
```

**FLOPs (Floating Point Operations):**
```
Theoretical computation cost per image:

ViT-B/16 @ 224×224:
  - Patch embedding: 196 × 768 = 150k flops
  - Transformer (12 layers): 12 × (attention + mlp)
    - Each attention: O(196²) = ~38k flops per head × 12 heads
    - Each MLP: 196 × 768 × 3072 / 768 = ~600k flops
    - Per layer: ~1.2M flops
    - Total: 12 × 1.2M = ~15M flops
  - Classification head: 768 × 1000 = ~770k flops
  - Total: ~16-17 Giga-FLOPS (multiply by 2 for matmul operations)

CNN Comparison (ResNet-152):
  - Total: ~11 Giga-FLOPS
  - ViT requires ~1.5× more computation despite better accuracy

Trade-off:
  - ViT: Higher compute, better accuracy
  - CNN: Lower compute, lower accuracy (without huge pre-training)
```

### ImageNet-1k Evaluation Protocol

**Data Split:**
```
Training Set: 1,281,167 images
  - 1000 classes
  - ~1280 images per class (varies)
  - Randomly shuffled during training
  - Standard train/val split

Validation Set: 50,000 images
  - 50 images per class
  - Standard ImageNet val set
  - Used for all official submissions

Test Set: ~100,000 images (labels withheld)
  - Not used in paper (val set results reported)
  - Available through official ImageNet challenge
```

**Evaluation Procedure:**
```python
def evaluate_on_imagenet_val():
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in val_dataloader:
        with torch.no_grad():
            logits = model(images)  # (batch_size, 1000)

        # Top-1 accuracy
        pred_top1 = torch.argmax(logits, dim=1)
        correct_top1 += (pred_top1 == labels).sum().item()

        # Top-5 accuracy
        _, pred_top5 = torch.topk(logits, 5, dim=1)
        correct_top5 += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    return {
        'top1_acc': correct_top1 / total,
        'top5_acc': correct_top5 / total
    }
```

### CIFAR-10/100 Evaluation Protocol

**CIFAR-10:**
```
Training: 50,000 images (32×32)
Validation: 10,000 images
Classes: 10
Standard metrics: Top-1 accuracy

ViT-B/32 (pre-trained on ImageNet-21k):
  - Accuracy: 98.13%
  - Demonstrates excellent transfer to small images

ViT-B/32 (trained from scratch):
  - Accuracy: 88.59%
  - Significant gap shows pre-training importance
```

**CIFAR-100:**
```
Training: 50,000 images (32×32)
Validation: 10,000 images
Classes: 100 (fine-grained)
Standard metrics: Top-1 accuracy

ViT-B/32 (pre-trained on ImageNet-21k):
  - Accuracy: 91.67%
  - Good performance on fine-grained classification

ViT-B/32 (trained from scratch):
  - Accuracy: 68.65%
  - Much larger gap on harder task (100 classes)
```

### Vision Task Adaptation Benchmark (VTAB-1k) Protocol

**Test Protocol:**
```
19 diverse vision tasks in 3 categories:

Natural:
  - CIFAR-10, CIFAR-100 (image classification)
  - Caltech-101, Flowers102, Pets, Sun397 (fine-grained classification)
  - EuroSAT (satellite images)
  - Resisc45 (remote sensing)

Specialized:
  - Retinopathy Detection (medical imaging)
  - Patch Camelyon (histopathology)

Structured:
  - DTD (texture classification)
  - SVHN (street view house numbers)
  - Malaria (medical)
  - Dmlab (3D scenes)
  - BigTransfer (multiple task-specific datasets)

Evaluation:
  - Pre-defined train (1000 examples) / val / test split
  - Report median accuracy across 3 runs
  - Fix random seed for reproducibility
  - Batch size: 32-64 (small datasets)
  - Learning rate: 1e-3 to 1e-4
  - Training steps: 2500 (roughly 10-25 epochs depending on task)
```

**ViT Results Summary (VTAB):**
```
Pre-training Dataset | Median Top-1 Accuracy | Improvement over random
ImageNet-21k        | 77.64%                | +44.5 percentage points
JFT-300M            | 81.88%                | +48.7 percentage points
Trained from scratch| ~33%                  | Very poor
```

### Few-Shot Evaluation Protocol

**Protocol:**
```
Evaluate transfer learning with limited labeled examples:
  - k-shot: Use only k training examples per class (k ∈ {1, 5, 10})
  - Train small linear classifier on frozen pre-trained features
  - Report accuracy on held-out test set

Setup:
  1. Extract [CLS] token features from frozen ViT
  2. Train softmax classifier on k examples per class
  3. Evaluate on official test set (50 images per class in ImageNet)

Example (ImageNet 5-shot):
  - Use 5 random training images per class (5,000 total)
  - Train linear layer on top of frozen features
  - Evaluate on 50 validation images per class
```

**Few-Shot Results (ViT-L/16, pre-trained on ImageNet-21k):**
```
k=1 (one-shot):  72.63% top-1 accuracy
k=5 (five-shot): 82.11% top-1 accuracy
k=10:            84.37% top-1 accuracy

Comparison with CNNs:
  - CNNs (ResNet-152): ~50% one-shot, ~70% five-shot
  - ViT significantly better at few-shot learning
  - Suggests better representation structure learned from pre-training
```

### Robustness Evaluation (Additional from Related Work)

**Not explicitly in main ViT paper, but important for understanding:**

1. **ImageNet-C (Corrupted):**
   - Test on images with various corruptions (blur, noise, weather, digital)
   - ViT more robust than ResNet to many corruptions

2. **ImageNet-R (Renditions):**
   - Test on artistic renditions of ImageNet classes
   - ViT generalizes better to distribution shift

3. **ImageNet-A (Adversarial):**
   - Challenging subset selected to fool ResNet-50
   - ViT achieves higher accuracy (~40% vs ~14% for ResNet-50)

---

## Section 10: Results Summary + Ablations

### Main Results: ViT vs CNN on ImageNet-1k

**Accuracy Comparison (Top-1):**
```
Model                      Pre-training  | ImageNet-1k Acc | Training Time
─────────────────────────────────────────────────────────────────────────
ResNet-152                 ImageNet-21k  | 82.96%          | [baseline]
EfficientNet-B7            ImageNet-21k  | 85.70%          | [baseline]
ResNet-152 (AutoAugment)   ImageNet-21k  | 83.48%          | [baseline]
─────────────────────────────────────────────────────────────────────────
ViT-B/32                   ImageNet-21k  | 84.86%          | Similar
ViT-B/16                   ImageNet-21k  | 86.45%          | ~53 hrs
ViT-L/16                   ImageNet-21k  | 87.76%          | ~178 hrs
ViT-H/14                   ImageNet-21k  | 88.16%          | ~290 hrs
─────────────────────────────────────────────────────────────────────────
ViT-L/16                   JFT-300M      | 88.55%          | ~1000 hrs
ViT-H/14                   JFT-300M      | 88.95%          | Much longer
ViT-g/14                   JFT-300M      | 90.45%          | ~2000+ hrs

Key Observations:
1. With ImageNet-21k: ViT-L outperforms all CNNs by 1-5%
2. With JFT-300M: ViT-L/H achieve 88-89% (state-of-the-art 2020)
3. Scaling: Larger models + more data = monotonic improvement
4. Smaller ViT-B/16 (86.45%) matches EfficientNet-B7 (85.70%)
```

**FLOPs vs Accuracy Trade-off:**
```
Accuracy
   ↑
90%|                          ViT-g/14
   |                    ViT-H/14
88%|               ViT-L/16
   |          ViT-B/16
86%|    ViT-B/32
   | EfficientNet-B7
84%|ResNet-152
   |
80%|
   |
   └──────────────────────────→ GFLOPs (training)
     0  10  20  30  40  50  60

Observation:
  - ViT requires ~1.5× more FLOPs than ResNet
  - But achieves 5-10% higher accuracy with pre-training
  - Trade-off: Compute for accuracy
```

### Scaling Behavior: The Power Law

**Empirical Observation:**
Accuracy improves predictably with model size and data size following a power law.

**Model Size Scaling (with fixed data):**
```
Accuracy
   ↑
88%|                    ViT-H/14
   |              ViT-L/16
86%|        ViT-B/16
   |   ViT-B/32
84%|
   |
82%|
   |
   └──────────────────────────→ Model Size (parameters)
     0  100M  200M  300M  400M  500M  600M
     ViT-B   ViT-L        ViT-H

Law: Accuracy ≈ a × (Parameters)^b
     where b ≈ 0.07 (diminishing returns)
```

**Data Size Scaling (with fixed model):**
```
Accuracy
   ↑
88%|          JFT-300M
   |    ImageNet-21k
84%|ImageNet-1k
   |
80%|
   |
   └──────────────────────────→ Data Size (images)
     1M     10M    100M   300M
     ImageNet  21k    300M

Observation:
  - ImageNet-1k: 84.86% (ViT-B/32)
  - ImageNet-21k: 86.45% (ViT-B/16)
  - JFT-300M: 88.55% (ViT-L/16)

Interpretation:
  - More data helps all models
  - ViT scales better than CNN with data
  - CNN saturates around 85%, ViT continues improving
```

### Ablation Studies

#### 1. Position Embedding Ablation

**Question:** How important are position embeddings?

**Experiment:**
```
ViT-B/16 on ImageNet-1k (pre-trained on ImageNet-21k)

Configuration          | Top-1 Accuracy
─────────────────────────────────────
Full model (with pos)  | 86.45%
No position embedding  | 81.93%
Sinusoidal position    | 85.94%  [not used in ViT]

Ablation Results:
  - Removing position embeddings: -4.52 percentage points
  - Suggests position embeddings encode important spatial information
  - But model still works reasonably well without them (81.93%)

  - Learnable > Sinusoidal: +0.51 pp
    (slight advantage of learning positions from data)
```

#### 2. Patch Size Ablation

**Question:** What patch size provides best trade-off?

**Experiment:**
```
ViT-B with different patch sizes (ImageNet-21k pre-training)

Patch Size | Tokens | Top-1 Acc | FLOPs | Throughput
─────────────────────────────────────────────────────
32×32      | 49     | 84.86%    | 0.5×  | 2.0× faster
16×16      | 196    | 86.45%    | 1.0×  | 1.0× baseline
8×8        | 784    | 87.38%    | 2.5×  | 0.4× (4× slower)

Ablation Results:
  - Smaller patches: Better accuracy but quadratic compute cost
  - 16×16 provides good balance for ViT-B/16
  - Larger models (ViT-H) use 14×14 patches (fine detail)
  - Smaller models can use 32×32 if speed is critical
```

#### 3. Transformer Depth Ablation

**Question:** How many layers do we need?

**Experiment:**
```
ViT-B with different numbers of transformer blocks

Num Layers | Parameters | Top-1 Acc | Training Time
─────────────────────────────────────────────────
4          | 60M        | 84.20%    | 40 hrs
8          | 73M        | 85.30%    | 45 hrs
12         | 86M        | 86.45%    | 53 hrs (full)
16         | 99M        | 86.42%    | 60 hrs
24         | 125M       | 86.37%    | 75 hrs

Ablation Results:
  - Diminishing returns beyond 12 layers
  - 12 layers: Good accuracy with reasonable compute
  - 24+ layers: More parameters but no significant improvement
  - Suggests ViT-B is near optimal depth for this size
```

#### 4. Attention Heads Ablation

**Question:** How many attention heads help?

**Experiment:**
```
ViT-B/16 with different numbers of attention heads

Num Heads | Head Dim | Top-1 Acc | Notes
─────────────────────────────────────
1         | 768      | 85.43%    | Single head (baseline BERT-style)
4         | 192      | 86.12%    | Still good
12        | 64       | 86.45%    | Optimal
16        | 48       | 86.38%    | Slightly worse
24        | 32       | 85.94%    | Degrades

Ablation Results:
  - 12 heads: Optimal for ViT-B/16
  - Benefits of multiple heads diminish with too many small heads
  - Head dimension seems to matter more than count
  - Maintains consistent 64-dim heads across ViT-B/L/H
```

#### 5. CLS Token vs Global Average Pooling

**Question:** Is the CLS token necessary?

**Experiment:**
```
ViT-L/16 on ImageNet-1k (pre-trained on ImageNet-21k)

Configuration           | Top-1 Accuracy | Notes
────────────────────────────────────────────
CLS token (used in ViT) | 87.76%         | Baseline
Global avg pooling      | 87.43%         | -0.33 pp
CLS + avg pooling       | 87.71%         | -0.05 pp

Ablation Results:
  - CLS token marginally better
  - Global average pooling works almost as well
  - Combination slightly worse than CLS alone

Interpretation:
  - CLS token: Allows specialization for classification
  - Global pooling: More generic, doesn't require special token
  - Either approach viable, CLS preferred for consistency with BERT
```

#### 6. Pre-training Dataset Ablation

**Question:** How much does pre-training data help?

**Experiment:**
```
ViT-B/16 fine-tuned on ImageNet-1k

Pre-training Data | Fine-tune Acc | Improvement | Training Cost
──────────────────────────────────────────────────────────────
None (random init) | 68.4%         | baseline    | -
ImageNet-1k only   | 73.6%         | +5.2 pp     | minimal
ImageNet-21k       | 86.45%        | +18.1 pp    | 14M images
JFT-300M           | 88.02%        | +19.6 pp    | 300M images
ALIGN (ViT-H)      | 90.8%*        | +22.4 pp    | 1.8B image-text pairs

*Different model size (ViT-H/14)

Key Observations:
  - Pre-training is CRITICAL for ViT
  - ImageNet-1k alone: ViT underperforms CNNs (73.6% vs 78%)
  - ImageNet-21k: ViT matches/exceeds best CNNs
  - JFT-300M: State-of-the-art (88%+)

  This is the key difference from CNNs:
  - CNN: Reasonable results with ImageNet-1k alone
  - ViT: Requires large-scale pre-training
```

#### 7. Resolution Ablation

**Question:** How does input resolution affect accuracy?

**Experiment:**
```
ViT-B/16 and ViT-L/16 at different resolutions

Model      | 224×224 | 384×384 | 512×512 | FLOPs Increase
───────────────────────────────────────────────────────────
ViT-B/16   | 86.45%  | 87.80%  | -       | 1× → 2.3×
ViT-L/16   | 87.76%  | 88.64%  | 88.89%  | 1× → 5.1×

Results:
  - Higher resolution: Consistent improvements
  - 224→384: +1.2-0.9 pp for B/L models
  - 384→512: +0.25 pp for ViT-L (diminishing returns)
  - Training trick: Pre-train at 224, fine-tune at higher res
    (interpolate position embeddings)

Trade-off:
  - Higher res: Better accuracy but 2-5× more compute
  - Sweet spot: 384×384 for good balance
```

#### 8. Feature Space Analysis

**Learned Representations (from paper analysis):**

```
Position Embedding Structure:
  - Nearest neighbor analysis of learned position embeddings
  - Nearby patches in image have similar position embeddings
  - Position embeddings learn 2D grid structure without explicit supervision

  Observation:
    cos_sim(pos_emb[i], pos_emb[j]) ∝ proximity(i, j)

Head Specialization:
  - Different attention heads focus on different aspects
  - Some heads attend to all patches (global context)
  - Some heads focus on local neighborhoods (fine details)
  - Multi-scale analysis emerges without explicit design

Spatial Resolution:
  - Early layers: Focus on local patch interactions
  - Middle layers: Build up to full receptive field
  - Late layers: Global, integrated representations
```

### Comparison with Related Architectures

**ViT vs Hybrid (ConvNet + Transformer):**
```
Model              | ImageNet-1k Acc | Parameters | FLOPs
───────────────────────────────────────────────────
DeiT-S (distilled) | 81.77%          | 22M        | 1.1G
DeiT-B (distilled) | 85.32%          | 86M        | 9.1G
ViT-B/16           | 86.45%          | 86M        | 17.6G

Hybrid (Conv + Transformer):
ResNet-50 + Trans  | 86.39%          | 104M       | 22.1G
DeiT (Transformer) | 84.56%          | 86M        | 9.1G

Result: Pure transformers > Hybrids, but hybrids more efficient
```

---

## Section 11: Practical Insights

### 10 Engineering Takeaways

1. **Pre-training is Everything**
   - Without large-scale pre-training, ViT severely underperforms CNNs
   - ViT-B from scratch: 68% accuracy
   - ViT-B with ImageNet-21k: 86% accuracy (+18 pp!)
   - Recommendation: Always pre-train on at least ImageNet-21k or equivalent

2. **Position Embeddings Matter But Aren't Critical**
   - Removing them: -4.5 percentage points
   - But model still works (81% vs 86%)
   - Learnable > sinusoidal
   - Can interpolate for different resolutions (simple hack)

3. **Larger Models Scale Better Than Larger Data Alone**
   - Both model size and data size follow power laws
   - Doubling model size: +0.5-1% accuracy gain
   - Doubling data size: +1-2% accuracy gain
   - Larger models are more sample-efficient at high-data regime

4. **Patch Size is a Critical Design Choice**
   - Smaller patches: Better accuracy but 4-8× slowdown
   - Larger patches: Faster but 1-2% accuracy loss
   - Sweet spot: 16×16 for standard 224×224 resolution
   - Recommendation: 14×14 for high-res (384×384+), 32×32 for speed-critical

5. **Weight Decay is Non-Negotiable for Pre-training**
   - Weight decay = 0.1 (10× higher than typical CNN practices)
   - Without it: Severe overfitting to pre-training classes
   - Suggests ViT learns memorization-prone patterns
   - Recommendation: Always use weight decay ~0.1 for pre-training

6. **Stochastic Depth as a Regularizer**
   - Drop probability: increases with depth (0.1 → 0.3 for ViT-H)
   - Acts as ensemble of sub-networks
   - Crucial for larger models
   - Not as important for ViT-B

7. **Batch Size and Learning Rate Correlation**
   - Large batches (4096): Need high learning rates (1e-3)
   - Small batches (128): Need low learning rates (5e-5)
   - Rule of thumb: LR ∝ sqrt(batch_size)
   - Pre-training needs large batches for stability

8. **Warmup is Essential, Cosine Annealing is Standard**
   - Linear warmup for 5-10% of training helps convergence
   - Cosine annealing from peak to ~1e-5 over remaining epochs
   - Without warmup: Unstable gradients early on
   - Alternative (less common): Exponential decay also works

9. **Resolution Transfer is Easy (with interpolation)**
   - Pre-train at 224×224, fine-tune at 384×384
   - Simply linearly interpolate position embeddings
   - No accuracy loss, cleaner than retraining
   - Enables flexible deployment (large model, variable input sizes)

10. **CLS Token is Convenient but Not Necessary**
    - CLS token: 87.76% accuracy (standard approach)
    - Global average pooling: 87.43% accuracy (only -0.33 pp)
    - CLS is more compatible with BERT-style fine-tuning
    - Either works in practice; choose based on other constraints

### 5 Gotchas (Common Pitfalls)

1. **Assuming ViT Works Like CNN Out of the Box**
   - Gotcha: Downloading pre-trained ViT and training on small dataset fails
   - Reality: ViT requires large-scale pre-training to work well
   - Fix: Either pre-train yourself (expensive) or use pre-trained weights
   - Timeline: Pre-training on ImageNet-21k takes 50-300 hours

2. **Overfitting to Fine-tuning Data**
   - Gotcha: Fine-tuning with high learning rate degrades pre-trained features
   - Reality: Must use much lower LR (1e-4 to 1e-3 vs 1e-3 for pre-training)
   - Fix: Use learning rate schedules (cosine decay or linear decay)
   - Symptom: Training loss decreases but val accuracy plateaus or decreases

3. **Ignoring Weight Decay During Fine-tuning**
   - Gotcha: Set weight_decay=0.1 during fine-tuning and accuracy drops
   - Reality: Weight decay should be 0.0 during fine-tuning
   - Reason: We want to preserve pre-trained weights, not regularize further
   - Fix: Use weight_decay=0.0 for fine-tuning, 0.1 for pre-training

4. **Not Accounting for Quadratic Attention Complexity**
   - Gotcha: Doubling input resolution (224→384) crashes out-of-memory
   - Reality: Attention is O(n²) in sequence length
   - Math: 224²/16² = 196 tokens vs 384²/14² = 753 tokens
   - 753/196 ≈ 3.8× more memory, not 2× as you'd expect
   - Fix: Use smaller batch sizes, gradient accumulation, or lower resolution

5. **Not Fine-tuning the Classification Head Properly**
   - Gotcha: Keeping pre-training MLP head for downstream task (wrong!)
   - Reality: Remove the hidden layer MLP, only keep first linear layer
   - Wrong: Linear(768→1000) + GELU + Linear(1000→1000)
   - Right: Linear(768→num_classes_downstream)
   - Fix: Discard pre-training head, initialize new head from scratch

### Overfit Plan: Debugging Checklist

**When Model Doesn't Converge on Fine-tuning Dataset:**

```
Symptom: Training loss decreases but validation accuracy stagnates

Step 1: Check Data Pipeline
  ├─ Verify no label corruption
  ├─ Check augmentation is appropriate (not too strong)
  ├─ Plot sample images to ensure they're correct
  └─ Verify class balance

Step 2: Check Hyperparameters
  ├─ Learning rate too high? → Reduce by 10×
  ├─ Learning rate too low? → Train longer (100+ epochs)
  ├─ Weight decay = 0 during fine-tuning? → Yes, should be 0
  ├─ Batch size too small? → Increase to 256+
  └─ Epochs too few? → Try 50-100 epochs

Step 3: Verify Model Configuration
  ├─ Using correct pre-trained weights?
  ├─ Classification head removed (MLP discarded)?
  ├─ New linear layer randomly initialized?
  └─ Optimizer starting from scratch (no stored states)?

Step 4: Monitor Training
  ├─ Plot learning curves (loss and accuracy over epochs)
  ├─ Check gradient norms (should be ~1.0)
  ├─ Verify batch norm statistics are updating (if used)
  └─ Confirm gradients are flowing (backprop check)

Step 5: Progressive Unfreezing
  ├─ Option A: Freeze encoder, only train classification head (10 epochs)
  ├─ Option B: Unfreeze last 2 layers, train with low LR (20 epochs)
  ├─ Option C: Unfreeze all with very low LR (50 epochs)
  └─ Monitor accuracy improvement at each stage

Step 6: Regularization Adjustments
  ├─ Increase stochastic depth (drop_path_rate)
  ├─ Increase dropout on embeddings (0.1 → 0.2)
  ├─ Try mixup or other augmentation strategies
  └─ Reduce model complexity (use ViT-B instead of ViT-L)

Step 7: Sanity Checks
  ├─ Can model overfit on tiny subset (10 images)?
  ├─ Is loss decreasing on training set?
  ├─ Are predictions changing during training?
  └─ Can you achieve >90% on training set with more epochs?

Common Fixes (in order of likelihood):
  1. Wrong learning rate (too high → oscillate, too low → no progress)
  2. Weight decay should be 0.0, not 0.1
  3. Classification head not properly replaced
  4. Data augmentation too aggressive
  5. Batch size too small (try 512+)
  6. Not enough training epochs (try 100+)
```

---

## Section 12: Minimal Reimplementation Checklist

This section provides a step-by-step guide to implementing ViT from scratch.

### Prerequisites
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
```

### Step 1: Patch Embedding Layer

```python
class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection: (patch_dim) -> embed_dim
        self.linear_proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, 224, 224)
        batch_size = x.shape[0]

        # Reshape to patches: (batch, 3, H//P, P, W//P, P)
        x = x.reshape(
            batch_size,
            3,
            self.img_size // self.patch_size,
            self.patch_size,
            self.img_size // self.patch_size,
            self.patch_size,
        )

        # Rearrange: (batch, H//P, W//P, 3, P, P)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.patch_dim)

        # Linear projection: (batch, num_patches, embed_dim)
        x = self.linear_proj(x)
        return x
```

### Step 2: Multi-Head Self-Attention

```python
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V: (batch, seq_len, 3*embed_dim)
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        # Rearrange: (3, batch, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention: Q @ K^T / sqrt(d_k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply to V
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

### Step 3: Transformer Encoder Block

```python
class TransformerBlock(nn.Module):
    """Transformer encoder block with attention and MLP."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # Attention
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=dropout,
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        # Random binary mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device)
        mask = random_tensor > self.drop_prob

        # Scale to maintain expected value
        scale = 1.0 / (1 - self.drop_prob)
        return x * mask.float() * scale
```

### Step 4: Vision Transformer Architecture

```python
class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Stochastic depth schedule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, num_layers)
        ]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=dpr[i],
            )
            for i in range(num_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Position embeddings and class token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Layer norms and linear layers
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, 224, 224)
        batch_size = x.shape[0]

        # Patch embedding: (batch, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend class token: (batch, num_patches+1, embed_dim)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Layer norm and classification
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token

        # Classification head
        logits = self.head(x)
        return logits
```

### Step 5: Training Loop

```python
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
```

### Step 6: Quick Start Example

```python
# Create model
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    mlp_dim=3072,
    dropout=0.1,
    drop_path_rate=0.1,
).to("cuda")

# Optimizer with proper settings
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,      # Lower for fine-tuning
    betas=(0.9, 0.999),
    weight_decay=0.0,  # Zero for fine-tuning!
)

# Learning rate schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-5,
)

criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, "cuda")
    val_acc = evaluate(model, val_loader, "cuda")
    scheduler.step()

    print(f"Epoch {epoch}: Loss={train_loss:.3f}, ValAcc={val_acc:.3f}")
```

### Implementation Checklist

```
Essential Components:
  ☐ PatchEmbedding: Divide images into patches and project
  ☐ MultiHeadAttention: Q,K,V projections, softmax attention
  ☐ TransformerBlock: Norm + Attention + MLP + residual
  ☐ DropPath: Stochastic depth for regularization
  ☐ VisionTransformer: Main model combining all components
  ☐ CLS token: Learnable parameter prepended to patches
  ☐ Position embeddings: Learnable, same shape as sequence

Training Configuration:
  ☐ Optimizer: Adam with beta1=0.9, beta2=0.999
  ☐ Learning rate: 1e-4 to 1e-3 (lower for fine-tuning)
  ☐ Learning rate schedule: Warmup + cosine annealing
  ☐ Weight decay: 0.1 for pre-training, 0.0 for fine-tuning
  ☐ Batch size: 512+ (larger is better, helps convergence)
  ☐ Gradient clipping: max_norm=1.0 (prevents exploding gradients)

Initialization:
  ☐ Patch embedding weights: He initialization or Xavier
  ☐ CLS token: Truncated normal (std=0.02)
  ☐ Position embeddings: Truncated normal (std=0.02)
  ☐ Attention/MLP weights: Truncated normal (std=0.02)
  ☐ Layer norm: Initialize to identity (weight=1, bias=0)

Data Pipeline:
  ☐ Image normalization: ImageNet mean/std
  ☐ Augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter
  ☐ Resolution: 224×224 for standard, 384×384 for fine-tuning
  ☐ Batch size: 512+ for distributed training

Common Mistakes to Avoid:
  ✗ Using high learning rate (1e-3) for fine-tuning
  ✗ Setting weight_decay=0.1 during fine-tuning
  ✗ Not using pre-trained weights
  ✗ Forgetting to remove pre-training MLP head
  ✗ Using small batch sizes (< 256)
  ✗ Not using layer norm before attention/MLP
  ✗ Initializing new head parameters too large
  ✗ Not using cosine annealing schedule
```

---

## Summary

This 12-section summary provides a comprehensive technical understanding of Vision Transformers:

1. **Overview**: Key novelty of pure transformer architecture for vision
2. **Problem Setup**: Image classification with shape tracking
3. **Geometry**: Patches as tokens, position embeddings, CLS token
4. **Architecture**: Detailed diagram and model variants (ViT-B/L/H)
5. **Forward Pass**: Complete shape-annotated pseudocode
6. **Losses & Heads**: Cross-entropy, pre-training vs fine-tuning strategies
7. **Data Pipeline**: Datasets, augmentation, resolution handling
8. **Training**: Optimizer, learning rate schedule, regularization
9. **Evaluation**: Protocols for ImageNet, CIFAR, VTAB, few-shot
10. **Results**: Scaling laws, ablations, comparisons with CNNs
11. **Insights**: Engineering takeaways, gotchas, debugging checklist
12. **Implementation**: Minimal PyTorch code to build ViT from scratch

The paper demonstrates that transformers, originally developed for NLP, can be successfully applied to vision when given sufficient pre-training data. The key insights are: (1) pre-training is critical, (2) larger models and data scale well together, and (3) simple architectural choices (learnable embeddings, CLS token, standard transformer blocks) work surprisingly well.

