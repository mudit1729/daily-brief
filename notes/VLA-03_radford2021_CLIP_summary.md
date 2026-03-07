# Learning Transferable Visual Models From Natural Language Supervision

**CLIP: Radford et al. (2021)** — ICML 2021, PMLR 139:8748-8763

---

<!-- AUDIT_NOTE_START -->
> Audit note (March 7, 2026): This block is the canonical, paper-backed teaching summary for this file. If later sections give exact tensor shapes, hyperparameters, deployment numbers, or pseudocode details that the paper does not explicitly report, treat them as explanatory reconstructions rather than direct paper claims.

## Paper-Backed Teaching Corrections
- **What the paper actually does:** learns image representations by contrastively matching images and text on 400M internet image-text pairs, then uses natural-language prompts for zero-shot transfer.
- **Why it matters for VLA:** CLIP is an enabling semantic backbone for later robotics work, but it is not itself a robot control paper.
- **What you should understand:** contrastive pretraining, prompt-based zero-shot classification, and why language supervision gives broader visual semantics than fixed class labels.
- **Important correction:** any robotics-specific action or deployment detail elsewhere in the file is downstream interpretation, not a claim made in the CLIP paper.
<!-- AUDIT_NOTE_END -->

## 1. One-Page Overview

| Aspect | Details |
|--------|---------|
| **Title** | Learning Transferable Visual Models From Natural Language Supervision |
| **Authors** | Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever |
| **Venue** | International Conference on Machine Learning (ICML) 2021, PMLR 139:8748-8763 |
| **arXiv** | 2103.00020 (Mar 26, 2021) |
| **Affiliation** | OpenAI |
| **Code & Models** | [github.com/openai/CLIP](https://github.com/openai/CLIP); Hugging Face; multiple variants |

### Problem Setting

Radford et al. tackle **visual recognition from natural language supervision** at scale. Instead of training on labeled datasets like ImageNet, CLIP learns visual representations from 400 million freely available (image, caption) pairs from the internet. The key insight is that predicting which caption goes with which image is an efficient and scalable way to learn state-of-the-art image representations.

### Inputs/Outputs

- **Input**:
  - Images (any resolution, normalized to 224×224 or 336×336)
  - Text captions (natural language descriptions)
- **Output**:
  - Image embeddings (512D or higher)
  - Text embeddings (same dimension)
  - Similarity scores for image-caption ranking

### Key Novelty Bullets

1. **Contrastive Language-Image Pre-training**: Train image and text encoders jointly to maximize similarity of correct (image, text) pairs and minimize similarity of incorrect pairs. No class labels required.

2. **Internet-Scale Dataset (400M pairs)**: Leverage 400 million weakly-paired (image, caption) pairs from the web, orders of magnitude larger than ImageNet (1.28M labeled images).

3. **Zero-Shot Transfer**: CLIP models can classify images without any task-specific fine-tuning by converting class names to text prompts and comparing image embeddings.

4. **Multiple Model Variants**: Provides trade-offs between accuracy and efficiency (ResNet-50, ResNet-101, ViT-B/32, ViT-B/16, ViT-L/14) for various deployment scenarios.

5. **Robustness and Generalization**: CLIP shows better generalization to new domains, robust to distribution shift, and performs competitively with task-specific supervised models.

### If You Only Remember 3 Things

1. **Contrastive learning at scale works**: Training image and text encoders to maximize similarity of matching pairs (and minimize for non-matching) scales to 400M image-caption pairs and produces highly transferable representations.

2. **Language supervision is powerful**: Natural language provides richer, more diverse supervision than class labels. A single image can be described many ways, capturing multiple visual concepts.

3. **Zero-shot capability is transformative**: By treating classification as text-image matching, CLIP can recognize any visual concept expressible in language without fine-tuning—enabling true zero-shot transfer.

---

## 2. Problem Setup and Outputs

### Training Objective

**Contrastive Loss**: For a batch of N image-caption pairs:

```
Given: Batch of N images {I₁, I₂, ..., Iₙ}
       Batch of N captions {T₁, T₂, ..., Tₙ}
       Pairing: (I_i, T_i) for i=1..N

Goal: Maximize similarity of diagonal entries (correct pairs)
      Minimize similarity of off-diagonal entries (incorrect pairs)
```

### Input Specification

| Input | Dimensions | Format | Notes |
|-------|-----------|--------|-------|
| **Image** | (H, W, 3) or (224, 224, 3) | uint8, [0,255] | Variable resolution; resized to 224×224 or 336×336 for training |
| **Caption/Text** | Variable length | Text string or tokens | 1–75 tokens; mean ~11 tokens, max ~77 tokens |
| **Batch** | N=32,768 | Paired (image, caption) | Very large batch size crucial for contrastive learning |

### Output Specification

| Output | Dimensions | Range | Semantics |
|--------|-----------|-------|-----------|
| **Image Embedding** | 512–1024D | float32 | Normalized L2 norm; unit length |
| **Text Embedding** | 512–1024D (same) | float32 | Normalized L2 norm; unit length |
| **Similarity Matrix** | (N, N) | [−1, 1] (dot product) | logits_ij = <I_i, T_j> × τ (temperature-scaled) |
| **Probability Matrix** | (N, N) | [0, 1] | softmax(logits) across rows and columns |
| **Contrastive Loss** | scalar | [0, ∞) | Average cross-entropy loss for image→text and text→image matching |

---

## 3. Coordinate Frames and Geometry

### Embedding Space Geometry

**Normalized Embeddings** (L2 normalization):
```
For any embedding vector v ∈ ℝᵈ:
  v_norm = v / ||v||₂

All embeddings lie on the surface of a d-dimensional unit hypersphere.

Dot product between normalized vectors = cosine similarity:
  cos(θ) = <v₁_norm, v₂_norm> = ||v₁|| · ||v₂|| · cos(θ_angle)
                                = cos(θ_angle)  (since ||v|| = 1)
```

### Similarity Metric

**Cosine Similarity**:
```
sim(I, T) = <φ_image(I), φ_text(T)>

where φ_image: ℝ^{H×W×3} → ℝᵈ (image encoder, outputs normalized)
      φ_text: Text → ℝᵈ (text encoder, outputs normalized)

Range: [-1, 1]
  +1: identical vectors (perfect match)
   0: orthogonal
  -1: opposite vectors (dissimilar)
```

### Batch Similarity Matrix

For batch of N pairs:
```
Similarity matrix S ∈ ℝ^{N×N}:
  S_ij = <φ_image(I_i), φ_text(T_j)>

Ideal (correct pairing):
  Diagonal entries S_ii should be high (≈ 1.0)
  Off-diagonal entries S_ij (i≠j) should be low (≈ 0.0 or negative)

Visual:
  ┌                    ┐
  │ +0.95  -0.10  -0.20 │
  │ -0.08  +0.98  -0.15 │
  │ -0.12  -0.11  +0.96 │
  └                    ┘
```

### Temperature Parameter τ

Scales similarity scores before softmax:
```
Logit_ij = S_ij / τ

where τ ∈ (0, 1) is a learnable temperature parameter

Effect:
  τ → 0 (small): Sharp softmax, high sensitivity to similarity differences
  τ → 1 (large): Soft softmax, low sensitivity

Typical value: τ ≈ 0.07 (learned, typically initialized to ~0.07)
Constraint: Clipped to avoid extreme values (e.g., τ ∈ [0.001, 1.0])
```

---

## 4. Architecture Deep Dive

### Overall System Block Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      CLIP ARCHITECTURE                       │
└──────────────────────────────────────────────────────────────┘

                        Training Phase

Image Batch (N, 3, H, W)              Text Batch (N, seq_len)
        │                                      │
        ▼                                      ▼
┌──────────────────┐              ┌──────────────────────┐
│ Image Encoder    │              │ Text Encoder         │
│ (ResNet or ViT)  │              │ (Transformer)        │
│ φ_I: I → I_emb  │              │ φ_T: Text → T_emb   │
└────────┬─────────┘              └──────────┬───────────┘
         │                                   │
         │ I_emb ∈ ℝ^{N×D}                 │ T_emb ∈ ℝ^{N×D}
         │ (unnormalized)                   │ (unnormalized)
         ▼                                   ▼
┌──────────────────┐              ┌──────────────────────┐
│ L2 Normalize     │              │ L2 Normalize         │
│ I_norm = I/||I|| │              │ T_norm = T/||T||     │
└────────┬─────────┘              └──────────┬───────────┘
         │                                   │
         └───────────────┬───────────────────┘
                         ▼
              ┌──────────────────────┐
              │ Compute Similarity   │
              │ Matrix S = I @ T^T   │
              │ S ∈ ℝ^{N×N}          │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Scale by Temperature │
              │ Logits = S / τ       │
              │ τ is learned         │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Cross-Entropy Loss   │
              │ L = (L_I2T + L_T2I)/2│
              └──────────────────────┘

                      Inference Phase

Image → [Image Encoder] → [L2 Norm] → Image Embedding (512D, unit norm)
                                             │
                                             └─→ Compare with text embeddings
                                                 (compute similarity)
```

### Encoder Architecture Variants

#### Vision Encoders

| Model | Architecture | Input Size | Hidden Dim | Output Dim | Parameters |
|-------|-------------|-----------|-----------|-----------|-----------|
| **ResNet-50** | 4-stage ResNet | 224×224 | Various | 2048 | ~25.6M |
| **ResNet-101** | 4-stage ResNet (larger) | 224×224 | Various | 2048 | ~44.5M |
| **ViT-B/32** | Vision Transformer, patch 32 | 224×224 | 768 | 512 | ~86M |
| **ViT-B/16** | Vision Transformer, patch 16 | 224×224 | 768 | 512 | ~86M |
| **ViT-L/14** | Vision Transformer, patch 14 | 336×336 | 1024 | 768 | ~304M |

**Vision Transformer (ViT) Details** (ViT-L/14 example):
```
Input: 336×336 RGB image
    ↓
Patch Embedding: Divide into 14×14 patches (24×24 pixels each)
    ↓ 784 patches × 1024 dimensions
Linear Projection to embedding dimension (1024)
    ↓
Transformer Encoder: 24 layers, 16 attention heads, MLP hidden 4096
    ↓
Global Average Pooling (or [CLS] token)
    ↓ 1024D
Linear Projection to embedding dimension (768D in this example)
    ↓ 768D
L2 Normalize
    ↓ (unit norm)
Output: 768D image embedding
```

#### Text Encoder

```
Input: Text sequence (e.g., "a dog wearing a hat")
    ↓
Byte-Pair Encoding (BPE): Tokenize to ~49,152 token vocabulary
    ↓ Token indices
Embedding Layer: Each token → 512D embedding
    ↓ (seq_len, 512)
Transformer Encoder: 12 layers, 8 attention heads, feedforward 2048
    ↓
Extract [EOS] token representation (end-of-sequence token)
    ↓ 512D
L2 Normalize
    ↓ (unit norm)
Output: 512D text embedding
```

### Module Architecture Table

| Module | Input | Output | Parameters | Notes |
|--------|-------|--------|-----------|-------|
| **Vision Encoder** | (B, 3, H, W) | (B, D_v) | Varies (11M–304M) | Pre-trained; frozen or fine-tuned |
| **Text Encoder** | (B, seq_len) | (B, D_t) | ~63M | Transformer-based |
| **Temperature Parameter** | Scalar | Scalar | 1 | Learned; initialized ~log(1/0.07) ≈ 2.66 |
| **Vision Projection** | (B, D_v) | (B, D_shared) | D_v × D_shared | Optional; projects to shared embedding space |
| **Text Projection** | (B, D_t) | (B, D_shared) | D_t × D_shared | Optional; projects to shared embedding space |
| **L2 Normalization** | (B, D) | (B, D) | 0 | No learnable parameters |

---

## 5. Forward Pass Pseudocode

```python
def forward_pass_clip(images, texts, image_encoder, text_encoder, tau=0.07):
    """
    Forward pass for CLIP training.

    Args:
        images: (B, 3, H, W) float32 image batch
        texts: (B, max_seq_len) int token indices
        image_encoder: ResNet or ViT model
        text_encoder: Transformer model
        tau: Temperature parameter (learned during training)

    Returns:
        loss: Scalar contrastive loss
        logits_per_image: (B, B) similarity matrix (scaled)
        logits_per_text: (B, B) similarity matrix (transposed)
    """

    # === Image Encoding ===
    image_features = image_encoder(images)  # (B, D_v)

    # Optional projection to shared space
    if has_vision_projection:
        image_features = vision_projection(image_features)  # (B, D_shared)

    # L2 normalization
    image_features_norm = image_features / (norm(image_features, axis=1, keepdims=True) + eps)
    # Shape: (B, D)

    # === Text Encoding ===
    text_features = text_encoder(texts)  # (B, D_t)

    # Optional projection
    if has_text_projection:
        text_features = text_projection(text_features)  # (B, D_shared)

    # L2 normalization
    text_features_norm = text_features / (norm(text_features, axis=1, keepdims=True) + eps)
    # Shape: (B, D)

    # === Compute Similarity Matrix ===
    # Dot product of all images with all texts
    logits_per_image = image_features_norm @ text_features_norm.T  # (B, B)
    # logits_per_image[i, j] = similarity(image_i, text_j)

    # Transpose for text-to-image matching
    logits_per_text = logits_per_image.T  # (B, B)

    # === Scale by Temperature ===
    # Temperature parameter τ is learned via backprop
    logits_per_image_scaled = logits_per_image / tau  # (B, B)
    logits_per_text_scaled = logits_per_text / tau    # (B, B)

    # === Compute Loss ===
    # Image-to-text matching: maximize diagonal of logits_per_image
    labels = arange(B)  # (B,) ground truth: image i matches text i
    loss_i2t = cross_entropy_loss(logits_per_image_scaled, labels)

    # Text-to-image matching: maximize diagonal of logits_per_text
    loss_t2i = cross_entropy_loss(logits_per_text_scaled, labels)

    # Average
    loss = (loss_i2t + loss_t2i) / 2

    return {
        'loss': loss,
        'logits_per_image': logits_per_image_scaled,
        'logits_per_text': logits_per_text_scaled,
        'image_features': image_features_norm,
        'text_features': text_features_norm,
    }


def zero_shot_classification(image, class_names, image_encoder, text_encoder, tau):
    """
    Perform zero-shot classification by converting class names to text.

    Args:
        image: (3, H, W) single image
        class_names: list of class name strings, e.g., ["dog", "cat", "bird"]
        image_encoder: Trained CLIP image encoder
        text_encoder: Trained CLIP text encoder
        tau: Temperature parameter

    Returns:
        class_probabilities: (num_classes,) softmax probabilities
    """

    # Encode image
    image_batch = image[None, ...]  # (1, 3, H, W)
    image_features = image_encoder(image_batch)  # (1, D)
    image_features_norm = image_features / norm(image_features)  # (1, D)

    # Convert class names to text prompts
    # Option 1: Direct class name
    # text_prompts = class_names  # ["dog", "cat", "bird"]

    # Option 2: Template-based (often works better)
    text_prompts = [f"a photo of a {name}" for name in class_names]
    # e.g., ["a photo of a dog", "a photo of a cat", "a photo of a bird"]

    # Tokenize text prompts
    text_tokens = [tokenize(prompt) for prompt in text_prompts]  # List of token arrays
    text_tokens_batch = stack(text_tokens)  # (num_classes, seq_len)

    # Encode text
    text_features = text_encoder(text_tokens_batch)  # (num_classes, D)
    text_features_norm = text_features / norm(text_features, axis=1, keepdims=True)

    # Compute similarity
    logits = (image_features_norm @ text_features_norm.T) / tau  # (1, num_classes)
    logits = logits.squeeze(0)  # (num_classes,)

    # Softmax to get probabilities
    probabilities = softmax(logits)  # (num_classes,)

    return probabilities
```

---

## 6. Heads, Targets, and Losses

### Output Representation

CLIP does **not** have explicit "heads" like traditional supervised models. Instead, it produces embeddings that serve as the basis for similarity computation:

| Component | Output Shape | Semantics |
|-----------|-------------|-----------|
| **Image Embedding** | (B, D) | Normalized feature vector per image |
| **Text Embedding** | (B, D) | Normalized feature vector per text |
| **Similarity Matrix** | (B, B) | Pairwise cosine similarities |
| **Scaled Logits** | (B, B) | Temperature-scaled similarities (input to softmax) |

### Training Targets

**Contrastive Labels** (implicit):
```
For batch of N images paired with N captions:
  Correct pairing: (I_i, T_i) for i = 1..N
  Incorrect pairing: (I_i, T_j) for i ≠ j

Target for image-to-text matching:
  labels_i2t = [0, 1, 2, ..., N-1]  (identity mapping)
  Interpreted as: "For image i, the correct caption is at index i"

Target for text-to-image matching:
  labels_t2i = [0, 1, 2, ..., N-1]  (identity mapping)
  Interpreted as: "For text j, the correct image is at index j"
```

### Loss Functions

**Contrastive Loss** (symmetric):
```
For image-to-text matching:
  loss_i2t = CrossEntropy(logits_per_image, labels)
  = - (1/B) * Σ_{i=0}^{B-1} log(softmax(logits_per_image[i])[i])
  = - (1/B) * Σ_{i=0}^{B-1} log(exp(logits_per_image[i,i]) / Σ_k exp(logits_per_image[i,k]))

For text-to-image matching:
  loss_t2i = CrossEntropy(logits_per_text, labels)
  = - (1/B) * Σ_{j=0}^{B-1} log(softmax(logits_per_text[j])[j])

Total loss (symmetric contrastive):
  L = (loss_i2t + loss_t2i) / 2
```

**Detailed Example** (B=4):
```
Image embeddings: I_1, I_2, I_3, I_4 (each 512D, normalized)
Text embeddings:  T_1, T_2, T_3, T_4  (each 512D, normalized)

Similarity matrix S = I @ T^T:
  ┌           ┐
  │ 0.95 -0.1 -0.2 -0.15 │  ← image 0 similarity with all texts
  │-0.08 0.98 -0.15 -0.1 │  ← image 1 similarity with all texts
  │-0.12 -0.11 0.96 -0.18│  ← image 2 similarity with all texts
  │-0.1  -0.08 -0.2  0.97│  ← image 3 similarity with all texts
  └           ┘

Target labels: [0, 1, 2, 3] (diagonal entries should be correct)

Softmax for image 0 (row 0):
  softmax([0.95, -0.1, -0.2, -0.15] / τ)
  = softmax([13.6, -1.4, -2.9, -2.1])  (if τ=0.07)
  ≈ [0.999, 0.00, 0.00, 0.00]

Cross-entropy loss for image 0:
  -log(0.999) ≈ 0.001  (very small, good!)

If model is poorly trained:
  Softmax([0.1, 0.2, 0.0, -0.1] / τ)
  = softmax([1.4, 2.9, 0, -1.4])
  ≈ [0.23, 0.62, 0.14, 0.01]

Cross-entropy loss:
  -log(0.23) ≈ 1.47  (large, incentivizes learning)
```

---

## 7. Data Pipeline and Augmentations

### Dataset Composition

| Aspect | Details |
|--------|---------|
| **Dataset Name** | WIT (WebImageText) |
| **Total Pairs** | 400 million (image, caption) pairs |
| **Source** | Publicly available internet images and alt-text |
| **Collection** | Curated from diverse domains: web pages, documents, products, scenes |
| **Language Diversity** | English captions; diverse vocabulary and writing styles |
| **Image Diversity** | Natural images, graphics, diagrams, photographs; varied resolutions |
| **Distribution** | Long-tailed; some concepts more common than others |

### Data Filtering & Curation

```python
def filter_and_prepare_wit_dataset(raw_dataset):
    """
    Prepare WIT dataset for CLIP training.
    """

    # 1. Remove near-duplicate images & captions
    #    Use perceptual hashing (dhash, phash) to detect duplicates
    unique_dataset = deduplicate(raw_dataset)

    # 2. Length filtering on captions
    #    Remove captions that are too short (< 3 words) or too long (> 100 words)
    #    Typical range: 5–50 words
    filtered = [
        (img, caption) for (img, caption) in unique_dataset
        if 3 <= len(caption.split()) <= 100
    ]

    # 3. Language detection
    #    Keep only English captions (using langdetect or similar)
    english_only = [
        (img, caption) for (img, caption) in filtered
        if detect_language(caption) == 'en'
    ]

    # 4. Optional: Keyword-based filtering
    #    Remove certain content (e.g., explicit, hate speech)
    #    Use keyword lists or trained classifiers

    # 5. Resolution filtering (optional)
    #    Discard very low-resolution images (e.g., < 100 pixels on short edge)
    min_resolution = 100
    good_resolution = [
        (img, caption) for (img, caption) in english_only
        if min(img.shape[:2]) >= min_resolution
    ]

    return good_resolution  # ~400M pairs after filtering
```

### Data Preprocessing & Augmentations

| Augmentation | Probability | Parameters | Purpose |
|--------------|-------------|-----------|---------|
| **Image Resizing** | 1.0 (always) | Resize to 224×224 or 336×336 | Standardize input size |
| **Random Crop** | 1.0 | Center or random crop from resized image | Reduce spatial bias |
| **Horizontal Flip** | 0.5 | Flip image left-right | Robustness to orientation |
| **Color Jitter** | 0.5 | Brightness 0.2, Contrast 0.2, Saturation 0.2, Hue 0.1 | Lighting & color variations |
| **Grayscale Conversion** | 0.1 | Convert to grayscale | Robustness to color loss |
| **Gaussian Blur** | 0.1 | σ ∈ [0.1, 2.0] | Motion blur or noise simulation |
| **Text Tokenization** | 1.0 (always) | Byte-pair encoding; truncate to 77 tokens | Standardize text input |
| **Text Augmentation** | 0.1 | Synonym replacement, word dropout | Minor text variation (not recommended; can break semantics) |

### Batch Construction

```python
def create_training_batch(dataset, batch_size=32768, num_workers=16):
    """
    Construct batch for CLIP training.

    Note: CLIP uses very large batch size (32,768) across multiple GPUs.
    Each GPU sees a subset; contrastive loss aggregates across GPUs.

    Args:
        dataset: WIT dataset (400M pairs)
        batch_size: 32,768 (total across all GPUs)
        num_workers: Number of data loading processes

    Returns:
        batch: {
            'images': (B, 3, H, W) float32,
            'texts': (B, max_seq_len) int,
        }
    """

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,  # Pin memory for fast GPU transfer
        sampler=DistributedSampler(dataset, shuffle=True),  # For multi-GPU
    )

    return dataloader
```

---

## 8. Training Pipeline

### Training Configuration

| Aspect | Details |
|--------|---------|
| **Total Training Time** | 32 epochs over 400M pairs (~12.8B samples) |
| **Batch Size** | 32,768 (distributed across 256 V100 GPUs or equivalent) |
| **Effective Batch Size per GPU** | 128 images + 128 texts (with gradient accumulation) |
| **Learning Rate Schedule** | Cosine annealing: 5e-4 → 1e-6 over 32 epochs |
| **Optimizer** | Adam with decoupled weight decay (AdamW) |
| **Total Training Steps** | ~13B samples / 32,768 batch = ~400K steps |
| **Training Duration** | ~12 days on 256 V100 GPUs |
| **Mixed Precision** | FP16 activations & gradients; FP32 weight updates (PyTorch AMP) |

### Hyperparameter Table

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| **Learning Rate (initial)** | 5e-4 | Cosine decay to 1e-6 |
| **Learning Rate Schedule** | Cosine annealing | 32 epochs; no warmup used |
| **Optimizer** | Adam (AdamW) | β₁ = 0.9, β₂ = 0.999, ε = 1e-8 |
| **Weight Decay** | Varies | 0.1 for ViT models, smaller for ResNets |
| **Batch Size** | 32,768 | Very large batch crucial for contrastive learning |
| **Temperature τ** | Learned | Initialized log(1/0.07) ≈ 2.66; clipped to [log(1/100), log(1/0.07)] |
| **Gradient Clipping** | 1.0 | Clip by global norm to stabilize training |
| **Dropout** | Varies | 0.0 (not used) in original; optional in some versions |
| **Label Smoothing** | 0.0 | Not used; contrastive loss already provides implicit smoothing |
| **Epochs** | 32 | Total passes over 400M pairs |
| **Warmup** | 0 steps | No learning rate warmup in original paper |

### Training Loop

```python
def train_clip_epoch(image_encoder, text_encoder, dataloader, device, epoch):
    """
    Train CLIP for one epoch.

    Args:
        image_encoder: Vision model (ResNet or ViT)
        text_encoder: Text Transformer
        dataloader: DataLoader with (image, text) pairs
        device: 'cuda' or 'cpu' (distributed across GPUs)
        epoch: Current epoch number

    Returns:
        loss_avg: Average loss over epoch
    """

    image_encoder.train()
    text_encoder.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        images = batch['images'].to(device)  # (B, 3, H, W)
        texts = batch['texts'].to(device)    # (B, seq_len)

        optimizer.zero_grad()

        # Forward pass
        output = forward_pass_clip(
            images, texts,
            image_encoder, text_encoder,
            tau=temperature
        )

        loss = output['loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(image_encoder.parameters()) + list(text_encoder.parameters()),
            max_norm=1.0
        )

        # Optimizer step
        optimizer.step()

        # Update temperature parameter (learned)
        temperature.backward(retain_graph=True)
        temperature_optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def train_clip_distributed(
    image_encoder, text_encoder,
    train_loader, epochs=32, lr=5e-4,
    num_gpus=256, device='cuda'
):
    """
    Training loop with distributed data parallelism (DDP).

    Args:
        image_encoder: Model to train
        text_encoder: Model to train
        train_loader: DataLoader (handles distributed sampling)
        epochs: Total epochs
        lr: Initial learning rate
        num_gpus: Number of GPUs
        device: Device type
    """

    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(image_encoder.parameters()) + list(text_encoder.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
    )

    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * epochs,
        eta_min=1e-6
    )

    # Temperature parameter (learned)
    log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))
    temperature_optimizer = torch.optim.SGD([log_temperature], lr=0.1)

    # Distributed data parallel
    image_encoder = DistributedDataParallel(image_encoder)
    text_encoder = DistributedDataParallel(text_encoder)

    # Training loop
    for epoch in range(epochs):
        train_loss = train_clip_epoch(
            image_encoder, text_encoder,
            train_loader, device, epoch
        )

        scheduler.step()

        # Evaluation on validation set (every N epochs)
        if epoch % 5 == 0:
            val_loss = evaluate_clip(
                image_encoder, text_encoder,
                val_loader, device
            )
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Checkpoint
        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'image_encoder_state': image_encoder.state_dict(),
                'text_encoder_state': text_encoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'temperature': log_temperature.item(),
            }, f"checkpoint_epoch_{epoch}.pt")
```

---

## 9. Dataset + Evaluation Protocol

### Dataset: WIT (WebImageText)

| Aspect | Details |
|--------|---------|
| **Size** | 400 million image-caption pairs |
| **Source** | Publicly available image-alt text from web |
| **Diversity** | Covers objects, animals, scenes, concepts, abstract ideas |
| **Language** | English captions; natural, unstructured language |
| **Quality** | Variable; no manual curation (all automatic) |
| **Distribution** | Long-tailed (some concepts frequent, others rare) |
| **Availability** | Not released publicly (due to licensing); similar datasets (e.g., LAION-400M) are available |

### Evaluation Benchmarks

| Benchmark | Task | # Classes | Evaluation | Notes |
|-----------|------|-----------|-----------|-------|
| **ImageNet** | Image classification | 1,000 | Top-1 accuracy | Standard supervised baseline |
| **ImageNet-A/R/V** | Distribution shift | 1,000 each | Top-1 accuracy | Robustness to visual variations |
| **Caltech-101** | Fine-grained classification | 101 | Top-1 accuracy | Different domain (objects) |
| **Pets** | Pet breed classification | 37 | Top-1 accuracy | Fine-grained; small dataset |
| **MNIST/CIFAR** | Low-resolution images | 10/100 | Top-1 accuracy | Out-of-distribution for pre-training |
| **Retinal Disease** | Medical imaging | 4 | Top-1 accuracy | Domain-specific; tests generalization |

### Evaluation Protocol

```python
def evaluate_clip_zero_shot(image_encoder, text_encoder, test_dataset):
    """
    Zero-shot evaluation: classify images without any task-specific training.

    Args:
        image_encoder: Trained CLIP image encoder
        text_encoder: Trained CLIP text encoder
        test_dataset: {
            'images': array of test images,
            'labels': ground truth class indices,
            'class_names': list of class name strings,
        }

    Returns:
        accuracy: Float in [0, 1]
    """

    image_encoder.eval()
    text_encoder.eval()

    class_names = test_dataset['class_names']
    images = test_dataset['images']
    labels_gt = test_dataset['labels']

    # Template-based prompts (often works better than direct class names)
    templates = [
        'a photo of a {}',
        'a {}',
        'a {} object',
        'the {}',
    ]

    text_prompts = []
    for class_name in class_names:
        for template in templates:
            text_prompts.append(template.format(class_name))

    # Encode all text prompts
    with torch.no_grad():
        text_tokens = [tokenize(prompt) for prompt in text_prompts]
        text_tokens_batch = torch.stack(text_tokens)  # (num_prompts, seq_len)
        text_features = text_encoder(text_tokens_batch)  # (num_prompts, D)
        text_features_norm = text_features / torch.norm(text_features, dim=1, keepdim=True)

        # Average embeddings per class (across templates)
        text_features_per_class = text_features_norm.reshape(
            len(class_names), len(templates), -1
        ).mean(dim=1)  # (num_classes, D)

    # Classify each image
    correct = 0
    total = 0

    for image in images:
        with torch.no_grad():
            image_batch = image[None, ...]  # (1, 3, H, W)
            image_features = image_encoder(image_batch)
            image_features_norm = image_features / torch.norm(image_features)

            # Compute similarity with all class embeddings
            similarities = image_features_norm @ text_features_per_class.T  # (1, num_classes)
            predicted_class = torch.argmax(similarities, dim=1).item()

            if predicted_class == labels_gt[total]:
                correct += 1

        total += 1

    accuracy = correct / total
    return accuracy


def evaluate_clip_linear_probe(image_encoder, text_encoder, train_loader, test_loader):
    """
    Linear probe evaluation: freeze CLIP encoders, train a linear classifier on top.

    This tests the quality of frozen CLIP representations.

    Args:
        image_encoder: Trained CLIP image encoder (frozen)
        text_encoder: Not used for linear probe
        train_loader: Training data for linear classifier
        test_loader: Test data

    Returns:
        test_accuracy: Float in [0, 1]
    """

    image_encoder.eval()

    # Extract features from all training images
    train_features = []
    train_labels = []

    for batch in train_loader:
        images = batch['images']
        labels = batch['labels']

        with torch.no_grad():
            features = image_encoder(images)  # (B, D)
            features_norm = features / torch.norm(features, dim=1, keepdim=True)

        train_features.append(features_norm.cpu().numpy())
        train_labels.append(labels.cpu().numpy())

    train_features = np.concatenate(train_features)  # (N, D)
    train_labels = np.concatenate(train_labels)      # (N,)

    # Train linear classifier (logistic regression)
    linear_classifier = LogisticRegression(max_iter=1000)
    linear_classifier.fit(train_features, train_labels)

    # Evaluate on test set
    test_features = []
    test_labels = []

    for batch in test_loader:
        images = batch['images']
        labels = batch['labels']

        with torch.no_grad():
            features = image_encoder(images)
            features_norm = features / torch.norm(features, dim=1, keepdim=True)

        test_features.append(features_norm.cpu().numpy())
        test_labels.append(labels.cpu().numpy())

    test_features = np.concatenate(test_features)
    test_labels = np.concatenate(test_labels)

    test_accuracy = linear_classifier.score(test_features, test_labels)
    return test_accuracy
```

---

## 10. Results Summary + Ablations

### Main Results on ImageNet

| Model | ImageNet Top-1 (%) | Zero-Shot | Linear Probe | Notes |
|-------|-----------------|-----------|-------------|-------|
| **CLIP (ViT-L/14)** | **76.2** | 76.2% | 88.5% | Best CLIP model |
| **CLIP (ViT-B/32)** | 63.3 | 63.3% | 80.2% | Smaller, faster |
| **CLIP (ResNet-50)** | 59.5 | 59.5% | 77.1% | Weaker than ViT |
| **ResNet-50 (supervised)** | 76.1 | N/A | N/A | Standard ImageNet baseline |
| **ResNet-50 (ViT-L/14 CLIP)** | 76.2 | — | — | CLIP matches supervised baseline |

### Key Results

1. **CLIP-ViT-L/14 matches ResNet-50 supervised learning on ImageNet** (76.2% vs 76.1%)
2. **Zero-shot CLIP generalizes across diverse datasets** without task-specific fine-tuning
3. **Linear probe performance (88.5%) >> zero-shot (76.2%)** indicates frozen representations are highly useful

### Generalization & Robustness Results

| Benchmark | CLIP Zero-Shot | ResNet-50 Supervised | Δ |
|-----------|----------------|---------------------|-----|
| **ImageNet** | 76.2 | 76.1 | +0.1 |
| **ImageNet-A** | 73.9 | 67.0 | +6.9 |
| **ImageNet-R** | 88.2 | 84.7 | +3.5 |
| **ImageNet-V** | 84.3 | 80.6 | +3.7 |
| **Caltech-101** | 94.0 | 91.3 | +2.7 |
| **Pets** | 91.4 | 92.7 | −1.3 |
| **CIFAR-10** | 95.3 | 98.5 | −3.2 |
| **STL-10** | 97.3 | 99.2 | −1.9 |

**Interpretation**: CLIP is more robust on distribution-shifted versions of ImageNet but less robust on low-resolution images (CIFAR-10). This is because CLIP trains on web images (typically high-res) unlike ImageNet (standardized 224×224).

### Ablation Studies

| Ablation | ImageNet Top-1 (%) | Δ | Finding |
|----------|------------------|----|----|
| **Full ViT-L/14** | 76.2 | — | Baseline |
| **Smaller batch (16K vs 32K)** | 74.8 | −1.4 | Batch size crucial for contrastive learning |
| **No L2 normalization** | 72.1 | −4.1 | Normalization essential; otherwise unstable |
| **Large τ (fixed at 0.5)** | 71.5 | −4.7 | Temperature parameter important; learnable better |
| **Text encoder frozen** | 64.2 | −12.0 | Text encoder must be trained jointly |
| **No image augmentation** | 73.8 | −2.4 | Augmentation helps; contrastive learning robust |
| **Random text captions** | 35.4 | −40.8 | Caption quality essential |
| **Only 10M pairs (vs 400M)** | 58.0 | −18.2 | Data scale critical for scaling |

---

## 11. Practical Insights

### Engineering Takeaways

1. **Large Batch Size is Critical**: CLIP uses 32K batch size. With smaller batches (<4K), performance degrades significantly. This is because contrastive loss requires many negative examples.

2. **L2 Normalization is Essential**: Always normalize both image and text embeddings to unit length. Without it, training becomes unstable and embeddings don't capture similarity well.

3. **Temperature Parameter τ is Learnable**: Initialize to log(1/0.07) ≈ 2.66. Learning τ automatically adapts to batch statistics and model capacity. Fixing τ is suboptimal.

4. **Scaling Laws Apply**: CLIP follows predictable scaling trends. Larger models (ViT-L vs ViT-B) and more data (400M vs 10M pairs) monotonically improve performance.

5. **Text Encoder Quality Matters**: Don't freeze the text encoder; train it jointly with the image encoder. Text encoder learns task-specific vocabulary and phrasing.

6. **Image Augmentation Works**: Even though CLIP trains on diverse internet data, adding standard augmentations (crop, color jitter, flip) provides modest improvements.

7. **Template Ensembling Improves Zero-Shot**: Instead of using bare class names ("dog"), use templates ("a photo of a dog", "a {}") and average embeddings. 2–3% improvement is typical.

8. **Mixed Precision Training Enables Scale**: FP16 activations + gradient accumulation reduce memory by 2–3×, enabling larger batches or models.

9. **Multi-GPU Synchronization**: Contrastive loss requires embeddings from all GPUs. Use DistributedDataParallel carefully; embeddings are gathered across GPUs before computing loss.

10. **Data Diversity Beats Data Size**: 400M diverse internet images > 1.28M ImageNet-style labels. Unlabeled diverse data with weak pairing is more valuable than highly curated small datasets.

### Gotchas & Pitfalls

1. **Embedding Dimension Must Match**: If projecting image and text embeddings to a shared space, ensure dimensions match. Mismatched dimensions silently break the model.

2. **Tokenization Vocabulary Size**: CLIP uses BPE with 49K tokens. Using a different tokenizer or vocabulary (e.g., 30K tokens) changes behavior; normalize across codebases.

3. **Temperature Clipping**: If τ is not clipped, it can become extremely small (τ → 0), making logits huge and causing numerical instability. Clip to [log(1/100), log(1/0.07)].

4. **Batch Ordering Bias**: In distributed training, if per-GPU batches are not well-shuffled, certain image-text pairs might be always together or always separated, biasing contrastive learning.

5. **Evaluation Mode Matters**: Text encoder behavior differs between train (dropout active) and eval (dropout inactive). Always use `model.eval()` for evaluation.

6. **Class Name Ambiguity**: "bass" can mean a fish or a musical instrument. Ambiguous class names hurt zero-shot performance. Use template-based prompts to disambiguate.

7. **Caption Quality**: CLIP quality depends directly on caption quality. Low-quality alt-text (e.g., "image.jpg") or mismatched captions significantly hurt performance.

8. **Transfer Learning from Supervised**: CLIP pre-training is different from supervised ImageNet pre-training. Cold-starting from CLIP requires different fine-tuning schedules.

9. **Vision Transformer Scaling**: ViT models are more compute-hungry than ResNets at inference time. Patch size (32 vs 16 vs 14) significantly affects speed/accuracy trade-off.

10. **Synthetic Data Limitations**: CLIP trained on internet data doesn't transfer well to synthetic/rendered images (large domain gap). Requires domain-specific fine-tuning or mixed training.

### Tiny-Subset Overfit Plan

```python
def debug_overfit_clip_on_1000_pairs():
    """
    Verify CLIP implementation on tiny dataset.
    """

    # 1. Create minimal 1,000-pair dataset (images + captions)
    tiny_dataset = create_synthetic_pairs(
        num_pairs=1000,
        image_size=(224, 224),
        caption_length=(5, 20),
    )

    # 2. Disable augmentation & regularization
    image_encoder.train()
    text_encoder.train()
    learning_rate = 1e-2  # Aggressive
    weight_decay = 0.0

    # 3. Small batch size (32, not 32K)
    batch_size = 32

    # 4. Train to overfit
    for epoch in range(100):
        total_loss = 0.0

        for batch_idx, batch in enumerate(DataLoader(tiny_dataset, batch_size=batch_size)):
            images = batch['images'].to(device)
            texts = batch['texts'].to(device)

            output = forward_pass_clip(images, texts, image_encoder, text_encoder)
            loss = output['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(DataLoader(tiny_dataset, batch_size=batch_size))

        if avg_loss < 0.01:
            print(f"✓ Overfit successful at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    # 5. Verify embeddings on training set
    with torch.no_grad():
        for idx in range(min(10, len(tiny_dataset))):
            image = tiny_dataset[idx]['image'][None, ...]
            text = tiny_dataset[idx]['text'][None, ...]

            image_feat = image_encoder(image)
            text_feat = text_encoder(text)

            image_feat_norm = image_feat / torch.norm(image_feat)
            text_feat_norm = text_feat / torch.norm(text_feat)

            sim = (image_feat_norm @ text_feat_norm.T).item()
            print(f"Pair {idx}: Similarity = {sim:.4f} (should be ~0.95–1.0 if overfit)")

    # 6. Diagnostics
    # - Check loss monotonically decreases?
    # - Check similarities on training pairs > 0.95?
    # - Check gradients non-zero in both encoders?
    # - Check for NaN/Inf in embeddings?
```

---

## 12. Minimal Reimplementation Checklist

### Core Components

- [ ] **Vision Encoder (ResNet or ViT)**
  - [ ] Load pre-trained or initialize from scratch
  - [ ] Remove classification head; extract intermediate features (2048D for ResNet-50, 1024D for ViT-L)
  - [ ] Optional: Projection layer to shared embedding dimension (512D or 256D)
  - [ ] Test: `forward((1, 3, 224, 224)) → (1, 512)`

- [ ] **Text Encoder (Transformer)**
  - [ ] Byte-pair encoding tokenizer (49K vocabulary)
  - [ ] Token embedding layer (512D)
  - [ ] Transformer: 12 layers, 8 heads, feedforward 2048D
  - [ ] Extract [EOS] token or use [CLS] token
  - [ ] Optional: Projection to shared embedding dimension
  - [ ] Test: `forward((1, 77)) → (1, 512)` (77 = max tokens)

- [ ] **L2 Normalization**
  - [ ] Normalize image embeddings: `image_features / ||image_features||_2`
  - [ ] Normalize text embeddings: `text_features / ||text_features||_2`
  - [ ] All embeddings should have unit norm (L2 norm = 1.0)

- [ ] **Similarity Matrix Computation**
  - [ ] Compute dot product: `S = image_features @ text_features.T`
  - [ ] Result shape: (B, B)
  - [ ] Each entry S_ij is cosine similarity (range [-1, 1])

- [ ] **Temperature Parameter τ**
  - [ ] Initialize: `log_tau = torch.log(torch.tensor(0.07))`
  - [ ] Learnable parameter: `torch.nn.Parameter(log_tau)`
  - [ ] Scale logits: `logits = S / exp(log_tau)`
  - [ ] Optionally clip: `log_tau = clip(log_tau, log(1/100), log(1/0.07))`

- [ ] **Contrastive Loss**
  - [ ] Create labels: `labels = torch.arange(B)`
  - [ ] Image-to-text loss: `cross_entropy(logits_per_image, labels)`
  - [ ] Text-to-image loss: `cross_entropy(logits_per_text, labels.T)`
  - [ ] Average: `loss = (loss_i2t + loss_t2i) / 2`

- [ ] **Optimizer & Scheduler**
  - [ ] Optimizer: Adam or AdamW
  - [ ] Learning rate: 5e-4 initial (cosine decay to 1e-6)
  - [ ] Scheduler: `CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-6)`
  - [ ] Gradient clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)`

- [ ] **Distributed Training (Optional but Recommended)**
  - [ ] DistributedDataParallel for multi-GPU
  - [ ] Gather embeddings across GPUs before computing loss
  - [ ] Synchronized batch norm (SyncBatchNorm)

- [ ] **Zero-Shot Evaluation**
  - [ ] Convert class names to text prompts (with templates)
  - [ ] Encode all prompts with text encoder
  - [ ] For each test image, encode with image encoder
  - [ ] Compute similarity; argmax to get predicted class
  - [ ] Compute accuracy

### Minimal Configuration

```yaml
# config.yaml for minimal CLIP training
architecture:
  image_encoder: "resnet50"
  text_encoder: "transformer_12L_512D"
  embedding_dim: 512

training:
  batch_size: 32768  # or smaller for testing (32–256)
  num_epochs: 32
  learning_rate: 5.0e-4
  warmup_steps: 0
  lr_scheduler: "cosine"
  optimizer: "adamw"
  weight_decay: 0.1
  gradient_clipping: 1.0

data:
  dataset: "wit"  # or custom dataset
  num_workers: 16
  prefetch_factor: 2
  image_size: 224

losses:
  contrastive: true
  temperature_init: 0.07
  temperature_learnable: true

hardware:
  num_gpus: 256  # or 1 for development
  mixed_precision: true
  distributed: true
```

### Quick Validation Checklist

```
□ Image encoder outputs correct shape:
    forward((B, 3, 224, 224)) → (B, 512)

□ Text encoder outputs correct shape:
    forward((B, 77)) → (B, 512)

□ L2 normalization works:
    ||image_features||_2 ≈ 1.0 for all samples

□ Similarity matrix correct shape:
    S = I @ T.T → (B, B)

□ Loss decreases over 100 iterations on tiny (32-pair) dataset

□ Contrastive loss is symmetric:
    loss_i2t ≈ loss_t2i (both computed, then averaged)

□ No NaN/Inf gradients (check first backward pass)

□ Inference time < 1 second for batch of 32 images on single GPU

□ Zero-shot classification works:
    Generate prompts → Encode → Compute similarity → Predict class

□ Model checkpointing:
    Save/load state_dict without errors

□ Distributed training (if applicable):
    All-gather embeddings works across GPUs
```

---

## References

- **[Radford et al. 2021]** Learning Transferable Visual Models From Natural Language Supervision. *International Conference on Machine Learning (ICML)*, PMLR 139:8748-8763, July 2021.
- **arXiv**: 2103.00020
- **GitHub**: [openai/CLIP](https://github.com/openai/CLIP)
- **Blog Post**: [OpenAI CLIP](https://openai.com/index/clip/)
- **Related Work**: Vision-language pre-training; contrastive learning; zero-shot transfer
