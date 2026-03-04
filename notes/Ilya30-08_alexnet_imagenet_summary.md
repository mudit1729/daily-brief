# ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
**Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012)**

---

## Section 1: One-Page Overview

### Metadata
- **Title**: ImageNet Classification with Deep Convolutional Neural Networks
- **Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- **Published**: NIPS 2012 (Advances in Neural Information Processing Systems)
- **Venue Impact**: Won ILSVRC 2012 competition
- **Citation Count**: 100,000+ citations (one of most influential papers in ML)

### Task Solved
- **Problem**: Classify 1,280,000 ImageNet training images into 1,000 object categories
- **Challenge**: Images are 256×256 RGB, high resolution, extreme intra-class variance
- **Prior SOTA**: ~26% top-5 error (traditional methods, 2011)
- **AlexNet Result**: **18.9% top-5 error, 39.7% top-1 error** (55% relative improvement)

### Key Novelties - "3 Things to Remember"
1. **ReLU Activation**: Non-saturating non-linearity (ReLU: max(0, x)) converges 6× faster than tanh, enables deeper networks to train successfully
2. **GPU Implementation**: Custom CUDA kernels for convolution operations reduced training time from ~weeks (CPU) to ~days (GPU), enabling 60M parameter networks
3. **Dropout Regularization**: Stochastic neuron silencing (p=0.5 in FC layers) reduced overfitting without requiring smaller models, proved essential for generalization

### Additional Key Insights
- Data augmentation (crops, flips, color perturbations) provided ~2% improvement
- Local Response Normalization (LRN) improved ILSVRC top-1 by 1.4%
- Ensemble of 7 networks achieved 15.4% top-5 error
- Architectural depth (8 layers) critical: 5-layer ablation ≈22% top-5 error

---

## Section 2: Problem Setup and Outputs

### ImageNet Dataset Specification
- **ILSVRC-2010 Training Set**: 1,200,000 images, 1,000 categories, variable resolution
- **ILSVRC-2010 Test Set**: 150,000 images, held-out labels
- **ILSVRC-2012 Training Set**: 1,280,000 images
- **Input Standardization**: Rescale to 256×256, then crop to 227×227 at training, center crop at test
- **Color Space**: RGB (3 channels), no conversion to grayscale

### Output Specification
- **Output Layer**: 1,000-way softmax output
- **Output Shape**: (batch_size, 1000)
- **Each Element**: Probability distribution over 1,000 classes
- **Decision Rule**: argmax(softmax(logits))

### Image Dimension Details
| Stage | Height | Width | Channels | Notes |
|-------|--------|-------|----------|-------|
| Original | Variable | Variable | 3 | Aspect-ratio preserved rescale |
| Rescaled | 256 | 256 | 3 | Smallest dimension = 256 |
| Cropped (Train) | 227 | 227 | 3 | Random 227×227 crop |
| Cropped (Test) | 227 | 227 | 3 | Center 227×227 crop |
| Mean-Subtracted | 227 | 227 | 3 | Per-pixel mean from train set subtracted |

### Tensor Shapes Through Network
```
Input:        (N, 3, 227, 227)
Conv1 out:    (N, 96, 55, 55)
Pool1 out:    (N, 96, 27, 27)
Conv2 out:    (N, 256, 27, 27)
Pool2 out:    (N, 256, 13, 13)
Conv3 out:    (N, 384, 13, 13)
Conv4 out:    (N, 384, 13, 13)
Conv5 out:    (N, 256, 13, 13)
Pool5 out:    (N, 256, 6, 6)
Flatten:      (N, 9216)
FC6 out:      (N, 4096)
FC7 out:      (N, 4096)
FC8 out:      (N, 1000)
Softmax out:  (N, 1000) [probabilities sum to 1]
```

### Batch Size & Memory
- **Training Batch Size**: 128 images
- **Per-GPU Memory**: ~2.8 GB (GTX 580, 1.5GB VRAM each, 2 GPUs)
- **Full Batch Tensor**: (128, 3, 227, 227) = ~80 MB raw, more with intermediate activations

---

## Section 3: Coordinate Frames and Geometry

### Image Space Coordinate Frame
```
Image Coordinate System:
- Origin: Top-left corner (0, 0)
- X-axis: Points right (column index)
- Y-axis: Points down (row index)
- Range: [0, 256) or [0, 227) depending on stage
- Aspect ratio: Normalized to square, then cropped

Cropping Geometry (Training):
  Original 256×256 image
  Random (y_offset, x_offset) ~ Uniform([0, 30))
  Crop window: [y_offset:y_offset+227, x_offset:x_offset+227]
  Result: 227×227 random patch
```

### Feature Map Geometry in Convolutional Layers

**Conv1 (kernel=11, stride=4, padding=0)**
```
Input:  227×227
Kernel: 11×11
Stride: 4
Output spatial dim: floor((227 - 11) / 4) + 1 = 55
Output feature map: 55×55, depth=96
Receptive field: 11×11 of input image
```

**Pool1 (kernel=3, stride=2, padding=0)**
```
Input: 55×55
Kernel: 3×3
Stride: 2
Output: floor((55 - 3) / 2) + 1 = 27
Output: 27×27, depth=96
```

**Conv2 (kernel=5, stride=1, padding=2)**
```
Input: 27×27
Kernel: 5×5
Padding: 2 (explicit in paper)
Output: floor((27 - 5 + 2*2) / 1) + 1 = 27
Output: 27×27, depth=256
Receptive field from input: 5×5 (local connectivity)
```

**Pool2 (kernel=3, stride=2)**
```
Input: 27×27
Output: floor((27 - 3) / 2) + 1 = 13
Output: 13×13, depth=256
```

**Conv3-5 (kernel=3, stride=1, padding=1)**
```
All maintain spatial dimension:
Input: 13×13
Kernel: 3×3
Padding: 1
Output: floor((13 - 3 + 2*1) / 1) + 1 = 13
Output: 13×13, varying depths
```

**Pool5 (kernel=3, stride=2)**
```
Input: 13×13
Output: floor((13 - 3) / 2) + 1 = 6
Output: 6×6×256 = 9,216 neurons flattened to FC layer
```

### Receptive Field Growth
```
Layer      | Kernel | Stride | Padding | Receptive Field (from input)
-----------|--------|--------|---------|-------------------------
Input      | -      | -      | -       | 1×1
Conv1      | 11     | 4      | 0       | 11×11
Pool1      | 3      | 2      | 0       | 15×15
Conv2      | 5      | 1      | 2       | 23×23
Pool2      | 3      | 2      | 0       | 27×27
Conv3      | 3      | 1      | 1       | 31×31
Conv4      | 3      | 1      | 1       | 35×35
Conv5      | 3      | 1      | 1       | 39×39
Pool5      | 3      | 2      | 0       | 43×43
```

Each Conv5 neuron "sees" a 43×43 patch of the original 227×227 image.

---

## Section 4: Architecture Deep Dive

### ASCII Block Diagram (Split Across 2 GPUs)

```
GPU 0 and GPU 1 process independently with selected parameter sharing:

                        INPUT (227×227×3)
                              |
                    [CONV1: 11×11×3, 96 filters]
                    Stride=4, ReLU, LRN
                         OUTPUT: 55×55×96
                              |
                         [Split into GPU 0 and GPU 1]
                    48 filters GPU 0 | 48 filters GPU 1
                              |
                    [MAX-POOL1: 3×3, stride=2]
                         OUTPUT: 27×27×96
                              |
                    [CONV2: 5×5, 256 filters] ← GPU to GPU connections
                    ReLU, LRN
                         OUTPUT: 27×27×256
                              |
                    [Split: 128 GPU 0 | 128 GPU 1]
                    [MAX-POOL2: 3×3, stride=2]
                         OUTPUT: 13×13×256
                              |
                    [CONV3: 3×3, 384 filters]
                    ReLU (no LRN)
                         OUTPUT: 13×13×384
                              |
                    [Split: 192 GPU 0 | 192 GPU 1]
                    [CONV4: 3×3, 384 filters] ← GPU to GPU
                    ReLU (no LRN)
                         OUTPUT: 13×13×384
                              |
                    [CONV5: 3×3, 256 filters] ← GPU to GPU
                    ReLU (no LRN)
                         OUTPUT: 13×13×256
                              |
                    [Split: 128 GPU 0 | 128 GPU 1]
                    [MAX-POOL5: 3×3, stride=2]
                         OUTPUT: 6×6×256
                              |
                      FLATTEN: 9216
                              |
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   [FC6: 4096] Dropout(p=0.5)          [FC6: 4096] Dropout(p=0.5)
        │                                           │
   [FC7: 4096] Dropout(p=0.5)          [FC7: 4096] Dropout(p=0.5)
        │                                           │
   [FC8: 1000] (50% GPU0)              [FC8: 1000] (50% GPU1)
        └─────────────────────┬─────────────────────┘
                              |
                    [Softmax(1000)]
                              |
                   OUTPUT: 1000-way probability
```

### Layer-by-Layer Architecture Table

| Layer | Type | Input Shape | Kernel | Stride | Padding | Output Filters | Output Shape | Params | ReLU | LRN | Dropout |
|-------|------|-------------|--------|--------|---------|----------------|--------------|--------|------|-----|---------|
| 1 | Conv | 227×227×3 | 11×11 | 4 | 0 | 96 | 55×55×96 | 34,944 | ✓ | ✓ | - |
| 2 | MaxPool | 55×55×96 | 3×3 | 2 | 0 | 96 | 27×27×96 | 0 | - | - | - |
| 3 | Conv | 27×27×96 | 5×5 | 1 | 2 | 256 | 27×27×256 | 614,656 | ✓ | ✓ | - |
| 4 | MaxPool | 27×27×256 | 3×3 | 2 | 0 | 256 | 13×13×256 | 0 | - | - | - |
| 5 | Conv | 13×13×256 | 3×3 | 1 | 1 | 384 | 13×13×384 | 884,992 | ✓ | - | - |
| 6 | Conv | 13×13×384 | 3×3 | 1 | 1 | 384 | 13×13×384 | 1,327,104 | ✓ | - | - |
| 7 | Conv | 13×13×384 | 3×3 | 1 | 1 | 256 | 13×13×256 | 884,992 | ✓ | - | - |
| 8 | MaxPool | 13×13×256 | 3×3 | 2 | 0 | 256 | 6×6×256 | 0 | - | - | - |
| 9 | Flatten | 6×6×256 | - | - | - | 9216 | 9216 | 0 | - | - | - |
| 10 | FC | 9216 | - | - | - | 4096 | 4096 | 37,748,736 | ✓ | - | ✓ (0.5) |
| 11 | FC | 4096 | - | - | - | 4096 | 4096 | 16,777,216 | ✓ | - | ✓ (0.5) |
| 12 | FC | 4096 | - | - | - | 1000 | 1000 | 4,096,000 | - | - | - |

**Total Parameters**: 60,965,224 (~60M)
- Conv layers: ~7.3M
- FC layers: ~58.6M (weighted toward FC6, FC7)

### GPU Split Strategy
```
GPU 0 processes:
  - Conv1-2 filters: 48 (first half)
  - Pool layers: All processing
  - Conv3-5: Split channels
  - FC layers: Split neurons

GPU 1 processes:
  - Conv1-2 filters: 48 (second half)
  - Conv3-5: Split channels
  - FC layers: Split neurons

Cross-GPU Communication:
  - Conv2 input: GPU 0 sends outputs to GPU 1, vice versa
  - Conv4 input: Selective pooling from both GPUs
  - Conv5 input: All feature maps concatenated from both GPUs
  - FC layers: Each GPU processes independent 50% of neurons
```

---

## Section 5: Forward Pass Pseudocode

```python
def forward_pass(x):
    """
    Forward pass with shape annotations.
    x: (128, 3, 227, 227) - batch of 128 images
    """

    # ==================== CONV LAYER 1 ====================
    # Input: (128, 3, 227, 227)
    # Kernel: 11×11×3×96 (96 filters, each 11×11×3)
    # Output formula: floor((227 - 11) / 4) + 1 = 55

    conv1 = Conv2d(
        in_channels=3,
        out_channels=96,
        kernel_size=11,
        stride=4,
        padding=0
    )(x)  # Output: (128, 96, 55, 55)

    relu1 = ReLU()(conv1)  # (128, 96, 55, 55)

    lrn1 = LocalResponseNorm(
        size=5,
        alpha=1e-4,
        beta=0.75,
        k=1.0
    )(relu1)  # (128, 96, 55, 55)

    # ==================== POOL LAYER 1 ====================
    # Input: (128, 96, 55, 55)
    # Kernel: 3×3, Stride: 2
    # Output: floor((55 - 3) / 2) + 1 = 27

    pool1 = MaxPool2d(
        kernel_size=3,
        stride=2,
        padding=0
    )(lrn1)  # Output: (128, 96, 27, 27)

    # GPU SPLIT: Split channel dimension
    # GPU 0: pool1[:, :48, :, :]  shape (128, 48, 27, 27)
    # GPU 1: pool1[:, 48:, :, :]  shape (128, 48, 27, 27)

    # ==================== CONV LAYER 2 ====================
    # Input: (128, 96, 27, 27) [concatenated from both GPUs]
    # Kernel: 5×5×96×256 with padding=2
    # Output: floor((27 - 5 + 2*2) / 1) + 1 = 27

    conv2 = Conv2d(
        in_channels=96,
        out_channels=256,
        kernel_size=5,
        stride=1,
        padding=2
    )(pool1)  # Output: (128, 256, 27, 27)

    relu2 = ReLU()(conv2)  # (128, 256, 27, 27)

    lrn2 = LocalResponseNorm(
        size=5,
        alpha=1e-4,
        beta=0.75,
        k=1.0
    )(relu2)  # (128, 256, 27, 27)

    # ==================== POOL LAYER 2 ====================
    # Input: (128, 256, 27, 27)
    # Output: floor((27 - 3) / 2) + 1 = 13

    pool2 = MaxPool2d(
        kernel_size=3,
        stride=2,
        padding=0
    )(lrn2)  # Output: (128, 256, 13, 13)

    # ==================== CONV LAYER 3 ====================
    # Input: (128, 256, 13, 13)
    # Kernel: 3×3×256×384 with padding=1
    # Output: floor((13 - 3 + 2*1) / 1) + 1 = 13

    conv3 = Conv2d(
        in_channels=256,
        out_channels=384,
        kernel_size=3,
        stride=1,
        padding=1
    )(pool2)  # Output: (128, 384, 13, 13)

    relu3 = ReLU()(conv3)  # (128, 384, 13, 13)
    # NO LRN after conv3

    # ==================== CONV LAYER 4 ====================
    # Input: (128, 384, 13, 13)
    # Kernel: 3×3×384×384 with padding=1
    # Output: floor((13 - 3 + 2*1) / 1) + 1 = 13

    conv4 = Conv2d(
        in_channels=384,
        out_channels=384,
        kernel_size=3,
        stride=1,
        padding=1
    )(relu3)  # Output: (128, 384, 13, 13)

    relu4 = ReLU()(conv4)  # (128, 384, 13, 13)
    # NO LRN after conv4

    # ==================== CONV LAYER 5 ====================
    # Input: (128, 384, 13, 13)
    # Kernel: 3×3×384×256 with padding=1
    # Output: floor((13 - 3 + 2*1) / 1) + 1 = 13

    conv5 = Conv2d(
        in_channels=384,
        out_channels=256,
        kernel_size=3,
        stride=1,
        padding=1
    )(relu4)  # Output: (128, 256, 13, 13)

    relu5 = ReLU()(conv5)  # (128, 256, 13, 13)
    # NO LRN after conv5

    # ==================== POOL LAYER 5 ====================
    # Input: (128, 256, 13, 13)
    # Output: floor((13 - 3) / 2) + 1 = 6

    pool5 = MaxPool2d(
        kernel_size=3,
        stride=2,
        padding=0
    )(relu5)  # Output: (128, 256, 6, 6)

    # ==================== FLATTEN ====================
    # Reshape from (128, 256, 6, 6) to (128, 9216)

    flattened = pool5.reshape(128, -1)  # (128, 9216)

    # ==================== FC LAYER 6 ====================
    # Input: (128, 9216)
    # Weight matrix: (9216, 4096)
    # Output: (128, 4096)

    fc6_out = Linear(
        in_features=9216,
        out_features=4096
    )(flattened)  # (128, 4096)

    relu6 = ReLU()(fc6_out)  # (128, 4096)

    # DROPOUT during training
    dropout6 = Dropout(p=0.5)(relu6)  # (128, 4096)
    # Note: During inference, dropout is disabled (p=0)

    # ==================== FC LAYER 7 ====================
    # Input: (128, 4096)
    # Weight matrix: (4096, 4096)
    # Output: (128, 4096)

    fc7_out = Linear(
        in_features=4096,
        out_features=4096
    )(dropout6)  # (128, 4096)

    relu7 = ReLU()(fc7_out)  # (128, 4096)

    # DROPOUT during training
    dropout7 = Dropout(p=0.5)(relu7)  # (128, 4096)

    # ==================== FC LAYER 8 (Output) ====================
    # Input: (128, 4096)
    # Weight matrix: (4096, 1000)
    # Output: (128, 1000) - logits

    fc8_logits = Linear(
        in_features=4096,
        out_features=1000
    )(dropout7)  # (128, 1000)

    # ==================== SOFTMAX ====================
    # Converts logits to probabilities

    output_probs = Softmax(dim=1)(fc8_logits)  # (128, 1000)

    return {
        'logits': fc8_logits,           # (128, 1000) - pre-softmax
        'probabilities': output_probs,  # (128, 1000) - normalized
        'predictions': argmax(output_probs, dim=1)  # (128,) - class indices
    }


# ==================== MEMORY FOOTPRINT (per batch) ====================
# Input:              128 × 3 × 227 × 227 × 4 bytes = 80 MB
# Conv1 output:       128 × 96 × 55 × 55 × 4 bytes = 92 MB
# Conv2 output:       128 × 256 × 27 × 27 × 4 bytes = 95 MB
# Conv3 output:       128 × 384 × 13 × 13 × 4 bytes = 33 MB
# Pool5 output:       128 × 256 × 6 × 6 × 4 bytes = 7 MB
# FC6 output:         128 × 4096 × 4 bytes = 2 MB
# FC7 output:         128 × 4096 × 4 bytes = 2 MB
# FC8 output:         128 × 1000 × 4 bytes = 0.5 MB
# ────────────────────────────────────────────────────
# Total activations ≈ 312 MB (combined with parameters, ~1.5GB per GPU)
```

---

## Section 6: Heads, Targets, and Losses

### Output Head Design

The final output layer (FC8) produces logits:
```
Logits shape: (batch_size, 1000)
z_i = sum_j (W_ij * h_j + b_i) for i in [0, 999]

where:
  W: (4096, 1000) weight matrix
  h: (batch_size, 4096) hidden features from FC7
  b: (1000,) bias vector
```

### Softmax Activation

```
Softmax(z) = [softmax(z_0), softmax(z_1), ..., softmax(z_999)]

softmax(z_i) = exp(z_i) / sum_k exp(z_k)

Properties:
  - All outputs in (0, 1)
  - Sum of all outputs = 1 (valid probability distribution)
  - Numerically stable with log-sum-exp trick in practice
```

### Target Format

**Ground Truth Format**: One-hot encoded class labels
```
For an image with true class c:
target = [0, 0, ..., 1, ..., 0]  (1 at position c, 0s elsewhere)
target shape: (batch_size, 1000)

Example: If true class is 42:
target = [0, 0, ..., 1, 0, ..., 0]  # 1 at index 42
         [0  1  42  999]
```

Alternative representation (used during loss computation):
```
target_classes = [c_0, c_1, ..., c_127]  # Class indices
target_classes shape: (batch_size,) with values in [0, 999]
```

### Cross-Entropy Loss

**Mathematical Formulation**:
```
CrossEntropy(p, target) = -sum_i (target_i * log(p_i))

For one-hot target at class c:
L = -log(p_c)

where p_c = softmax(z)_c = exp(z_c) / sum_k exp(z_k)

Interpretation:
  - If p_c = 0.9: loss = -log(0.9) ≈ 0.105 (small, good)
  - If p_c = 0.1: loss = -log(0.1) ≈ 2.303 (large, bad)
  - If p_c = 0.5: loss = -log(0.5) ≈ 0.693 (medium)
```

**Per-Batch Loss**:
```
L_batch = (1/batch_size) * sum_b=0^127 L_b
        = (1/128) * sum_b (-log(p_{b,c_b}))
```

**Gradient with respect to logits**:
```
dL/dz_i = (softmax(z)_i - target_i)

This simplifies to:
- dL/dz_c = (p_c - 1) where c is the true class
- dL/dz_j = p_j for j ≠ c

Physical interpretation: Gradients directly equal the "error" in the softmax output.
```

### Top-1 and Top-5 Error Metrics

**Top-1 Error** (Primary metric in ILSVRC-2012):
```
Top1_Error = (number of misclassified samples) / total_samples

Prediction is correct if:
  argmax_i softmax(z)_i == true_class

AlexNet achieved: 39.7% top-1 error on ILSVRC-2012 test set
```

**Top-5 Error** (Tolerates multiple guesses):
```
Top5_Error = (samples not in top-5 predictions) / total_samples

Prediction is correct if true_class in top_5_indices

where top_5_indices = argsort(softmax(z))[-5:]
                    = the 5 indices with highest softmax probabilities

AlexNet achieved: 18.9% top-5 error on ILSVRC-2012 test set
```

**Comparison**:
```
Top-5 error is always ≤ Top-1 error because:
  - Top-1: Need exact match (1 chance out of 1000)
  - Top-5: Need match within 5 guesses (5 chances out of 1000)

AlexNet improvement over ILSVRC 2011 winner:
  - Top-5 error: 26% → 18.9% (38% relative improvement)
  - Top-1 error: ~44% → 39.7% (10% relative improvement)
```

### Loss Computation Example

```
Batch of 4 images, 3 classes:

Image 0: true class = 0
Image 1: true class = 2
Image 2: true class = 1
Image 3: true class = 0

Targets (one-hot):
target = [[1, 0, 0],
          [0, 0, 1],
          [0, 1, 0],
          [1, 0, 0]]

Logits from FC8:
logits = [[2.0, 1.0, 0.1],
          [0.5, 1.0, 2.0],
          [1.0, 2.0, 0.5],
          [1.5, 0.8, 0.5]]

Softmax computation:
p_0 = exp([2.0, 1.0, 0.1]) / sum = [0.659, 0.242, 0.099]
p_1 = exp([0.5, 1.0, 2.0]) / sum = [0.090, 0.244, 0.665]
p_2 = exp([1.0, 2.0, 0.5]) / sum = [0.181, 0.665, 0.154]
p_3 = exp([1.5, 0.8, 0.5]) / sum = [0.542, 0.286, 0.172]

Cross-entropy losses:
L_0 = -log(0.659) = 0.415
L_1 = -log(0.665) = 0.407
L_2 = -log(0.665) = 0.407
L_3 = -log(0.542) = 0.613

Batch loss:
L_batch = (0.415 + 0.407 + 0.407 + 0.613) / 4 = 0.461

Top-1 accuracy = 3/4 = 75% → Top-1 error = 25%
Top-5 accuracy = 4/4 = 100% (all classes correct) → Top-5 error = 0%
```

---

## Section 7: Data Pipeline and Augmentations

### Image Loading and Preprocessing

**Stage 1: Image I/O**
```
Raw JPEG → numpy array
Shape transformation from variable to fixed:
  1. Load JPEG file from disk
  2. Decode to RGB array (variable dimensions)
  3. Aspect-ratio-preserving rescale: min(height, width) = 256
  4. Store rescaled 256×256 image
```

**Stage 2: Mean Subtraction (Per-Channel)**
```
Training set statistics (computed once):
  - Iterate through entire training set
  - Compute per-channel mean for all pixels

Typical ImageNet means (BGR order in original paper):
  mean_B ≈ 103.939
  mean_G ≈ 116.779
  mean_R ≈ 123.680

Applied to each image:
  image_normalized = image - mean

Shape: (227, 227, 3)
Each pixel: [R, G, B] - [mean_R, mean_G, mean_B]
```

### Data Augmentation Techniques

**1. Random Cropping (Training)**
```
Input: 256×256×3 image
Output: 227×227×3 image

Random offset (y_offset, x_offset) ~ Uniform([0, 30))
Crop window: [y_offset:y_offset+227, x_offset:x_offset+227]

Purpose: Increase effective training set size (256² / 227² ≈ 1.27× per image)
Effect: Prevents model from memorizing spatial patterns

Code-like pseudocode:
  top = random.randint(0, 30)      # [0, 29]
  left = random.randint(0, 30)     # [0, 29]
  crop = image[top:top+227, left:left+227, :]
```

**2. Center Crop (Testing/Validation)**
```
Input: 256×256×3 image
Output: 227×227×3 image

Fixed offset: center the crop
  top = (256 - 227) // 2 = 14.5 → 14 (truncated)
  left = (256 - 227) // 2 = 14.5 → 14 (truncated)
  crop = image[14:14+227, 14:14+227, :]

Purpose: Deterministic evaluation, ensures reproducibility
Note: Original paper uses left-aligned or slightly off-center crop
```

**3. Horizontal Flip (Training)**
```
Applied with probability p = 0.5

image_flipped = image[:, ::-1, :]  # Flip horizontally (left-right)

Purpose: Dataset augmentation without changing content
Effect: Many ImageNet classes are symmetric (cars, dogs, buildings)
Cost: Negligible (bit-reversal operation)
```

**4. PCA Color Augmentation (Training)**
```
Motivation: Natural images have correlated RGB channels
            (e.g., shadows darken all channels together)

Algorithm:
  1. Compute PCA on pixel intensities across training set
  2. Extract principal components and eigenvalues
  3. For each training image:
     - Sample alpha_i ~ N(0, 0.1) for each principal component i
     - Compute: delta = sum_i (sqrt(lambda_i) * alpha_i * v_i)
     - Add delta to each RGB pixel: image += delta

Mathematical formulation:
  delta = [p_1, p_2, p_3] * [lambda_1*alpha_1, lambda_2*alpha_2, lambda_3*alpha_3]^T

where:
  p_i: i-th principal component (eigenvector)
  lambda_i: i-th eigenvalue
  alpha_i: random Gaussian coefficient

Effect:
  - ~1% reduction in top-1 error
  - Simulates natural lighting variations
  - Brightens or darkens images in a correlated way
```

### Augmentation Pipeline (Training)

```
For each training image:

  1. Load JPEG file
  2. Aspect-ratio preserving resize to 256×256
  3. Subtract ImageNet mean (per-channel)
  4. Random crop to 227×227
  5. Flip horizontally with p=0.5
  6. Apply PCA color augmentation
  7. Normalize to [0, 1] or [-1, 1] (optional)
  8. Pack into batch of 128

Total augmentation effect: Each epoch sees different crops, flips, and colors
                          Effective dataset size: ~1.27 × (2 flips) × ∞ (color aug)
                          ≈ 2.5-3× effective enlargement
```

### Augmentation Pipeline (Testing)

```
For each test image:

  1. Load JPEG file
  2. Aspect-ratio preserving resize to 256×256
  3. Subtract ImageNet mean (same values as training)
  4. Center crop to 227×227 (NO random crop, deterministic)
  5. NO horizontal flip (deterministic)
  6. NO color augmentation (deterministic)
  7. Pack into batch for inference

Testing is deterministic to ensure reproducibility and fair comparison.

Alternative (rarely used): 10-crop evaluation
  - 4 corner crops (top-left, top-right, bottom-left, bottom-right)
  - 1 center crop
  - 5 flipped versions (same crops)
  - Total: 10 patches per image
  - Average softmax outputs
  - Reported as "10-crop" error
```

---

## Section 8: Training Pipeline

### SGD with Momentum Optimizer

**Update Rule**:
```
v_t = momentum * v_{t-1} - learning_rate * gradient_t
theta_{t+1} = theta_t + v_t

where:
  v_t: velocity (momentum accumulator)
  momentum: typically 0.9
  learning_rate: α, typically starting at 0.01
  gradient_t: dL/dθ at iteration t
  theta_t: model parameters
```

**Momentum Effect**:
```
Without momentum:
  theta_{t+1} = theta_t - α * grad_t
  (Noisy updates, slow convergence in shallow directions)

With momentum = 0.9:
  theta_{t+1} = theta_t + 0.9 * v_{t-1} - 0.01 * grad_t
  (Accelerates in consistent directions, dampens oscillations)

Effect: 6-7% acceleration in convergence
```

### Weight Decay (L2 Regularization)

**Loss with Weight Decay**:
```
L_total = L_cross_entropy + λ * ||w||_2^2

where:
  L_cross_entropy: Standard CE loss on batch
  λ: weight decay coefficient (≈ 0.0005 in AlexNet)
  ||w||_2^2: Sum of squared weights across all layers
```

**Update with Weight Decay**:
```
Effective update:
  v_t = momentum * v_{t-1} - learning_rate * (gradient_CE + 2*λ*w)
  theta_{t+1} = theta_t + v_t

Interpretation:
  - Each weight is slightly shrunk toward zero
  - Larger weights are penalized more (quadratic term)
  - Prevents overfitting by limiting model complexity
```

### Learning Rate Schedule

**Schedule Used in AlexNet**:
```
Epoch 0-30:   learning_rate = 0.01
Epoch 30-60:  learning_rate = 0.001 (×0.1)
Epoch 60-90:  learning_rate = 0.0001 (×0.1 again)

Total training: 90 epochs
Batch size: 128
Training set: 1,280,000 images
Iterations per epoch: 1,280,000 / 128 = 10,000
Total iterations: ~900,000

Iteration schedule:
  Iterations 0-300,000:        α = 0.01
  Iterations 300,000-600,000:  α = 0.001
  Iterations 600,000-900,000:  α = 0.0001
```

**Rationale**:
```
- High learning rate (0.01) early: Quick descent in loss landscape
- Decay at milestone: Fine-tune weights as loss plateaus
- Final decay: Converge to minima with small steps
- Typical improvement: ~1-2% top-1 error from scheduling
```

### Dropout Regularization

**Dropout Mechanism During Training**:
```
Applied to FC6 and FC7 outputs with p=0.5

Forward pass with dropout:
  h_dropped = h * mask

where:
  h: hidden activation from ReLU, shape (128, 4096)
  mask: random binary mask ~ Bernoulli(p=0.5)
  mask shape: (128, 4096)

Each neuron output is:
  - Multiplied by 0 with probability 0.5 (dropped)
  - Multiplied by 1 with probability 0.5 (kept)
  - Passed through unchanged (binary decision)
```

**Interpretation**:
```
Physical interpretation:
  - Co-adaptation prevention: Each neuron can't rely on others
  - Implicit ensemble: Each forward pass samples different architecture
  - Computational: ~50% of FC neurons "used" per batch

Dropped neurons during training:
  - No forward activation computed
  - No gradient computed in backward pass
  - Parameters not updated

During testing/inference:
  - Dropout is disabled (p=0)
  - All neurons active
  - Weights are typically scaled by (1-p) = 0.5 (inverted dropout)
```

**Effect on Generalization**:
```
Without dropout:
  Training error: ~18% top-1 error (severe overfitting to train set)
  Test error: ~45% top-1 error

With dropout (p=0.5 in FC layers):
  Training error: ~25% top-1 error (higher, but less overfit)
  Test error: ~39.7% top-1 error (much better generalization)

Improvement: ~5-6% test error reduction
Note: Conv layers use NO dropout (stable spatial features)
```

### Training Hyperparameters Summary

| Parameter | Value | Reason |
|-----------|-------|--------|
| Optimizer | SGD + Momentum | Standard, proven for CNNs |
| Momentum | 0.9 | Accelerates convergence |
| Learning Rate (Phase 1) | 0.01 | Aggressive initial descent |
| Learning Rate (Phase 2) | 0.001 | Fine-tuning |
| Learning Rate (Phase 3) | 0.0001 | Convergence |
| Weight Decay (λ) | 0.0005 | L2 regularization strength |
| Dropout (FC layers) | p=0.5 | Prevent overfitting |
| Batch Size | 128 | GPU memory constraint |
| Training Epochs | 90 | ~900k iterations |
| Initialization | Gaussian (σ=0.01) | Small random weights |
| Bias Initialization | 0 | Standard |

### Training Procedure Pseudocode

```python
def train_epoch(model, data_loader, optimizer, epoch):
    """
    One training epoch over 1,280,000 images
    """
    model.train()  # Set dropout, BN to training mode
    total_loss = 0

    for iteration, (images, labels) in enumerate(data_loader):
        # images shape: (128, 3, 227, 227)
        # labels shape: (128,) with class indices in [0, 999]

        # Adjust learning rate at specific iterations
        if iteration == 300_000:  # After 30 epochs
            learning_rate = 0.001
            update_lr(optimizer, learning_rate)
        elif iteration == 600_000:  # After 60 epochs
            learning_rate = 0.0001
            update_lr(optimizer, learning_rate)

        # Forward pass
        logits = model(images)  # (128, 1000)

        # Compute cross-entropy loss
        loss = cross_entropy_loss(logits, labels)

        # L2 regularization on weights
        l2_loss = 0.0005 * sum(w.norm()**2 for w in model.parameters())
        total_loss_with_reg = loss + l2_loss

        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        total_loss_with_reg.backward()  # Compute gradients

        # SGD update with momentum
        optimizer.step()  # Apply momentum update

        total_loss += loss.item()

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, Loss: {total_loss / (iteration+1):.4f}")

    return total_loss / len(data_loader)


def train():
    """
    Full training for 90 epochs
    """
    model = AlexNet()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(90):
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        val_loss, val_top1, val_top5 = validate(model, val_loader)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
              f"Val Top-1: {val_top1:.1%}, Val Top-5: {val_top5:.1%}")
```

---

## Section 9: Dataset and Evaluation Protocol

### ILSVRC-2010 vs ILSVRC-2012

**ILSVRC-2010** (First paper version trained on):
```
- Total images: 1,261,406
- Training set: 1,200,000 images
- Validation set: 50,000 images
- Test set: 100,000 images (labels held out during competition)
- Classes: 1,000 object categories
- Resolution: Variable (256≤min(H,W), ≤512)

Results on ILSVRC-2010:
  Top-1 error: ~40.7%
  Top-5 error: ~18.3%
```

**ILSVRC-2012** (Larger, higher quality dataset):
```
- Total images: 1,281,167
- Training set: 1,200,000 images
- Validation set: 50,000 images
- Test set: 100,000 images (labels held out)
- Classes: 1,000 object categories
- Resolution: Variable (256≤min(H,W), ≤512)

Results on ILSVRC-2012 (used for final published results):
  Top-1 error: 39.7% (single network)
  Top-5 error: 18.9% (single network)
  Top-1 error: 38.1% (5-model ensemble)
  Top-5 error: 16.4% (7-model ensemble)
```

### Class Distribution

**ImageNet-1000 Classes** (examples):
```
Animal categories:
  - n02084075: dog (207 subcategories)
  - n01503061: bird (321 subcategories)
  - n01674464: lizard
  - n02084442: cat (subclasses)

Vehicle categories:
  - n02814860: automobile
  - n02930766: aircraft
  - n04209133: boat

Object categories:
  - n04379243: table
  - n03201941: dining table
  - n04398044: teapot
  - n02933112: cabinet

Plant categories:
  - n12985857: mushroom
  - n11939491: daisy

Total: 1000 leaf nodes in ImageNet hierarchy
```

### Train/Val/Test Split

**Training Set** (1,200,000 images):
```
- Used to train model weights
- All augmentations applied (crop, flip, color)
- Sampled in shuffled batches of 128
- Distributed across 1000 classes (~1200 images per class)
- Multiple epochs: 90 full passes through data

Class distribution:
  min_class_samples = 737 images (class 'pottery')
  max_class_samples = 1300 images (class 'dog')
  median = ~1200 images/class
```

**Validation Set** (50,000 images):
```
- Used to tune hyperparameters and select best model
- No augmentations (center crop only)
- Processed in batches of 128
- Distributed as: ~50 images per class
- Used during training to monitor progress
- Final metrics computed with ensemble methods
```

**Test Set** (100,000 images):
```
- Held-out evaluation set (labels not released to teams)
- Used for final competition scoring
- Distributed as: ~100 images per class
- Organized by ILSVRC organizers
- Submitted predictions, organizers compute error
- Results used for competition ranking
```

### Evaluation Protocol (During Training)

```
Validation cycle (every N iterations):

  1. Set model to eval mode (disable dropout, BN)
  2. Forward pass on validation set
  3. Compute top-1 and top-5 errors
  4. Record metrics in log
  5. If best validation error: save checkpoint
  6. Return to training mode

Validation error monitoring:
  - Computed on 50,000 validation images
  - Split into batches of 128
  - For each batch:
      logits = model(images)
      probs = softmax(logits)
      top1_idx = argmax(probs, dim=1)
      top5_idx = argsort(probs, descending=True)[:5]
      top1_err += (top1_idx != true_labels).sum()
      top5_err += all(true_labels not in top5_idx).sum()

  Final: top1_error = top1_err / 50000, top5_error = top5_err / 50000
```

### Test Set Evaluation (Final Competition)

```
Test submission protocol:

  1. Load trained model
  2. Forward pass on 100,000 test images
  3. Extract top-1 predictions (argmax softmax)
  4. Format predictions as:
     image_id prediction_class_id
  5. Submit file to ILSVRC organizers
  6. Organizers compare against held-out labels
  7. Compute top-1 and top-5 errors
  8. Publish results

Results announced in September 2012:
  - AlexNet (single model): 39.7% top-1, 18.9% top-5
  - Previous SOTA (2011): 25.8% top-5 error
  - AlexNet improvement: 26.9% relative reduction (2nd place: ~27%)
  - Clear winner with >11% absolute gap over 2nd place
```

### Ensemble Evaluation

```
Ensemble methodology (combining multiple trained models):

Procedure:
  1. Train 5-7 independently initialized networks
  2. All networks have same architecture
  3. Different random seeds, different data shuffles
  4. For each test image:
     - Forward through all networks: logits_k = model_k(image)
     - Average softmax outputs: p_avg = (1/K) * sum_k softmax(logits_k)
     - Predict: class = argmax(p_avg)

Results:
  - 5-model ensemble: 38.1% top-1 error
  - 7-model ensemble: 16.4% top-5 error
  - Ensemble wins: ~1.6% reduction from single model
  - Standard practice in ImageNet competitions

Trade-off:
  + Better accuracy (state-of-the-art)
  - 7× inference cost
  - Not practical for real-time applications
```

---

## Section 10: Results Summary and Ablations

### Main Results (ILSVRC-2012)

```
AlexNet Single Network:
  Top-1 Error:  39.7%
  Top-5 Error:  18.9%

AlexNet Ensemble (7 networks):
  Top-1 Error:  38.1%
  Top-5 Error:  16.4%

Comparison with Previous SOTA:
  2011 ILSVRC Winner (traditional methods):
    Top-5 Error: 25.8%

  AlexNet vs 2011:
    Absolute improvement: 25.8% - 18.9% = 6.9%
    Relative improvement: 6.9 / 25.8 = 26.7%

  AlexNet vs 2nd place (2012):
    AlexNet: 18.9%
    2nd place: ~26.2%
    Margin: 7.3% absolute gap
```

### Ablation Study Results

**Ablation 1: Removing ReLU (Use Tanh Instead)**
```
Architecture: Same as AlexNet but with tanh instead of ReLU

tanh(x) = (e^x - e^-x) / (e^x + e^-x)  ∈ (-1, 1)

Results:
  Training time: ~6× longer to reach same validation accuracy
  Convergence: Significantly slower
  Final performance: Comparable accuracy if trained long enough

Conclusion: ReLU critical for training speed, not accuracy
Effect: ~6-7× faster convergence (equivalent to 300k more SGD iterations)

Hypothesis (authors):
  - Tanh has horizontal asymptotes at ±1 (saturation)
  - Gradient = 0 in saturated regions → slow learning
  - ReLU never saturates for x > 0 → always non-zero gradient
```

**Ablation 2: Removing Local Response Normalization (LRN)**
```
Architecture: Same as AlexNet but without LRN after Conv1 and Conv2

LRN formula (not critical to understand):
  b_x,y^i = a_x,y^i / (k + α * sum_j (a_x,y^j)^2)^β

  where j ranges over neighboring channels

Results (without LRN):
  Top-1 error (ILSVRC-2010): 41.4%
  Top-1 error (with LRN):     40.7%
  Improvement: 0.7% from LRN alone

  Top-5 error (ILSVRC-2010): 19.0%
  Top-5 error (with LRN):     18.3%
  Improvement: 0.7% from LRN

Conclusion: LRN provides ~0.7-1.4% improvement
Reason: LRN acts as implicit regularization, normalizes activations
Note: Later architectures (ResNet, Inception) replaced LRN with BatchNorm
```

**Ablation 3: Removing Dropout**
```
Architecture: Same as AlexNet but p=0 in FC layers (no dropout)

Results:
  Without dropout:
    Training top-1 error: ~17% (severe overfitting)
    Validation top-1 error: ~43-45%

  With dropout (p=0.5):
    Training top-1 error: ~25% (higher but less overfit)
    Validation top-1 error: ~39.7%

  Improvement: ~5% reduction in test error from dropout
  Overfitting gap (train - test):
    Without dropout: 17% - 45% = 28% gap (severe)
    With dropout: 25% - 39.7% = 14.7% gap (moderate)

Conclusion: Dropout critical for generalization
Effect: Reduces overfitting by ~50%, enables training 60M parameter model
Note: Could train smaller model without dropout, but 60M enables better accuracy
```

**Ablation 4: Depth Ablation (Remove Conv Layers)**
```
Architecture: 5-layer version (remove Conv3, Conv4, or both)

Comparison:
  Full 8-layer (5 conv + 3 FC): 39.7% top-1 error
  7-layer (remove Conv5):       41.2% top-1 error → +1.5% error
  6-layer (remove Conv4+5):     42.1% top-1 error → +2.4% error
  5-layer (remove Conv3+4+5):   42.9% top-1 error → +3.2% error

Pattern: Each additional conv layer reduces error by ~1% absolute

Conclusion: Depth is critical (diminishing returns but positive)
Reason: Deeper networks have larger receptive fields, capture hierarchical features

Receptive field growth:
  5-layer: receptive field ≈ 27×27
  8-layer: receptive field ≈ 43×43

Better spatial understanding with more layers.
```

**Ablation 5: Data Augmentation Impact**

```
Training without augmentations:
  - No random crops (only center crop)
  - No horizontal flips
  - No PCA color augmentation

Results:
  Top-1 error without augmentation: ~41.7%
  Top-1 error with augmentation:    39.7%
  Improvement: 2.0% from augmentation

Breakdown of contributions:
  Random crops: ~1.0-1.2% improvement
  Horizontal flips: ~0.3-0.5% improvement
  PCA color augmentation: ~0.3-0.5% improvement
  Total: ~2.0%

Conclusion: Augmentation crucial for generalization
Effect: Prevents overfitting to specific crop positions and lighting
```

### Result Visualization Summary

```
ImageNet Performance Over Time (Simplified):

2010 ILSVRC Winner (SVM-based): 25.8% top-5 error
        ↑
2012 Competitor (Traditional):  ~27% top-5 error
        ↑
2012 AlexNet (single):          18.9% top-5 error ← 26% relative improvement!
        ↑
2012 AlexNet (ensemble 7):      16.4% top-5 error


Historical context:
- Previous decade: Incremental 0.1-0.2% improvements
- AlexNet: 6.9% absolute jump
- Marked inflection point for deep learning adoption
```

---

## Section 11: Practical Insights

### 10 Engineering Takeaways

1. **GPU Implementation is Crucial**
   - CPU training would take weeks, GPU takes 6 days
   - Custom CUDA kernels for convolution critical
   - ~50× speedup possible on GPU (GTX 580 in 2012)
   - Modern GPUs: 100-1000× speedups over CPU

2. **ReLU > Tanh for Deep Networks**
   - ReLU enables 6× faster convergence
   - No saturation gradient problem
   - Simple to implement (max(0, x))
   - Became standard in industry
   - Later: Leaky ReLU, GELU variants still use this principle

3. **Data Augmentation is Cheap & Effective**
   - 2% improvement for minimal computational cost
   - Apply augmentations during training only
   - Crop + flip + color augmentation essential for ImageNet
   - Modern approach: AutoAugment, RandAugment build on this

4. **Dropout Regularization Works**
   - 5% improvement without increasing model size
   - p=0.5 in FC layers (too aggressive in conv layers)
   - Essential when training very large models
   - Modern alternative: BatchNorm, but dropout still useful

5. **Learning Rate Scheduling Matters**
   - Fixed LR plateau is common (0.01 → 0.001 → 0.0001)
   - Decay by 10× at specific epochs is simple & effective
   - 1-2% improvement from scheduling alone
   - Modern: Cosine annealing, warm restarts

6. **Weight Decay is Essential**
   - λ=0.0005 prevents overfitting on 60M parameters
   - Simple L2 penalty on loss
   - ~2-3% improvement in test error
   - Standard practice across all architectures

7. **Batch Normalization Would Help (Not Available Then)**
   - LRN was approximation of what BN does
   - BatchNorm (2015): More effective, standard today
   - Enables higher learning rates, faster convergence
   - AlexNet paved way for better normalization techniques

8. **Ensemble Methods Provide ~1-2% Gains**
   - 5-7 independently trained networks
   - Average softmax outputs (not voting on predictions)
   - Cheap way to improve accuracy pre-inference optimization
   - Worth doing when model size allows

9. **Memory Management on Limited VRAM**
   - Split model across 2 GPUs (1.5GB each)
   - Batch size 128 carefully chosen for memory fit
   - Strategic tensor shapes to minimize memory
   - Modern GPUs have more memory, but principle applies to larger models

10. **Careful Initialization and Hyperparameter Tuning**
    - Gaussian initialization σ=0.01 prevents saturation
    - Momentum=0.9 standard for SGD
    - Learning rate schedule critical (not one-size-fits-all)
    - Multiple restarts with different initializations helps

### 5 Common Gotchas

1. **Dropout at Test Time**
   - **Gotcha**: Forgetting to disable dropout during evaluation
   - **Symptom**: Validation error much worse than training
   - **Fix**: Set model.eval() or drop_rate=0 explicitly
   - **Impact**: Can reduce accuracy by 5-10% if overlooked
   - **Modern**: Framework handles this automatically

2. **Mean Subtraction Must Use Training Set Statistics**
   - **Gotcha**: Computing mean on validation/test set
   - **Symptom**: Data leakage, artificially good test performance
   - **Fix**: Compute mean only on training set, reuse for val/test
   - **Impact**: Can inflate accuracy by 2-5%
   - **Lesson**: All preprocessing must be fit on training only

3. **Learning Rate Too High Causes Divergence**
   - **Gotcha**: Starting with α=0.1 instead of 0.01
   - **Symptom**: Loss increases, then NaN
   - **Fix**: Use learning rate schedule, start conservative
   - **Impact**: Training fails completely
   - **Principle**: Be cautious with learning rates, easy to overshoot

4. **Batch Size Affects Learning Dynamics**
   - **Gotcha**: Using batch_size=32 then switching to 128
   - **Symptom**: Different accuracy, different convergence speed
   - **Fix**: Choose batch size early, stick with it
   - **Impact**: 1-2% accuracy differences
   - **Note**: Larger batches → noisier gradients → need higher LR

5. **Class Imbalance in ILSVRC**
   - **Gotcha**: Assuming uniform distribution across 1000 classes
   - **Reality**: Some classes have 2× more images than others
   - **Fix**: Use weighted sampling or class weights in loss
   - **Impact**: Biased performance on frequent vs rare classes
   - **Lesson**: Always check dataset distribution first

### Overfitting Prevention Strategy

**AlexNet's Multi-Pronged Overfitting Prevention** (Model has 60M params, only 1.2M train images):

```
Layer 1: Data-Level Augmentation
  - Random cropping: ~1.27× dataset expansion
  - Horizontal flips: 2× dataset expansion
  - PCA color augmentation: ~continuous expansion
  - Total effective: ~3× larger training distribution

Layer 2: Architectural Regularization
  - Dropout p=0.5: Implicitly regularizes
  - ReLU: Simpler activation than tanh
  - LRN: Channel-wise normalization

Layer 3: Optimization-Level Regularization
  - Weight decay λ=0.0005: L2 penalty on weights
  - Momentum: Smoother gradients, less noisy

Layer 4: Early Stopping
  - Monitor validation error every epoch
  - Save best model checkpoint
  - Stop if no improvement after K epochs

Overfitting metrics to watch:
  Train-Test Gap:
    Overfitting: gap > 15-20%
    Normal: gap 5-10%
    Underfitting: gap < 2%

  Loss curves:
    Overfitting: training loss decreases, validation increases
    Healthy: both decrease together
```

---

## Section 12: Minimal Reimplementation Checklist

### Prerequisites
```
Libraries needed:
  ✓ PyTorch or TensorFlow 2.x
  ✓ numpy, scipy (numerical ops)
  ✓ PIL/Pillow (image loading)
  ✓ torchvision or similar (datasets)
  ✓ matplotlib (visualization)

Hardware minimum:
  ✓ GPU with ≥6GB VRAM (or train smaller batch)
  ✓ 1-2 weeks for full training (1.2M images, 90 epochs)
  ✓ ~200GB disk space (ImageNet dataset)
```

### Step 1: Data Preparation
```
[ ] Download ImageNet dataset (ILSVRC-2012)
[ ] Extract images to directory structure:
    ImageNet/
      train/
        n02084075/  (dog class)
          *.jpg
        ...
      val/
        n02084075/
          *.jpg
[ ] Compute training set mean (per-channel, RGB)
    - Script: iterate train set, collect pixel statistics
[ ] Verify image shapes (variable resolution OK)
[ ] Create train/val splits (1.2M / 50K)
```

### Step 2: Data Loading Pipeline
```
[ ] Implement image augmentation:
    - Rescale to 256 in shortest dimension (preserve aspect)
    - Random crop to 227×227 (training)
    - Center crop to 227×227 (testing)
    - Horizontal flip with p=0.5 (training)
    - PCA color augmentation (optional, ~1% improvement)
    - Mean subtraction (per-channel)

[ ] Create DataLoader with batch_size=128
[ ] Verify output shapes: (128, 3, 227, 227)
[ ] Check loading speed: target ~1-2 images/ms
[ ] Test on small subset (1000 images) first
```

### Step 3: Model Architecture
```
[ ] Implement Conv2d layers:
    def conv_block(in_ch, out_ch, k, s, p):
        return Sequential(
            Conv2d(in_ch, out_ch, k, s, p),
            ReLU(),
            LocalResponseNorm(...)  # or skip
        )

[ ] Layer 1: Conv(3→96, k=11, s=4, p=0) + LRN
[ ] MaxPool (k=3, s=2)
[ ] Layer 2: Conv(96→256, k=5, s=1, p=2) + LRN
[ ] MaxPool (k=3, s=2)
[ ] Layer 3: Conv(256→384, k=3, s=1, p=1)
[ ] Layer 4: Conv(384→384, k=3, s=1, p=1)
[ ] Layer 5: Conv(384→256, k=3, s=1, p=1)
[ ] MaxPool (k=3, s=2)
[ ] Flatten
[ ] FC6: Linear(9216→4096) + ReLU + Dropout(0.5)
[ ] FC7: Linear(4096→4096) + ReLU + Dropout(0.5)
[ ] FC8: Linear(4096→1000)

[ ] Initialize weights:
    - Conv: Gaussian(σ=0.01)
    - Bias: 0
    - Can use torch.nn.init.normal_()

[ ] Test forward pass with dummy input:
    x = torch.randn(1, 3, 227, 227)
    y = model(x)  # should output (1, 1000)
```

### Step 4: Loss and Optimization
```
[ ] Use CrossEntropyLoss (combines softmax + CE)
[ ] Optimizer: SGD with momentum=0.9
[ ] Loss = CE + 0.0005 * L2(weights)
    - Can use weight_decay parameter in SGD

[ ] Learning rate schedule:
    - Phase 1 (epochs 0-30): α=0.01
    - Phase 2 (epochs 30-60): α=0.001
    - Phase 3 (epochs 60-90): α=0.0001
    - Use StepLR or manual update

[ ] Implement lr_scheduler:
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler.step()  # after each epoch
```

### Step 5: Training Loop
```
[ ] Main training function:

for epoch in range(90):
    model.train()  # Enable dropout
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward
        logits = model(images)
        loss = criterion(logits, labels)

        # Add L2 regularization
        l2_reg = 0.0005 * sum(p.norm()**2 for p in model.parameters())
        total_loss_with_reg = loss + l2_reg

        # Backward
        optimizer.zero_grad()
        total_loss_with_reg.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    scheduler.step()  # Update learning rate

    # Validation
    val_loss, val_top1, val_top5 = validate(model, val_loader)
    print(f"Epoch {epoch}: Val Top-1: {val_top1:.1%}, Top-5: {val_top5:.1%}")

    if val_top1 < best_accuracy:
        best_accuracy = val_top1
        save_checkpoint(model, f'best_model.pth')
```

### Step 6: Evaluation
```
[ ] Validation function:

def validate(model, val_loader):
    model.eval()  # Disable dropout
    top1_correct, top5_correct = 0, 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            # Top-1
            top1_pred = probs.argmax(dim=1)
            top1_correct += (top1_pred == labels).sum().item()

            # Top-5
            top5_pred = probs.topk(5, dim=1)[1]
            for i in range(len(labels)):
                if labels[i] in top5_pred[i]:
                    top5_correct += 1

            total += labels.size(0)

    top1_error = 1 - (top1_correct / total)
    top5_error = 1 - (top5_correct / total)

    return top1_error, top5_error

[ ] Test evaluation on small val subset first
[ ] Verify top-1 and top-5 metrics
```

### Step 7: Inference
```
[ ] Load best checkpoint:
    model.load_state_dict(torch.load('best_model.pth'))

[ ] Test-time evaluation:
    model.eval()

    with torch.no_grad():
        logits = model(test_image)
        probs = F.softmax(logits, dim=1)
        prediction = probs.argmax(dim=1)

[ ] (Optional) Ensemble with multiple models
[ ] (Optional) 10-crop evaluation
```

### Step 8: Debugging and Optimization
```
[ ] Common issues to check:

  Issue: Very high training loss
  [ ] Check learning rate (might be too high)
  [ ] Verify data preprocessing (mean subtraction)
  [ ] Check loss computation (cross-entropy)

  Issue: Training loss decreases but val plateaus
  [ ] Increase dropout rate
  [ ] Increase weight decay
  [ ] Check learning rate schedule

  Issue: Overfitting (train loss ↓, val loss ↑)
  [ ] Add more data augmentation
  [ ] Increase dropout (p=0.5 → 0.7)
  [ ] Increase weight decay λ
  [ ] Use earlier stopping

  Issue: Training too slow
  [ ] Check GPU utilization (nvidia-smi)
  [ ] Increase batch size (if VRAM allows)
  [ ] Use mixed precision (torch.cuda.amp)
  [ ] Profile data loading bottleneck

  Issue: OOM (Out of Memory)
  [ ] Decrease batch size (128 → 64)
  [ ] Use gradient checkpointing
  [ ] Enable mixed precision
  [ ] Split model across multiple GPUs
```

### Step 9: Expected Results Timeline
```
After 10 epochs:   ~55% top-1 error (untrained)
After 30 epochs:   ~42% top-1 error (learning)
After 60 epochs:   ~40% top-1 error (converging)
After 90 epochs:   ~39.7% top-1 error (AlexNet-level)

With ensemble: ~38% top-1 error

If worse than this:
  - Check data preprocessing
  - Verify augmentation (should see ~2% improvement)
  - Check dropout disabled at test time
```

### Step 10: Checklist Summary
```
✓ Dataset downloaded and preprocessed
✓ Data augmentation pipeline working
✓ DataLoader yields (128, 3, 227, 227)
✓ AlexNet model defined and initialized
✓ Forward pass produces (batch, 1000) logits
✓ Loss function + L2 regularization configured
✓ SGD + momentum optimizer ready
✓ Learning rate schedule implemented
✓ Training loop runs without errors
✓ Validation metrics (top-1, top-5) computed correctly
✓ Model checkpoint saving/loading works
✓ Evaluation reaches ~39-40% top-1 error
✓ Results reproducible with fixed seed
```

---

## Bonus: Modern Improvements on AlexNet's Approach

### What's Different Today (2024+):

**Architectural Changes**:
- VGG (deeper), ResNet (skip connections), DenseNet (dense connections)
- EfficientNet (scaling rules), Vision Transformers (attention)
- But AlexNet's core ideas (ReLU, dropout, augmentation) still fundamental

**Training Changes**:
- BatchNorm replaced LocalResponseNorm
- Adam optimizer competes with SGD
- Learning rate warmup + cosine annealing
- Gradient accumulation for larger effective batches

**Data Changes**:
- AutoAugment, RandAugment, MixUp, CutMix (better augmentation)
- Larger models + bigger datasets (ImageNet-21k pretraining)
- Transfer learning standard (pretrain on large dataset, finetune on small)

**Inference Changes**:
- Quantization (8-bit instead of 32-bit)
- Pruning (removing unimportant weights)
- Knowledge distillation (student from teacher)
- ONNX / TensorRT optimization

**But Key Principles Remain**:
- Deep hierarchical features capture patterns
- Regularization prevents overfitting
- Data augmentation improves generalization
- GPU acceleration enables training
- Systematic evaluation on held-out test set

---

## References & Additional Reading

1. Original Paper: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." NIPS 2012.

2. ILSVRC Dataset: http://www.image-net.org/

3. Related Architectures:
   - VGG (2014): Depth matters
   - ResNet (2015): Skip connections enable even deeper networks
   - DenseNet (2016): Dense connections improve gradient flow

4. Optimization Advances:
   - Adam optimizer (2014): Adaptive learning rates
   - BatchNorm (2015): Normalization enables faster training
   - LARS (2017): Large-batch training

5. Augmentation Evolution:
   - AutoAugment (2018): Learned augmentation policies
   - RandAugment (2019): Simpler, data-dependent augmentation
   - MixUp (2017), CutMix (2019): New augmentation strategies

---

**Summary in One Paragraph**:
AlexNet proved that deep convolutional neural networks could dramatically outperform hand-crafted features on ImageNet, winning the 2012 ILSVRC competition with an 18.9% top-5 error (vs 25.8% previous best). The key innovations—ReLU activations for fast training, dropout for regularization, GPU acceleration, and data augmentation—enabled training a 60-million parameter model on 1.2 million images. While modern architectures have superseded AlexNet's specific design, the principles it established (depth + scale + regularization + augmentation) remain foundational to deep learning.
