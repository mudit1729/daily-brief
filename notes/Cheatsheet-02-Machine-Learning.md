# Machine Learning Cheatsheet — Scale Robotics Interview

> Dense reference: loss functions, optimizers, architectures, training tricks, and theory you need to recall under pressure.

---

## 0. HIGH-YIELD INTERVIEW ANSWERS

### Fast Decision Rules
```
Cross-Entropy vs MSE for classification:
- CE is preferred because it models class probabilities and gives stronger gradients
- MSE on one-hot labels usually trains slower and is a worse fit to classification

Softmax CE vs BCE:
- Softmax CE: exactly one class is correct
- BCE / sigmoid: multi-label, classes independent

Precision vs Recall:
- Precision matters when false positives are costly
- Recall matters when false negatives are costly
- On imbalanced data, PR-AUC is usually more informative than ROC-AUC

AdamW vs SGD:
- AdamW: easier to tune, standard for transformers and many modern pipelines
- SGD+momentum: often strong final generalization for CNN-heavy vision tasks

BatchNorm vs LayerNorm vs GroupNorm:
- BN: CNNs with decent batch size
- LN: transformers / sequence models
- GN: small-batch detection / segmentation
```

### High-Value One-Liners
- Overfitting: train improves while validation degrades.
- Underfitting: both train and validation are poor.
- Data leakage can dominate model choice; always verify the split.
- Accuracy can be misleading under class imbalance.

---

## 1. LOSS FUNCTIONS

### Regression Losses

```
MSE (L2):        L = (1/n) Σ (yᵢ - ŷᵢ)²
                 Gradient: 2(ŷ - y)/n
                 Sensitive to outliers

MAE (L1):        L = (1/n) Σ |yᵢ - ŷᵢ|
                 Gradient: sign(ŷ - y)/n
                 Robust to outliers, not smooth at 0

Huber (Smooth L1):
                 Let e = y-ŷ
                 L = { 0.5e²               if |e| ≤ δ
                     { δ|e| - 0.5δ²        otherwise
                 Combines MSE (small errors) + MAE (large errors)
                 Used in Faster R-CNN bbox regression
```

### Classification Losses

```
Cross-Entropy (CE):
  Multi-class:   L = -Σ yₖ log(ŷₖ)     (y is one-hot)
                 = -log(ŷ_c)            (c = true class)

  Binary CE:     L = -[y log(ŷ) + (1-y) log(1-ŷ)]

  Softmax:       ŷₖ = exp(zₖ) / Σⱼ exp(zⱼ)

  Numerically stable: subtract max(z) before exp
  In practice: pass raw logits to framework loss (`CrossEntropyLoss`)
```

```
Focal Loss:      L = -αₜ (1-pₜ)^γ log(pₜ)
                 pₜ = p if y=1, else 1-p
                 γ = focusing parameter (typically 2)
                 α = class weight

                 Purpose: down-weight easy negatives
                 Used in: RetinaNet, one-stage detectors
```

```
Label Smoothing:  y_smooth = (1-ε)*y_onehot + ε/K
                  Prevents overconfident predictions
                  Typical ε = 0.1
```

### Metric Learning Losses

```
Contrastive:     L = y*d² + (1-y)*max(0, margin-d)²
                 d = ‖f(x₁) - f(x₂)‖
                 y = 1 if same class, 0 if different

Triplet:         L = max(0, d(a,p) - d(a,n) + margin)
                 a = anchor, p = positive, n = negative
                 Hard mining: select hardest negatives

InfoNCE / NT-Xent (used in SimCLR, CLIP):
                 L = -log[ exp(sim(zᵢ,zⱼ)/τ) / Σₖ exp(sim(zᵢ,zₖ)/τ) ]
                 sim = cosine similarity
                 τ = temperature (0.07 typical)
```

### Reconstruction / Generative Losses

```
L1 Reconstruction:   L = ‖x - x̂‖₁    (sharper than L2)
Perceptual Loss:     L = Σₗ ‖φₗ(x) - φₗ(x̂)‖²  (VGG features)
Adversarial (GAN):   L_D = -E[log D(x)] - E[log(1-D(G(z)))]
                     L_G = -E[log D(G(z))]

KL Divergence (VAE): KL(q‖p) = -0.5 Σ (1 + log σ² - μ² - σ²)
                     ELBO = E[log p(x|z)] - KL(q(z|x) ‖ p(z))
```

---

## 2. ACTIVATION FUNCTIONS

```
ReLU:        f(x) = max(0, x)           f'(x) = {1 if x>0, 0 if x≤0}
             Problem: dead neurons (if stuck at 0)

LeakyReLU:   f(x) = max(αx, x)          α=0.01 typical
             Fixes dying ReLU

GELU:        f(x) = x * Φ(x)            Φ = CDF of standard normal
             ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
             Used in: Transformers, BERT, GPT

SiLU/Swish:  f(x) = x * σ(x)            σ = sigmoid
             Used in: EfficientNet, YOLOv5+

Sigmoid:     f(x) = 1/(1+e⁻ˣ)           f'(x) = f(x)(1-f(x))
             Range: (0,1). Vanishing gradient for |x| >> 0

Tanh:        f(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)  f'(x) = 1-f(x)²
             Range: (-1,1). Zero-centered (better than sigmoid)

Softmax:     f(xᵢ) = exp(xᵢ)/Σⱼexp(xⱼ)
             ∂fᵢ/∂xⱼ = fᵢ(δᵢⱼ - fⱼ)    (Jacobian)
```

---

## 3. OPTIMIZERS

### SGD + Momentum
```
vₜ = β*vₜ₋₁ + ∇L(θₜ)          (momentum, β=0.9 typical)
θₜ₊₁ = θₜ - lr * vₜ

Nesterov:
vₜ = β*vₜ₋₁ + ∇L(θₜ - β*vₜ₋₁)   (look-ahead gradient)
```

### Adam
```
mₜ = β₁*mₜ₋₁ + (1-β₁)*gₜ              (1st moment, mean)
vₜ = β₂*vₜ₋₁ + (1-β₂)*gₜ²             (2nd moment, variance)
m̂ₜ = mₜ/(1-β₁ᵗ)                        (bias correction)
v̂ₜ = vₜ/(1-β₂ᵗ)                        (bias correction)
θₜ₊₁ = θₜ - lr * m̂ₜ / (√v̂ₜ + ε)

Defaults: β₁=0.9, β₂=0.999, ε=1e-8
```

### AdamW (Decoupled Weight Decay)
```
Adaptive step:   θₜ₊₁ = θₜ - lr * m̂ₜ/(√v̂ₜ + ε)
Decay step:      θₜ₊₁ = θₜ₊₁ - lr * λ * θₜ

Key difference from Adam + L2:
  Adam+L2: weight decay interacts with adaptive learning rate
  AdamW: weight decay applied directly to params (decoupled)
  → Better generalization, standard for transformers
```

### Learning Rate Schedules
```
Step Decay:        lr = lr₀ * γ^(epoch // step_size)
Cosine Annealing:  lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(πt/T))
Warmup + Cosine:   Linear warmup for N steps → cosine decay
OneCycleLR:        Warmup → max → anneal (superconvergence)
ReduceOnPlateau:   Reduce lr when metric stops improving
```

---

## 4. NORMALIZATION

```
Batch Norm:   μ, σ² over (N, H, W) per channel     — depends on batch
              At inference: use running mean/var
              y = γ * (x-μ)/√(σ²+ε) + β

              Problem: small batch sizes, sequence data
              Good for: CNNs with large batches

Layer Norm:   μ, σ² over feature dimension(s) per sample/token
              Used in: Transformers
              Stable for any batch size
              In NLP: usually normalize over hidden dimension only

Instance Norm: μ, σ² over (H, W) per channel per sample
              Used in: Style transfer

Group Norm:   μ, σ² over groups of channels per sample
              G groups (typically 32), compromise between LN and IN
              Used when batch size is small (detection, segmentation)
```

---

## 5. REGULARIZATION

```
L2 (Weight Decay):   L_reg = λ/2 * Σ wᵢ²
                     Shrinks weights, prevents large magnitudes

L1 (Lasso):          L_reg = λ * Σ |wᵢ|
                     Induces sparsity (many weights → 0)

Dropout:             Randomly zero out neurons with probability p
                     At test time: multiply by (1-p) OR use inverted dropout
                     Inverted dropout (standard): scale by 1/(1-p) during training

                     DropPath/Stochastic Depth: drop entire layers/blocks

Data Augmentation:   Random crop, flip, color jitter, Cutout, Mixup, CutMix,
                     RandAugment, AutoAugment, mosaic (YOLO)

Mixup:               x̃ = λxᵢ + (1-λ)xⱼ,  ỹ = λyᵢ + (1-λ)yⱼ
CutMix:              Paste patch from one image onto another, mix labels by area ratio

Label Smoothing:     Soft targets instead of hard one-hot

EMA (Exponential Moving Average):
                     θ_ema = α*θ_ema + (1-α)*θ     (α=0.999)
                     Use EMA weights at test time
```

---

## 6. BACKPROPAGATION

### Chain Rule
```
∂L/∂w = ∂L/∂ŷ * ∂ŷ/∂z * ∂z/∂w

For a network: x → z₁=W₁x+b₁ → a₁=σ(z₁) → z₂=W₂a₁+b₂ → ŷ=softmax(z₂) → L

Forward pass: compute all activations
Backward pass: compute gradients layer by layer (reverse order)
```

### Key Derivatives
```
Linear:     z = Wx + b    → ∂z/∂W = xᵀ, ∂z/∂x = Wᵀ, ∂z/∂b = I
ReLU:       a = max(0,z)  → ∂a/∂z = 1{z>0}
Sigmoid:    σ(z)          → σ(z)(1-σ(z))
Softmax+CE: L = -log(ŷ_c) → ∂L/∂zᵢ = ŷᵢ - yᵢ  (very clean!)
BatchNorm:  complex but computed by autograd
```

### Gradient Issues
```
Vanishing:  Gradients → 0 in deep nets (sigmoid/tanh saturation)
            Fix: ReLU, residual connections, proper init, normalization

Exploding:  Gradients → ∞
            Fix: gradient clipping (clip by norm or value), proper init

Gradient Clipping:
  By norm:  if ‖g‖ > max_norm: g = g * max_norm/‖g‖
  By value: g = clamp(g, -clip_value, clip_value)
```

### Weight Initialization
```
Xavier/Glorot:  W ~ N(0, 2/(fan_in + fan_out))    — for sigmoid/tanh
Kaiming/He:     W ~ N(0, 2/fan_in)                 — for ReLU
                fan_in = number of input units
```

---

## 7. TRANSFORMER ARCHITECTURE

### Self-Attention
```
Q = XWq,  K = XWk,  V = XWv     (X is input, W are learned)

Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V

√dₖ = scaling factor (dₖ = dimension of keys)
Purpose: prevents dot products from getting too large → softmax saturation
```

### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) Wₒ
headᵢ = Attention(QWᵢq, KWᵢk, VWᵢv)

h = number of heads (typically 8 or 16)
dₖ = d_model / h    (each head operates on reduced dimension)

Total params: 4 * d_model² (for Wq, Wk, Wv, Wo)
```

### Positional Encoding (Sinusoidal)
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Learned positional embeddings also common (BERT, GPT)
Rotary Position Embedding (RoPE): encode position in rotation of Q,K
```

### Transformer Block
```
Pre-norm block:
  x = x + MultiHeadAttention(LN(x), LN(x), LN(x))
  x = x + FFN(LN(x))

FFN(x) = W₂ * GELU(W₁x + b₁) + b₂
W₁: d_model → 4*d_model
W₂: 4*d_model → d_model

Pre-norm (GPT-2+): LayerNorm BEFORE attention/FFN (more stable training)
Post-norm (original): LayerNorm AFTER residual add
```

### Cross-Attention (Decoder)
```
Q from decoder, K and V from encoder:
CrossAttention = softmax(Q_dec * K_encᵀ / √dₖ) * V_enc

Used in: DETR object detection, image captioning, Stable Diffusion
```

---

## 8. CNN ARCHITECTURES

### ResNet
```
Residual connection: y = F(x) + x
  - Skip connection every 2 layers (BasicBlock) or 3 layers (Bottleneck)
  - Bottleneck: 1x1 (reduce) → 3x3 → 1x1 (expand)
  - Solves vanishing gradient in deep networks
  - ResNet-50: [3, 4, 6, 3] bottleneck blocks
```

### Key Design Principles
```
VGG:         3x3 convs stacked, simple and deep
Inception:   Parallel branches (1x1, 3x3, 5x5, pool) → concat
ResNeXt:     Grouped convolutions (cardinality)
DenseNet:    Dense connections — each layer receives all previous features
EfficientNet: Compound scaling (depth × width × resolution)
ConvNeXt:    Modernized ResNet with transformer tricks (7x7 depthwise, GELU, LN)
```

### Receptive Field
```
RF = 1 + Σₗ (kₗ - 1) * Πᵢ₌₁ˡ⁻¹ sᵢ

k = kernel size, s = stride at each layer
Dilated/atrous conv: increases RF without more params
  effective_k = k + (k-1)(d-1)    where d = dilation rate
```

---

## 9. VISION TRANSFORMERS

### ViT (Vision Transformer)
```
1. Split image into patches (16x16)
2. Flatten patches → linear projection → patch embeddings
3. Prepend [CLS] token + add positional embeddings
4. Feed through standard Transformer encoder
5. [CLS] token → MLP head for classification

Input: (B, 3, 224, 224) → patches: (B, 196, 768) for 16x16 patches
Requires large-scale pretraining (ImageNet-21k or JFT)
```

### DeiT
```
Data-efficient Image Transformer
- Knowledge distillation from CNN teacher
- Distillation token (like CLS token) trained to match teacher output
- Makes ViT trainable on ImageNet-1k alone
```

### Swin Transformer
```
- Shifted Window attention (local, not global)
- Window size typically 7x7
- Shifted by half window every other layer for cross-window connections
- Hierarchical: patch merging for multi-scale features
- Linear complexity O(n) vs O(n²) for standard attention
- Good backbone for detection/segmentation (replaces ResNet in FPN)
```

---

## 10. SELF-SUPERVISED LEARNING

```
Contrastive:
  SimCLR:    Two augmented views → same encoder → NT-Xent loss
  MoCo:      Momentum encoder + queue of negative keys
  CLIP:      Image-text contrastive (align image & text embeddings)

Non-contrastive:
  BYOL:      Online + target (EMA) network, no negatives needed
  DINO:      Self-distillation, centering + sharpening, no labels

Masked Image Modeling:
  MAE:       Mask 75% of patches → reconstruct pixels (ViT encoder-decoder)
  BEiT:      Mask patches → predict visual tokens (dVAE tokenizer)

Key idea: Learn representations without labels, then fine-tune
```

---

## 11. SEGMENTATION

```
Semantic:    Per-pixel class label (no instance distinction)
Instance:    Detect + segment each object instance
Panoptic:    Semantic + Instance (stuff + things)

FCN:         Fully convolutional, skip connections from encoder
U-Net:       Encoder-decoder with skip connections (concat)
DeepLab:     Atrous/dilated convolutions + ASPP (multi-scale) + CRF
Mask R-CNN:  Faster R-CNN + mask branch (per-instance binary mask)
SAM:         Segment Anything Model — prompted segmentation (foundation model)

Loss: CE per pixel, Dice loss (good for imbalanced)
Dice = 2|A∩B|/(|A|+|B|)
Dice Loss = 1 - Dice
```

---

## 12. GENERATIVE MODELS

### Diffusion Models
```
Forward:   q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)
           q(xₜ|x₀) = N(xₜ; √ᾱₜx₀, (1-ᾱₜ)I)    (closed form)
           ᾱₜ = Π αₛ,  αₜ = 1-βₜ

Reverse:   pθ(xₜ₋₁|xₜ) — learned by neural network
           Network predicts noise ε (or x₀, or v)

Training:  L = E[‖ε - εθ(xₜ, t)‖²]     (simple MSE on noise)

Sampling:  xₜ₋₁ = (1/√αₜ)(xₜ - (βₜ/√(1-ᾱₜ))εθ(xₜ,t)) + σₜz

DDPM: 1000 steps, slow sampling
DDIM: deterministic, skip steps, faster
Classifier-free guidance: ε̃ = εᵤ + w*(ε_c - εᵤ)    (w=7.5 typical)
```

### VAE
```
Encoder:  q(z|x)  ≈ N(μ, σ²)
Decoder:  p(x|z)
Loss:     L = -E_q[log p(x|z)] + KL(q(z|x) ‖ p(z))
          = Reconstruction + KL divergence
Reparameterization trick: z = μ + σ * ε,  ε ~ N(0,1)
```

---

## 13. TRAINING TRICKS & PRACTICAL TIPS

```
Mixed Precision (FP16/BF16):
  - Use `torch.autocast(...)` + `GradScaler` when FP16 is enabled
  - 2x faster, half memory, minimal accuracy loss
  - BF16 preferred (larger dynamic range, no scaling needed)

Gradient Accumulation:
  - Simulate larger batch by accumulating gradients over N steps
  - optimizer.step() every N mini-batches

Multi-GPU:
  - DataParallel (DP): simple but suboptimal (master GPU bottleneck)
  - DistributedDataParallel (DDP): one process per GPU, all-reduce
  - FSDP: Fully Sharded Data Parallel (shard params across GPUs)

EMA:
  - θ_ema = α*θ_ema + (1-α)*θ
  - Smoother, often better test performance
  - α = 0.999 or 0.9999

Debugging Checklist:
  ✓ Overfit on 1 batch first (loss should → 0)
  ✓ Check learning rate (too high → diverge, too low → stuck)
  ✓ Visualize predictions during training
  ✓ Monitor gradient norms
  ✓ Check data loading (augmentations, normalization)
  ✓ Verify loss is going down on training set
```

---

## 14. EVALUATION METRICS SUMMARY

```
Classification:
  Accuracy, Top-5 Accuracy
  Precision = TP/(TP+FP)
  Recall = TP/(TP+FN)
  F1 = 2*P*R/(P+R)
  AUC-ROC, AUC-PR

Detection:
  mAP@0.5, mAP@[.5:.95] (COCO)
  AP per class

Segmentation:
  mIoU = mean over classes of IoU per class
  Pixel Accuracy
  Dice coefficient

Tracking:
  MOTA, MOTP, IDF1, HOTA

Depth:
  Abs Rel = |d-d*|/d*
  δ < 1.25 (% of pixels within ratio threshold)
  RMSE

Reconstruction:
  PSNR = 10*log10(MAX²/MSE)
  SSIM = structural similarity (luminance, contrast, structure)
  LPIPS = perceptual distance (lower = more similar)

Calibration:
  ECE (Expected Calibration Error)
  Brier score
```

---

## 15. LAST-MINUTE DEBUGGING & TRIAGE

```
Train loss ↓, val loss ↑:
  Overfitting → more data/augmentation, regularize, early stop, smaller model

Train loss high, val loss high:
  Underfitting or optimization issue → larger model, train longer, reduce regularization, tune LR

Loss explodes / NaNs:
  LR too high, bad normalization, mixed-precision instability, bad labels, division by zero

Accuracy looks good but product performance is bad:
  Wrong metric, class imbalance, threshold issue, leakage, distribution shift

Model is confident and wrong:
  Miscalibration → label noise, distribution shift, overfitting; consider temperature scaling
```

### Questions Worth Answering Out Loud
- What is the actual target metric, and does it match the business cost?
- What changed first: data, labels, augmentations, model, or optimization?
- Can the model overfit a tiny subset?
- Is the baseline actually competitive?
