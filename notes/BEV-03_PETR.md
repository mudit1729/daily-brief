# Paper 3: PETR

**Full title:** PETR: Position Embedding Transformation for Multi-View 3D Object Detection
**Authors:** Yingfei Liu, Tiancai Wang, Xiangyu Zhang, Jian Sun (MEGVII)
**Venue:** ECCV 2022
**arXiv:** 2203.05625

---

## 1) One-page Overview

**Tasks solved:** Multi-view 3D object detection [PETR | Sec 1]

**Sensors assumed:** Multi-camera (6 cameras on nuScenes) [PETR | Sec 4]

**Key novelty:**

- **3D Position Embedding (3D PE):** Encodes 3D positional information into 2D image features, enabling standard transformer cross-attention to implicitly reason in 3D [PETR | Sec 3.2]
- Generates 3D coordinates for every pixel via camera frustum, transforms them to a shared 3D space, and encodes them as positional embeddings [PETR | Sec 3.2]
- Eliminates the need for explicit 3D-to-2D projection (as in DETR3D) — cross-attention between queries and 3D-PE-enhanced features handles geometry implicitly [PETR | Sec 3.1]
- Simple architecture: backbone → 3D PE → standard transformer decoder → detection heads [PETR | Sec 3]
- Extends to temporal via PETRv2 (concatenating multi-frame features with aligned 3D PEs) [PETRv2]

> **If you only remember 3 things:**
>
> 1. 3D position embeddings on image features replace explicit geometric projection
> 2. Standard cross-attention (not feature sampling) — queries attend to all 3D-PE features
> 3. Simpler than DETR3D (no projection + grid_sample), but more compute for cross-attention

---

## 2) Problem Setup and Outputs (Precise)

**Input tensors:** Not specified. Plausible defaults (not from paper): reuse the corresponding input definition from the referenced baseline paper.

| Tensor | Shape | Description |
|--------|-------|-------------|
| Images | `(B, N, 3, H, W)` | N=6 cameras |
| Intrinsics | `(B, N, 3, 3)` | Camera K matrices |
| Extrinsics | `(B, N, 4, 4)` | Camera-to-ego transforms |

**Outputs:** Not specified. Plausible defaults (not from paper): reuse the corresponding output definition from the referenced baseline paper.

| Output | Shape | Description |
|--------|-------|-------------|
| 3D boxes | `(B, Q, 10)` | Center, size, rotation, velocity |
| Class scores | `(B, Q, C_cls)` | Per-query class probabilities |

Q = 900 queries [PETR | Sec 4].

---

## 3) Coordinate Frames and Geometry

**Frames:**

- **Camera frustum frame:** For each pixel (u, v) and depth d, create 3D point `K^{-1} · (u·d, v·d, d)^T`
- **Ego/LiDAR frame:** All frustum points transformed to a shared frame via extrinsics [PETR | Sec 3.2]

**3D Position Embedding generation:**

1. For each pixel in each camera's feature map, create D 3D coordinates along the camera ray (D depths, linearly spaced)
2. Transform all coordinates to ego frame using extrinsics
3. Normalize coordinates to [0, 1] range
4. Pass through a small MLP to produce the 3D PE [PETR | Sec 3.2, Eq 3]

**Key difference from DETR3D:** PETR creates a dense 3D representation (every pixel at multiple depths) as position embeddings, while DETR3D projects sparse reference points. The geometry is baked into the PE, not into the attention mechanism.

**Geometry sanity checks:**

| # | What to verify | Expected outcome | Failure signature |
|---|---------------|------------------|-------------------|
| 1 | 3D coordinates for each camera form correct frustum in ego frame | 6 fan-shaped frustums covering 360° | Frustums overlapping incorrectly or pointing wrong direction |
| 2 | Normalized coordinates in [0, 1] | All values between 0 and 1 after normalization | Values outside [0,1] → PE MLP sees out-of-distribution inputs |
| 3 | After extrinsic transform, 3D PEs from different cameras are in same frame | Points from adjacent cameras overlap at boundaries | Discontinuities at camera boundaries |
| 4 | PE MLP output magnitude | Comparable to image feature magnitude | PE overwhelms or is negligible compared to image features |
| 5 | Depth sampling covers object range | Depth bins span from near to far field | Missing short-range or long-range objects |

---

## 4) Architecture Deep Dive (Module-by-module)

**Block diagram:**

```
Images (B,N,3,H,W)
       │
       ▼
┌─────────────┐
│  Backbone    │  ResNet-50/101 + FPN
│  + Neck      │
└──────┬──────┘
       │ (B, N, C, H/s, W/s)
       ▼
┌──────────────────────┐
│  3D Position Embed   │  Frustum coords → extrinsic transform → normalize → MLP
│  (B, N, C, H/s, W/s)│
└──────┬───────────────┘
       │
       ▼  image_feats + 3D_PE (added or concatenated)
┌──────────────────────┐
│  Transformer Decoder  │
│  Object queries (Q,C) │
│  Cross-attn: queries  │  ← attends to (N*H/s*W/s) key/value tokens with 3D PE
│  Self-attn: queries   │
│  (6 layers)           │
└──────┬───────────────┘
       │
       ▼
┌──────────────┐
│  Det Heads    │  cls + reg
└──────────────┘
```

**Modules:**

| Module | Purpose | Input → Output | Key operations |
|--------|---------|---------------|----------------|
| Backbone + FPN | Extract features | `(B*N, 3, H, W)` → `(B*N, 256, H/16, W/16)` | ResNet + FPN |
| **3D PE Generator** | Create position embeddings | Intrinsics + extrinsics → `(B, N, 256, H/16, W/16)` | Create frustum → transform → normalize → MLP [PETR | Sec 3.2] |
| Key/Value features | Combine image feats + 3D PE | Two tensors → `(B, N*H'*W', 256)` | Keys = image_feat + 3D_PE; Values = image_feat |
| Query embeddings | Learnable | → `(Q, 256)` | Learned parameters |
| Decoder self-attn | Query interaction | `(Q, 256)` → `(Q, 256)` | Multi-head self-attention |
| Decoder cross-attn | Query ↔ image features | `(Q, 256)` query, `(N*H'*W', 256)` kv → `(Q, 256)` | **Standard cross-attention — this is where 3D reasoning happens via 3D PE** |
| Cls head | Classification | `(Q, 256)` → `(Q, C_cls)` | Linear |
| Reg head | Box regression | `(Q, 256)` → `(Q, 10)` | Linear; center + size + rot + vel |

**Where BEV happens:** No explicit BEV grid. 3D reasoning is entirely implicit through cross-attention between object queries and 3D-PE-enhanced image features [PETR | Sec 3.1].

---

## 5) Forward Pass Pseudocode (Shape-annotated)

```python
def forward(images, intrinsics, extrinsics):
    B, N, _, H, W = images.shape

    # 1. Backbone + FPN
    feats = backbone_fpn(images.view(B*N, 3, H, W))  # (B*N, 256, H', W')
    feats = feats.view(B, N, 256, H_f, W_f)           # (B, N, 256, H', W')

    # 2. Generate 3D position embeddings
    # Create frustum: for each (u,v) pixel, D depths
    frustum = create_frustum(D, H_f, W_f, intrinsics)  # (B, N, D, H', W', 3)
    frustum_ego = apply_extrinsics(frustum, extrinsics)  # (B, N, D, H', W', 3)
    frustum_norm = normalize_coords(frustum_ego)          # (B, N, D, H', W', 3) in [0,1]

    # Mean pool over depth dimension to get per-pixel 3D coord
    coords = frustum_norm.mean(dim=2)    # (B, N, H', W', 3)
    # OR: concatenate over D → (B, N, D*3, H', W') then MLP

    pos_embed_3d = pe_mlp(coords)        # (B, N, 256, H', W')

    # 3. Prepare keys/values for cross-attention
    keys = feats + pos_embed_3d           # (B, N, 256, H', W')
    keys = keys.flatten(1, 3)             # (B, N*H'*W', 256)
    values = feats.flatten(1, 3)          # (B, N*H'*W', 256)

    # 4. Object queries
    queries = query_embed.weight.unsqueeze(0).expand(B, Q, 256)  # (B, Q, 256)

    # 5. Decoder (6 layers)
    for layer in decoder_layers:
        queries = layer.self_attn(queries)                    # (B, Q, 256)
        queries = layer.cross_attn(queries, keys, values)     # (B, Q, 256)
        queries = layer.ffn(queries)                          # (B, Q, 256)

    # 6. Heads
    cls_scores = cls_head(queries)   # (B, Q, C_cls)
    box_params = reg_head(queries)   # (B, Q, 10)

    return cls_scores, box_params
```

**Activation/scale sanity:**

- `pos_embed_3d`: should be same magnitude as image features (both ~O(1) after normalization)
- `keys`: addition of features + PE should not blow up (both O(1))
- Cross-attention over N*H'*W' tokens: for 6 cameras at 50×88 feature map = 26,400 tokens — attention matrix is Q × 26,400 — this is the memory bottleneck
- `cls_scores`: raw logits for focal loss

---

## 6) Heads, Targets, and Losses

Same structure as DETR3D: focal loss for classification, L1 loss for regression, Hungarian matching.

| Loss | Formula | Weight |
|------|---------|--------|
| Focal loss (cls) | α=0.25, γ=2.0 | 2.0 |
| L1 (reg) | Normalized box parameters | 0.25 |

*Plausible defaults (not from paper): exact weights may follow DETR3D conventions.*

**Loss debugging checklist:**

| # | Bug | Symptom | Quick test |
|---|-----|---------|------------|
| 1 | 3D PE not aligned with image features | Cross-attention learns nothing from position | Visualize attention weights: should peak near query's predicted 3D location |
| 2 | PE normalization range wrong | MLP produces garbage embeddings | Check input to PE MLP is in expected range |
| 3 | Cross-attention memory overflow | OOM with large feature maps | Reduce image resolution or use deformable attention |
| 4 | Queries collapse (same output) | All detections at same location | Add diversity loss or check self-attention is working |
| 5 | Missing PE at eval time | Performance drops dramatically at inference | Ensure PE generation uses eval-time calibration |

---

## 7) Data Pipeline and Augmentations

Similar to DETR3D. Key difference: if image augmentation changes the geometry (resize, crop), the frustum coordinates and 3D PE must be recomputed with updated intrinsics [PETR | Sec 3.2].

**Temporal (PETRv2):** Concatenate features from previous frame with ego-motion-aligned 3D PEs.

**Augmentation safety:**

| Augment | Risk | Validation |
|---------|------|------------|
| Any geometric image augment | 3D PE must be recomputed with updated intrinsics | Verify PE coordinates match augmented image geometry |
| Color jitter | No geometric effect | Safe — 3D PE unaffected |

---

## 8) Training Pipeline

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-4 [PETR | Sec 4] |
| Backbone LR | 2e-5 (10× lower) |
| Epochs | 24 on nuScenes [PETR | Sec 4] |
| Batch size | 8 (1 per GPU × 8 GPUs) *Plausible defaults (not from paper)* |
| GPUs | 8× V100 *Plausible defaults (not from paper)* |
| LR schedule | Cosine or step decay |

**Stability & convergence:**

| Failure | Symptom | Fix |
|---------|---------|-----|
| Cross-attention OOM | Crash during forward pass | Use deformable attention (as in PETRv2) or reduce resolution |
| 3D PE dominates image features | Position-only prediction without appearance | Scale down PE magnitude; use addition not concatenation |
| Slow convergence | Worse than DETR3D at same epochs | Check that 3D PEs are correct; standard attention is slower to converge than feature sampling |

---

## 9) Dataset + Evaluation Protocol

Not specified. Plausible defaults (not from paper): use DETR3D-style nuScenes evaluation protocol (mAP + NDS + TP metrics).

---

## 10) Results Summary + Ablations

**Main results [PETR | Table 1]:** On nuScenes val with ResNet-50:

- Not specified in provided inputs. Plausible defaults (not from paper): consult the cited paper table/benchmark entry for exact values.
- With ResNet-101 + CBGS: competitive with DETR3D

**Top 3 ablations [PETR | Table 2–4]:**

1. **3D PE vs. 2D PE:** 3D PE improves over 2D PE in ablations [PETR | Table 2–4]; exact delta Not specified in provided inputs.
2. **Number of depth bins for PE:** D=64 or D=128 frustum depths for PE construction; more bins give marginal improvement
3. **PE architecture:** MLP-based PE outperforms sinusoidal encoding of 3D coordinates

---

## 11) Practical Insights

**10 engineering takeaways:**

1. PETR is architecturally simpler than DETR3D (no projection + grid_sample), but the full cross-attention over all image tokens is expensive.
2. The 3D PE MLP is lightweight and critical for performance [PETR | Sec 3.2]; exact layer count Not specified in provided inputs.
3. PETRv2 addresses PETR's temporal limitation by concatenating aligned multi-frame features.
4. Deformable attention (as in Deformable DETR) dramatically reduces PETR's memory cost.
5. 3D PE provides an elegant alternative to explicit geometry but makes debugging harder (geometry is implicit).
6. For high-resolution features, PETR's memory scales as Q × (N × H' × W') — can be prohibitive.
7. PE normalization to [0, 1] is critical; unnormalized coordinates cause MLP training instability.
8. The camera frustum depth range for PE should cover the detection range (not just near-field).
9. PETR generalizes to different camera setups more easily than LSS (no BEV grid to reconfigure).
10. Combined with temporal alignment, PETR-style 3D PE becomes a strong baseline for streaming detection.

**5 gotchas:**

1. Forgetting to recompute 3D PE after image augmentation
2. Attention memory explosion with high-resolution multi-view features
3. PE magnitude mismatch with image features (need careful normalization)
4. Depth range for frustum PE must match dataset's object distance range
5. PETRv2 temporal alignment requires correct ego-motion compensation of 3D PE coordinates

**Tiny-subset overfit plan:**

| Parameter | Value |
|-----------|-------|
| Subset | 4 scenes |
| Augments | None |
| Expected | mAP > 0.7 on training set within 3000 steps |
| If fails | Check 3D PE generation; visualize attention patterns; verify coordinate frame alignment |

---

## 12) Minimal Reimplementation Checklist

**Build order:**

1. Data loading (nuScenes, 6 cameras)
2. Backbone + FPN
3. **3D PE generator:** frustum creation → extrinsic transform → normalize → MLP
4. Standard transformer decoder with cross-attention
5. Det heads + Hungarian matching + focal loss + L1 loss
6. Evaluation

**Unit tests:**

| Test | Check |
|------|-------|
| `test_frustum_coords` | Frustum points form correct shape; depths monotonically increasing |
| `test_extrinsic_alignment` | Frustum points from all cameras in same ego frame |
| `test_pe_normalization` | All normalized coords in [0, 1] |
| `test_cross_attn_shape` | Output (B, Q, C) with Q=900 |
| `test_pe_gradient` | PE MLP receives non-zero gradients |

**Minimal sanity scripts:**

**Script 1:** Visualize 3D PE by coloring image pixels by their 3D z-coordinate → should show depth gradient
**Script 2:** Overfit 4 scenes → mAP > 0.7 in 3000 steps
**Script 3:** Compare PETR predictions with GT boxes projected onto BEV → should overlap

---

_Source: extracted and adapted from `BEV_Papers_Technical_Dossier.md` in this workspace._
