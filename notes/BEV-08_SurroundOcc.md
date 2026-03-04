# Paper 8: SurroundOcc

**Full title:** SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving
**Authors:** Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, Jiwen Lu (Tsinghua / PhiGent)
**Venue:** ICCV 2023
**arXiv:** 2303.09551

---

## 1) One-page Overview

**Tasks solved:** 3D semantic occupancy prediction (voxel-level scene understanding) [SurroundOcc | Sec 1]

**Sensors assumed:** Multi-camera (6 cameras on nuScenes / Occ3D benchmark) [SurroundOcc | Sec 4]

**Key novelty:**

- **Multi-scale 3D occupancy prediction:** Predicts dense 3D semantic occupancy at multiple resolutions using a coarse-to-fine approach [SurroundOcc | Sec 3.2]
- **Dense occupancy GT generation:** Proposes a pipeline to create dense occupancy labels from sparse LiDAR using multi-frame aggregation + Poisson reconstruction + semantic propagation [SurroundOcc | Sec 3.1]
- Uses 2D-to-3D feature lifting (LSS-style or spatial cross-attention) to build 3D volume features [SurroundOcc | Sec 3.2]
- 3D UNet-style decoder for coarse-to-fine refinement of occupancy predictions
- Evaluated on nuScenes-Occupancy (Occ3D) benchmark

> **If you only remember 3 things:**
>
> 1. Dense 3D occupancy prediction (not just boxes or BEV segmentation — full voxel grid)
> 2. Coarse-to-fine 3D UNet refinement from multi-scale voxel features
> 3. Novel dense GT generation pipeline from sparse LiDAR

---

## 2) Problem Setup and Outputs (Precise)

**Inputs:** Standard multi-camera setup.

| Tensor | Shape | Description |
|--------|-------|-------------|
| Images | `(B, N, 3, H, W)` | N=6 cameras |
| Intrinsics | `(B, N, 3, 3)` | Camera K |
| Extrinsics | `(B, N, 4, 4)` | Camera-to-ego |

**Outputs:**

| Output | Shape | Description |
|--------|-------|-------------|
| Occupancy grid | `(B, C_cls, X, Y, Z)` | Per-voxel semantic class; typical 200×200×16 |

**Voxel grid [SurroundOcc | Sec 3]:**

| Parameter | Value |
|-----------|-------|
| X range | [-40m, 40m] or [-50m, 50m] |
| Y range | [-40m, 40m] or [-50m, 50m] |
| Z range | [-5m, 3m] or [-1m, 5.4m] |
| Resolution | 0.4m or 0.5m per voxel |
| Grid size | 200×200×16 (typical) |
| Classes | 16 or 17 (including free space) |

---

## 3) Coordinate Frames and Geometry

**Frames:** Camera, ego, voxel grid (3D extension of BEV).

**Voxel grid:** 3D grid in ego frame. Unlike BEV which collapses height, occupancy preserves the Z dimension.

**2D-to-3D lifting:** Either LSS-style (lift to 3D volume, then pool to voxels) or spatial cross-attention (BEVFormer-style, but querying 3D voxels instead of 2D BEV).

**Multi-scale approach [SurroundOcc | Sec 3.2]:**

- Level 1: Coarse grid (e.g., 50×50×4)
- Level 2: Medium grid (e.g., 100×100×8)
- Level 3: Fine grid (e.g., 200×200×16)
- 3D UNet decoder upsamples coarse → fine with skip connections

**Geometry sanity checks:**

| # | What to verify | Expected outcome | Failure signature |
|---|---------------|------------------|-------------------|
| 1 | Voxel grid covers driving scene | Z range includes ground and typical vehicle heights | Vehicles or ground clipped |
| 2 | 3D feature lifting covers full volume | Features distributed across all Z levels | All features at Z=0 (collapsed to ground) |
| 3 | Multi-scale consistency | Coarse prediction roughly matches fine prediction | Conflicting predictions at different scales |
| 4 | Dense GT quality | Filled voxels match visible surfaces | Holes in GT or incorrect semantic labels |
| 5 | Free-space vs. unknown handling | Model distinguishes empty from unobserved | Predicting "free" everywhere or ignoring occlusion |

---

## 4) Architecture Deep Dive (Module-by-module)

**Block diagram:**

```
Images (B,N,3,H,W)
       │
       ▼
┌──────────────┐
│ 2D Backbone  │  ResNet-101 + FPN
│ + Neck       │
└──────┬───────┘
       │ Multi-scale 2D features
       ▼
┌──────────────────┐
│ 2D → 3D Lifting  │  LSS-style or spatial cross-attention
│                   │  → 3D volume features at multiple scales
└──────┬───────────┘
       │ 3D volume: (B, C, X/s, Y/s, Z/s) per scale
       ▼
┌──────────────────┐
│ 3D UNet Decoder  │  Coarse → fine with 3D deconv + skip connections
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Occupancy Head   │  Per-voxel classification at each scale
└──────────────────┘
  → (B, C_cls, 200, 200, 16)
```

**Modules:**

| Module | Purpose | Input → Output | Key operations |
|--------|---------|---------------|----------------|
| 2D Backbone + FPN | Image features | `(B*N, 3, H, W)` → multi-scale 2D features | ResNet-101 + FPN |
| 2D → 3D Lifting | Create 3D volumes | 2D features + calibration → `(B, C, X', Y', Z')` | LSS lift or cross-attention to 3D queries |
| 3D UNet Encoder | Downsample 3D | `(B, C, X, Y, Z)` → multi-scale 3D features | 3D convolution + pooling |
| 3D UNet Decoder | Upsample 3D | Multi-scale → `(B, C, X, Y, Z)` | 3D deconv + skip connections |
| Occupancy Head | Per-voxel class | `(B, C, X, Y, Z)` → `(B, C_cls, X, Y, Z)` | 1×1×1 conv at each scale |

**Where BEV happens:** The 2D→3D lifting creates a 3D volume (not just BEV). BEV is a 2D slice of this; occupancy uses the full 3D. The lifting module is the critical step.

---

## 5) Forward Pass Pseudocode (Shape-annotated)

```python
def forward(images, intrinsics, extrinsics):
    B, N, _, H, W = images.shape

    # 1. 2D features
    feats_2d = backbone_fpn(images.view(B*N, 3, H, W))  # multi-scale 2D

    # 2. 2D → 3D lifting (at coarsest scale)
    vol_coarse = lift_to_3d(feats_2d, intrinsics, extrinsics)  # (B, C, 50, 50, 4)

    # 3. 3D UNet: encode → decode with multi-scale
    # Encoder
    vol_s2 = conv3d_down(vol_coarse)   # (B, 2C, 25, 25, 2)
    vol_s4 = conv3d_down(vol_s2)       # (B, 4C, 12, 12, 1)

    # Decoder (with skip connections)
    vol_up2 = deconv3d(vol_s4) + vol_s2  # (B, 2C, 25, 25, 2)
    vol_up1 = deconv3d(vol_up2) + vol_coarse  # (B, C, 50, 50, 4)
    vol_fine = upsample_3d(vol_up1)      # (B, C, 200, 200, 16)

    # 4. Occupancy heads at each scale
    occ_coarse = occ_head_coarse(vol_coarse)  # (B, C_cls, 50, 50, 4)
    occ_mid = occ_head_mid(vol_up1)           # (B, C_cls, 100, 100, 8)
    occ_fine = occ_head_fine(vol_fine)         # (B, C_cls, 200, 200, 16)

    return occ_fine, [occ_coarse, occ_mid]  # multi-scale supervision
```

---

## 6) Heads, Targets, and Losses

**Head:** Per-voxel classification (1×1×1 conv).

| Loss | Formula | Weight |
|------|---------|--------|
| Cross-entropy (per-voxel, per-scale) | `CE(occ_pred, occ_gt)` | Weighted by scale: coarse=1, mid=1, fine=1 |
| Lovász-softmax (optional) | IoU-friendly surrogate | *Plausible defaults (not from paper)* |
| Scene-class affinity loss | Encourages spatial smoothness | *Plausible defaults (not from paper)* |

**Class-balanced CE:** Free space dominates (~90% of voxels) → use class-balanced sampling or class weights [SurroundOcc | Sec 3].

**Loss debugging checklist:**

| # | Bug | Symptom | Quick test |
|---|-----|---------|------------|
| 1 | Free-space dominance | Predicts all-free (high accuracy but useless) | Check per-class IoU; add class weights |
| 2 | Voxel GT misaligned with predictions | Loss doesn't decrease | Visualize GT occupancy alongside input images |
| 3 | Z-axis convention mismatch | Occupied voxels at wrong height | Check Z range and sign convention |
| 4 | Multi-scale GT resolution | GT at wrong resolution for each scale | Downsample GT correctly for coarse supervision |
| 5 | Semantic label mapping | Wrong class IDs between GT and prediction | Verify class mapping matches dataset specification |

---

## 7) Data Pipeline and Augmentations

**Dense GT generation [SurroundOcc | Sec 3.1]:**

1. Aggregate LiDAR points from multiple sweeps (±5 frames)
2. Transform all to current ego frame using ego poses
3. Apply Poisson reconstruction to fill gaps
4. Propagate semantic labels from annotated 3D boxes and map
5. Voxelize the dense point cloud

**Augmentations:** Similar to BEVDet — image augments + 3D augments on voxel labels.

---

## 8) Training Pipeline

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 2e-4 |
| Epochs | 24 *Plausible defaults (not from paper)* |
| GPUs | 8× A100 *Plausible defaults (not from paper)* |
| Backbone | ResNet-101 pretrained |

---

## 9) Dataset + Evaluation Protocol

**Dataset:** nuScenes-Occupancy (Occ3D benchmark) or custom dense occupancy labels.

**Metrics:**

| Metric | Description |
|--------|-------------|
| mIoU | Mean IoU across all semantic classes (volumetric) |
| IoU per class | Per-class volumetric IoU |

- Not specified in provided inputs. Plausible defaults (not from paper): consult the cited paper table/benchmark entry for exact values.

**Top ablations:**

1. Dense GT (multi-frame) vs. sparse GT: dense GT provides significantly better training signal
2. Multi-scale supervision improves over single-scale in ablations [SurroundOcc | Table 6]; exact delta Not specified in provided inputs.
3. 3D UNet depth: more decoder layers improve fine-grained predictions

---

## 10) Results Summary + Ablations

- Not specified in provided inputs. Plausible defaults (not from paper): summarize main benchmark numbers directly from cited tables.

---

## 11) Practical Insights

**10 engineering takeaways:**

1. Occupancy prediction extends BEV to full 3D — but memory scales as X×Y×Z (much larger than X×Y).
2. Dense GT generation is critical — sparse LiDAR GT limits occupancy quality.
3. Class imbalance is severe: free space >> occupied → class weighting essential.
4. 3D UNet is effective for coarse-to-fine refinement but memory-hungry.
5. Multi-scale supervision at coarse levels helps gradient flow to early layers.
6. The 2D→3D lifting module is the same as for BEV detection — reuse existing code.
7. Occupancy is complementary to detection: it handles non-boxable objects (walls, vegetation, terrain).
8. Z-range design matters: too tall wastes voxels on sky; too short clips elevated objects.
9. Inference is slower than BEV detection due to the 3D decoder — consider sparse representations.
10. Occupancy evaluation (mIoU) is stricter than detection (mAP) — small errors in geometry hurt significantly.

**5 gotchas:**

1. Z-axis convention: nuScenes Z is up, but some code uses Z as height index (inverted) → verify
2. Free-space handling: some benchmarks distinguish "free" from "unknown" — different loss treatment
3. Dense GT quality depends on LiDAR aggregation — moving objects create artifacts
4. 3D convolution memory: 200×200×16 × C channels is large — reduce C or resolution
5. Voxel grid origin: centered on ego? Corner? Off-by-half-voxel errors are common

**Tiny-subset overfit plan:**

| Parameter | Value |
|-----------|-------|
| Subset | 2 scenes |
| Grid | Small (50×50×4) for speed |
| Expected | mIoU > 0.6 on training set within 2000 steps |
| If fails | Check GT voxel visualization; verify 2D→3D lifting quality |

---

## 12) Minimal Reimplementation Checklist

**Build order:**

1. Dense occupancy GT generation pipeline (multi-frame LiDAR aggregation)
2. 2D backbone + FPN
3. 2D→3D lifting (reuse LSS or BEVFormer-style)
4. 3D UNet encoder-decoder
5. Per-voxel classification head + class-balanced CE loss
6. Multi-scale supervision
7. Evaluation (mIoU)

**Sanity scripts:**

**Script 1:** Visualize GT occupancy voxels colored by class → should match scene structure
**Script 2:** Overfit 2 scenes at low resolution → mIoU > 0.6
**Script 3:** Compare predicted occupancy with LiDAR point cloud → should align

---

_Source: extracted and adapted from `BEV_Papers_Technical_Dossier.md` in this workspace._
