# Paper 1: Lift, Splat, Shoot (LSS)

**Full title:** Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D
**Authors:** Jonah Philion, Sanja Fidler (NVIDIA / U of T / Vector Institute)
**Venue:** ECCV 2020
**arXiv:** 2008.05711

---

## 1) One-page Overview

**Tasks solved:** BEV semantic segmentation (vehicle segmentation, road segmentation); motion planning via cost-map shooting [LSS | Sec 1, Sec 5]

**Sensors assumed:** Multi-camera (tested with nuScenes 6-camera rig); no LiDAR at inference [LSS | Sec 4]

**Key novelty:**

- Introduces the **Lift-Splat** paradigm: predict a categorical depth distribution per pixel, outer-product with image features to create a frustum point cloud, then scatter (splat) onto a BEV grid [LSS | Sec 3.1–3.2]
- Learns depth implicitly via task loss alone — no explicit depth supervision required [LSS | Sec 3.1]
- Handles arbitrary numbers and configurations of cameras by design — the representation is permutation-invariant over cameras [LSS | Sec 3.2]
- Demonstrates end-to-end planning ("Shoot") by scoring template trajectories on a BEV cost map [LSS | Sec 5]
- Entire pipeline is fully differentiable, enabling joint optimization of depth and BEV features [LSS | Sec 3]

**If you only remember 3 things:**

1. Per-pixel categorical depth distribution (softmax over D bins) outer-producted with image features → frustum point cloud → pillar pooling onto BEV grid
2. No explicit depth supervision — depth emerges from the task loss alone
3. This is the foundational "LSS-style lift" used by BEVDet, BEVDepth, BEVFusion, and many others

---

## 2) Problem Setup and Outputs (Precise)

**Input tensors:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| Images | `(B, N, 3, H_img, W_img)` | RGB images from N cameras; nuScenes: N=6, H=128, W=352 after resize [LSS | Sec 4] |
| Intrinsics | `(B, N, 3, 3)` | Camera intrinsic matrices K |
| Extrinsics | `(B, N, 4, 4)` | Camera-to-ego transformation matrices [R|t] |
| Ego pose | `(B, 4, 4)` | Ego-to-global transform (used for temporal alignment if multi-frame) |

**Outputs:**

| Output | Shape | Description |
|--------|-------|-------------|
| BEV segmentation | `(B, C_cls, Y_bev, X_bev)` | Per-class semantic probability map in BEV; C_cls = number of classes |
| (Optional) Cost map | `(B, 1, Y_bev, X_bev)` | For planning: cost at each BEV cell |

**Label format:** Binary or multi-class BEV segmentation maps rasterized from 3D annotations onto BEV grid. Vehicle footprints projected top-down; road/lane polygons rasterized [LSS | Sec 4].

**Value ranges:** BEV grid covers [-50m, 50m] × [-50m, 50m] at 0.5m resolution → 200×200 grid [LSS | Sec 4]. Depth bins: 4m to 45m, 1m spacing, D=41 bins. *Plausible defaults (not from paper): exact depth range may vary; 4–45m with 1m spacing yielding D=41 is the commonly reported configuration in follow-up works.*

---

## 3) Coordinate Frames and Geometry

**Frames used:**

- **Camera frame:** Origin at optical center; Z forward, X right, Y down (OpenCV convention)
- **Ego frame:** Origin at vehicle rear axle (nuScenes convention); X forward, Y left, Z up
- **BEV grid frame:** Discretized ego frame; origin at grid center; X → columns, Y → rows (or vice versa — see pitfalls)

**Transforms:**

1. **Pixel → camera ray:** `(u, v, 1)^T = K · p_cam` inversion gives ray direction
2. **Camera → ego:** `p_ego = T_ego←cam · p_cam` using extrinsic 4×4 matrix
3. **Ego → BEV grid:** Discretize (x_ego, y_ego) into grid indices via `idx = (x - x_min) / resolution`

**Depth bins:** D discrete depths `d_k ∈ {d_min, d_min+Δ, ..., d_max}`. For each pixel feature at (u, v), create D copies at each depth: `p_cam(u,v,d_k) = d_k · K^{-1} · (u, v, 1)^T` [LSS | Sec 3.1]

**BEV grid definition:**

| Parameter | Value |
|-----------|-------|
| X range | [-50m, 50m] |
| Y range | [-50m, 50m] |
| Resolution | 0.5 m/cell |
| Grid size | 200 × 200 |
| Height | Collapsed (no height bins — sum/pool along Z) |
| Origin | Center of grid = ego position |

**Common pitfalls:**

- X/Y axis swap between ego frame (X-forward) and image-like BEV tensor (row = Y, col = X)
- nuScenes extrinsics are sensor-to-ego, not ego-to-sensor — applying the wrong direction produces mirrored projections
- Intrinsic matrices must be adjusted when images are resized/cropped — failure to do this shifts all projections

**Geometry sanity checks:**

| # | What to verify | Expected outcome | Failure signature |
|---|---------------|------------------|-------------------|
| 1 | Project 3D box corners onto image using K and extrinsics | Boxes overlay correctly on objects in image | Boxes in wrong location or mirrored |
| 2 | Create frustum points for a single camera, transform to ego, plot top-down | Fan-shaped frustum aligned with camera FOV | Frustum pointing wrong direction or behind camera |
| 3 | Splat uniform features and check BEV activation pattern | Each camera covers its expected sector; overlaps in multi-camera regions | Dead zones or all features in one corner |
| 4 | Verify depth bin spacing by plotting bin edges | Monotonically increasing from d_min to d_max | Non-monotonic or negative depths |
| 5 | Check that resized intrinsics match resized image | Focal length scaled by resize factor; principal point shifted | Off-by-one or unscaled intrinsics |

---

## 4) Architecture Deep Dive (Module-by-module)

**Block diagram (ASCII):**

```
Images (B,N,3,H,W)
       │
       ▼
┌─────────────┐
│  Backbone    │  (EfficientNet-B0 or ResNet)
│  + Neck      │
└──────┬──────┘
       │ (B,N,C,H/s,W/s)     s = downsample factor
       ▼
┌─────────────┐
│ Depth Net   │  → (B,N, D+C, H/s, W/s)
│ (1×1 conv)  │    split → depth: (B,N,D,H/s,W/s)
└──────┬──────┘            context: (B,N,C,H/s,W/s)
       │
       ▼
┌─────────────────────┐
│  Outer Product       │  depth_prob × context_feat
│  → Frustum Features │  → (B,N,D,C,H/s,W/s)
└──────┬──────────────┘
       │
       ▼
┌──────────────────────┐
│  Voxel Pooling       │  Frustum → 3D voxels → sum-pool along Z
│  (Splat to BEV grid) │  → (B, C, Y_bev, X_bev)
└──────┬───────────────┘
       │
       ▼
┌─────────────┐
│  BEV Encoder │  (ResNet-like 2D CNN on BEV plane)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Seg Head    │  → (B, C_cls, Y_bev, X_bev)
└─────────────┘
```

**Modules in order:**

| Module | Purpose | Input → Output shape | Key operations |
|--------|---------|---------------------|----------------|
| Image Backbone | Extract multi-scale features | `(B*N, 3, H, W)` → `(B*N, C_feat, H/s, W/s)` | EfficientNet-B0 [LSS | Sec 4]; standard conv + BN + ReLU |
| Neck (FPN or simple) | Combine feature scales | Multi-scale → `(B*N, C, H/s, W/s)` | Upsample + concat or add |
| Depth + Context head | Predict depth distribution + context | `(B*N, C, H/s, W/s)` → `(B*N, D+C, H/s, W/s)` | Single 1×1 conv → split into depth logits (→softmax) and context features |
| **Frustum creation** | Create 3D point cloud per pixel | `(B,N,D,H/s,W/s)` coords + `(B,N,D,C,H/s,W/s)` feats | **This is where BEV happens (step 1)**: outer product `softmax(depth) × context` |
| **Voxel pooling / Splat** | Scatter frustum points onto BEV grid | `(B,N,D,C,H/s,W/s)` → `(B,C,Y,X)` | **This is where BEV happens (step 2)**: transform coords cam→ego, discretize to grid, cumulative sum (pillar pooling) |
| BEV Encoder | Refine BEV features | `(B,C,Y,X)` → `(B,C',Y,X)` | 2D ResNet blocks |
| Segmentation Head | Predict per-cell class | `(B,C',Y,X)` → `(B,C_cls,Y,X)` | 1×1 conv |

**Where BEV happens:** The frustum outer-product + voxel pooling (splat) steps. Specifically: (1) each pixel's depth softmax creates a weighted distribution over D depths; (2) each depth location gets the pixel's context feature scaled by the depth probability; (3) all these weighted features are scattered to BEV grid cells by coordinate transformation and summation [LSS | Sec 3.1–3.2].

**Depth representation:** Categorical distribution over D bins. Softmax converts raw logits to probabilities. The outer product `α(d) · c` where α is the depth probability at bin d and c is the context feature vector creates the "lifted" frustum [LSS | Eq 2–3].

**Depth failure modes:**

| Failure | Symptom | Check |
|---------|---------|-------|
| Depth collapse (all probability on one bin) | BEV features form a thin shell at fixed depth | Visualize per-pixel depth distribution entropy; should be >0 |
| Uniform depth (no learning) | BEV features smeared everywhere | Check depth logit gradients; they should be non-zero |
| Wrong depth range | Objects placed at wrong BEV locations | Overlay predicted frustum on ground truth boxes |

---

## 5) Forward Pass Pseudocode (Shape-annotated)

```python
def forward(images, intrinsics, extrinsics):
    B, N, C_in, H, W = images.shape

    # 1. Backbone: extract image features
    imgs = images.view(B*N, C_in, H, W)
    feats = backbone(imgs)                    # (B*N, C_bb, H//s, W//s)  e.g. s=16
    feats = neck(feats)                       # (B*N, C, H//s, W//s)    C=64 typical

    # 2. Depth + context head
    depth_context = depth_head(feats)         # (B*N, D+C, H//s, W//s)
    depth_logits = depth_context[:, :D]       # (B*N, D, H//s, W//s)
    context = depth_context[:, D:]            # (B*N, C, H//s, W//s)
    depth_probs = softmax(depth_logits, dim=1)  # (B*N, D, H//s, W//s)  ← sums to 1 over D

    # 3. Create frustum coordinates (precomputed once)
    # For each (u, v) in feature map and each depth bin d_k:
    # p_cam = d_k * K^{-1} * [u*s, v*s, 1]^T
    frustum_xyz = create_frustum(D, H//s, W//s, intrinsics)  # (B, N, D, H//s, W//s, 3)

    # 4. Transform frustum to ego frame
    frustum_ego = apply_extrinsics(frustum_xyz, extrinsics)   # (B, N, D, H//s, W//s, 3)

    # 5. Outer product: weight context by depth probability
    volume = depth_probs.unsqueeze(2) * context.unsqueeze(1)
    # depth_probs: (B*N, D, 1, H//s, W//s) × context: (B*N, 1, C, H//s, W//s)
    # → volume: (B*N, D, C, H//s, W//s)
    volume = volume.view(B, N, D, C, H//s, W//s)

    # 6. Splat: scatter to BEV grid by pillar pooling
    bev = voxel_pooling(frustum_ego, volume, bev_grid_config)  # (B, C, Y_bev, X_bev)
    # Implementation: discretize ego coords → grid indices, use cumsum trick for differentiable scatter

    # 7. BEV encoder
    bev = bev_encoder(bev)                    # (B, C', Y_bev, X_bev)

    # 8. Segmentation head
    seg_logits = seg_head(bev)                # (B, C_cls, Y_bev, X_bev)

    return seg_logits
```

**Activation/scale sanity notes:**

- `depth_probs`: each spatial location should sum to 1.0 over D. Not specified in provided inputs: expected entropy range.
- `context` features: typical magnitude O(1) after BN in backbone
- `volume` after outer product: same magnitude as context (scaled by probabilities ≤1)
- `bev` after pooling: magnitude scales with number of contributing pixels per cell; cells with many projections (near cameras) will have larger magnitudes than far cells
- `seg_logits`: unconstrained; apply sigmoid or softmax downstream

---

## 6) Heads, Targets, and Losses

**Heads:**

| Head | Output shape | Activation | Purpose |
|------|-------------|------------|---------|
| Segmentation | `(B, C_cls, 200, 200)` | Sigmoid (binary per class) | BEV semantic segmentation |

**Loss terms:**

| Loss | Formula | Target | Default weight |
|------|---------|--------|----------------|
| Binary cross-entropy | `BCE(σ(logits), y)` per class | Binary BEV masks rasterized from 3D annotations | 1.0 [LSS | Sec 4] |

*Plausible defaults (not from paper): some implementations add focal loss weighting or positive-class overweighting to handle class imbalance (most BEV cells are background).*

**Assignment strategy:** Dense per-pixel — every BEV cell has a ground-truth label. No Hungarian matching needed [LSS | Sec 4].

**Loss debugging checklist:**

| # | Bug | Symptom | Quick test |
|---|-----|---------|------------|
| 1 | BEV label X/Y axes swapped vs. prediction | Loss doesn't decrease; predictions mirrored | Visualize GT labels overlaid on predicted BEV; should align |
| 2 | Class imbalance not addressed | Predicts all-background (low loss but useless) | Check per-class recall; add positive weight to BCE |
| 3 | Sigmoid vs. softmax mismatch | Multi-class: probabilities sum > 1 or network can't assign multiple classes | Verify head activation matches loss (BCE→sigmoid, CE→softmax) |
| 4 | Wrong spatial resolution for labels | Labels and predictions don't align | Assert `labels.shape == logits.shape` before loss |
| 5 | Depth not learning (flat gradient) | BEV features uniform → loss stuck | Check gradient magnitude on depth logits; visualize depth predictions |

---

## 7) Data Pipeline and Augmentations

**Data loading flow:**

1. Load N camera images + intrinsics + extrinsics from nuScenes
2. Resize images to target resolution (e.g., 128×352) [LSS | Sec 4]
3. Adjust intrinsics K to match resize: scale fx, fy, cx, cy
4. Rasterize 3D annotations to BEV grid as binary masks
5. Collate into batch

**Geometric augments:**

- Random image resize/crop → must update intrinsics accordingly
- Random horizontal flip → must mirror extrinsics (negate y-translation, flip rotation)
- *Plausible defaults (not from paper): color jitter, random rotation of BEV grid*

**Temporal sampling:** Single-frame only in original LSS — no temporal fusion [LSS | Sec 3].

**Tricks:**

- Efficient "cumulative sum" trick for differentiable voxel pooling (avoids scatter_add which is slow) [LSS | Sec 3.2]
- *Plausible defaults (not from paper): pre-computing frustum coordinates once per camera config speeds up training*

**Augmentation safety:**

| Augment | Risk | Validation |
|---------|------|------------|
| Image resize without intrinsic update | All projections shift | Project known 3D point after augment; should land on correct pixel |
| Horizontal flip without extrinsic flip | Left-right cameras swap but labels don't | Visualize flipped frustum in ego frame |
| BEV random rotation | Must rotate BEV labels identically | Overlay augmented BEV labels on augmented predictions |

**Minimal-augment debug config:** Disable all augments; use fixed resize only with correctly adjusted intrinsics.

---

## 8) Training Pipeline (Reproducible) (Reproducible) (Reproducible)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam [LSS | Sec 4] |
| Learning rate | 1e-3 *Plausible defaults (not from paper)* |
| Batch size | 4 *Plausible defaults (not from paper)* |
| Epochs | ~50k steps *Plausible defaults (not from paper)* |
| Warmup | Not specified |
| Mixed precision | Not specified |
| Gradient clipping | Not specified |
| Multi-stage | None — single-stage end-to-end |
| GPUs | Not specified. *Plausible defaults (not from paper): 4× V100 or 8× 2080Ti based on era* |

**Stability & convergence:**

| Failure mode | Symptom | Fix |
|-------------|---------|-----|
| Depth collapses to single bin | BEV features look like a thin ring | Lower LR for depth head; add entropy regularization on depth distribution |
| Loss stuck at log(2) | Model predicts 0.5 everywhere | Increase positive class weight; verify labels are loading correctly |
| NaN in voxel pooling | Numerical overflow in cumsum | Use float32 for pooling; clip very large coordinate values |
| BEV features have dead regions | Some cameras not contributing | Check extrinsic signs; verify all cameras project within BEV range |
| Slow convergence | >20k steps with no meaningful improvement | Increase LR; simplify to single-class (vehicle only) first |

---

## 9) Dataset + Evaluation Protocol

**Dataset:** nuScenes [LSS | Sec 4]

- 700 training scenes, 150 validation scenes
- 6 cameras: front, front-left, front-right, back, back-left, back-right
- 2Hz keyframes (annotated), 12Hz sweeps (unannotated)

**Metrics:** Intersection-over-Union (IoU) for BEV segmentation per class [LSS | Sec 4, Table 1]

**Evaluation settings:**

- BEV range: [-50m, 50m] × [-50m, 50m]
- Resolution: 0.5m
- Classes evaluated: vehicle, drivable area (varies by experiment)

**Data gotchas:**

| Gotcha | Detail |
|--------|--------|
| nuScenes coordinate convention | X-forward, Y-left, Z-up in ego frame; easy to confuse with X-right conventions |
| Missing camera frames | Some keyframes may have dropped camera images; need fallback |
| Annotation sparsity | Only keyframes (2Hz) are annotated; sweeps between keyframes have no labels |
| Vehicle size variation | Trucks vs. cars → large IoU sensitivity to large vehicles |

---

## 10) Results Summary + Ablations

**Main results:** LSS achieves strong BEV vehicle segmentation IoU on nuScenes validation set, outperforming prior baselines (OFT, PON) that use monocular depth or IPM [LSS | Table 1]. *Exact numbers not confirmed in this session — refer to Table 1 in paper.*

**Top 3 ablations:**

1. **Number of depth bins:** More bins (D=41) outperforms fewer (D=10); diminishing returns beyond ~41 [LSS | Sec 4, ablations]
2. **Effect of number of cameras:** Performance increases with more cameras; model gracefully handles missing cameras [LSS | Sec 4]
3. **Robustness to calibration noise:** LSS is robust to small perturbations in extrinsics, demonstrating that learned depth compensates for calibration error [LSS | Sec 4]

**Failure modes:** (1) Far-field objects poorly resolved due to depth discretization; (2) Heavy occlusion in ego-near regions; (3) Depth without supervision can be inaccurate in absolute metric terms even if BEV segmentation is reasonable.

---

## 11) Practical Insights

**10 engineering takeaways:**

1. The outer-product lift is memory-hungry: D×C×H×W per camera. Reduce D or feature resolution for memory savings.
2. Pillar pooling (sum along Z) is the simplest height collapse — sufficient for flat-ground driving scenarios.
3. Efficient voxel pooling via cumulative-sum trick is essential for GPU speed; naive scatter_add is slow.
4. Learned depth without supervision tends to be "soft" — broad distributions rather than sharp peaks. This still works for segmentation but limits 3D detection accuracy.
5. The architecture naturally handles variable camera counts — just concatenate frustums before pooling.
6. BEV encoder quality matters as much as the lift quality — a deeper BEV CNN compensates for noisy depth.
7. Pre-computing frustum coordinates for each camera config (once per dataset) speeds up data loading significantly.
8. Intrinsic adjustment after image resize is the #1 source of bugs in reimplementations.
9. The BEV grid origin convention (center vs. corner) must be consistent between grid creation and label rasterization.
10. LSS-style lift is deterministic given depth — no sampling, no learned queries — making it very debuggable.

**5 gotchas:**

1. Forgetting to adjust intrinsics after image resize → all projections wrong
2. Using ego-to-camera instead of camera-to-ego extrinsics (or vice versa) → mirrored BEV
3. X/Y axis convention mismatch between BEV tensor layout and ego frame → rotated predictions
4. Depth range not covering actual object distances in dataset → objects outside frustum
5. Memory overflow with full-resolution images and D=41 → must downsample images or features

**5 debugging checks:**

1. **Projection sanity:** Project GT 3D boxes onto each camera image → should overlay vehicles
2. **Frustum coverage:** Plot all frustum points in BEV → should form 6 overlapping fans covering 360°
3. **Depth histogram:** Plot predicted depth probabilities over dataset → should roughly match GT depth distribution
4. **BEV activation map:** Sum absolute BEV features → should show higher activation where cameras point
5. **Single-camera test:** Train with 1 camera → BEV should only activate in that camera's FOV

**Tiny-subset overfit plan:**

| Parameter | Value |
|-----------|-------|
| Subset size | 2 scenes (Not specified exact frame count in provided inputs) |
| Heads to keep | Vehicle segmentation only |
| Augments to disable | ALL — fixed resize only |
| Expected behavior | IoU > 0.9 on training set within 500 steps |
| If fails to overfit | Check projection sanity; check label loading; reduce to 1 camera + 1 frame |

---

## 12) Minimal Reimplementation Checklist

**Step-by-step build list:**

1. **Data preprocessing:** Load nuScenes images + calibration; resize images; adjust intrinsics; rasterize BEV labels
2. **Frustum creation:** Create (D, H_feat, W_feat, 3) coordinate grid per camera using K^{-1}
3. **Coordinate transform:** Apply extrinsics to transform frustum from camera to ego frame
4. **Backbone + neck:** EfficientNet-B0 or ResNet-18 → FPN neck → feature maps
5. **Depth + context head:** 1×1 conv → split to D depth logits + C context features → softmax on depth
6. **Outer product:** `depth_probs * context` → frustum volume
7. **Voxel pooling:** Scatter frustum volume to BEV grid cells; sum within each cell
8. **BEV encoder:** 2D ResNet blocks on BEV features
9. **Segmentation head:** 1×1 conv → binary cross-entropy loss
10. **Evaluation:** IoU on BEV segmentation

**Unit tests:**

| Test | Check |
|------|-------|
| `test_intrinsic_resize` | After 2× downsample, focal lengths halved, principal point halved |
| `test_frustum_shape` | Frustum has shape (D, H_feat, W_feat, 3) with Z monotonically increasing |
| `test_extrinsic_transform` | Known 3D point transforms to expected ego coordinates |
| `test_outer_product_shape` | Output is (B, N, D, C, H_feat, W_feat) |
| `test_voxel_pooling_conservation` | Sum of BEV grid ≈ sum of input frustum features (no features lost) |
| `test_bev_output_shape` | Final output matches expected (B, C_cls, Y_bev, X_bev) |

**Minimal sanity scripts:**

**Script 1: Projection sanity visualization**
```python
# Project GT 3D box corners to all 6 camera images
# Expected: colored boxes overlaid on vehicles in images
# Run: python vis_projection.py --scene 0 --sample 0
for cam in cameras:
    pts_2d = K[cam] @ extrinsic[cam] @ pts_3d  # project
    draw_boxes_on_image(img[cam], pts_2d)
    # CHECK: boxes should tightly enclose vehicles
```

**Script 2: Tiny-overfit training**
```python
# Overfit on 2 scenes, vehicle class only, no augmentation
# Expected: training IoU > 0.9 within 500 steps
# If not: check projection, label loading, loss function
train(scenes=[0,1], classes=['vehicle'], augment=False, steps=500)
```

**Script 3: Eval sanity**
```python
# Run trained model on training scene; compute IoU
# Expected: IoU matches training metric (no eval-time bugs)
# Common issue: forgetting to adjust intrinsics at eval time
eval(model, scenes=[0,1], check_shapes=True)
```

---
---

---

_Source: extracted and adapted from `BEV_Papers_Technical_Dossier.md` in this workspace._
