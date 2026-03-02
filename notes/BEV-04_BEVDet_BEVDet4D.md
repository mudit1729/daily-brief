# Paper 4: BEVDet (and BEVDet4D)

**Full title:** BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye-View
**Authors:** Junjie Huang, Guan Huang, Zheng Zhu, Dalong Du (PhiGent Robotics)
**Venue:** arXiv 2021 (workshop NeurIPS 2022)
**arXiv:** 2112.11790

---

## 1) One-page Overview

**Tasks solved:** Multi-camera 3D object detection [BEVDet | Sec 1]

**Sensors assumed:** Multi-camera (6 cameras on nuScenes) [BEVDet | Sec 4]

**Key novelty:**

- **Paradigm unification:** Frames the multi-camera 3D detection pipeline as four modules: image encoder, view transformer (LSS-style), BEV encoder, and task head — establishing a modular design template [BEVDet | Sec 3]
- **BEV-space data augmentation:** Applies random flip, rotation, and scaling in BEV space (not just image space), which is critical for BEV detection performance [BEVDet | Sec 3.4]
- Uses LSS-style lift (depth distribution + outer product + pillar pooling) as the view transformer [BEVDet | Sec 3.2]
- Task-specific head: CenterPoint-style center-based detection head on BEV features [BEVDet | Sec 3.3]
- BEVDet4D extension: concatenates BEV features from adjacent timestamps with ego-motion alignment for temporal fusion [BEVDet4D | Sec 3]

**If you only remember 3 things:**

1. LSS view transformer + CenterPoint head = BEVDet (modular 4-part design)
2. BEV-space augmentation (flip, rotate, scale in BEV) is surprisingly important — often the single biggest contributor
3. BEVDet4D adds temporal by concatenating ego-aligned BEV features from previous frames

---

## 2) Problem Setup and Outputs (Precise)

**Inputs:** Same multi-camera setup as LSS + CenterPoint-style targets.

| Tensor | Shape | Description |
|--------|-------|-------------|
| Images | `(B, N, 3, H, W)` | N=6, typical H=256, W=704 after resize |
| Intrinsics | `(B, N, 3, 3)` | Camera K |
| Extrinsics | `(B, N, 4, 4)` | Camera-to-ego |
| (BEVDet4D) Prev images | `(B, N, 3, H, W)` | Previous frame images |
| (BEVDet4D) Ego motion | `(B, 4, 4)` | Relative ego pose between frames |

**Outputs:**

| Output | Shape | Description |
|--------|-------|-------------|
| Heatmaps | `(B, C_cls, H_bev, W_bev)` | Center heatmap per class (CenterPoint style) |
| Box regression | `(B, 8, H_bev, W_bev)` | Offset (2) + depth (1) + size (3) + rotation (2) maps |
| Velocity | `(B, 2, H_bev, W_bev)` | vx, vy per BEV cell |

**BEV grid:** Typically [-51.2m, 51.2m] × [-51.2m, 51.2m] at 0.8m resolution → 128×128 grid [BEVDet | Sec 4]. *Some configs use [-50m, 50m] at 0.5m → 200×200.*

---

## 3) Coordinate Frames and Geometry

**Frames:** Same as LSS — camera, ego, BEV grid.

**BEV grid definition [BEVDet | Sec 4]:**

| Parameter | Value (typical) |
|-----------|----------------|
| X range | [-51.2m, 51.2m] |
| Y range | [-51.2m, 51.2m] |
| Resolution | 0.8 m/cell |
| Grid size | 128 × 128 |
| Depth bins | 1m to 60m, 1m spacing, D=59 |

**BEV-space augmentation transforms [BEVDet | Sec 3.4]:**

- Random flip (horizontal in BEV) — must also flip GT boxes
- Random rotation ([-22.5°, 22.5°] typical) — must also rotate GT boxes
- Random scaling ([0.95, 1.05] typical) — must also scale GT boxes

These augmentations are applied to the BEV feature tensor AND the BEV-space ground truth simultaneously.

**BEVDet4D temporal alignment [BEVDet4D | Sec 3]:**

- BEV features from time t-1 are warped to time t using ego-motion:
  `BEV_{t-1→t} = warp(BEV_{t-1}, T_{ego_t ← ego_{t-1}})`
- Warping via grid_sample with the relative ego pose
- Concatenate `[BEV_t, BEV_{t-1→t}]` along channel dimension

**Geometry sanity checks:**

| # | What to verify | Expected outcome | Failure signature |
|---|---------------|------------------|-------------------|
| 1 | BEV augmentation + label consistency | After BEV flip, GT boxes also flipped | Boxes on wrong side of ego |
| 2 | BEV rotation center | Rotation around BEV grid center (ego position) | Objects shift spatially after rotation |
| 3 | Temporal warp correctness | Stationary objects align after ego-motion warp | Temporal ghosting (double objects) |
| 4 | Depth range covers objects | Depth bins span 1–60m | Far objects missing from BEV |
| 5 | CenterPoint heatmap GT generation | Gaussian peaks at GT box centers in BEV | Peaks at wrong locations or missing |

---

## 4) Architecture Deep Dive (Module-by-module)

**Block diagram:**

```
Images (B,N,3,H,W)
       │
       ▼
┌──────────────┐
│ Image Encoder │  ResNet-50 + FPN
└──────┬───────┘
       │ (B,N,C,H/16,W/16)
       ▼
┌──────────────────┐
│ View Transformer  │  LSS-style: depth pred → outer product → voxel pool
│ (LSS Lift+Splat) │
└──────┬───────────┘
       │ (B,C,Y_bev,X_bev)
       ▼
┌──────────────────┐
│ BEV-space Augment │  Random flip/rotate/scale on BEV features + GT
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│ BEV Encoder   │  ResNet-like 2D CNN on BEV
└──────┬───────┘
       │ (B,C',Y_bev,X_bev)
       ▼
┌──────────────────┐
│ CenterPoint Head  │  Heatmap + regression + velocity
└──────────────────┘
```

**For BEVDet4D, insert before BEV Encoder:**

```
BEV_t (B,C,Y,X) + BEV_{t-1} warped → concat → (B,2C,Y,X) → conv → (B,C,Y,X)
```

**Modules:**

| Module | Purpose | Input → Output | Key operations |
|--------|---------|---------------|----------------|
| Image encoder | Feature extraction | `(B*N, 3, H, W)` → `(B*N, C, H/16, W/16)` | ResNet-50 + FPN or SecondFPN |
| View transformer | LSS lift+splat | `(B, N, C, H', W')` + calibration → `(B, C_bev, Y, X)` | Depth pred (1×1 conv → softmax), outer product, pillar pooling |
| BEV augmentation | Training-time augment | `(B, C, Y, X)` → `(B, C, Y, X)` | grid_sample with random affine; also apply to GT |
| (BEVDet4D) Temporal concat | Fuse adjacent frames | `(B, C, Y, X)` × 2 → `(B, 2C, Y, X)` → `(B, C, Y, X)` | Ego-warp + channel concat + 1×1 conv |
| BEV encoder | Refine BEV | `(B, C, Y, X)` → `(B, C', Y, X)` | ResNet-18 style blocks on BEV plane |
| CenterPoint head | Detection | `(B, C', Y, X)` → heatmap + reg maps | Separate conv heads for each attribute |

**Where BEV happens:** Same as LSS — the view transformer module performs depth-based lift and pillar pooling [BEVDet | Sec 3.2].

---

## 5) Forward Pass Pseudocode (Shape-annotated)

```python
def forward(images, intrinsics, extrinsics, prev_bev=None, ego_motion=None):
    B, N, _, H, W = images.shape

    # 1. Image encoder
    feats = image_encoder(images.view(B*N, 3, H, W))  # (B*N, C, H', W')
    feats = feats.view(B, N, C, H_f, W_f)

    # 2. View transformer (LSS-style)
    depth_logits = depth_net(feats)                    # (B, N, D, H_f, W_f)
    depth_probs = softmax(depth_logits, dim=2)         # (B, N, D, H_f, W_f)
    context = context_net(feats)                       # (B, N, C_ctx, H_f, W_f)
    volume = depth_probs.unsqueeze(3) * context.unsqueeze(2)  # (B, N, D, C_ctx, H_f, W_f)
    bev = voxel_pooling(volume, frustum_coords, grid_config)  # (B, C_bev, Y, X)

    # 3. (BEVDet4D) Temporal fusion
    if prev_bev is not None:
        prev_bev_aligned = warp_bev(prev_bev, ego_motion)  # (B, C_bev, Y, X)
        bev = conv1x1(cat([bev, prev_bev_aligned], dim=1))  # (B, C_bev, Y, X)

    # 4. BEV augmentation (training only)
    if training:
        bev, gt = bev_augment(bev, gt_boxes)  # random flip/rotate/scale

    # 5. BEV encoder
    bev = bev_encoder(bev)  # (B, C', Y, X)

    # 6. CenterPoint head
    heatmap = heatmap_head(bev)       # (B, C_cls, Y, X)
    reg = regression_head(bev)         # (B, 8, Y, X)
    vel = velocity_head(bev)           # (B, 2, Y, X)

    return heatmap, reg, vel
```

---

## 6) Heads, Targets, and Losses

**Heads (CenterPoint style) [BEVDet | Sec 3.3]:**

| Head | Output | Target |
|------|--------|--------|
| Heatmap | `(B, C_cls, Y, X)` | Gaussian peaks at GT box centers, σ based on box size |
| Offset | `(B, 2, Y, X)` | Sub-pixel offset from cell center to exact box center |
| Height | `(B, 1, Y, X)` | Box center z-coordinate |
| Size | `(B, 3, Y, X)` | (w, l, h) in meters |
| Rotation | `(B, 2, Y, X)` | (sin(yaw), cos(yaw)) |
| Velocity | `(B, 2, Y, X)` | (vx, vy) in m/s |

**Loss terms:**

| Loss | Formula | Weight |
|------|---------|--------|
| Gaussian focal loss (heatmap) | Modified focal loss for Gaussian heatmaps | 1.0 |
| L1 loss (offset, height, size, rotation, velocity) | Per-positive-cell L1 regression | Varies: typically 0.25 for offset/height, 0.25 for size, 1.0 for rotation |

*Plausible defaults (not from paper): loss weights follow CenterPoint conventions.*

**Assignment strategy:** Center-based — no Hungarian matching. Each GT box is assigned to the BEV cell containing its center. Only cells with GT centers contribute to regression losses. The heatmap uses Gaussian focal loss at all cells [BEVDet | Sec 3.3].

**Loss debugging checklist:**

| # | Bug | Symptom | Quick test |
|---|-----|---------|------------|
| 1 | Heatmap GT Gaussian size wrong | Too many false positives (σ too large) or misses (σ too small) | Visualize GT heatmap; Gaussians should cover object centers |
| 2 | BEV augment applied to features but not GT | Loss doesn't converge; boxes misaligned | Disable BEV augment; loss should drop immediately |
| 3 | Regression target encoding mismatch | Predicted boxes have wrong sizes/orientations | Check encoding: are targets (w,l,h) or (l,w,h)? Is yaw sin/cos or angle? |
| 4 | Velocity target frame mismatch | Velocity predictions systematically wrong | Verify velocity is in current ego frame after temporal alignment |
| 5 | Height (z) target convention | Boxes floating above or below ground | Check z is center height above ground vs. absolute z |
| 6 | Missing positive cells | Regression loss is zero | Verify GT box centers fall within BEV grid range |

---

## 7) Data Pipeline and Augmentations

**Image-space augments:** Random resize, crop, flip, color jitter [BEVDet | Sec 4]

**BEV-space augments [BEVDet | Sec 3.4]:**

- **Random BEV flip:** Flip BEV features + GT boxes along X or Y axis
- **Random BEV rotation:** Rotate BEV features + GT by angle ∈ [-22.5°, 22.5°]
- **Random BEV scale:** Scale BEV features + GT by factor ∈ [0.95, 1.05]

This is applied AFTER the view transformer and BEFORE the BEV encoder.

**BEVDet4D temporal:** Previous frame BEV is cached (detached); no gradient flows through temporal fusion [BEVDet4D | Sec 3].

**Augmentation safety:**

| Augment | Risk | Validation |
|---------|------|------------|
| BEV flip | GT must flip identically; rotation angles must negate | Overlay augmented GT on augmented BEV features |
| BEV rotation | bilinear interpolation artifacts at grid edges | Check that objects near edges aren't clipped |
| Image resize | Intrinsics must adjust | Standard LSS check |

---

## 8) Training Pipeline (Reproducible) (Reproducible) (Reproducible)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW [BEVDet | Sec 4] |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Epochs | 20 on nuScenes [BEVDet | Sec 4] |
| Batch size | 8 (1 per GPU × 8 GPUs) |
| LR schedule | Cyclic or step [BEVDet | Sec 4] |
| Backbone pretrain | ImageNet pretrained ResNet-50 |
| GPUs | 8× A100 *Plausible defaults (not from paper)* |

**Stability & convergence:**

| Failure | Symptom | Fix |
|---------|---------|-----|
| BEV augmentation too strong | BEV features distorted; loss unstable | Reduce augment ranges; disable during initial epochs |
| (BEVDet4D) Temporal misalignment | Ghost objects in BEV | Verify ego-motion warp direction and magnitude |
| Depth not learning | BEV features noisy | Consider adding depth supervision (→ BEVDepth) |
| Heatmap loss dominates | Regression heads underfit | Increase regression loss weights |

---

## 9) Dataset + Evaluation Protocol

Not specified. Plausible defaults (not from paper): use DETR3D-style nuScenes evaluation protocol (mAP + NDS + TP metrics).

---

## 10) Results Summary + Ablations

**Main results [BEVDet | Table 1]:**

- Not specified in provided inputs. Plausible defaults (not from paper): consult the cited paper table/benchmark entry for exact values.
- BEVDet4D adds a notable NDS improvement via temporal fusion [BEVDet4D | Table 1]; exact delta Not specified in provided inputs.

**Top 3 ablations [BEVDet | Table 2–4]:**

1. **BEV augmentation:** BEV-space augmentation provides a major mAP gain in ablations [BEVDet | Table 4]; exact delta Not specified in provided inputs.
2. **BEV encoder depth:** Deeper BEV encoder improves metrics in ablations [BEVDet | Table 6]; exact delta Not specified in provided inputs.
3. **Image resolution:** Higher resolution (900×1600 vs. 256×704) improves but at large compute cost

---

## 11) Practical Insights

**10 engineering takeaways:**

1. BEV-space augmentation is the #1 trick — implement it early and correctly.
2. The modular design (image encoder / view transformer / BEV encoder / head) makes ablation and debugging clean.
3. CenterPoint head is simpler than Hungarian matching — easier to debug, faster to train.
4. BEVDet4D's temporal fusion is simple (concat + conv) but effective — a strong baseline for temporal BEV.
5. Detaching previous-frame BEV (no gradient through time) is practical and avoids GPU memory explosion.
6. BEV grid resolution directly trades off accuracy vs. memory: 128×128 is lightweight; 200×200 is more accurate.
7. The depth network quality limits the entire pipeline — improvements here (→ BEVDepth) yield large gains.
8. Multi-task heads (add segmentation or mapping) can share the BEV encoder with minimal overhead.
9. BEV augmentation must be applied consistently to features AND all label types (boxes, heatmaps, velocities).
10. For BEVDet4D, caching BEV features for sequential scenes avoids recomputing previous-frame features.

**5 gotchas:**

1. BEV augmentation applied to features but not GT → model learns nothing
2. BEV rotation must rotate velocity vectors too → `(vx', vy') = R · (vx, vy)`
3. CenterPoint Gaussian size must match BEV resolution — wrong σ causes poor heatmap learning
4. BEVDet4D ego-motion warp: using inverse instead of forward transform → temporal ghosting
5. Depth bins must match between training and inference — changing resolution changes bin assignment

**Tiny-subset overfit plan:**

| Parameter | Value |
|-----------|-------|
| Subset | 2 scenes |
| Augments | Disable ALL (especially BEV augments) |
| Expected | Heatmap loss < 0.1, box regression converges within 1000 steps |
| If fails | Check GT heatmap generation; verify CenterPoint assignment |

---

## 12) Minimal Reimplementation Checklist

**Build order:**

1. Data loading (nuScenes)
2. Image encoder (ResNet-50 + FPN)
3. View transformer (LSS lift+splat — can reuse LSS code)
4. BEV encoder (ResNet-18 blocks on BEV)
5. CenterPoint head (heatmap + regression heads)
6. Gaussian heatmap GT generation
7. BEV-space augmentation
8. (BEVDet4D) Temporal BEV warp + concatenation
9. Training + evaluation

**Unit tests:**

| Test | Check |
|------|-------|
| `test_bev_flip` | Flipped BEV + flipped GT produce same loss as unflipped |
| `test_bev_rotate` | Small rotation should preserve object locations with only minor perturbation |
| `test_heatmap_gt` | Gaussian peak at GT box center, peak value = 1.0 |
| `test_ego_warp` | Stationary object at same BEV location after warp |
| `test_centerpoint_decode` | Known heatmap peak → correct box center after offset |

**Minimal sanity scripts:**

**Script 1:** Visualize BEV features (sum channels) + GT heatmap overlay → should align
**Script 2:** Overfit 2 scenes without augmentation → heatmap loss < 0.1 in 1000 steps
**Script 3:** Decode predicted boxes and plot in BEV alongside GT → should overlap

---
---

---

_Source: extracted and adapted from `BEV_Papers_Technical_Dossier.md` in this workspace._
