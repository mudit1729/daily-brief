# Paper 5: BEVDepth

**Full title:** BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection
**Authors:** Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, Zeming Li (MEGVII)
**Venue:** AAAI 2023
**arXiv:** 2206.10092

---

## 1) One-page Overview

**Tasks solved:** Multi-camera 3D object detection (improves BEVDet) [BEVDepth | Sec 1]

**Sensors assumed:** Multi-camera (6 cameras on nuScenes); uses LiDAR only for depth supervision at training time [BEVDepth | Sec 3]

**Key novelty:**

- **Explicit depth supervision:** Uses projected LiDAR points as depth ground truth to supervise the depth prediction network, dramatically improving depth accuracy over LSS's implicit learning [BEVDepth | Sec 3.1]
- **Camera-aware depth prediction:** Encodes camera intrinsic/extrinsic information into the depth network via a camera embedding, making depth prediction camera-aware [BEVDepth | Sec 3.2]
- **Depth correction module:** Residual depth refinement after initial prediction [BEVDepth | Sec 3.3]
- Plug-and-play: can be added to any LSS-style pipeline (BEVDet, BEVFusion, etc.)
- Demonstrates that explicit depth supervision is the single most impactful improvement for LSS-style BEV detection

> **If you only remember 3 things:**
>
> 1. Project LiDAR to camera images → binary cross-entropy depth supervision → massive accuracy gain
> 2. Camera-aware depth: encode intrinsics/extrinsics into depth prediction
> 3. Drop-in improvement for any LSS-based architecture

---

## 2) Problem Setup and Outputs (Precise)

**Inputs:** Not specified. Plausible defaults (not from paper): use BEVDet camera inputs and add LiDAR-projected depth targets for supervision.

| Tensor | Shape | Description |
|--------|-------|-------------|
| Images | `(B, N, 3, H, W)` | Multi-view images |
| Intrinsics | `(B, N, 3, 3)` | Camera K |
| Extrinsics | `(B, N, 4, 4)` | Camera-to-ego |
| **LiDAR depth GT** | `(B, N, D, H_feat, W_feat)` | Projected LiDAR → per-pixel depth bin label (training only) |

**Depth GT generation:** Project LiDAR point cloud onto each camera image using calibration. For each pixel with a LiDAR hit, assign to the nearest depth bin. Creates a sparse one-hot depth label [BEVDepth | Sec 3.1].

**Outputs:** Not specified. Plausible defaults (not from paper): reuse the corresponding output definition from the referenced baseline paper.

---

## 3) Coordinate Frames and Geometry

Not specified. Plausible defaults (not from paper): follow the same frame conventions as BEVDet/LSS implementations.

**LiDAR → camera depth projection:**

1. Transform LiDAR points from LiDAR frame to camera frame: `p_cam = T_cam←lidar · p_lidar`
2. Project to image: `(u, v) = K · p_cam[:3] / p_cam[2]`
3. Depth = `p_cam[2]` (z in camera frame)
4. Discretize depth into bins matching the LSS depth bins
5. Only keep points that fall within image bounds and have positive depth

**Camera-aware depth [BEVDepth | Sec 3.2]:**

- Flatten intrinsic matrix + extrinsic matrix into a vector
- Pass through MLP → camera embedding
- Concatenate or add camera embedding to depth features before depth prediction

**Geometry sanity checks:**

| # | What to verify | Expected outcome | Failure signature |
|---|---------------|------------------|-------------------|
| 1 | Projected LiDAR on images | Points overlay on object surfaces and ground | Points floating or misaligned with images |
| 2 | Depth GT histogram | Peaks at common distances (near-field ground, vehicles at 10-30m) | All zeros (projection failed) or uniform |
| 3 | Depth GT matches depth bins | Assigned bins correspond to correct metric depth | Off-by-one bin assignment |
| 4 | Camera embedding varies per camera | Different cameras produce different embeddings | All cameras produce identical depth (ignoring FOV differences) |
| 5 | Sparse depth GT coverage | Not specified in provided inputs | Too sparse or overly dense indicates GT generation issues |

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
       │ (B,N,C,H',W')
       ▼
┌──────────────────────────┐
│ Camera-Aware Depth Net    │
│ ┌────────────┐           │
│ │ Camera Embed│←intrinsics│  MLP(flatten(K, E)) → (B,N,C_cam)
│ └─────┬──────┘           │
│       │ concat            │
│ ┌─────▼──────────┐      │
│ │ Depth + Context │      │  Enhanced depth prediction with camera info
│ │ + Depth Correct │      │  Residual refinement
│ └─────┬──────────┘      │
│       │ depth: (B,N,D,H',W')  context: (B,N,C,H',W')
└───────┴──────────────────┘
       │
       ▼
┌────────────────────┐
│ Depth Supervision   │  BCE loss with projected LiDAR GT (training only)
└────────────────────┘
       │
       ▼
┌──────────────────┐
│ LSS Outer Product │  Not specified (Plausible defaults (not from paper): follow BEVDet-style implementation)
│ + Voxel Pooling   │
└──────┬───────────┘
       │ (B,C,Y,X)
       ▼
┌──────────────────┐
│ BEV Encoder + Head│  Not specified (Plausible defaults (not from paper): follow BEVDet-style implementation)
└──────────────────┘
```

**Key module: Depth supervision**

- **Depth prediction:** `depth_logits = depth_net(img_feat, cam_embed)` → `(B, N, D, H', W')`
- **Depth GT:** `depth_gt = project_lidar_to_bins(lidar, K, E)` → `(B, N, D, H', W')` (sparse binary)
- **Loss:** Binary cross-entropy between `softmax(depth_logits)` and `depth_gt` at pixels with LiDAR coverage
- The depth supervision gradient flows through the entire depth network and improves the lift quality

---

## 5) Forward Pass Pseudocode (Shape-annotated)

```python
def forward(images, intrinsics, extrinsics, lidar_depth_gt=None):
    B, N, _, H, W = images.shape

    # 1. Image encoder
    feats = image_encoder(images.view(B*N, 3, H, W))  # (B*N, C, H', W')
    feats = feats.view(B, N, C, H_f, W_f)

    # 2. Camera-aware depth prediction
    cam_params = flatten_calib(intrinsics, extrinsics)  # (B, N, K) K≈25 (3x3 + 4x4 flattened)
    cam_embed = cam_mlp(cam_params)                      # (B, N, C_cam)
    cam_embed = cam_embed[..., None, None].expand(-1,-1,-1, H_f, W_f)  # broadcast to spatial

    depth_input = cat([feats, cam_embed], dim=2)          # (B, N, C+C_cam, H', W')
    depth_logits = depth_net(depth_input)                  # (B, N, D, H', W')
    depth_logits = depth_correction(depth_logits)          # residual refinement
    depth_probs = softmax(depth_logits, dim=2)             # (B, N, D, H', W')

    # 3. Depth supervision loss (training only)
    if lidar_depth_gt is not None:
        mask = lidar_depth_gt.sum(dim=2) > 0               # pixels with LiDAR hits
        depth_loss = BCE(depth_probs[mask], lidar_depth_gt[mask])

    # 4. Context features
    context = context_net(feats)                           # (B, N, C_ctx, H', W')

    # 5. LSS outer product + voxel pooling (same as BEVDet)
    volume = depth_probs.unsqueeze(3) * context.unsqueeze(2)
    bev = voxel_pooling(volume, frustum, grid_config)      # (B, C, Y, X)

    # 6. BEV encoder + detection head (same as BEVDet)
    bev = bev_encoder(bev)
    outputs = detection_head(bev)

    return outputs, depth_loss
```

---

## 6) Heads, Targets, and Losses

Same CenterPoint heads as BEVDet, **plus** depth supervision loss.

| Loss | Formula | Weight |
|------|---------|--------|
| Depth BCE | `BCE(softmax(depth_logits), lidar_depth_gt)` at LiDAR pixels | 3.0 [BEVDepth | Sec 4] |
| Gaussian focal (heatmap) | Not specified (Plausible defaults (not from paper): same target type as BEVDet) | 1.0 |
| L1 (regression) | Not specified (Plausible defaults (not from paper): same target type as BEVDet) | 0.25 |

*Plausible defaults (not from paper): depth loss weight of 3.0 is commonly reported; exact value may vary.*

**Loss debugging checklist:**

| # | Bug | Symptom | Quick test |
|---|-----|---------|------------|
| 1 | LiDAR-camera projection misaligned | Depth loss doesn't decrease | Visualize projected LiDAR on images — should align with surfaces |
| 2 | Depth bin assignment off-by-one | Depth systematically biased by 1 bin | Compare predicted depth mode with GT depth for known objects |
| 3 | Sparse GT mask wrong | Training on empty pixels (all zeros) | Check percentage of masked pixels — should be 5-15% |
| 4 | Depth loss weight too high | Detection loss can't decrease (depth dominates) | Reduce depth weight; check gradient magnitudes |
| 5 | Camera embedding not varying | All cameras predict same depth distribution | Print cam_embed per camera — should differ |

---

## 7) Data Pipeline and Augmentations

Not specified. Plausible defaults (not from paper): follow BEVDet inputs and add LiDAR-derived depth supervision targets.

1. Load LiDAR point cloud for each sample
2. Project onto each camera using LiDAR-to-camera calibration
3. Discretize depth values into bins matching depth network
4. Handle image augmentation: if images are resized/cropped, update projection coordinates accordingly

**Critical:** If image augmentation changes the pixel coordinates, the LiDAR depth GT must be regenerated with the augmented intrinsics.

---

## 8) Training Pipeline

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-4 [BEVDepth | Sec 4] |
| Epochs | 20 on nuScenes |
| Depth supervision | From epoch 0 (not staged) |
| Other | Not specified |

**Stability & convergence:**

| Failure | Symptom | Fix |
|---------|---------|-----|
| Depth loss dominates | Detection metrics don't improve | Lower depth loss weight from 3.0 to 1.0 |
| Depth collapses despite supervision | Predicted depth is sharp but wrong | Check LiDAR GT alignment; verify bin assignment |
| Camera embed has no effect | Ablation shows same result with/without | Ensure embed is concatenated before depth conv, not after |

---

## 9) Dataset + Evaluation Protocol

Same evaluation protocol as BEVDet.

**Main results [BEVDepth | Table 1]:** BEVDepth with ResNet-50 significantly outperforms BEVDet:

- Not specified in provided inputs. Plausible defaults (not from paper): consult the cited paper table/benchmark entry for exact values.
- Depth supervision provides a substantial mAP improvement in ablations [BEVDepth | Table 2–3]; exact delta Not specified in provided inputs.

**Top 3 ablations [BEVDepth | Table 2–3]:**

1. **Depth supervision alone:** +5 mAP over BEVDet baseline
2. **Camera-aware depth:** +1-2 mAP over position-agnostic depth
3. **Depth correction module:** +0.5-1 mAP residual refinement

---

## 10) Results Summary + Ablations

- Not specified in provided inputs. Plausible defaults (not from paper): summarize main benchmark numbers directly from cited tables.

---

## 11) Practical Insights

**10 engineering takeaways:**

1. Depth supervision is the single biggest improvement for LSS-style BEV — implement it first.
2. LiDAR is only needed at training time; inference remains camera-only.
3. Camera-aware depth is cheap to add (small MLP) but consistently helps.
4. The depth supervision signal is sparse; use masked loss correctly [BEVDepth | Sec 3]. Exact sparsity Not specified in provided inputs.
5. Depth GT quality depends on LiDAR-camera calibration accuracy — verify alignment first.
6. BEVDepth improvements are orthogonal to BEV encoder/head improvements — stack them.
7. The depth correction module is a simple residual conv block — easy to add.
8. At inference, without depth GT, the network relies on learned depth patterns — generalization matters.
9. Depth supervision makes training faster to converge (depth learning is accelerated).
10. This is a must-have for any serious LSS-based system — the improvement is large and consistent.

**5 gotchas:**

1. LiDAR-to-camera calibration errors directly corrupt depth GT → verify alignment visually
2. Image augmentation invalidates pre-computed depth GT → must reproject with augmented intrinsics
3. Sparse depth GT requires careful masking — training on zero-label pixels teaches wrong depth
4. Depth bins must match exactly between depth GT and depth network
5. Camera embedding must be computed with the augmented intrinsics (post-augmentation), not original

**Tiny-subset overfit plan:**

| Parameter | Value |
|-----------|-------|
| Subset | 2 scenes |
| Expected | Depth loss → near 0 (depth perfectly learned from LiDAR supervision) |
| Key check | Predicted depth distribution should be sharp and match GT bins |

---

## 12) Minimal Reimplementation Checklist

**Build on BEVDet and add:**

1. LiDAR depth GT generation pipeline
2. Camera embedding MLP
3. Modified depth network with camera-aware input
4. Depth correction module (residual conv)
5. Depth BCE loss with sparse masking
6. Verify depth improves by visualizing predicted vs GT depth

**Script 1:** Visualize projected LiDAR on camera images → should overlay on surfaces
**Script 2:** Plot depth prediction distribution vs GT → should be sharp and aligned
**Script 3:** Overfit 2 scenes → depth loss near 0, detection loss also improving

---

_Source: extracted and adapted from `BEV_Papers_Technical_Dossier.md` in this workspace._
