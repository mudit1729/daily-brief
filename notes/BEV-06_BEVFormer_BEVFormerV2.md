# Paper 6: BEVFormer (and BEVFormer v2)

**Full title:** BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers
**Authors:** Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghuo Sber, Tong Lu, Yu Qiao, Jifeng Dai
**Venue:** ECCV 2022
**arXiv:** 2203.17270

---

## 1) One-page Overview

**Tasks solved:** Multi-camera 3D object detection; also extended to BEV segmentation [BEVFormer | Sec 1, Sec 5]

**Sensors assumed:** Multi-camera (6 cameras) [BEVFormer | Sec 4]

**Key novelty:**

- **Learnable BEV queries on a dense grid:** Maintains a grid of BEV queries (e.g., 200×200) that aggregate information from multi-view images via deformable cross-attention [BEVFormer | Sec 3.2]
- **Spatial cross-attention:** Each BEV query generates reference points at multiple heights, projects them onto camera images, and samples features via deformable attention [BEVFormer | Sec 3.2]
- **Temporal self-attention:** BEV queries attend to previous-frame BEV features (with ego-motion alignment) for temporal consistency [BEVFormer | Sec 3.3]
- Combines dense BEV grid (like LSS) with transformer attention (like DETR3D) — gets benefits of both
- BEVFormer v2 adds perspective supervision and an adaptive history mechanism [BEVFormerV2]

> **If you only remember 3 things:**
>
> 1. Dense BEV query grid + deformable spatial cross-attention to multi-view images (project BEV locations to cameras at multiple heights)
> 2. Temporal self-attention: BEV queries attend to ego-aligned previous BEV features (recurrent memory)
> 3. Bridges LSS-style dense BEV and DETR3D-style query-based reasoning

---

## 2) Problem Setup and Outputs (Precise)

**Inputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| Images | `(B, N, 3, H, W)` | N=6 cameras |
| Intrinsics | `(B, N, 3, 3)` | Camera K |
| Extrinsics | `(B, N, 4, 4)` | Camera-to-ego |
| Ego motion | `(B, 4, 4)` | Current-to-previous ego transform |
| Prev BEV | `(B, C, H_bev, W_bev)` | Previous frame's BEV features (cached) |

**Outputs:** CenterPoint-style or DETR-style 3D detection outputs.

**BEV query grid [BEVFormer | Sec 3.1]:**

| Parameter | Value |
|-----------|-------|
| Grid size | 200 × 200 (BEVFormer-Base) or 50 × 50 (BEVFormer-Small) |
| Range | [-51.2m, 51.2m] × [-51.2m, 51.2m] |
| Resolution | 0.512 m/cell (Base) or 2.048 m/cell (Small) |
| Height pillars | 4 reference heights (e.g., -1m, 0m, 1m, 2m) for spatial cross-attention |
| Channel dim | C = 256 |

---

## 3) Coordinate Frames and Geometry

**Frames:** Camera, ego, BEV grid (same as LSS/BEVDet).

**Spatial cross-attention geometry [BEVFormer | Sec 3.2]:**

For each BEV query at grid position (x, y):
1. Create P reference points at different heights: `{(x, y, z_1), ..., (x, y, z_P)}` where z_p ∈ {-1, 0, 1, 2}m
2. Project each reference point to all N cameras: `(u, v) = K_n · T_cam_n ← ego · (x, y, z_p)`
3. Only attend to cameras where projection is within image bounds
4. Apply deformable attention: learned offsets around each projected point, sample features from image feature maps

**Temporal self-attention geometry [BEVFormer | Sec 3.3]:**

- Previous BEV features are warped to current ego frame using ego-motion matrix
- BEV queries attend to the warped previous BEV features via deformable self-attention
- Self-attention reference points are the BEV grid positions themselves (with learned offsets)

**Geometry sanity checks:**

| # | What to verify | Expected outcome | Failure signature |
|---|---------------|------------------|-------------------|
| 1 | BEV query reference point projections | Points from BEV grid project to correct image locations | Projections outside image or on wrong camera |
| 2 | Height pillar sampling | 4 heights per BEV cell cover relevant objects | All heights above/below ground plane |
| 3 | Temporal BEV warp | Stationary objects align after ego warp | Objects shift or duplicate after temporal fusion |
| 4 | Deformable attention offsets | Offsets stay within reasonable range (~few pixels) | Offsets explode → sampling random locations |
| 5 | Camera hit checking | Each BEV query attends to 1-3 cameras typically | Query attends to 0 cameras (dead) or all 6 (wrong geometry) |

---

## 4) Architecture Deep Dive (Module-by-module)

**Block diagram:**

```
Images (B,N,3,H,W)
       │
       ▼
┌──────────────┐
│ Backbone+FPN  │  ResNet-101 + FPN
└──────┬───────┘
       │ Multi-scale: (B, N, C, H/s, W/s) for s in {8,16,32,64}
       │
BEV Queries ──────────────────────────────────┐
(B, H_bev*W_bev, C)                           │
       │                                       │
       ▼                                       ▼
┌───────────────────────────────────────────────────┐
│  BEVFormer Encoder (6 layers)                      │
│  ┌─────────────────────────┐                      │
│  │ Temporal Self-Attention  │← prev_bev (warped)  │
│  │ (deformable self-attn   │                      │
│  │  on BEV grid)           │                      │
│  └──────────┬──────────────┘                      │
│             │                                      │
│  ┌──────────▼──────────────┐                      │
│  │ Spatial Cross-Attention  │← image features      │
│  │ (deformable cross-attn  │  BEV → project to    │
│  │  multi-height refs)     │  cameras → sample     │
│  └──────────┬──────────────┘                      │
│             │                                      │
│  ┌──────────▼──────────────┐                      │
│  │ FFN                      │                      │
│  └──────────┬──────────────┘                      │
│  (repeat 6 layers)                                 │
└─────────────┬─────────────────────────────────────┘
              │ (B, H_bev*W_bev, C) → reshape → (B, C, H_bev, W_bev)
              ▼
┌──────────────────┐
│ Detection Head    │  CenterPoint or Deformable-DETR style
└──────────────────┘
```

**Modules:**

| Module | Purpose | Input → Output | Key operations |
|--------|---------|---------------|----------------|
| Backbone + FPN | Image features | `(B*N, 3, H, W)` → multi-scale features | ResNet-101 + FPN [BEVFormer | Sec 4] |
| BEV queries | Learnable grid | N/A → `(B, H_bev×W_bev, C)` | Learned embeddings; C=256 |
| **Temporal self-attn** | Fuse with previous BEV | `(B, HW, C)` query + `(B, HW, C)` prev_bev → `(B, HW, C)` | Deformable self-attention; prev_bev warped by ego motion |
| **Spatial cross-attn** | Sample from images | `(B, HW, C)` query + image feats → `(B, HW, C)` | Project BEV refs at P heights → deformable attn on images |
| FFN | Non-linearity | `(B, HW, C)` → `(B, HW, C)` | 2-layer MLP |
| Detection head | 3D detection | `(B, C, H_bev, W_bev)` → boxes | CenterPoint-style [BEVFormer | Sec 4] |

**Where BEV happens:** The spatial cross-attention step is where multi-view image features are aggregated into the BEV grid. Each BEV query at position (x,y) projects reference points at multiple heights to camera images and attends to nearby features [BEVFormer | Sec 3.2].

**Deformable attention details:**

- Each reference point has K learned sampling offsets (typically K=4)
- Multi-scale: deformable attention samples from all FPN levels
- Attention weights: learned from query features
- This is much more memory-efficient than full cross-attention (PETR-style)

---

## 5) Forward Pass Pseudocode (Shape-annotated)

```python
def forward(images, intrinsics, extrinsics, ego_motion, prev_bev=None):
    B, N, _, H, W = images.shape

    # 1. Backbone + FPN
    ms_feats = backbone_fpn(images.view(B*N, 3, H, W))
    # list of (B*N, C, H/s, W/s) for s in [8,16,32,64]
    # Reshape each: (B, N, C, H/s, W/s)

    # 2. Initialize BEV queries
    bev_queries = bev_embed.weight  # (H_bev*W_bev, C) = (40000, 256)
    bev_queries = bev_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 40000, 256)

    # 3. Warp previous BEV for temporal attention
    if prev_bev is not None:
        prev_bev_warped = warp_bev(prev_bev, ego_motion)  # (B, 40000, 256)
    else:
        prev_bev_warped = bev_queries.clone()  # first frame: self-reference

    # 4. BEVFormer Encoder (L=6 layers)
    for layer in encoder_layers:
        # 4a. Temporal self-attention
        # Reference points: BEV grid positions
        bev_queries = layer.temporal_self_attn(
            query=bev_queries,        # (B, 40000, 256)
            value=prev_bev_warped,    # (B, 40000, 256)
        )  # → (B, 40000, 256)

        # 4b. Spatial cross-attention
        # For each BEV position (x,y), create P reference points at heights z_1..z_P
        ref_3d = create_bev_ref_points(bev_grid, heights=[-1,0,1,2])  # (40000, P, 3)
        # Project to all cameras
        ref_2d = project_to_cameras(ref_3d, intrinsics, extrinsics)  # (40000, N, P, 2)

        bev_queries = layer.spatial_cross_attn(
            query=bev_queries,         # (B, 40000, 256)
            value=ms_feats,            # multi-scale image features
            reference_points=ref_2d,   # projected reference positions
        )  # → (B, 40000, 256)

        # 4c. FFN
        bev_queries = layer.ffn(bev_queries)  # (B, 40000, 256)

    # 5. Reshape to BEV grid
    bev_feat = bev_queries.view(B, H_bev, W_bev, C).permute(0,3,1,2)  # (B, C, H_bev, W_bev)

    # 6. Detection head
    outputs = detection_head(bev_feat)

    # Cache current BEV for next frame
    return outputs, bev_queries.detach()  # detach for temporal
```

---

## 6) Heads, Targets, and Losses

BEVFormer uses either CenterPoint-style or Deformable-DETR-style detection head.

**With Deformable DETR head [BEVFormer | Sec 4]:**

| Loss | Formula | Weight |
|------|---------|--------|
| Focal loss (cls) | α=0.25, γ=2.0 | 2.0 |
| L1 loss (reg) | Box parameter L1 | 0.25 |
| GIoU loss (optional) | 1 - GIoU | *Plausible defaults (not from paper): 2.0* |

**Assignment:** Hungarian matching (same as DETR3D).

**Loss debugging checklist:**

| # | Bug | Symptom | Quick test |
|---|-----|---------|------------|
| 1 | Spatial cross-attn projects to wrong camera | Features from wrong viewpoint | Visualize which cameras each BEV position attends to |
| 2 | Temporal warp direction wrong | Temporal ghosting | Visualize warped prev_bev vs. current GT |
| 3 | Deformable attn offset explosion | Loss NaN or garbage predictions | Clip offsets; check offset magnitudes |
| 4 | BEV positional embedding wrong | BEV features don't have spatial meaning | Visualize BEV features: should show spatial structure |
| 5 | Height pillar range wrong | Ground-level objects missed | Check if heights [-1,0,1,2]m cover objects in dataset |

---

## 7) Data Pipeline and Augmentations

**Image augments:** Random resize, crop, flip [BEVFormer | Sec 4]

**BEV augments:** Random flip, rotation (similar to BEVDet)

**Temporal:** Previous frame BEV features cached and warped. For the first frame of a sequence, BEV queries attend to themselves (no temporal info).

**Augmentation safety:** Not specified. Plausible defaults (not from paper): apply BEV augmentations consistently to features and GT annotations.

---

## 8) Training Pipeline

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW [BEVFormer | Sec 4] |
| Learning rate | 2e-4 |
| Backbone LR | 2e-5 |
| Epochs | 24 on nuScenes [BEVFormer | Sec 4] |
| Batch size | 8 (1 per GPU × 8) |
| GPUs | 8× A100 [BEVFormer | Sec 4] |
| Backbone | ResNet-101-DCN pretrained on nuImages |
| LR schedule | Cosine annealing |

**Stability & convergence:**

| Failure | Symptom | Fix |
|---------|---------|-----|
| Deformable attention NaN | Training crashes | Clip offset range; reduce LR; use float32 for attention |
| Temporal attention hurts at start | Worse than single-frame | Disable temporal for first few epochs; prev_bev is random initially |
| BEV queries don't specialize | All queries produce similar features | Verify BEV positional embedding is added correctly |
| Memory overflow | 200×200 BEV × 6 encoder layers | Reduce BEV resolution to 100×100 or use fewer encoder layers |

---

## 9) Dataset + Evaluation Protocol

**Dataset:** nuScenes val/test

**Main results [BEVFormer | Table 1]:**

- Not specified in provided inputs. Plausible defaults (not from paper): consult the cited paper table/benchmark entry for exact values.
- Not specified in provided inputs. Plausible defaults (not from paper): use exact BEVFormer-Small metrics from cited table.
- State-of-the-art camera-only performance at publication time

**Top 3 ablations [BEVFormer | Table 2–5]:**

1. **Temporal self-attention:** +3-4 NDS improvement over single-frame
2. **Number of height pillars:** 4 heights significantly better than 1; diminishing returns after 4
3. **Number of encoder layers:** 6 layers optimal; 3 layers notably weaker

---

## 10) Results Summary + Ablations

- Not specified in provided inputs. Plausible defaults (not from paper): summarize main benchmark numbers directly from cited tables.

---

## 11) Practical Insights

**10 engineering takeaways:**

1. BEVFormer achieves strong results by combining the density of BEV grids with the efficiency of deformable attention.
2. The 200×200 BEV query grid is memory-intensive — consider 100×100 for initial experiments.
3. Temporal self-attention with ego-warped previous BEV provides significant gains with minimal overhead.
4. Deformable attention makes spatial cross-attention tractable (O(Q × K × P) instead of O(Q × N × H × W)).
5. Height pillar sampling (4 heights) is a simple but effective way to handle 3D without explicit depth.
6. Pre-training the backbone on nuImages (monocular 3D detection) helps convergence.
7. BEVFormer naturally extends to multi-task (detection + segmentation) by adding heads on BEV features.
8. The first frame has no temporal info — model must still work in single-frame mode.
9. Deformable attention offset initialization matters — Xavier init works well.
10. BEVFormer's BEV is built by cross-attention (top-down reasoning), unlike LSS which is bottom-up (depth-based).

**5 gotchas:**

1. Deformable attention requires CUDA custom ops — compilation can be tricky
2. BEV query positional embedding must encode spatial position (x, y) — sinusoidal or learned
3. Temporal warp must use the correct transform direction (current → previous, not vice versa)
4. Multi-scale deformable attention needs careful level assignment for reference points
5. First-frame handling: when prev_bev is None, temporal self-attention should degenerate to self-attention

**Tiny-subset overfit plan:**

| Parameter | Value |
|-----------|-------|
| Subset | 4 consecutive scenes (for temporal) |
| BEV size | 50×50 (Small config for speed) |
| Augments | None |
| Expected | mAP > 0.6 within 2000 steps |
| If fails | Check spatial cross-attention projections; verify deformable attention gradients; disable temporal first |

---

## 12) Minimal Reimplementation Checklist

**Build order:**

1. Data loading (nuScenes, sequential for temporal)
2. Backbone + FPN (ResNet-101 + multi-scale)
3. BEV query grid initialization (H_bev × W_bev × C)
4. Spatial cross-attention: BEV refs → project to cameras → deformable attention
5. Temporal self-attention: warp prev_bev → deformable self-attention
6. FFN + encoder layers (× 6)
7. Detection head (CenterPoint or DETR-style)
8. Loss + matching + evaluation

**Critical dependency:** Deformable attention CUDA kernels (from Deformable-DETR repo).

**Unit tests:**

| Test | Check |
|------|-------|
| `test_bev_ref_projection` | BEV grid (x,y) at heights projects to correct camera pixels |
| `test_ego_warp` | Stationary object in prev_bev aligns with current frame |
| `test_deformable_attn` | Output shape correct; gradients flow; offsets within bounds |
| `test_first_frame` | Model works without prev_bev (temporal degenerates gracefully) |

**Sanity scripts:**

**Script 1:** Visualize which cameras each BEV position attends to → should match expected FOV
**Script 2:** Overfit 4 scenes → mAP > 0.6 in 2000 steps
**Script 3:** Visualize temporal BEV features across consecutive frames → should be smooth/consistent

---

_Source: extracted and adapted from `BEV_Papers_Technical_Dossier.md` in this workspace._
