# Paper 2: DETR3D

**Full title:** DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries
**Authors:** Yue Wang, Vitor Guizilini, Timanyu Zhang, Yilun Wang, Hang Zhao, Justin Solomon
**Venue:** CoRL 2021
**arXiv:** 2110.06922

---

## 1) One-page Overview

**Tasks solved:** 3D object detection from multi-view cameras [DETR3D | Sec 1]

**Sensors assumed:** Multi-camera only (6 cameras on nuScenes) [DETR3D | Sec 4]

**Key novelty:**

- **3D-to-2D query projection:** Instead of lifting 2D features to 3D (like LSS), DETR3D uses learnable 3D reference points that project onto 2D feature maps to sample features [DETR3D | Sec 3.2]
- Adapts DETR-style set prediction to 3D: object queries → transformer decoder → bipartite matching [DETR3D | Sec 3.1]
- No explicit depth estimation or BEV grid construction — 3D reasoning happens implicitly through query-to-image attention [DETR3D | Sec 3]
- No NMS required (set-based prediction with Hungarian matching) [DETR3D | Sec 3.3]
- Multi-camera fusion is implicit: each query's reference point projects to whichever cameras it's visible in [DETR3D | Sec 3.2]

**If you only remember 3 things:**

1. Learnable 3D reference points projected to 2D → sample image features → no explicit BEV grid
2. DETR-style set prediction with Hungarian matching for 3D detection
3. Inverse of LSS philosophy: instead of lifting all pixels to 3D, query specific 3D locations and pull 2D features

---

## 2) Problem Setup and Outputs (Precise)

**Input tensors:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| Images | `(B, N, 3, H, W)` | Multi-view images; N=6 for nuScenes |
| Intrinsics | `(B, N, 3, 3)` | Camera K matrices |
| Extrinsics | `(B, N, 4, 4)` | Camera-to-ego (or world) transforms |
| (Optional) Ego pose | `(B, 4, 4)` | For temporal alignment |

**Outputs:**

| Output | Shape | Description |
|--------|-------|-------------|
| 3D bounding boxes | `(B, Q, 10)` | Per-query: (cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy) |
| Classification scores | `(B, Q, C_cls+1)` | Per-query class probabilities (+ no-object class) |

Where Q = number of queries (typically 900) [DETR3D | Sec 4].

**Label format:** nuScenes 3D bounding boxes: center (x, y, z) in global frame, size (w, l, h) in meters, rotation as quaternion (converted to yaw), velocity (vx, vy) [DETR3D | Sec 4].

---

## 3) Coordinate Frames and Geometry

**Frames used:**

- **Camera frame:** Per-camera; standard pinhole model
- **Ego frame:** Vehicle-centric
- **Global/world frame:** Used for reference points and targets (nuScenes global coordinates)

**Transforms:**

- **3D reference point → 2D projection:** For each query's reference point `p_3d = (x, y, z)` in 3D (global frame), project to each camera: `p_2d = K · T_cam←world · p_3d`. Only sample features from cameras where the projection falls within the image bounds [DETR3D | Sec 3.2].
- **Ego-motion:** Reference points live in the current ego frame or global frame; temporal alignment requires ego-pose compensation.

**BEV grid definition:** No explicit BEV grid — DETR3D operates in 3D query space [DETR3D | Sec 3].

**Common pitfalls:**

- Reference point projections must check image bounds (including behind-camera rejection: z_cam > 0)
- Coordinate frame of reference points must match coordinate frame of ground truth for matching
- Projecting to feature map coordinates requires dividing by stride after camera projection

**Geometry sanity checks:**

| # | What to verify | Expected outcome | Failure signature |
|---|---------------|------------------|-------------------|
| 1 | Project learned reference points to images after training | Points cluster on objects | Points randomly scattered or all in one camera |
| 2 | Check reference point z_cam after projection | Positive for visible points | Negative z means behind camera — should be masked |
| 3 | Feature map coordinate scaling | Projected (u,v) divided by stride matches feature map pixels | Off-by-one errors; features sampled from wrong locations |
| 4 | Hungarian matching costs | Matched pairs have small L1 distance | All queries match to same GT or no matches found |
| 5 | Multi-camera visibility | Each query samples from 1–3 cameras typically | Query sampling from 0 cameras (dead query) or all 6 (wrong projection) |

---

## 4) Architecture Deep Dive (Module-by-module)

**Block diagram:**

```
Images (B,N,3,H,W)
       │
       ▼
┌─────────────────┐
│  Backbone (Res101│  + FPN
│  or VoVNet)      │
└──────┬──────────┘
       │ Multi-scale features: (B,N,C,H/s,W/s) for s in {8,16,32,64}
       ▼
┌─────────────────────────┐
│  Transformer Decoder     │
│  (6 layers)              │
│  ┌─────────────────┐    │
│  │ Object Queries   │    │  Q = 900 learned queries
│  │ (Q, C=256)       │    │
│  └────────┬────────┘    │
│           │              │
│  ┌────────▼────────┐    │
│  │ Self-Attention   │    │  Queries attend to each other
│  └────────┬────────┘    │
│           │              │
│  ┌────────▼────────────┐│
│  │ 3D-to-2D Cross-Attn ││  Reference pts → project to 2D → sample features
│  │ (Feature Sampling)   ││
│  └────────┬────────────┘│
│           │              │
│  │ (repeat 6 layers)    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────┐
│  Detection Heads │  cls + reg (applied at each decoder layer)
└─────────────────┘
```

**Modules in detail:**

| Module | Purpose | Input → Output | Key operations |
|--------|---------|---------------|----------------|
| Backbone | Image features | `(B*N, 3, H, W)` → multi-scale `{(B*N, 256, H/8, W/8), ...}` | ResNet-101 + FPN [DETR3D | Sec 4] |
| Object queries | Learnable embeddings | N/A → `(Q, 256)` | Learned parameters; Q=900 |
| Reference point head | Predict 3D reference from query | `(Q, 256)` → `(Q, 3)` | Linear → sigmoid → scale to 3D range |
| **3D-to-2D feature sampling** | Project ref pts to images, sample features | `(Q, 3)` + image feats → `(Q, 256)` | **This is how 3D info enters:** project reference to each camera, bilinear sample from feature maps, average over cameras [DETR3D | Sec 3.2] |
| Self-attention | Query-query interaction | `(Q, 256)` → `(Q, 256)` | Standard multi-head self-attention |
| FFN | Non-linear transform | `(Q, 256)` → `(Q, 256)` | Two-layer MLP with ReLU |
| Cls head | Classify each query | `(Q, 256)` → `(Q, C_cls)` | Linear layer(s) |
| Reg head | Regress box parameters | `(Q, 256)` → `(Q, 10)` | Linear layer(s); center as offset from reference point |

**Where BEV happens:** There is no explicit BEV grid. The "3D reasoning" happens implicitly through the 3D reference points that live in 3D space and pull features from 2D images [DETR3D | Sec 3.2]. This is a sparse, query-based representation rather than a dense BEV grid.

**Transformer details:**

- **Query types:** Learned object queries (Q=900); each associated with a 3D reference point
- **Attention pattern:** Self-attention among queries + cross-attention via feature sampling (not standard cross-attention — features are gathered by projection, not by attention weights)
- **Positional embeddings:** 3D reference point coordinates serve as positional information
- **Matching:** Hungarian matching between predictions and ground truth using cost = λ_cls · cls_cost + λ_L1 · L1_cost [DETR3D | Sec 3.3]

---

## 5) Forward Pass Pseudocode (Shape-annotated)

```python
def forward(images, intrinsics, extrinsics):
    B, N, _, H, W = images.shape

    # 1. Extract multi-scale image features
    imgs = images.view(B*N, 3, H, W)
    feat_maps = backbone_fpn(imgs)  # list of (B*N, 256, H/s, W/s) for s in [8,16,32,64]
    # Reshape: each level → (B, N, 256, H/s, W/s)

    # 2. Initialize queries and reference points
    queries = object_queries.weight  # (Q, 256)  learned
    queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, 256)
    ref_pts = reference_head(queries)  # (B, Q, 3)  → sigmoid → scaled to [-61.2, 61.2]m range

    # 3. Decoder layers (L=6)
    for layer in decoder_layers:
        # 3a. Self-attention
        queries = layer.self_attn(queries)  # (B, Q, 256)

        # 3b. 3D-to-2D feature sampling
        sampled_feats = torch.zeros(B, Q, 256)
        for cam in range(N):
            # Project reference points to camera
            pts_cam = extrinsics[cam] @ ref_pts  # (B, Q, 3)  in camera frame
            valid = pts_cam[..., 2] > 0          # behind-camera check
            pts_2d = intrinsics[cam] @ pts_cam   # (B, Q, 2)  pixel coords
            pts_2d = pts_2d / stride              # feature map coords

            # Bilinear sample from feature maps (multi-scale)
            for level, feat in enumerate(feat_maps):
                sampled = F.grid_sample(feat[:, cam], pts_2d)  # (B, Q, 256)
                sampled_feats += sampled * valid

        sampled_feats /= num_valid_cameras  # average over visible cameras

        # 3c. Cross-attention (add sampled features)
        queries = queries + sampled_feats  # simplified; actual uses more complex mixing

        # 3d. FFN
        queries = layer.ffn(queries)  # (B, Q, 256)

        # 3e. Update reference points
        ref_pts = ref_pts + ref_offset_head(queries)  # refine reference points

    # 4. Detection heads (applied at each layer for auxiliary loss)
    cls_scores = cls_head(queries)   # (B, Q, C_cls)
    box_params = reg_head(queries)   # (B, Q, 10)
    # box center = ref_pts + predicted offset

    return cls_scores, box_params
```

**Activation/scale sanity:**

- `ref_pts` after sigmoid: in [0, 1], then scaled to metric range; typical: [-61.2m, 61.2m]
- `queries` magnitude: O(1) due to layer normalization
- `sampled_feats`: similar magnitude to image features; O(1) after normalization
- `cls_scores`: raw logits; sigmoid applied for focal loss

---

## 6) Heads, Targets, and Losses

**Heads:**

| Head | Output shape | Activation | Purpose |
|------|-------------|------------|---------|
| Classification | `(B, Q, C_cls)` | Sigmoid (for focal loss) | Object class prediction |
| Regression | `(B, Q, 10)` | None (direct regression) | Box center offset, size, rotation, velocity |

**Loss terms:**

| Loss | Formula | Target | Default weight |
|------|---------|--------|----------------|
| Focal loss (cls) | `FL(p, y) = -α(1-p)^γ · log(p)` | Matched GT class; α=0.25, γ=2.0 [DETR3D | Sec 3.3] | λ_cls = 2.0 |
| L1 loss (reg) | `L1(pred_box, gt_box)` | Matched GT box parameters | λ_L1 = 0.25 |

*Plausible defaults (not from paper): loss weights follow DETR convention; exact values may differ.*

**Assignment strategy:** Hungarian matching [DETR3D | Sec 3.3]:

- Cost matrix: `C = λ_cls · cls_cost + λ_L1 · L1_cost` over all (query, GT) pairs
- Optimal assignment via scipy.linear_sum_assignment
- Unmatched queries assigned "no-object" class
- Applied at each decoder layer (auxiliary losses) [DETR3D | Sec 3.3]

**Loss debugging checklist:**

| # | Bug | Symptom | Quick test |
|---|-----|---------|------------|
| 1 | Reference points outside dataset spatial range | All queries match to background | Print ref_pts statistics; should be within [-61.2, 61.2]m |
| 2 | Wrong projection matrix order (K × E vs E × K) | Features sampled from wrong locations | Visualize projected ref pts on images; should land on objects |
| 3 | Hungarian matching cost explosion | Matching degenerates; loss very high | Print cost matrix statistics; L1 cost should be O(1) not O(100) |
| 4 | Missing auxiliary losses from intermediate layers | Slow convergence | Verify losses from all 6 decoder layers are being summed |
| 5 | Velocity target in wrong frame | vx, vy don't match predictions | Verify velocity is in ego frame and consistent with box center frame |
| 6 | No-object class weight too low | Predicts objects everywhere | Increase background class weight or tune focal loss α |

---

## 7) Data Pipeline and Augmentations

**Data loading:** Same nuScenes format as LSS: 6 cameras, 3D box annotations, calibration.

**Augmentations [DETR3D | Sec 4]:**

- Image augmentation: random resize, crop, color jitter
- *Plausible defaults (not from paper): no BEV-space augments since there's no explicit BEV grid*
- GT sampling may or may not be used (common in 3D detection but not always for camera-only)

**Temporal sampling:** Single-frame in base DETR3D [DETR3D | Sec 3]. No temporal fusion.

**Augmentation safety:**

| Augment | Risk | Validation |
|---------|------|------------|
| Image resize/crop | Must update intrinsics and re-check projection validity | Project GT boxes after augmentation; should still overlay correctly |
| Horizontal flip | Must also flip extrinsics and GT boxes | Verify left/right cameras swap consistently with boxes |

**Minimal-augment debug config:** No augmentation; original resolution images.

---

## 8) Training Pipeline (Reproducible) (Reproducible) (Reproducible)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW [DETR3D | Sec 4] |
| Learning rate | 2e-4 (backbone: 2e-5, 10× lower) [DETR3D | Sec 4] |
| Batch size | 8 (1 per GPU × 8 GPUs) *Plausible defaults (not from paper)* |
| Epochs | 24 epochs on nuScenes [DETR3D | Sec 4] |
| LR schedule | Step decay or cosine *Plausible defaults (not from paper)* |
| Warmup | Linear warmup for 500 iterations *Plausible defaults (not from paper)* |
| Backbone | Pretrained ResNet-101 or VoVNet-99 [DETR3D | Sec 4] |
| GPUs | 8× V100 [DETR3D | Sec 4] |

**Stability & convergence:**

| Failure mode | Symptom | Fix |
|-------------|---------|-----|
| Queries all predict same location | Low diversity in ref pts | Initialize ref pts with wider distribution; add query noise |
| Matching instability (oscillating assignments) | Loss bounces | Lower LR; use gradient clipping (max_norm=35) |
| Backbone overfitting | val mAP drops while train mAP rises | Lower backbone LR further; add dropout |
| Slow convergence (DETR-like) | 24 epochs insufficient | Standard for DETR-style; consider deformable attention for faster convergence |
| Dead queries (never matched) | Many queries predict background only | Initialize queries more uniformly in 3D space |

---

## 9) Dataset + Evaluation Protocol

**Dataset:** nuScenes detection benchmark [DETR3D | Sec 4]

**Metrics:**

| Metric | Description |
|--------|-------------|
| mAP | Mean Average Precision over 10 classes with BEV distance thresholds {0.5, 1, 2, 4}m |
| NDS | nuScenes Detection Score = weighted combination of mAP + TP metrics |
| mATE | Mean Average Translation Error |
| mASE | Mean Average Scale Error |
| mAOE | Mean Average Orientation Error |
| mAVE | Mean Average Velocity Error |
| mAAE | Mean Average Attribute Error |

**Evaluation range:** Detection within 50m radius from ego vehicle.

**Data gotchas:**

| Gotcha | Detail |
|--------|--------|
| Class imbalance | Construction vehicle, trailer very rare vs. car |
| Velocity annotation | Derived from consecutive frames; can be noisy at low speeds |
| Detection range | Objects beyond 50m still exist but aren't evaluated |
| Annotation frequency | Only 2Hz keyframes annotated; if using sweeps, need interpolation |

---

## 10) Results Summary + Ablations

**Main results [DETR3D | Table 1]:** On nuScenes val set with ResNet-101 backbone:

- Not specified in provided inputs. Plausible defaults (not from paper): consult the cited paper table/benchmark entry for exact values.
- Not specified in provided inputs. Plausible defaults (not from paper): use exact VoVNet-99 + CBGS metrics from cited table.

**Top 3 ablations [DETR3D | Table 2–3]:**

1. **Feature sampling vs. global attention:** 3D-to-2D feature sampling significantly outperforms projecting all features globally; spatial precision matters
2. **Multi-scale features:** Improves over single-scale in paper ablations [DETR3D | Table 2–3]; exact delta Not specified in provided inputs.
3. **Number of decoder layers:** 6 layers substantially outperforms 1-3 layers; diminishing returns after 6

**Failure modes:** (1) Small distant objects hard to detect (few feature pixels); (2) Depth ambiguity — no explicit depth supervision means z-coordinate accuracy limited; (3) Set-based matching can be unstable in early training.

---

## 11) Practical Insights

**10 engineering takeaways:**

1. DETR3D avoids the memory cost of explicit BEV grids — no D×H×W frustum volume needed.
2. Feature sampling is the key innovation: bilinear interpolation from multi-scale FPN maps at projected 2D locations.
3. Reference point initialization matters: too narrow → queries compete for same objects; too wide → many queries wasted on empty space.
4. Behind-camera masking (z_cam > 0) is critical — without it, features from wrong cameras leak in.
5. Multi-camera averaging (not concatenation) of sampled features keeps the model invariant to camera count.
6. Auxiliary losses at every decoder layer (standard DETR trick) significantly speed up convergence.
7. DETR3D's query-based approach naturally extends to tracking (associate queries across frames).
8. No NMS needed — but in practice, a light confidence threshold is applied.
9. Backbone LR should be 10× lower than heads — pretrained features should be fine-tuned gently.
10. Feature map stride alignment is crucial: projected (u,v) must be divided by the correct stride for each FPN level.

**5 gotchas:**

1. Projection must handle behind-camera points (z < 0); forgetting to mask these causes hallucinated detections
2. Grid_sample expects normalized coordinates in [-1, 1]; forgetting to normalize causes blank features
3. Hungarian matching is O(N^3) — with Q=900 queries and many GT objects, this can be slow
4. Reference points should be in the same coordinate frame as GT boxes (usually ego or global)
5. Velocity regression: nuScenes annotates velocity in global frame; may need conversion to ego frame

**5 debugging checks:**

1. **Ref pt visualization:** Plot learned reference points in 3D space → should distribute in driving-relevant region
2. **Sampling visualization:** For top-scoring queries, highlight sampled 2D locations on images → should be on objects
3. **Matching statistics:** Print how many queries get matched per frame → typically 5-30 for nuScenes
4. **Per-class AP:** Check per-class performance → large vehicles detected first, small objects (pedestrians) harder
5. **Depth accuracy:** Compare predicted z to GT z → check systematic bias

**Tiny-subset overfit plan:**

| Parameter | Value |
|-----------|-------|
| Subset size | 4 scenes (Not specified exact frame count in provided inputs) |
| Heads to keep | All (cls + reg) |
| Augments to disable | ALL |
| Expected behavior | mAP > 0.8 on training set within 2000 steps |
| If fails to overfit | Check reference point projection; verify Hungarian matching is finding correct pairs; print cost matrix |

---

## 12) Minimal Reimplementation Checklist

**Build order:**

1. **Data preprocessing:** Load nuScenes; extract images, calibration matrices, 3D box annotations
2. **Backbone + FPN:** ResNet-101 pretrained + FPN with 256 channels
3. **Object queries:** Initialize Q=900 learnable embeddings
4. **Reference point head:** Linear → sigmoid → scale to 3D range
5. **3D-to-2D projection:** Implement projection + behind-camera masking + grid_sample
6. **Decoder layers:** Self-attention + feature sampling + FFN × 6
7. **Detection heads:** Cls (focal loss) + reg (L1) at each layer
8. **Hungarian matching:** Implement cost matrix + scipy matching
9. **Evaluation:** nuScenes eval toolkit

**Unit tests:**

| Test | Check |
|------|-------|
| `test_projection` | Known 3D point projects to expected 2D pixel in each camera |
| `test_behind_camera` | Point behind camera (z<0) is masked out |
| `test_grid_sample_normalization` | Features sampled at image center return center-pixel feature |
| `test_hungarian_matching` | Identity GT (queries = GT boxes) produces perfect matching |
| `test_loss_gradient` | Loss has non-zero gradients w.r.t. queries and backbone |

**Minimal sanity scripts:**

**Script 1:** Project GT box centers onto images; visualize as dots → should be on object centers
**Script 2:** Overfit 4 scenes; should reach near-zero L1 loss within 2000 steps
**Script 3:** Eval on training set; mAP should match training metric (no eval-time projection bugs)

---
---

---

_Source: extracted and adapted from `BEV_Papers_Technical_Dossier.md` in this workspace._
