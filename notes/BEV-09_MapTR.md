# Paper 9: MapTR

**Full title:** MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction
**Authors:** Bencheng Liao, Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Wenyu Liu, Chang Huang (HUST / Horizon Robotics)
**Venue:** ICLR 2023
**arXiv:** 2208.14437

---

## 1) One-page Overview

**Tasks solved:** Online HD map construction — predicting map elements (lane dividers, road boundaries, pedestrian crossings) as vectorized polylines from camera images [MapTR | Sec 1]

**Sensors assumed:** Multi-camera (6 cameras on nuScenes) [MapTR | Sec 4]

**Key novelty:**

- **Vectorized map representation:** Instead of rasterized BEV segmentation, predicts map elements as ordered sets of control points (polylines/polygons) [MapTR | Sec 3.1]
- **Permutation-equivalent set prediction:** Uses a hierarchical set-based matching strategy: first match map elements (outer level), then match point orderings within each element (inner level) [MapTR | Sec 3.2]
- **Map query design:** Each map element is represented by a fixed number of queries (N_pts points per element); M total elements [MapTR | Sec 3.3]
- End-to-end transformer architecture: BEV features → map decoder → polyline predictions
- No post-processing (no NMS, no polyline fitting)

**If you only remember 3 things:**

1. Predicts HD map elements as ordered point sets (polylines), not rasterized masks
2. Hierarchical bipartite matching: element-level Hungarian + point-permutation matching
3. End-to-end from multi-view images to vectorized map without post-processing

---

## 2) Problem Setup and Outputs (Precise)

**Inputs:** Standard multi-camera setup + BEV features (from any BEV encoder).

**Outputs:**

| Output | Shape | Description |
|--------|-------|-------------|
| Element classes | `(B, M, C_cls)` | Class per map element (lane, boundary, crossing) |
| Point coordinates | `(B, M, N_pts, 2)` | Ordered (x, y) BEV coordinates for each element's polyline |

Where M is number of map queries and N_pts is points per element. Not specified in provided inputs: exact defaults.

**Label format:** Each GT map element is a polyline/polygon with an ordered list of 2D BEV points + class label [MapTR | Sec 4].

---

## 3) Coordinate Frames and Geometry

**Frames:** BEV coordinate frame (ego-centric; x-forward, y-left).

**Map element representation [MapTR | Sec 3.1]:**

- Polylines: ordered sequence of (x, y) points in BEV
- Polygons: closed polylines (last point connects to first)
- Points are in ego-centric BEV coordinates (meters)
- Coordinate range: typically [-30m, 30m] × [-15m, 15m] or [-50m, 50m] × [-25m, 25m]

**Permutation equivalence [MapTR | Sec 3.2]:**

- A polyline (A, B, C, D) is equivalent to (D, C, B, A) (reversed direction)
- A polygon (A, B, C) is equivalent to (B, C, A), (C, A, B) (cyclic permutations) and reversed versions
- The matching considers all valid permutations and picks the one with minimum cost

**Geometry sanity checks:**

| # | What to verify | Expected outcome | Failure signature |
|---|---------------|------------------|-------------------|
| 1 | Predicted points within BEV range | All (x,y) within [-50m, 50m] | Points at extreme coordinates or NaN |
| 2 | Polyline ordering | Points form connected line when drawn in order | Zigzag or crossing patterns |
| 3 | Permutation matching | Matched GT-prediction pairs have similar shapes | High Chamfer distance despite good visual overlap (wrong permutation) |
| 4 | Map element count | Not specified in provided inputs | Too many (hundreds) or too few (0-1) predictions |
| 5 | Class balance | Lane dividers most common; crossings rarer | All predictions same class |

---

## 4) Architecture Deep Dive (Module-by-module)

**Block diagram:**

```
Images (B,N,3,H,W)
       │
       ▼
┌──────────────┐
│ BEV Encoder  │  Any: LSS, BEVFormer, etc.
└──────┬───────┘
       │ (B, C, H_bev, W_bev)
       ▼
┌─────────────────────────┐
│  Map Decoder             │
│  ┌───────────────────┐  │
│  │ Map Element Queries│  │  M queries × N_pts points each
│  │ (M*N_pts, C)       │  │
│  └─────────┬─────────┘  │
│            │             │
│  ┌─────────▼─────────┐  │
│  │ Self-Attention     │  │  Queries attend to each other
│  └─────────┬─────────┘  │
│            │             │
│  ┌─────────▼─────────┐  │
│  │ Cross-Attention    │  │  Queries attend to BEV features
│  │ (deformable)       │  │
│  └─────────┬─────────┘  │
│            │             │
│  │ (repeat L layers)    │
└────────────┬────────────┘
             │
             ▼
┌────────────────────┐
│ Classification Head │  → (B, M, C_cls)
│ Point Regression    │  → (B, M, N_pts, 2)
└────────────────────┘
```

**Modules:**

| Module | Purpose | Input → Output | Key operations |
|--------|---------|---------------|----------------|
| BEV Encoder | Create BEV features | Images → `(B, C, H_bev, W_bev)` | Any BEV method |
| Map queries | Learnable | → `(M × N_pts, C)` | M elements × N_pts points |
| Self-attention | Query interaction | `(M×N_pts, C)` → same | Inter- and intra-element attention |
| Cross-attention | BEV feature sampling | Queries + BEV → queries | Deformable cross-attention on BEV features |
| Cls head | Element classification | `(M, C)` → `(M, C_cls)` | Aggregate N_pts queries per element → classify |
| Point head | Point regression | `(M, N_pts, C)` → `(M, N_pts, 2)` | Per-point (x, y) prediction (sigmoid → scaled) |

---

## 5) Forward Pass Pseudocode (Shape-annotated)

```python
def forward(bev_features):
    # bev_features: (B, C, H_bev, W_bev) from upstream BEV encoder
    B = bev_features.shape[0]

    # 1. Initialize map queries
    # M map elements, each with N_pts point queries
    queries = map_query_embed.weight  # (M * N_pts, C)
    queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, M*N_pts, C)

    # 2. Reference points (initial point predictions)
    ref_pts = ref_point_head(queries)  # (B, M*N_pts, 2) → sigmoid

    # 3. Decoder layers
    for layer in decoder_layers:
        queries = layer.self_attn(queries)  # (B, M*N_pts, C)
        queries = layer.cross_attn(queries, bev_features, ref_pts)  # deformable attn
        queries = layer.ffn(queries)

        # Update reference points
        ref_pts = ref_pts + offset_head(queries)  # refine

    # 4. Reshape to elements
    queries = queries.view(B, M, N_pts, C)       # (B, M, N_pts, C)

    # 5. Heads
    elem_feat = queries.mean(dim=2)               # (B, M, C)  average over points
    cls_scores = cls_head(elem_feat)              # (B, M, C_cls)
    pts_coords = pts_head(queries)                # (B, M, N_pts, 2)  sigmoid → BEV coords

    return cls_scores, pts_coords
```

---

## 6) Heads, Targets, and Losses

**Losses [MapTR | Sec 3.3]:**

| Loss | Formula | Weight |
|------|---------|--------|
| Focal loss (cls) | Element classification | 2.0 |
| Point-to-point L1 | L1 between matched predicted and GT point sets | *Plausible defaults (not from paper): 5.0* |
| Edge direction loss | Consistency of polyline segment directions | *Plausible defaults (not from paper): 0.005* |
| Chamfer distance (optional) | Chamfer between predicted and GT polylines | *Plausible defaults (not from paper)* |

**Hierarchical matching [MapTR | Sec 3.2]:**

1. **Element-level:** Hungarian matching between M predicted elements and GT elements using cost = cls_cost + point_cost
2. **Point-level:** For each matched pair, find the optimal point permutation (considering valid permutations: forward, reverse, cyclic shifts) that minimizes the total point L1 cost

**Loss debugging checklist:**

| # | Bug | Symptom | Quick test |
|---|-----|---------|------------|
| 1 | Permutation matching not implemented | High loss even when polylines visually overlap | Compare Chamfer distance (permutation-invariant) with L1 |
| 2 | Point coordinates outside BEV range | Sigmoid scaling wrong | Check that predicted coords are in expected meter range |
| 3 | Element count mismatch | Too many/few elements matched | Print matching statistics per frame |
| 4 | Polyline direction loss confusing forward/reverse | Predicted polylines systematically reversed | Visualize matched pairs; check permutation cost |
| 5 | Class imbalance (few crossings) | Model never predicts crossings | Add class weights or oversample |

---

## 7) Data Pipeline and Augmentations

**Data:** nuScenes map annotations (lane dividers, road boundaries, pedestrian crossings) converted to polylines.


## 8) Training Pipeline (Reproducible)

**Training [MapTR | Sec 4]:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 6e-4 *Plausible defaults (not from paper)* |
| Epochs | 24 on nuScenes |
| BEV encoder | BEVFormer or LSS-based |

---

## 9) Dataset + Evaluation Protocol

**Metrics [MapTR | Sec 4]:**

| Metric | Description |
|--------|-------------|
| mAP | Mean AP over map element classes; thresholds based on Chamfer distance (0.5, 1.0, 1.5m) |
| Chamfer distance | Avg distance between predicted and GT polyline point sets |

- Not specified in provided inputs. Plausible defaults (not from paper): consult the cited paper table/benchmark entry for exact values.

**Top ablations:**

1. Hierarchical matching outperforms naive matching in ablations [MapTR | Table 1/ablation tables]; exact delta Not specified in provided inputs.
2. Number of points per element (N_pts): 20 points is a good balance
3. BEV encoder quality: better BEV features directly improve map quality

---

## 10) Results Summary + Ablations

- Not specified in provided inputs. Plausible defaults (not from paper): summarize main benchmark numbers directly from cited tables.

## 11) Practical Insights

**10 engineering takeaways:**

1. Vectorized maps are much more useful for planning than rasterized maps.
2. The hierarchical matching is the key algorithmic contribution — get it right.
3. Point ordering within polylines matters — the permutation-equivalent matching solves this elegantly.
4. MapTR works with any BEV encoder — it's a modular map head.
5. The number of map queries (M) should exceed the typical number of map elements per frame.
6. Points per element (N_pts=20) determines the resolution of predicted polylines.
7. Chamfer distance is the right metric — it's invariant to point ordering.
8. Cross-attention to BEV features means map predictions leverage the full BEV representation.
9. Post-processing free: no NMS, no polyline simplification needed.
10. MapTRv2 adds one-to-many matching and auxiliary dense losses for further improvement.

**5 gotchas:**

1. Permutation matching cost is expensive — precompute all valid permutations
2. Polyline direction: lanes have direction; crossings don't — handle differently
3. Map annotations in nuScenes are global; must transform to ego frame
4. Short map elements (few GT points) need careful resampling to N_pts points
5. BEV feature resolution limits map precision — use high-resolution BEV for mapping

**Tiny-subset overfit plan:**

| Parameter | Value |
|-----------|-------|
| Subset | 4 scenes |
| Classes | Lane divider only |
| Expected | mAP > 0.6 within 3000 steps |
| If fails | Check matching; visualize predicted vs. GT polylines |

---

## 12) Minimal Reimplementation Checklist

**Build order:**

1. Map annotation loading + polyline extraction
2. BEV encoder (reuse existing)
3. Map query design (M elements × N_pts points)
4. Transformer decoder (self-attn + cross-attn on BEV)
5. Hierarchical matching (element-level Hungarian + point permutation)
6. Losses (focal + point L1 + direction)
7. Evaluation (Chamfer-based mAP)

**Sanity scripts:**

**Script 1:** Visualize GT polylines on BEV grid → should trace roads/lanes
**Script 2:** Overfit 4 scenes → Chamfer < 0.5m within 3000 steps
**Script 3:** Plot predicted vs. GT polylines → should overlap closely

---
---

---

_Source: extracted and adapted from `BEV_Papers_Technical_Dossier.md` in this workspace._
