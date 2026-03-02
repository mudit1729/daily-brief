# Paper 10: UniAD

**Full title:** Planning-oriented Autonomous Driving
**Authors:** Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghuo Sber, Xizhou Zhu, Zhe Wang, Haifeng Lu, Jifeng Dai, Yu Qiao, Hongyang Li, et al.
**Venue:** CVPR 2023 (Best Paper)
**arXiv:** 2212.10156

---

## 1) One-page Overview

**Tasks solved:** End-to-end autonomous driving: perception (detection + tracking + mapping + occupancy) → prediction (motion forecasting) → planning [UniAD | Sec 1]

**Sensors assumed:** Multi-camera (6 cameras on nuScenes) [UniAD | Sec 4]

**Key novelty:**

- **Unified multi-task architecture:** A single model performs detection, tracking, online mapping, motion forecasting, occupancy prediction, and planning — all connected via query-based representations [UniAD | Sec 3]
- **Planning-oriented design:** All intermediate tasks are designed to serve the ultimate planning objective; each module passes information to the next [UniAD | Sec 3.6]
- **Query-based pipeline:** Track queries propagate across frames; map queries predict vectorized maps; motion queries forecast future trajectories; planning queries output ego trajectory [UniAD | Sec 3]
- BEV backbone: BEVFormer-style spatial-temporal BEV encoder
- Demonstrates that joint training of all tasks improves planning performance over separate models

**If you only remember 3 things:**

1. Full stack: detect → track → map → forecast → plan — all in one model, all differentiable
2. Query-based design throughout: track queries evolve over time and feed into motion forecasting
3. CVPR 2023 Best Paper — demonstrates planning-oriented multi-task learning is superior to isolated modules

---

## 2) Problem Setup and Outputs (Precise)

**Inputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| Images | `(B, N, 3, H, W)` | N=6 cameras, sequential frames |
| Intrinsics | `(B, N, 3, 3)` | Camera K |
| Extrinsics | `(B, N, 4, 4)` | Camera-to-ego |
| Ego pose history | `(B, T_hist, 4, 4)` | Past ego poses |
| HD map (optional) | — | For evaluation comparison |

**Outputs:**

| Output | Shape | Description |
|--------|-------|-------------|
| Detection | `(B, Q_det, 10)` | 3D boxes |
| Tracking | `(B, Q_track, 10)` | Tracked box associations across frames |
| Online map | `(B, M, N_pts, 2)` | Vectorized map elements |
| Motion forecast | `(B, Q_track, K_modes, T_fut, 2)` | Multi-modal future trajectories for each agent |
| Occupancy forecast | `(B, T_fut, H_bev, W_bev, C_occ)` | Future occupancy predictions |
| **Planning** | `(B, T_plan, 3)` | Ego trajectory: (x, y, heading) × T_plan future steps |

Where T_fut = 6 (3 seconds at 2Hz), T_plan = 6, K_modes = 6 trajectory modes [UniAD | Sec 4].

---

## 3) Coordinate Frames and Geometry

**Frames:** Camera, ego (current), ego (past/future), BEV grid.

**Temporal alignment:** All representations are in the current ego frame. Past and future ego poses transform historical/future coordinates to the current frame [UniAD | Sec 3.1].

**Track query propagation:** Track queries from frame t are ego-motion-compensated to frame t+1, then updated with new observations [UniAD | Sec 3.2].

**Geometry sanity checks:**

| # | What to verify | Expected outcome | Failure signature |
|---|---------------|------------------|-------------------|
| 1 | Track query temporal consistency | Same object's query tracks it across frames | Query jumps between objects |
| 2 | Motion forecast coordinate frame | Future positions relative to current ego | Trajectories in wrong frame (absolute vs. relative) |
| 3 | Planning output frame | Ego trajectory in current ego frame | Planned path drifts or rotates incorrectly |
| 4 | BEV feature alignment across time | BEVFormer temporal features align | Double objects (ghosting) |
| 5 | Occupancy forecast alignment | Future occupancy matches actual future scene | Occupancy doesn't evolve realistically |

---

## 4) Architecture Deep Dive (Module-by-module)

**Block diagram:**

```
Images (B,N,3,H,W) × T frames
              │
              ▼
┌────────────────────┐
│ BEVFormer Backbone  │  Spatial-temporal BEV encoder
└──────────┬─────────┘
           │ (B, C, H_bev, W_bev)
           ▼
┌──────────────────────────────────────────────────────┐
│                    UniAD Pipeline                      │
│                                                        │
│  ┌────────────┐    ┌────────────┐    ┌──────────────┐│
│  │ TrackFormer │───→│ MapFormer  │───→│ MotionFormer ││
│  │ (Detect +   │    │ (Online    │    │ (Forecast    ││
│  │  Track)     │    │  Map)      │    │  trajectories││
│  └──────┬─────┘    └──────┬─────┘    └──────┬───────┘│
│         │                 │                  │        │
│         └─────────────────┴──────────────────┘        │
│                           │                           │
│                    ┌──────▼───────┐                   │
│                    │ OccFormer    │                   │
│                    │ (Future occ) │                   │
│                    └──────┬───────┘                   │
│                           │                           │
│                    ┌──────▼───────┐                   │
│                    │ Planner      │                   │
│                    │ (Ego traj)   │                   │
│                    └──────────────┘                   │
└──────────────────────────────────────────────────────┘
```

**Modules:**

| Module | Purpose | Input → Output | Key operations |
|--------|---------|---------------|----------------|
| BEVFormer | BEV features | Multi-view images → `(B, C, H_bev, W_bev)` | Spatial + temporal cross-attention [UniAD | Sec 3.1] |
| **TrackFormer** | Detection + tracking | BEV + track queries → updated queries + boxes | DETR-style detection with temporal query propagation; newborn queries for new objects [UniAD | Sec 3.2] |
| **MapFormer** | Online mapping | BEV + map queries → vectorized map | Similar to MapTR; polyline prediction [UniAD | Sec 3.3] |
| **MotionFormer** | Motion forecasting | Track queries + map features → future trajectories | Cross-attention between agent queries and map/scene context; multi-modal outputs [UniAD | Sec 3.4] |
| **OccFormer** | Future occupancy | BEV + agent forecasts → future occupancy | Predict occupancy at future timesteps [UniAD | Sec 3.5] |
| **Planner** | Ego planning | BEV + occ + agent forecasts → ego trajectory | GRU-based planning with collision cost; selects among K modes [UniAD | Sec 3.6] |

**Information flow:** Detection → Tracking → Mapping → Motion prediction → Occupancy prediction → Planning. Each module's output queries/features feed into the next [UniAD | Sec 3].

---

## 5) Forward Pass Pseudocode (Shape-annotated)

```python
def forward(images_seq, calibration, ego_poses, prev_track_queries=None):
    B = images_seq.shape[0]

    # 1. BEV backbone (BEVFormer)
    bev_feat = bevformer(images_seq, calibration, ego_poses)  # (B, C, 200, 200)

    # 2. TrackFormer: detect + track
    if prev_track_queries is not None:
        # Ego-motion compensate previous track queries
        track_queries = ego_compensate(prev_track_queries, ego_motion)
    else:
        track_queries = None

    det_queries = learnable_det_queries  # (Q_det, C)
    all_queries = concat([track_queries, det_queries]) if track_queries else det_queries

    track_out = track_decoder(all_queries, bev_feat)  # queries → boxes + scores
    # Output: updated track queries (B, Q_track, C), boxes (B, Q_track, 10)

    # 3. MapFormer: online mapping
    map_queries = learnable_map_queries  # (M * N_pts, C)
    map_out = map_decoder(map_queries, bev_feat)  # polylines (B, M, N_pts, 2)

    # 4. MotionFormer: trajectory forecasting
    # Agent queries cross-attend to BEV, map features, and other agents
    motion_queries = track_out.queries  # (B, Q_track, C)
    for layer in motion_decoder_layers:
        motion_queries = layer.self_attn(motion_queries)     # agent-agent interaction
        motion_queries = layer.map_cross_attn(motion_queries, map_out.features)  # agent-map
        motion_queries = layer.bev_cross_attn(motion_queries, bev_feat)          # agent-scene

    future_traj = traj_head(motion_queries)  # (B, Q_track, K_modes, T_fut, 2)
    traj_scores = mode_head(motion_queries)  # (B, Q_track, K_modes)

    # 5. OccFormer: future occupancy
    future_occ = occ_decoder(bev_feat, future_traj)  # (B, T_fut, H_bev, W_bev, C_occ)

    # 6. Planner: ego trajectory
    # Cross-attend to BEV features and future occupancy
    ego_query = learnable_ego_query  # (1, C)
    for t in range(T_plan):
        ego_query = planner_attn(ego_query, bev_feat, future_occ[:, t])
        ego_traj[t] = traj_regressor(ego_query)  # (x, y, heading)

    # Cache track queries for next frame
    return {
        'detection': track_out.boxes,
        'tracking': track_out.ids,
        'map': map_out.polylines,
        'motion': future_traj,
        'occupancy': future_occ,
        'planning': ego_traj,  # (B, T_plan, 3)
    }, track_out.queries.detach()
```

---

## 6) Heads, Targets, and Losses

**Multi-task losses [UniAD | Sec 3, Table 6]:**

| Task | Loss | Weight |
|------|------|--------|
| Detection | Focal (cls) + L1 (reg) | λ_det |
| Tracking | Track association loss | λ_track |
| Mapping | Focal (cls) + point L1 + direction | λ_map |
| Motion forecasting | min-ADE over K modes + collision loss | λ_motion |
| Occupancy | BCE (per-voxel-per-timestep) | λ_occ |
| **Planning** | L2 trajectory error + collision penalty | λ_plan |

*Plausible defaults (not from paper): exact λ values tuned via multi-task optimization; typical ranges 0.1–10.0.*

**Loss debugging checklist:**

| # | Bug | Symptom | Quick test |
|---|-----|---------|------------|
| 1 | Task gradient conflict | One task improves while others degrade | Monitor per-task losses independently; use gradient normalization |
| 2 | Tracking query propagation broken | Track queries reset every frame | Check ego-compensation; verify query ID persistence |
| 3 | Motion forecasting multi-modal collapse | All K modes predict same trajectory | Check min-ADE loss (winner-take-all); verify mode diversity |
| 4 | Planning ignores other agents | Ego trajectory collides with predicted agent paths | Verify collision loss is active; check cross-attention to motion queries |
| 5 | Occupancy prediction is static | Same prediction for all future timesteps | Verify temporal encoding; check that agent forecasts influence occupancy |
| 6 | Task ordering dependencies | Earlier tasks not converging → downstream tasks can't learn | Train earlier tasks first (staged training) |

---

## 7) Data Pipeline and Augmentations

**Sequential data:** UniAD requires sequential frames for tracking and temporal BEV [UniAD | Sec 4].

**Augmentations:** Standard image augments; BEV augments applied consistently to all tasks.

**Planning labels:** Ego trajectory from nuScenes ego poses (future 3 seconds) [UniAD | Sec 4].

---

## 8) Training Pipeline (Reproducible) (Reproducible) (Reproducible)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 2e-4 |
| Epochs | 20 on nuScenes [UniAD | Sec 4] |
| **Multi-stage** | Stage 1: BEVFormer pretrain → Stage 2: joint multi-task [UniAD | Sec 4] |
| GPUs | 8× A100 [UniAD | Sec 4] |
| Batch size | 8 (1 per GPU) |
| BEV backbone | BEVFormer with ResNet-101 |

**Multi-stage training [UniAD | Sec 4]:**

1. **Stage 1:** Pretrain BEVFormer backbone on detection task alone
2. **Stage 2:** Fine-tune entire UniAD model jointly on all tasks

**Stability & convergence:**

| Failure | Symptom | Fix |
|---------|---------|-----|
| Upstream task failure cascades | If detection is bad, everything downstream fails | Pretrain upstream tasks; freeze during early downstream training |
| Multi-task gradient conflict | Some tasks improve, others degrade | Use uncertainty-based loss weighting or GradNorm |
| Planning not learning | L2 error doesn't decrease | Verify planning labels; check that ego trajectory GT is in correct frame |
| Memory overflow | Full pipeline is very large | Reduce BEV resolution; fewer decoder layers; gradient checkpointing |

---

## 9) Dataset + Evaluation Protocol

**Metrics [UniAD | Table 1–4]:**

| Task | Metric |
|------|--------|
| Detection | mAP, NDS |
| Tracking | AMOTA, AMOTP |
| Mapping | mAP (Chamfer-based) |
| Motion | minADE, minFDE, MissRate |
| Occupancy | IoU |
| **Planning** | **L2 error (1s, 2s, 3s), Collision rate** |

**Main results [UniAD | Table 1]:**

- Not specified in provided inputs. Plausible defaults (not from paper): consult the cited paper table/benchmark entry for exact values.
- Detection performance improves with stronger BEV backbones [UniAD | Sec 3]; exact mAP range Not specified in provided inputs.
- Joint training improves planning over separate-module baselines

**Top ablations [UniAD | Table 5–6]:**

1. **Joint training vs. separate modules:** Joint training improves planning by reducing collision rate
2. **Which tasks help planning most:** Motion forecasting and occupancy have largest impact on planning
3. **Query-based communication:** Query passing between tasks outperforms independent BEV feature sharing

---

## 10) Results Summary + Ablations

- Not specified in provided inputs. Plausible defaults (not from paper): summarize main benchmark numbers directly from cited tables.

## 11) Practical Insights

**10 engineering takeaways:**

1. UniAD demonstrates that end-to-end multi-task learning is viable and beneficial for planning.
2. Query-based information flow (detect → track → forecast → plan) is the key architectural innovation.
3. Multi-stage training is essential: pretrain perception, then jointly train everything.
4. The planning module is relatively simple (GRU + regression) — the strength comes from upstream representations.
5. Track query propagation across frames enables both tracking and motion forecasting.
6. Collision penalty in the planning loss is critical — without it, ego trajectory ignores other agents.
7. The architecture is very large — training requires significant GPU resources (8× A100).
8. Each module can be evaluated independently, making debugging more tractable.
9. The BEVFormer backbone is the foundation — its quality limits all downstream tasks.
10. UniAD's design philosophy: every perception task should serve planning, not exist in isolation.

**5 gotchas:**

1. Multi-task loss balancing is extremely sensitive — wrong weights cause cascading failures
2. Track query ego-compensation must be exact — small errors accumulate across frames
3. Planning labels must be in current ego frame — not global frame
4. Motion forecasting min-ADE loss: must use winner-take-all correctly (only best mode penalized)
5. Sequential training data: scenes must be loaded in order; random shuffling breaks temporal tracking

**Tiny-subset overfit plan:**

| Parameter | Value |
|-----------|-------|
| Subset | 4 consecutive scenes |
| Simplify | Detection + planning only (skip mapping, occupancy) |
| Expected | Detection mAP > 0.7, planning L2 < 0.2m within 3000 steps |
| If fails | Train detection alone first; then add planning |

---

## 12) Minimal Reimplementation Checklist

**Build order (following 5 milestones):**

1. BEVFormer backbone (reuse existing implementation)
2. TrackFormer (DETR-style detection + temporal query propagation)
3. MapFormer (MapTR-style polyline prediction)
4. MotionFormer (trajectory forecasting with agent-agent and agent-map attention)
5. OccFormer (future occupancy prediction)
6. Planner (GRU-based ego trajectory generation with collision cost)
7. Multi-task joint training with balanced losses

**Sanity scripts:**

**Script 1:** Verify track query persistence: same object → same query across frames
**Script 2:** Overfit detection + planning on 4 scenes → L2 < 0.2m
**Script 3:** Visualize full pipeline: images → BEV → detections → tracks → forecast → plan

---
---
---

---

_Source: extracted and adapted from `BEV_Papers_Technical_Dossier.md` in this workspace._
