# DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving

> **Authors:** Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, Junchi Yan
> **Venue:** arXiv 2025 | **arXiv:** [2505.16278](https://arxiv.org/abs/2505.16278)
> **Impact:** ★★★★ (MoE architecture for driving VLA, Bench2Drive SOTA)

---

## 1. Problem & Motivation

Current driving VLA models use **monolithic architectures** where the same parameters handle all driving scenarios. But driving encompasses vastly different skills:
- Highway cruising (steady speed, lane keeping)
- Intersection navigation (yielding, signal recognition)
- Parking (low-speed maneuvering)
- Emergency braking (rapid response)

A single set of parameters must average across all these modes, leading to **mode averaging** — the model compromises on each skill instead of excelling at any.

## 2. Core Idea

Apply **Mixture-of-Experts (MoE)** to driving VLA at two levels:
1. **Scene-Specialized Vision MoE:** Dynamically selects which cameras matter for the current context (don't process rear cameras when driving straight on highway)
2. **Skill-Specialized Action MoE:** Activates different expert modules for different driving behaviors (lane change expert vs intersection expert)

Built on top of the **Drive-π₀** VLA baseline.

## 3. Architecture & Method

```
Multi-view Camera Images (6 cameras)
         │
         ▼
┌────────────────────────────────┐
│  Scene-Specialized Vision MoE  │
│                                │
│  Router: "What cameras matter  │
│  for this driving context?"    │
│                                │
│  Highway: front + side cameras │
│  Parking: all cameras          │
│  Intersection: front + sides   │
│                                │
│  → Selective camera processing │
└──────────┬─────────────────────┘
           │ Context-aware visual features
           ▼
┌────────────────────────────────┐
│  VLM Backbone (Drive-π₀)      │
│  Language understanding +      │
│  scene reasoning               │
└──────────┬─────────────────────┘
           │
           ▼
┌────────────────────────────────┐
│  Skill-Specialized Action MoE  │
│                                │
│  Router: "What driving skill   │
│  is needed here?"              │
│                                │
│  ┌────────┐ ┌────────┐        │
│  │Expert 1│ │Expert 2│ ...    │
│  │Highway │ │Intersec│        │
│  │Cruising│ │tion Nav│        │
│  └────────┘ └────────┘        │
│                                │
│  → Skill-specific trajectory   │
└──────────┬─────────────────────┘
           │
           ▼
    Trajectory Output
```

**Two MoE routers:**

| Router | Input | Decision | Benefit |
|--------|-------|----------|---------|
| Vision MoE | Driving context | Which cameras to process | Computational efficiency |
| Action MoE | Scene understanding | Which expert to activate | Skill specialization |

## 4. Key Results

| Benchmark | Performance |
|-----------|------------|
| **Bench2Drive** (closed-loop) | **State-of-the-art** |
| Mode averaging | Eliminated by expert specialization |
| Computational efficiency | Improved via selective camera processing |

## 5. Why Significant

1. **First MoE for driving VLA:** Introduces the MoE scaling paradigm to autonomous driving
2. **Solves mode averaging:** Each expert specializes in a driving skill instead of compromising
3. **Dual MoE (vision + action):** Novel two-level expert routing
4. **Efficient:** Vision MoE reduces computation by skipping irrelevant cameras
5. **Bench2Drive SOTA:** Validated on the standard closed-loop benchmark

## 6. The Mode Averaging Problem

```
Monolithic Model:
  Training sees:  70% highway, 20% intersections, 10% parking
  Result: Model excels at highway, mediocre at intersections, poor at parking
  Why: Same parameters must handle all modes → weighted average behavior

MoE Model (DriveMoE):
  Highway Expert:      Trained on highway → excels at highway
  Intersection Expert: Trained on intersections → excels at intersections
  Parking Expert:      Trained on parking → excels at parking
  Router: Selects correct expert based on context
  Result: Expert-level performance across all modes
```

## 7. MoE in Modern AI Context

| System | MoE Application | Domain |
|--------|----------------|--------|
| Mixtral | Language token experts | Text generation |
| Switch Transformer | Sparse expert routing | NLP |
| **DriveMoE** | Vision + Action experts | **Autonomous driving** |
| V-MoE | Vision token experts | Image recognition |

**DriveMoE is unique** in applying MoE at both the vision (camera selection) and action (skill selection) levels.

## 8. Connections

- **← Drive-π₀:** Base VLA model that DriveMoE extends with MoE
- **← Mixtral/Switch:** MoE architectural pattern from NLP
- **↔ ORION:** Both achieve Bench2Drive SOTA, different approaches (reasoning vs experts)
- **↔ SimLingo:** Both closed-loop; DriveMoE uses skill specialization vs alignment
- **→ Future:** MoE may become the standard scaling approach for driving VLAs

## 9. Interview Quick-Hits

**Q: What is the mode averaging problem in driving VLA?**
A: When a single model handles all driving scenarios, it learns a weighted average behavior that compromises on each skill. Highway driving gets good because it's most common, but rare scenarios (intersections, parking) get poor performance. MoE solves this by having separate experts for each driving mode.

**Q: How does DriveMoE's Vision MoE work?**
A: A learned router examines the driving context and decides which cameras are relevant. On a highway, it processes front and side cameras but skips rear cameras. At an intersection, it processes all available views. This saves computation and focuses attention where it matters.

**Q: Why two levels of MoE?**
A: Vision MoE handles *what to look at* (camera selection based on scene type), while Action MoE handles *how to drive* (skill selection based on maneuver type). The combination addresses both the perception and planning dimensions of the mode averaging problem.

---

*VLA fit: Full VLA with MoE scaling. Key contribution to efficient, specialized driving VLA architectures.*
