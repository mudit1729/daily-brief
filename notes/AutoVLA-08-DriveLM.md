# DriveLM: Driving with Graph Visual Question Answering

> **Authors:** Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jian Luo, Ping Luo, Andreas Geiger, Hongyang Li
> **Venue:** ECCV 2024 (Oral) | **arXiv:** [2312.14150](https://arxiv.org/abs/2312.14150)
> **Citations:** ~593 | **Impact:** ★★★★★ (GVQA framework + DriveLM-Data benchmark)
> **DOI:** [10.1007/978-3-031-72943-0_15](https://doi.org/10.1007/978-3-031-72943-0_15)

---

## 1. Problem & Motivation

Driving requires **multi-step reasoning** across objects and their interactions—single-round VQA is an insufficient proxy. When approaching an intersection, a driver reasons about: (1) what objects are present, (2) how they'll move, (3) what the traffic rules dictate, (4) what discrete behavior to take, (5) what specific trajectory to follow. These are logically **dependent** reasoning steps, not independent questions.

## 2. Core Idea

Introduce **Graph Visual Question Answering (GVQA)** where QA pairs are connected via logical dependencies, forming a reasoning graph. Build **DriveLM-Data** covering the full driving stack:

```
Perception QAs → Prediction QAs → Planning QAs → Behavior → Motion
     ↓                ↓                ↓             ↓         ↓
"What objects?"   "Where will    "What rules    "Turn left"  "(x1,y1)..."
                   they go?"      apply?"
```

## 3. Architecture & Method

```
Scene Observation (multi-view cameras)
              │
              ▼
     ┌────────────────────────────┐
     │    DriveLM Reasoning Graph │
     ├────────────────────────────┤
     │  Perception QAs            │ ← "What objects are in the scene?"
     │       ↓ (dependency)       │
     │  Prediction QAs            │ ← "Will the pedestrian cross?"
     │       ↓                    │
     │  Planning QAs              │ ← "Should I yield?"
     │       ↓                    │
     │  Behavior Token            │ ← Discrete: "Decelerate and wait"
     │       ↓                    │
     │  Motion Output             │ ← Continuous: trajectory waypoints
     └────────────────────────────┘
```

**DriveLM-Data:**
- Built on **nuScenes + CARLA**
- Graph-structured QA pairs with logical dependencies
- Covers perception → prediction → planning → behavior → motion
- Each node in the graph has explicit parent dependencies

**DriveLM-Agent:** Baseline VLM model that uses DriveLM-Data for both GVQA reasoning and end-to-end driving.

**Metrics:**
- Language QA metrics (including GPT-based scoring)
- Driving performance metrics (trajectory quality)
- Graph consistency metrics (reasoning coherence)

## 4. Key Results

- GVQA forces models to maintain reasoning coherence across steps
- DriveLM-Agent achieves competitive driving performance while being interpretable
- Graph structure exposes reasoning failures that flat QA would miss

## 5. Why Seminal

1. **GVQA framework:** First to formalize driving reasoning as a dependency graph, not isolated questions
2. **DriveLM-Data:** Reusable benchmark covering the **full stack** (perception → motion)
3. **Very high citations** (~593) for an ECCV paper — widely adopted
4. **Reasoning graph** framing increasingly used as bridge between VLM reasoning and action generation
5. **Both nuScenes + CARLA** data — cross-domain coverage

## 6. Limitations

- QA generation is partially **synthetic** (GPT-based scoring can be noisy)
- Graph structure may not match how real users interact with driving systems
- CARLA component limits external validity
- Reasoning correctness ≠ driving safety guarantee

## 7. Graph VQA vs Flat VQA

| Aspect | Flat VQA | Graph VQA (DriveLM) |
|--------|----------|---------------------|
| Structure | Independent questions | Dependency graph |
| Reasoning | Single-step | Multi-step chains |
| Consistency | Not enforced | Graph dependencies enforced |
| Coverage | Partial | Full stack (P→Pr→Pl→B→M) |
| Failure detection | Hard | Graph reveals broken chains |

## 8. The Full Stack

```
P(erception)  → "3 vehicles, 1 pedestrian, traffic light green"
    ↓
Pr(ediction)  → "Pedestrian likely to cross in 2s"
    ↓
Pl(anning)    → "Yield to pedestrian, then proceed"
    ↓
B(ehavior)    → "Decelerate" (discrete action token)
    ↓
M(otion)      → [(x1,y1,t1), (x2,y2,t2), ...] (trajectory)
```

## 9. Connections

- **← Textual Explanations:** Evolves from free-form to structured reasoning
- **← GPT-Driver:** Extends planning-as-language to full reasoning chains
- **→ Reason2Drive:** Addresses evaluation metrics for reasoning chains
- **→ SimLingo:** Tests whether graph reasoning is aligned with actions
- **→ ORION:** Uses reasoning tokens to bridge LLM space and action space

## 10. Interview Quick-Hits

**Q: What is Graph VQA (GVQA)?**
A: A reasoning framework where QA pairs about a driving scene are connected by logical dependencies forming a DAG. Unlike flat VQA (independent questions), GVQA captures that prediction depends on perception, planning depends on prediction, etc.

**Q: What does DriveLM-Data cover?**
A: The full driving reasoning stack: perception (what's there), prediction (what will happen), planning (what should I do), behavior (discrete intent), and motion (continuous trajectory). Built on nuScenes + CARLA.

**Q: Why is graph structure important for driving VLMs?**
A: It exposes reasoning failures that flat QA would miss. If a model correctly identifies a pedestrian (perception) but incorrectly predicts they'll stay still (prediction), the graph reveals the broken chain rather than scoring each QA independently.

---

*VLA fit: Full VLA framework (vision + language reasoning + action). Landmark structured reasoning benchmark for driving.*
