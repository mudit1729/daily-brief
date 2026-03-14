# ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation

> **Authors:** Haoyu Fu, Dingkang Yang, Zhi Qu, Peng Zhai, Lihua Zhang
> **Venue:** ICCV 2025 | **arXiv:** [2503.19755](https://arxiv.org/abs/2503.19755)
> **Citations:** ~74 | **Impact:** ★★★★★ (reasoning-action gap solution)
> **PDF:** [arXiv](https://arxiv.org/pdf/2503.19755)

---

## 1. Problem & Motivation

VLMs excel at **semantic reasoning** but driving requires **numerical trajectories**. Directly generating numeric outputs via autoregressive text models can underperform in complex scenes because:
- Token-level generation isn't suited for precise coordinate sequences
- LLM reasoning space ≠ physical action space
- The gap between "I should turn left because..." and "(x₁,y₁), (x₂,y₂), ..." is significant

This **"reasoning space vs action space" gap** is the core problem ORION addresses.

## 2. Core Idea

Bridge semantic reasoning and physical action generation with a **three-component architecture:**
1. **QT-Former:** Aggregates long-term visual context and connects vision tokens to LLM space
2. **LLM Reasoning:** Performs driving scenario reasoning and produces a special **planning token**
3. **Generative Planner:** Converts the planning token into multimodal trajectories

The planning token is the key: it's a learned embedding that captures the LLM's reasoning decision in a format the generative planner can use.

## 3. Architecture & Method

```
Multi-view Camera Sequence
         │
         ▼
  ┌──────────────┐
  │  QT-Former   │ ← Temporal aggregation + visual tokenization
  │  (Query-     │
  │   Temporal)  │
  └──────┬───────┘
         │ Visual tokens
         ▼
  ┌──────────────┐     VQA Queries
  │    LLM       │◄──── "What is the safe action?"
  │  Reasoning   │
  │    Core      │──► VQA Answers (interpretability)
  │              │
  │              │──► Planning Token [P]
  └──────┬───────┘        │
         │                │
         ▼                ▼
  ┌──────────────────────────┐
  │   Generative Planner     │
  │   (diffusion/GMM-based)  │
  │                          │
  │   Condition: [P] token   │
  │   Output: Multimodal     │
  │           trajectories   │
  └──────────────────────────┘
         │
         ▼
  Trajectory: [(x₁,y₁), (x₂,y₂), ...]
```

## 4. Key Design: The Planning Token

```
Why a planning token?
─────────────────────
LLM outputs text tokens → poor for precise coordinates
LLM outputs numeric text → imprecise, slow, error-prone

ORION solution:
LLM outputs → [P] embedding ← learned representation
                  │               of driving decision
                  ▼
         Generative Planner ← optimized for
                  │               trajectory quality
                  ▼
         Precise trajectories ← multimodal, feasible
```

The planning token captures the *semantic decision* ("turn left, yield to pedestrian") as a dense embedding, while the generative planner handles the *geometric realization* as precise waypoints.

## 5. Key Results

| Metric | ORION | Context |
|--------|-------|---------|
| Bench2Drive Driving Score | **77.74** | Large gains over prior methods |
| Bench2Drive Success Rate | **54.62%** | State-of-the-art |
| VQA Performance | Competitive | Reasoning capability preserved |

## 6. Why Seminal

1. **Explicit statement of the key VLA challenge:** converting semantics into physically grounded trajectories
2. **Planning token:** elegant bridge between reasoning space and action space
3. **Generative planner:** dedicated architecture for trajectory feasibility and multimodality
4. **Strong Bench2Drive results** — validates the approach in closed-loop evaluation
5. **Clean separation:** LLM reasons, planner generates — each does what it's best at

## 7. Limitations

- **Complexity and training cost** — three-component architecture with multiple training stages
- **Simulator benchmarks only** — Bench2Drive/CARLA, not real-world deployment
- **LLM reasoning robustness** under distribution shift ≠ safety guarantee
- **Planning token interpretability** — what the embedding encodes isn't directly inspectable

## 8. Reasoning Space vs Action Space

```
The Core Problem:
┌────────────────────┐      ┌────────────────────┐
│  Reasoning Space   │  GAP │   Action Space      │
│  (LLM tokens)      │ ──── │   (trajectories)    │
│                    │      │                    │
│  "Yield to         │  →?  │  [(3.2, 0.1, 1.0), │
│   pedestrian"      │      │   (3.5, -0.2, 2.0),│
│                    │      │   ...]              │
└────────────────────┘      └────────────────────┘

ORION's Solution:
┌────────────────────┐ [P]  ┌────────────────────┐
│  LLM Reasoning     │──────│  Generative Planner │
│                    │      │                    │
│  Semantic decision │ emit │  Trajectory         │
│  encoded in [P]    │ ────►│  conditioned on [P] │
│                    │      │                    │
└────────────────────┘      └────────────────────┘
```

## 9. Connections

- **← GPT-Driver:** ORION addresses GPT-Driver's precision problem
- **← DriveLM:** Uses similar multi-step reasoning but adds explicit action generation
- **← LMDrive:** Shares closed-loop evaluation but adds generative planning head
- **↔ SimLingo:** Complementary alignment approaches
- **→ Future:** Reinforces pattern of keeping LLM reasoning + dedicated planning head

## 10. Interview Quick-Hits

**Q: What is the "reasoning-action gap" in driving VLAs?**
A: VLMs reason in semantic/token space ("yield to pedestrian") but driving requires precise numeric trajectories. Directly generating coordinates as text tokens is imprecise. ORION bridges this gap with a planning token that the generative planner converts to feasible trajectories.

**Q: What is the QT-Former?**
A: Query-Temporal Former — it aggregates long-term visual context from multi-view camera sequences and converts visual features into tokens the LLM can process. It handles the temporal dimension that single-frame processing misses.

**Q: Why use a generative planner instead of direct LLM trajectory output?**
A: LLMs excel at semantic reasoning but struggle with precise coordinate generation. A dedicated generative planner (diffusion/GMM-based) is optimized for trajectory quality, multimodality (multiple valid paths), and physical feasibility. The planning token bridges both worlds.

---

*VLA fit: Full VLA with explicit reasoning-action alignment. Key contribution to the action generation problem in driving VLMs.*
