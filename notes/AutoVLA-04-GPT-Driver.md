# GPT-Driver: Learning to Drive with GPT

> **Authors:** Jiageng Mao, Yuxi Qian, Hang Zhao, Yue Wang
> **Venue:** NeurIPS 2023 FMDM Workshop | **arXiv:** [2310.01415](https://arxiv.org/abs/2310.01415)
> **Citations:** ~456 | **Impact:** ★★★★★ (LLM-as-planner paradigm)
> **OpenReview:** [link](https://openreview.net/forum?id=Pvjk9lxLJK)

---

## 1. Problem & Motivation

Traditional motion planners rely on hand-crafted cost functions or specialized neural architectures with limited generalization. **Long-tail driving scenarios** (rare events, novel interactions) are hard to handle because planners lack the "common sense" and reasoning that large language models provide. Can we reframe planning itself as **language modeling**?

## 2. Core Idea

**Reformulate motion planning as a language modeling problem.** Represent planner inputs (scene context, ego state) and outputs (waypoint trajectories) as language tokens. Use an LLM (GPT-3.5) to generate trajectory waypoints via prompting + fine-tuning, paired with natural-language reasoning traces.

## 3. Architecture & Method

```
Scene Context (tokenized)          Ego State (tokenized)
    │                                  │
    └──────────┬───────────────────────┘
               ▼
        Prompt Construction
        "Given scene: ... ego at (x,y,θ)
         Plan the next 3s trajectory"
               │
               ▼
        ┌─────────────┐
        │   LLM (GPT) │ ← fine-tuned for driving
        └─────────────┘
               │
               ├─► Reasoning trace: "The car ahead is
               │   braking, I should decelerate..."
               │
               └─► Waypoints: "(x1,y1), (x2,y2), ..."
```

**Key design choices:**
- **Tokenization:** Convert numeric coordinates to text tokens
- **Prompting:** Include scene description, traffic rules, ego state
- **Fine-tuning:** Adapt LLM on driving trajectory data (nuScenes)
- **Reasoning:** Generate natural-language chain-of-thought before trajectory

## 4. Key Results

- Evaluated on nuScenes planning benchmark (open-loop)
- Competitive trajectory prediction performance
- LLM reasoning traces provide interpretable planning rationale
- Demonstrates feasibility of planning-as-language paradigm

## 5. Why Seminal

1. **First heavily-cited "LLM-as-planner" driving paper** — catalyzed an entire research direction
2. **Planning-as-language:** showed trajectories can be generated as token sequences
3. **Reasoning traces:** LLM provides chain-of-thought for planning decisions
4. **Inspired EMMA, ORION, and many follow-ups** that treat planning as token generation

## 6. Limitations

- **Ambiguous VLA fit:** "vision" may be abstracted into structured features, not raw images
- **Open-loop evaluation only** (nuScenes) — no closed-loop validation
- **Numeric precision:** LLMs can struggle with precise coordinate generation
- **Latency:** LLM inference may not meet real-time planning requirements
- **Reasoning causality:** reasoning traces may not be causally tied to trajectory quality

## 7. Key Insight: Planning as Language

| Traditional Planner | GPT-Driver |
|---------------------|------------|
| Cost function optimization | Token generation |
| Hand-crafted rules | Learned from data + LLM priors |
| Domain-specific architecture | General-purpose LLM |
| Limited reasoning | Chain-of-thought reasoning |
| Hard to generalize | World knowledge from pretraining |

## 8. Connections

- **← CIL:** Replaces discrete commands with full language reasoning
- **→ EMMA:** Scales "everything as language" to industry-level driving
- **→ ORION:** Addresses the numeric precision gap with a dedicated generative planner
- **→ DriveLM:** Extends reasoning to graph-structured QA chains

## 9. Interview Quick-Hits

**Q: What's the core idea of GPT-Driver?**
A: Reformulate motion planning as a language modeling problem—represent scene context and trajectories as text tokens, and use an LLM to generate waypoints with chain-of-thought reasoning.

**Q: What's the main limitation of LLM planners?**
A: Numeric precision—LLMs work in token space which isn't ideal for precise coordinate generation. ORION later addresses this with a dedicated generative planner head.

**Q: Why is planning-as-language important for VLA?**
A: It unifies perception, reasoning, and action in a single token-generation framework, enabling natural language to mediate between visual understanding and driving decisions.

---

*VLA fit: Ambiguous (LLM planner; vision may be abstracted). Catalyzed the LLM-for-driving paradigm.*
