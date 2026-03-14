# EMMA: End-to-End Multimodal Model for Autonomous Driving

> **Authors:** Jyh-Jing Hwang et al. (Waymo Research)
> **Venue:** TMLR 2025 | **OpenReview:** [link](https://openreview.net/forum?id=kH3t5lmOU8)
> **Citations:** ~212 | **Impact:** ★★★★★ (industry-scale generalist driving model)
> **Project:** [waymo.com/research/emma](https://waymo.com/research/emma/)

---

## 1. Problem & Motivation

Current autonomous driving systems are **siloed**: separate modules for perception, prediction, planning, and mapping. Each module optimizes independently, leading to error accumulation and limited cross-task knowledge sharing. Can we build a **single multimodal model** that handles everything—leveraging the world knowledge of large foundation models?

## 2. Core Idea

**"Everything as language tokens."** EMMA unifies multiple driving tasks in a single prompt-driven multimodal model:
- **Inputs:** Camera images + text (navigation instructions, ego status)
- **Outputs:** Planning trajectories, perception objects, road graph elements—all represented as natural language

One model, multiple driving tasks, unified language interface.

## 3. Architecture & Method

```
Raw Camera Sequences
         │
         ▼
   Multimodal Foundation Model
   (large-scale pretrained)
         │
         ├──► Task 1: Planning
         │    Prompt: "Plan 3s trajectory given..."
         │    Output: "(x1,y1), (x2,y2), ..." as text
         │
         ├──► Task 2: Perception
         │    Prompt: "Detect objects in scene..."
         │    Output: "Car at (x,y,z), Pedestrian at..."
         │
         ├──► Task 3: Road Graph
         │    Prompt: "Describe road structure..."
         │    Output: "Lane 1: [(x1,y1)→(x2,y2)], ..."
         │
         └──► Non-sensor inputs as text
              Navigation: "Turn right in 200m"
              Ego status: "Speed: 30km/h, heading: 45°"
```

**Key insight:** By representing everything as language, the model can leverage cross-task knowledge and foundation model world priors. A single architecture handles what traditionally required 3-5 separate modules.

## 4. Key Results

| Benchmark | Performance |
|-----------|------------|
| nuScenes Planning | Strong trajectory quality |
| Waymo Benchmarks | Competitive perception + planning |
| Multi-task | Joint planning + perception + road graph |

## 5. Why Seminal

1. **Industry-scale demonstration** of "everything as language" for driving
2. **Generalist model:** one model for planning + perception + mapping
3. **Waymo credibility:** research prototype from the leading AV company
4. **Unified prompt interface:** tasks specified via text prompts, not architecture changes
5. **Highly cited** (~212) — pushing open-source replications

## 6. Limitations (Acknowledged by Waymo)

| Limitation | Impact |
|-----------|--------|
| **Small number of image frames** | Cannot process long temporal sequences |
| **No LiDAR/radar** | Missing depth and velocity sensor redundancy |
| **Computationally expensive** | May not meet real-time constraints |
| **Camera-only** | Safety-critical applications need sensor diversity |
| **Research prototype** | Not deployed on production vehicles |

## 7. "Everything as Language" Paradigm

```
Traditional ADS:
  Camera → [Perception Module] → Objects
  Objects → [Prediction Module] → Futures
  Futures → [Planning Module] → Trajectory
  Each module: separate architecture, separate training

EMMA:
  Camera → [Unified Foundation Model] → Everything
  Planning:   "Prompt: plan..." → trajectory as text
  Perception: "Prompt: detect..." → objects as text
  Road Graph: "Prompt: map..." → road structure as text
  Single model, single training, shared knowledge
```

## 8. Connections

- **← GPT-Driver:** Scaled up planning-as-language to full multi-task
- **← DriveGPT4:** From single-task VLM to generalist
- **↔ ORION:** Different approach—EMMA keeps everything as language, ORION separates reasoning from action generation
- **→ Future:** Motivates open-source generalist driving models

## 9. EMMA vs Other Approaches

| Approach | Modality | Tasks | Language Role |
|----------|----------|-------|---------------|
| DriveGPT4 | Camera | Control + explanation | Output |
| LMDrive | Camera + LiDAR | Instruction-following | Input |
| DriveLM | Camera | VQA + driving | Reasoning |
| ORION | Camera | VQA + planning | Reasoning + planning token |
| **EMMA** | Camera | **All tasks** | **Everything** |

## 10. Interview Quick-Hits

**Q: What makes EMMA different from other driving VLMs?**
A: EMMA is a true generalist—one model handles planning, perception, and road graph understanding. Other works focus on one or two tasks. EMMA represents everything (inputs, outputs, tasks) as language, enabling cross-task knowledge sharing.

**Q: What are EMMA's key limitations?**
A: Acknowledged by Waymo: limited frame processing, no LiDAR/radar, and high computational cost. These are exactly the factors that determine real-world deployability.

**Q: How does EMMA relate to the "language-as-internal-representation" vs "language-as-interface" distinction?**
A: EMMA uses language as both—navigation instructions and ego status are language inputs (interface), while all outputs are language representations (internal). This is the most extreme version of "everything as language."

---

*VLA fit: Ambiguous (language as internal representation/prompting). Industry-scale demonstration of unified language-space driving.*
