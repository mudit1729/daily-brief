# VLP: Vision Language Planning for Autonomous Driving

> **Authors:** Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G. Allievi, Senem Velipasalar, Liu Ren
> **Venue:** CVPR 2024 | **DOI:** [10.1109/CVPR52733.2024.01398](https://doi.org/10.1109/CVPR52733.2024.01398)
> **Citations:** ~176 | **Impact:** ★★★★ (LM priors for BEV planning)
> **PDF:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Pan_VLP_Vision_Language_Planning_for_Autonomous_Driving_CVPR_2024_paper.pdf)

---

## 1. Problem & Motivation

Vision-only driving models struggle with **reasoning, generalization, and long-tail scenarios**. They lack the "common sense" knowledge that humans use when driving. Language models encode broad world knowledge—can we inject this into the planning process without requiring explicit language instructions?

## 2. Core Idea

Integrate **language model representations** into a BEV-based autonomous driving system. Use LM features as a semantic prior to enhance planning, without requiring language input at inference time. Two components:

1. **ALP (Aligned Language Planning):** Align local BEV features with pretrained LM feature space for richer semantic representation
2. **SLP (Structured Language Planning):** Use LM comprehension to align planning queries with goals and ego status

## 3. Architecture & Method

```
Multi-view Camera Images
         │
         ▼
   BEV Feature Extraction
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
  ALP Module                   SLP Module
  (Align BEV features          (Align planning queries
   with LM feature space)       with LM goal/status
   ┌──────────┐                  comprehension)
   │ LM Prior │                 ┌──────────┐
   │ Features │                 │ LM Goal  │
   └──────────┘                 │ Parsing  │
    │                           └──────────┘
    │                             │
    └────────────┬────────────────┘
                 ▼
         Planning Head
                 │
                 ▼
         Trajectory Output
```

**Key insight:** Language is used as an **internal prior** (feature space), not as explicit user instructions. This means the LM knowledge improves planning without requiring language input at test time.

## 4. Key Results

- Improvements across multiple driving tasks on nuScenes
- Enhanced performance in long-tail / rare scenarios
- LM features help with semantic understanding of complex scenes

## 5. Why Seminal

1. **Early exemplar of "LM distillation into BEV planning"** — a distinct approach from instruction-following VLA
2. **Language as internal knowledge** rather than external interface
3. **Practical:** no language input needed at inference → compatible with existing ADS pipelines
4. **Strong CVPR citation traction** (~176 citations)

## 6. Limitations

- **Ambiguous VLA fit:** language functions as internal prior, not explicit instruction-following
- **Open-loop evaluation** primarily
- **Limited interpretability:** LM features improve performance but don't generate human-readable reasoning
- **Dependency on LM feature quality** and alignment

## 7. Two Approaches to Language in Driving

| Approach | Example | Language Role | At Inference? |
|----------|---------|---------------|---------------|
| Language as interface | LMDrive, SimLingo | User gives instructions | Yes, explicit |
| Language as prior | VLP | LM features enhance planning | No, implicit |

## 8. Interview Quick-Hits

**Q: How does VLP differ from instruction-following VLA like LMDrive?**
A: VLP uses language model representations as an internal feature prior to enhance BEV planning—no language input at inference. LMDrive takes explicit language instructions and generates control signals. VLP is "LM knowledge distillation" vs "LM interaction."

**Q: What are ALP and SLP in VLP?**
A: ALP aligns local BEV features with pretrained LM feature space for richer semantics. SLP uses LM comprehension to align planning queries with goals/ego status. Together they inject world knowledge into the planner.

---

*VLA fit: Ambiguous (LM features; not language-in-the-loop). Important conceptual bridge between pure vision planning and full VLA.*
