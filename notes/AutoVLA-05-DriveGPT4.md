# DriveGPT4: Interpretable End-to-End Autonomous Driving via Large Language Model

> **Authors:** Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kenneth K.Y. Wong, Zhenguo Li, Hengshuang Zhao
> **Venue:** IEEE Robotics and Automation Letters (2024) | **DOI:** [10.1109/LRA.2024.3440097](https://doi.org/10.1109/LRA.2024.3440097)
> **Citations:** ~723 | **Impact:** ★★★★★ (widely-copied VLM driving recipe)
> **PDF:** [HKU](https://i.cs.hku.hk/~kykwong/publications/zxu_ral2024.pdf)

---

## 1. Problem & Motivation

End-to-end driving models lack **interpretability and interaction**. A user can't ask "why did you brake?" and get a meaningful answer. The model can't describe what it sees or justify its actions. How can we add language-based interpretability while maintaining control prediction quality?

## 2. Core Idea

Apply **multimodal instruction tuning** (LLaVA-style) to driving. Build a driving-specific visual instruction dataset from BDD-X using ChatGPT to generate diverse QA pairs. Train a VLM that simultaneously:
1. Predicts low-level control signals (speed, turning angle)
2. Describes and justifies driving actions in natural language

## 3. Architecture & Method

```
Front-view Video Frames
        │
        ▼
   Vision Encoder (pretrained)
        │
        ▼
   Visual Tokens
        │
   ┌────┴────┐
   │  LLM    │ ← instruction-tuned on driving QA
   │ (frozen │
   │  + LoRA)│
   └────┬────┘
        │
        ├──► Control Predictions
        │    (speed RMSE, turning angle)
        │
        └──► Language Outputs
             - Action description
             - Justification / reasoning
             - Scene narration
```

**Training data construction:**
- Start with BDD-X (video + action descriptions + justifications)
- Use ChatGPT to expand into diverse instruction-following QA pairs
- Include control prediction as a language task

**Evaluation:**
- Open-loop control: speed RMSE, turning angle RMSE, threshold accuracies
- Language tasks: description quality, justification quality, QA accuracy

## 4. Key Results

- Competitive open-loop control prediction on BDD-X
- Generates fluent, contextually appropriate driving explanations
- Multi-task training (control + language) doesn't degrade either capability

## 5. Why Seminal

1. **Highest-cited driving VLM** in the listed corpus (~723 citations)
2. **Widely-copied recipe:** driving-domain visual instruction tuning + joint language+control
3. **Made BDD-X the de facto benchmark** for language+control models
4. **Accelerated the "VLM agent" framing** for driving
5. **Motivated closed-loop extensions** (DriveGPT4-V2 explicitly contrasts with V1's open-loop focus)

## 6. Limitations

- **Open-loop only** — no closed-loop validation of safety under compounding error
- **Language reasoning can be fluent but non-causal** — model may generate plausible but incorrect explanations
- **Synthetic instruction tuning** via ChatGPT can introduce biases
- **No instruction following** — language is output, not input for controlling behavior

## 7. Comparison: DriveGPT4 vs DriveGPT4-V2

| Aspect | DriveGPT4 (V1) | DriveGPT4-V2 |
|--------|----------------|--------------|
| Evaluation | Open-loop (BDD-X) | Closed-loop (CARLA) |
| Control | RMSE on logged data | Interactive driving |
| Error handling | No compounding errors | Handles compounding |
| Temporal | Frame-by-frame | Sequential reasoning |

## 8. Connections

- **← BDD-X / Textual Explanations:** Builds on the BDD-X dataset and explanation paradigm
- **← LLaVA:** Adapts multimodal instruction tuning recipe for driving
- **→ SimLingo:** Extends to closed-loop + language-action alignment
- **→ DriveLM:** Structures reasoning into graph dependencies vs free-form

## 9. Interview Quick-Hits

**Q: How does DriveGPT4 combine language and control?**
A: It treats both as outputs of a multimodal LLM—control signals are predicted alongside language descriptions/justifications using instruction tuning on BDD-X data augmented with ChatGPT.

**Q: Why is open-loop evaluation insufficient for driving VLMs?**
A: Open-loop scores (RMSE on logged data) don't capture compounding errors or interactive behavior. A model might score well frame-by-frame but fail catastrophically in continuous driving because small errors accumulate. This motivated DriveGPT4-V2 and LMDrive to move to closed-loop CARLA evaluation.

---

*VLA fit: Full VLA (vision input → language reasoning → control output). Most influential template for driving VLMs.*
