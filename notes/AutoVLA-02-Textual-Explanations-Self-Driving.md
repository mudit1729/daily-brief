# Textual Explanations for Self-Driving Vehicles

> **Authors:** Jinkyu Kim, Anna Rohrbach, Trevor Darrell, John Canny, Zeynep Akata
> **Venue:** ECCV 2018 | **DOI:** [10.1007/978-3-030-01216-8_35](https://doi.org/10.1007/978-3-030-01216-8_35)
> **Citations:** ~555 | **Impact:** ★★★★★ (created BDD-X benchmark)
> **PDF:** [CVF Open Access](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jinkyu_Kim_Textual_Explanations_for_ECCV_2018_paper.pdf)

---

## 1. Problem & Motivation

End-to-end driving models are **opaque**: they produce control signals without any rationale. Users cannot trust a system that can't explain *why* it's braking or turning. Explanations improve interpretability and acceptance, but generating them requires grounding language in the visual regions that actually influenced the driving decision.

## 2. Core Idea

Train two coupled models:
1. **Visual attention-based controller** (images → control) with spatial attention maps
2. **Video-to-text explanation model** that generates natural-language justifications

Then **align their attentions** so that explanations are grounded in the same visual regions that influenced control output. This ensures explanations are not just plausible rationalizations but are connected to the actual decision process.

## 3. Architecture & Method

```
Video Frames
    │
    ├──► Controller (CNN + Attention)
    │        │
    │        ├─► Control Signals (steer, accel)
    │        └─► Controller Attention Map
    │
    └──► Explanation Generator (Encoder-Decoder + Attention)
             │
             ├─► "Slowing down because pedestrian crossing"
             └─► Explanation Attention Map
                      │
                      ▼
              Attention Alignment Loss
              (controller ↔ explanation)
```

**Key distinction:** *Introspective explanations* (causally related to decision) vs *rationalizations* (post-hoc plausible but not necessarily causal).

**BDD-X Dataset:** Built on BDD100K with human-written descriptions and explanations of driving actions. Each video segment paired with action description + justification.

## 4. Key Results

- Attention alignment improves explanation quality (human evaluation)
- Explanations reference decision-relevant visual regions (not arbitrary scene elements)
- BDD-X provides a reusable benchmark for explainable driving

## 5. Why Seminal

1. **Created BDD-X** — the standard testbed for explainable driving, reused by DriveGPT4 and many subsequent works
2. **Established attention alignment** as the canonical mechanism for grounding language output to action-relevant evidence
3. **Distinguished introspective vs rationalization** — a philosophical frame still debated in 2025 VLA work
4. **First language-output driving system** with explicit grounding mechanism

## 6. Limitations

- **Faithfulness problem:** aligned attention ≠ proof of causal correctness; language can be plausible yet not truly decision-causing
- **Offline evaluation only:** no closed-loop safety validation
- **Language quality** limited by template-like descriptions in BDD-X
- **No instruction following:** language is output only, not input

## 7. Key Concepts

| Concept | Description |
|---------|-------------|
| Introspective explanation | Text grounded in the actual features used for the driving decision |
| Rationalization | Post-hoc plausible explanation not necessarily tied to the causal decision |
| Attention alignment | Loss that encourages explanation attention to match controller attention |
| BDD-X | Dataset with paired (video, action description, justification) triples |

## 8. Connections to Later Work

- **→ DriveGPT4:** Directly trains on BDD-X using LLM instruction tuning for explanations + control
- **→ SimLingo:** Extends alignment idea to Action Dreaming (language-action consistency)
- **→ DriveLM:** Structures reasoning as graphs rather than free-form explanations
- **→ Reason2Drive:** Creates reasoning *chains* rather than single-shot explanations

## 9. Interview Quick-Hits

**Q: Why is explaining driving decisions hard?**
A: The core tension is faithfulness—a model can generate fluent, plausible explanations that don't reflect what actually drove the decision. Attention alignment helps but doesn't guarantee causal faithfulness.

**Q: What is the BDD-X dataset?**
A: Video segments from BDD100K paired with human-written (1) action descriptions ("car is slowing down") and (2) justifications ("because a pedestrian is crossing"). ~7K video clips with language annotations.

**Q: How does attention alignment work?**
A: The controller's spatial attention map (which visual regions influence steering/acceleration) is regularized to match the explanation generator's attention map. This encourages the explanation to reference what the controller actually "looked at."

---

*VLA fit: Ambiguous (language is output/explanation, not input command). Foundational for language-grounded driving interpretability.*
