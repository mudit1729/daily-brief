# Reason2Drive: Towards Interpretable and Chain-Based Reasoning for Autonomous Driving

> **Authors:** Ming Nie, Renyuan Peng, Chunwei Wang, Xinyue Cai, Jianhua Han, Hang Xu, Li Zhang
> **Venue:** ECCV 2024 | **DOI:** [10.1007/978-3-031-73347-5_17](https://doi.org/10.1007/978-3-031-73347-5_17)
> **Citations:** ~142 | **Impact:** ★★★★ (large-scale reasoning chain benchmark)
> **PDF:** [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03786.pdf)

---

## 1. Problem & Motivation

Driving-oriented VLM research lacks datasets with **annotated reasoning chains** that explain decision-making. Existing benchmarks provide isolated QA pairs, not connected reasoning sequences. Standard language metrics (BLEU, CIDEr) are **ambiguous** for evaluating driving reasoning—a correct but differently-worded chain might score low while a fluent but incorrect one scores high.

## 2. Core Idea

Create a **large-scale video-text reasoning chain dataset** (>600K pairs) by automatically collecting QA pairs from multiple open-source driving datasets. Characterize driving as a sequence of:

```
Perception → Prediction → Reasoning → Decision
```

Introduce an **aggregated evaluation metric** designed to reduce ambiguity of standard language metrics in driving reasoning contexts.

## 3. Dataset Construction

```
Source Datasets:
├── nuScenes
├── Waymo Open Dataset
└── ONCE

     │ Automatic QA Collection
     ▼

Reason2Drive Dataset (>600K pairs)
├── Perception chains: "What objects are present?"
├── Prediction chains: "How will they move?"
├── Reasoning chains: "What does this imply for my driving?"
└── Decision chains: "What action should I take and why?"
```

**Aggregated evaluation metric:** Designed to handle the ambiguity of evaluating reasoning chains where multiple valid phrasings exist. Combines multiple scoring components to reduce noise from standard metrics.

## 4. Key Results

- >600K video-text pairs spanning multiple driving datasets
- Reveals limitations of BLEU/CIDEr for driving reasoning evaluation
- Aggregated metric correlates better with human judgment of reasoning quality
- Baseline VLM models show significant room for improvement on chain reasoning

## 5. Why Seminal

1. **Benchmark-first contribution** targeting the core weakness of VLM driving agents
2. **Cross-dataset:** Built from nuScenes + Waymo + ONCE (diverse scenarios)
3. **New evaluation metric** addressing fundamental problems with standard language metrics
4. **>600K pairs** — largest reasoning chain dataset for driving at time of release
5. **Part of broader movement** to stress-test chain-of-thought driving models

## 6. Limitations

- **Ambiguous VLA fit:** reasoning benchmark, not an action policy
- **Automatic QA generation** may encode biases from source datasets and scripts
- **Reasoning evaluation** remains fundamentally difficult even with improved metrics
- **Actions are indirect** — evaluates reasoning about actions, not action generation itself

## 7. The Evaluation Problem

| Metric | Problem for Driving Reasoning |
|--------|-------------------------------|
| BLEU | Penalizes valid paraphrases; rewards n-gram overlap regardless of correctness |
| CIDEr | Similar issues with consensus-based scoring |
| GPT-Score | Expensive, variable, may not capture driving-specific correctness |
| **Reason2Drive Aggregated** | Combines multiple aspects to reduce individual metric biases |

## 8. Connections

- **← DriveLM:** Both address structured driving reasoning, DriveLM via graphs, Reason2Drive via chains
- **← Textual Explanations:** Extends single explanations to full reasoning chains
- **→ ORION:** Benefits from reasoning chain evaluation methodology
- **→ SimLingo:** Raises the bar for what "reasoning" means in driving VLMs

## 9. Interview Quick-Hits

**Q: Why are BLEU/CIDEr scores problematic for driving reasoning?**
A: A correct reasoning chain phrased differently from the reference gets a low score, while a fluent but factually wrong chain can score high. Driving reasoning has many valid ways to describe the same correct analysis—standard metrics don't capture semantic correctness.

**Q: How does Reason2Drive differ from DriveLM?**
A: DriveLM structures reasoning as a dependency graph with explicit node connections. Reason2Drive focuses on sequential chains (perception → prediction → reasoning → decision) across a much larger dataset (>600K pairs from multiple datasets) and emphasizes evaluation methodology.

---

*VLA fit: Ambiguous (benchmark; actions indirect). Critical for evaluating whether driving VLMs actually reason correctly.*
