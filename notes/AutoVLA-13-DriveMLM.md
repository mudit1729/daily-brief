# DriveMLM: Aligning Multi-Modal LLMs with Behavioral Planning States

> **Authors:** Wenhai Wang et al.
> **Venue:** arXiv 2023 | **arXiv:** [2312.09245](https://arxiv.org/abs/2312.09245)
> **Citations:** ~237 | **Impact:** ★★★★ (LLM module in existing AD stack)

---

## 1. Problem & Motivation

Most driving VLMs are **end-to-end replacements** for the entire driving stack. But existing modular AD systems (Apollo, Autoware) have mature perception and control. Can we instead plug an **LLM module** into existing stacks for better behavioral planning—without replacing everything?

## 2. Core Idea

Standardize **behavioral planning states** using an off-the-shelf motion planning module. Train a multimodal LLM to:
1. Model behavioral planning from sensor inputs + user commands
2. Output driving decisions + natural-language explanations
3. Integrate as a **plug-and-play module** into existing AD stacks

## 3. Architecture

```
Sensor Inputs (camera/LiDAR)    User Commands
          │                          │
          ▼                          ▼
   ┌──────────────────────────────────────┐
   │        Multimodal LLM               │
   │   (behavioral planning module)       │
   │                                      │
   │   Input: visual features + commands  │
   │   Output: behavioral state +         │
   │           explanation                │
   └──────────────┬───────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │   Existing AD Stack          │
   │   (Apollo / Autoware)        │
   │   Motion Planning Module     │
   │                              │
   │   Behavior → Trajectory      │
   └──────────────────────────────┘
```

## 4. Why Important

- **"LLM in the stack" design pattern** — doesn't require replacing everything end-to-end
- **Practical integration:** compatible with Apollo and similar frameworks
- **CARLA driving score improvements** reported
- **Frequently cited** (~237) as an alternative to pure end-to-end approaches

## 5. Limitations

- Not a top-venue archival paper
- Simulator-based validation only
- Integration quality depends on external stack interfaces
- Behavioral state abstraction may lose information

## 6. Interview Quick-Hit

**Q: How does DriveMLM differ from end-to-end driving VLMs?**
A: DriveMLM is a plug-and-play module for existing AD stacks—it handles behavioral planning while the existing stack handles perception and motion planning. End-to-end models (LMDrive, EMMA) replace the entire pipeline.

---

# LLaDA: Driving Everywhere with Large Language Model Policy Adaptation

> **Authors:** Boyi Li et al.
> **Venue:** CVPR 2024 | **DOI:** [10.1109/CVPR52733.2024.01416](https://doi.org/10.1109/CVPR52733.2024.01416)
> **Citations:** ~176 | **Impact:** ★★★★ (geographic rule adaptation via LLM)

---

## Problem

Deploying driving systems across different countries/regions requires adapting to **local traffic laws and customs**. Manual rule engineering is expensive and doesn't scale.

## Core Idea

Use LLMs to **interpret traffic rules from local driver handbooks** and adapt driving policies accordingly. Language serves as the medium for transferring geographic driving knowledge.

## Why Important

- Pushes "language as external knowledge source" into policy design
- Addresses real deployment challenge (geo-fencing)
- Supported by user studies and real-world dataset experiments

## VLA Fit

Ambiguous — language-driven adaptation layer, not a unified VLA policy. Conceptually important for geographically robust autonomy.
