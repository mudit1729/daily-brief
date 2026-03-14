# Seminal Vision-Language-Action Papers in Autonomous Driving

## Executive Summary

VLA in autonomous driving has progressed in **three waves:**

### Wave 1: Controllable End-to-End Driving (2017-2018)
Made end-to-end driving controllable by conditioning on symbolic "commands" (proto-language). Command-conditional imitation learning showed that high-level intent resolves perceptual ambiguity at intersections.

### Wave 2: Language for Driving (2018-2022)
Introduced human-facing language—either as **explanation output** (BDD-X, attention-aligned justifications) or as **command grounding** (Talk2Car: natural-language command → referred object in driving scenes).

### Wave 3: Foundation Model Era (2023-2025)
LLMs and VLMs used to: (i) convert planning into text-token generation, (ii) do multi-step reasoning about scenes, and (iii) align reasoning with physically grounded actions in closed-loop simulators.

---

## Ranked Papers by Impact

| Rank | Paper | Venue | Citations | VLA Fit |
|------|-------|-------|-----------|---------|
| 1 | [Conditional Imitation Learning](AutoVLA-01) | ICRA 2018 | ~1651 | Proto-VLA |
| 2 | [Textual Explanations / BDD-X](AutoVLA-02) | ECCV 2018 | ~555 | Language-output |
| 3 | [DriveGPT4](AutoVLA-05) | IEEE RAL 2024 | ~723 | Full VLA |
| 4 | [DriveLM / GVQA](AutoVLA-08) | ECCV 2024 | ~593 | Full VLA |
| 5 | [LMDrive / LangAuto](AutoVLA-06) | CVPR 2024 | ~373 | Full VLA |
| 6 | [GPT-Driver](AutoVLA-04) | NeurIPS 2023 | ~456 | Planning-as-language |
| 7 | [VLP](AutoVLA-07) | CVPR 2024 | ~176 | LM-as-prior |
| 8 | [Talk2Car](AutoVLA-03) | EMNLP 2019 | ~209 | Grounding |
| 9 | [Reason2Drive](AutoVLA-09) | ECCV 2024 | ~142 | Benchmark |
| 10 | [SimLingo](AutoVLA-10) | CVPR 2025 | ~67 | Alignment-first VLA |
| 11 | [ORION](AutoVLA-11) | ICCV 2025 | ~74 | Reasoning-action bridge |
| 12 | [EMMA](AutoVLA-12) | TMLR 2025 | ~212 | Generalist model |

---

## Three Design Axes

### Where Language Enters/Exits

```
Language as input:     LMDrive, SimLingo, ORION
Language as output:    Textual Explanations, DriveGPT4
Language as internal:  GPT-Driver, EMMA, VLP
```

### Action Representation

```
Low-level controls:    CIL, LMDrive, DriveGPT4
Trajectories:          VLP, GPT-Driver, ORION, EMMA
Discrete behaviors:    DriveLM (behavior → motion)
```

### Evaluation Regime

```
Open-loop:   nuScenes planning, BDD-X prediction
Closed-loop: CARLA Leaderboard, Bench2Drive, LangAuto
```

---

## Timeline

```
2017  CARLA simulator
2018  Conditional Imitation Learning (proto-VLA)
      Textual Explanations + BDD-X (language output)
2019  Talk2Car (NL command grounding)
2022  Talk2Car-Trajectory (NL → trajectory)
2023  GPT-Driver (planning as language modeling)
      DriveMLM (LLM module in AD stack)
2024  LMDrive + LangAuto (closed-loop instruction following)
      DriveGPT4 (VLM instruction tuning)
      VLP (LM priors for BEV planning)
      DriveLM + GVQA (graph reasoning)
      Reason2Drive (reasoning chain benchmark)
      LLaDA (geographic rule adaptation)
2025  SimLingo (language-action alignment, CARLA winner)
      ORION (reasoning-action gap, Bench2Drive SOTA)
      EMMA (generalist "everything as language")
```

---

## Key Benchmarks & Datasets

| Benchmark | Type | Source |
|-----------|------|--------|
| **CARLA / Leaderboard** | Closed-loop simulator | CoRL 2017 (~9250 citations) |
| **Bench2Drive** | Closed-loop, scenario-disentangled | CARLA v2, 220 routes |
| **LangAuto** | Language-guided closed-loop | LMDrive, ~64K clips |
| **DriveLM-Data** | Graph VQA + full stack | nuScenes + CARLA |
| **Reason2Drive** | Reasoning chains, >600K pairs | nuScenes + Waymo + ONCE |
| **BDD-X** | Explanations + control | BDD100K + language |
| **Talk2Car** | NL command grounding | nuScenes + commands |
| **nuScenes** | Perception + planning | ~10K+ citations |
| **Waymo Open** | Perception + planning | ~4984 citations |

---

## Unsolved Problems

1. **Closed-loop sim wins ≠ deployment** — all leading VLA papers are simulator-only for actions
2. **Language-action consistency** — language outputs can be inconsistent with driving behavior
3. **Semantic → numeric gap** — converting "yield to pedestrian" to precise trajectories
4. **Data realism** — synthetic paraphrases / LLM-generated instructions ≠ real user language
5. **Computation + sensors** — EMMA flags frame limits, no LiDAR, and high cost
6. **Safety guarantees** — no VLA paper provides formal safety guarantees under distribution shift

---

## Key Recurring Lessons

> **"Evaluation protocol choices shape what success looks like."**

- Open-loop L2 errors can miss compounding failures
- Closed-loop benchmarks (Bench2Drive) created to address this
- Even within CARLA, long-route vs short-route scoring changes incentives

> **"Language-action consistency is the primary failure mode."**

- SimLingo: VQA understanding ≠ driving competence unless aligned
- ORION: semantic reasoning ≠ feasible trajectories unless bridged

> **"The hardest unsolved problem is guaranteeing that language-conditioned reasoning is causally tied to safe, feasible actions under distribution shift."**
