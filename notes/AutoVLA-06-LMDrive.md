# LMDrive: Closed-Loop End-to-End Driving with Large Language Models

> **Authors:** Hao Shao, Yuxuan Hu, Letian Wang, Steven Waslander, Yu Liu, Hongsheng Li
> **Venue:** CVPR 2024 | **arXiv:** [2312.07488](https://arxiv.org/abs/2312.07488)
> **Citations:** ~373 | **Impact:** ★★★★★ (first closed-loop language-conditioned driving benchmark)
> **PDF:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Shao_LMDrive_Closed-Loop_End-to-End_Driving_with_Large_Language_Models_CVPR_2024_paper.pdf)

---

## 1. Problem & Motivation

Prior LLM driving works (DriveGPT4, GPT-Driver) evaluate **open-loop** only—replaying logged data without interactive feedback. Real driving involves **compounding errors** and **temporal consistency** that open-loop metrics miss entirely. Furthermore, no benchmark existed for testing whether LLM-based drivers can **follow natural-language instructions** in a closed-loop setting.

## 2. Core Idea

Build an **instruction-following multimodal LLM** that:
1. Takes multi-sensor input (camera + LiDAR) + natural-language instructions
2. Outputs vehicle control signals directly
3. Operates in **closed-loop** CARLA simulation where errors compound and interact

Introduce **LangAuto**, the first language-guided closed-loop driving benchmark.

## 3. Architecture & Method

```
Multi-view Cameras    LiDAR Point Cloud
       │                     │
       ▼                     ▼
  Vision Encoder        LiDAR Encoder
       │                     │
       └──────┬──────────────┘
              ▼
     Multimodal Adapter
              │
    ┌─────────┴─────────┐
    │   Language         │
    │   Instructions     │──► Text Tokenizer
    │   (navigation +    │        │
    │    notice)         │        │
    └────────────────────┘        │
              │                   │
              ▼                   ▼
        ┌──────────────────────────┐
        │     LLM Core (frozen)    │
        │   + learnable adapters   │
        └──────────┬───────────────┘
                   │
                   ▼
            Control Output
            (steer, throttle, brake)
```

**Key design choices:**
- **LLM frozen** — preserves language reasoning; adapts via encoders + adapters
- **Camera + LiDAR** — multimodal sensor fusion (not vision-only)
- **Instruction types:**
  - Navigation: "Turn left at the next intersection"
  - Notice: "Watch out for the pedestrian on the right"
  - Multiple phrasings for diversity
  - Misleading/infeasible instructions (test robustness)

## 4. LangAuto Benchmark

| Feature | Description |
|---------|-------------|
| Environment | CARLA simulator |
| Instructions | Natural-language navigation + notice |
| Dataset | ~64K instruction-following clips |
| Evaluation | Closed-loop driving metrics |
| Robustness tests | Misleading and infeasible instructions |
| Instruction variants | Multiple phrasings per scenario |

## 5. Key Results

- Successfully follows language instructions in closed-loop CARLA driving
- Handles diverse instruction types including cautionary notices
- Robustness to misleading instructions (doesn't blindly follow unsafe commands)
- Temporal consistency maintained across driving sequences

## 6. Why Seminal

1. **First named closed-loop language-conditioned driving benchmark** (LangAuto)
2. **Released ~64K instruction-following clips** — reproducible benchmark
3. **Addressed the critical open-loop → closed-loop gap** in driving VLMs
4. **Multi-sensor (camera + LiDAR)** — realistic sensor setup
5. **Instruction robustness** — tested misleading/infeasible commands

## 7. Limitations

- **CARLA only** — simulator proxy, not real-world deployment
- **Instruction diversity** partially synthetic (ChatGPT augmentation)
- **Frozen LLM trade-off:** preserves reasoning but limits driving-specific adaptation
- **No world model** — doesn't predict future states for planning

## 8. Open-Loop vs Closed-Loop: Why It Matters

```
Open-Loop (GPT-Driver, DriveGPT4-V1):
  Log Replay → Predict Action → Score vs Ground Truth
  ✗ No compounding errors
  ✗ No interaction with environment
  ✗ Can't test temporal consistency

Closed-Loop (LMDrive, SimLingo):
  Live Environment → Predict Action → Execute → Observe → Repeat
  ✓ Errors compound realistically
  ✓ Must react to changed environment
  ✓ Tests full driving loop
```

## 9. Connections

- **← CIL:** Upgrades discrete commands to natural language
- **← DriveGPT4:** Moves from open-loop to closed-loop evaluation
- **→ SimLingo:** Adds language-action alignment to closed-loop driving
- **→ ORION:** Uses similar LLM core but adds generative planner for trajectories
- **→ Bench2Drive:** Provides standardized closed-loop evaluation

## 10. Interview Quick-Hits

**Q: Why is LMDrive significant compared to earlier driving LLM works?**
A: It's the first to demonstrate and benchmark LLM-based driving in closed-loop, where errors compound. Prior work (GPT-Driver, DriveGPT4) only evaluated on replayed logs, which can't capture interactive driving competence.

**Q: What is LangAuto?**
A: A language-guided CARLA benchmark where agents are navigated by natural-language instructions (not GPS waypoints or discrete commands). Includes ~64K instruction-following clips with navigation, notice, and adversarial (misleading) instructions.

**Q: How does LMDrive handle misleading instructions?**
A: The frozen LLM retains reasoning capability to assess instruction feasibility. When an instruction conflicts with safety (e.g., "run the red light"), the system prioritizes safe driving behavior over instruction compliance.

---

*VLA fit: Full VLA (multi-sensor vision + language instructions + control actions). Landmark closed-loop benchmark.*
