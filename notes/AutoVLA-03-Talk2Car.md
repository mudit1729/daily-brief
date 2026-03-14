# Talk2Car: Taking Control of Your Self-Driving Car

> **Authors:** Thierry Deruyttere, Simon Vandenhende, Duber Grez, Luc De Raedt, Marie-Francine Moens
> **Venue:** EMNLP-IJCNLP 2019 | **DOI:** [10.18653/v1/D19-1215](https://doi.org/10.18653/v1/D19-1215)
> **Citations:** ~209 | **Impact:** ★★★★ (canonical NL driving grounding dataset)
> **Dataset:** [talk2car.github.io](https://talk2car.github.io/)

---

## 1. Problem & Motivation

For autonomous vehicles to be truly useful, passengers should be able to issue **natural-language commands** like *"park behind that blue car"* or *"follow the bus ahead."* The system must **ground** the referred object in the visual scene so downstream planning can act on it. This requires spatial language understanding in driving contexts.

## 2. Core Idea

Build a **natural-language command grounding dataset** on top of nuScenes, where each command refers to a specific object in the driving scene. Evaluate models on their ability to predict the correct bounding box for the referred object, using AP/IoU-style metrics.

## 3. Dataset & Evaluation

```
Input:  nuScenes camera image + natural language command
        "Park behind the white car on the right"
                    │
                    ▼
        Vision-Language Grounding Model
                    │
                    ▼
Output: Bounding box of referred object
        Evaluate: IoU > 0.5 → AP50
```

**Dataset statistics:**
- Built on nuScenes driving scenes
- Natural-language commands written by human annotators
- Each command refers to a visible object in the scene
- Commands express actions the vehicle should take w.r.t. the object

**Metrics:** AP50, AP (average precision at IoU thresholds)

## 4. Why Seminal

1. **First widely-referenced NL command dataset for driving** — one of the earliest to position natural-language interaction as a driving interface
2. **Practical framing:** commands are actions for the vehicle grounded via visible objects
3. **Reusable benchmark** that enabled a stream of driving-focused language grounding models
4. **Bridge:** stepping stone from "language + scene understanding" to "language-conditioned planning/control"

## 5. Limitations

- Benchmark is **object grounding**, not full end-to-end driving—action semantics are implicit
- Language diversity limited by annotation process
- Inherits nuScenes scene coverage limitations
- No trajectory or control output evaluation

## 6. Connections

| Work | Relationship |
|------|-------------|
| Talk2Car-Trajectory | Extension: adds trajectory prediction for commands |
| LMDrive | Full VLA: takes language instructions → closed-loop control |
| nuScenes | Base driving dataset Talk2Car builds on |

---

# Talk2Car-Trajectory: Language to Trajectory

> **Authors:** Thierry Deruyttere et al.
> **Venue:** IEEE Access 2022 | **DOI:** [10.1109/ACCESS.2022.3224144](https://doi.org/10.1109/ACCESS.2022.3224144)

## Extension

Moves beyond "which object?" to **"what path should the car follow?"** for a given command. Decomposes into:
1. **Object referral** — align command with candidate objects
2. **Trajectory prediction** — generate a path that executes the command

**Why important:** Early, clear step toward language-conditioned driving trajectories (language → action in trajectory form).

**Limitation:** Open-loop/local trajectory prediction without closed-loop validation.

## Interview Quick-Hits

**Q: What does Talk2Car evaluate?**
A: Given a camera image and a natural-language passenger command, predict the bounding box of the referred object. Metric: AP at IoU > 0.5.

**Q: How does Talk2Car-Trajectory extend this?**
A: Instead of just localizing the referred object, it predicts a trajectory the car should follow to execute the command. This bridges grounding and planning.

---

*VLA fit: Vision (nuScenes images) + Language (commands) + Action (grounding → trajectory). Clear precursor to full VLA driving systems.*
