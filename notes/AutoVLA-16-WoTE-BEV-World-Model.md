# WoTE: End-to-End Driving with Online Trajectory Evaluation via BEV World Model

> **Authors:** Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, Zhaoxiang Zhang
> **Venue:** arXiv 2025 | **arXiv:** [2504.01941](https://arxiv.org/abs/2504.01941)
> **Impact:** ★★★★ (BEV world model for trajectory evaluation, NAVSIM + Bench2Drive SOTA)
> **Code:** GitHub (public)

---

## 1. Problem & Motivation

End-to-end autonomous driving models generate trajectory proposals, but how do you **evaluate which trajectory is actually safe** before executing it? Traditional approaches either:
- Trust a single trajectory output (risky — no fallback)
- Use image-level world models to simulate futures (computationally expensive)
- Rely on hand-crafted cost functions (limited generalization)

The key insight: a **BEV-space world model** can efficiently predict future states and evaluate trajectory safety, without the computational cost of image-level rendering.

## 2. Core Idea

**WoTE (World model for Trajectory Evaluation):** Use a BEV world model to:
1. Generate multiple candidate trajectories
2. **Predict future BEV states** conditioned on each trajectory
3. **Evaluate** which trajectory leads to the safest/best outcome
4. Execute the winning trajectory

All within a fully differentiable framework integrating perception, prediction, and planning.

## 3. Architecture & Method

```
Multi-view Camera Images
         │
         ▼
┌────────────────────────┐
│  BEV Feature Extraction│
│  (Perception)          │
└──────────┬─────────────┘
           │ BEV features
           ▼
┌────────────────────────┐
│  Trajectory Proposal   │
│  Generator             │
│                        │
│  Output: K candidate   │
│  trajectories          │
│  τ₁, τ₂, ..., τₖ      │
└──────────┬─────────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
┌────────────────────────┐
│  BEV World Model       │
│                        │
│  For each trajectory τᵢ:│
│  Predict future BEV    │
│  states at t+1, t+2... │
│                        │
│  Evaluate: collision?  │
│  off-road? smooth?     │
└──────────┬─────────────┘
           │ Scores per trajectory
           ▼
┌────────────────────────┐
│  Trajectory Selection  │
│                        │
│  Pick best τ* based on │
│  world model evaluation│
└──────────┬─────────────┘
           │
           ▼
    Execute τ* (best trajectory)
```

**Why BEV space (not image space)?**
```
Image-level world model:
  ✗ Must render full RGB images for each future step
  ✗ Computationally expensive (diffusion models, etc.)
  ✗ Much of the rendered detail is irrelevant to safety

BEV world model (WoTE):
  ✓ Compact representation (top-down 2D map)
  ✓ Directly encodes what matters: positions, lanes, obstacles
  ✓ Efficient to simulate multiple trajectories
  ✓ Compatible with existing BEV traffic simulators
```

## 4. Key Results

| Benchmark | Performance |
|-----------|------------|
| **NAVSIM** | State-of-the-art |
| **Bench2Drive** (closed-loop CARLA) | State-of-the-art |
| Computational efficiency | Significantly faster than image-level world models |

## 5. Why Significant

1. **Online trajectory evaluation** — don't just predict a trajectory, *simulate its consequences* before executing
2. **BEV efficiency** — world model operates in compact BEV space, not expensive image space
3. **Fully differentiable** — perception→prediction→planning trained end-to-end
4. **Dual benchmark SOTA** — strong on both NAVSIM (open-loop) and Bench2Drive (closed-loop)
5. **Connects to BEV literature** — bridges the BEV perception stack (BEVFormer, UniAD) with planning

## 6. World Model Approaches Comparison

| Approach | Representation | Cost | Quality | Example |
|----------|---------------|------|---------|---------|
| No world model | N/A | Lowest | No future eval | Basic E2E |
| Image-level WM | RGB pixels | Very high | Rich visual detail | GAIA-1, DriveDreamer |
| **BEV world model** | Top-down map | **Low** | Sufficient for planning | **WoTE** |
| Latent WM | Learned features | Medium | Depends on training | MILE |

## 7. Connections

- **← UniAD:** Shares the BEV-based unified perception-prediction-planning paradigm
- **← BEVFormer:** Builds on BEV feature extraction technology
- **← Bench2Drive:** Uses the same closed-loop evaluation benchmark as SimLingo/ORION
- **↔ ORION:** Different approach — ORION uses LLM reasoning, WoTE uses world model simulation
- **→ Future:** BEV world models may complement VLA reasoning for safety verification

## 8. Key Insight: Simulate Before You Act

```
Traditional E2E:
  Observe → Plan → Act → Hope for the best

WoTE:
  Observe → Generate candidates → Simulate each in world model
                                         │
                                  "What happens if I take τ₁?"
                                  "What happens if I take τ₂?"
                                  "What happens if I take τ₃?"
                                         │
                                  Pick safest/best → Act
```

This is analogous to **tree search in game playing** (AlphaGo): look ahead, evaluate outcomes, choose the best move. WoTE brings this principle to driving.

## 9. Interview Quick-Hits

**Q: What is WoTE's core contribution?**
A: Using a BEV-space world model to evaluate trajectory candidates *before* executing them. Instead of trusting a single trajectory output, WoTE simulates multiple futures and picks the safest path. BEV space is key — it's computationally efficient compared to image-level world models.

**Q: Why BEV instead of image-level world models?**
A: Image-level world models must render full RGB images for each simulated timestep — expensive and wasteful since most pixel-level detail is irrelevant for safety evaluation. BEV compactly encodes what matters (positions, lanes, obstacles) and is fast enough to evaluate multiple trajectories in real time.

**Q: How does this relate to AlphaGo's tree search?**
A: Same principle — look ahead before acting. AlphaGo evaluates candidate moves by simulating game outcomes. WoTE evaluates candidate trajectories by simulating future driving states. Both use a learned model (value network / world model) to score outcomes without playing out the full sequence.

---

*VLA fit: Vision + Action (no explicit language component). Important for the trajectory evaluation / world model dimension of autonomous driving. Complements language-based VLA with physics-based safety verification.*
