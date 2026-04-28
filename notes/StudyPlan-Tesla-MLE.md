# Tesla MLE Study Plan

A focused prep plan for the Tesla Machine Learning Engineer technical screen, mapped to existing notes in this viewer. Based on candidate-reported screen patterns: live coding (28%), ML/CV fundamentals (16%), NumPy/CNN arithmetic (20%), metrics/SQL/latency (24%), and short MLOps checks (12%).

---

## MUST (Single Highest-ROI Action)

If you only have **one week**, focus exclusively on these four notes. They cover ~80% of what the deep research predicts you'll see in the screen.

### 1. `Blind-75.md` — Timed Coding Practice (highest priority)

**Why**: Live LeetCode mediums in 25–40 min are the most common Tesla screen format. Blind-75 covers 4 of the 12 candidate-reported questions directly:
- Decode Ways (DP)
- Palindrome Linked List
- Reverse Words in a String
- Top K Frequent Elements

**How to use**: Time every problem (25 min hard cap). Speak complexity out loud before coding. Keep an error log of what you missed.

### 2. `Tesla-ML-Engineer-Study-Guide.md` — Tesla-Specific Recipes

**Why**: This is the densest, most Tesla-aligned guide. §1.1 contains ready-to-write implementations for:
- `softmax`, `cross_entropy_from_logits` (stable versions)
- `iou_matrix`, `nms` (vectorized)
- `waypoint_suffix_distances` (exact candidate-reported question)
- `braking_decision` (stopping distance physics)
- `conv2d_forward_nchw`, `scaled_dot_product_attention`
- `confusion_matrix`, `precision_recall_f1`
- `top_k_hardest`, `first_crossing` (binary search)

**How to use**: Type out every implementation from memory. Don't copy-paste. The screen tests muscle memory.

### 3. `Tesla-NumPy-ML-Interview-Guide.md` — Vectorization Drills

**Why**: NumPy/vectorization is 10% of the screen and a known recruiter-mentioned focus area for Tesla Optimus and AI roles. Covers:
- Broadcasting rules
- Stable softmax / cross entropy
- Pairwise distances / cosine similarity
- IoU / NMS vectorized
- Conv2D output shape and forward pass
- Top-25 NumPy questions with timed solutions

**How to use**: Implement each "Top Asked" question without looking. Aim for under 10 min per problem.

### 4. `Cheatsheet-02-Machine-Learning.md` — Stats & Regularization Quick Answers

**Why**: Confusion matrix and L1/L2 regularization were both candidate-reported from Tesla data-science screens. You need fast, crisp verbal answers — not lengthy derivations.

**How to use**: Read once, then close the file and answer aloud:
- Define TP/FP/FN/TN, then derive precision/recall/F1
- When does PR-AUC beat ROC-AUC?
- L1 vs L2 — when is each preferred?
- Bias/variance diagnosis from train/val curves

---

## GAP-FILLING NOTES (High-Priority Adds)

These four notes were added specifically to plug critical gaps the standard prep material doesn't cover. Promote any of these into MUST if your screen has been confirmed to include the topic.

### `Tesla-SQL-Cheatsheet.md` — SQL Deep-Dive
Covers what `Tesla-ML-Engineer-Study-Guide.md` only sketches: window functions (RANK, LAG, ROW_NUMBER), CTEs, dedup patterns, anti-joins, gaps-and-islands, time-bucketing, NULL traps. 12 Tesla-flavored worked questions (drive_events, fleet logs, calibration drift). Read if your screen has a SQL round.

### `Tesla-Model-Optimization.md` — Quantization, Pruning, Latency
The "your model is 3x too slow" answer scaffold. PTQ vs QAT, AMP with `GradScaler`, structured vs unstructured pruning, knowledge distillation with temperature, FlashAttention, profiling workflow, deployment (ONNX/TensorRT). Read if your screen mentions edge inference, latency, or efficient deployment.

### `Tesla-Tracking-Classical-CV.md` — Kalman, SORT, HOG
Older Tesla CV interviews still ask Kalman + HOG. Predict/update math, 1D NumPy implementation, EKF/UKF/particle filter, SORT/DeepSORT/ByteTrack, Hungarian assignment, optical flow, HOG/SIFT/ORB, camera calibration, Mahalanobis gating, track lifecycle. Read if your screen has CV depth or autonomy follow-ups.

### `Tesla-Probability-Bayes-Drill.md` — Probability Puzzles
15 classic puzzles (Monty Hall, disease testing/base rates, two-child, birthday, coupon collector, reservoir sampling, A/B sample size, conditional independence/colliders) plus ML-specific Bayes (cross-entropy as KL, MAP=MLE+regularization, GDA, Naive Bayes). Read if your screen mentions stats, calibration, or uncertainty.

---

## GOOD TO HAVE (Additional Reading Material)

If you have **2–4 weeks**, add these for depth, polish, and Tesla-specific talking points.

### Coding Depth
- **`Amazon-150.md`** — broader pattern coverage including "Build Tree from Preorder + Inorder" (candidate-reported from Senior MLE process)
- **`Graph-Problems-Guide.md`** — tree/graph traversal patterns (BFS, DFS, topological sort)
- **`Torc-Live-Code-Pair-Question-Bank.md`** — live-coding format under interviewer pressure
- **`DeepML-Easy.md`, `DeepML-Medium.md`, `DeepML-Hard.md`** — implement-from-scratch ML problems

### CNN & Vision Depth
- **`CMU-Lec09_CNN_Part1.md` → `CMU-Lec12_CNN_Part4.md`** — the 4-lecture deep CNN sequence (output-shape formulas, receptive field, BatchNorm params, dilated/grouped conv)
- **`Cheatsheet-01-Computer-Vision.md`** — CV fundamentals reference
- **`Cheatsheet-04-CV-Interview-Revision-Handbook.md`** — CV revision

### PyTorch Production Skills
- **`Tesla-PyTorch-ML-Interview-Guide.md`** — autograd, training loops, AMP, debugging playbook
- **`PyTorch-Refresher-Cheatsheet.md`** — quick syntax refresh

### ML Theory & Interview Q&A
- **`MLTheory-Classical-ML-Interview-Questions.md`** — classical ML (SVM, decision trees, ensembles, calibration)
- **`MLTheory-Interview-QA.md`** — broader theory Q&A
- **`Cheatsheet-03-ML-Coding.md`** — additional ML coding patterns

### Transformer/ViT (for follow-up depth)
- **`DeepML-Attention-Is-All-You-Need.md`** — attention from scratch with theory + interview Q&A
- **`DeepML-Vision-Transformer.md`** — ViT pipeline with theory + interview Q&A
- **`MLPaper-04-ViT-Vision-Transformers.md`** — paper-level ViT understanding

### Tesla-Specific Autonomy (BEV — Tesla uses these networks)
Pick **2–3** for talking points if asked about perception:
- **`BEV-06_BEVFormer_BEVFormerV2.md`** — most cited BEV transformer
- **`BEV-07_BEVFusion.md`** — multi-sensor fusion
- **`BEV-10_UniAD.md`** — planning-aware perception (Tesla-relevant)

### ML System Design & Production
- **`MLSystemDesign-Prep.md`** — for production/MLOps follow-up questions (rollout, monitoring, drift, regression gates)

### Trajectory/Planning (one only, optional)
- **`Planner-04_pluto_summary.md`** OR **`Planner-08_large_trajectory_models_summary.md`** — for closed-loop / motion-planning context

---

## SKIP (Not Relevant for Screen)

These are great long-term ML knowledge but won't move the needle for a Tesla MLE technical screen:

- **All `Ilya30-*`** (deep paper summaries — long-term study, not screen)
- **All `VLA-*`** (manipulation robotics — not Tesla-relevant)
- **All `AutoVLA-*`** (skim 1–2 only if you want one talking point on language+driving)
- **`RL-*`** (RL is not in screen scope per the research article)
- **`Scale-*`** (different company)
- **`Async-*`** (not Tesla-specific)
- **`CMU-Lec13`+** (RNN/Transformers/GANs — Tesla screens are CV-heavy, not generative)
- **`MLPaper-02-LoRA`, `03-PEFT`, `05-VAE`, `06-GANs`** (not screen-critical)
- **`Paper-AlphaGo.md`** (off-topic)

---

## 2-Week Sprint Schedule

| Day | Focus | Notes |
|-----|-------|-------|
| 1 | Arrays / hash maps + complexity narration | `Blind-75` (arrays) |
| 2 | Trees + Build Tree pattern | `Amazon-150` (trees) + `Graph-Problems-Guide` |
| 3 | DP — Decode Ways family | `Blind-75` (DP) |
| 4 | Strings + Linked Lists | `Blind-75` (Reverse Words, Palindrome LL) |
| 5 | Heaps / Top-K + SQL refresh | `Blind-75` (heap) + `Tesla-Study-Guide` SQL section |
| 6 | CNN arithmetic | `CMU-Lec09-12` + `Tesla-Study-Guide` §1.1 |
| 7 | NumPy drills | `Tesla-NumPy-Guide` (timed top-25) |
| 8 | 2D conv coding + im2col | `Tesla-NumPy-Guide` Q14 + `DeepML-Hard` |
| 9 | Stats: confusion matrix, PR/ROC, L1/L2 | `Cheatsheet-02-ML` + `MLTheory-Classical-ML-QA` |
| 10 | TTC, has_collided, scenario slicing | `Tesla-Study-Guide` §1.1 (waypoint, braking, IoU, NMS) |
| 11 | SQL / data pipelines | `Tesla-Study-Guide` SQL Q7 + practice |
| 12 | Latency / deployment short answers | `Tesla-PyTorch-Guide` (AMP, debugging) + `Tesla-Study-Guide` §3.4 |
| 13 | **Mock Screen A** — coding-first, 45 min | timed |
| 14 | **Mock Screen B** — ML-first, 50 min + error log review | timed |

---

## Daily Cadence

For each prep day:
- **75–90 min** timed coding (Blind-75 or DeepML)
- **45–60 min** ML/CV or NumPy/CNN work (Tesla guides or CMU CNN lectures)
- **20–30 min** metrics / SQL drill (cheatsheets)
- **10–15 min** spoken-answer rehearsal (talk through confusion matrix, L1/L2, conv shapes out loud)

---

## Recovery Script (If You Freeze Mid-Screen)

> "Let me restate the inputs, outputs, and constraints. Here is the simplest correct approach. Here is the complexity. If we have time, I will optimize."

Boring but it saves screens. Memorize it.

---

## Self-Score Rubric (After Each Mock)

| Dimension | Weight |
|-----------|--------|
| Clarification & framing | 15 |
| Correctness | 20 |
| Complexity & trade-offs | 15 |
| ML/CV depth | 15 |
| Metrics & data judgment | 10 |
| Communication | 10 |
| Time management | 10 |
| Production judgment | 5 |

- **85–100**: screen-ready
- **70–84**: close but brittle
- **<70**: still leaking time on foundations
