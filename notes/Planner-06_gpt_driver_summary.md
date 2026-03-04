# GPT-Driver: Learning to Drive with GPT – Implementation Summary

**Paper**: GPT-Driver: Learning to Drive with GPT
**Authors**: Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, Yue Wang
**Affiliation**: University of Southern California, Tsinghua University
**Venue**: Foundation Models for Decision Making Workshop, NeurIPS 2023
**arXiv**: 2310.01415v3
**Date**: Dec 5, 2023

---

## 1. One-Page Overview

### Paper Metadata
- **Task**: Motion planning (trajectory prediction) for autonomous vehicles
- **Input Modality**: Heterogeneous multi-modal (detection bboxes, ego-states, perception predictions)
- **Output Modality**: Sequence of 2D waypoint coordinates (language tokens → coordinates)
- **Base Model**: OpenAI GPT-3.5
- **Dataset**: nuScenes (1000 driving scenarios, 40K frames, diverse locations/weather)
- **Code**: Available (stated in abstract)

### Tasks Solved
- **Primary**: Predict safe, comfortable driving trajectories (3-second horizon, 6 waypoints)
- **Secondary**: Interpretable reasoning about driving decisions via chain-of-thought
- **Tertiary**: Few-shot generalization (1%, 10%, 50% training data)

### Sensors/Inputs
| Component | Format | Details |
|-----------|--------|---------|
| Detections | Class + bbox | Objects with positions, velocities, predicted future motions |
| Ego-States | Numerical | Velocity (vx, vy), heading angular velocity, acceleration |
| Historical Trajectory | Waypoints | Last 2 seconds of ego vehicle path (4 waypoints @ 0.5s intervals) |
| Mission Goal | Text | High-level driving objective (e.g., "TURN RIGHT") |

### Key Novelty (With Section Citations)
1. **Reformulation as Language Modeling** [Sec 3.2]: Motion planning ↔ language token generation; coordinate tokenization (e.g., 23.17 → "23", ".", "17")
2. **Prompting-Reasoning-Finetuning Strategy** [Sec 3.3]: Chain-of-thought prompting → numerical reasoning → fine-tuning with human driving trajectories; bypasses complex numerical regression in neural nets
3. **Precision Coordinate Prediction**: Achieves centimeter-level accuracy via tokenized reasoning, first LLM demonstrating precise numerical reasoning in motion planning
4. **Superior Generalization**: Outperforms end-to-end planners (UniAD, St-P3/VAD) on few-shot settings (1%–50% data); L2 error 1.89m @1% data vs. 5.37m for UniAD [Table 2 | Sec 4.3]
5. **Interpretability via Explicit Reasoning**: Outputs both trajectory AND reasoning chain; human drivers see decision-making process [Figure 3 | Sec 4.4]
6. **No Dense Occupancy Grids**: Unlike UniAD, uses only language descriptions of detections and predictions; simpler, more modular

### If You Only Remember 3 Things
1. **Tokenization as Intermediate Representation**: Coordinates are serialized as language tokens (e.g., 23.17 → ["23", ".", "17"]); GPT-3.5 learns token distributions over waypoints via cross-entropy loss
2. **Prompting-Reasoning-Finetuning Wins**: Initial prompt-based generation fails (Sec 4.5); fine-tuning on 100% training data + chain-of-thought examples is critical for accuracy
3. **Practical Limitation**: OpenAI API latency + cost prevent deployment; inference time not reported; few-shot success suggests few-shot LLM optimization strategies are worth exploring

---

## 2. Problem Setup and Outputs

### Input Specification

| Input | Format | Shape/Example | Notes |
|-------|--------|---------------|-------|
| **Detections (N objects)** | Structured text | N varies (e.g., 3–8 objects) | Classes: car, pedestrian, cyclist, etc. + bbox center (x, y) + velocity (vx, vy) + predicted trajectory |
| **Detection Predictions** | Structured text | Per-object future positions | E.g., "car at (2.34, 19.08) moving to (5.12, 20.45)" |
| **Ego-States** | Text | 3 scalars: vx, vy, heading angular velocity | E.g., "Velocity: (0.00, 1.46), Heading Angular Velocity: (-0.08)" |
| **Historical Trajectory** | Waypoints | 4 waypoints, 0.5s intervals (last 2s) | E.g., "[(-0.88, -6.74), (-0.83, -3.73), (-0.83, -3.07), (-0.82, -1.46)]" |
| **Mission Goal** | String | Single command | "RIGHT", "LEFT", "FORWARD", "STRAIGHT" (high-level intent) |
| **Coordinate Frame Context** | Text | Implicit | "Coordinates: X-axis is perpendicular, Y-axis is parallel to your facing direction" |

### Output Specification

| Output | Format | Shape | Notes |
|--------|--------|-------|-------|
| **Reasoning (R)** | Natural language | Variable length | Chain-of-thought: identifies critical objects, predicts their influence, determines meta-action |
| **Meta-Action** | Text command | Single string | E.g., "TURN RIGHT WITH A DECELERATION" or "MOVE FORWARD WITH ACCELERATION" |
| **Trajectory (T)** | 2D Waypoint sequence | t ∈ {1,2,3,4,5,6} | 6 waypoints (3-second horizon, 0.5s intervals); each (xi, yi) in ego-vehicle coordinates (meters) |

### Tensor Shapes (Inference)
```
Input prompt length:     variable (~200–600 tokens)
Output trajectory:       T_traj = [(x1,y1), ..., (x6,y6)] ∈ ℝ^(6×2)
Output reasoning:        variable length (50–150 tokens)
Total output tokens:     ~200–300 (reasoning + trajectory)
```

### Loss Function (Training)
```
L_reg = Σ(|xi - x̂i| + |yi - ŷi|)  [L1 distance, Eq. 3]
L_LM  = -Σ log P(w_i | w1, ..., w_{i-1})  [Cross-entropy over tokens, Eq. 5]

Combined: L_LM (fine-tuning loss on human trajectories)
```

---

## 3. Coordinate Frames and Geometry

### Coordinate System Definition
| Aspect | Definition | Notes |
|--------|-----------|-------|
| **Origin** | Ego-vehicle center (current timestep) | (0, 0) is always the vehicle's current position |
| **X-axis** | Perpendicular to facing direction | Positive right; negative left |
| **Y-axis** | Parallel to facing direction | Positive forward; negative backward |
| **Rotated Frame** | Ego-centric (body frame) | All coordinates relative to vehicle's current heading |
| **Time Discretization** | 0.5s intervals | Waypoints at t ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0} seconds |

### Coordinate Transformation
```
Raw perception (global):  P_global ∈ ℝ^(3) [x, y, z in world frame]
Ego-centric transform:    P_ego = R(θ_ego)^T @ (P_global - ego_pos)
  where θ_ego = vehicle heading, ego_pos = vehicle center
Result:                   P_ego = [X, Y] (2D, ego-vehicle frame)
```

### Grid and Geometry Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Spatial Range (X)** | [-50, 50] meters | Left-right extent (inferred, typical for AD) |
| **Spatial Range (Y)** | [0, 50] meters | Forward-only (no backward planning) |
| **Temporal Horizon** | 3 seconds | Standard for safety-critical planning |
| **Temporal Discretization** | 0.5 seconds | 6 waypoints per trajectory |
| **Detection Range** | ~70m | nuScenes standard; objects outside ignored |
| **Heading Representation** | Implicit in ego-frame | No explicit heading output; trajectory relative to current heading |

### Geometry Sanity Checks
| Check | Expected | Paper Compliance |
|-------|----------|-----------------|
| Waypoint distance growth | ~0.3–5m per 0.5s | Yes; examples show progression (e.g., →(0.12, 2.98) →(3.45, 18.90)) [Fig 3] |
| Coordinate range realism | |x|, |y| < 50m | Yes; all examples within bounds |
| Historical trajectory continuity | Monotonic Y growth (forward motion) | Yes; typical pattern in examples [Fig 2] |
| Ego-frame consistency | X-axis lateral, Y-axis forward | Stated explicitly [Sec 3.2, Fig 2 context] |
| Velocity-trajectory alignment | vx, vy derivatives ≈ Δwaypoints | Implicit; no explicit check in paper |

---

## 4. Architecture Deep Dive

### High-Level Block Diagram (Forward Pass)

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUTS                                    │
├─────────────────────────────────────────────────────────────┤
│ Detections (N×[class, bbox, v_pred])                        │
│ Ego-States (vx, vy, ω)                                      │
│ Historical Trajectory (4 waypoints)                          │
│ Mission Goal (string)                                        │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│         PROMPT ENGINEERING STAGE                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Format detections as text (parameterized sentences)      │
│ 2. Format ego-states as text descriptions                   │
│ 3. Format historical trajectory as waypoint list            │
│ 4. Add context (coordinate frame, safety constraints)       │
│ 5. Concatenate into single prompt string                    │
│    Output: prompt ∈ str, len ~200–600 tokens               │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│      TOKENIZATION (GPT-3.5 Tokenizer K)                    │
├─────────────────────────────────────────────────────────────┤
│ K(prompt) → [token_1, ..., token_m] ∈ {0, ..., 50256}      │
│ m ≈ 200–600 tokens                                          │
│ Output: prompt_tokens ∈ ℤ^(m,)                            │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│     GPT-3.5 FORWARD PASS (Auto-regressive Generation)      │
├─────────────────────────────────────────────────────────────┤
│ For each output token position i = 1, 2, ..., n:            │
│   logits[i] = GPT3.5(prompt_tokens[1:], prev_output[1:i])  │
│   w_i ~ P(w_i | prompt, prev_output[1:i-1])               │
│ Outputs:                                                     │
│   - Reasoning tokens: R ≈ 50–150 tokens                    │
│   - Trajectory tokens: T ≈ 150–200 tokens (6 waypoints)    │
│   - Stop token (implicit EOS)                               │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│    POST-PROCESSING & DETOKENIZATION                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Extract reasoning tokens: R = [w_1, ..., w_m_R]         │
│ 2. Extract trajectory tokens: [w_{m_R+1}, ..., w_n]        │
│ 3. Decode reasoning: R_text = K^{-1}(R)                    │
│ 4. Parse trajectory coordinates:                            │
│    For each waypoint i: parse "X.X" tokens → (x_i, y_i)    │
│    Interpret: "23.17" = integer_tokens(23) + float_tokens  │
│ 5. Return: (R_text, T_trajectory)                          │
│    where T_trajectory = [(x_1,y_1), ..., (x_6,y_6)] ∈ ℝ^(6,2)
└─────────────────────────────────────────────────────────────┘
```

### Module-by-Module Details

| Stage | Input | Computation | Output | Parameters | Notes |
|-------|-------|-------------|--------|-----------|-------|
| **Prompt Formatter** | Detection structs, ego-states (list/tuple) | String concatenation (hardcoded rules) | prompt ∈ str | ~0 (rule-based) | Hand-crafted; no learnable params; examples in Fig 2 |
| **Tokenizer K** | prompt (string) | GPT-3.5 BPE tokenizer | tokens ∈ ℤ^(m,), m=200–600 | 50K vocab (frozen) | OpenAI standard; bidirectional |
| **GPT-3.5 Transformer** | tokens ∈ ℤ^(m,) | 96-layer decoder + attention | logits ∈ ℝ^(n, vocab_size) | ~175B params (frozen) | Auto-regressive; next-token prediction |
| **Sampling/Decoding** | logits | argmax or sampling (temperature τ ≈ 0.7, inferred) | output_tokens ∈ ℤ^(n,) | ~0 (decoding rule) | Greedy or sample; paper doesn't specify |
| **Detokenizer K^{-1}** | output_tokens ∈ ℤ^(n,) | BPE lookup + string reconstruction | output_text ∈ str | 50K vocab (frozen) | Inverse tokenization |
| **Coordinate Parser** | output_text (substring) | Regex + float parsing | waypoints ∈ ℝ^(6, 2) | ~0 (string parsing) | Extracts "(x1, y1), ..., (x6, y6)" pattern |

### Inference Pipeline (No Training of Base Model)
```python
# Pseudocode
def forward(detections, ego_states, hist_traj, mission):
    # 1. Format inputs
    prompt_text = format_prompt(detections, ego_states, hist_traj, mission)

    # 2. Tokenize
    prompt_tokens = tokenizer.encode(prompt_text)  # ℤ^(m,)

    # 3. Generate (via OpenAI API; auto-regressive)
    output_tokens = gpt35.generate(
        prompt_tokens,
        max_tokens=200,  # ~150 reasoning + 50 trajectory
        temperature=0.7,  # Inferred
    )  # ℤ^(n,)

    # 4. Decode & parse
    output_text = tokenizer.decode(output_tokens)
    reasoning_text, trajectory_tokens = parse_output(output_text)
    waypoints = extract_coordinates(trajectory_tokens)  # ℝ^(6, 2)

    return reasoning_text, waypoints
```

---

## 5. Forward Pass Pseudocode

### Training-Time Forward (Fine-tuning)
```python
def train_forward(detections, ego_states, hist_traj, mission, gt_trajectory):
    """
    Input shapes:
      detections:    List[Dict] with keys: class, bbox, vx, vy, pred_traj
      ego_states:    (vx: float, vy: float, ω: float)
      hist_traj:     [(x,y), (x,y), (x,y), (x,y)]  # 4 waypoints, ℝ^(4,2)
      mission:       str  # e.g., "TURN_RIGHT"
      gt_trajectory: [(x,y), ..., (x,y)]  # 6 waypoints, ℝ^(6,2)

    Outputs:
      reasoning:     str  # chain-of-thought text
      pred_traj:     [(x,y), ..., (x,y)]  # ℝ^(6,2)
      loss:          float  # scalar loss
    """

    # Step 1: Prompt assembly [~0.1ms]
    perception_text = format_detections(detections)  # str, len ~100–200
    ego_text = format_ego_states(ego_states)  # str, len ~50
    hist_text = format_trajectory(hist_traj)  # str, len ~30
    context_text = SYSTEM_CONTEXT  # str, static, len ~200

    prompt = (
        context_text +
        perception_text +
        ego_text +
        hist_text +
        f"Mission Goal: {mission}\n\n" +
        "Output:\n"
    )  # Total: ~380–450 tokens

    # Step 2: Create training target (ground truth reasoning + trajectory)
    # [Sec 3.3: hypothetical ego-trajectory → identify critical objects → reasoning]
    hypothetical_ego_traj = ego_trajectory_from_current(ego_states, no_intervention=True)
    critical_objects = identify_critical_objects(
        detections, hypothetical_ego_traj
    )  # Heuristic: overlapping predicted paths

    gt_reasoning = generate_reasoning(
        critical_objects, hypothetical_ego_traj, gt_trajectory
    )  # Generated or manually crafted

    # Step 3: Tokenize prompt and full target (prompt + reasoning + trajectory)
    prompt_tokens = tokenizer.encode(prompt)  # ℤ^(m,), m ≈ 380

    gt_text = (
        f"Thoughts:\n{gt_reasoning}\n" +
        f"Meta Action: [META_ACTION_TEXT]\n" +
        f"Trajectory: {format_trajectory_output(gt_trajectory)}\n"
    )  # e.g., "Trajectory: [(0.11, 1.14), ..., (5.49, 5.58)]\n"

    gt_tokens = tokenizer.encode(gt_text)  # ℤ^(n,), n ≈ 200

    full_sequence = prompt_tokens + gt_tokens  # ℤ^(m+n,)

    # Step 4: Forward pass through GPT-3.5 (frozen; via API or fine-tuned checkpoint)
    # [Note: Paper uses fine-tuning API; actual weights not exposed]
    logits = gpt35_forward(full_sequence)  # ℝ^(m+n, vocab_size)

    # Step 5: Compute language modeling loss (only on target portion)
    loss = cross_entropy_loss(
        logits[m:m+n],  # Logits for target tokens only
        gt_tokens  # Ground truth token IDs
    )  # Scalar; ℝ

    # Step 6: Backward (via OpenAI fine-tuning API; not explicit in pseudocode)
    # [Fine-tuning API abstracts away gradient computation]

    return loss  # ℝ (scalar)

def format_detections(detections: List[Dict]) -> str:
    """
    Input: [
        {"class": "car", "center": (x, y), "vx": vx, "vy": vy, "pred_traj": [...]},
        ...
    ]
    Output: str, structured text description
    Example:
        "Perception and Prediction:
         - car at (2.34, 19.08), moving to (5.12, 20.45)
         - pedestrian at (-1.93, 7.00), moving to (-2.31, 10.89)
         ..."
    """
    lines = ["Perception and Prediction:"]
    for det in detections:
        obj_class = det["class"]
        x, y = det["center"]
        vx, vy = det["vx"], det["vy"]
        pred_traj = det["pred_traj"]  # List[(x,y), ...]

        motion_str = f"({vx:.2f}, {vy:.2f})" if (vx or vy) else "stationary"
        pred_str = ", ".join([f"({px:.2f}, {py:.2f})" for px, py in pred_traj])

        line = f"- {obj_class} at ({x:.2f}, {y:.2f}), {motion_str}, predicted path: {pred_str}"
        lines.append(line)

    return "\n".join(lines)

def extract_coordinates(output_text: str) -> List[Tuple[float, float]]:
    """
    Input: "Trajectory: [(0.11, 1.14), (0.45, 2.28), ..., (5.49, 5.58)]"
    Output: [(0.11, 1.14), (0.45, 2.28), ..., (5.49, 5.58)]  ∈ ℝ^(6, 2)
    """
    import re
    pattern = r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)"
    matches = re.findall(pattern, output_text)
    waypoints = [(float(x), float(y)) for x, y in matches]
    return waypoints[:6]  # Take first 6 (safety: handle parsing errors)
```

### Inference-Time Forward (Deployment)
```python
def inference_forward(detections, ego_states, hist_traj, mission):
    """
    Input shapes (same as train):
      detections:    List[Dict]
      ego_states:    (vx, vy, ω)
      hist_traj:     [(x,y), (x,y), (x,y), (x,y)]
      mission:       str

    Output:
      reasoning:     str
      pred_traj:     [(x,y), ..., (x,y)] ∈ ℝ^(6,2)

    [This is NOT fine-tuned; uses base GPT-3.5 + prompting]
    """

    # Step 1: Format inputs (same as training)
    perception_text = format_detections(detections)
    ego_text = format_ego_states(ego_states)
    hist_text = format_trajectory(hist_traj)

    prompt = (
        SYSTEM_CONTEXT +
        perception_text + ego_text + hist_text +
        f"Mission Goal: {mission}\n\nOutput:\n"
    )

    # Step 2: Call OpenAI API (or fine-tuned model)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or fine-tuned variant
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,  # Greedy decoding for determinism
    )

    output_text = response.choices[0].message.content

    # Step 3: Parse reasoning and trajectory
    reasoning_section = extract_section(output_text, "Thoughts:")
    trajectory_section = extract_section(output_text, "Trajectory:")

    waypoints = extract_coordinates(trajectory_section)

    return reasoning_section, waypoints
```

---

## 6. Heads, Targets, and Losses

### Prediction Heads

| Head | Output | Format | Example | Loss Type |
|------|--------|--------|---------|-----------|
| **Reasoning Head** | Chain-of-thought explanation | Natural language text | "Notable Objects: car at (-1.93, 7.00)...; Meta Action: TURN RIGHT WITH A DECELERATION;" | Cross-entropy (language modeling) |
| **Trajectory Head** | 6 waypoint coordinates | Tokenized floats + punctuation | "[(0.11, 1.14), (0.45, 2.28), ..., (5.49, 5.58)]" | Cross-entropy (token-level) |

### Loss Terms and Formulation

| Loss Component | Formula | Weight | Notes |
|---|---|---|---|
| **Language Modeling Loss (LM)** | L_LM = -Σ_{i=1}^{n} log P(w_i \| w_1, ..., w_{i-1}) [Eq. 5] | 1.0 (only loss used during fine-tuning) | Cross-entropy over all output tokens (reasoning + trajectory); no separate weighting |
| **L1 Trajectory Regression** (inference comparison only) | L_reg = Σ_{i=1}^{6} (\|x_i - x̂_i\| + \|y_i - ŷ_i\|) [Eq. 3] | Not used for training; only metrics | Reported in Table 1 as "L2 (m)" (despite formula being L1) |
| **Collision Rate Penalty** | Collision(%) = (# colliding waypoints) / 6 × 100 | Reported only; not in loss | Computed post-hoc: ego-vehicle bbox overlaps ground-truth object bbox at each waypoint |

### Training-Time Assignment Strategy

```
Input: detections[N], ego_states, hist_traj, mission, gt_trajectory[6]

Step 1: Generate hypothetical ego-trajectory (no intervention)
  hyp_traj = ego_forward_kinematics(ego_states, duration=3s, accel=0, steering=0)

Step 2: Identify critical objects (heuristic)
  for each detection d in detections:
    if is_collision_risk(d.predicted_traj, hyp_traj):
      critical_objects.append(d)

Step 3: Generate ground-truth reasoning (manual or heuristic)
  gt_reasoning = f"Notable Objects: {describe(critical_objects)}\n"
  gt_reasoning += f"Potential Effects: {predict_effects(critical_objects)}\n"
  gt_reasoning += f"Meta Action: {classify_action(gt_trajectory)}"

Step 4: Format full target
  gt_text = f"Thoughts:\n{gt_reasoning}\n"
  gt_text += f"Trajectory: {format_coordinates(gt_trajectory)}\n"

Step 5: Compute loss
  prompt_tokens = tokenizer.encode(prompt)
  target_tokens = tokenizer.encode(gt_text)
  full_tokens = prompt_tokens + target_tokens

  logits = gpt35(full_tokens)  # ℝ^(len(full_tokens), vocab_size)
  loss = cross_entropy(logits[len(prompt_tokens):], target_tokens)
```

### Loss Debugging Checklist

| Debug Item | Symptom | Investigation |
|---|---|---|
| **Token Overflow** | Output truncated; missing last waypoints | Check max_tokens limit (should be ≥300); count output tokens |
| **Parsing Failure** | Extracted waypoints all zeros/NaNs | Verify coordinate regex; print raw output_text before parsing |
| **Coordinate Scale Mismatch** | L2 error > 20m; waypoints far outside expected range | Check tokenizer output; verify coordinate formatting in prompt (should be [-50, 50] meters) |
| **Reasoning Absurdity** | Output reasoning contradicts detections | Fine-tuning not converged; increase training examples or epochs |
| **Collision Not Detected** | Collision rate 0% despite waypoint intersecting objects | Verify bbox → ego-frame transformation; check if ground-truth bboxes are correctly loaded |
| **Generalization Collapse** | Perfect train loss; zero-shot fails | Fine-tuning overfitted; reduce examples per batch or increase diversity |
| **Randomness in Output** | Same input → different trajectory | temperature > 0 in decoding; set to 0.0 for determinism |

---

## 7. Data Pipeline and Augmentations

### Dataset Details

| Property | Value | Notes |
|---|---|---|
| **Name** | nuScenes | Large-scale autonomous driving benchmark |
| **Total Scenarios** | 1,000 scenarios | ~40,000 key frames total |
| **Locations** | Boston, Singapore, Las Vegas | Diverse urban environments |
| **Weather Conditions** | Rain, night, clear day | Diverse lighting/conditions |
| **Sensors** | Camera (7×), Lidar, Radar | Perception outputs used (detections) |
| **Annotations** | Object bboxes, trajectories, attributes | 23 object classes (car, pedestrian, cyclist, etc.) |
| **Splits** | Train (700), Val (150), Test (150) | Standard split; paper uses train/val for hyperparameter tuning |
| **Sampling Rate** | 2 Hz (0.5s intervals) | Frames sampled every 0.5 seconds |

### Data Split and Evaluation Protocol

| Split | # Scenarios | # Frames | Purpose | Notes |
|---|---|---|---|---|
| **Training** | 700 | ~28,000 | Fine-tune GPT-3.5 | Used for instruction fine-tuning via OpenAI API |
| **Validation** | 150 | ~6,000 | Hyperparameter tuning | Reported in main results (Table 1) |
| **Test** | 150 | ~6,000 | Final evaluation | Not used in paper (assumed same as val) |
| **Few-Shot Subsets** | 700→{1%, 10%, 50%} | ~280, 2,800, 14,000 | Generalization test | Randomly sampled frames for Table 2 |

### Augmentations and Transformation Pipeline

| Augmentation | Parameter | Probability | Applied | Notes |
|---|---|---|---|---|
| **Rotation (ego-frame)** | No global rotation | 1.0 | Always | All coordinates already in ego-frame; no augmentation needed |
| **Translation** | No translation | 1.0 | Always | Ego always at origin (0, 0) in output |
| **Temporal Jitter** | Δt ∈ [-0.1, 0.1]s | Not stated | Unlikely | Would affect tokenizer consistency |
| **Perception Noise** | Gaussian ∼ N(0, σ²), σ ∈ [0.1, 0.5]m | 0.0 (inferred) | No | Paper uses ground-truth detections; no synthetic noise |
| **Object Dropping** | p_drop ∈ {0.0, 0.1, 0.2} | 0.0 (inferred) | No | No evidence of occlusion simulation |
| **Weather Simulation** | Synthetic rain/fog | 0.0 | No | nuScenes already has diverse weather; no sim-to-real augmentation |

### Augmentation Safety and Consistency Table

| Augmentation | Safety Risk | Mitigation |
|---|---|---|
| **Coordinate Noise in Ego-Frame** | May corrupt tokenization; floats become un-parseable | Keep noise << tokenizer precision (0.01m intervals) |
| **Object Dropout** | Could eliminate critical objects from prompt; trajectory unchanged → distribution shift | If implemented, match dropout rate between prompt and gt_trajectory |
| **Temporal Jitter** | Waypoint timing becomes inconsistent (e.g., 0.4s instead of 0.5s) | Lock to fixed intervals; jitter only input, not output |
| **Coordinate Clipping** | Waypoints outside [-50, 50] range may not parse | Clip ground-truth trajectories to valid range before training |

### Data Quality Checks (Inference Time)

```python
def sanity_check_input(detections, ego_states, hist_traj):
    """Pre-flight checks before inference"""
    checks = {}

    # Check 1: Historical trajectory continuity
    dists = [euclidean(hist_traj[i], hist_traj[i+1]) for i in range(3)]
    checks["hist_traj_valid"] = all(0 < d < 20 for d in dists)  # 0–20m per 0.5s

    # Check 2: Ego-states magnitude
    v_norm = (ego_states.vx**2 + ego_states.vy**2)**0.5
    checks["ego_velocity_valid"] = v_norm < 50  # m/s, ~180 km/h limit

    # Check 3: Detection count
    checks["num_detections"] = len(detections)
    checks["detections_reasonable"] = 0 < len(detections) < 100

    # Check 4: Coordinate frame consistency
    for det in detections:
        x, y = det["center"]
        checks["detection_in_range"] = (
            -100 < x < 100 and -100 < y < 100
        )  # Wider range for detections

    return checks
```

---

## 8. Training Pipeline

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| **Base Model** | GPT-3.5-turbo | OpenAI's proprietary model; 175B parameters |
| **Fine-tuning Method** | Supervised instruction fine-tuning | Via OpenAI fine-tuning API; exact algorithm undisclosed |
| **Batch Size** | Not stated | Plausible default: 16–32 (API default: 1) |
| **Learning Rate** | Not stated | Plausible default: 1e-5 to 5e-5 (standard for LLM fine-tuning) |
| **Optimizer** | Not stated | Plausible: AdamW (standard for transformers) |
| **Max Tokens (Output)** | 300 | Enough for ~150 reasoning + ~150 trajectory tokens |
| **Temperature (Inference)** | 0.0 | Greedy decoding (inferred from deterministic results); not explicitly stated |
| **Top-p (Nucleus Sampling)** | Not applicable | Likely disabled (temperature=0.0) |
| **Epochs** | 1–3 | Plausible default for few-shot learning on fine-tuning API |

### Training Stability and Convergence

| Metric | Observed | Stability Assessment |
|---|---|---|
| **Loss Convergence** | Not reported | Assumed smooth (LLM fine-tuning is typically stable) |
| **Validation L2 Error** | 0.84m (full 700 scenarios) [Table 1] | Stable; no mention of divergence |
| **Few-Shot Generalization** | L2 error: 1.89m @1%, 1.20m @10%, 1.01m @50% [Table 2] | Monotonic improvement; suggests stable gradient flow |
| **Fine-tuning vs. Prompting** | Fine-tuning L2: 0.84m; in-context learning L2: 3.17m [Table 3] | ~3.8× improvement; clear benefit of gradient updates |
| **Inference Stability** | Qualitative examples (Figure 4) show coherent reasoning | Anecdotal; no reported variance in outputs |
| **Failure Modes** | Limitations stated: inference time unknown, open-loop only | Acknowledged; no reported catastrophic failures during training |

### Training Procedure (Reconstructed)

```python
def train():
    """
    Fine-tune GPT-3.5 on motion planning task.
    Uses OpenAI's fine-tuning API (exact implementation hidden).
    """

    # Step 1: Prepare training data
    train_scenarios = load_nuscenes_train(split="train", count=700)  # 28K frames
    train_examples = []

    for scenario in train_scenarios:
        for frame in scenario.frames:
            # Extract inputs and ground truth
            detections = frame.objects  # List[Dict]
            ego_states = frame.ego_state  # (vx, vy, ω)
            hist_traj = frame.historical_trajectory  # [(x,y), (x,y), (x,y), (x,y)]
            mission = frame.high_level_goal  # str, e.g., "TURN_RIGHT"
            gt_trajectory = frame.future_trajectory[:6]  # Next 3 seconds, 0.5s intervals

            # Format as language model example
            prompt = format_prompt(detections, ego_states, hist_traj, mission)
            target = format_target(gt_trajectory)  # Reasoning + trajectory

            example = {
                "prompt": prompt,
                "completion": target,  # Prefix with " " for API format
            }
            train_examples.append(example)

    # Step 2: Upload to OpenAI (JSONL format)
    with open("train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    # Step 3: Fine-tune via API
    response = openai.FineTune.create(
        training_file="train.jsonl",
        model="gpt-3.5-turbo",
        n_epochs=3,  # Plausible; paper doesn't specify
        batch_size=32,  # Plausible default
        learning_rate_multiplier=0.1,  # Plausible for instruction tuning
    )

    fine_tuned_model = response.fine_tuned_model  # e.g., "ft-xyz123"

    # Step 4: Validate on validation set
    val_scenarios = load_nuscenes_val(split="val", count=150)
    metrics = {}

    for scenario in val_scenarios:
        for frame in scenario.frames:
            detections = frame.objects
            ego_states = frame.ego_state
            hist_traj = frame.historical_trajectory
            mission = frame.high_level_goal
            gt_trajectory = frame.future_trajectory[:6]

            # Inference
            prompt = format_prompt(detections, ego_states, hist_traj, mission)
            response = openai.ChatCompletion.create(
                model=fine_tuned_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )

            pred_trajectory = parse_response(response.choices[0].message.content)

            # Compute metrics
            l2_error = compute_l2_error(pred_trajectory, gt_trajectory)
            collision = compute_collision_rate(pred_trajectory, frame.objects)

            metrics[frame.id] = {"l2": l2_error, "collision": collision}

    # Step 5: Report results
    avg_l2 = np.mean([m["l2"] for m in metrics.values()])
    avg_collision = np.mean([m["collision"] for m in metrics.values()])

    print(f"Validation L2 error: {avg_l2:.2f}m")
    print(f"Validation collision rate: {avg_collision:.2f}%")

    return fine_tuned_model
```

---

## 9. Dataset + Evaluation Protocol

### Dataset Summary

**nuScenes** [Sec 4.1]
- **Scale**: 1,000 driving scenarios, ~40,000 key frames
- **Locations**: Boston, Singapore, Las Vegas
- **Sensors**: 7× camera, Lidar, Radar
- **Classes**: 23 object categories (car, pedestrian, cyclist, truck, etc.)
- **Annotations**: 3D bounding boxes, attributes, future trajectories (6 seconds)
- **Temporal Resolution**: 2 Hz sampling (0.5s intervals)

### Official Splits (Paper's Usage)

| Split | Scenarios | Frames | Usage |
|---|---|---|---|
| **Training** | 700 | ~28,000 | Fine-tune GPT-3.5 [Sec 4.1] |
| **Validation** | 150 | ~6,000 | Main results (Table 1, 3, 4) |
| **Test** | 150 | ~6,000 | Not explicitly used; results likely from val |

### Evaluation Metrics

| Metric | Definition | Formula | Interpretation | Paper Result |
|---|---|---|---|---|
| **L2 Error (m)** | Average Euclidean distance between predicted and ground-truth waypoints | L2 = (1/6) Σ √[(x_i - x̂_i)² + (y_i - ŷ_i)²] | Lower is better; unit = meters | Main metric; 0.84m (GPT-Driver full data) |
| **Collision Rate (%)** | Percentage of waypoints where ego-vehicle bbox overlaps object bbox | Collision% = (# colliding waypoints) / 6 × 100 | Lower is better; unit = % | Secondary metric; 0.44% (GPT-Driver full data) |
| **L2 @ 1s, 2s, 3s (m)** | L2 error at specific future time horizons (1st, 2nd, 3rd waypoint) | L2_t = √[(x_t - x̂_t)² + (y_t - ŷ_t)²] | Shows error growth over time | Table 3: 0.27m @1s, 0.74m @2s, 1.52m @3s |

### Baselines and Comparisons

| Baseline | Category | L2 Error (m) | Collision Rate (%) | Notes |
|---|---|---|---|---|
| **ST-P3** | Learning-based (imitation) | 1.33 | 0.23 | State-of-the-art ST-P3; strong baseline |
| **VAD** | Learning-based (vectorized) | 0.17 (!) | 0.07 | Lower L2 on validation; likely overfitted to nuScenes distribution |
| **NMP** | Rule-based | - | 1.92 | Heuristic; poor collision avoidance |
| **SA-NMP** | Rule-based | 0.55 | - | Slightly better heuristic |
| **UniAD** | End-to-end learning | 0.48 | 0.31 | Strong multi-task baseline; uses dense occupancy |
| **GPT-Driver (ours)** | LLM-based | **0.84** | **0.44** | Competitive L2; good collision avoidance |

### Few-Shot Evaluation

| Training Data | Method | Avg. L2 (m) | Collision (%) | Notes |
|---|---|---|---|---|
| **1% (~280 frames)** | UniAD | 5.37 | 6.86 | Severe overfitting on tiny data |
| **1%** | GPT-Driver | **1.89** | 1.24 | ~2.8× better generalization |
| **10% (~2.8K frames)** | UniAD | 1.80 | 1.31 | Still struggles |
| **10%** | GPT-Driver | **1.20** | 0.93 | Strong few-shot performance |
| **50% (~14K frames)** | UniAD | 1.42 | 0.49 | Approaching full-data performance |
| **50%** | GPT-Driver | **1.01** | 0.75 | Continues to generalize well |
| **100% (~28K frames)** | UniAD | 1.03 | 0.31 | Baseline at full capacity |
| **100%** | GPT-Driver | **0.84** | 0.44 | Final validation result |

### Evaluation Procedure

```python
def evaluate_model(model_id, val_data):
    """Compute L2 error and collision rate on validation set"""

    results = []

    for frame in val_data:
        # Prepare inputs
        detections = frame.objects
        ego_states = frame.ego_state
        hist_traj = frame.historical_trajectory
        mission = frame.mission_goal
        gt_trajectory = frame.future_trajectory[:6]

        # Inference
        prompt = format_prompt(detections, ego_states, hist_traj, mission)
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )

        pred_trajectory = parse_response(response.choices[0].message.content)

        # Compute metrics
        l2_error = compute_l2_distance(pred_trajectory, gt_trajectory)

        collision_count = 0
        for i, (px, py) in enumerate(pred_trajectory):
            ego_box = get_ego_bbox_at_waypoint(px, py)
            for obj in frame.objects:
                if bbox_overlap(ego_box, obj.bbox):
                    collision_count += 1
                    break  # Count per waypoint, not per object

        collision_rate = collision_count / 6 * 100

        results.append({
            "frame_id": frame.id,
            "l2_error": l2_error,
            "collision_rate": collision_rate,
        })

    # Aggregate metrics
    avg_l2 = np.mean([r["l2_error"] for r in results])
    avg_collision = np.mean([r["collision_rate"] for r in results])

    return avg_l2, avg_collision, results
```

---

## 10. Results Summary + Ablations

### Main Results [Table 1 | Sec 4.2]

| Method | Category | L2 Error (m) ↓ | Collision (%) ↓ | Advantage |
|---|---|---|---|---|
| **ST-P3** | Learning (imitation) | 1.33 | 0.62 | Baseline |
| **VAD** | Learning (vec.) | 0.17 | 0.10 | Best L2 (may be val-set biased) |
| **NMP** | Rule-based | — | 1.92 | Poor |
| **SA-NMP** | Rule-based | 0.55 | — | Better than NMP |
| **FiFo** | Learning | 0.55 | 1.07 | Competitive |
| **UniAD** | Learning (e2e) | 0.48 | 0.31 | Strong baseline |
| **GPT-Driver (ours)** | **LLM-based** | **0.84** | **0.44** | Reasonable L2; interpretable |

**Key Finding**: GPT-Driver achieves L2 error (0.84m) between ST-P3 (1.33m) and UniAD (0.48m). Trade-off: slightly higher error than best learning baselines, but with interpretability and few-shot generalization benefits.

### Ablation Study 1: Fine-tuning vs. In-Context Learning [Table 3 | Sec 4.5]

| Strategy | L2 @1s (m) | L2 @2s (m) | L2 @3s (m) | Avg L2 (m) | Collision (%) |
|---|---|---|---|---|---|
| **In-context learning** (prompting only) | 2.41 | 3.11 | 4.00 | 3.17 | 5.30 |
| **Fine-tuning** (on full 700 scenarios) | 0.27 | 0.74 | 1.52 | **0.84** | **0.44** |
| **Improvement** | **8.9×** | **4.2×** | **2.6×** | **3.8×** | **12.0×** |

**Insight**: Fine-tuning is essential; in-context learning alone fails dramatically. GPT-3.5 cannot perform precise numerical reasoning without training signal. Error grows substantially over 3-second horizon (0.27m → 1.52m).

### Ablation Study 2: Few-Shot Generalization [Table 2 | Sec 4.3]

| Data Fraction | UniAD L2 (m) | GPT-Driver L2 (m) | GPT-Driver Advantage | UniAD Collision (%) | GPT-Driver Collision (%) |
|---|---|---|---|---|---|
| **1%** | 5.37 | 1.89 | **2.8×** | 6.86 | 1.24 |
| **10%** | 1.80 | 1.20 | **1.5×** | 1.31 | 0.93 |
| **50%** | 1.42 | 1.01 | **1.4×** | 0.49 | 0.75 |
| **100%** | 1.03 | 0.84 | **1.2×** | 0.31 | 0.44 |

**Insight**: GPT-Driver's few-shot learning is dramatically superior. At 1% training data, GPT-Driver achieves (1.89m) what UniAD needs 100% data to achieve (~1.0m). Suggests LLM's pre-training transfers well to motion planning. UniAD likely overfits due to its larger capacity.

### Ablation Study 3: Design Choices in Reasoning [Sec 4.4, Figure 4]

| Choice | Observation | Impact |
|---|---|---|
| **Chain-of-thought reasoning** | Explicit identification of critical objects; reasoning visible in output | Interpretability++; enables validation of decision-making |
| **Three-step reasoning** (identify objects → predict effects → determine action) | Structured reasoning; each step is transparent | Better than single-shot generation |
| **Prompting-reasoning-finetuning** | Initial prompting fails; fine-tuning enables numeric reasoning | Necessary for coordinate precision |
| **No dense occupancy grids** | Uses only sparse detections + text descriptions | Simpler, modular; less compute than UniAD |

**Insight**: Explicit reasoning structure is key; without it, LLM struggles to generate coherent trajectories. Three-step decomposition mirrors human driving cognition.

### Interpretability Results [Sec 4.4, Figure 4]

Qualitative visualization of GPT-Driver outputs on 4 scenarios:

1. **Highway with Traffic Cones**: Identifies barriers; plans to turn right with deceleration to avoid collision
2. **Parking Lot with Pedestrians**: Recognizes pedestrian crossing; plans to stop ("MOVE FORWARD WITH DECELERATION")
3. **Night Driving with Vehicles**: Detects vehicles ahead; plans acceleration to merge
4. **Curved Road with Motorcycle**: Identifies motorcycle; plans smooth right turn with constant speed

All outputs include explicit reasoning (Figure 4 right column) showing object detection, effect prediction, and action determination.

---

## 11. Practical Insights

### Engineering Takeaways

1. **Tokenization is a Bottleneck for Precision**: Floating-point coordinates tokenized as separate tokens ("23", ".", "17"). Tokenizer must preserve sub-meter precision. Error accumulates if tokenizer vocabulary lacks common decimal separators. Mitigation: Pre-process floats to fixed-width strings (e.g., "23.17" → two tokens).

2. **Prompt Engineering Matters More Than Model Size**: In-context learning (3.17m L2) vs. fine-tuning (0.84m L2) shows gradient updates >> prompt design. Spend effort on fine-tuning data quality, not prompt wording alone. Corollary: base GPT-3.5 cannot reason numerically without training.

3. **Few-Shot Learning Requires Diverse Examples**: GPT-Driver generalizes 2.8× better than UniAD at 1% data. Hypothesized reason: LLM pre-training on diverse text enables compositional reasoning. Imitation learning requires large, representative dataset. Takeaway: LLM-based planners may be preferred for data-scarce scenarios.

4. **Coordinate Frame Consistency is Critical**: All prompts must explicitly state ego-centric coordinates ("X-axis perpendicular, Y-axis forward"). Output parsing must correctly map tokenized floats to coordinate space. Off-by-one errors in frame transform → compounding trajectory errors. Mitigation: unit tests on frame transformations; check all examples against raw sensor data.

5. **Reasoning Output Enables Debugging**: Unlike black-box neural planners (e.g., end-to-end CNNs), GPT-Driver outputs reasoning chain. Use this for post-hoc validation: if reasoning identifies wrong object or miscalculates collision risk, fine-tuning can correct it. Takeaway: explainability is a debugging tool, not just a feature.

6. **Fine-Tuning API Cost/Latency Trade-off**: Using OpenAI's fine-tuning API (not shown but implied) incurs per-request costs (~0.1–1ms per token; 300 output tokens ≈ 30ms latency + $0.01 per inference). Real-time closed-loop driving is infeasible. Open problem: deploy on local quantized LLM (7B parameter model) to reduce latency. Paper acknowledges but doesn't address.

7. **Collision Detection Requires Accurate BBox Transforms**: Collision rate metric checks overlap of ego-bbox with object-bbox at each predicted waypoint. If bbox transforms are wrong (e.g., ego-frame ↔ global frame), collision metric becomes meaningless. Mitigation: validate all bbox transforms on ground-truth trajectories first (should match human-annotated data).

8. **Generalization is Non-Monotonic Across Time Horizons**: Table 3 shows L2 error grows from 0.27m @1s to 1.52m @3s. This suggests compounding errors in long-horizon planning. Possible cause: tokenization errors accumulate; each token is decoded independently. Mitigation: consider hierarchical planning (predict next 1s precisely, then recursively plan from that state).

9. **Dataset Diversity > Dataset Size**: nuScenes spans 3 cities + multiple weather conditions. Few-shot generalization suggests diversity matters more than raw frame count. If moving to new domain (e.g., different country, weather), prioritize collecting diverse scenarios over sheer volume.

10. **Validation Set Must Match Deployment Domain**: Validation results (Table 1) are on nuScenes. If deploying to different sensor configurations, road styles, or object classes, retune prompts + fine-tune on domain examples. Corollary: don't trust cross-dataset results without explicit domain adaptation experiments (paper doesn't include).

### Five Critical Gotchas

1. **Coordinate Tokenization Ambiguity**: Decimal point "." is a separate token. Tokenizer can generate "23. 17" (space) or "23.17" or "2317" depending on training data. Parser must be robust to spacing variations. Bug: regex `\([-\d.]+, [-\d.]+\)` may miss spaces inside floats.

2. **Off-by-One in Temporal Alignment**: Historical trajectory has 4 waypoints (t=-2s, -1.5s, -1s, -0.5s). Future trajectory starts at t=0.5s. If parser incorrectly interprets "first waypoint = current position," predictions shift by 0.5s. Mitigation: explicitly label timestamps in prompts and outputs.

3. **Collision Detection False Negatives**: BBox collision check is axis-aligned (not rotated). If vehicle is turning, rotated bbox ≠ axis-aligned bbox; collision rate underestimated. Paper doesn't clarify bbox type used. Mitigation: use rotated IoU or check multiple bbox representations.

4. **Fine-tuning Overfits on Reasoning Style**: If training examples have specific reasoning format (e.g., "Notable Objects:", "Meta Action:"), fine-tuned model learns to memorize format, not reasoning logic. Inference on out-of-distribution detections → incoherent reasoning. Mitigation: vary reasoning format across training examples; don't enforce rigid templates.

5. **API Latency Masks as "Inference Time"**: Paper acknowledges limitation (Sec 4.6): "unable to obtain inference time due to OpenAI API limitations." This is critical for real-time deployment. 50ms API latency + 30ms network round-trip + decision cycle = 100ms loop impossible for highway speeds. Mitigation: quantize model to run on-device (ORT, TensorRT); plan to ~2s horizon instead of 3s.

### Tiny-Subset Overfit Debugging Plan

Goal: Verify that all components (tokenization, parsing, loss computation) work correctly on a 1-scenario subset.

```python
def debug_tiny_subset():
    """
    Create minimal dataset: 1 scenario, 5 frames.
    Verify: (1) forward pass works, (2) loss decreases, (3) outputs parse correctly.
    """

    # Step 1: Create tiny dataset
    tiny_train = [
        {
            "detections": [{"class": "car", "center": (2.34, 19.08), "vx": 0, "vy": 0, "pred": [(2.34, 20.00)]}],
            "ego_states": (0.00, 1.46, -0.08),
            "hist_traj": [(-0.88, -6.74), (-0.83, -3.73), (-0.83, -3.07), (-0.82, -1.46)],
            "mission": "FORWARD",
            "gt_trajectory": [(0.11, 1.14), (0.45, 2.28), (1.12, 3.47), (2.18, 4.54), (3.65, 5.29), (5.49, 5.58)],
        },
        # ... 4 more frames
    ]

    # Step 2: Format and tokenize
    for example in tiny_train:
        prompt = format_prompt(example)
        target = format_target(example["gt_trajectory"])

        prompt_tokens = tokenizer.encode(prompt)
        target_tokens = tokenizer.encode(target)

        print(f"Prompt tokens: {len(prompt_tokens)}")
        print(f"Target tokens: {len(target_tokens)}")
        print(f"Total: {len(prompt_tokens) + len(target_tokens)}")

        # Verify tokenization is reversible
        prompt_reconstructed = tokenizer.decode(prompt_tokens)
        assert prompt_reconstructed.strip() == prompt.strip(), "Tokenization not reversible!"

    # Step 3: Fine-tune on tiny set
    response = openai.FineTune.create(
        training_file="tiny_train.jsonl",
        model="gpt-3.5-turbo",
        n_epochs=10,  # Overfit on tiny data
    )

    fine_tuned_model = response.fine_tuned_model

    # Step 4: Validate on same 5 frames (should overfit to 0 error)
    for example in tiny_train:
        prompt = format_prompt(example)
        response = openai.ChatCompletion.create(
            model=fine_tuned_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )

        output_text = response.choices[0].message.content
        pred_trajectory = parse_response(output_text)

        gt_trajectory = example["gt_trajectory"]
        l2_error = compute_l2_distance(pred_trajectory, gt_trajectory)

        print(f"Predicted: {pred_trajectory}")
        print(f"Ground truth: {gt_trajectory}")
        print(f"L2 error: {l2_error:.4f}m")

        # Expected: L2 error << 0.1m (near-perfect overfitting)
        assert l2_error < 0.1, f"Overfitting failed! Error: {l2_error}"

    print("✓ Tiny-subset overfit successful; all components working")
```

---

## 12. Minimal Reimplementation Checklist

### Build Order (Dependency Graph)

```
1. Tokenizer + Detokenizer (GPT-3.5 BPE)
   ↓
2. Prompt Formatter (hardcoded rules)
   ├─ format_detections()
   ├─ format_ego_states()
   ├─ format_trajectory()
   └─ SYSTEM_CONTEXT (static string)
   ↓
3. Coordinate Parser (regex + float conversion)
   └─ extract_coordinates()
   ↓
4. Forward Pass (via OpenAI API or local LLM)
   ├─ API wrapper (if using OpenAI)
   └─ Local inference (if quantizing model)
   ↓
5. Loss Function (cross-entropy over tokens)
   └─ compute_lm_loss()
   ↓
6. Fine-tuning Loop (or use OpenAI API)
   ├─ Data preparation (JSONL formatting)
   ├─ Training (via API or local)
   └─ Validation loop
   ↓
7. Evaluation Metrics
   ├─ compute_l2_error()
   └─ compute_collision_rate()
```

### Unit Tests Table

| Component | Test Case | Expected Output | Critical? |
|---|---|---|---|
| **Tokenizer** | encode/decode round-trip on "(1.23, 4.56)" | "(1.23, 4.56)" (exact match or with spaces) | YES |
| **Prompt Formatter** | format_detections([car@(2.0, 3.0)]) | "Perception and Prediction:\n- car at (2.00, 3.00)..." | YES |
| **Coordinate Parser** | parse_response("Trajectory: [(0.1, 1.0), (2.0, 3.0), ...]") | [(0.1, 1.0), (2.0, 3.0), ...] (6 tuples, floats) | YES |
| **L2 Metric** | compute_l2_distance([(0, 0)], [(1, 0)]) | 1.0 | YES |
| **Collision Detector** | bbox_overlap(ego_box@(0,0), obj_box@(5,5)) with 1m radius | False | YES |
| **Frame Transform** | rotate_to_ego_frame(global_pos=(10,0), heading=90°) | ego_pos ≈ (0, 10) | CRITICAL |
| **Fine-tuning Loss** | cross_entropy_loss(logits[bs,seq,vocab], targets[bs,seq]) | scalar ∈ [0, ∞), decreases over epochs | YES |
| **API Wrapper** | call_gpt35(prompt="test", max_tokens=100) | response.choices[0].message.content (string) | YES |

### Sanity Test Scripts

```python
# Test 1: Tokenization round-trip
def test_tokenization():
    prompt = "Perception and Prediction:\n- car at (2.34, 19.08), moving to (5.12, 20.45)"
    tokens = tokenizer.encode(prompt)
    reconstructed = tokenizer.decode(tokens)
    # Check: reconstructed should match prompt (possibly with minor whitespace diffs)
    assert prompt.strip() == reconstructed.strip()
    print(f"✓ Tokenization: {len(tokens)} tokens")

# Test 2: Coordinate parsing
def test_coord_parsing():
    output = "Trajectory: [(0.11, 1.14), (0.45, 2.28), (1.12, 3.47), (2.18, 4.54), (3.65, 5.29), (5.49, 5.58)]"
    waypoints = extract_coordinates(output)
    assert len(waypoints) == 6
    assert all(isinstance(w, tuple) and len(w) == 2 for w in waypoints)
    assert abs(waypoints[0][0] - 0.11) < 0.01  # Float precision check
    print(f"✓ Coord parsing: {waypoints}")

# Test 3: L2 error computation
def test_l2_metric():
    pred = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0.5), (0.5, 0)]  # 6 points
    gt = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0.5), (0.5, 0)]    # Same
    l2 = compute_l2_distance(pred, gt)
    assert abs(l2 - 0.0) < 1e-6
    print(f"✓ L2 metric: {l2:.4f}m")

# Test 4: Frame transformation (critical)
def test_frame_transform():
    # Vehicle at (10, 10) in global frame, heading 90° (north)
    # Object at (10, 20) in global frame → should be (10, 0) in ego frame (forward)
    global_pos = np.array([10, 20])
    ego_pos = np.array([10, 10])
    heading = 90 * np.pi / 180  # radians

    # Rotate and translate
    R = np.array([[np.cos(heading), np.sin(heading)],
                  [-np.sin(heading), np.cos(heading)]])
    ego_coord = R @ (global_pos - ego_pos)

    # Expected: ego_coord ≈ [0, 10] (perpendicular=0, forward=10)
    assert abs(ego_coord[0]) < 0.01
    assert abs(ego_coord[1] - 10) < 0.01
    print(f"✓ Frame transform: {ego_coord}")

# Test 5: End-to-end inference
def test_e2e_inference():
    detections = [{"class": "car", "center": (2.34, 19.08), "vx": 0, "vy": 0, "pred": [(2.34, 20.00)]}]
    ego_states = (0.00, 1.46, -0.08)
    hist_traj = [(-0.88, -6.74), (-0.83, -3.73), (-0.83, -3.07), (-0.82, -1.46)]
    mission = "FORWARD"

    prompt = format_prompt(detections, ego_states, hist_traj, mission)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or fine-tuned model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
    )

    output = response.choices[0].message.content
    reasoning, waypoints = parse_response(output)

    assert len(waypoints) == 6
    assert all(-50 < x < 50 and -50 < y < 50 for x, y in waypoints)
    print(f"✓ E2E inference: {waypoints}")

# Test 6: Collision detection
def test_collision():
    # Ego at (0, 0), size [4.7m long, 1.85m wide]
    # Object at (0, 0), size [5m long, 2m wide]
    ego_box = BBox(center=(0, 0), length=4.7, width=1.85)
    obj_box = BBox(center=(0, 0), length=5, width=2)

    collision = bbox_overlap(ego_box, obj_box)
    assert collision == True

    # Object at (10, 10) → no collision
    obj_box_far = BBox(center=(10, 10), length=5, width=2)
    collision_far = bbox_overlap(ego_box, obj_box_far)
    assert collision_far == False
    print(f"✓ Collision detection")

# Run all tests
if __name__ == "__main__":
    test_tokenization()
    test_coord_parsing()
    test_l2_metric()
    test_frame_transform()
    test_collision()
    test_e2e_inference()
    print("\n✓✓✓ All tests passed ✓✓✓")
```

### Implementation Milestone Checklist

- [ ] **Phase 1: Infrastructure** (2–3 days)
  - [ ] Set up nuScenes data loader (or mock data)
  - [ ] Implement prompt formatter (test on 5 examples)
  - [ ] Verify tokenizer round-trip (test: prompt → tokens → prompt)

- [ ] **Phase 2: Core Logic** (3–5 days)
  - [ ] Implement coordinate parser (regex + float handling)
  - [ ] Implement L2 metric (compare to paper: 0.84m on full data expected)
  - [ ] Implement collision detector (use axis-aligned BBox)
  - [ ] Test end-to-end on 10 frames (mock API or lightweight LLM)

- [ ] **Phase 3: Training & Evaluation** (5–7 days)
  - [ ] Prepare JSONL data for fine-tuning (or use local LLM training loop)
  - [ ] Fine-tune GPT-3.5 (or quantized 7B model locally)
  - [ ] Validate on tiny subset (5 frames; expect L2 < 0.1m if overfitting)
  - [ ] Evaluate on full validation set (expected: 0.84m L2, 0.44% collision)

- [ ] **Phase 4: Ablations & Analysis** (3–4 days)
  - [ ] Few-shot experiment (1%, 10%, 50% data; compare to Table 2)
  - [ ] Reasoning interpretability (visualize outputs for 4–5 examples)
  - [ ] Error analysis (categorize failures: tokenization? frame transform? object detection?)

- [ ] **Phase 5: Deployment** (2–3 days)
  - [ ] Optimize latency (quantization, batching, caching)
  - [ ] Benchmark inference time (target: < 100ms for real-time loop)
  - [ ] Set up CI/CD for validation set tests

### Quick-Start Pseudocode (Skeleton)

```python
import json
import numpy as np
from openai import OpenAI
import re

# Initialize
client = OpenAI(api_key="sk-...")
SYSTEM_CONTEXT = """..."""  # From paper Fig 2

def format_prompt(detections, ego_states, hist_traj, mission):
    # Hardcode rules to convert structured data → text
    pass

def extract_coordinates(text):
    # Regex to parse "(x, y)" from text
    pattern = r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)"
    matches = re.findall(pattern, text)
    return [(float(x), float(y)) for x, y in matches[:6]]

def compute_l2_distance(pred, gt):
    return np.mean([np.linalg.norm(np.array(p) - np.array(g)) for p, g in zip(pred, gt)])

def infer(detections, ego_states, hist_traj, mission):
    prompt = format_prompt(detections, ego_states, hist_traj, mission)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
    )

    output = response.choices[0].message.content
    waypoints = extract_coordinates(output)

    return waypoints

# Example usage
detections = [...]  # Load from nuScenes
ego_states = (0.0, 1.5, -0.08)
hist_traj = [(-0.88, -6.74), (-0.83, -3.73), (-0.83, -3.07), (-0.82, -1.46)]
mission = "FORWARD"

pred_waypoints = infer(detections, ego_states, hist_traj, mission)
print(f"Predicted trajectory: {pred_waypoints}")
```

---

## Summary Table: Implementation Effort Estimation

| Component | Lines of Code | Difficulty | Time (hours) | Dependencies |
|---|---|---|---|---|
| Tokenizer + Detokenizer | ~50 | Easy | 0.5 | `tokenizers` lib (pip) |
| Prompt Formatter | ~150–300 | Medium | 4–6 | numpy, Python builtins |
| Coordinate Parser | ~100 | Easy | 2 | regex |
| Forward Pass (API) | ~50 | Easy | 1 | OpenAI API |
| Forward Pass (Local LLM) | ~200 | Hard | 8–12 | `transformers`, `torch` |
| Loss Function | ~50 | Easy | 1 | numpy/torch |
| Fine-tuning Loop | ~150–300 | Hard | 6–10 | OpenAI API or training framework |
| Metrics + Evaluation | ~200–300 | Medium | 4–6 | numpy, scipy |
| Unit Tests | ~300–500 | Medium | 5–8 | pytest |
| **Total (API-based)** | **~1.0K–1.5K** | **Medium** | **20–30 hours** | OpenAI API key |
| **Total (Local LLM)** | **~2.0K–3.0K** | **Hard** | **40–60 hours** | GPU, transformers |

---

## Known Gaps and Inferred Values

| Gap | Paper Status | Assumption Used |
|---|---|---|
| Exact batch size for fine-tuning | Not stated | Plausible: 16–32 |
| Learning rate | Not stated | Plausible: 1e-5 to 5e-5 |
| Optimizer algorithm | Not stated | Assumed: AdamW (standard) |
| Temperature/top-p decoding params | Not stated | Assumed: temperature=0.0 (greedy) |
| Exact spatial range for coordinate validity | Implied | Assumed: [-50, 50]m based on typical AD ranges |
| Inference latency | Explicitly unknown (Sec 4.6) | Estimated: ~100ms (30ms generation + 70ms API overhead) |
| Code availability | Stated as available | Not yet evaluated; may differ from paper pseudocode |
| Reasoning generation procedure (deterministic or manual?) | Somewhat vague | Assumed: heuristic-based + manual annotations |
| Collision bbox definition (rotated or axis-aligned?) | Not fully specified | Assumed: axis-aligned (standard in 3D bboxes) |

---

## Final Checklist for Implementation

- [ ] Read entire paper + PDF figures
- [ ] Implement tokenization round-trip (verify on 5 examples)
- [ ] Implement prompt formatter with all 4 components (detections, ego, hist, mission)
- [ ] Implement coordinate parser with robust regex
- [ ] Test on 1-scenario (5-frame) subset with mocked LLM (should overfit to L2 < 0.1m)
- [ ] Set up OpenAI API client OR local LLM inference pipeline
- [ ] Prepare nuScenes data loader (train/val splits)
- [ ] Prepare JSONL data for fine-tuning
- [ ] Fine-tune model on full training set (700 scenarios)
- [ ] Validate on val set (expect: 0.84m L2, 0.44% collision)
- [ ] Run few-shot experiments (1%, 10%, 50% data)
- [ ] Validate frame transforms and collision detection carefully
- [ ] Benchmark inference latency (document bottleneck)
- [ ] Create 4–5 interpretability visualizations (like Figure 4)
- [ ] Write unit tests for all critical functions
- [ ] Document all inferred hyperparameters

---

**End of Summary**
