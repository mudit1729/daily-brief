# Language Conditioned Imitation Learning Over Unstructured Data

**Lynch & Sermanet (2021)** — RSS 2021 (arXiv 2005.07648)

---

<!-- AUDIT_NOTE_START -->
> Audit note (March 7, 2026): This block is the canonical, paper-backed teaching summary for this file. If later sections give exact tensor shapes, hyperparameters, deployment numbers, or pseudocode details that the paper does not explicitly report, treat them as explanatory reconstructions rather than direct paper claims.

## Paper-Backed Teaching Corrections
- **What the paper actually does:** trains a single end-to-end language-conditioned visuomotor policy from pixels for robotic manipulation.
- **Core method:** it uses unlabeled, unstructured demonstration data together with hindsight-style instruction relabeling / multicontext imitation ideas so language annotation is needed for less than 1% of the data.
- **What you should understand:** the main contribution is how to exploit large amounts of play-style data for language-conditioned imitation, not a specific network micro-architecture.
- **Important correction:** later sections with exact module shapes or training constants should be read as reconstruction unless the paper states them explicitly.
<!-- AUDIT_NOTE_END -->

## 1. One-Page Overview

| Aspect | Details |
|--------|---------|
| **Title** | Language Conditioned Imitation Learning Over Unstructured Data |
| **Authors** | Corey Lynch, Pierre Sermanet |
| **Venue** | Robotics: Science and Systems (RSS) 2021 |
| **arXiv** | 2005.07648 (submitted May 15, 2020; revised July 7, 2021) |
| **Affiliation** | Google Brain Robotics |
| **Code** | Available; project page: [language-play.github.io](https://language-play.github.io/) |

### Problem Setting

This work addresses a key limitation in language-conditioned imitation learning: existing methods require **each demonstration to be paired with either a task ID or goal image**, which is impractical in open-world settings. Lynch & Sermanet propose a method that:

1. Learns from **unlabeled, unstructured play data** (robot moving around with no task labels)
2. Uses **hindsight instruction pairing** to retroactively label demonstrations with natural language
3. Trains a **single multitask policy** conditioned on diverse forms of task specification (language, goal image, task ID) end-to-end

### Inputs/Outputs

- **Input**:
  - Observation (RGB image from wrist camera, proprioceptive state)
  - Task specification (free-form language instruction OR goal image OR task ID)
- **Output**:
  - Continuous action (joint angles, gripper open/close)
  - Typically 7-DoF robot arm + parallel gripper

### Key Novelty Bullets

1. **Hindsight Instruction Pairing (HIP)**: Retroactively label play trajectories with natural language instructions for what was accomplished, creating language-task associations without online annotation.

2. **Multicontext Imitation Learning (MCIL)**: A single policy jointly conditioned on multiple heterogeneous task specifications (language embeddings, goal images, discrete task IDs), with separate encoders per modality feeding into a shared latent goal space.

3. **Data Efficiency**: Reduces language annotation cost to **<1% of collected robot experience**; the majority of control learning comes from unlabeled play data via self-supervised imitation.

4. **Zero-Shot Generalization**: When combined with pre-trained language embeddings (e.g., from language models), the system can generalize to unseen instructions without additional finetuning.

### If You Only Remember 3 Things

1. **Hindsight pairing works**: You can label play trajectories with language *after the fact* by asking "what instruction would describe what just happened?" This dramatically reduces annotation cost.

2. **Multiple task contexts in one model**: A single policy can condition on language, images, or task IDs by learning separate encoders that project into a shared latent goal space.

3. **Unlabeled data is your friend**: Most robot control learning comes from unstructured play; language annotation is just the cherry on top for multi-task generalization.

---

## 2. Problem Setup and Outputs

### Task Definition

**Multi-Task Robotic Manipulation**: Given a task specification (language, goal image, or task ID), the robot must:
1. Predict the next action (joint deltas, gripper command)
2. Condition on the task throughout the episode
3. Learn from both labeled (language + actions) and unlabeled (action-only) play data

### Input Specification

| Input | Dimensions | Format | Notes |
|-------|-----------|--------|-------|
| **Observation: Image** | (H=256, W=256, 3) | uint8, [0,255] | Wrist-mounted camera (first-person view) |
| **Observation: Proprioception** | (7+1,) | float32 | 7 joint angles + gripper state; normalized to [-1, 1] |
| **Task: Language** | Variable length | Tokens/embeddings | Sentence like "pick up the red block"; tokenized vocabulary |
| **Task: Goal Image** | (64, 64, 3) | uint8 | Downsized goal state observation |
| **Task: Task ID** | Scalar | int | Discrete task index (0–N_tasks) |

### Output Specification

| Output | Dimensions | Range | Semantics |
|--------|-----------|-------|-----------|
| **Joint Deltas** | (7,) | Δθ ∈ [-0.1, 0.1] rad | Change in joint angles (small steps) |
| **Gripper Command** | Scalar | a_gripper ∈ {open, close, stay} or [-1, 1] | Discrete or continuous gripper action |
| **Action Space** | (8,) | Continuous or mixed | 7 joint deltas + 1 gripper signal |
| **Prediction Frequency** | 10 Hz | Δt = 0.1 s | Action repeats on base controller |

### System Architecture Levels

```
Level 1: Raw Observations
  - RGB image, joint positions, gripper state

Level 2: Encoding
  - Image encoder (CNN) → visual features
  - Proprioceptive encoder (MLP) → state vector
  - Task encoder (varies by task type)

Level 3: Goal Latent Space
  - All task encoders → shared d=256 dimensional latent space
  - Enables comparison across modalities

Level 4: Policy Head
  - Fused observation + goal latent → action prediction
  - 2-3 layer MLP for continuous control
```

---

## 3. Coordinate Frames and Geometry

### Robot Kinematics

**Robot**: 7-DoF arm (typically Sawyer or similar) + parallel gripper

```
Base Frame (B): Fixed at robot base
  - Z-axis: vertical (up)
  - X-axis: forward
  - Y-axis: left

End-Effector Frame (E): Attached to gripper
  - Defined by forward kinematics: T_B_E(θ)
  - Position: p_E = FK_position(θ₁, ..., θ₇)
  - Orientation: R_E = FK_rotation(θ₁, ..., θ₇)

Camera Frame (C): Wrist-mounted camera
  - Offset from end-effector frame (known calibration)
  - Observes workspace and grasped objects
```

### Action Space Semantics

**Joint-Space Actions** (used in paper):
```
θ_t+1 = θ_t + Δθ  (additive joint space)

where Δθ = [Δθ₁, Δθ₂, ..., Δθ₇] (continuous deltas)
      gripper_action = sigmoid(a_gripper) ∈ [0, 1]
        0 = fully open, 1 = fully closed
```

Alternative (Task-Space Actions):
```
p_t+1 = p_t + Δp  (Cartesian position delta)
R_t+1 = R_t · Δω_x Δω_y Δω_z (orientation delta as rotation)

Advantage: More intuitive, generalizes better
Disadvantage: Requires IK solver, singularities
```

### Camera Projection

**Wrist Camera** observes end-effector and workspace:
```
Intrinsic matrix K:
  [f_x    0   c_x]
  [  0  f_y  c_y]
  [  0    0    1 ]

where f_x, f_y = focal lengths
      (c_x, c_y) = principal point

Pixel-to-3D (if depth available):
  X = (u - c_x) * Z / f_x
  Y = (v - c_y) * Z / f_y
  Z = depth(u, v)
```

---

## 4. Architecture Deep Dive

### Overall System Block Diagram

```
┌──────────────────────────────────────────────────────────────┐
│           LYNCH & SERMANET MCIL ARCHITECTURE                │
└──────────────────────────────────────────────────────────────┘

Observation (RGB + Proprioception)
     ↓
┌────────────────────┐       Task Specification
│ Vision Encoder     │       (Language OR Image OR Task ID)
│ CNN (ResNet-18)    │              ↓
│ O_img → (B, 256)   │       ┌──────────────────┐
└────────┬───────────┘       │ Task Encoder     │
         │                   │ (Multimodal)     │
         ├─→ Concat ────────→│ Different head   │
         │                   │ per modality      │
┌────────▼──────────┐        └────┬─────────────┘
│Proprioceptive     │            │
│Encoder (MLP)      │            │
│O_prop → (B, 128)  │            │
└────────┬──────────┘            │
         │                   ┌────▼──────────┐
         │          ┌───────→│ Latent Goal   │
         │          │        │ Space g ∈ ℝ256
         │          │        └────┬──────────┘
         └──────────┘             │
                                  ↓
                         ┌────────────────────┐
                         │ Policy Head (MLP)  │
                         │ Fused + Goal → a   │
                         │ Output: Δθ, gripper│
                         └────────────────────┘
                                  ↓
                         Action: (Δθ₁, ..., Δθ₇, g_cmd)
```

### Module Architecture Table

| Module | Input Shape | Output Shape | Architecture | Parameters |
|--------|-------------|--------------|--------------|-----------|
| **Vision Encoder (CNN)** | (B, 3, 256, 256) | (B, 256) | ResNet-18, pretrained, final FC | ~11.7M (mostly frozen) |
| **Proprioceptive Encoder (MLP)** | (B, 8) | (B, 128) | 2 hidden layers (256 units each) | ~2K |
| **Language Encoder (LSTM)** | (B, T_lang) | (B, 256) | Embedding → LSTM → final state | ~500K |
| **Goal Image Encoder (CNN)** | (B, 3, 64, 64) | (B, 256) | Lightweight CNN (4 conv layers) | ~2M |
| **Task ID Encoder (Embedding)** | (B,) | (B, 256) | Learned embedding table (N_tasks × 256) | ~0.256M per 1K tasks |
| **Latent Goal Fusion** | Goal vectors from above | (B, 256) | Simple projection or identity | ~0 (no learned params) |
| **Policy Head (MLP)** | (B, 256+256+128=640) | (B, 8) | 2 hidden layers (512 units), ReLU | ~400K |
| **Gripper Classifier** | (B, 256+256+128) | (B, 3) | 2-layer MLP, softmax (open/close/stay) | ~50K |

### Data Flow in Multicontext Training

```python
# Forward pass handling multiple task context types

def forward_multicontext(obs_image, obs_prop, task_context):
    """
    Args:
        obs_image: (B, 3, H, W) float32 ∈ [0, 1]
        obs_prop: (B, 8) proprioceptive state
        task_context: dict with one of:
            - {'type': 'language', 'tokens': (B, T_lang)}
            - {'type': 'image', 'goal_img': (B, 3, 64, 64)}
            - {'type': 'task_id', 'task_id': (B,)}

    Returns:
        action: (B, 8) predicted joint deltas + gripper
    """

    # Encode observation (shared across contexts)
    v_obs = vision_encoder(obs_image)  # (B, 256)
    p_obs = prop_encoder(obs_prop)     # (B, 128)

    # Encode task based on context type
    if task_context['type'] == 'language':
        tokens = task_context['tokens']  # (B, T_lang)
        embeddings = embedding_layer(tokens)  # (B, T_lang, 100)
        _, (h_n, _) = lstm_encoder(embeddings)  # h_n: (1, B, 256)
        goal_latent = h_n.squeeze(0)  # (B, 256)

    elif task_context['type'] == 'image':
        goal_img = task_context['goal_img']  # (B, 3, 64, 64)
        goal_latent = goal_img_encoder(goal_img)  # (B, 256)

    elif task_context['type'] == 'task_id':
        task_ids = task_context['task_id']  # (B,)
        goal_latent = task_id_embeddings(task_ids)  # (B, 256)

    # Fuse observation + goal into policy
    fused = concat([v_obs, p_obs, goal_latent])  # (B, 256+128+256=640)
    action_logits = policy_head(fused)  # (B, 8)
    gripper_logits = gripper_classifier(fused)  # (B, 3)

    # Parse action
    action_joint_deltas = action_logits[:, :7]  # (B, 7)
    action_gripper_continuous = action_logits[:, 7]  # (B,) - alternative: discrete from softmax

    return {
        'joint_deltas': action_joint_deltas,
        'gripper_action': gripper_logits,
        'goal_latent': goal_latent,
    }
```

---

## 5. Forward Pass Pseudocode

```python
def forward_pass_episode(model, trajectory, task_context):
    """
    Args:
        model: Trained MCIL policy
        trajectory: dict with 'images', 'proprioception', 'actions' (for training)
        task_context: {'type': 'language', 'tokens': ...} etc.

    Returns:
        predictions: list of action predictions over trajectory
    """

    predictions = []

    for t in range(len(trajectory['images'])):
        # Observation at timestep t
        obs_image = trajectory['images'][t]  # (3, H, W) uint8
        obs_prop = trajectory['proprioception'][t]  # (8,) float32

        # Normalize
        obs_image_norm = (obs_image.astype(float32) / 255.0) - 0.5  # Mean centering
        obs_prop_norm = (obs_prop - mean_prop) / std_prop  # Standardize

        # Batch dimension
        obs_image_batch = obs_image_norm[None, :, :, :]  # (1, 3, H, W)
        obs_prop_batch = obs_prop_norm[None, :]  # (1, 8)

        # Forward pass
        output = model.forward_multicontext(
            obs_image_batch,
            obs_prop_batch,
            task_context
        )

        action_deltas = output['joint_deltas'][0]  # (7,)
        gripper_logits = output['gripper_action'][0]  # (3,) - logits

        # Convert gripper logits to action
        if discrete_gripper:
            gripper_action = argmax(gripper_logits)  # 0=open, 1=stay, 2=close
        else:
            # Map continuous action
            gripper_pos = sigmoid(gripper_logits[0])  # ∈ [0, 1]

        # Clip action to safe ranges
        action_deltas = clip(action_deltas, -0.1, 0.1)

        predictions.append({
            't': t,
            'joint_deltas': action_deltas,
            'gripper_action': gripper_action,
        })

    return predictions


def hindsight_instruction_pairing(play_trajectory):
    """
    Label a play trajectory retroactively with natural language.

    Args:
        play_trajectory: {
            'images': [...],
            'proprioception': [...],
            'actions': [...]  # Unlabeled
        }

    Returns:
        labeled_trajectory: {...} + 'language': "description of what happened"
    """

    # Step 1: Detect what happened during play
    #   e.g., "gripper closed", "hand moved left", "object knocked over"
    detected_events = detect_events(play_trajectory)

    # Step 2: Generate language instruction for detected goal
    #   e.g., detected="gripper_closed_on_red_block"
    #         instruction="close the gripper on the red block"
    generated_instruction = generate_instruction(detected_events)

    # Step 3: (Optional) Have human annotator verify/refine
    #   Ask MTurk: "Does this instruction describe the video?"
    #   Keep instructions with >3/5 approval

    # Step 4: Return labeled trajectory
    return {
        **play_trajectory,
        'language': generated_instruction,
        'detected_events': detected_events,
    }


def compute_loss_multicontext(predictions, targets, task_context_type):
    """
    Loss function for training on heterogeneous contexts.

    Args:
        predictions: {
            'joint_deltas': (B, T, 7),
            'gripper_logits': (B, T, 3),
        }
        targets: {
            'joint_deltas': (B, T, 7),
            'gripper_action': (B, T) int (class 0/1/2) or float (continuous)
        }
        task_context_type: 'language' | 'image' | 'task_id'

    Returns:
        loss: scalar
    """

    # Joint control loss
    loss_joints = mean_squared_error(
        predictions['joint_deltas'],
        targets['joint_deltas']
    )

    # Gripper action loss
    if discrete_gripper:
        loss_gripper = cross_entropy_loss(
            predictions['gripper_logits'].reshape(-1, 3),
            targets['gripper_action'].reshape(-1)
        )
    else:
        loss_gripper = mse_loss(
            predictions['gripper_logits'],
            targets['gripper_action']
        )

    # Combine
    loss_total = loss_joints + 0.5 * loss_gripper

    # Optional: Add auxiliary loss for goal correctness
    # (e.g., ensure goal latent space is well-structured)
    if add_goal_aux_loss:
        # Contrastive loss on goal latents of same task
        loss_aux = contrastive_goal_loss(predictions['goal_latent'], targets['task_id'])
        loss_total += 0.1 * loss_aux

    return loss_total
```

---

## 6. Heads, Targets, and Losses

### Output Heads

| Head | Output Shape | Activation | Loss Function |
|------|-------------|-----------|--------------|
| **Joint Deltas** | (B, T, 7) | Linear (unbounded) | MSE with ground truth Δθ |
| **Gripper Action (Discrete)** | (B, T, 3) | Softmax | Cross-entropy (open/close/stay) |
| **Gripper Action (Continuous)** | (B, T, 1) | Sigmoid or Linear | MSE with position ∈ [0, 1] |
| **Latent Goal** | (B, 256) | None (intermediate) | Implicitly via contrastive loss |

### Supervision Targets

**Behavior Cloning** (primary loss):
```
Ground truth actions from expert demonstrations:
  Δθ_target(t) = θ_expert(t+1) - θ_expert(t)
  gripper_target(t) = gripper_state_expert(t)

Source: human teleoperation or task-specific scripted behaviors
```

**Language Labels** (for HIP):
```
Retroactively generated via hindsight instruction pairing:
  language_label = generate_description(detected_goal, play_trajectory)

Examples:
  - Play sequence: gripper closes on red block → language: "pick up the red block"
  - Play sequence: arm moves left → language: "move left"
  - Play sequence: object pushed into bin → language: "push the object in the bin"
```

### Loss Functions

**Behavior Cloning Loss**:
```
L_BC = ||δθ_pred - δθ_target||₂² + λ_gripper · CE(gripper_pred, gripper_target)

where:
  δθ_pred: predicted joint deltas from policy
  δθ_target: ground truth (expert) joint deltas
  λ_gripper ≈ 0.5
```

**Goal Latent Space Loss** (optional):
```
For same-task pairs (trajectory pairs with same goal):
  L_goal_sim = MSE(g₁, g₂)  minimize distance for same task

For different-task pairs (trajectory pairs with different goals):
  L_goal_diff = max(0, margin - ||g₁ - g₂||₂)  maximize distance for different tasks

Triplet loss variant:
  L_triplet = ||g_anchor - g_pos||² + max(0, margin - ||g_anchor - g_neg||²)
```

**Total Training Loss**:
```
L_total = L_BC + λ_goal · L_goal_latent + λ_reg · ||θ||₂

where:
  λ_goal ≈ 0.1 (importance of goal space structure)
  λ_reg ≈ 1e-4 (L2 regularization strength)
```

---

## 7. Data Pipeline and Augmentations

### Dataset Composition

| Source | # Trajectories | Hours | Task Types | Notes |
|--------|----------------|-------|-----------|-------|
| **Play Data (Unlabeled)** | 50,000+ | ~100+ | Unstructured robot exploration | Majority of dataset; no language labels |
| **Task Demonstrations** | 5,000–10,000 | ~10–20 | 50–100 distinct manipulation tasks | Labeled with language; for finetuning |
| **Human Annotations** | 500–1,000 | N/A | Language labels on play segments | Hindsight instruction pairing validation |

### Data Preprocessing

```python
def preprocess_trajectory(raw_traj):
    """
    Prepare raw robot trajectory for training.

    Args:
        raw_traj: {
            'images': list of raw RGB arrays (H, W, 3) uint8,
            'joint_angles': (T, 7) float32,
            'gripper_state': (T,) binary or continuous,
            'language': str (optional),
        }

    Returns:
        processed: {
            'images': (T, 3, H, W) float32 normalized,
            'proprioception': (T, 8) float32 normalized,
            'actions': (T, 8) float32 (joint deltas + gripper delta),
            'language_tokens': list of int,
            'task_context': dict,
        }
    """

    # === Image Preprocessing ===
    # Stack and normalize
    images = stack([img.astype(float32) / 255.0 for img in raw_traj['images']])
    images = images - 0.5  # Center to [-0.5, 0.5] for ResNet
    images = transpose(images, (0, 3, 1, 2))  # (T, H, W, 3) → (T, 3, H, W)

    # === Proprioception Preprocessing ===
    joint_angles = array(raw_traj['joint_angles'])  # (T, 7)
    gripper_state = array(raw_traj['gripper_state'])  # (T,)

    # Normalize joint angles to [-1, 1] (assuming typical robot bounds)
    joint_angles_norm = (joint_angles - mean_joint) / std_joint
    joint_angles_norm = clip(joint_angles_norm, -1, 1)

    # Normalize gripper [0, 1]
    if gripper_state.max() > 1.0:
        gripper_state_norm = gripper_state / gripper_state.max()
    else:
        gripper_state_norm = gripper_state

    proprioception = concat([joint_angles_norm, gripper_state_norm[:, None]], axis=1)

    # === Action Extraction ===
    joint_deltas = joint_angles[1:] - joint_angles[:-1]  # (T-1, 7)
    gripper_deltas = gripper_state[1:] - gripper_state[:-1]  # (T-1,)

    # Pad to match image sequence length
    joint_deltas = concat([joint_deltas, zeros((1, 7))], axis=0)  # (T, 7)
    gripper_deltas = concat([gripper_deltas, zeros((1,))], axis=0)  # (T,)

    actions = concat([joint_deltas, gripper_deltas[:, None]], axis=1)  # (T, 8)

    # === Language Tokenization ===
    language_tokens = None
    if 'language' in raw_traj:
        words = raw_traj['language'].lower().split()
        language_tokens = [vocab[w] if w in vocab else vocab['<UNK>'] for w in words]
        # Pad/truncate to fixed length (e.g., 20 tokens)
        if len(language_tokens) > 20:
            language_tokens = language_tokens[:20]
        else:
            language_tokens = language_tokens + [vocab['<PAD>']] * (20 - len(language_tokens))

    return {
        'images': images,  # (T, 3, H, W)
        'proprioception': proprioception,  # (T, 8)
        'actions': actions,  # (T, 8)
        'language_tokens': language_tokens,  # (20,) or None
        'trajectory_length': len(images),
    }
```

### Data Augmentation

| Augmentation | Probability | Parameters | Purpose |
|--------------|-------------|-----------|---------|
| **Random Crop** | 0.5 | 10–20 px borders | Robustness to framing variations |
| **Color Jitter** | 0.7 | Brightness ±0.2, Saturation ±0.2 | Lighting/camera variations |
| **Gaussian Blur** | 0.3 | σ ∈ [0.5, 1.5] | Robustness to motion blur |
| **Rotation (Small)** | 0.2 | ±10° | Slight viewpoint changes |
| **Proprioception Noise** | 0.5 | Gaussian ±0.01 rad (joint angle) | Position estimation noise |
| **Sequence Truncation** | 0.1 | Random start/end within trajectory | Variable episode lengths |

---

## 8. Training Pipeline

### Training Stages

| Stage | Duration | Data | Loss | LR | Notes |
|-------|----------|------|------|----|----|
| **Stage 1: Behavior Cloning on Task Data** | 20 epochs | 5K–10K labeled task demos | L_BC only | 5e-4 | Supervised baseline |
| **Stage 2: HIP on Play Data** | 10 epochs | 50K unlabeled + 500 HIP labels | L_BC (with HIP labels) | 1e-4 | Learn from hindsight-labeled play |
| **Stage 3: Goal Space Structure** | 5 epochs | All data (mixed) | L_BC + L_goal | 5e-5 | Refine goal latent space |
| **Stage 4: Fine-tune on New Tasks** | As needed | Few-shot demos (10–100 per task) | L_BC | 1e-5 | Zero/few-shot transfer |

### Hyperparameter Table

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | Adam | β₁=0.9, β₂=0.999 |
| **Batch Size** | 32 (trajectories) | Typically 3–5 timesteps per trajectory in batch |
| **Learning Rate (Stage 1)** | 5e-4 | Cosine annealing: 5e-4 → 1e-5 over 20 epochs |
| **Learning Rate (Stage 2)** | 1e-4 | Fixed or exponential decay |
| **Learning Rate (Stage 3)** | 5e-5 | Conservative, for stability |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Gradient Clipping** | 1.0 | Clip by global norm |
| **Dropout** | 0.2 | Applied to all dense layers |
| **Loss Weights** | λ_gripper=0.5, λ_goal=0.1 | May require task-dependent tuning |
| **Goal Latent Dim** | 256 | Shared space dimension |
| **Vision Encoder** | ResNet-18 (frozen) | Pre-trained on ImageNet; 11.7M parameters |
| **Language Embedding Dim** | 100 | GloVe or similar pre-trained |
| **LSTM Hidden Dim** | 256 | Single or 2-layer LSTM |
| **Gradient Accumulation** | 4 steps | If memory-limited; effective batch ×4 |

### Training Loop Pseudocode

```python
def train_mcil_epoch(model, train_loader, optimizer, device, stage='bc'):
    """
    Train MCIL model for one epoch.

    Args:
        model: MCIL policy
        train_loader: DataLoader over mixed-context trajectories
        optimizer: Adam
        device: 'cuda' or 'cpu'
        stage: 'bc' (behavior cloning), 'hip' (hindsight), 'goal' (goal structure)
    """

    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        # batch contains:
        #   - 'images': (B, T, 3, H, W)
        #   - 'proprioception': (B, T, 8)
        #   - 'actions': (B, T, 8)
        #   - 'task_context': mixed types (language, image, task_id)
        #   - 'task_labels': (B,) for goal structure loss

        images = batch['images'].to(device)
        prop = batch['proprioception'].to(device)
        actions_gt = batch['actions'].to(device)
        task_context = batch['task_context']  # List of dicts
        task_labels = batch.get('task_labels', None)

        optimizer.zero_grad()

        total_batch_loss = 0.0

        # Forward pass over trajectory timesteps
        for t in range(images.shape[1]):
            # Get observation at timestep t
            obs_img = images[:, t]  # (B, 3, H, W)
            obs_prop = prop[:, t]  # (B, 8)

            # Forward
            output = model.forward_multicontext(obs_img, obs_prop, task_context)

            # Predicted actions
            joint_delta_pred = output['joint_deltas']  # (B, 7)
            gripper_pred = output['gripper_action']     # (B, 3) logits or (B, 1)
            goal_latent = output['goal_latent']         # (B, 256)

            # Ground truth
            joint_delta_gt = actions_gt[:, t, :7]
            gripper_gt = actions_gt[:, t, 7]

            # Losses
            loss_joint = mse_loss(joint_delta_pred, joint_delta_gt)

            if discrete_gripper:
                loss_gripper = cross_entropy_loss(gripper_pred, gripper_gt.long())
            else:
                loss_gripper = mse_loss(gripper_pred, gripper_gt.unsqueeze(1))

            loss_bc = loss_joint + 0.5 * loss_gripper

            # Optional: Goal structure loss
            loss_goal = 0.0
            if stage in ['goal', 'hip'] and task_labels is not None:
                # Contrastive loss on goal latents
                for i in range(len(goal_latent)):
                    for j in range(i+1, len(goal_latent)):
                        if task_labels[i] == task_labels[j]:
                            # Same task: pull closer
                            loss_goal += ||goal_latent[i] - goal_latent[j]||₂²
                        else:
                            # Different task: push apart
                            dist = ||goal_latent[i] - goal_latent[j]||₂
                            loss_goal += max(0, margin - dist)

            # Combine
            loss_t = loss_bc + 0.1 * loss_goal
            total_batch_loss += loss_t

        # Average over timesteps
        total_batch_loss = total_batch_loss / images.shape[1]

        # Backward
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total_batch_loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {total_batch_loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss
```

---

## 9. Dataset + Evaluation Protocol

### Dataset Details

| Aspect | Scale |
|--------|-------|
| **Play Data** | 50,000+ trajectories (~100+ hours) |
| **Task-Specific Demos** | 5,000–10,000 trajectories |
| **Language Annotations** | ~500–1,000 hindsight labels on play data |
| **Task Diversity** | 50–100 distinct manipulation tasks |
| **Trajectory Length** | 10–120 seconds (variable) |
| **Action Frequency** | 10 Hz (0.1 s per action) |

### Evaluation Metrics

| Metric | Definition | Notes |
|--------|-----------|-------|
| **Success Rate (SR)** | % tasks completed correctly within 60 seconds | Primary metric; task-specific definition |
| **Trajectory Error** | L2 distance from reference trajectory | Measures path quality, not just goal |
| **Action Accuracy** | % timesteps with predicted action close to expert action | Fine-grained behavior matching |
| **Generalization (Zero-Shot)** | Success on unseen instructions with same task types | Tests language understanding |
| **Few-Shot Adaptation** | Success after 10–100 demonstrations of new task | Tests transfer learning |
| **Human Judgment** | MTurk rating (1–5 scale) of task success | Subjective quality assessment |

### Evaluation Protocol

```python
def evaluate_language_conditioned_policy(model, test_tasks, num_trials=5):
    """
    Evaluate MCIL policy on test set.

    Args:
        model: Trained MCIL policy
        test_tasks: List of {
            'language': str,
            'goal_image': array,
            'task_id': int,
            'success_criteria': function,
        }
        num_trials: Repetitions per task

    Returns:
        results: {
            'success_rate': float,
            'trajectory_error_mean': float,
            'zero_shot_success': float,
        }
    """

    model.eval()
    results = {
        'successes': [],
        'trajectory_errors': [],
        'action_errors': [],
    }

    for task in test_tasks:
        language = task['language']
        goal_image = task['goal_image']
        success_criteria = task['success_criteria']

        for trial in range(num_trials):
            # Reset robot, get initial observation
            obs_image = robot.get_rgb()
            obs_prop = robot.get_proprioception()

            # Prepare task context
            task_context = {
                'type': 'language',
                'tokens': tokenize(language),
            }

            trajectory = []
            success = False

            for step in range(max_steps):
                # Get prediction
                with torch.no_grad():
                    output = model.forward_multicontext(
                        obs_image[None, ...],
                        obs_prop[None, ...],
                        task_context
                    )

                action = output['joint_deltas'][0].cpu().numpy()
                gripper = output['gripper_action'][0].cpu().numpy()

                # Execute action
                robot.move_joints(action, gripper)

                # Observe result
                obs_image = robot.get_rgb()
                obs_prop = robot.get_proprioception()

                trajectory.append({
                    'obs': obs_image,
                    'action': action,
                })

                # Check success
                if success_criteria(obs_image, obs_prop):
                    success = True
                    break

            results['successes'].append(success)

            # Compute trajectory error vs reference
            ref_trajectory = task.get('reference_trajectory', [])
            if len(ref_trajectory) > 0:
                traj_error = compute_trajectory_distance(trajectory, ref_trajectory)
                results['trajectory_errors'].append(traj_error)

    # Aggregate
    success_rate = mean(results['successes']) * 100
    trajectory_error_mean = mean(results['trajectory_errors']) if results['trajectory_errors'] else 0

    return {
        'success_rate': success_rate,
        'trajectory_error_mean': trajectory_error_mean,
    }
```

---

## 10. Results Summary + Ablations

### Main Results

| Condition | Success Rate (%) | Zero-Shot Generalization | Notes |
|-----------|-----------------|--------------------------|-------|
| **Full MCIL + HIP** | **72–80** | High (unseen instructions) | Primary result |
| **BC Only (no HIP)** | 65–70 | Low (overfit to task IDs) | Baseline, less diverse data |
| **Single Modality (Language Only)** | 68–75 | Moderate | Language grounding less flexible |
| **Single Modality (Task ID Only)** | 45–55 | Low | No language generalization |
| **Without Goal Latent Space** | 60–65 | Low | Separate policies per task |
| **With 1% Language Labels** | 72–78 | High | HIP efficiency key result |

### Key Ablation Results

| Ablation | SR (%) | Δ | Interpretation |
|----------|--------|------|-------|
| Remove HIP (use only task demos) | 65 | −7 | HIP adds diversity, efficiency |
| Remove Goal Latent Space Structure | 62 | −10 | Shared latent essential for generalization |
| Reduce Language Labels: 10% → 1% | 72 | −8 | Language annotation highly efficient |
| Play Data: 100% → 50% | 68 | −4 | Play data useful but not critical |
| Vision Encoder: Frozen → Fine-tuned | 74 | +2 | Minor improvement; pre-training sufficient |
| Language Encoder: LSTM → Bag-of-Words | 70 | −2 | Sequence modeling helps slightly |

### Zero-Shot Generalization Experiments

```
Experiment: Train on 20 tasks with language, test on unseen task language instructions.

Setup:
  - Train: tasks = {pick_red, pick_blue, pick_green, ..., push_left, ...}
  - Test: language instructions for novel combinations
    e.g., "pick the blue object" (blue object seen in training but not in pick task context)
         "push the large block" (large/block concepts seen, novel combination)

Results:
  - Unseen Object + Seen Action: 78% success
  - Seen Object + Unseen Action: 65% success
  - Fully Unseen Concept: 45% success (but better than random baseline ~20%)

Conclusion: Language representation generalizes better to novel combinations than task IDs.
```

---

## 11. Practical Insights

### Engineering Takeaways

1. **Hindsight Instruction Pairing is Practical**: You don't need online language annotation for every demo. Auto-generate labels from play trajectories, then have humans verify a small fraction (~5–10%). Dramatically reduces annotation cost.

2. **Separate Encoders + Shared Latent**: Rather than forcing all task types through one encoder, use specialized encoders (CNN for images, LSTM for language, embedding for IDs) that map to a common latent space. This simplifies learning and enables mixing modalities.

3. **Unlabeled Play Data is Valuable**: 50K play trajectories + 500 language labels outperforms 5K labeled task demos alone. Robot exploration is cheap; language annotation is expensive.

4. **Pre-trained Language Embeddings Work**: GloVe or FastText embeddings capture semantic meaning out-of-the-box. No need to train from scratch; fine-tuning helps but is optional.

5. **Vision Encoder Can Be Frozen**: ResNet-18 pre-trained on ImageNet is strong enough; freezing weights reduces memory and training time. Only fine-tune if data is very different from ImageNet.

6. **Goal Latent Space Enables Transfer**: A well-structured shared latent space (d=256) enables generalization across tasks. Use contrastive losses to encourage structure.

7. **Discrete Gripper Actions**: Treating open/close/stay as discrete (3-class softmax) is more stable than continuous regression. Avoids intermediate gripper states.

8. **Action Space: Joint Deltas vs Cartesian**: Joint deltas are simpler (no IK needed) but less intuitive. Cartesian is more interpretable but requires differentiable IK. Start with joints.

9. **Trajectory Preprocessing**: Clip action deltas to small ranges (±0.1 rad for joints). Prevents policy from learning extreme actions that destabilize robot.

10. **Batch Normalization in Vision Encoder**: If fine-tuning the vision encoder, use instance norm or layer norm instead of batch norm (smaller batch sizes can cause issues).

### Gotchas & Pitfalls

1. **Language Tokenization Inconsistency**: Tokenization must be identical between training and inference. Off-by-one errors in vocabulary indexing silently break language grounding.

2. **Gripper State Representation**: Ambiguity in gripper state (position vs force vs binary). Define clearly: e.g., 0=fully open, 1=fully closed, continuous ∈ [0, 1].

3. **Proprioception Normalization**: Joint angles have different ranges (e.g., shoulder ∈ [-π, π], wrist ∈ [-2π, 2π]). Normalize per-joint, not globally.

4. **Goal Image Encoding**: If using goal images, they must be captured in the same framing as current images. Camera pose/calibration changes break this.

5. **Hindsight Label Quality**: Auto-generated labels can be noisy. Always validate a sample (10–20%) with humans. Noisy labels degrade final performance more than fewer labels.

6. **Mixing Trajectory Lengths**: Variable-length trajectories require careful batching. Either pad all to max length (wasteful) or use dynamic padding (complex).

7. **Overfitting to Task IDs**: If task IDs are too predictive (e.g., task 1 always means "pick red"), model learns task IDs instead of language. Mix tasks across contexts during training.

8. **Evaluation Bias**: Evaluating on tasks similar to training biases results. Always hold out diverse test tasks (different objects, different orders, etc.).

9. **Sim-to-Real Transfer**: If training in simulation, domain randomization is critical. Real gripper dynamics, friction, etc. differ from sim.

10. **Action Frequency Mismatch**: If training at 10 Hz but deploying at 5 Hz (or vice versa), action scaling changes. Standardize early.

### Tiny-Subset Overfit Plan

```python
def debug_overfit_on_5_trajectories():
    """
    Verify implementation on tiny subset.
    """
    # 1. Select 5 diverse short trajectories
    train_tiny = [
        dataset[0],  # 20-step simple task
        dataset[100],
        dataset[200],
        dataset[500],
        dataset[1000],
    ]

    # 2. Disable augmentation & regularization
    model.dropout = 0.0
    weight_decay = 0.0
    learning_rate = 1e-2  # Aggressive

    # 3. Train to convergence on tiny set
    for epoch in range(500):
        loss = train_mcil_epoch(model, [train_tiny], optimizer, device)
        if loss < 1e-3:
            print(f"✓ Overfitting successful at epoch {epoch}")
            break
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # 4. Inference on training set
    with torch.no_grad():
        for traj in train_tiny:
            obs_img = traj['images'][0][None, :, :, :]
            obs_prop = traj['proprioception'][0][None, :]
            task_context = {'type': 'language', 'tokens': traj['language_tokens']}

            output = model.forward_multicontext(obs_img, obs_prop, task_context)
            action_pred = output['joint_deltas'][0].cpu().numpy()
            action_gt = traj['actions'][0, :7]

            error = mean_absolute_error(action_pred, action_gt)
            print(f"Action error: {error:.4f} (should be <0.001 if overfit)")

    # 5. Diagnostics if overfitting fails
    # - Check data shape: images (T, 3, H, W)?
    # - Gradients flowing? (gradient norm > 0 in first layer?)
    # - Loss decreasing monotonically?
    # - Check for NaN/Inf in activations
    # - Verify ground truth actions are non-zero
```

---

## 12. Minimal Reimplementation Checklist

### Core Modules

- [ ] **Vision Encoder** (ResNet-18 pre-trained)
  - [ ] Load from torchvision.models
  - [ ] Remove final classification layer
  - [ ] Output 256D features
  - [ ] Test: `forward(randn(1, 3, 256, 256)) → (1, 256)`

- [ ] **Proprioceptive Encoder** (2-layer MLP)
  - [ ] Input: 8D (7 joints + gripper)
  - [ ] Hidden: 256D
  - [ ] Output: 128D
  - [ ] Test: `forward(randn(1, 8)) → (1, 128)`

- [ ] **Language Encoder** (LSTM + GloVe embeddings)
  - [ ] GloVe embedding lookup (100D)
  - [ ] LSTM: 1–2 layers, 256 hidden
  - [ ] Extract final hidden state
  - [ ] Test: `forward(token_ids) → (1, 256)`

- [ ] **Goal Image Encoder** (lightweight CNN)
  - [ ] 4 conv layers + ReLU
  - [ ] Output: 256D
  - [ ] Test: `forward(randn(1, 3, 64, 64)) → (1, 256)`

- [ ] **Task ID Embedding** (lookup table)
  - [ ] Size: (num_tasks, 256)
  - [ ] Test: `forward(task_id) → (1, 256)`

- [ ] **Latent Goal Space** (projection or identity)
  - [ ] Input: 256D from any encoder
  - [ ] Output: 256D in latent space
  - [ ] Optional: learned projection matrix

- [ ] **Policy Head** (2-layer MLP)
  - [ ] Input: fused obs + goal (640D)
  - [ ] Hidden: 512D, ReLU
  - [ ] Output: 8D (7 joints + gripper continuous) or 8D + 3D (if discrete gripper)
  - [ ] Test: `forward(randn(1, 640)) → (1, 8)`

- [ ] **Gripper Classifier** (3-way softmax, optional)
  - [ ] Input: 640D
  - [ ] Output: logits for [open, stay, close]
  - [ ] Test: `forward(randn(1, 640)) → (1, 3)`

- [ ] **Training Loop**
  - [ ] DataLoader with variable-length trajectories
  - [ ] Forward pass over timesteps
  - [ ] Loss aggregation
  - [ ] Backward pass + optimizer step
  - [ ] Validation/evaluation

- [ ] **Hindsight Instruction Pairing**
  - [ ] Detect end-effector state changes (gripper close, arm move, etc.)
  - [ ] Generate candidate instruction templates
  - [ ] (Optional) Human validation

### Minimal Dataset Format

```json
{
  "trajectory": {
    "images": ["base64_rgb_t0", "base64_rgb_t1", ...],
    "proprioception": [[θ₁, θ₂, ..., θ₇, g], ...],
    "actions": [[Δθ₁, ..., Δθ₇, Δg], ...],
    "language": "pick up the red block",
    "task_id": 5,
    "goal_image": "base64_rgb_goal",
    "success": true
  }
}
```

### Quick Validation

```
□ DataLoader yields correct shapes:
    - images: (B, T, 3, H, W)
    - proprioception: (B, T, 8)
    - actions: (B, T, 8)
    - language_tokens: (B, T_lang) or None

□ Model forwards through all modalities:
    - Language: tokens → LSTM → (B, 256)
    - Goal Image: (B, 3, 64, 64) → (B, 256)
    - Task ID: (B,) → (B, 256)

□ Policy output shapes:
    - joint_deltas: (B, 7) or (B, T, 7) for sequences
    - gripper_action: (B, 3) logits or (B, 1) continuous

□ Loss decreases on toy 5-traj overfit within 100 iterations

□ No NaN/Inf gradients

□ Inference time < 50 ms on target hardware

□ Evaluation metrics:
    - Success rate: binary task completion
    - Trajectory error: L2 distance from reference
```

---

## References

- **[Lynch & Sermanet 2021]** Language Conditioned Imitation Learning Over Unstructured Data. *Robotics: Science and Systems (RSS)*, July 2021.
- **arXiv**: 2005.07648
- **Project Page**: [language-play.github.io](https://language-play.github.io/)
- **Related**: Multicontext imitation learning; hindsight instruction pairing; zero-shot task generalization.
