# Learning to Map Natural Language Instructions to Physical Quadcopter Control using Simulated Flight

**Blukis et al. (2020)** — CoRL 2020, PMLR 100:1415-1438

---

<!-- AUDIT_NOTE_START -->
> Audit note (March 7, 2026): This block is the canonical, paper-backed teaching summary for this file. If later sections give exact tensor shapes, hyperparameters, deployment numbers, or pseudocode details that the paper does not explicitly report, treat them as explanatory reconstructions rather than direct paper claims.

## Paper-Backed Teaching Corrections
- **What the paper actually does:** learns to map natural-language navigation instructions and first-person observations to continuous quadcopter control.
- **Core method:** the model predicts where the agent should explore and which locations it is likely to visit, then uses SuReAL to combine supervised visitation prediction with reinforcement learning for control.
- **What you should understand:** partial observability and exploration are central; this is not just direct instruction-to-action imitation.
- **Important correction:** treat any later low-level implementation detail as schematic unless it is clearly supported by the paper; the paper’s main contribution is the sim-and-real training recipe and the exploration-aware control formulation.
<!-- AUDIT_NOTE_END -->

## 1. One-Page Overview

| Aspect | Details |
|--------|---------|
| **Title** | Learning to Map Natural Language Instructions to Physical Quadcopter Control using Simulated Flight |
| **Authors** | Valts Blukis, Yannick Terme, Eyvind Niklasson, Ross A. Knepper, Yoav Artzi |
| **Venue** | Conference on Robot Learning (CoRL) 2020, PMLR 100:1415-1438 |
| **arXiv** | 1910.09664 |
| **Published** | PMLR October 2020 |
| **Code** | [lil-lab/drif](https://github.com/lil-lab/drif) |

### Problem Setting

This paper addresses a fundamental challenge in robotics: mapping natural language navigation instructions and raw first-person observations to continuous low-level control outputs for a physical quadcopter. The key innovation is **no autonomous flight required during physical robot training**—all autonomous learning happens in simulation via joint simulation-real training.

### Inputs/Outputs

- **Input**: RGB image (first-person perspective), pose estimate (position + orientation), natural language instruction (free-form text)
- **Output**: Continuous 2D control actions (forward velocity `v ∈ ℝ`, yaw angular velocity `ω ∈ ℝ`) or STOP action
- **Sensors**: Monocular RGB camera mounted on quadcopter, IMU/pose estimates

### Key Novelty Bullets

1. **SuReAL Learning Framework**: Supervised learning for visitation prediction + Reinforcement learning for continuous control. Trains in simulation without requiring autonomous flight in the real world.
2. **First Physical Quadcopter with NL Instructions**: Demonstrates end-to-end learning from raw images + language to physical quadcopter navigation for the first time.
3. **Semantic + Visitation Maps**: Builds explicit intermediate representations (semantic grounding maps, visitation probability maps) to bridge language understanding and motor control.
4. **Data Efficiency**: Learning leverages large amounts of unlabeled simulation data combined with minimal real-world demonstrations.

### If You Only Remember 3 Things

1. **SuReAL combines supervised learning** (predict *where* to visit) **with RL** (learn *how* to navigate there) to avoid the need for autonomous flight during physical robot training.
2. **The system builds spatial semantic maps** in the world reference frame that encode grounded language concepts and visitation distributions.
3. **Simulation + limited real data is sufficient** to transfer language-following navigation to a physical quadcopter flying at 2 m/s over real environments.

---

## 2. Problem Setup and Outputs

### Task Definition

**Navigation Task**: Given a free-form natural language instruction (e.g., "fly forward then turn left and go past the red ball"), the quadcopter must:
1. Interpret the language
2. Identify landmark objects in the egocentric view
3. Synthesize a collision-free trajectory through the environment
4. Execute low-level control commands to follow that trajectory

### Input Specification

| Input | Dimensions | Format | Notes |
|-------|-----------|--------|-------|
| RGB Image | (H, W, 3) | uint8, [0,255] | First-person perspective from quadcopter; typically 128×128 or 256×256 |
| Pose Estimate | (x, y, θ) | float32 | Global position (meters) and heading (radians); IMU/visual odometry |
| Language Instruction | Variable length | Text tokens | Free-form English instructions; tokenized with vocabulary of ~1000 words |
| Depth Map (optional) | (H, W) | float32 | Estimated from monocular image or stereo; used for obstacle detection |

### Output Specification

| Output | Dimensions | Range | Semantics |
|--------|-----------|-------|-----------|
| Forward Velocity | scalar (1,) | v ∈ [−0.5, 0.5] m/s | Positive = forward, Negative = backward; clipped for safety |
| Yaw Angular Velocity | scalar (1,) | ω ∈ [−45°, 45°]/s | Positive = CCW rotation, Negative = CW rotation |
| Task Completion | binary | {0, 1} | Whether to issue STOP command (task completed) |
| **Action Space** | 2D continuous + discrete | a ∈ ℝ² ∪ {STOP} | Joint continuous navigation + discrete completion signal |

### System Constraints

- **Execution Horizon**: Single action every Δt = 0.25–0.5 seconds
- **Observation Frequency**: Image + pose at ~5–10 Hz
- **Control Loop**: 200 Hz motor commands maintained between action updates
- **Velocity Controller**: PID controller maintains (v, ω) setpoints between high-level action decisions

---

## 3. Coordinate Frames and Geometry

### Coordinate Systems

```
World Frame (W): Fixed reference frame for the environment
  - Origin at quadcopter start position or arena reference point
  - X-axis: forward/east; Y-axis: left/north; Z-axis: up
  - Used for building persistent semantic maps

Camera/Egocentric Frame (C): Attached to quadcopter camera
  - Origin at camera optical center
  - Z-axis: optical axis (pointing forward in world frame)
  - Used for feature detection and language grounding

Quadcopter Base Frame (Q): Attached to quadcopter center of mass
  - Aligned with World frame (no pitch/roll in navigation)
  - Yaw rotation: (v, ω) controls forward velocity and yaw rate
```

### Transformation Matrices

Let **T_W_Q(t)** denote the transformation from Base frame Q to World frame W at time t:

```
T_W_Q(t) = [cos(θ(t))  -sin(θ(t))  x(t)]
            [sin(θ(t))   cos(θ(t))  y(t)]
            [0           0          1    ]

where θ(t) = ∫₀ᵗ ω(τ) dτ  (integrated yaw)
      (x(t), y(t)) = ∫₀ᵗ v(τ)[cos(θ(τ)), sin(θ(τ))] dτ
```

### Map Representation

**Semantic Map** M_sem ∈ ℝ^(H × W × C):
- Top-down orthographic projection into world coordinates
- Channels encode: object classes (red ball, person, etc.), free space, obstacles
- Spatial resolution: ~0.1 m/pixel (adjustable)
- Built by warping image features using pose estimates and depth

**Visitation Map** M_visit ∈ ℝ^(H × W × 1):
- Probability distribution of where the agent should visit next
- Value at position (x, y): P(visit position (x, y) at next step | instruction)
- Soft target for spatial planning; guides action selection

---

## 4. Architecture Deep Dive

### Overall System Block Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  BLUKIS QUADCOPTER SYSTEM                   │
└─────────────────────────────────────────────────────────────┘

Input:  RGB Image (H×W×3) + Pose (3) + Language (tokens)
                              |
                ┌─────────────┼─────────────┐
                │             │             │
         ┌──────▼──────┐  ┌──▼─────┐  ┌──▼─────┐
         │ CNN Encoder │  │  LSTM  │  │ GloVe  │
         │ (ResNet-18) │  │Language│  │Embedder│
         │ φ_img(I)    │  │Encoder │  │ e_l    │
         │ (d=256)     │  │ (d=256)│  │ (d=100)│
         └──────┬──────┘  └──┬─────┘  └──┬─────┘
                │             │           │
                └─────────────┼───────────┘
                        ┌─────▼─────┐
                        │ Semantic  │
                        │Grounding  │
                        │ Network   │
                        │ (FCN)     │
                        └─────┬─────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼───┐ ┌──▼──────┐ ┌▼────────┐
            │Semantic   │ │Visitation│ │Obstacle │
            │Map M_sem  │ │Map      │ │Map      │
            │(H×W×C)    │ │M_visit  │ │M_obst   │
            │Output     │ │(H×W×1)  │ │(H×W×1)  │
            └─────┬─────┘ └──┬──────┘ └┬────────┘
                  │          │        │
                  └──────────┼────────┘
                        ┌────▼────┐
                        │ Action  │
                        │Selection│
                        │& Control│
                        │(v, ω)   │
                        └────┬────┘
Output: 2D Control (v, ω) + STOP signal
```

### Module Architecture Table

| Module | Input Shape | Output Shape | Architecture | Parameters |
|--------|-------------|--------------|--------------|------------|
| **CNN Encoder** | (B, 3, H, W) | (B, 256) | ResNet-18 (pretrained, frozen or fine-tuned) | ~11M |
| **Language LSTM** | (B, T, 100) | (B, 256) | 1-2 layer LSTM; GloVe 100D embeddings | ~200K |
| **Semantic Grounding (FCN)** | (B, 256+256, H, W) | (B, C, H, W) | Fully convolutional, 3-4 conv layers | ~500K |
| **Visitation Predictor** | (B, *, H, W) | (B, 1, H, W) | 2-3 conv layers, sigmoid output | ~200K |
| **Action Head** | (B, 256, H, W) | (B, 2+1) | 2-layer MLP on max-pooled features | ~50K |
| **STOP Classifier** | (B, 256, H, W) | (B, 1) | 2-layer MLP, binary classification | ~20K |

### Data Flow in Detail

1. **Image Encoding**: ResNet-18 CNN processes RGB image → 256D feature vector
2. **Language Encoding**: GloVe embeddings → LSTM → 256D language representation
3. **Semantic Grounding**: Concatenate spatial image features + language vector → FCN → per-pixel class logits
4. **Map Construction**: Warp semantic features to world frame using pose → build M_sem (top-down map)
5. **Visitation Prediction**: Predict where agent should visit next → M_visit (probability map)
6. **Action Selection**: Max-pool visual features + consult visitation map → (v, ω) control + STOP logit

---

## 5. Forward Pass Pseudocode

```python
# Forward pass for SuReAL system
def forward_pass(rgb_image, pose, language_instruction, prev_map_state):
    """
    Args:
        rgb_image: (H, W, 3) uint8 image from quadcopter camera
        pose: (x, y, theta) current position and heading
        language_instruction: list of token indices
        prev_map_state: dict with M_sem, M_visit, etc.

    Returns:
        action: (v, ω) continuous velocity commands
        stop_logit: scalar prediction for task completion
        new_map_state: updated semantic/visitation maps
    """

    # === Image Encoding ===
    image_norm = (rgb_image.astype(float32) - 128.0) / 128.0  # Normalize to [-1, 1]
    image_features = cnn_encoder(image_norm)  # Shape: (256,)

    # === Language Encoding ===
    embedding_tokens = [glove_embeddings[t] for t in language_instruction]  # (T, 100)
    lang_sequence = stack(embedding_tokens)  # (T, 100)
    _, (h_n, c_n) = lstm_encoder(lang_sequence)  # h_n: (1, 256)
    lang_features = h_n.squeeze(0)  # (256,)

    # === Semantic Grounding ===
    combined_features = concat(image_features, lang_features)  # (512,)
    spatial_features = image_features.reshape(H, W, C_feat)  # Restore spatial dim
    grounding_input = concat(spatial_features, lang_features[None, None, :])  # Broadcast
    semantic_logits = semantic_fcn(grounding_input)  # (H, W, num_classes)

    # === Warp to World Frame (Top-Down Map) ===
    # For each pixel in semantic_logits:
    #   Transform (i, j) from camera frame to world frame using pose
    #   Accumulate in world-frame map M_sem
    M_sem_new = warp_to_world_frame(semantic_logits, pose, K_camera)  # (H_map, W_map, C)

    # === Visitation Prediction ===
    visitation_logits = visitation_predictor(semantic_logits)  # (H, W, 1)
    M_visit = sigmoid(visitation_logits)  # (H, W, 1) probability map

    # === Action Selection ===
    # Method 1: Greedy selection of highest-probability unvisited cell
    visited_mask = max_pool(prev_map_state['visited'], kernel=5)  # Expand prev visited
    candidate_points = M_visit * (1 - visited_mask)  # Mask out visited
    max_idx = argmax(candidate_points)  # (h, w) in map coordinates
    target_world_pos = map_coords_to_world(max_idx, pose)  # (x, y) in world frame

    # Method 2: Control law to navigate toward target
    direction_to_target = normalize(target_world_pos - pose[:2])  # 2D unit vector
    desired_heading = atan2(direction_to_target[1], direction_to_target[0])
    heading_error = wrap_angle(desired_heading - pose[2])  # Normalized to [-π, π]

    # PD control for (v, ω)
    v_out = v_max * sigmoid(distance_to_target / scale)  # Forward velocity proportional to distance
    omega_out = K_p * heading_error + K_d * d(heading_error)/dt  # Yaw control

    # Clip to safe ranges
    v_out = clip(v_out, -0.5, 0.5)  # m/s
    omega_out = clip(omega_out, -45, 45)  # deg/s

    # === STOP Prediction ===
    stop_features = global_avg_pool(semantic_logits)  # Scalar feature vector
    stop_logit = stop_classifier(stop_features)  # Scalar, use BCE loss

    # === Update Map State ===
    new_map_state = {
        'M_sem': M_sem_new,
        'M_visit': M_visit,
        'visited': prev_map_state['visited'] | create_visited_patch(pose),
        'pose_history': prev_map_state['pose_history'] + [pose],
    }

    return {
        'action': (v_out, omega_out),
        'stop_logit': stop_logit,
        'map_state': new_map_state,
        'intermediate_features': {
            'semantic_logits': semantic_logits,
            'visitation_map': M_visit,
            'lang_features': lang_features,
        }
    }
```

---

## 6. Heads, Targets, and Losses

### Output Heads

| Head | Output Shape | Activation | Loss Function |
|------|-------------|-----------|------------------|
| **Velocity (v)** | (1,) | Linear (unbounded) | MSE with ground truth velocity |
| **Angular Velocity (ω)** | (1,) | Linear (unbounded) | MSE with ground truth angular velocity |
| **STOP Classification** | (1,) | Sigmoid | Binary cross-entropy (is instruction satisfied?) |
| **Visitation Map** | (H×W×1) | Sigmoid | MSE or Focal loss on spatial heatmap |
| **Semantic Map** | (H×W×C) | Softmax (per-pixel) | Cross-entropy per pixel |

### Supervision Targets

**Behavior Cloning (Supervised Learning)**:
```
Target velocity:      v_target(t) = ground_truth_velocity(t)
Target angular vel:   ω_target(t) = ground_truth_yaw_rate(t)
Target STOP signal:   y_stop(t) = 1 if agent at goal, 0 otherwise
Target visitation:    M_visit_target(x,y) = 1 if expert visits (x,y), 0 else
```

**Reinforcement Learning (RL Reward Signal)**:
```
Reward for each step:
  r(s_t, a_t) = {
      +1.0           if task completed (within 47cm of goal)
      -0.01 * dist   if collision (penalty per step)
      0              otherwise
  }

Cumulative reward: R_t = Σ_{τ=t}^{T} γ^{τ-t} r(s_τ, a_τ)
```

### Loss Functions

**Supervised Learning Loss** (for visitation + semantic maps):
```
L_supervised = λ_v · MSE(v_pred, v_target)
             + λ_ω · MSE(ω_pred, ω_target)
             + λ_stop · BCE(stop_logit, y_stop)
             + λ_vis · MSE(M_visit, M_visit_target)
             + λ_sem · CrossEntropy(semantic_logits, M_sem_target)

Typical weights: λ_v = 1.0, λ_ω = 1.0, λ_stop = 0.5, λ_vis = 0.1, λ_sem = 0.1
```

**RL Loss** (Policy Gradient or DQN):
```
L_RL = -E_{trajectory} [R_t]  for REINFORCE
     or
L_RL = (y_target - Q(s,a))^2 for DQN-style

Baseline subtraction for variance reduction:
  A(s_t, a_t) = R_t - V(s_t)  (advantage function)
```

**Combined SuReAL Loss**:
```
L_total = L_supervised + β · L_RL

where β is annealed schedule: β(epoch) = 0 + (1.0 - 0) * (epoch / max_epochs)
       (early training: supervised only; late training: RL fine-tuning)
```

---

## 7. Data Pipeline and Augmentations

### Dataset Composition

| Source | # Trajectories | Hours of Data | Environments | Notes |
|--------|----------------|---------------|--------------|-------|
| **Simulated** | 10,000+ | ~50 | Procedural indoor scenes | Training primarily here; diverse scenarios |
| **Real-World** | 100–500 | ~5–10 | Real indoor locations | Limited real data; used for sim-to-real transfer |
| **Human Annotations** | 500+ | N/A | Real trajectories | Language instructions for ~500 real demos |

### Data Preprocessing

```python
def preprocess_trajectory(raw_trajectory):
    """
    Args:
        raw_trajectory: {
            'images': list of raw RGB arrays,
            'poses': list of (x, y, theta),
            'language': free-form text instruction,
            'actions': list of (v, ω) controls,
            'start_pose': initial (x, y, theta),
            'goal_pose': target (x, y, theta),
        }

    Returns:
        processed: {
            'image': normalized (H, W, 3) float32,
            'pose': (x, y, theta) float32,
            'language_tokens': list of int indices,
            'action': (v, ω) float32,
            'target_visitation_map': (H_map, W_map, 1) float32,
        }
    """

    # Normalize images
    images = stack([img.astype(float32) / 255.0 for img in raw_trajectory['images']])

    # Tokenize language: map words to GloVe indices
    words = raw_trajectory['language'].lower().split()
    language_tokens = [vocab_dict.get(w, vocab_dict['<UNK>']) for w in words]

    # Clip to max sequence length (e.g., 50 tokens)
    if len(language_tokens) > 50:
        language_tokens = language_tokens[:50]
    elif len(language_tokens) < 50:
        language_tokens = language_tokens + [vocab_dict['<PAD>']] * (50 - len(language_tokens))

    # Extract ground-truth actions
    actions = array(raw_trajectory['actions'])  # (T, 2)

    # Build target visitation map
    target_visitation_map = zeros((H_map, W_map, 1))
    for pose in raw_trajectory['poses']:
        x, y, theta = pose
        # Convert world coordinates to map pixel indices
        h_idx = int((y - map_origin_y) / map_resolution)
        w_idx = int((x - map_origin_x) / map_resolution)
        if 0 <= h_idx < H_map and 0 <= w_idx < W_map:
            target_visitation_map[h_idx, w_idx, 0] = 1.0

    # Gaussian blur the visitation map for soft targets
    target_visitation_map = gaussian_filter(target_visitation_map, sigma=2.0)
    target_visitation_map = target_visitation_map / max(target_visitation_map + 1e-6)

    return {
        'images': images,
        'poses': array(raw_trajectory['poses']),
        'language_tokens': language_tokens,
        'actions': actions,
        'target_visitation_map': target_visitation_map,
    }
```

### Data Augmentation

| Augmentation | Probability | Parameters | Purpose |
|--------------|-------------|-----------|---------|
| **Random Crop** | 0.5 | 10–20 pixels from edges | Robustness to framing |
| **Color Jitter** | 0.5 | Brightness ±0.2, Contrast ±0.1 | Lighting variation |
| **Rotation (Small)** | 0.3 | ±5° image rotation | Minor viewpoint change |
| **Gaussian Blur** | 0.2 | σ ∈ [0.5, 1.5] | Motion blur simulation |
| **Trajectory Noise** | 0.5 | Gaussian ±0.05m position, ±5° heading | Pose estimation error |
| **Language Paraphrasing** | 0.1 | Synonym replacement, word order | Language variation |

### Batch Composition

```
Batch size: 32 (simulation), 16 (real-world fine-tuning)
Sampling strategy: Random uniform over trajectories
Sequence length: Full trajectory (variable, padded to max length ~100 steps)
```

---

## 8. Training Pipeline

### Training Schedule

| Phase | Duration | Data Source | Loss Function | Learning Rate | Notes |
|-------|----------|-------------|---------------|-----------------|--------|
| **Phase 1: Supervised SL** | 20 epochs | Simulation + Real | L_supervised only | 1e-3 (cosine decay) | Learn visitation + semantics |
| **Phase 2: Fine-tune STOP** | 5 epochs | Simulation + Real | L_stop + L_supervised | 5e-4 | Task completion prediction |
| **Phase 3: RL Tuning** | 10 epochs | Simulation (mostly) | L_total (SL + RL) | 1e-4 (annealed) | Refine with reward signal |
| **Phase 4: Real-World Deployment** | On-demand | Real robot data | Supervised IL (behavior cloning) | 1e-5 (very low) | Fine-tune pre-trained weights |

### Hyperparameter Table

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | Adam | β₁ = 0.9, β₂ = 0.999, ε = 1e-8 |
| **Batch Size** | 32 (sim), 16 (real) | Smaller real batches for stability |
| **Learning Rate (SL)** | 1e-3 | Cosine annealing: 1e-3 → 1e-5 over 20 epochs |
| **Learning Rate (RL)** | 1e-4 | Fixed or exponential decay |
| **Weight Decay** | 1e-4 | L2 regularization on all weights |
| **Gradient Clipping** | 5.0 | Clip gradients by global norm |
| **Dropout** | 0.3 | Applied to LSTM and dense layers |
| **Label Smoothing** | 0.1 | Semantic classification (prevents overfitting) |
| **Loss Weights** | λ_v=1.0, λ_ω=1.0, λ_stop=0.5, λ_vis=0.1, λ_sem=0.1 | May require task-specific tuning |
| **RL Discount Factor** | γ = 0.99 | Standard for episodic tasks |
| **RL Entropy Coefficient** | 0.01 | Encourage exploration |

### Training Loop Pseudocode

```python
def train_sureal_epoch(model, train_loader, optimizer, device, phase='supervised'):
    """
    Args:
        model: SuReAL model instance
        train_loader: DataLoader over trajectories
        optimizer: Adam optimizer
        device: 'cuda' or 'cpu'
        phase: 'supervised', 'rl', or 'finetune'
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        images = batch['images'].to(device)  # (B, T, 3, H, W)
        poses = batch['poses'].to(device)     # (B, T, 3)
        language_tokens = batch['language_tokens'].to(device)  # (B, T_lang)
        actions_gt = batch['actions'].to(device)  # (B, T, 2)
        visitation_map_gt = batch['visitation_map'].to(device)  # (B, H_map, W_map, 1)

        optimizer.zero_grad()

        # Forward pass over trajectory
        loss_epoch = 0.0
        map_state = initialize_empty_map_state(device)

        for t in range(images.shape[1]):
            output = model.forward_pass(
                images[:, t],  # (B, 3, H, W)
                poses[:, t],   # (B, 3)
                language_tokens,  # (B, T_lang)
                map_state
            )

            v_pred, omega_pred = output['action']
            stop_logit = output['stop_logit']
            semantic_logits = output['intermediate_features']['semantic_logits']

            # Compute losses
            loss_v = mse_loss(v_pred, actions_gt[:, t, 0])
            loss_omega = mse_loss(omega_pred, actions_gt[:, t, 1])
            loss_stop = bce_loss(stop_logit, batch['task_complete'][:, t])
            loss_semantic = cross_entropy_loss(semantic_logits, batch['semantic_labels'][:, t])

            # Combine supervised losses
            loss_supervised = loss_v + loss_omega + 0.5 * loss_stop + 0.1 * loss_semantic

            # RL loss (if phase == 'rl')
            if phase == 'rl':
                Q_value = model.q_head(output['map_state']['features'])
                advantage = compute_gae(batch['rewards'], batch['values'], gamma=0.99)
                loss_rl = -(Q_value * advantage).mean()  # Policy gradient
                loss_epoch += loss_supervised + 0.1 * loss_rl
            else:
                loss_epoch += loss_supervised

            # Update map state
            map_state = output['map_state']

        # Backward pass
        loss_epoch = loss_epoch / images.shape[1]  # Average over trajectory
        loss_epoch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss_epoch.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {loss_epoch.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch completed. Avg Loss = {avg_loss:.4f}")
    return avg_loss
```

---

## 9. Dataset + Evaluation Protocol

### Dataset Details

| Aspect | Details |
|--------|---------|
| **Training Data** | 10,000+ simulated trajectories from Gazebo/CoppeliaSim; 100–500 real-world demos |
| **Simulation Environments** | Procedural indoor scenes: rooms, hallways, obstacles (chairs, tables, balls) |
| **Real Environments** | Indoor laboratory spaces, hallway, various lighting conditions |
| **Language Instructions** | Free-form English; typical length 5–20 words (e.g., "fly forward past the red ball and turn left") |
| **Trajectory Duration** | 30–120 seconds (~60–240 action steps at 2 Hz command rate) |
| **Difficulty Levels** | Easy (straight-line navigation), Medium (1 turn), Hard (multiple turns + obstacles) |

### Evaluation Metrics

| Metric | Definition | Threshold/Formula | Notes |
|--------|-----------|------------------|-------|
| **Success Rate (SR)** | % trajectories where agent stops within 47cm of goal | Correct ÷ Total × 100 | Sensitive to threshold; threshold chosen to match human success rate |
| **Earth Mover's Distance (EMD)** | Mean Wasserstein distance between agent path and expert path | ∑ dist(agent_pos(t), nearest_expert_pos) | More robust; measured in meters |
| **Goal Accuracy (GA)** | Human judgment: is final position correct? | 5-point Likert scale (1–5) | MTurk evaluation; averaged over 5 workers |
| **Path Following (PF)** | Human judgment: does agent follow instruction trajectory? | 5-point Likert scale (1–5) | Evaluates intermediate behavior, not just final location |
| **Collision Rate** | % trajectories with collisions | Collisions ÷ Total × 100 | Safety metric; should be near 0% |

### Evaluation Protocol

```python
def evaluate_quadcopter_system(model, test_trajectories, num_trials=5):
    """
    Args:
        model: Trained SuReAL model
        test_trajectories: List of test scenarios
        num_trials: Number of physical deployments per scenario

    Returns:
        metrics: {
            'success_rate': float (0–100),
            'emd_mean': float (meters),
            'goal_accuracy_mean': float (1–5),
            'path_following_mean': float (1–5),
            'collision_rate': float (0–100),
        }
    """

    results = {
        'success': [],
        'emd_values': [],
        'collisions': [],
        'final_distances': [],
    }

    for traj in test_trajectories:
        for trial in range(num_trials):
            # Initialize quadcopter at start pose
            agent_pose = traj['start_pose']
            goal_pose = traj['goal_pose']
            language = traj['language']

            # Run episode
            trajectory_executed = []
            map_state = initialize_empty_map_state()

            for step in range(max_steps):
                # Get image + pose from quadcopter
                image = quadcopter.get_rgb()
                pose = quadcopter.get_pose()

                # Model prediction
                output = model.forward_pass(image, pose, language, map_state)
                v, omega = output['action']
                stop_logit = output['stop_logit']

                # Execute action
                quadcopter.set_velocity(v, omega)
                trajectory_executed.append(pose)
                map_state = output['map_state']

                # Check for collision
                if quadcopter.check_collision():
                    results['collisions'].append(True)
                    break

                # Check for task completion
                if sigmoid(stop_logit) > 0.5:
                    break

            # Evaluation
            final_pos = pose[:2]
            goal_pos = goal_pose[:2]
            final_distance = euclidean_distance(final_pos, goal_pose[:2])

            success = final_distance < 0.47  # 47cm threshold
            results['success'].append(success)
            results['final_distances'].append(final_distance)

            # Compute EMD
            emd = earth_mover_distance(trajectory_executed, traj['expert_trajectory'])
            results['emd_values'].append(emd)

    # Aggregate metrics
    metrics = {
        'success_rate': mean(results['success']) * 100,
        'emd_mean': mean(results['emd_values']),
        'emd_std': std(results['emd_values']),
        'collision_rate': mean(results['collisions']) * 100,
        'final_distance_mean': mean(results['final_distances']),
    }

    return metrics
```

### Human Evaluation

```
MTurk Setup:
  - Show worker: (1) instruction, (2) expert trajectory video, (3) agent trajectory video
  - Ask: "Does the agent correctly follow the instruction?" (1=No, 5=Clearly Yes)
  - Repeat for goal accuracy (final position) and path following (trajectory quality)
  - Aggregate: report mean ± std over 5 workers

Results (typical):
  - Goal Accuracy: 3.8 ± 0.6 (scale 1–5)
  - Path Following: 3.5 ± 0.8
  - Good agreement between SR and human GA (r² ≈ 0.7)
```

---

## 10. Results Summary + Ablations

### Main Results

| Experiment | SR (%) | EMD (m) | Goal Score (1–5) | Path Score (1–5) | Notes |
|-----------|--------|---------|------------------|------------------|-------|
| **SuReAL (Full)** | **30.6** | **1.23** | **3.8** | **3.5** | Full system with supervised + RL |
| **Supervised Only** | 24.3 | 1.56 | 3.2 | 2.9 | No RL fine-tuning |
| **RL Only** | 12.1 | 2.87 | 2.1 | 1.8 | No supervised pretraining (unstable) |
| **Human Performance** | 39.7 | 0.45 | 4.6 | 4.3 | Baseline upper bound |
| **Simulation Transfer** | 28.5 | 1.35 | 3.6 | 3.3 | Zero-shot to real (slight drop) |

### Ablation Studies

| Ablation | SR (%) | Δ SR | Key Finding |
|----------|--------|------|-------------|
| **Remove Semantic Map** | 22.1 | −8.5 | Semantic grounding is critical for language understanding |
| **Remove Visitation Map** | 25.3 | −5.3 | Visitation map essential for spatial planning |
| **Freeze Language Encoder** | 27.2 | −3.4 | Fine-tuning language LSTM helps |
| **Remove LSTM (use avg embedding)** | 26.8 | −3.8 | Sequence modeling captures instruction structure |
| **Remove Real Data** | 19.5 | −11.1 | Real data crucial for sim-to-real transfer |
| **Reduce Data: 50% trajectories** | 27.4 | −3.2 | Learning curve shows saturation region |

### Key Findings

1. **SuReAL > Supervised or RL Alone**: Combining supervised learning (for high-level planning) with RL (for continuous control) outperforms either approach independently.
2. **Semantic Grounding Necessary**: Language understanding must be grounded in visual concepts (colors, object types) for effective navigation.
3. **Real Data Matters**: Transfer from pure simulation results in ~8% drop in SR; even 100–200 real demos improve substantially.
4. **Sample Efficiency**: With SuReAL, physical deployment requires minimal real-world autonomous flight (mostly pre-training in simulation).

---

## 11. Practical Insights

### Engineering Takeaways

1. **Simulation Fidelity Matters**: Ensure simulator accurately models camera optics, pose estimation noise, and control latency. Realistic simulation reduces sim-to-real gap.

2. **Build Explicit Spatial Maps**: Instead of end-to-end learning of control directly, intermediate semantic + visitation maps provide interpretability and enable better generalization.

3. **Supervise the Intermediate Steps**: Use multiple supervision signals (velocity, angular velocity, visitation heatmaps, semantic maps) rather than just end-to-end control loss.

4. **Two-Phase Training**: Supervised pretraining first (fast convergence), then RL fine-tuning (better exploration) outperforms end-to-end RL.

5. **Language Embeddings**: Pre-trained GloVe embeddings (or BERT) work well; fine-tuning the embeddings helps but is not strictly necessary.

6. **Avoid End-to-End Black Boxes**: CNNs can predict (v, ω) directly from images, but adding semantic/spatial maps improves robustness and interpretability.

7. **Control Integration**: Simple PID or velocity controller on top of learned (v, ω) setpoints is more stable than end-to-end learned control.

8. **Pose Estimation Noise**: Train with noisy pose estimates (Gaussian blur in image, jitter in (x, y, θ)) to handle real odometry errors.

9. **Batch Size and Gradient Accumulation**: Use gradient accumulation if GPU memory is limited; effective batch size matters for stable training.

10. **Evaluation in Diverse Settings**: Test on both seen and unseen environments; generalization to new buildings/layouts is a critical metric.

### Gotchas & Pitfalls

1. **Quadcopter Instability**: Controlling yaw while moving forward requires careful tuning; aggressive acceleration can cause loss of control.

2. **Language Ambiguity**: Free-form instructions like "turn left" are ambiguous (left relative to what?). Need clear annotation guidelines.

3. **Sim-to-Real Domain Gap**: Simulated lighting, textures, and sensor noise differ from reality. Even with augmentation, real performance is 8–15% lower.

4. **Collision Detection Failures**: Simulator collision detection may miss narrow obstacles. Validate with conservative safety margins (e.g., expand collision geometry by 20cm).

5. **Localization Drift**: Over long trajectories (>60s), pose estimates accumulate error. Real systems need loop-closure or external localization.

6. **Training Instability with RL**: Pure RL (no supervised pretraining) can diverge. Always start with supervised behavior cloning.

7. **Overfitting to Simulation**: Models train easily on simulation data but transfer poorly. Requires domain randomization + real data for robustness.

8. **Language Preprocessing**: Tokenization and vocabulary size critically affect performance. Use consistent preprocessing between training and deployment.

9. **Gradual Sim-to-Real Transition**: Deploying on real hardware requires incremental fine-tuning with real data, not one-shot transfer.

10. **Hyperparameter Sensitivity**: Loss weights (λ_v, λ_ω, λ_stop) are task-dependent; require per-task tuning or learned weighting schemes.

### Tiny-Subset Overfit Plan

**Goal**: Overfit on 10 trajectories to verify implementation correctness.

```python
def debug_overfit_10_trajectories():
    """
    Use tiny subset to debug training pipeline
    """
    # 1. Select 10 diverse trajectories
    train_subset = train_dataset[:10]  # Very small batch

    # 2. Disable all augmentation and regularization
    model.dropout = 0.0
    model.weight_decay = 0.0
    batch_norm.momentum = 0.0  # Use batch statistics only
    learning_rate = 1e-2  # Aggressive learning

    # 3. Train without validation
    for epoch in range(1000):
        loss = train_sureal_epoch(model, train_subset, optimizer, device)
        if loss < 1e-3:
            print("✓ Model can overfit to 10 trajectories")
            break
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # 4. Check predictions on training set
    with torch.no_grad():
        for traj in train_subset:
            output = model.forward_pass(traj['image'], traj['pose'],
                                       traj['language'], map_state={})
            v_pred, omega_pred = output['action']
            v_gt, omega_gt = traj['action']

            print(f"v: pred={v_pred:.3f}, gt={v_gt:.3f}, err={abs(v_pred - v_gt):.3f}")
            print(f"ω: pred={omega_pred:.3f}, gt={omega_gt:.3f}, err={abs(omega_pred - omega_gt):.3f}")

    # 5. If overfitting fails:
    #    - Check data shapes: are images (B, 3, H, W)?
    #    - Check loss backprop: do gradients flow? (grad_norm > 0?)
    #    - Check for NaN/Inf: print intermediate activations
    #    - Verify ground truth targets are reasonable (not all zeros)
```

---

## 12. Minimal Reimplementation Checklist

### Core Components to Implement

- [ ] **CNN Image Encoder** (ResNet-18 backbone, output 256D features)
  - [ ] Load pre-trained weights (ImageNet)
  - [ ] Option to freeze or fine-tune
  - [ ] Test on dummy image: `forward(randn(1, 3, 256, 256)) → (1, 256)`

- [ ] **Language LSTM Encoder** (1–2 layers, hidden 256D)
  - [ ] GloVe embedding lookup (100D word vectors)
  - [ ] LSTM with hidden state extraction
  - [ ] Padding for variable-length sequences
  - [ ] Test: `forward(token_ids) → (1, 256)` for batch of 10 tokens

- [ ] **Semantic Grounding FCN** (fully convolutional)
  - [ ] Upsampling layers to match input spatial resolution (H×W)
  - [ ] Fusion of image features + language context
  - [ ] Softmax output for class logits
  - [ ] Test on dummy: `forward(img_feats, lang_feat) → (B, num_classes, H, W)`

- [ ] **Visitation Predictor** (2–3 conv layers)
  - [ ] Stride-1 convolutions to preserve spatial resolution
  - [ ] Sigmoid activation for heatmap
  - [ ] Test: `forward(semantic_logits) → (B, 1, H, W)` with values in [0, 1]

- [ ] **Action Heads** (velocity + angular velocity + STOP)
  - [ ] Global average pooling on spatial features → 256D vector
  - [ ] 2-layer MLP for each head
  - [ ] Test: `forward(features) → (v, ω, stop_logit)`

- [ ] **Map Projection & Warping** (warp image to top-down world frame)
  - [ ] Pinhole camera model (K matrix, intrinsics)
  - [ ] Depth estimation or constant height assumption
  - [ ] Pose-based transformation (rotation + translation)
  - [ ] Test on synthetic image + pose

- [ ] **Training Loop**
  - [ ] DataLoader with batch sampling
  - [ ] Forward + backward pass
  - [ ] Loss aggregation (multiple heads)
  - [ ] Optimizer step + gradient clipping
  - [ ] Validation loop with metrics

- [ ] **Sim2Real Transfer**
  - [ ] Domain randomization (color jitter, blur, rotation)
  - [ ] Fine-tuning on real data (small learning rate)
  - [ ] Evaluation protocol (success rate, EMD)

### Minimal Dataset Example

```json
{
  "trajectory_id": "sim_2024_001",
  "start_pose": [0.0, 0.0, 0.0],
  "goal_pose": [2.0, 1.5, 0.0],
  "language": "fly forward then turn left",
  "steps": [
    {
      "t": 0,
      "image": "base64_encoded_rgb.png",
      "pose": [0.0, 0.0, 0.0],
      "action": [0.2, 0.0],
      "action_type": "move"
    },
    {
      "t": 1,
      "image": "base64_encoded_rgb.png",
      "pose": [0.2, 0.0, 0.0],
      "action": [0.2, 0.05],
      "action_type": "turn"
    }
  ]
}
```

### Quick Validation Checklist

```
□ DataLoader yields correct shapes:
    - image: (B, 3, H, W)
    - pose: (B, 3) or (B, T, 3) for sequences
    - language_tokens: (B, T_lang)
    - actions: (B, T, 2) or (B, 2)

□ Model forward pass produces outputs in correct range:
    - v_pred ∈ [-0.5, 0.5] (m/s)
    - omega_pred ∈ [-45, 45] (deg/s)
    - stop_logit ∈ ℝ (unbounded, use sigmoid for probability)

□ Loss decreases over 100 training iterations on toy data (overfit test)

□ No NaN/Inf gradients in backward pass

□ Inference time < 100 ms on target hardware (e.g., Jetson Xavier)

□ Model checkpointing: save/load state_dict correctly

□ Evaluation metrics computed correctly:
    - Success rate: count(final_distance < 0.47m) / total
    - EMD: sklearn.metrics.wasserstein_distance(...)
```

---

## References

- **[Blukis et al. 2020]** Learning to Map Natural Language Instructions to Physical Quadcopter Control using Simulated Flight. *Conference on Robot Learning (CoRL)*, PMLR 100:1415-1438, October 2020.
- **GitHub**: [lil-lab/drif](https://github.com/lil-lab/drif)
- **arXiv**: 1910.09664
