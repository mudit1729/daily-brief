# KiGRAS: Kinematic-Driven Generative Model for Realistic Agent Simulation

## 1. One-Page Overview

| Metadata | Value |
|----------|-------|
| **Paper Title** | KiGRAS: Kinematic-Driven Generative Model for Realistic Agent Simulation |
| **Authors** | Zhao et al. (Mach Drive + USTC + Tsinghua) |
| **Venue** | arXiv:2407.12940 (July 2024) |
| **Challenge** | Waymo SimAgents Challenge 2024 (1st place, 0.7M params) |
| **Realism Score** | 0.7597 (Realism Meta-metric) |
| **Core Task** | Multi-agent trajectory generation in autonomous driving (8-second rollout) |

### Key Novelty (3-6 bullets with section citations)

1. **Action-Space Reformulation** [Sec III-B]: Reframe trajectory distribution modeling from state space P(S_{1:T}) to control action space P(U_{0:T-1}), eliminating massive redundancy. Achieves orders-of-magnitude reduction in prediction space complexity.

2. **Kinematic-Driven Constraint** [Sec III-D]: Use kinematic bicycle model K as hard constraint: S_{t+1} = K(S_t, U_t). All generated states guaranteed physically feasible. Inverse kinematics via rolling-horizon MPC extract training labels.

3. **Unified Architecture** [Sec III-D]: Single spatial encoder for agents & map lines (vectorized representation). Temporal transformer with causal attention learns P(U_t | S^w_{≤t}, S^a_{≤t}) at each step. Only 0.7M parameters.

4. **Discrete Action Binning** [Sec III-C]: 63×63 action space (acceleration [-5,5] m/s², yaw rate [-1.5,1.5] rad/s). Breaks dependency on Gaussian/Laplace priors; empirically models natural driving distributions.

5. **Post-Training Customization** [Sec III-E]: Fine-tune with Discriminative Policy Optimization (DPO) for task-specific driving preferences (Safety, Fast, Comfort) without retraining.

6. **SOTA Performance**: 0.7597 Realism meta-metric with 7-170× fewer parameters than competitors (SMART-96M, BehaviorGPT-3M, GUMP-523M).

### If You Only Remember 3 Things
1. **Predict actions, not trajectories**: Kinematic model creates bijection action↔trajectory; model only P(U) and apply K for feasibility guarantee.
2. **Rolling-horizon MPC for labels**: Extract ground-truth actions by solving trajectory-fitting MPC problem per sequence (Fig. 3).
3. **Unified + Causal = Minimal**: One spatial encoder + causal transformer on 64-dim scene tokens + discrete softmax over 3969 actions = 0.7M params, SOTA results.

---

## 2. Problem Setup and Outputs

### Input/Output Tensors

| **Category** | **Symbol** | **Dimension** | **Description** |
|--------------|-----------|---------------|-----------------|
| **Historical Agent State** | S^a_{≤0} | (N_hist, 4) | N_hist frames; (x, y, θ, v) per agent |
| **Historical World State** | S^w_{≤0} | variable | Map + other agents' states |
| **Historical Map** | S^{map} | variable | Polyline sequences (vectorized) |
| **Future Horizon** | T | scalar | 16 frames @ 2 Hz = 8 seconds |
| **Output: Action Trajectory** | U_{0:T-1} | (T-1, 2) | (acceleration, yaw_rate) at each step |
| **Output: State Trajectory** | S^a_{1:T} | (T, 4) | Future (x, y, θ, v) derived via K(·) |

### Task Formulation

**Training Objective (standard):**
```
arg max_θ P_θ(S^a_{1:T} | S^w_{≤0}, S^a_{≤0})    [Eq. 2]
```

**Reformulated Objective (KiGRAS):**
```
arg max_θ' ∏_{t=0}^{T-1} P_θ'(U^a_t | S^w_{≤t}, S^a_{≤t})

subject to: S^a_{τ+1} = K(S^a_τ, U^a_τ) for τ ∈ {0,…,t}    [Eq. 6]
```

### Scene Representation (at time t)

- **Target agent state**: (x, y, θ, v)
- **Other agents**: N nearest, each (x, y, θ, v)
- **Map lines**: Polylines as vector sequences (diff between consecutive points)
- **Traffic lights**: State per light (optional; not emphasized in ablations)
- **Agent bounding box representation**: 4 corner-point vectors (unified with map polylines)

### Data Frequency & Horizon

| Parameter | Value |
|-----------|-------|
| Historical lookback | n frames (paper: ~2-4 sec typical) |
| Training frame rate | 2 Hz |
| Prediction horizon T | 16 frames (8 sec) |
| Action update rate | 1 per frame (2 Hz) |

---

## 3. Coordinate Frames and Geometry

### Coordinate Frame Definitions

| **Frame** | **Origin/Reference** | **Usage** | **Key Transform** |
|-----------|---------------------|----------|-------------------|
| **World Frame** | Global map origin | State representation S, map lines S^{map} | Fixed throughout scenario |
| **Agent-Centric Frame** | Per agent center + heading θ | Scene encoding at each t; separate for each agent | Rotation by -θ, translation by -(x,y) |
| **Local Kinematic State** | Instantaneous S_t = (x,y,θ,v) | Intermediate in forward/inverse K | Updated via K(S_t, U_t) |

### Kinematic Model K (Bicycle Model, CTRA)

**State**: s = (x, y, θ, v) where v is forward velocity

**Control Action**: u = (a, ψ̇) where:
- a ∈ [-5, 5] m/s² (acceleration)
- ψ̇ ∈ [-1.5, 1.5] rad/s (yaw rate)

**Continuous Kinematic Update** (single timestep dt ≈ 0.5 sec):
```
ẋ = v cos(θ)
ẏ = v sin(θ)
θ̇ = ψ̇
v̇ = a
```

**Discrete Version** [from Sec III-D]:
```
s_{t+1} = K(s_t, u_t)
# Plausible: Euler integration or RK4 over dt
```

### Geometric Sanity Checks

| **Check** | **Constraint** | **Enforcement** |
|-----------|----------------|-----------------|
| **Velocity bounds** | v ≥ 0 (forward only) | Implicit in training data; may clip negative v in K forward pass |
| **Acceleration feasibility** | a ∈ [-5, 5] m/s² | Binned into 63 discrete levels; hard constraint during discrete action selection |
| **Yaw rate feasibility** | ψ̇ ∈ [-1.5, 1.5] rad/s | Binned into 63 discrete levels; no continuous yaw singularity |
| **Heading continuity** | θ should not wrap discontinuously | K updates θ additively; no explicit wrapping mentioned (likely handled in map transform) |
| **Collision-free** | Not geometrically enforced | Only via learned behavior; post-hoc evaluation with bounding boxes |

### Grid Parameters (Scene Encoding)

| **Parameter** | **Value** | **Purpose** |
|---------------|-----------|------------|
| Agent vector dimension | 4 | Corner points of bounding box |
| Map polyline vector dimension | 2+ | Consecutive point differences |
| Unified embedding dimension | 64 | After spatial encoder |
| Scene token dimension | 64 | Output of 3-layer self-attention over agents+map |
| Nearest obstacles to track | 64 | Sparsification: focus only on N=64 closest |
| Historical frames | Variable | Plausible: 2-5 seconds @ 2 Hz = 4-10 frames |

### World State Update [Eq. 8]

```
S^w_{t+1} = φ(S^{map}_t, {K(S^a_i_t, u^a_i_t)}^N_{i=1})
```

All N agents' next states computed via kinematic model with sampled actions. Map S^{map} unchanged. Then transform to each agent's agent-centric frame for next encoding.

---

## 4. Architecture Deep Dive

### ASCII Block Diagram (Forward Pass)

```
┌─────────────────────────────────────────────────────────────────┐
│                        KIGRAS Forward Pass (Training Phase)      │
└─────────────────────────────────────────────────────────────────┘

TIME STEP t:
                ┌─────────────────────┐
                │   Scene at time t   │
    Agents: {S^a_i_t}^N  │ (x,y,θ,v)  │
      Maps: S^{map}_t    │   polylines │
   Traffic: S^{light}_t  │  (optional) │
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │  Spatial Info    │  Attribute Info  │
        ▼                  ▼                  ▼
    ┌────────────┐   ┌──────────┐   ┌─────────────┐
    │Agent-Attr  │   │Map-Attr  │   │Agents/Map   │
    │Encoder     │   │Encoder   │   │Spatial Enc. │
    │(64-dim)    │   │(64-dim)  │   │(Vectorized) │
    └────┬───────┘   └────┬─────┘   └──────┬──────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────┐
        │  U-Embedding (add previous action u_{t-1})  │
        │  → fused into scene token (64-dim)          │
        └────────────────┬────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │   UNIFIED SPATIAL ENCODER      │
         │  (3-layer Self-Attention)      │
         │  Output: scene_token (64-dim)  │
         └───────────────┬────────────────┘
                         │
                    [STACK ALL T TOKENS]
                         │
         ┌───────────────▼────────────────┐
         │ AUTOREGRESSIVE TRANSFORMER     │
         │  - 3 layers causal attention   │
         │  - Input: all T scene tokens   │
         │  - Mask: only t' ≤ t visible   │
         │  - Output: T × 64 features     │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │ CONTROL ACTION DECODER HEAD    │
         │  - Linear: 64 → 3969 (63²)     │
         │  - Softmax over action space   │
         │  - Output: P(U_t | ...)        │
         └───────────────┬────────────────┘
                         │
                    [SAMPLE or ARGMAX]
                    u*_t: (a, ψ̇)
                         │
         ┌───────────────▼────────────────┐
         │ KINEMATIC FORWARD UPDATE [K]   │
         │  S^a_t, u*_t → S^a_{t+1}       │
         │  (compute next agent state)    │
         └───────────────┬────────────────┘
                         │
                    S^w_{t+1} = φ(S^{map},
                    {K(S^a_i_t, u^a_i_t)})
                         │
              ┌──────────┴──────────┐
              │ Transform to        │
              │ agent-centric frame │
              │ for next step       │
              └─────────────────────┘

[Repeat for t = 0, 1, ..., T-1]
```

### Module-by-Module Architecture Table

| **Module** | **Input Shape** | **Output Shape** | **Key Hyperparams** | **Purpose** |
|------------|-----------------|------------------|---------------------|------------|
| **Agent Attr Encoder** | (N_agents, state_dim=4) | (N_agents, 64) | FC layers (paper: not detailed) | Encode agent metadata (type, role) |
| **Map Attr Encoder** | (N_lines, attr_dim) | (N_lines, 64) | FC layers (paper: not detailed) | Encode map metadata |
| **Unified Spatial Encoder** | Agent vectors: (N, 4, 2) Map vectors: (N_lines, var, 2) | (N + N_lines, 64) | Vectorized; shared embedding | Uniform treatment of agents/map as polylines; corner-point representation |
| **U-Embedding** | u_{t-1}: (2,) | (64,) | Linear or FC | Embed previous action; concatenate to scene token |
| **Scene Encoder** | 64 scene tokens (agents+map) | (1, 64) | 3-layer self-attention, MHSA | Fuse all spatial info into one token per timestep |
| **Autoregressive Transformer** | Scene tokens: (T, 64) | (T, 64) | 3 layers MHSA, causal mask, d_model=64 | Temporal reasoning; causality prevents future leakage |
| **Action Decoder Head** | (T, 64) from transformer | (T, 63, 63) | Linear 64→3969, softmax | Output P(U_t) as discrete distribution |
| **Kinematic Model K** | (S_t, u_t): (4,) + (2,) | S_{t+1}: (4,) | CTRA model (continuous) | Forward integration S → S' |
| **World State Updater φ** | {S^a_i, u^a_i}^N, S^{map} | S^w_{t+1} | Geometric transform | Update all agents, transform to agent-centric |

### Notation Summary

- **N**: Number of agents in scene
- **T**: Prediction horizon (16 frames)
- **d_model**: 64 (uniform embedding dimension)
- **Action space size**: 63 × 63 = 3,969 bins
- **Total parameters**: 0.7M (vs. SMART-96M, BehaviorGPT-3M)

---

## 5. Forward Pass Pseudocode

```python
# KIGRAS Forward Pass (Training with Teacher Forcing)

def forward_pass(agent_states, world_state, target_agent_idx):
    """
    Args:
        agent_states: Dict[agent_id → (x, y, θ, v)]  # All N agents
        world_state: {map_lines, traffic_lights}
        target_agent_idx: int (which agent to predict for)

    Returns:
        action_logits: (T, 63, 63)  # Logits for action space
        predicted_actions: (T-1, 2)  # Sampled or argmax actions
        predicted_trajectory: (T, 4)  # Future states via K
    """

    T = 16  # prediction horizon
    d_model = 64
    n_nearest = 64

    # ─────────────────────────────────────────────────
    # STEP 1: Encode Static Attributes
    # ─────────────────────────────────────────────────

    agent_attr = agent_attr_encoder(agent_types)  # (N, 64)
    map_attr = map_attr_encoder(map_metadata)      # (N_lines, 64)

    # ─────────────────────────────────────────────────
    # STEP 2: Autoregressive Loop (t = 0 to T-1)
    # ─────────────────────────────────────────────────

    scene_tokens = []  # Store all (T, 64)
    predicted_actions = []
    predicted_trajectory = [agent_states[target_agent_idx]]  # S_0

    u_prev = zeros(2,)  # u_{-1} = [0, 0]

    for t in range(T):

        # ─── 2a. Build Agent-Centric Scene ───
        target_agent = agent_states[target_agent_idx]
        # Transform all agents to target-agent frame (rotation + translation)
        agents_local = transform_to_agent_centric(agent_states, target_agent)

        # Keep nearest 64
        agents_local_filtered = nearest_k(agents_local, k=64)  # (≤64, 4)

        # ─── 2b. Get Map Around Agent ───
        map_local = get_local_map(world_state[map], target_agent)  # Polylines

        # ─── 2c. Vectorize Spatial Info ───
        # Agents: corner points (4 per agent) → vectors
        agent_vectors = [
            [agents_local[i, :2],  agents_local[i, :2] + [1, 0],  # corners
             agents_local[i, :2] + [1, 1], agents_local[i, :2] + [0, 1]]
            for i in range(len(agents_local))
        ]  # (64, 4, 2)

        # Map: polyline vectors
        map_vectors = [
            polyline_to_vectors(polyline) for polyline in map_local
        ]  # (N_lines, var, 2)

        # ─── 2d. Spatial Encoding ───
        agent_spatial = unified_spatial_encoder(agent_vectors)  # (64, 64)
        map_spatial = unified_spatial_encoder(map_vectors)      # (N_lines, 64)

        # Combine agent and map spatial features
        all_spatial = concatenate([agent_spatial, map_spatial])  # (64+N_lines, 64)

        # ─── 2e. Attribute Encoding (broadcast) ───
        agent_attr_subset = agent_attr[0:64]  # (64, 64)
        map_attr_subset = map_attr[0:len(map_vectors)]

        # ─── 2f. Fuse Spatial + Attribute ───
        spatial_attr = concatenate([
            agent_spatial + agent_attr_subset,
            map_spatial + map_attr_subset
        ])  # (≤128, 64)

        # ─── 2g. Embed Previous Action ───
        u_embedding = action_embedding_layer(u_prev)  # (64,)

        # ─── 2h. Scene Encoder (3-layer self-attention) ───
        scene_tokens_agg = spatial_attr  # (≤128, 64)
        for layer in range(3):
            scene_tokens_agg = self_attention_layer(
                scene_tokens_agg,
                add_positional_encoding=True
            )  # (≤128, 64)

        # Pool to single token
        scene_token_t = mean_pool(scene_tokens_agg)  # (64,)

        # Add u-embedding
        scene_token_t = scene_token_t + u_embedding  # (64,)

        scene_tokens.append(scene_token_t)  # Store for transformer

    # ─────────────────────────────────────────────────
    # STEP 3: Temporal Transformer (Causal Attention)
    # ─────────────────────────────────────────────────

    scene_tokens_stacked = stack(scene_tokens)  # (T, 64)

    transformer_input = scene_tokens_stacked  # (T, 64)
    for t_layer in range(3):
        # Causal mask: position i can only attend to positions ≤ i
        causal_mask = tril(ones(T, T))  # Lower triangular

        transformer_input = self_attention_layer(
            transformer_input,
            attention_mask=causal_mask,
            add_positional_encoding=True
        )  # (T, 64)

    transformer_output = transformer_input  # (T, 64)

    # ─────────────────────────────────────────────────
    # STEP 4: Action Decoder Head
    # ─────────────────────────────────────────────────

    action_logits = action_decoder_head(transformer_output)  # (T, 3969)
    # Reshape to (T, 63, 63) for clarity
    action_logits_2d = reshape(action_logits, (T, 63, 63))

    # ─────────────────────────────────────────────────
    # STEP 5: Forward Kinematic Update
    # ─────────────────────────────────────────────────

    predicted_actions_list = []
    current_state = agent_states[target_agent_idx]  # S_0: (4,)

    for t in range(T-1):
        # During training: use ground-truth action (teacher forcing)
        # During inference: sample or argmax from action_logits[t]

        if training:
            # Teacher forcing: use ground-truth action from dataset
            u_t = ground_truth_actions[t]  # (2,) from training data
        else:
            # Inference: argmax or sample
            action_idx = argmax(action_logits[t])  # Scalar index
            a_idx, psi_dot_idx = unravel_index(action_idx, (63, 63))
            u_t = action_space[a_idx, psi_dot_idx]  # (2,)

        predicted_actions_list.append(u_t)

        # Apply kinematic model
        next_state = kinematic_model_K(current_state, u_t)  # S_{t+1}: (4,)
        predicted_trajectory.append(next_state)

        current_state = next_state
        u_prev = u_t  # For next iteration's u-embedding

    # ─────────────────────────────────────────────────
    # STEP 6: Return
    # ─────────────────────────────────────────────────

    return {
        "action_logits": action_logits_2d,          # (T, 63, 63)
        "predicted_actions": stack(predicted_actions_list),  # (T-1, 2)
        "predicted_trajectory": stack(predicted_trajectory),  # (T, 4)
        "scene_tokens": scene_tokens_stacked,       # (T, 64) [for inspection]
    }

# ─────────────────────────────────────────────────
# Kinematic Model K (CTRA: Coordinated Turn Rate Acceleration)
# ─────────────────────────────────────────────────

def kinematic_model_K(state, action, dt=0.5):
    """
    state: (x, y, θ, v)
    action: (a, ψ̇)  where a ∈ [-5, 5], ψ̇ ∈ [-1.5, 1.5]
    dt: time step (0.5 sec for 2 Hz)

    Returns:
        next_state: (x', y', θ', v')
    """
    x, y, theta, v = state
    a, psi_dot = action

    # Continuous ODE (simplified Euler)
    x_next = x + v * cos(theta) * dt
    y_next = y + v * sin(theta) * dt
    theta_next = theta + psi_dot * dt
    v_next = v + a * dt

    # Clamp velocity (optional)
    v_next = max(0, v_next)

    # Wrap heading to [-π, π]
    theta_next = wrap_angle(theta_next)

    return array([x_next, y_next, theta_next, v_next])

# ─────────────────────────────────────────────────
# Training Loss
# ─────────────────────────────────────────────────

def training_step(batch):
    """
    batch: List[(agent_states_hist, world_state, target_idx, action_labels)]

    action_labels: Extracted via inverse kinematics (MPC method) [Sec III-D]
    """

    total_loss = 0

    for sample in batch:
        agent_states, world_state, target_idx, action_labels_mpc = sample

        # Forward pass
        output = forward_pass(agent_states, world_state, target_idx)
        action_logits = output["action_logits"]  # (T, 63, 63)

        # ─── Loss Computation ───
        # action_labels_mpc: (T-1, 2) → discretized to (T-1,) indices
        action_indices = discretize_actions(action_labels_mpc)  # (T-1,)

        # Cross-entropy loss
        ce_loss = cross_entropy(
            action_logits[:-1].reshape(T-1, 3969),  # (T-1, 3969)
            action_indices                           # (T-1,)
        )

        total_loss += ce_loss

    return total_loss / len(batch)
```

### Key Design Decisions in Pseudocode

1. **Teacher Forcing During Training**: Feed ground-truth actions extracted via MPC, not predicted actions. Ensures consistency in each timestep task.

2. **Causal Attention**: Transformer can only attend to past tokens (t' ≤ t). Prevents information leakage from future.

3. **Agent-Centric Transforms**: Each prediction target gets its own coordinate frame. Enables multi-agent simulations without agent confusion.

4. **Kinematic Constraint as Hard Rule**: K is not learned; it's a deterministic forward integrator. This is the critical ingredient ensuring physical feasibility.

5. **Discrete Action Sampling**: At test time, argmax from logits or top-p sampling. No continuous sampling; discrete bins enforce realistic action distributions.

---

## 6. Heads, Targets, and Losses

### Prediction Heads Table

| **Head** | **Output Shape** | **Activation** | **Target Labels** | **Loss Function** |
|----------|------------------|----------------|-------------------|-------------------|
| **Control Action Decoder** | (T, 63, 63) or (T, 3969) | Softmax over action bins | Ground-truth actions from MPC inverse kinematics | Cross-Entropy (CE) |

### Loss Terms and Formulas

| **Loss Type** | **Formula** | **Weight/Scale** | **When Applied** | **Notes** |
|---------------|-----------|------------------|------------------|-----------|
| **Cross-Entropy (CE)** | L_CE = -∑_t ∑_{a ∈ U} [a == a*_t] log(P_θ(a \| S^w_{≤t}, S^a_{≤t})) | 1.0 (primary) | Every training step | Supervises action probability distribution; only positive class (ground-truth action) contributes gradient |
| **Kinematic Constraint** | None (hard constraint) | ∞ (enforced deterministically) | Forward pass (no gradient) | S_{t+1} = K(S_t, U_t) is deterministic; no learnable params in K |
| **DPO Loss** (fine-tuning only) | L_DPO = -E_{(x,y_w,y_l) ~ D} [ log σ(β log π_θ(y_w\|x) / π_ref(y_w\|x) - β log π_θ(y_l\|x) / π_ref(y_l\|x)) ] | β = 1.0 | Fine-tuning phase (Section IV-E) | Preference optimization; encourages policy π_θ to prefer winner y_w over loser y_l relative to π_ref |

### Action Space Assignment Strategy

**During Training** (Labels from Inverse Kinematics):

1. **Rolling Horizon MPC** [Eq. 7, Sec III-D, Fig. 3]:
   ```
   min_{u_{t:t+k}} ∑_{τ=t}^{t+k} ||s^ctl_{τ+1} - s_{τ+1}||²

   subject to: s^ctl_{τ+1} = K(s_τ, u_τ)
   ```
   - Solve k-step control sequence to match observed trajectory
   - Use previous solution s^ctl*_t as warm-start for stability
   - Output: u'*_t (optimal continuous action)

2. **Discretization**:
   - Find nearest bin in 63×63 discrete grid: u*_t = argmin_{u ∈ U} ||u'*_t - u||²
   - If multiple actions equally close, pick first in canonical order

3. **Label Extraction**:
   - Per training trajectory: extract U*_{0:T-1} via rolling MPC
   - These become ground-truth labels for CE loss

**During Inference**:

- **Argmax (Deterministic)**: u_t = argmax_{u} P_θ(u | S^w_{≤t}, S^a_{≤t})
- **Top-p Sampling** (for diversity): Sample from top p% of action probabilities [used in Section IV-E with p=0.9]

### Loss Debugging Checklist

| **Issue** | **Diagnostic** | **Likely Cause** | **Fix** |
|-----------|----------------|-----------------|--------|
| **CE Loss >> 1.0 (not improving)** | Check action_logits distribution; validate MPC labels accurate | MPC solver stuck or wrong formulation | Verify kinematic model K matches actual vehicle; tune MPC horizon k |
| **CE Loss drops fast, then plateaus** | Early overfitting to seen trajectories | Insufficient data aug; task imbalance (some agents more predictable) | Add augmentations [Sec VII]; weight samples by diversity |
| **Predicted trajectory diverges from target** | Run forward pass with ground-truth actions; compare output trajectory to label | Kinematic model K has bug (e.g., angle wrapping, sign error) | Print intermediate states (x, y, θ, v) step-by-step; verify dt integration |
| **Action logits all uniform** | Transformer outputs zero gradients or low variance | Causal mask broken; no supervision signal reaching decoder | Check causal_mask is lower-triangular; validate CE loss backprop reaches action head |
| **DPO fine-tuning unstable (loss → ∞)** | Winning/losing sample selection too aggressive | Expert rules select unrealistic trajectories (e.g., collision vs non-collision too extreme) | Inspect sample pairs; choose closer margins (faster vs slower, not collision vs non-collision) |
| **Prediction trajectory self-collides** | Visualize x,y positions over time; check bounding box overlap | Action bins have poor coverage of deceleration; K model over-integrates | Verify action grid uniform coverage; reduce dt or subdivide acceleration range |

---

## 7. Data Pipeline and Augmentations

### Dataset Details

| **Aspect** | **Value** |
|-----------|-----------|
| **Source** | Waymo Motion Dataset v1.2 [30] |
| **Total Scenarios** | 103,354 real human driving scenarios |
| **Training/Val/Test Split** | Not explicitly stated (Plausible default: 70/10/20) |
| **Frequency** | 2 Hz (0.5 sec per frame) |
| **Modalities** | LiDAR, Camera, HD Map |
| **Scenario Duration** | ~10 sec each (historical + future) |
| **Agents per Scene** | Typically 2-50 (all dynamic agents + maps) |

### Augmentations Table

| **Augmentation** | **Type** | **Parameter Range** | **Applied** | **Purpose** |
|-----------------|---------|---------------------|-----------|------------|
| **Agent Type Masking** | Categorical | Plausible default: mask 0-50% agent type info | Training | Robustness to agent classification errors |
| **Noise on State** | Gaussian | Plausible: σ ∈ [0.05, 0.2] m on (x,y), [0.01, 0.05] rad on θ | Inference time | Simulate sensor noise; test robustness |
| **Speed Jitter** | Gaussian | Plausible: σ ∈ [0.1, 0.5] m/s on v | Training | Handle tracking uncertainty |
| **Dropout on Agents** | Random | Plausible: drop 10-30% of non-target agents | Training | Robustness to missing agent detections |
| **Map Polyline Simplification** | Geometric | Plausible: remove ≤ 20% of consecutive points | Training | Sparse map representation; reduce memory |
| **Rotation (Ego-Centric)** | Geometric | Plausible: rotate entire scene by θ ∈ [0, 2π] | Training | Invariance to heading; improves generalization |
| **Translation** | Geometric | Plausible: shift scene by (δx, δy) ∈ [-10, 10] m | Training | Translational invariance |
| **Scale** | Geometric | Plausible: scale distances by factor ∈ [0.8, 1.2] | Not mentioned | May not apply (absolute velocity units) |

### Augmentation Safety Checks

| **Check** | **Constraint** | **Enforcement** |
|----------|----------------|-----------------|
| **No label corruption** | Augmentation applied to inputs only, not action labels | Actions extracted pre-augmentation; applied consistently at train/test |
| **Velocity consistency** | Noise on (x, y, θ) consistent with v | If state noise added, recompute v = ||Δx/Δt||; don't double-apply |
| **Agent identity preservation** | Target agent index invariant to permutation | Store agent_id separately; don't shuffle agent order mid-trajectory |
| **Kinematic feasibility post-augment** | Action noise doesn't violate bounds | Noise applied to state space, not action space; K ensures feasibility |
| **Temporal consistency** | Same augmentation seed per trajectory | Use fixed random seed per sequence to avoid state jitter flipping signs |

### No Explicit Augmentations Mentioned in Paper

The paper **does not detail data augmentations** in the main text or experiments. The above table reflects **plausible defaults** based on common practices in trajectory prediction (inspired by SMART, BehaviorGPT, TrafficBots).

**Actual augmentations (if any) likely in closed codebase or appendix.**

---

## 8. Training Pipeline

### Hyperparameters Table

| **Hyperparameter** | **Value** | **Category** | **Rationale / Notes** |
|-------------------|-----------|--------------|----------------------|
| **Batch Size** | 256 | Optimizer | Large batch for stable gradient estimates over diverse scenarios |
| **Learning Rate** | 2e-4 | Optimizer | OneCycleLR schedule; typical for transformer-based models |
| **LR Scheduler** | OneCycleLR | Optimizer | Warmup → peak → decay; helps escape local minima |
| **Optimizer** | Not specified | - | Plausible default: AdamW |
| **Weight Decay** | Not specified | Regularization | Plausible default: 1e-4 |
| **Gradient Clipping** | Not specified | Stability | Plausible default: max_norm=1.0 |
| **Training Iterations** | 1.5M (for ablation) | Duration | ~1-2 epochs on 103K scenarios |
| **Validation Frequency** | Not specified | Monitoring | Plausible: every 10K steps |
| **Num Layers (Spatial Encoder)** | 3 | Architecture | Per Sec IV-B |
| **Num Layers (Transformer)** | 3 | Architecture | Causal attention layers |
| **Embedding Dimension (d_model)** | 64 | Architecture | Compact; balanced with 0.7M param budget |
| **Num Attention Heads** | Not specified | Architecture | Plausible default: 4 (64 / 4 = 16-dim per head) |
| **Action Space Bins** | 63 × 63 | Discretization | Covers acceleration [-5, 5] m/s² and yaw rate [-1.5, 1.5] rad/s |
| **Nearest Obstacles to Track** | 64 | Scene Encoding | Sparsification; prunes distant agents |
| **Historical Window** | Variable (Plausible: 2-5 sec) | Input | Feed enough history for context |
| **Prediction Horizon T** | 16 frames (8 sec @ 2 Hz) | Task | Match Waymo benchmark |
| **Rolling Horizon MPC Window k** | Not specified | Inverse Kinematics | Plausible: k ∈ [4, 8] frames for stability |
| **Teacher Forcing Ratio** | 1.0 (always ground truth) | Training | Use MPC labels during training; no scheduled sampling |
| **DPO β Parameter** | 1.0 | Fine-tuning | Weight for preference learning; scaling factor in Eq. 9 |

### Stability & Convergence Table

| **Aspect** | **Status** | **Metric / Evidence** | **Notes** |
|-----------|-----------|----------------------|-----------|
| **Convergence** | ✓ Achieved | CE loss: 1.966 → 1.786 over 1.5M iterations (9.2% drop per ablation) | Smooth downward trend; no divergence reported |
| **Gradient Flow** | ✓ Healthy | Ablations show measurable CE improvements per component | Each module contributes; no dead layers |
| **Training Stability** | ✓ Stable | No loss spikes, NaN, or divergence reported | OneCycleLR + batch size 256 likely sufficient |
| **Validation Generalization** | ✓ Good | Test Realism=0.7597; outperforms peers with 7-170× fewer params | No evidence of severe overfitting |
| **Kinematic Constraint Enforcement** | ✓ Enforced | All predicted trajectories satisfy S_{t+1} = K(S_t, U_t) by construction | Hard constraint; no learning required |
| **Action Space Discretization** | ✓ Adequate | Discrete 63×63 bins cover realistic driving; ablation not needed | Paper notes flexibility; can be refined if needed |
| **Teacher Forcing Robustness** | ✓ Safe | Consistent task at each timestep; no distribution shift between train/test | Encoder sees consistent input distribution |

### Training Dynamics & Key Insights

1. **Loss Decomposition** (Table III):
   - Base model (no ablations): CE = 1.966
   - +Causal-Attn: CE = 1.879 (Δ -0.087, -4.4%)
   - +USR: CE = 1.839 (Δ -0.04, -2.1%)
   - +U-Embedding: CE = 1.786 (Δ -0.053, -2.8%)
   - **Total improvement: -9.2% over baseline**

2. **Causal Attention Most Impactful**: 4.4% loss reduction suggests temporal ordering/causality critical for action prediction.

3. **No Reported Overfitting Mitigation**: Beyond implicit regularization (batch norm, dropout in transformer). **Data augmentation likely missing from paper writeup.**

4. **DPO Fine-Tuning Stability** (Section IV-E):
   - Sampled 256 trajectories per target; selected winner/loser per preference rule
   - No reported instability; suggests sample selection (collision vs non-collision) sufficiently separated
   - Drove observable behavior changes (Safety: -1.6 collisions/8s; Fast: +0.49 m/s speed)

---

## 9. Dataset + Evaluation Protocol

### Dataset Details (Repeated from Section 7)

| **Property** | **Value** |
|--------------|-----------|
| **Name** | Waymo Motion Dataset v1.2 |
| **Scenarios** | 103,354 |
| **Modalities** | LiDAR, camera, HD maps |
| **Frequency** | 2 Hz |
| **Agents per Scenario** | Typically 2-50 (variable) |
| **Closed-Loop Horizon** | 8 seconds (16 frames @ 2 Hz) |

### Splits

| **Split** | **Scenarios** | **Usage** |
|-----------|--------------|----------|
| **Train** | ~72,348 (Plausible 70%) | Model training; action labels via MPC |
| **Validation** | ~10,335 (Plausible 10%) | Hyperparameter tuning; CE loss monitoring |
| **Test** | ~20,671 (Plausible 20%) | Official Waymo SimAgents leaderboard evaluation |

(Exact splits not stated in paper; above is reasonable assumption.)

### Evaluation Metrics (Waymo SimAgents Challenge)

| **Metric** | **Dimension** | **Formula / Definition** | **Higher/Lower Better** | **Key Insights from KiGRAS Results** |
|-----------|---------------|---------------------------|------------------------|--------------------------------------|
| **REALISM (Real.)** | Overall | Weighted combination of Kin., Inter., Map. | ↑ Higher better | **0.7597** (SOTA; +0.56% vs SMART-7M) |
| **Kinematic (Kin.)** | Physics | Speed, acceleration, jerk violations | ↑ Higher | **0.4691** (competitive; no collision constraint in reward) |
| **Interaction (Inter.)** | Multi-agent | Agent proximity, collision rate, yielding fidelity | ↑ Higher | **0.8064** (best among comparators) |
| **Map-Related (Map.)** | Lane Keeping | Distance to lane center, off-road rate | ↑ Higher | **0.8658** (best; constrained to road graph) |
| **minADE** | Distance | Min Average Displacement Error (over K best-of-many predictions) | ↓ Lower | **1.4383** m (competitive; slightly worse than SMART-7M @ 1.4062) |
| **Acceleration (Acc)** | Kinematics | Avg absolute acceleration (m/s²) | Report only | 0.402 (PDriver); natural range ±2 m/s² typical |
| **Jerk** | Kinematics | Avg jerk (m/s³) | ↓ Lower (smoother) | 0.065 (PDriver); 0.051 (CDriver) after fine-tuning |
| **Collision Rate (%)** | Safety | Frequency of bounding-box overlaps | ↓ Lower | 7.959‰ (0.7959%) @ 8s (PDriver); 6.354‰ (SDriver) |

### Evaluation Protocol

1. **Closed-Loop Simulation**:
   - Run all agents' predictions forward for 8 seconds
   - At each timestep, sample or argmax actions from model
   - Update world state via kinematic model K
   - Re-encode scene for next prediction

2. **Multi-Agent Consistency**:
   - All N agents predicted in parallel (Section III-D, Eq. 8)
   - World state φ updates all agents; transform to agent-centric frame
   - No hierarchical or sequential prediction; all agents independent

3. **Metric Aggregation**:
   - Per scenario: compute all metrics for all agents
   - Macro-average: mean over scenarios
   - Weighted meta-metric (exact weights not disclosed, likely: Realism = 0.5×Kin + 0.25×Inter + 0.25×Map)

### Qualitative Evaluation (Section IV-D, Fig. 4)

Four representative scenarios tested:
- **(a) Multi-agent interaction**: Vehicle yields to pedestrian; tests interaction modeling
- **(b) Lane keeping**: Vehicles maintain lanes on complex roads; tests map compliance
- **(c) Unprotected turn**: Vehicles yield; tests interaction + decision-making
- **(d) Corner case**: Vehicle parked against traffic flow (rare in training); tests generalization

**Key Observation**: Model generalizes to out-of-distribution scenario (d); demonstrates robustness.

---

## 10. Results Summary + Ablations

### Main Results (Table I: SimAgents Challenge Leaderboard)

| **Method** | **Params** | **Real.** | **Kin.** | **Inter.** | **Map.** | **minADE** | **Rank** |
|-----------|-----------|----------|---------|-----------|---------|-----------|----------|
| **KiGRAS** | 0.7M | **0.7597** | **0.4691** | **0.8064** | **0.8658** | 1.4383 | **1st** |
| SMART-7M | 7M | 0.7591 | 0.4759 | 0.8039 | 0.8632 | **1.4062** | 2nd |
| BehaviorGPT | 3M | 0.7473 | 0.4333 | 0.7997 | 0.8593 | 1.4147 | 3rd |
| GUMP | 523M | 0.7431 | 0.4780 | 0.7887 | 0.8359 | 1.6041 | 4th |
| MVTE | ? | 0.7302 | 0.4503 | 0.7706 | 0.8381 | 1.6770 | 5th |
| VBD | 12M | 0.7200 | 0.4169 | 0.7819 | 0.8137 | 1.4743 | 6th |
| TrafficBOTv1.5 | 10M | 0.6988 | 0.4304 | 0.7114 | 0.8360 | 1.8825 | 7th |

**Key Insight**: KiGRAS wins overall (0.7597) with **0.7M params** (10× fewer than SMART-7M, 750× fewer than GUMP). Slight minADE loss vs SMART-7M (1.4383 vs 1.4062, +0.023) but wins critical metrics (Interaction +0.0025, Map +0.0026).

### Top 3 Ablations (Table III)

| **ID** | **Causal-Attn** | **USR** | **U-Embedding** | **CE Loss** | **Insight** |
|--------|-----------------|---------|-----------------|-----------|------------|
| 1 (Base) | ✗ | ✗ | ✗ | 1.966 | Baseline: no temporal structure, no action history, no unified encoding |
| 2 | ✓ | ✗ | ✗ | 1.879 | **Δ -0.087 (-4.4%)**: Temporal causality is critical; causal mask prevents information leakage and helps model learn conditional independence |
| 3 | ✓ | ✓ | ✗ | 1.839 | **Δ -0.04 (-2.1%)**: Unified spatial encoding (shared representation for agents/maps) improves feature interaction; ~50% smaller loss gain than causal attention |
| 4 (Full) | ✓ | ✓ | ✓ | 1.786 | **Δ -0.053 (-2.8%)**: Previous action embedding provides momentum context; most localized impact (smaller gain than other two) |

### Ablation Insights

1. **Causal Attention is MVP**: Largest single contributor (-4.4%). Suggests temporal ordering of actions is fundamental; without it, model struggles to learn Markovian dynamics.

2. **Unified Spatial Representation Effective**: -2.1% shows that treating agents and maps as geometric objects (polylines + vectors) enables better feature sharing. Contrast with separate pathways in BehaviorGPT/SMART.

3. **U-Embedding Additive, Not Critical**: -2.8% suggests action history is useful but not essential. Model learns implicit action patterns via transformer; explicit embedding refines by ~2-3%.

4. **No Ablation on Kinematic Constraint K**: All runs enforce K; no "learned next state" baseline. Paper argues K is essential feature, not a hyperparameter to ablate.

### Performance Over Baselines

**Why KiGRAS Wins Despite Fewer Parameters**:

1. **Action Space Compression**: 3,969-dim action space far smaller than trajectory space (1000s of continuous dimensions in SMART/BehaviorGPT regressive outputs).

2. **Kinematic Causality**: Hard K constraint eliminates massive trajectory redundancy; model only needs to learn P(U), not P(S) ∩ physical feasibility.

3. **Unified Architecture**: Single encoder for all scene elements; minimal task-specific heads. SMART/BehaviorGPT likely have separate pathways (agent prediction, map interaction, social pooling).

4. **Teacher Forcing Consistency**: Each timestep uses ground-truth history (not predicted states). Reduces compounding error and distributional shift.

---

## 11. Practical Insights

### 10 Engineering Takeaways

1. **Inverse Kinematics for Labels is Brilliant**: Rolling-horizon MPC (Fig. 3, Eq. 7) extracts ground-truth actions automatically from trajectories. Eliminates need for manual action annotation or heuristics. Key to scaling.

2. **Discrete Action Space > Continuous Outputs**: 63×63 bins (3,969 classes) eliminates assumption that driving follows Gaussian distribution. Softmax over discrete bins directly models empirical action distribution.

3. **Causal Attention is Non-Negotiable**: 4.4% CE loss reduction is the single biggest ablation win. Must mask future tokens during training to learn conditional dependencies correctly.

4. **Agent-Centric Frames Enable Multi-Agent Modeling**: Transforming each agent's local frame allows batch-processing different agents without permutation invariance overhead. Critical for N-agent simulation.

5. **Vectorized Geometry Unifies Representation**: Treating agent bounding boxes and map polylines as vector sequences (corner points and line segments) enables shared spatial encoder. Elegant generalization.

6. **U-Embedding (Previous Action) Provides Momentum Context**: Small 2.8% gain but important for predicting smooth action sequences. Models natural continuity; vehicle doesn't flip acceleration signs every frame.

7. **Teacher Forcing Throughout Training**: No curriculum or annealing of teacher forcing. Always use ground-truth states/actions during training; inference uses predictions. Simplifies implementation; avoids distribution mismatch debugging.

8. **World State Update φ After Each Agent**: Equation 8 ensures all agents update consistently. Prevents temporal inconsistency (e.g., vehicle A stops, vehicle B still sees A moving). Cheap to implement; critical for believable multi-agent scenes.

9. **Scene Token Pooling: Mean > Attention**: Paper pools 64+ spatial features to single (64,) scene token via mean. Likely simpler, faster than learned attention-based pooling. Works because self-attention already captures inter-dependencies.

10. **Nearest-64 Sparsification is Adequate**: Focusing only on 64 nearest obstacles removes 90%+ of potential agents (typical scenes have thousands of road users in coverage area). No performance loss; significant memory savings.

### 5 Common Gotchas

1. **Kinematic Model Singularities**: Heading θ wraps at ±π. If not careful, Δθ can jump from +π to -π (discontinuity). Must use angle wrapping (atan2) in K and any loss that compares headings. Easy to miss in initial implementation.

2. **MPC Rolling Horizon Drift**: Solving k-step optimization independently at each t can accumulate error if k is too small (e.g., k=2 looks only 1 sec ahead, misses curves). Paper doesn't specify k; guess k=4-8 frames. Validate on a single trajectory.

3. **Action Discretization Misalignment**: If discrete action grid not uniformly spaced in [a_min, a_max] × [ψ̇_min, ψ̇_max], coverage is nonuniform. Decelerations may under-represented if bins cluster near zero. Use linspace(a_min, a_max, 63) not ad-hoc bins.

4. **Teacher Forcing Leakage in Validation**: If validation also uses teacher forcing (ground-truth actions), CE loss on validation set looks artificially low (model hasn't learned prediction, only Q-function). Must validate with predicted actions to catch overfitting.

5. **Agent Identity Confusion in Batching**: When stacking multiple scenarios, agent_id must be unique per batch (e.g., (scenario_id, agent_id)). Easy to accidentally reuse agent_id 0 across scenarios, causing embedding collisions or identity swaps in attention.

### Tiny-Subset Overfit Plan

**Goal**: Verify all components work end-to-end before large-scale training.

```
STEP 1: Prepare Minimal Dataset
  - Take 1 driving scenario (multi-agent, 10 sec)
  - Extract 10 unique target agents
  - Total: 10 training samples
  - Manual visual inspection: ensure ground truth correct

STEP 2: Extract Action Labels via MPC
  - Run inverse kinematics on all 10 samples
  - Inspect 3 trajectories manually: verify MPC solutions reasonable
  - Check u*_t fits within action bins (no out-of-range values)

STEP 3: Build Minimal Model
  - d_model = 32 (instead of 64)
  - 1 layer spatial encoder, 1 layer transformer
  - Total params: ~10K
  - Train to CE loss → 0 (should overfit tiny set)

STEP 4: Verify Forward Pass
  - Confirm action_logits shape: (T, 3969) ✓
  - Predicted trajectory shape: (T, 4) ✓
  - Kinematic constraint: visualize S_{t+1} = K(S_t, U_t) ✓
  - No NaN in loss ✓

STEP 5: Check Gradient Flow
  - Print gradients at each layer: all non-zero ✓
  - Causal mask working: future tokens masked ✓
  - Loss decreasing: first 10 steps ✓

STEP 6: Inference Sanity
  - Run closed-loop inference for 8 sec
  - Trajectory should remain in scene bounds ✓
  - Speed should not become negative ✓
  - Yaw rate changes should be smooth (no 360° flips) ✓

STEP 7: Validate Kinematic Model K
  - Unit test K on known motions (straight line, 90° turn)
  - Verify K is invertible for small Δt (no singularities)
  - Check numerical stability over T=16 steps

STEP 8: Scale to Full Data
  Once 10-sample overfit works, scale to 1K samples (1 scenario with 100 agents or 10 scenarios with 10 agents each). Validate loss curve is smooth; no divergence.
```

---

## 12. Minimal Reimplementation Checklist

### Build Order (Dependency Graph)

```
Level 1 (Foundations):
  ✓ Kinematic Model K (CTRA)
    - forward_integrate(state, action, dt) → next_state
    - angle_wrap() helper
    - Unit tests: straight line, turn, negative velocity clamp

  ✓ Discretization Utils
    - continuous_to_discrete_action(a, psi_dot) → (i, j) bin indices
    - discrete_to_continuous_action(i, j) → (a, psi_dot) values
    - action_bins = create_action_space(63, [-5,5], [-1.5,1.5])

  ✓ Inverse Kinematics (MPC Rolling Horizon)
    - mpc_rolling_horizon(trajectory_states, horizon=4) → actions
    - Uses scipy.optimize or custom optimizer
    - Depends on: K (level 1)

Level 2 (Scene Encoding):
  ✓ Spatial Encoder
    - vectorize_agent_bounding_box(state) → (4, 2) corner vectors
    - vectorize_map_polyline(polyline) → (N, 2) segment vectors
    - spatial_encoder_layer(vectors) → (64,) embeddings
    - Depends on: PyTorch nn.Module, torch.nn.attention (self-attention)

  ✓ Attribute Encoders
    - agent_attr_encoder(agent_type, traffic_role) → (64,)
    - map_attr_encoder(line_type, lane_id) → (64,)
    - Simple MLPs

  ✓ U-Embedding
    - action_embedding(u_prev) → (64,)
    - Linear layer: (2,) → (64,)

Level 3 (Architecture):
  ✓ Scene Encoder (Temporal)
    - 3-layer self-attention module
    - Pool to single (64,) token per timestep
    - Depends on: Spatial encoder (L2), attention module

  ✓ Autoregressive Transformer
    - 3 transformer blocks with causal masking
    - Input: (T, 64) scene tokens
    - Output: (T, 64) hidden states
    - Depends on: PyTorch Transformer (or custom attention)

  ✓ Action Decoder Head
    - Linear(64, 3969) + softmax
    - Output: (T, 63, 63) or (T, 3969) logits
    - Depends on: PyTorch nn.Linear

Level 4 (Training):
  ✓ Loss & Optimization
    - cross_entropy_loss(action_logits, action_targets)
    - OneCycleLR scheduler
    - AdamW optimizer
    - Gradient clipping (max_norm=1.0)

Level 5 (Inference):
  ✓ Closed-Loop Simulator
    - forward_pass(scene_t) → action_logits_t
    - argmax or sample actions
    - kinematic_update(state, action) → next_state
    - transform_to_agent_centric(world_state, target_agent)
    - Depends on: All above levels

Level 6 (Evaluation):
  ✓ Metrics
    - compute_ade(pred_trajectory, gt_trajectory)
    - compute_collision_rate(trajectory, bounding_boxes)
    - compute_jerk(trajectory)
    - compute_map_compliance(trajectory, road_graph)
```

### Unit Tests Table

| **Component** | **Test Case** | **Expected Behavior** | **Critical?** |
|---------------|---------------|----------------------|--------------|
| **Kinematic Model K** | Straight line, zero acceleration, small dt | x, y advance by v·dt; θ constant | ✓✓ Critical |
| **Kinematic Model K** | 90° turn, constant yaw_rate | θ increases by ψ̇·dt per step | ✓✓ Critical |
| **Kinematic Model K** | Negative velocity input | v clipped to 0; state doesn't reverse | ✓✓ Critical |
| **Kinematic Model K** | Large dt, high acceleration | No overflow; heading wraps correctly | ✓ Important |
| **Action Discretization** | a=0, ψ̇=0 → bin indices | Should map to center bin (31, 31) | ✓✓ Critical |
| **Action Discretization** | a=5 m/s², ψ̇=-1.5 rad/s | Correct corner of action grid | ✓ Important |
| **Inverse Kinematics MPC** | Straight line trajectory | MPC returns zero acceleration, zero yaw_rate | ✓✓ Critical |
| **Inverse Kinematics MPC** | Figure-8 trajectory | MPC actions non-constant; match k-step horizon output | ✓ Important |
| **Spatial Encoder** | Agent bounding box input | Output shape (4, 64) for 4 corners | ✓✓ Critical |
| **Spatial Encoder** | Map polyline input | Output shape (N_lines, 64) | ✓✓ Critical |
| **Scene Encoder (Pool)** | 64 spatial features → 1 token | Output shape (1, 64); no NaN | ✓ Important |
| **Transformer Causal Mask** | Position t attends to t-1 only | Future positions masked (log(0) = -∞) | ✓✓ Critical |
| **Transformer Causal Mask** | Position 0 attends to nothing | Works (self-attention only) | ✓ Important |
| **Action Decoder Head** | (T, 64) input → (T, 3969) logits | Shape correct; sum(softmax) = 1 per t | ✓✓ Critical |
| **Cross-Entropy Loss** | Correct action index (one-hot) | Loss → 0 (theoretical) | ✓ Important |
| **Cross-Entropy Loss** | Random action index | Loss ≈ -log(1/3969) ≈ 8.3 nats | ✓ Important |
| **World State Update φ** | All agents move, update frames | Agent-centric transform consistent; no permutation errors | ✓✓ Critical |
| **Teacher Forcing** | Ground-truth actions input | Predicted trajectory close to ground truth | ✓✓ Critical |
| **Inference (No Teacher)** | Predicted actions only | Closed-loop rollout completes without NaN | ✓ Important |
| **Collision Detection** | Overlapping bboxes | Returns True | ✓ Important |

### Minimal Sanity Scripts

```python
# ─────────────────────────────────────────────────────────────
# Script 1: Verify Kinematic Model K
# ─────────────────────────────────────────────────────────────

def test_kinematic_model():
    from kigras.kinematic import kinematic_model_K, wrap_angle

    # Test 1: Straight line motion
    state = [0, 0, 0, 10]  # x, y, θ, v (moving east at 10 m/s)
    action = [0, 0]         # no accel, no yaw

    for t in range(5):
        state = kinematic_model_K(state, action, dt=0.5)
        assert state[0] >= 0, "X should increase"
        assert state[1] == 0, "Y should stay 0"
        assert state[3] == 10, "Velocity should be constant"

    print("✓ Test 1 passed: straight line")

    # Test 2: Right turn
    state = [0, 0, 0, 5]
    action = [0, 0.5]  # yaw_rate = 0.5 rad/s (right turn)

    initial_theta = state[2]
    for t in range(4):
        state = kinematic_model_K(state, action, dt=0.5)

    final_theta = state[2]
    expected_delta_theta = 0.5 * 0.5 * 4  # ψ̇ * dt * steps
    actual_delta_theta = final_theta - initial_theta

    assert abs(actual_delta_theta - expected_delta_theta) < 0.01, \
        f"Expected Δθ={expected_delta_theta}, got {actual_delta_theta}"

    print("✓ Test 2 passed: right turn")

    # Test 3: Acceleration
    state = [0, 0, 0, 0]  # stationary
    action = [2, 0]        # accel 2 m/s²

    for t in range(5):
        state = kinematic_model_K(state, action, dt=0.5)

    expected_v = 0 + 2 * 0.5 * 5  # a * dt * steps = 5 m/s
    actual_v = state[3]

    assert abs(actual_v - expected_v) < 0.01, \
        f"Expected v={expected_v}, got {actual_v}"

    print("✓ Test 3 passed: acceleration")

# ─────────────────────────────────────────────────────────────
# Script 2: Verify Action Discretization
# ─────────────────────────────────────────────────────────────

def test_action_discretization():
    from kigras.utils import create_action_space, continuous_to_discrete, discrete_to_continuous

    action_space = create_action_space(n_bins=63, a_range=[-5, 5], psi_dot_range=[-1.5, 1.5])

    # Test 1: Center bin
    a, psi_dot = 0, 0  # Centered actions
    i, j = continuous_to_discrete(a, psi_dot, action_space)
    assert i == 31 and j == 31, f"Center should be (31, 31), got ({i}, {j})"
    print("✓ Test 1: center action maps to center bin")

    # Test 2: Extremes
    a, psi_dot = 5, 1.5  # Max accel, max yaw
    i, j = continuous_to_discrete(a, psi_dot, action_space)
    assert i == 62 and j == 62, f"Max should be (62, 62), got ({i}, {j})"
    print("✓ Test 2: max actions map to max bin")

    # Test 3: Round-trip
    a_orig, psi_dot_orig = 2.3, -0.8
    i, j = continuous_to_discrete(a_orig, psi_dot_orig, action_space)
    a_reconstructed, psi_dot_reconstructed = discrete_to_continuous(i, j, action_space)

    assert abs(a_reconstructed - a_orig) < 0.16, "Discretization loss acceptable"
    assert abs(psi_dot_reconstructed - psi_dot_orig) < 0.05, "Discretization loss acceptable"
    print("✓ Test 3: round-trip discretization within tolerance")

# ─────────────────────────────────────────────────────────────
# Script 3: Verify Inverse Kinematics (MPC)
# ─────────────────────────────────────────────────────────────

def test_inverse_kinematics():
    from kigras.inverse_kinematics import mpc_rolling_horizon
    from kigras.kinematic import kinematic_model_K
    import numpy as np

    # Create a straight-line ground-truth trajectory
    n_steps = 10
    traj = []
    state = [0, 0, 0, 10]  # Start: (0,0), east, 10 m/s

    for _ in range(n_steps):
        traj.append(state.copy())
        state = kinematic_model_K(state, [0, 0], dt=0.5)  # Straight line

    # Run MPC to recover actions
    actions_recovered = mpc_rolling_horizon(traj, horizon=4)

    # Check: actions should be ~[0, 0]
    assert len(actions_recovered) == n_steps - 1

    for i, (a, psi_dot) in enumerate(actions_recovered):
        assert abs(a) < 0.5, f"Step {i}: accel should be ~0, got {a}"
        assert abs(psi_dot) < 0.1, f"Step {i}: yaw_rate should be ~0, got {psi_dot}"

    print("✓ Test: MPC recovered near-zero actions for straight-line trajectory")

# ─────────────────────────────────────────────────────────────
# Script 4: Verify Spatial Encoder Output
# ─────────────────────────────────────────────────────────────

def test_spatial_encoder():
    from kigras.model import UnifiedSpatialEncoder
    import torch

    encoder = UnifiedSpatialEncoder(input_dim=2, output_dim=64)

    # Test 1: Agent corner vectors
    agent_corners = torch.randn(4, 2)  # 4 corners, 2 coords each
    output = encoder(agent_corners)

    assert output.shape == (4, 64), f"Expected (4, 64), got {output.shape}"
    assert not torch.isnan(output).any(), "No NaN in output"
    print("✓ Test 1: agent encoding shape correct")

    # Test 2: Map polyline vectors
    polyline = torch.randn(20, 2)  # 20-point polyline
    output = encoder(polyline)

    assert output.shape == (20, 64), f"Expected (20, 64), got {output.shape}"
    assert not torch.isnan(output).any(), "No NaN in output"
    print("✓ Test 2: map encoding shape correct")

# ─────────────────────────────────────────────────────────────
# Script 5: Verify Transformer Causal Mask
# ─────────────────────────────────────────────────────────────

def test_transformer_causal_mask():
    import torch
    from kigras.model import ActionTransformer

    T = 8  # sequence length
    d_model = 64

    transformer = ActionTransformer(d_model=d_model, num_layers=1)

    # Create dummy input: (T, d_model)
    x = torch.randn(T, d_model)

    # Forward pass with causal masking
    output = transformer(x, causal=True)

    assert output.shape == (T, d_model), f"Shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "No NaN in output"

    # Verify: output[0] should only depend on x[0]
    # (Hard to test directly; rely on gradient flow test below)
    print("✓ Test: transformer causal output shape correct")

# ─────────────────────────────────────────────────────────────
# Script 6: End-to-End Forward Pass
# ─────────────────────────────────────────────────────────────

def test_forward_pass_e2e():
    from kigras.model import KiGRAS
    import torch

    model = KiGRAS(d_model=64, num_agents=10, num_map_lines=20)

    # Dummy scene at time t
    agent_states = torch.randn(10, 4)  # (N, 4): x, y, θ, v
    map_lines = torch.randn(20, 5, 2)  # (N_lines, var_points, 2)

    # Forward pass
    action_logits = model(agent_states, map_lines)

    assert action_logits.shape == (1, 3969), \
        f"Expected (1, 3969), got {action_logits.shape}"

    # Check softmax: should sum to ~1
    probs = torch.softmax(action_logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities don't sum to 1"

    print("✓ Test: E2E forward pass produces valid action distribution")

# ─────────────────────────────────────────────────────────────
# Script 7: Verify No Information Leakage in Teacher Forcing
# ─────────────────────────────────────────────────────────────

def test_teacher_forcing_consistency():
    from kigras.model import KiGRAS
    from kigras.kinematic import kinematic_model_K
    import torch

    model = KiGRAS(d_model=64)

    # Create a batch with ground-truth actions
    agent_states = torch.randn(2, 16, 4)  # (B, T, 4)
    actions_gt = torch.randint(0, 3969, (2, 15))  # (B, T-1) action indices

    # Forward pass with teacher forcing
    action_logits = model(agent_states, teacher_forcing=True)  # (B, T, 3969)

    # Verify: loss gradient flows back
    loss = torch.nn.functional.cross_entropy(
        action_logits[:, :-1].reshape(-1, 3969),
        actions_gt.reshape(-1)
    )

    loss.backward()

    # Check: all parameters have gradients
    for param in model.parameters():
        assert param.grad is not None, "Gradient not computed for some params"

    print("✓ Test: teacher forcing enables full gradient flow")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running sanity checks...\n")

    test_kinematic_model()
    print()

    test_action_discretization()
    print()

    test_inverse_kinematics()
    print()

    test_spatial_encoder()
    print()

    test_transformer_causal_mask()
    print()

    test_forward_pass_e2e()
    print()

    test_teacher_forcing_consistency()
    print()

    print("\n✅ All sanity checks passed!")
```

### Implementation Priority Matrix

| **Priority** | **Component** | **Effort** | **Risk** | **Validation** |
|--------------|---------------|-----------|----------|----------------|
| **P0 (Critical)** | Kinematic Model K | 2 hours | High | Unit tests (Script 1) |
| **P0 (Critical)** | Action Discretization | 1 hour | High | Unit tests (Script 2) |
| **P0 (Critical)** | Inverse Kinematics MPC | 4 hours | High | Unit tests (Script 3), visual inspection |
| **P1 (High)** | Unified Spatial Encoder | 3 hours | Medium | Unit tests (Script 4), output shape checks |
| **P1 (High)** | Transformer with Causal Mask | 4 hours | Medium | Unit tests (Script 5), gradient flow checks |
| **P1 (High)** | Action Decoder Head | 1 hour | Low | Unit tests (Script 6) |
| **P2 (Medium)** | Scene Encoder (Pooling) | 2 hours | Low | Ablation study (verify 2.1% impact) |
| **P2 (Medium)** | U-Embedding | 1 hour | Low | Ablation study (verify 2.8% impact) |
| **P3 (Nice-to-Have)** | DPO Fine-Tuning | 3 hours | Medium | Compare PDriver, SDriver, FDriver behaviors |
| **P3 (Nice-to-Have)** | Evaluation Metrics (ADE, Collision) | 2 hours | Low | Compare to Waymo leaderboard |

---

## Summary Table: Quick Reference

| **Section** | **Key Takeaway** | **Critical Details** |
|-------------|-----------------|---------------------|
| **1. Overview** | 0.7M params, SOTA on Waymo, action-space reformulation | Realism=0.7597, 1st place, 7-170× fewer params |
| **2. Problem Setup** | Input: S^a_{≤0}, S^w_{≤0} | Output: U_{0:T-1} or S^a_{1:T} via K |
| **3. Coordinates** | World + agent-centric frames; CTRA kinematic model | Hard constraint: S_{t+1}=K(S_t, U_t) |
| **4. Architecture** | Spatial encoder (3-layer) + Transformer (3-layer causal) + Action head | 64-dim embedding, 3969-class softmax |
| **5. Forward Pass** | Encode scene → Transformer → Action logits → Sample U_t → K forward | Teacher forcing in training; argmax/sample in inference |
| **6. Loss & Targets** | Cross-entropy on action bins; labels from MPC rolling horizon | No explicit kinematic loss; K is deterministic |
| **7. Data Pipeline** | Waymo Motion Dataset v1.2, 103K scenarios, 2 Hz, 8-sec horizon | Action labels extracted via inverse kinematics (MPC) |
| **8. Training** | Batch size 256, LR 2e-4, OneCycleLR, 1.5M iterations | Causal attention (-4.4% CE), USR (-2.1%), U-embedding (-2.8%) |
| **9. Evaluation** | Realism=0.7597 (overall), Kin=0.4691, Inter=0.8064, Map=0.8658 | minADE=1.4383 (slightly higher than SMART-7M) |
| **10. Ablations** | Causal attention is MVP; unified encoding helps; action embedding refines | Top 3: Causal (-4.4%), USR (-2.1%), U-Embedding (-2.8%) |
| **11. Practical Insights** | MPC for labels, discrete actions, causal attention, agent-centric frames | Gotchas: angle wrap, MPC drift, discretization alignment, validation leakage, identity confusion |
| **12. Implementation** | Build K → discretization → MPC → spatial encoder → transformer → decoder | 7 level dependency graph; 20 unit tests; 7 sanity scripts |

---

## References & Gaps

### Inferred Values (Not Explicitly in Paper)

1. **MPC Rolling Horizon k**: Inferred k ∈ [4, 8] frames (paper says "k consecutive states" but not k value).
2. **Train/Val/Test Split**: Assumed 70/10/20 (paper only says "training set").
3. **Num Attention Heads**: Assumed 4 (typical for d_model=64).
4. **Optimizer & Weight Decay**: Inferred AdamW, 1e-4 weight decay (common defaults).
5. **Augmentations**: Paper silent on data augmentations; table reflects common practices.
6. **Historical Lookback n**: Assumed 2-5 sec (typical for trajectory prediction; not stated).
7. **dt in Kinematic Model**: Assumed 0.5 sec (inverse of 2 Hz frame rate).

### Honest Gaps

1. **Exact Waymo Metric Weights**: Paper doesn't disclose how Realism = f(Kin, Inter, Map); assumed weighted average.
2. **Fine-Tuning Sample Selection Details**: Section IV-E describes expert rules but not quantitative thresholds (collision threshold, speed thresholds for Fast/Comfort drivers).
3. **Failure Modes**: Paper doesn't discuss scenarios where KiGRAS underperforms (e.g., minADE slightly higher than SMART-7M).
4. **Computational Cost**: No FLOPs, inference latency, or memory usage reported (only param count).
5. **Dataset Imbalance**: No analysis of agent-type distribution, scene complexity, or rare-event frequency in training data.
6. **Generalization**: Only tested on Waymo; unclear how it transfers to other datasets (NGSIM, Argoverse, nuScenes).

---

**Document Version**: 1.0
**Date**: March 2025
**Purpose**: Implementation reference for ML engineers reimplementing KiGRAS
**Audience**: Researchers, practitioners, code reviewers

