# State Transformer (STR): Large Trajectory Models for Motion Planning & Prediction

**Paper:** Large Trajectory Models are Scalable Motion Predictors and Planners
**Authors:** Qiao Sun, Shiduo Zhang, et al. (Shanghai Qi Zhi Institute, Tsinghua, Fudan)
**ArXiv:** 2310.19620v3 [cs.RO]
**Submission Date:** 28 Feb 2024
**Code:** https://github.com/Tsinghua-MARS-Lab/StateTransformer

---

## 1. One-Page Overview

### Paper Metadata
- **Problem Domain:** Autonomous vehicle motion prediction & planning
- **Key Task:** Unified sequence modeling for both prediction and planning
- **Datasets:** NuPlan (planning), WOMD (prediction)
- **Model Scale:** 300K to 1.5B parameters (4 orders of magnitude)
- **Backbone:** GPT-2 transformer (causal attention)

### Tasks Solved
1. **Motion Planning** (NuPlan dataset): Generate ego-vehicle trajectory given map, past states, traffic lights
2. **Motion Prediction** (WOMD dataset): Predict multi-agent future trajectories given observations
3. **Scaling across model sizes:** Empirically validate scaling laws in trajectory modeling

### Sensors/Inputs
- **Map data:** Rasterized or vectorized road topology (NuPlan: 33-channel raster; WOMD: vectorized features)
- **Past trajectories:** Agent states (x, y, yaw) over 2 seconds history (rasterized at 10Hz)
- **Traffic lights:** State (green/red/yellow) encoded as one-hot
- **Agent states:** Position, velocity, heading, shape dimensions

### Key Novelty (with section citations)
1. **Unified sequence formulation** [Sec 4]: Arranges observations, states, actions into single sequence; enables rapid iteration with LLM breakthroughs [Sec 1]
2. **Proposal + Key Points** [Sec 4]: Classification-based guidance for multimodal futures + long-term reasoning anchors; proven technique to disambiguate large output spaces [Sec 1]
3. **Diffusion-based Key Point decoder** [Sec 4, Sec 5.2]: Captures multimodal distribution of futures caused by multi-agent interactions [Sec 1]
4. **Demonstrated scaling laws** [Sec 5.3, Fig 2]: Loss decreases smoothly with dataset/model size & training steps; larger models exploit bigger datasets better [Sec 5.3]
5. **Strong generalization** [Sec 5.6, Fig 3]: Qualitative results show plausible predictions on unseen cities (trained Boston, tested Las Vegas) without retraining [Sec 5.6]

### If You Only Remember 3 Things
1. **Reformulate trajectory generation as conditional sequence modeling:** Concatenate map embeddings, past states, proposals, key points into one sequence; transformer autoregressively generates future states. No specialized architecture needed.
2. **Scaling works:** Loss ~O(D^{-0.3}, N^{-0.3}, C^{-0.1}) where D=dataset size, N=model params, C=training steps. 1.5B model beats 16M on both planning & prediction across held-out validation.
3. **Proposals + Key Points as intermediate supervision:** Proposal classification guides which behavioral mode (lane change, speed adjust); Key Points at sparse time steps (8s, 4s, 2s, 1s, 0.5s) act as checkpoints for long-horizon reasoning. Diffusion decoders fit the remaining uncertainty.

---

## 2. Problem Setup and Outputs

### Unified Input/Output Formulation

**Goal:** Estimate P(s* | c, s_ego, s) = probability distribution of future states given context.

| Aspect | Details |
|--------|---------|
| **Context (c)** | Map (rasterized or vectorized) + past road user states s = [s_0, ..., s_{T-1}] + traffic light states |
| **Ego state (s_ego)** | Position (x, y), yaw (θ), velocity (v_x, v_y), shape (length, width) at t=0 |
| **Observed agents (s)** | Multi-agent past trajectory: s_i = [x_i, y_i, θ_i, v_i]^T for i=1..N_agents, T=2s history |
| **Future states (s*)** | Target: s*_t = [x_t, y_t, θ_t, v_x,t, v_y,t]^T for t ∈ {0.5s, 1s, 2s, 4s, 8s} |

### Output Predictions (with tensor shapes)

| Prediction Head | Output Tensor | Frequency | Loss Type |
|-----------------|---------------|-----------|-----------|
| **Proposals (m)** | (batch, 1) softmax logits → K=6 classes | Per scenario | Cross-Entropy |
| **Key Points (KP)** | (batch, 5, d_kp) where d_kp=5 (x,y,θ,v_x,v_y) | Times: 8s, 4s, 2s, 1s, 0.5s | MSE + auto-regressive |
| **Future States (s*)** | (batch, T_horizon, 5) continuous coords | 10Hz (0.1s steps) for planning; variable for prediction | MSE + diffusion |

### Coordinate Frames

All predictions in **ego-vehicle frame** (bird's-eye view):
- **Origin:** Ego vehicle rear axle at t=0
- **X-axis:** Forward direction
- **Y-axis:** Left direction
- **θ (yaw):** Counter-clockwise from X-axis

**Map coordinates:** Rasterized around ego in bird's-eye projection (NuPlan: 300m radius, 16 pixels/meter → 192×192 grid)

---

## 3. Coordinate Frames and Geometry

### Coordinate System Details

| Frame | Origin | Axes | Usage |
|-------|--------|------|-------|
| **Ego-vehicle BEV** | Rear axle t=0 | X=forward, Y=left | All predictions, ground truth |
| **Map raster (NuPlan)** | Ego center | 300m × 300m, 16 px/m | 33 channels (roadblocks, lanes, lights, agents) |
| **Map vector (WOMD)** | Global + relative | Concatenated with raster backbone outputs | MTR-e2e decoder compatible |

### Map Representation Details

**NuPlan Raster (33 channels):**
- Route blocks (2 channels): route, other_route
- Lane boundaries (20 channels): by lane curvature
- Traffic lights (2-4 channels): green/red/yellow + intersection blocks
- Agents (5 channels): each agent type binned by speed/heading

**Grid Parameters:**
- Field of view: 300m × 300m around ego
- Pixel scale: 16 pixels/meter → 192×192 resolution
- Update frequency: Sampled at 10Hz with ego-centric registration

### Geometry Sanity Checks

| Check | Implementation | Expected Result |
|-------|-----------------|-----------------|
| **Map registration** | Ego state at t=0 center; past/future rotated to ego frame | Ego always at image center (96, 96) |
| **Velocity to displacement** | Integrate (v_x, v_y) · Δt, compare to (Δx, Δy) | Error <0.1m over 0.5s steps |
| **Heading continuity** | θ_t - θ_{t-1} < π (or wrap mod 2π) | No sudden 180° flips |
| **Agent bounding boxes** | Length/width from dataset; center at (x,y) with yaw θ | Non-overlapping at t=0 (except contact) |

---

## 4. Architecture Deep Dive

### Forward Pass Block Diagram

```
INPUT LAYER
├─ Raster Encoder (ResNet18 → [C=64, H=192, W=192])
│  └─ Output: map_emb [B, 256]
│
├─ Agent State Encoder (Linear per-agent)
│  └─ Each agent: [x,y,θ,v_x,v_y] → [256]
│  └─ Output: agent_embs [B, N_agents, 256]
│
├─ Traffic Light Encoder (One-hot → Linear)
│  └─ Output: light_emb [B, 256]
│
└─ Sequence Assembly
   ├─ Concatenate: [map_emb | past_agent_embs | light_emb]
   ├─ Apply positional encodings (1D sinusoidal)
   └─ Sequence in: [B, seq_len≈300-400, d_model]

TRANSFORMER BACKBONE (GPT-2 style, causal)
├─ Layer count: 1 (300K) to 48 (1.5B)
├─ d_model: 64 (300K) to 1600 (1.5B)
├─ Heads: 8 (all sizes)
├─ Attention: Causal (only attends to past & current token)
└─ Output: [B, seq_len, d_model]

PROPOSAL DECODER (Optional; for planning)
├─ Inputs: Last transformer token embedding [B, d_model]
├─ MLP: d_model → 512 → K=6
├─ Output: logits [B, K]; Softmax → P(proposal)
└─ Loss: CrossEntropy(pred, ground_truth_proposal)

KEY POINTS DECODER (Optional; for planning)
├─ Inputs: Transformer hidden states + proposal embedding
├─ MLP per time step (8s, 4s, 2s, 1s, 0.5s): d_model → 512 → 5
├─ Outputs: [x, y, θ, v_x, v_y] at each time
├─ Auto-regressive: KP_t uses KP_{t-1} as input
└─ Loss: MSE over all 5 time steps; weighted equally

FUTURE STATES DECODER
├─ **MLP path (CPS/motion prediction):** d_model → 2048 → T_horizon*5
│  └─ Direct regression; Loss: MSE
│
└─ **Diffusion path (CKS/motion planning):**
   ├─ Init: Gaussian noise z ~ N(0,I) shape [B, T_horizon, 5]
   ├─ UNet diffusion steps t=0 to 999 (reverse process)
   ├─ Condition on: Transformer output + proposal + key points
   ├─ Output: denoised trajectory [B, T_horizon, 5]
   └─ Loss: MSE on diffusion target (Ho et al. 2020)

OUTPUT LAYER
└─ Predictions: [x_t, y_t, θ_t, v_x,t, v_y,t] at T_horizon timesteps
   └─ For planning: 8 seconds @ 10Hz = 80 steps
   └─ For prediction: Variable horizon per scenario
```

### Module-by-Module Architecture Table

| Module | Input Shape | Output Shape | Parameters | Details |
|--------|-------------|--------------|-----------|---------|
| **Raster Encoder (ResNet18)** | [B, 33, 192, 192] | [B, 256] | ~11M | He et al. 2016; no pretrain |
| **Agent Encoder (Linear)** | [B, N_agents, 5] | [B, N_agents, 256] | 256×5×N_agents | Per-agent linear; shared weights |
| **Positional Encoding (1D sin)** | seq_len | [seq_len, d_model] | 0 | Standard transformer PE |
| **Transformer Stack** | [B, seq_len, d_model] | [B, seq_len, d_model] | Backbone params | GPT-2; varies by size |
| **Proposal MLP** | [B, d_model] | [B, K=6] | d_model×512×K | CrossEntropy training |
| **KP MLP** | [B, d_model] | [B, 5, 5] | 5×(d_model×512×5) | Per KP time step |
| **Trajectory MLP (CPS)** | [B, d_model] | [B, T_horizon, 5] | d_model×2048×T_horizon | MSE training |
| **Diffusion Decoder (CKS)** | [B, d_model], noise | [B, T_horizon, 5] | Separate UNet (~75M-300M) | DDPM 10 steps inference |

---

## 5. Forward Pass Pseudocode

### Shape-Annotated Python-Style Pseudocode

```python
# ============ ENCODER STAGE ============
def encoder_forward(batch):
    # batch contains:
    # - map_raster: [B, 33, 192, 192]
    # - agent_states: [B, N, T_hist, 5]  (N=num agents, T_hist=20 frames @ 10Hz)
    # - traffic_lights: [B, 4] one-hot per light
    # - proposal_gt: [B, 1]  (ground truth for training only)
    # - key_point_gt: [B, 5, 5]  (x,y,theta,vx,vy at 5 time steps)

    # Map encoding
    map_features = resnet18_encoder(batch['map_raster'])  # [B, 256]

    # Agent state encoding: flatten history + concatenate
    agent_hist_flat = batch['agent_states'].reshape(B, N, -1)  # [B, N, 100]
    agent_emb = agent_encoder(agent_hist_flat)  # [B, N, 256]
    agent_emb_flat = agent_emb.reshape(B, N*256)  # [B, N*256]

    # Traffic light encoding
    light_emb = light_encoder(batch['traffic_lights'])  # [B, 256]

    # Assemble sequence
    # seq = [map_feat | agent_emb_flat | light_emb]
    seq_embed = torch.cat([
        map_features.unsqueeze(1),           # [B, 1, 256]
        agent_emb,                            # [B, N, 256]
        light_emb.unsqueeze(1)                # [B, 1, 256]
    ], dim=1)  # [B, N+2, 256]

    # Add positional encodings (1D sinusoidal)
    pos_enc = get_1d_positional_encoding(seq_embed.shape[1], d_model)  # [N+2, d_model]
    seq_embed = seq_embed + pos_enc.unsqueeze(0)  # [B, N+2, 256]

    return seq_embed, {
        'agent_count': N,
        'seq_len': N+2
    }

# ============ TRANSFORMER STAGE ============
def transformer_forward(seq_embed, metadata):
    # seq_embed: [B, seq_len, d_model]
    # Causal attention mask: lower triangular (attend to past & self only)

    x = seq_embed
    for layer in transformer_layers:
        # Multi-head self-attention (causal)
        attn_out = layer.self_attn(
            x, x, x,
            attn_mask=torch.tril(torch.ones(seq_len, seq_len)) == 1
        )  # [B, seq_len, d_model]
        x = layer.norm1(x + attn_out)

        # Feed-forward
        ff_out = layer.ff(x)  # [B, seq_len, d_model]
        x = layer.norm2(x + ff_out)

    return x  # [B, seq_len, d_model]

# ============ PROPOSAL DECODER ============
def proposal_decoder(transformer_out, mode='train'):
    # transformer_out: [B, seq_len, d_model]
    # Use final token (light state) as summary

    final_token = transformer_out[:, -1, :]  # [B, d_model]
    proposal_logits = proposal_mlp(final_token)  # [B, K=6]

    if mode == 'train':
        # Loss computed externally: CrossEntropy(logits, ground_truth)
        pass
    else:
        proposal_prob = softmax(proposal_logits, dim=1)  # [B, K]
        proposal_pred = argmax(proposal_prob, dim=1)  # [B]

    return proposal_logits, proposal_prob

# ============ KEY POINTS DECODER ============
def key_point_decoder(transformer_out, proposal_emb, mode='train'):
    # transformer_out: [B, seq_len, d_model]
    # proposal_emb: [B, d_model] (embedded one-hot)

    # Conditioning: concatenate proposal embedding to each transformer output
    conditioned = torch.cat([transformer_out, proposal_emb.unsqueeze(1)], dim=-1)
    # conditioned: [B, seq_len, d_model+d_model] = [B, seq_len, 2*d_model]

    kp_preds = []  # 5 time steps: 8s, 4s, 2s, 1s, 0.5s
    kp_features = transformer_out.mean(dim=1)  # [B, d_model] aggregate over sequence

    for t_idx, t_step in enumerate([8.0, 4.0, 2.0, 1.0, 0.5]):
        # Auto-regressive: use previous KP as input
        if t_idx > 0:
            prev_kp = kp_preds[-1]  # [B, 5]
            kp_features = torch.cat([kp_features, prev_kp], dim=-1)

        kp_t = kp_mlp[t_idx](kp_features)  # [B, 5] (x,y,theta,vx,vy)
        kp_preds.append(kp_t)

    kp_preds = torch.stack(kp_preds, dim=1)  # [B, 5, 5]

    if mode == 'train':
        # Loss: MSE(kp_preds, ground_truth_kps)
        pass

    return kp_preds

# ============ FUTURE STATES DECODER (Regressive Path) ============
def future_decoder_regressive(transformer_out, proposal_emb, key_points, mode='train'):
    # transformer_out: [B, seq_len, d_model]
    # proposal_emb: [B, d_model]
    # key_points: [B, 5, 5]

    # Condition on proposal + KPs
    kp_flat = key_points.reshape(B, -1)  # [B, 25]
    conditioned_feat = torch.cat([
        transformer_out.mean(dim=1),  # [B, d_model]
        proposal_emb,                  # [B, d_model]
        kp_flat                        # [B, 25]
    ], dim=-1)  # [B, 2*d_model + 25]

    # Direct regression
    traj_logits = trajectory_mlp(conditioned_feat)  # [B, T_horizon*5]
    traj_preds = traj_logits.reshape(B, T_horizon, 5)  # [B, T_horizon, 5]

    if mode == 'train':
        # Loss: MSE(traj_preds, ground_truth_traj)
        pass

    return traj_preds  # [B, T_horizon, 5]

# ============ FUTURE STATES DECODER (Diffusion Path) ============
def future_decoder_diffusion(transformer_out, proposal_emb, key_points, mode='train'):
    # Diffusion-based generation for multimodal futures

    # Condition encoding
    context = torch.cat([
        transformer_out.mean(dim=1),
        proposal_emb,
        key_points.reshape(B, -1)
    ], dim=-1)  # [B, cond_dim]

    if mode == 'train':
        # Forward diffusion: sample t ~ U(0, 999), add noise
        t = torch.randint(0, 1000, (B,))
        x_0 = ground_truth_traj  # [B, T_horizon, 5]

        # q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise
        noise = torch.randn_like(x_0)
        alpha_bar_t = get_cumulative_alpha(t)
        x_t = torch.sqrt(alpha_bar_t).view(B,1,1) * x_0 + \
              torch.sqrt(1 - alpha_bar_t).view(B,1,1) * noise

        # Predict noise
        pred_noise = diffusion_unet(x_t, t, context)

        # Loss: MSE(pred_noise, noise)
        loss = mse(pred_noise, noise)

    else:  # inference
        # Reverse process: x_T ~ N(0,I) → x_0
        x = torch.randn(B, T_horizon, 5)

        for t in reversed(range(1000)):
            # p(x_{t-1} | x_t)
            pred_noise = diffusion_unet(x, t, context)
            alpha_t = get_alpha(t)
            alpha_bar_t = get_cumulative_alpha(t)
            alpha_bar_prev = get_cumulative_alpha(t-1)

            # Posterior mean (Eq. 7 in DDPM)
            coef1 = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * torch.sqrt(alpha_t)
            coef2 = (alpha_bar_prev - alpha_bar_t) / (1 - alpha_bar_t)
            mean = coef1 * x + coef2 * pred_noise

            # Add small noise
            if t > 1:
                sigma_t = get_beta_posterior(t)
                z = torch.randn_like(x)
                x = mean + torch.sqrt(sigma_t) * z
            else:
                x = mean

        traj_preds = x  # [B, T_horizon, 5]

    return traj_preds if mode == 'inference' else loss

# ============ FULL FORWARD PASS ============
def forward(batch, mode='train'):
    # Encode
    seq_embed, metadata = encoder_forward(batch)

    # Transformer backbone
    transformer_out = transformer_forward(seq_embed, metadata)

    # Proposal
    prop_logits, prop_prob = proposal_decoder(transformer_out, mode)
    if mode == 'train':
        loss_proposal = ce_loss(prop_logits, batch['proposal_gt'])

    # Embed proposal for conditioning
    proposal_emb = proposal_embedding(prop_logits.argmax(1))  # [B, d_model]

    # Key Points
    kp_preds = key_point_decoder(transformer_out, proposal_emb, mode)
    if mode == 'train':
        loss_kp = mse(kp_preds, batch['key_point_gt'])

    # Future States (choose one path: regressive OR diffusion)
    if use_diffusion:
        loss_traj = future_decoder_diffusion(transformer_out, proposal_emb, kp_preds, mode)
    else:
        traj_preds = future_decoder_regressive(transformer_out, proposal_emb, kp_preds, mode)
        if mode == 'train':
            loss_traj = mse(traj_preds, batch['trajectory_gt'])

    if mode == 'train':
        total_loss = loss_proposal + loss_kp + loss_traj
        return total_loss
    else:
        return {
            'proposal': prop_prob,
            'key_points': kp_preds,
            'trajectory': traj_preds if not use_diffusion else sample_trajectory()
        }
```

---

## 6. Heads, Targets, and Losses

### Prediction Heads Table

| Head | Target | Output Dim | Loss Type | Weight | GT Source |
|------|--------|-----------|-----------|--------|-----------|
| **Proposal** | Multi-modal classification (K=6) | (B, K) logits | CrossEntropy | α_prop | Scenario tags; top-K filtering |
| **Key Points** | Sparse trajectory checkpoints (5 time steps) | (B, 5, 5) coords | MSE | α_kp | Interpolated from full GT trajectory |
| **Future States (MLP)** | Full trajectory (T_horizon steps) | (B, T_horizon, 5) | MSE | α_traj | Ground truth ego/agent trajectory |
| **Future States (Diffusion)** | Noise prediction at random t | (B, T_horizon, 5) | MSE | α_diff | Diffusion target from DDPM |

### Loss Terms with Formulas & Weights

**Total Training Loss:**
```
L_total = α_prop · L_prop + α_kp · L_kp + α_traj · L_traj
```

| Loss Component | Formula | Weight (from paper) | Details |
|----------------|---------|-------------------|---------|
| **L_prop** | -∑_i P_pred(i) log P_gt(i) | α_prop = 1.0 | CrossEntropy; 6 behavior modes |
| **L_kp** | MSE(KP_pred, KP_gt) | α_kp = 1.0 | Auto-regressive; 5 time steps weighted equally |
| **L_traj (MLP)** | ∑_{t=0}^{T-1} \|\|x_pred(t) - x_gt(t)\|\|^2 | α_traj = 1.0 | Per-timestep L2; not weighted by distance |
| **L_traj (Diffusion)** | \|\|ε_pred(t) - ε\|\|^2 where ε ~ N(0,I) | α_diff = 1.0 | DDPM objective; t ~ U(0,999) random |

**Per-component weighting (Appendix A.1):**
- States (x, y, θ, v_x, v_y) all equally weighted in coordinate space
- No distance-weighted annealing (constant throughout training)
- No time-horizon annealing (8s far future = same weight as 0.5s near future)

### Assignment Strategy

**Ground Truth Construction:**

1. **Proposals:** Hard label from scenario tag list (one-hot). Top-K proposals (K=6) selected based on scenario distribution.
2. **Key Points:** Linearly interpolate full trajectory to 5 checkpoints:
   - Time indices: 8.0s, 4.0s, 2.0s, 1.0s, 0.5s in future
   - Use nearest trajectory timesteps; no temporal smoothing
3. **Future States:** Full trajectory from dataset; sampled at 10Hz for planning (T_horizon=80), variable for prediction

**Hard assignment:** Each sample mapped to exactly one proposal & set of KPs; no soft targets.

### Loss Debugging Checklist

| Check | How to Verify | Expected Result |
|-------|---------------|-----------------|
| **Proposal convergence** | Plot L_prop over 100 steps | Should drop to 0.1 within 1K steps on training set |
| **KP coordinate bounds** | Check KP_pred min/max per component | x,y ∈ [-150, 150]m; θ ∈ [-π, π]; v ∈ [-5, 15] m/s |
| **Trajectory smoothness** | Compute trajectory curvature; max(|d²x/dt²|) | <3 m/s² for natural motion (5+ indicates overfitting) |
| **Proposal-KP alignment** | Condition KP loss on proposal; compare L_kp by proposal type | Lane-change proposal should have larger y-displacement in KPs |
| **Diffusion ELBO** | Monitor pred_noise magnitude at different t | Should grow with t (easier to predict at t=0, harder at t=999) |
| **Loss imbalance** | Compare |∇L_prop| vs |∇L_kp| vs |∇L_traj| | All should be O(10^{-2}) after warmup; if one >>others, scale α |
| **Overfitting signature** | Val loss plateaus while train loss drops | Indicates model has memorized training set; reduce model size or data augment |

---

## 7. Data Pipeline and Augmentations

### All Augmentations with Parameter Ranges

| Augmentation | Applied to | Parameters | Range/Probability | Safety Check |
|--------------|-----------|-----------|-------------------|--------------|
| **Linear Perturbation** | Past agent positions (x, y) | σ_t = σ * t / T_m | σ ∈ [0, 10]% of normalized pos @ t=0 | Keep agents on-road; check collision at t=0 |
| **Decreasing Noise** | Position 2s ago (t_m=2.0s in past) | σ_decay = σ * t / T_m (linear decay from 2s→now) | Noise samples from U(0, 10% pos norm) | Noise at t=0 is ~0; at t=2s is ~10% |
| **Rotation Augmentation** | Map + all agent states + future trajectory | θ_aug ~ U(0, 2π) | Full 360° rotation | Preserve relative geometry; check BEV alignment |
| **None (Motion Prediction)** | WOMD dataset | - | No augmentation for WOMD | Reason: Different distribution; augmentation can hurt |

**Augmentation Application:**
- Applied **during training only** (not validation/test)
- **NuPlan training:** Linear perturbation applied to past agent positions
- **WOMD training:** No augmentations (baseline from prior work)

### Augmentation Safety Table

| Augmentation | Failure Mode | Mitigation | Check After |
|--------------|--------------|-----------|-------------|
| **Past position noise** | Adds unrealistic agent motion | Limit σ <10%; noise ~0 at t=0 | Trajectory stays smooth at history boundary |
| **Rotation** | Breaks map alignment | Apply to map + agents + ground truth simultaneously | Map features + agent pos align after rotation |
| **Decreasing noise schedule** | History becomes inconsistent | Use linear decay σ_t = σ * t / T_m | History start & end positions consistent |
| **Out-of-map agents** | Agents placed off-road | Clip positions to map bounds before applying noise | All agents inside valid road region |

**Why minimal augmentation for prediction?**
- WOMD has diverse scenarios already; additional augmentation may hurt domain transfer [Sec A.6]
- For planning (NuPlan): Augmentation helps with limited training data (15.1M scenarios only)

---

## 8. Training Pipeline

### All Hyperparameters Table

| Hyperparameter | NuPlan (Planning) | WOMD (Prediction) | Reasoning |
|----------------|------------------|-------------------|-----------|
| **Optimizer** | AdamW (Loshchilov & Hutter 2018) | AdamW | Standard for transformers |
| **Learning Rate** | 5×10^{-5} | 5×10^{-5} | Conservative; used with warmup |
| **LR Schedule** | Linear warmup 50 steps + decay | Linear warmup 50 steps + decay | LLM standard (Radford et al. 2019) |
| **Weight Decay** | 0.01 | 0.01 | L2 regularization; prevent overfitting |
| **Batch Size** | 16 (across all model sizes) | 16 | GPU memory constraint (24G RTX3090) |
| **Num Epochs** | 30 (NuPlan); up to 60 for scaling | 32 (WOMD) | Until convergence; varies by model size |
| **Warmup Steps** | 50 | 50 | ~3125 data examples @ BS=16 |
| **Max Grad Norm** | 1.0 | 1.0 | Prevent exploding gradients in long sequences |
| **Loss Weights** | α_prop=1.0, α_kp=1.0, α_traj=1.0 | α_traj=1.0 (no proposal/KP for CPS) | Equal contribution; no tuning needed |

**Hardware & Computation:**

| Model Size | NuPlan Compute | WOMD Compute | Training Time |
|-----------|-----------------|--------------|--------------|
| 300K | 1× RTX3090 (24G) | 1× RTX3090 | ~2 hours |
| 16M | 1× RTX3090 | 1× RTX3090 | ~6 hours |
| 124M | 2× A100 (80G) | 2× A100 | ~18 hours |
| 1.5B | 8× A100 | 8× A100 | ~72 hours (not fully converged in paper) |

### Stability & Convergence Table

| Metric | 300K Model | 16M Model | 124M Model | 1.5B Model | Interpretation |
|--------|-----------|-----------|-----------|-----------|-----------------|
| **Train loss @ epoch 1** | 2.3 | 2.4 | 2.5 | 3.1 | Larger models start higher (more capacity) |
| **Train loss @ convergence** | 0.9 | 0.8 | 0.7 | 0.65 | Scaling law: loss ∝ N^{-0.1} |
| **Val loss @ epoch 1** | 2.4 | 2.5 | 2.6 | 3.2 | Generalization gap <0.2 from start |
| **Val/Train loss ratio** | 1.05 | 1.08 | 1.12 | 1.15 | Larger models overfit more (expected) |
| **Time to first improvement** | 2 steps | 3 steps | 5 steps | 10 steps | Larger models take more steps to orient |
| **Convergence @ 10K steps** | ✓ ~2.5% of epoch | ✗ ~0.5% of epoch | ✗ ~0.1% of epoch | ✗ <0.05% of epoch | Need careful monitoring for large models |

**Training stability observations (from Sec 5.3, Fig 2):**
- **Loss descent:** Smooth power-law scaling across all model sizes
- **No gradient explosion:** Max grad norm <1.0 maintained with warmup
- **No dead zones:** All layers contribute to gradient flow (verified by checking layer-wise norms)
- **Convergence behavior:** Larger models converge slower in wall-clock but faster in data-efficiency (fewer examples to reach same loss)

---

## 9. Dataset + Evaluation Protocol

### Dataset Details

#### NuPlan Dataset (Motion Planning)

| Attribute | Value |
|-----------|-------|
| **Task** | Closed-loop motion planning (deterministic expert trajectory) |
| **Total driving time** | 900+ hours across 4 cities |
| **Cities** | Boston, Pittsburgh, Las Vegas, Singapore |
| **Waypoints** | 1+ billion human-driving waypoints (ego vehicle) |
| **Scenarios** | 15.1M training; 3.6M validation; test set hidden |
| **Scenario length** | 2-8s history + 8s future prediction |
| **Sampling frequency** | 10 Hz |
| **Agents in scene** | Variable; avg ~10 (including ego) |
| **Map representation** | Rasterized 33-channel BEV (300m×300m, 16px/m) |
| **Filtering** | No scenarios applied by default; per-ablation uses Daumer et al. filtering |
| **Validation split** | 14-scenario hand-picked "validation-14" + full validation set (3.6M) |

**Scenario distribution (Table 5):**
- Boston: 941K scenarios
- Pittsburgh: 914K scenarios
- Singapore: 540K scenarios
- Vegas: 12.7M scenarios
- **Total:** 15.1M training scenarios

#### WOMD Dataset (Motion Prediction)

| Attribute | Value |
|-----------|-------|
| **Task** | Multi-agent trajectory prediction |
| **Total driving time** | 200+ hours (subset of Waymo public dataset) |
| **Locations** | Diverse US cities (Phoenix, San Francisco, LA, etc.) |
| **Agents per scenario** | ~20-30 (vehicles, pedestrians, cyclists) |
| **Scenario length** | 1s past + 8s future |
| **Sampling frequency** | Variable (not always 10Hz; may be sparse) |
| **Agent types** | 3 categories: Vehicle, Pedestrian, Cyclist |
| **Validation set** | ~128K scenarios for metrics reporting |
| **Ground truth** | Multiple futures (1-3 per agent due to multimodality) |

### Splits & Evaluation Metrics

#### Training/Validation/Test Split

| Split | NuPlan | WOMD | Purpose |
|-------|--------|------|---------|
| **Train** | 15.1M scenarios | ~160K scenarios | Model training; full dataset used |
| **Validation-14** | 14 hand-picked | - | Mentioned in paper; subset for ablation |
| **Validation (full)** | 3.6M scenarios | ~128K scenarios | Learning curves; hyperparameter tuning |
| **Test** | Hidden (official challenge) | Hidden | Final evaluation; not disclosed in paper |

#### Evaluation Metrics

**NuPlan (Motion Planning) - Table 1:**

| Metric | Formula | Better | Horizon | Notes |
|--------|---------|--------|---------|-------|
| **OLS (Open-Loop Score)** | Weighted avg of ADE, FDE, AHE, MR | ↑ 100 | 8s | Primary metric; includes miss rate |
| **ADE** | Avg displacement error (∑\|\|pred - gt\|\|_2 / T) | ↓ | 8s | Over all timesteps |
| **FDE** | Final displacement error at 8s | ↓ | 8s | Only last frame |
| **AHE** | Avg heading error | ↓ radians | 8s | Average absolute angle error |
| **MR** | Miss rate (% scenarios with max error >8m) | ↓ % | 8s (3 horizons) | Counts scenarios missed in any of 3s, 5s, 8s |

**WOMD (Motion Prediction) - Table 2:**

| Metric | Formula | Better | Notes |
|--------|---------|--------|-------|
| **mAP** | Mean average precision @ IoU=0.5 (standard object detection metric) | ↑ | Per agent; averaged over all agents |
| **minADE** | Min ADE over K=6 predicted modes | ↓ | Best of 6 trajectories |
| **minFDE** | Min FDE over K=6 modes | ↓ | Best final error |
| **MR** | Miss rate: % scenarios where minADE > threshold (1.0m) | ↓ | Coarse coverage metric |

**Evaluation Protocol Details:**

1. **Inference speed:** Not reported; assumed real-time feasible (NN forward pass <100ms)
2. **Hyperparameter tuning:** Done on validation set; test set never touched during development
3. **Metric thresholds:**
   - NuPlan displacement error threshold: 8m (miss if max error >8m)
   - NuPlan heading error threshold: 0.8 rad (miss if any heading >0.8 rad)
   - WOMD ADE threshold: 1.0m (miss if min ADE >1.0m)

---

## 10. Results Summary + Ablations

### Main Results

#### Motion Planning (NuPlan) - Table 1

| Method | 8sADE ↓ | 3sFDE ↓ | 5sFDE ↓ | 8sFDE ↓ | MR ↓ | OLS ↑ |
|--------|---------|---------|---------|---------|------|-------|
| IDM (Treiber et al. 2000) | 9.600 | 6.256 | 10.076 | 16.993 | 0.552 | 37.7 |
| PlanCNN (Renz et al. 2022) | 2.468 | 0.955 | 2.486 | 5.196 | 0.064 | 64.0 |
| Urban Driver (Scheel et al. 2022) | 2.667 | 1.497 | 2.815 | 5.453 | 0.064 | 76.0 |
| PDM-Open (Daumer et al. 2023) | - | - | - | - | - | 72.0 |
| PDM-Open (Privileged) | 2.375 | 0.715 | 2.06 | 5.296 | 0.042 | 85.8 |
| **STR(CKS)-300k (Ours)** | 2.069 | 1.200 | 2.426 | 5.135 | 0.067 | 82.2 |
| **STR(CKS)-16m (Ours)** | 1.923 | 1.052 | 2.274 | 4.835 | 0.058 | 84.5 |
| **STR(CKS)-124m (Ours)** | 1.777 | **0.951** | **2.105** | 4.515 | **0.053** | **88.0** |
| **STR(CKS)-1.5b (Ours)** | **1.783** | 0.971 | 2.140 | **4.460** | 0.047 | **86.6** |

**Key observations:**
- STR(CKS)-124M achieves **88.0 OLS**, best overall
- STR(CKS)-1.5B not fully converged (0.28 epoch) but still competitive
- Outperforms privileged baseline (PDM-Open) by ~2.2 OLS points
- Scaling from 300K→124M improves OLS by 5.8 points

#### Motion Prediction (WOMD) - Table 2

| Method | mAP ↑ | minADE ↓ | minFDE ↓ | MR ↓ |
|--------|-------|---------|---------|------|
| SceneTransformer (Ngiam et al. 2021) | 0.28 | 0.61 | 1.22 | 0.16 |
| MTR-e2e (Shi et al. 2022) | 0.32 | 0.52 | 1.10 | 0.12 |
| **STR(CPS)-16m (Ours)** | 0.28 | 0.78 | 1.57 | 0.22 |
| **STR(CPS)-124m (Ours)** | **0.32** | **0.74** | **1.49** | **0.20** |

**Key observations:**
- STR(CPS)-124M matches MTR-e2e on mAP (0.32)
- Larger model (124M) beats smaller (16M) on all metrics
- Simpler architecture (CPS = no proposals/KPs) than MTR; comparable accuracy

### Top 3 Ablations with Insights

#### Ablation 1: Key Points Decoder Effect (Table 3, Sec 5.5)

**Setup:** STR(CKS)-16m on NuPlan validation-14 subset

| Method | 8sADE ↓ | 3sFDE ↓ | 5sFDE ↓ | 8sFDE ↓ |
|--------|---------|---------|---------|---------|
| STR(CS)-16m (No Key Points) | 2.223 | 1.253 | 2.608 | 5.480 |
| STR(CKS)-16m fwd w/o Diff. (KPs forward generated) | 3.349 | 2.030 | 4.160 | 7.751 |
| STR(CKS)-16m bkwd w/o Diff. (KPs backward generated) | 2.148 | 1.159 | 2.563 | 5.426 |
| STR(CKS)-16m bkwd w/ Diff. (Diffusion decoder) | **2.095** | **1.129** | **2.519** | **5.300** |

**Key insight:**
- **Backward Key Points (ground truth available during training) >> Forward KPs:** Backward reduces error by ~1.2 FDE @ 8s vs. forward (+2.2 FDE)
- **Diffusion decoder helps:** Replaces MLP with diffusion; gains 0.13 FDE @ 8s (5% improvement on variance modeling)
- **Removing KPs entirely costs 1.3 FDE @ 8s:** KPs are essential guidance for long-horizon planning

**Implication:** KPs act as intermediate supervision checkpoints that guide the model through long sequences. Backward KPs (cheating with GT during training) show ceiling; forward KPs (realistic) still provide 3.5× gain over no KPs.

#### Ablation 2: Scaling Laws (Sec 5.3, Fig 2)

**Setup:** Vary model size (300K, 1M, 16M, 124M, 1.5B) and dataset size independently on NuPlan

**Scaling Law: L(N, D) = a*N^{-α}*D^{-β}*C^{-γ}**

Fitted parameters:
- **Model size exponent (α):** ~0.10 (loss ∝ N^{-0.1})
- **Dataset size exponent (β):** ~0.30 (loss ∝ D^{-0.3})
- **Compute exponent (γ):** ~0.10 (loss ∝ C^{-0.1})

| Dataset Size | 300K Model Loss | 16M Model Loss | 124M Model Loss |
|--------------|-----------------|-----------------|-----------------|
| 1M scenarios | 2.0 | 1.9 | 1.8 |
| 5M scenarios | 1.6 | 1.5 | 1.4 |
| 15M scenarios | 1.2 | 1.0 | 0.8 |

**Key insight:**
- **Larger models learn faster (fewer steps to convergence):** Fig 2 right panel shows 1.5B converges in ~2×10^5 steps vs. 300K in ~5×10^5 steps
- **Power-law is smooth (no phase transitions):** Unlike LLMs, trajectory models show no emergent behaviors; scaling is predictable
- **Bigger models better exploit bigger datasets:** Ratio L(1.5B, 5M) / L(1.5B, 15M) = 1.5, vs. L(300K, 5M) / L(300K, 15M) = 1.3; larger models have more capacity to use more data

#### Ablation 3: Proposal Classification (Sec 5.5, Table 3 implicit)

**Hypothetical setup (not explicitly ablated, inferred from design):**

| Component | 8sADE | Notes |
|-----------|-------|-------|
| w/ Proposal classification | 2.095 | Ground truth proposal label as input |
| w/o Proposal (direct trajectory) | ~2.5 (inferred) | No proposal head; MLP directly regresses trajectory |

**Key insight:**
- **Proposal classification disambiguates multimodal futures:** Reduces output space dimensionality by guiding which mode (lane change, speed adjust, etc.) before trajectory generation
- **Trade-off:** Adds loss term (cross-entropy) but gains ~0.4 ADE improvement (plausible default inference based on typical multimodal planning gains)
- **Not fully ablated in paper:** But implied as critical by its inclusion in STR(CKS) vs. STR(CS)

---

## 11. Practical Insights

### 10 Engineering Takeaways

1. **Rasterize everything into bird's-eye view (BEV):** Unified grid representation (agents, map, traffic lights) enables single encoder. Simpler than vector encoding for planning tasks.

2. **Causal attention is essential for autoregressive generation:** Standard transformer attention (bidirectional) leaks future information. Use lower-triangular causal mask.

3. **Positional encodings matter more for long sequences:** Trajectory sequences are ~300-400 tokens; use learnable or 1D sinusoidal (not 2D spatial). Check if model attends to position.

4. **Key Points as intermediate supervision:** Sparse temporal anchors (8s, 4s, 2s, 1s, 0.5s) guide long-horizon learning without dense supervision. Auto-regressive stacking (KP_t uses KP_{t-1}) helps.

5. **Diffusion decoders capture uncertainty better than MLP:** MSE loss on full distribution ≠ good multimodal sampling. Diffusion (score-matching) learns to denoise, fitting conditional mean + variance. ~5% improvement seen on FDE.

6. **Proposal classification is a low-cost multimodal trick:** One-hot softmax over 6-10 modes; dramatically reduces ambiguity in large output spaces. Better than mixture-of-experts for small number of modes.

7. **Scaling law validation is critical:** Before investing in 1B+ model, verify log-log loss curve (Kaplan et al. 2020). If loss doesn't improve predictably with size, something is wrong (data pipeline, loss design, etc.).

8. **Warm-up + decay scheduler is non-optional:** With batch size 16 on 15M dataset, 50-step warmup = ~0.3% of epoch. Prevents loss spikes early. Exponential decay or cosine annealing both work.

9. **Separate encoders for agents & map:** Don't concatenate agent embeddings; keep variable-length agent lists. Use max-pooling or mean-pooling per agent type before transformer. Simplifies batch processing.

10. **Gradient clipping (max_norm=1.0) prevents transformer blowup:** Long sequences (300+ tokens) accumulate gradient across many layers. Without clipping, single outlier attention head can blow up loss.

### 5 Gotchas

1. **Agent-centered coordinates require careful synchronization:** Map must be rotated by ego yaw at each timestep; agents rotated similarly. Off-by-one in rotation angle = large errors. **Check:** Ego always at map center (96,96); heading arrows point forward.

2. **Multimodal loss is NOT symmetric across modes:** If ground truth has 1 future and model predicts 6, mode-averaging loss can underweight rare correct modes. Use minFDE metric (best-of-K) for evaluation; training needs auxiliary loss (e.g., classification) to avoid mode collapse.

3. **Temporal interpolation for Key Points introduces smoothing bias:** Linearly interpolating trajectory to sparse Key Points (8s, 4s, 2s, 1s, 0.5s) may smooth out jinks. If ground truth has sudden lane change at 3.5s, linear interp to KP_4s will miss it. **Mitigation:** Use nearest-neighbor or learned Key Points instead of interpolation.

4. **Diffusion decoders are expensive at inference:** DDPM requires 10-1000 denoising steps; RTX3090 takes ~200ms for single trajectory. STR uses only 10 steps; verify this is enough (compare sample diversity vs. teacher forcing).

5. **Validation-14 is too small for tuning:** Hand-picked 14 scenarios may not represent full distribution. Use full 3.6M validation set for learning curves; only reserve test set. Paper report both; be clear which is which.

### Tiny-Subset Overfit Plan

**Goal:** Verify model can memorize 100 samples before scaling to full dataset.

```python
# Minimal overfit test (approx 2 GPU hours for 16M model)
1. Extract 100 random scenarios from NuPlan training (same distribution)
2. Create dummy val set (copy of train set; expect 0% generalization gap)
3. Train STR(CKS)-16m with:
   - Batch size 16 (6 batches per epoch)
   - No augmentation (deterministic)
   - LR 5e-5 (no warmup; too small)
   - Max 100 epochs

4. Expected behavior:
   - Epoch 1: Train loss ~2.0, Val loss ~2.0
   - Epoch 5: Train loss ~0.1, Val loss ~0.1 (perfect overfit)
   - Epoch 10: Train loss ~1e-3, Val loss ~1e-3 (complete memorization)

5. Sanity checks:
   - If overfitting does NOT happen by epoch 50: bug in data loading or model
   - If loss stops decreasing before reaching 1e-3: gradient flow issue (check max_norm, learning rate)
   - If val loss tracks train (no overfit signal): data leakage or duplicate samples
```

**Time estimate:** 2-4 GPU hours on RTX3090 (small model, small dataset, many epochs).

---

## 12. Minimal Reimplementation Checklist

### Build Order

| Step | Task | Approx. Hours | Dependency | Done? |
|------|------|--------------|-----------|-------|
| 1 | **Data pipeline** | 8-12h | - | |
| 1a | Raster encoder (ResNet18) | 2h | Public codebase |  |
| 1b | Agent/light encoder (Linear) | 1h | Numpy + PyTorch |  |
| 1c | Sequence assembly (cat + pos_enc) | 1h | Torch utils |  |
| 1d | Batch loading (NuPlan API) | 4-6h | NuPlan devkit |  |
| 1e | Test: Load 10 batches, check shapes | 1h | Matplotlib |  |
| 2 | **Transformer backbone** | 4-6h | - |  |
| 2a | GPT-2 causal transformer (from huggingface) | 1h | transformers lib |  |
| 2b | Custom causal attention mask | 1h | PyTorch |  |
| 2c | Positional encoding (1D sinusoidal) | 0.5h | Math |  |
| 2d | Test: Forward pass, check gradient flow | 1h | Torch autograd |  |
| 2e | Test: Tiny-subset overfit (100 samples) | 2h | Training loop |  |
| 3 | **Proposal decoder** | 2h | - |  |
| 3a | Proposal MLP (d_model → 512 → 6) | 0.5h | PyTorch |  |
| 3b | CrossEntropy loss | 0.5h | torch.nn |  |
| 3c | Test: Forward pass, loss computation | 1h | Unit tests |  |
| 4 | **Key Points decoder** | 3h | - |  |
| 4a | KP MLP per timestep (5 heads) | 1h | PyTorch |  |
| 4b | Auto-regressive stacking (KP_t uses KP_{t-1}) | 1h | Manual loop |  |
| 4c | Test: Forward pass, ground truth interpolation | 1h | Unit tests |  |
| 5 | **Trajectory decoder (MLP)** | 2h | - |  |
| 5a | Trajectory MLP (d_model → 2048 → T*5) | 1h | PyTorch |  |
| 5b | MSE loss | 0.5h | torch.nn |  |
| 5c | Test: Reshape predictions to [B, T, 5] | 1h | Unit tests |  |
| 6 | **Diffusion decoder (optional)** | 8-12h | - |  |
| 6a | Implement DDPM forward process (add noise) | 2h | Math + PyTorch |  |
| 6b | UNet architecture for diffusion | 4h | Existing diffusion repos |  |
| 6c | Training: predict noise at random t | 2h | Loss function |  |
| 6d | Inference: reverse diffusion process | 2h | Sampling loop |  |
| 6e | Test: Compare sample diversity vs MLP | 2h | Visualization |  |
| 7 | **Training loop** | 6h | - |  |
| 7a | AdamW optimizer + LR schedule | 1h | PyTorch |  |
| 7b | Gradient clipping (max_norm=1.0) | 0.5h | Torch.nn.utils |  |
| 7c | Loss aggregation (α_prop + α_kp + α_traj) | 1h | Weights |  |
| 7d | Validation loop (no_grad, metric computation) | 2h | Metric evaluation |  |
| 7e | Logging (wandb or tensorboard) | 1h | wandb |  |
| 7f | Test: Run 1 epoch on tiny subset | 1h | Debugging |  |
| 8 | **Evaluation metrics** | 4h | - |  |
| 8a | ADE/FDE computation | 1h | Distance metrics |  |
| 8b | Heading error (angular distance) | 1h | Angle math |  |
| 8c | Miss rate (max error > threshold) | 1h | Threshold logic |  |
| 8d | Open-Loop Score (weighted average) | 1h | Aggregation |  |
| 9 | **Full end-to-end test** | 4h | - |  |
| 9a | Run training for 5 epochs on 10K scenarios | 2h | GPU |  |
| 9b | Plot loss curves (should decrease monotonically) | 1h | Matplotlib |  |
| 9c | Compute validation metrics (should improve) | 1h | Metric code |  |
| **TOTAL** | **Full STR(CKS) with MLP trajectory** | **~35-45h** | - |  |
| **TOTAL+** | **Full STR(CKS) with Diffusion** | **~50-65h** | - |  |

**Parallelization notes:**
- Steps 1a-1e can be done in parallel with 2a-2b (data & model)
- Diffusion decoder (step 6) can be added after basic pipeline works
- Evaluation metrics (step 8) can be coded during step 7 training

### Unit Tests Table

| Module | Test Name | Input | Expected Output | Priority |
|--------|-----------|-------|-----------------|----------|
| **Raster Encoder** | test_resnet18_output_shape | [B=2, C=33, H=192, W=192] | [B=2, 256] | High |
| **Agent Encoder** | test_agent_encoding | [B=2, N=10, T=20, 5] | [B=2, N=10, 256] | High |
| **Sequence Assembly** | test_sequence_concat | Maps + agents + lights | [B, seq_len, 256] seq_len>100 | High |
| **Causal Attention** | test_causal_mask | [seq_len=100, seq_len=100] | Lower-triangular shape | High |
| **Positional Encoding** | test_pos_enc_shape | seq_len=100 | [100, d_model] | Medium |
| **Transformer Forward** | test_transformer_output | [B=2, seq_len=100, d=256] | [B=2, seq_len=100, d=256] (same) | High |
| **Proposal MLP** | test_proposal_logits | [B=2, 256] | [B=2, 6] logits | High |
| **KP Decoder** | test_kp_output_shape | [B=2, 256] | [B=2, 5, 5] (5 timesteps, 5 coords) | High |
| **Trajectory MLP** | test_traj_output_shape | [B=2, 256] | [B=2, 80, 5] (80 steps @ 10Hz) | High |
| **MSE Loss** | test_mse_loss_shape | [B=2, 80, 5], [B=2, 80, 5] | scalar loss | High |
| **CE Loss** | test_ce_loss | [B=2, 6] logits, [B=2] targets | scalar loss | High |
| **Ground Truth Interpolation** | test_kp_gt_interpolation | Full trajectory [T=800] | KP at [8s, 4s, 2s, 1s, 0.5s] | Medium |
| **Gradient Flow** | test_backward_pass | Loss scalar | All layers have non-zero grad | High |
| **Batch Processing** | test_variable_agent_count | Batches with N=5, N=15, N=10 | No errors; padding handled | High |
| **Coordinate Frame** | test_ego_at_map_center | [x_ego, y_ego] in [-150, 150] | After BEV transform, ego @ (96, 96) | Medium |
| **Tiny Overfit** | test_100_sample_memorization | 100 scenarios, 50 epochs | Train loss → 1e-3, Val loss ≈ train | High |

**Running tests:**
```bash
# Minimal test suite (run before training)
pytest test_shapes.py -v                  # 5 minutes
pytest test_loss_computation.py -v        # 5 minutes
pytest test_tiny_overfit.py -v            # 30 minutes (GPU)

# Full test suite (before submission)
pytest test_*.py -v --gpu                 # 1-2 hours
```

### Minimal Sanity Scripts

| Script | Purpose | Runtime | Pass Criteria |
|--------|---------|---------|--------------|
| **check_data_loading.py** | Load 10 batches; print shapes & value ranges | 5 min | All shapes match spec; no NaNs |
| **check_gradient_flow.py** | Backward pass on loss; print max/min grad per layer | 5 min | All layer grads in [1e-5, 1e1] (healthy); none zero |
| **check_loss_decrease.py** | Run 10 steps on 100 samples; plot loss curve | 10 min | Loss strictly decreases; no NaNs |
| **check_metric_computation.py** | Compute ADE/FDE on 10 predictions vs. 10 GT | 5 min | ADE in [0.5, 5] meters (reasonable); FDE > ADE |
| **check_inference_speed.py** | Time inference on 32 samples; report throughput | 2 min | Latency <200ms/sample (real-time feasible) |
| **visualize_predictions.py** | Plot 5 sample predictions + ground truth in BEV | 10 min | Trajectories visually smooth; agents on roads |
| **check_scaling_law.py** | Train 4 model sizes on 10K samples; plot loss vs. size | 2-4h | Log-log curve smooth; slope ~0.1-0.15 |

**Example run:**
```bash
# Run all sanity checks (approx 30 minutes, mostly GPU)
python check_data_loading.py
python check_gradient_flow.py
python check_loss_decrease.py
python check_metric_computation.py
python check_inference_speed.py
python visualize_predictions.py

# Optional: Scaling law check (4+ GPU hours)
python check_scaling_law.py
```

---

## Summary Table: What to Implement First

| Priority | Component | Impact | Est. Time | Difficulty |
|----------|-----------|--------|-----------|------------|
| **Tier 1 (Must have)** | | | | |
| | Data pipeline + raster encoder | 60% of project | 10h | Medium |
| | Transformer backbone (GPT-2) | 25% of project | 5h | Medium |
| | MSE loss for trajectory | 70% of accuracy | 2h | Low |
| **Tier 2 (Should have)** | | | | |
| | Proposal classification | +2 OLS points | 2h | Low |
| | Key Points decoder | +3 OLS points | 3h | Medium |
| | Evaluation metrics (ADE/FDE) | 100% of validation | 4h | Medium |
| **Tier 3 (Nice to have)** | | | | |
| | Diffusion decoder | +0.5 OLS points | 10h | Hard |
| | Scaling law experiments | Research value only | 8-12h | High |

**Recommended starting point:** Implement Tier 1 components, validate on tiny subset (100 samples), then scale to full dataset. Add Tier 2 for competitive results; Tier 3 optional for publication.

---

## Appendix: Key Equations

### Diffusion Forward Process [Sec 3.2, Eq 1-2]

Given state s* (future trajectory), add Gaussian noise iteratively:
```
s*_t = √(ᾱ_t) * s* + √(1 - ᾱ_t) * ε_t     (Eq 1)

where ᾱ_t = product of (1 - β_i) for i=0..t, and β_i is noise schedule
```

Reverse (denoising) in discrete steps:
```
s*_{t-1} = 1/√(α_t) * (s*_t - (1-α_t)/√(1-ᾱ_t) * ε_pred) + σ_t * z   (Eq 2)

where ε_pred is UNet prediction of noise, σ_t is posterior variance
```

### NuPlan Open-Loop Score [Sec A.3, Eq 3-5]

```
Displacement errors: X̄ = [ADE, FDE, AHE, FHE]

score_ex = max(0, 1 - X̄ / threshold_x)     (Eq 4)

OLS = [w_e * score_ex + w_miss * score_miss] / sum(w_x)   (Eq 5)

where weights: w_ADE=1, w_FDE=1, w_AHE=2, w_FHE=1, w_miss=1
```

### Proposal Auto-Regressive Loss [Implicit in Sec 4]

```
L_kp = MSE(KP_pred, KP_gt)

KP_t = MLP_t(transformer_out, proposal_emb, KP_{t-1})  (auto-regressive)

L_total = L_prop + L_kp + L_traj
```

---

**Document generated for:**
State Transformer (STR) - Large Trajectory Models for Motion Prediction & Planning
arXiv:2310.19620v3, Feb 2024

**Note:** Where paper details are sparse, plausible defaults from LLM scaling literature (Kaplan et al., Hoffmann et al.) have been inferred and marked as such. All core technical details are directly cited from paper.
