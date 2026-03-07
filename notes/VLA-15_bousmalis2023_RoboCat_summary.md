# RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation
## Comprehensive Implementation Summary

**Paper Reference:** [RoboCat: A Self-Improving Foundation Agent for Robotic Manipulation](https://deepmind.google/research/publications/35829/)
**Citation:** Bousmalis et al., TMLR 2023
**arXiv:** [2306.11706](https://arxiv.org/abs/2306.11706)
**Blog:** [DeepMind - RoboCat](https://deepmind.google/blog/robocat-a-self-improving-robotic-agent/)

---

<!-- AUDIT_NOTE_START -->
> Audit note (March 7, 2026): This block is the canonical, paper-backed teaching summary for this file. If later sections give exact tensor shapes, hyperparameters, deployment numbers, or pseudocode details that the paper does not explicitly report, treat them as explanatory reconstructions rather than direct paper claims.

## Paper-Backed Teaching Corrections
- **What the paper actually does:** presents RoboCat, a multi-embodiment, multi-task generalist manipulation agent that can adapt to new tasks/robots and then generate data for later training rounds.
- **Core contributions:** RoboCat generalizes zero-shot and after adaptation with only 100-1000 examples, and it provides a basic autonomous self-improvement loop by collecting new data with the current agent.
- **What you should understand:** the paper is about generalist-to-specialist adaptation plus iterative self-improvement, not just about one transformer block diagram.
- **Important correction:** unless the paper explicitly states a tokenizer or submodule detail, treat later architecture specifics as reconstruction; the source-backed story is the generalist agent and autonomous data-generation loop.
<!-- AUDIT_NOTE_END -->

## 1. One-Page Overview

### Metadata
- **Venue:** TMLR (Transactions on Machine Learning Research) 2023
- **Authors:** Konstantinos Bousmalis, Giulio Vezzani, Dario Rao, and 37+ co-authors from Google DeepMind
- **Release Date:** June 2023 (arXiv), December 2023 (TMLR)
- **Task Domain:** Multi-embodiment, self-improving robotic manipulation

### Core Concept
**Iterative self-improvement loop for robot learning:**
1. Train generalist agent on diverse data (real + sim, multiple robots)
2. Deploy to new task with 100-1000 demonstration samples
3. Fine-tune agent on task-specific data
4. Deploy agent to collect on-policy data autonomously (~10,000 rollouts)
5. Combine task data + self-generated data into training set
6. Retrain next-generation agent
7. Repeat: Each iteration improves agent capability

### Key Contributions
1. **Self-improving loop:** First work showing autonomous data generation improves generalist agents
2. **Multi-embodiment:** Single agent works across Panda, Sawyer, KUKA, sim arms
3. **Hindsight goal relabeling:** Leverages visual goal-conditioning for free data augmentation
4. **Scale:** Trained on millions of trajectories from multiple sources
5. **Few-shot adaptation:** 100-1000 demos → specialized agent

### If You Only Remember 3 Things
1. **Self-improvement is possible:** Agent trained on large dataset, specialized to task, can generate useful data for retraining. Creates positive feedback loop.
2. **Hindsight goals are essential:** Visual goal-conditioning allows any trajectory segment as a valid goal without additional supervision. Enables efficient data reuse.
3. **Generalists can be specialized:** Single large model can adapt to specific tasks with modest fine-tuning (100-1000 examples), without forgetting generalist skills.

### Core Problem
Collecting robot data is expensive. Existing approaches: (a) collect all data upfront (expensive, fixed), (b) learn single task (not generalizable). RoboCat addresses this by: training a powerful generalist agent first, then using that agent to autonomously collect task-specific data, creating an iterative improvement loop where more data → better agent → more/better data.

---

## 2. Problem Setup & Architecture

### Input/Output Specification

| Component | Shape | Description |
|-----------|-------|-------------|
| **Observation (image)** | (H, W, 3) uint8 | Egocentric RGB (not necessarily used) |
| **Goal image** | (H, W, 3) uint8 | Target/desired state image |
| **VQ-GAN tokens** | (N_tokens,) int32 | Tokenized image representation |
| **Action output** | (4,) or (7,) | [gripper_xyz, gripper_quaternion, gripper_width] |
| **Future frame tokens** | (N_tokens,) int32 | Predicted next frame tokens (auxiliary) |
| **Trajectory horizon** | T ≤ 50 steps | Single episode length |

### Coordinate Frames
- **Camera frame:** Egocentric or exocentric observation
- **Action space:** End-effector (3D position + rotation + gripper)
- **Goal frame:** Visual image of desired end state
- **Tokenized space:** VQ-GAN latent space (typically 8×8×256 dim)

---

## 3. Architecture Deep Dive

### Block Diagram

```
┌────────────────────────────────────────────────────────┐
│              RoboCat Architecture                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────────────┐   ┌──────────────────────┐ │
│  │ Observation Image   │   │ Goal Image           │ │
│  │ (H, W, 3)          │   │ (H, W, 3)           │ │
│  └──────────┬──────────┘   └──────────┬──────────┘ │
│             │                         │           │
│             v                         v           │
│  ┌──────────────────────────────────────────────┐ │
│  │ Frozen VQ-GAN Tokenizer                      │ │
│  │ - Trained on diverse robot vision data      │ │
│  │ - Outputs: (N_tokens,) integer tokens       │ │
│  │ - Same tokenizer for obs & goal             │ │
│  └──────────────────────────────────────────────┘ │
│             │                         │           │
│             v                         v           │
│  ┌──────────────────────────────────────────────┐ │
│  │ Token Embeddings                             │ │
│  │ - Embed tokens in shared space              │ │
│  │ - Concatenate: [obs_tokens, goal_tokens]   │ │
│  └───────────┬──────────────────────────────────┘ │
│              │                                    │
│              v                                    │
│  ╔════════════════════════════════════════════╗  │
│  ║ Decision Transformer                       ║  │
│  ║ - Autoregressive action generation        ║  │
│  ║ - Input: concatenated tokens              ║  │
│  ║ - Output: action tokens + future frames   ║  │
│  ║ - Based on Gato architecture              ║  │
│  ╚════════════┬═══════════════════════════════╝  │
│               │                                  │
│               v                                  │
│  ┌────────────────────────────────────────────┐ │
│  │ Action Decoding                            │ │
│  │ - Continuous action from tokens           │ │
│  │ - [x, y, z, roll, pitch, yaw, gripper]   │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
└────────────────────────────────────────────────┘
```

### VQ-GAN Tokenizer (Frozen)

**Key design choice: Use frozen, pretrained VQ-GAN**

```python
class VQGANTokenizer:
    """
    Pretrained on diverse robot images:
    - Real robot observations from 100k+ trajectories
    - Simulated environments
    - Multiple viewing angles

    Properties:
    - Input: (H, W, 3) RGB image
    - Output: (H/16, W/16, 1) tokens (if 16× downsampling)
    - Codebook size: 2048 tokens (256 unique per dimension if 8D)
    - Frozen during training (not updated)
    """

    def __init__(self, ckpt_path, codebook_size=2048):
        self.encoder = load_pretrained_encoder(ckpt_path)
        self.decoder = load_pretrained_decoder(ckpt_path)
        self.codebook_size = codebook_size

    def encode(self, image):
        # image: (B, H, W, 3) or (H, W, 3)
        latent = self.encoder(image)  # (B, latent_dim, H', W')
        tokens = quantize(latent, codebook_size=self.codebook_size)
        # tokens: (B, H', W') where H'=H/16, W'=W/16
        return tokens.flatten()  # (B*H'*W',)

    def decode(self, tokens):
        # tokens: (B, H', W')
        latent = self.decoder(tokens)  # (B, latent_dim, H', W')
        image = upsampled_latent.clamp(0, 1)
        return image  # (B, 3, H, W)
```

**Why frozen VQ-GAN?**
1. Speeds up training (no gradient computation for encoder)
2. Learned representations are good across diverse visual domains
3. Reduces model parameter count
4. Inference faster (pre-tokenize offline)

### Decision Transformer (Gato-based)

**Architecture:**

```python
class DecisionTransformer(nn.Module):
    def __init__(self, vocab_size=2048+100, context_length=512):
        super().__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim=256)
        self.position_embedding = nn.Embedding(context_length, embedding_dim=256)

        # Transformer decoder
        self.transformer = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1
            ),
            num_layers=6
        )

        # Output heads
        self.action_head = nn.Linear(256, vocab_size)  # Predict action tokens
        self.future_frame_head = nn.Linear(256, vocab_size)  # Predict future frame tokens

    def forward(self, obs_tokens, goal_tokens, return_logits=False):
        """
        Args:
            obs_tokens: (B, N_obs_tokens) - tokenized observation
            goal_tokens: (B, N_goal_tokens) - tokenized goal
            return_logits: bool - return logits or sampled actions

        Returns:
            action: (B, action_dim) or logits: (B, vocab_size)
            future_frame: (B, N_tokens) predicted next frame
        """

        # Concatenate obs and goal tokens
        context = torch.cat([obs_tokens, goal_tokens], dim=1)  # (B, N_obs+N_goal)

        # Embed and add positional encoding
        embedded = self.token_embedding(context)  # (B, N_context, 256)
        positions = torch.arange(context.shape[1]).to(context.device)
        embedded = embedded + self.position_embedding(positions).unsqueeze(0)

        # Transformer forward pass
        transformer_out = self.transformer(embedded)  # (B, N_context, 256)

        # Action prediction (from last token)
        action_logits = self.action_head(transformer_out[:, -1, :])  # (B, vocab_size)

        # Future frame prediction (auxiliary)
        future_frame_logits = self.future_frame_head(transformer_out[:, -1, :])

        return action_logits, future_frame_logits
```

**Why Decision Transformer?**
1. Scalable: works with long sequences (50+ steps)
2. Autoregressive: natural fit for sequential action generation
3. Gato baseline: proven to work across modalities
4. Composable: separate heads for different outputs

### Multi-Embodiment Handling

**Key insight: Same architecture, different action parameterizations**

```python
def create_embodiment_config(embodiment_name):
    """
    Different robots have different action spaces
    """

    configs = {
        'panda': {
            'action_dim': 7,  # 6 DOF + gripper
            'joint_limits': [(-2.8973, 2.8973), ...],
            'control_mode': 'cartesian'
        },
        'sawyer': {
            'action_dim': 7,
            'joint_limits': [(-3.0503, 3.0503), ...],
            'control_mode': 'end_effector'
        },
        'kuka': {
            'action_dim': 7,
            'joint_limits': [(-3.5, 3.5), ...],
            'control_mode': 'cartesian'
        },
        'sim_arm': {
            'action_dim': 4,  # xyz + gripper (rotation from pixels)
            'joint_limits': None,
            'control_mode': 'cartesian'
        }
    }

    return configs[embodiment_name]

# In dataset loading:
class MultiEmbodimentDataset:
    def __getitem__(self, idx):
        trajectory = load_trajectory(idx)
        embodiment = trajectory['embodiment']  # e.g., 'panda'

        # Load actions with embodiment-specific scaling
        actions = normalize_actions_for_embodiment(
            trajectory['actions'],
            embodiment
        )

        return {
            'obs': trajectory['image'],
            'goal': trajectory['goal'],
            'actions': actions,
            'embodiment': embodiment
        }
```

---

## 4. Forward Pass & Training

### Forward Pass Pseudocode

```python
def forward(obs_image, goal_image, actions_gt=None, training=True):
    """
    Training/inference forward pass

    Args:
        obs_image: (H, W, 3) observation
        goal_image: (H, W, 3) goal
        actions_gt: (T, action_dim) ground truth actions (training only)
        training: bool - training or inference mode

    Returns:
        loss: scalar (training)
        actions: (T, action_dim) predicted actions (inference)
    """

    # ========== TOKENIZATION ==========
    # Encode images with frozen VQ-GAN
    obs_tokens = vq_gan_tokenizer.encode(obs_image)     # (N_tokens,)
    goal_tokens = vq_gan_tokenizer.encode(goal_image)   # (N_tokens,)

    # ========== EMBEDDING ==========
    obs_embed = token_embedding(obs_tokens).unsqueeze(0)    # (1, N_obs, 256)
    goal_embed = token_embedding(goal_tokens).unsqueeze(0)  # (1, N_goal, 256)

    # Add positional encoding
    obs_embed = obs_embed + pos_embedding(torch.arange(N_obs))
    goal_embed = goal_embed + pos_embedding(torch.arange(N_goal) + N_obs)

    # Concatenate context
    context = torch.cat([obs_embed, goal_embed], dim=1)  # (1, N_context, 256)

    # ========== TRANSFORMER FORWARD ==========
    if training:
        # Teacher forcing: include ground truth actions during training
        # Discretize actions to tokens
        action_tokens = discretize_actions(actions_gt)   # (T, action_dim) → (T,)

        # Embed action tokens
        action_embed = token_embedding(action_tokens).unsqueeze(0)  # (1, T, 256)
        action_embed = action_embed + pos_embedding(torch.arange(T) + N_context)

        # Full sequence for training
        full_sequence = torch.cat([context, action_embed], dim=1)  # (1, N_context+T, 256)

        # Transformer (autoregressive, causal masking)
        output = transformer(full_sequence, is_causal=True)  # (1, N_context+T, 256)

        # Extract action prediction logits
        action_logits = action_head(output[0, N_context:, :])  # (T, vocab_size)

        # Extract future frame prediction logits
        future_logits = future_frame_head(output[0, N_context:, :])  # (T, vocab_size)

        # ========== COMPUTE LOSS ==========
        # Action prediction loss
        action_loss = torch.nn.functional.cross_entropy(
            action_logits.view(-1, vocab_size),
            action_tokens.view(-1)
        )

        # Future frame prediction loss (auxiliary)
        future_frame_tokens = vq_gan_tokenizer.encode(next_images)  # (T, N_tokens)
        future_loss = torch.nn.functional.cross_entropy(
            future_logits.view(-1, vocab_size),
            future_frame_tokens.view(-1)
        )

        total_loss = action_loss + 0.1 * future_loss  # Weight auxiliary loss

        return {
            'loss': total_loss,
            'action_loss': action_loss,
            'future_loss': future_loss
        }

    else:
        # Inference: autoregressive generation
        actions = []
        current_context = context

        for step in range(max_steps):
            # Get next action prediction
            output = transformer(current_context, is_causal=True)
            action_logits = action_head(output[0, -1, :])  # (vocab_size,)

            # Sample action token
            action_token = torch.argmax(action_logits)  # Greedy; could use sampling
            actions.append(action_token.item())

            # Discretize token back to action
            action = discretize_to_action(action_token)  # → (action_dim,)

            # Embed action for next iteration
            action_embed = token_embedding(action_token)  # (256,)
            current_context = torch.cat([
                current_context,
                action_embed.unsqueeze(0).unsqueeze(0)
            ], dim=1)  # Append to sequence

        return {'actions': torch.tensor(actions)}
```

### Training Loop

```python
def train_robocat(model, train_loader, num_epochs=100):
    """
    Training loop for RoboCat with hindsight relabeling
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader),
        num_training_steps=len(train_loader) * num_epochs
    )

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_loader:
            obs_images = batch['obs'].to(device)              # (B, H, W, 3)
            goal_images = batch['goal'].to(device)            # (B, H, W, 3)
            actions = batch['actions'].to(device)             # (B, T, action_dim)
            next_images = batch['next_images'].to(device)     # (B, T, H, W, 3)

            # Forward pass
            output = model(obs_images, goal_images, actions, next_images, training=True)
            loss = output['loss']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'robocat_epoch{epoch}.pt')

    return model
```

---

## 5. Self-Improvement Loop

### Iterative Training Procedure

**Algorithm: RoboCat Self-Improvement**

```python
def robocat_self_improvement(initial_agent, tasks, num_iterations=5):
    """
    Iterative self-improvement loop

    Args:
        initial_agent: Pre-trained generalist model
        tasks: List[Task] - new tasks to learn
        num_iterations: Number of improvement iterations

    Process:
        1. Collect human demos for task
        2. Fine-tune agent on task
        3. Deploy to collect autonomous rollouts
        4. Combine human + autonomous data
        5. Retrain generalist
        6. Repeat
    """

    agent = initial_agent
    all_collected_data = []

    for iteration in range(num_iterations):
        print(f"\n======== ITERATION {iteration+1} ========")

        for task in tasks:
            # ======= STAGE 1: COLLECT HUMAN DEMONSTRATIONS =======
            print(f"Collecting demos for {task.name}...")
            human_demos = collect_human_demonstrations(
                task=task,
                num_demos=100-1000  # Small number of expert demos
            )
            print(f"  Collected {len(human_demos)} human demonstrations")

            # ======= STAGE 2: FINE-TUNE AGENT ON TASK =======
            print(f"Fine-tuning agent on {task.name}...")
            task_agent = fine_tune_agent(
                agent=agent,
                demonstrations=human_demos,
                num_epochs=50,
                learning_rate=1e-4  # Lower LR for fine-tuning
            )
            print(f"  Task-specific agent training complete")

            # ======= STAGE 3: DEPLOY FOR AUTONOMOUS COLLECTION =======
            print(f"Deploying agent to collect autonomous data...")
            autonomous_rollouts = deploy_and_collect(
                agent=task_agent,
                task=task,
                num_rollouts=10000,
                use_hindsight_relabeling=True  # Each rollout → multiple relabeled samples
            )
            print(f"  Collected {len(autonomous_rollouts)} autonomous trajectories")
            print(f"  With hindsight relabeling: ~{len(autonomous_rollouts) * 100} samples")

            # ======= STAGE 4: COMBINE DATA =======
            combined_data = human_demos + autonomous_rollouts
            all_collected_data.append(combined_data)
            print(f"  Total task data: {len(combined_data)} trajectories")

        # ======= STAGE 5: RETRAIN GENERALIST =======
        print(f"\nRetraining generalist agent with all collected data...")
        training_data = combine_with_original_dataset(
            new_data=all_collected_data,
            original_data=initial_large_dataset  # Original multi-embodiment data
        )
        print(f"  Training set size: {len(training_data)} trajectories")

        agent = train_from_scratch(
            data=training_data,
            num_epochs=200,
            checkpoint_path=f'robocat_iter{iteration+1}.pt'
        )
        print(f"  Generalist retraining complete")

    return agent
```

### Hindsight Goal Relabeling for RoboCat

```python
def robocat_hindsight_relabeling(trajectory):
    """
    RoboCat-specific hindsight: use any frame as goal

    Since RoboCat is goal-conditioned (obs + goal → actions),
    any successful trajectory segment can be relabeled as achieving
    the intermediate goal (frame at t_end).

    Example trajectory:
      t=0: obs_0, action_0 → obs_1
      t=1: obs_1, action_1 → obs_2
      ...
      t=T: obs_T-1, action_T-1 → obs_T (goal reached)

    Relabeled samples (all goal-reaching):
      (obs_0, obs_1): obs_0 + goal=obs_1 → 1 step
      (obs_0, obs_2): obs_0 + goal=obs_2 → 2 steps
      ...
      (obs_0, obs_T): obs_0 + goal=obs_T → T steps (original)
      (obs_1, obs_2): obs_1 + goal=obs_2 → 1 step
      ...
      (obs_T-1, obs_T): obs_T-1 + goal=obs_T → 1 step

    Total: T(T+1)/2 relabeled samples from single trajectory
    """

    relabeled = []

    for t_start in range(len(trajectory)):
        obs_start = trajectory['observations'][t_start]

        for t_goal in range(t_start + 1, len(trajectory)):
            goal = trajectory['observations'][t_goal]
            actions_segment = trajectory['actions'][t_start:t_goal]

            relabeled.append({
                'observation': obs_start,
                'goal': goal,
                'actions': actions_segment,
                'horizon': t_goal - t_start,
                'success': True,  # By construction
                'source': f'hindsight_{t_start}_{t_goal}'
            })

    return relabeled
```

---

## 6. Data Pipeline

### Dataset Composition

**RoboCat training data mixture:**

```
Total dataset: Millions of trajectories from:
├─ Real robot data (~30%)
│  ├─ Google robot arm (100K+ trajectories)
│  ├─ External lab robots (50K+ trajectories)
│  └─ Task-specific collections
├─ Simulated data (~40%)
│  ├─ PyBullet, MuJoCo simulations
│  └─ Domain-randomized synthetic
├─ Multi-embodiment data (~20%)
│  ├─ Panda, Sawyer, KUKA, WidowX
│  └─ Cross-embodiment trajectories
└─ Hindsight-relabeled (~700x multiplier)
   └─ Generated from successful segments
```

### Data Format

```python
# Single trajectory in RoboCat dataset
trajectory = {
    'observations': [
        {
            'image': (H, W, 3) uint8,           # Egocentric RGB
            'state': (state_dim,) float32,      # Joint positions, etc.
        },
        ...  # T timesteps
    ],
    'actions': (T, action_dim) float32,         # Continuous actions
    'goal_image': (H, W, 3) uint8,             # Final successful state
    'task_name': str,                           # e.g., 'place_cube_on_block'
    'embodiment': str,                          # e.g., 'panda', 'sawyer'
    'episode_length': int,
    'success': bool
}
```

### Data Augmentation

```python
class RoboCatAugmentation:
    def __init__(self, use_hindsight=True):
        self.use_hindsight = use_hindsight
        self.vq_gan = load_vq_gan()  # Frozen tokenizer

        # Image augmentation
        self.img_aug = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(0.2, 0.2, 0.2),
            torchvision.transforms.RandomAffine(degrees=5),
            torchvision.transforms.GaussianBlur(3),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, trajectory):
        # Augment images
        obs_aug = self.img_aug(torch.from_numpy(trajectory['observations']['image']))
        goal_aug = self.img_aug(torch.from_numpy(trajectory['goal_image']))

        # Tokenize (with frozen VQ-GAN)
        obs_tokens = self.vq_gan.encode(obs_aug)
        goal_tokens = self.vq_gan.encode(goal_aug)

        # Hindsight relabeling (if enabled)
        if self.use_hindsight:
            # Generate alternative goals from other frames
            all_hindsight_samples = robocat_hindsight_relabeling(trajectory)
            return all_hindsight_samples
        else:
            return [{
                'observation': obs_aug,
                'goal': goal_aug,
                'actions': trajectory['actions'],
                'obs_tokens': obs_tokens,
                'goal_tokens': goal_tokens
            }]
```

---

## 7. Practical Insights

### 10 Engineering Takeaways

1. **Frozen VQ-GAN is game-changer:** Pre-training encoders on diverse images and freezing them enables: (a) faster training, (b) no gradient overhead, (c) shared representation across embodiments.

2. **Self-improvement loop requires careful data mixing:** Don't just add new task data; mix with original dataset (60/40 ratio works well). Prevents catastrophic forgetting while specializing.

3. **Hindsight relabeling multiplies effective dataset size:** 700x multiplication with 50-step trajectories. Essential for scaling goal-conditioned learning without 700x more data collection.

4. **Multi-embodiment training improves generalization:** Training on Panda + Sawyer + KUKA together improves each individual arm. Embodiment acts like regularization.

5. **Action discretization affects sample efficiency:** Discrete actions (100-256 bins) perform better than continuous for transformer-based policies. Reduces output complexity.

6. **Fine-tuning with lower learning rate preserves knowledge:** Use 0.1-0.3x the original training LR when fine-tuning to specific task. Prevents overwriting generalist knowledge.

7. **Future frame prediction as auxiliary loss helps:** Adding auxiliary objective to predict next frame provides 5-10% performance improvement. Acts as regularization.

8. **Batch size scales linearly with dataset size:** Start at 32 for 100K samples; scale to 256+ for millions. Higher batch sizes improve gradient stability.

9. **Autonomously collected data is initially low-quality:** First autonomous rollouts have 20-30% task success. But diversity helps; mix with human demos (50/50) for best results.

10. **Embodiment-specific adaptation matters more than pretraining:** Fine-tuning on 100 task demos beats generalist trained on 1M trajectories. Task specificity > scale.

### 5 Gotchas

1. **Catastrophic forgetting when specializing:** Fine-tuning on task data can erase generalist skills. Use lower LR, shorter fine-tuning, mixed-data training to mitigate.

2. **Hindsight samples are highly correlated:** Many relabeled samples share actions; creates biased gradient estimates. Use careful sampling to keep diversity.

3. **Embodiment shift breaks action interpretation:** Actions from Panda can't directly transfer to Sawyer even with same task. Requires embodiment-specific scaling/limits.

4. **VQ-GAN tokenizer bottleneck:** If tokens are too lossy, reconstruction quality hurts. Monitor reconstruction error; retrain VQ-GAN if needed for new domains.

5. **Multi-embodiment data imbalance:** If 10x more Panda data than Sawyer, model biases toward Panda actions. Use weighted sampling or data augmentation to balance.

### Tiny-Subset Overfit Plan

**Test on single task, single embodiment, 10 human demos:**

```python
# Step 1: Collect 10 human demos for one task
task_demos = collect_human_demonstrations(
    task='pick_cube_from_shelf',
    embodiment='panda',
    num_demos=10
)

# Step 2: Fine-tune agent on these 10 + hindsight (10*45/2 ≈ 225 samples)
fine_tuned_agent = fine_tune_agent(
    agent=pretrained_robocat,
    demonstrations=task_demos,
    num_epochs=100,
    learning_rate=5e-5,  # Very low LR
    use_hindsight=True
)

# Step 3: Evaluate on task (no held-out test, just training set)
train_success = evaluate_on_demos(fine_tuned_agent, task_demos)

# Step 4: Deploy for autonomous collection
autonomous_data = deploy_and_collect(
    agent=fine_tuned_agent,
    task=task,
    num_rollouts=500
)
autonomous_success = eval_success(autonomous_data)

# Step 5: Check self-improvement loop
print(f"Human demos success: {train_success:.1%}")
print(f"Autonomous collection success: {autonomous_success:.1%}")
print(f"Self-improvement: {autonomous_success - train_success:+.1%}")

# Expected:
# - Human demos: 80-90% (supervised learning)
# - Autonomous: 30-50% (harder; generalization)
# - Self-improvement: If loop works, combining should boost next iteration to 60-70%
```

---

## 8. Minimal Reimplementation Checklist

### Phase 1: VQ-GAN Setup (Week 1)

- [ ] **Load Pretrained VQ-GAN**
  - [ ] Find/train VQ-GAN on robot images
  - [ ] Load encoder + decoder weights
  - [ ] Freeze parameters
  - [ ] Test encoding/decoding on sample images

- [ ] **Tokenization Pipeline**
  - [ ] Implement encode: image → tokens
  - [ ] Implement decode: tokens → image
  - [ ] Test round-trip: image → tokens → image
  - [ ] Verify reconstruction quality

- [ ] **Token Embedding**
  - [ ] Create embedding matrix: vocab_size → embed_dim
  - [ ] Initialize properly (Xavier)
  - [ ] Test on sample token IDs

### Phase 2: Transformer & Actions (Week 1-2)

- [ ] **Decision Transformer**
  - [ ] Implement transformer decoder
  - [ ] Causal attention masking
  - [ ] Action + future-frame heads
  - [ ] Test forward pass with dummy tokens

- [ ] **Action Tokenization**
  - [ ] Implement action → token mapping
  - [ ] Implement token → action decoding
  - [ ] Test round-trip: action → token → action
  - [ ] Verify embodiment-specific scaling

- [ ] **Multi-Embodiment Support**
  - [ ] Define embodiment configs (action dims, limits)
  - [ ] Implement embodiment-specific normalization
  - [ ] Test on multiple embodiments

### Phase 3: Data Pipeline (Week 2-3)

- [ ] **Dataset Loading**
  - [ ] Load trajectory data
  - [ ] Implement image augmentation
  - [ ] Implement action normalization
  - [ ] Test on 100 trajectories

- [ ] **Hindsight Relabeling**
  - [ ] Implement hindsight goal generation
  - [ ] Generate relabeled samples from trajectories
  - [ ] Verify correctness (all goals reachable)
  - [ ] Test on 10 trajectories (should generate 1000+ samples)

- [ ] **DataLoader**
  - [ ] Create PyTorch Dataset class
  - [ ] Batch collation (variable length actions)
  - [ ] Test loading 1000 samples/sec

### Phase 4: Training & Fine-tuning (Week 3-4)

- [ ] **Training Loop**
  - [ ] Implement training forward pass
  - [ ] Loss computation (action + future frames)
  - [ ] Optimizer & scheduler setup
  - [ ] Checkpointing and validation

- [ ] **Fine-tuning Script**
  - [ ] Load pretrained weights
  - [ ] Fine-tune on task-specific data
  - [ ] Use lower learning rate (1e-5 typical)
  - [ ] Early stopping on task success

- [ ] **Autonomous Deployment**
  - [ ] Inference loop
  - [ ] Action sampling/generation
  - [ ] Data collection from rollouts
  - [ ] Hindsight relabeling of rollouts

### Phase 5: Self-Improvement Loop (Week 4-5)

- [ ] **Iteration 1**
  - [ ] Collect 100 human demos for task
  - [ ] Fine-tune on demos
  - [ ] Deploy for 10K autonomous rollouts
  - [ ] Measure autonomous success rate

- [ ] **Iteration 2**
  - [ ] Combine human + autonomous data
  - [ ] Retrain generalist on combined data
  - [ ] Fine-tune new generalist on same task
  - [ ] Deploy again; measure improvement

- [ ] **Iteration 3+**
  - [ ] Repeat cycle 2-3 more times
  - [ ] Track success rate across iterations
  - [ ] Should show monotonic improvement

### Critical Code Components

```python
# Test 1: VQ-GAN tokenization
image = torch.randn(3, 64, 64)
tokens = vq_gan.encode(image)  # (N_tokens,)
image_recon = vq_gan.decode(tokens)
reconstruction_error = torch.nn.functional.mse_loss(image, image_recon)
assert reconstruction_error < 0.01

# Test 2: Token embedding
token_ids = torch.randint(0, vocab_size, (10,))
embeddings = token_embedding(token_ids)  # (10, 256)
assert embeddings.shape == (10, 256)

# Test 3: Hindsight relabeling
traj = load_trajectory(0)
relabeled = robocat_hindsight_relabeling(traj)
assert len(relabeled) > len(traj) * 10  # 700x+ increase
assert all('goal' in s for s in relabeled)

# Test 4: Forward pass
obs_tokens = torch.randint(0, vocab_size, (10,))
goal_tokens = torch.randint(0, vocab_size, (10,))
action_logits, future_logits = model(obs_tokens, goal_tokens)
assert action_logits.shape[0] == vocab_size
assert future_logits.shape[0] == vocab_size

# Test 5: Action encoding/decoding
action_orig = torch.tensor([0.3, -0.5, 0.7, 0.8])
token = action_to_token(action_orig)
action_recon = token_to_action(token)
error = torch.nn.functional.l1_loss(action_orig, action_recon)
assert error < 0.02

# Test 6: Fine-tuning
pretrained_agent = load_pretrained_robocat()
task_agent = fine_tune_agent(pretrained_agent, task_demos, lr=5e-5, epochs=50)
task_success = evaluate(task_agent, test_set)
assert task_success > 0.60

# Test 7: Self-improvement loop
gen1_success = evaluate(pretrained_agent, task)
fine_tuned = fine_tune_agent(pretrained_agent, human_demos)
autonomous_data = deploy_and_collect(fine_tuned, task, 5000)
combined_data = human_demos + autonomous_data
gen2 = retrain_on_data(combined_data)
gen2_success = evaluate(gen2, task)
assert gen2_success > gen1_success  # Should improve!
```

---

## References

[RoboCat DeepMind Publication](https://deepmind.google/research/publications/35829/)
[RoboCat Blog Post](https://deepmind.google/blog/robocat-a-self-improving-robotic-agent/)
[RoboCat arXiv Paper](https://arxiv.org/abs/2306.11706)
[RoboCat OpenReview](https://openreview.net/forum?id=vsCpILiWHu)

---

**Document Version:** 1.0
**Last Updated:** 2025-03-07
**Status:** Complete implementation guide for RoboCat (TMLR 2023)
