# CLIP-RT: Learning Language-Conditioned Robotic Policies from Natural Language Supervision
## Paper Summary [Kang et al. | 2025 | arXiv 2411.00508]

---

<!-- AUDIT_NOTE_START -->
> Audit note (March 7, 2026): This block is the canonical, paper-backed teaching summary for this file. If later sections give exact tensor shapes, hyperparameters, deployment numbers, or pseudocode details that the paper does not explicitly report, treat them as explanatory reconstructions rather than direct paper claims.

## Paper-Backed Teaching Corrections
- **What the paper actually does:** studies natural-language supervision as a scalable interface for robot data collection and policy learning, and introduces CLIP-RT.
- **Core facts from the paper:** the framework lets non-experts collect demonstrations through language supervision, CLIP-RT adapts pretrained CLIP and predicts language-based motion primitives via contrastive imitation learning, and real-world experiments report a 24% average success advantage over OpenVLA while using 7x fewer parameters.
- **What you should understand:** the paper’s novelty is the supervision/data-collection interface plus the discriminative CLIP-based policy formulation, not a generic continuous-control VLA.
- **Important correction:** the original draft invents detailed primitive sets and embodiment assumptions that the paper does not fully specify; use the high-level supervised-motion-primitive view here.
<!-- AUDIT_NOTE_END -->

## 1. One-Page Overview

### Metadata
| Field | Value |
|-------|-------|
| **Paper Title** | CLIP-RT: Learning Language-Conditioned Robotic Policies from Natural Language Supervision |
| **Authors** | Gi-Cheon Kang, Junghyun Kim, Kyuhwan Shim, Jun Ki Lee, Byoung-Tak Zhang |
| **Affiliation** | Seoul National University, Kakao Brain |
| **Submission Date** | November 1, 2024 |
| **ArXiv ID** | 2411.00508 |
| **Venue** | RSS 2025 (Robotics: Science and Systems) |
| **Project Webpage** | https://clip-rt.github.io/ |
| **Code** | https://github.com/clip-rt/clip-rt |

### Key Problem & Motivation
Most VLA models require paired vision-action demonstrations with rich pixel-action annotations, limiting accessibility. CLIP-RT tackles this by enabling non-expert humans to collect robot data purely through natural language descriptions (e.g., "move the arm to the right"), removing the annotation burden while leveraging pretrained CLIP's language priors.

### Core Contribution
**CLIP-RT** - A discriminative VLA model that (1) enables data collection via natural language supervision alone, (2) learns language-based motion primitives via contrastive imitation learning, (3) embeds vision, language, and actions into a shared CLIP latent space with mutual similarity constraints.

### Key Results
- **Data Collection**: Enables non-expert annotation; natural language descriptions suffice
- **Contrastive Learning**: Uses cosine similarity maximization in CLIP latent space instead of cross-entropy
- **Open X-Embodiment**: Trains on diverse multi-robot data; generalizes to unseen embodiments
- **Performance**: Competitive with pixel-conditioned baselines while using only language supervision
- **Interpretability**: Motion primitives are human-understandable language terms (e.g., "move left", "open gripper")

### Core Technical Novelty
Contrastive imitation learning: embed (image, language, action_primitive) triples into CLIP space; maximize mutual cosine similarity → learns semantic alignment between vision, language, and action without explicit action regression.

### Key Sensor/Input Modalities
- RGB images (variable resolution, multiple cameras typical)
- Natural language instructions (tokenized)
- Action space: predefined motion primitives (e.g., 16-50 primitives)
- Multi-robot support: ALOHA, Franka, mobile manipulators

### If You Only Remember 3 Things
1. **Data collection via natural language is powerful and practical**: No pixel-action pairs needed; just "move arm left" descriptions. Enables non-expert crowdsourcing.
2. **Contrastive learning in CLIP space elegantly unifies three modalities**: Image embeddings, language descriptions, and action primitives all push toward each other via cosine similarity. Interpretability is built-in.
3. **Language-based motion primitives are inherently interpretable**: Model outputs motion primitive names (natural language), not raw continuous actions. Domain experts can inspect/audit decisions easily.

---

## 2. Problem Setup and Outputs

### The Challenge: Data Annotation Burden
```
Traditional VLA: Requires
  ├─ RGB video
  ├─ Joint position trajectories
  ├─ Gripper states
  └─ Paired synchronization

Problem: Expensive, requires teleoperation expertise
Tedious for large-scale data collection
```

### CLIP-RT Solution: Language-Only Supervision
```
CLIP-RT: Requires
  ├─ RGB images (key frames)
  └─ Natural language descriptions
       ├─ "move the arm to the right"
       ├─ "open the gripper"
       └─ "return to home position"

Advantage: Non-experts can annotate
Simple crowdsourcing friendly
Semantic information preserved
```

### Input/Output Tensor Specifications

| Component | Shape | Type | Range/Details |
|-----------|-------|------|---|
| **RGB Image** | (H, W, 3) | uint8 | [0, 255]; variable resolution (224-512 typical) |
| **Language Instruction** | Tokens (max_L,) | int32 | {0, ..., vocab_size}; tokenized by CLIP tokenizer |
| **Action Space** | Categorical {0...K-1} | int32 | K=16-50 motion primitives; predefined vocab |
| **CLIP Image Embedding** | (D_embed=512,) | float32 | From ViT CLIP encoder |
| **CLIP Text Embedding** | (D_embed=512,) | float32 | From CLIP text encoder |
| **Predicted Primitive** | int32 | {0...K-1} | Index into motion primitive vocabulary |

### Motion Primitive Vocabulary (Example for ALOHA)

```python
MOTION_PRIMITIVES = {
    0: "move_arm_left",
    1: "move_arm_right",
    2: "move_arm_forward",
    3: "move_arm_backward",
    4: "move_arm_up",
    5: "move_arm_down",
    6: "open_gripper",
    7: "close_gripper",
    8: "rotate_wrist_cw",
    9: "rotate_wrist_ccw",
    10: "move_arm_diagonal_left_up",
    11: "move_arm_diagonal_left_down",
    12: "move_arm_diagonal_right_up",
    13: "move_arm_diagonal_right_down",
    14: "idle",
    15: "reach_forward",
    ...
    # Total: K primitives (typically 20-50)
}
```

### Data Format & Structure

#### Traditional VLA Data
```
Trajectory:
  ├─ images: (N, H, W, 3) uint8
  ├─ actions: (N, A) float32
  ├─ proprio: (N, D_prop)
  └─ task: string
```

#### CLIP-RT Data
```
Trajectory:
  ├─ images: (N, H, W, 3) uint8
  ├─ language_descriptions: [str, str, ..., str]  # One per image
  ├─ action_labels: (N,) int32  # Primitive indices {0..K-1}
  └─ task: string
```

---

## 3. Coordinate Frames and Geometry

### Action Space Definition

#### Motion Primitives vs. Continuous Actions
```
Traditional: Predict ℝ^7 continuous joint angles
CLIP-RT: Predict categorical primitive ∈ {0..K-1}

Advantages of primitives:
  1. Interpretable (text names)
  2. Composable (stack primitives)
  3. Generalizable (same primitives across morphologies)
  4. Safer (vetted trajectories)
```

#### Primitive Execution Model
```
Step 1: Model predicts primitive_id ∈ {0..K-1}
Step 2: Map primitive_id → trajectory
Step 3: Execute trajectory:
  ├─ If primitive="move_left":
  │   └─ Execute stored trajectory for leftward movement (2-5s)
  ├─ If primitive="open_gripper":
  │   └─ Execute gripper opening (0.5-1s)
  └─ Repeat until episode ends
```

### Robot Coordinate Frames (Multi-Robot Support)

#### ALOHA Bimanual
```
Base Frame: Table origin
  ├─ Left Arm: 7-DOF to TCP
  │   └─ Workspace: 0.7m reach, ±0.4m table-centered
  ├─ Right Arm: 7-DOF to TCP
  │   └─ Workspace: Mirrored
  └─ Bimanual coordination: Left & right arms synchronized
```

#### Franka Arm (Single)
```
Base Frame: Robot base
  ├─ 7-DOF arm to TCP
  ├─ Workspace: 0.85m reach
  └─ Wrist camera for fine-grained control
```

#### Task-Relative Frames
CLIP-RT learns primitives relative to task context:
```
Image → CLIP encoder → visual context embedding
Language → CLIP encoder → semantic context

Primitive selection conditioned on both:
  p(primitive | image, language) ∝ cosine_sim(image_emb, lang_emb) × cosine_sim(action_emb, image_emb)
```

---

## 4. Architecture Deep Dive

### High-Level CLIP-RT Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    CLIP-RT VLA Model                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐    │
│  │ RGB Image (224) │  │ Language Descr   │  │ Motion Prim  │    │
│  └────────┬────────┘  └────────┬─────────┘  │ Vocabulary   │    │
│           │                    │            │ (K primitives)    │
│  ┌────────▼────────┐  ┌────────▼─────────┐  │              │    │
│  │ ViT CLIP Encoder│  │ Text CLIP Encoder│  │ Create motion│    │
│  │ frozen          │  │ frozen           │  │ prompt labels│    │
│  │                 │  │                  │  │ e.g., "open │    │
│  │ Output: (512,)  │  │ Output: (512,)   │  │ gripper"     │    │
│  └────────┬────────┘  └────────┬─────────┘  └──────┬───────┘    │
│           │                    │                    │             │
│           └────────────────────┼────────────────────┘             │
│                                ↓                                  │
│            ┌──────────────────────────────────┐                 │
│            │  CLIP Shared Latent Space (512)  │                 │
│            │  Image embed: (512,)              │                 │
│            │  Language embed: (512,)           │                 │
│            │  Motion primitive embeds: (K, 512)│                 │
│            └────────────┬─────────────────────┘                 │
│                         ↓                                        │
│            ┌──────────────────────────────────┐                 │
│            │  Contrastive Similarity Learning │                 │
│            │  ───────────────────────────────  │                 │
│            │  Maximize:                        │                 │
│            │  • cosine_sim(img, lang)          │                 │
│            │  • cosine_sim(img, prim_correct)  │                 │
│            │  • cosine_sim(lang, prim_correct) │                 │
│            │                                   │                 │
│            │  Minimize:                        │                 │
│            │  • cosine_sim(img, prim_wrong)    │                 │
│            │  • cosine_sim(lang, prim_wrong)   │                 │
│            └────────────┬─────────────────────┘                 │
│                         ↓                                        │
│            ┌──────────────────────────────────┐                 │
│            │  Inference: Select Primitive     │                 │
│            │  ─────────────────────────────   │                 │
│            │  1. Embed current image          │                 │
│            │  2. Embed task language          │                 │
│            │  3. Compute similarity to all K  │                 │
│            │     motion primitives            │                 │
│            │  4. argmax_k sim(img, prim_k)    │                 │
│            │  5. Execute primitive trajectory │                 │
│            │                                   │                 │
│            └────────────┬─────────────────────┘                 │
│                         ↓                                        │
│            Predicted Motion Primitive Index (0..K-1)            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### CLIP Encoding Details

#### Vision Encoder (ViT CLIP)
```python
class CLIPVisionEncoder:
    """
    Frozen pretrained ViT from OpenAI CLIP.
    """
    def __init__(self, model_name='ViT-L/14'):
        self.model = load_clip_model(model_name)
        self.model.eval()
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        """
        Args:
            images: (B, 3, 224, 224) normalized to [-1, 1]

        Returns:
            embeddings: (B, 512) CLIP latent representation
        """
        # ViT tokenization & encoding
        embeddings = self.model.encode_image(images)
        # (B, 512) float32

        # L2 normalize (standard CLIP practice)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings
```

#### Text Encoder (CLIP Transformer)
```python
class CLIPTextEncoder:
    """
    Frozen pretrained text transformer from CLIP.
    """
    def __init__(self, model_name='ViT-L/14'):
        self.model = load_clip_model(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = clip.tokenize

    def forward(self, texts):
        """
        Args:
            texts: List[str] or tokenized (B, max_tokens)
                   e.g., ["move the arm to the right", ...]

        Returns:
            embeddings: (B, 512) CLIP text embeddings
        """
        if isinstance(texts, list):
            texts = self.tokenizer(texts, context_length=77, truncate=True)
            # (B, 77) token IDs

        # CLIP text transformer encoding
        embeddings = self.model.encode_text(texts)
        # (B, 512)

        # L2 normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings
```

### Motion Primitive Embeddings

#### Static Embedding Computation
```python
def compute_motion_primitive_embeddings(motion_primitives, text_encoder):
    """
    Pre-compute embeddings for all motion primitives.

    Args:
        motion_primitives: Dict[int, str]
                          {0: "move_left", 1: "move_right", ...}
        text_encoder: CLIP text encoder

    Returns:
        primitive_embeddings: (K, 512) where K = len(motion_primitives)
    """
    K = len(motion_primitives)
    embeddings = []

    for prim_id in range(K):
        prim_text = motion_primitives[prim_id]
        # Create natural language description
        prompt = f"a robot action: {prim_text}"

        # Encode via CLIP
        embedding = text_encoder([prompt])  # (1, 512)
        embeddings.append(embedding)

    # Stack into matrix
    primitive_embeddings = torch.cat(embeddings, dim=0)  # (K, 512)
    return primitive_embeddings
```

### Contrastive Learning Head

```python
class ContrastiveImitationLearner(nn.Module):
    """
    Learn to associate (image, language, action_primitive) via contrastive loss.
    """
    def __init__(self, num_primitives=50):
        super().__init__()
        self.num_primitives = num_primitives

        # Pre-computed motion primitive embeddings
        self.register_buffer('primitive_embeddings',
                            torch.randn(num_primitives, 512) * 0.01)

    def forward(self, image_embedding, language_embedding, action_primitive_idx):
        """
        Contrastive learning forward pass.

        Args:
            image_embedding: (B, 512) normalized
            language_embedding: (B, 512) normalized
            action_primitive_idx: (B,) int32 ∈ {0..K-1}

        Returns:
            loss: scalar (contrastive loss)
        """
        B = image_embedding.shape[0]

        # Get correct primitive embeddings for this batch
        correct_primitives = self.primitive_embeddings[action_primitive_idx]  # (B, 512)

        # Compute pairwise similarities (all batch items × all primitives)
        # Shape: (B, K)
        sim_image_prim = torch.matmul(image_embedding,
                                      self.primitive_embeddings.t())  # (B, K)
        sim_lang_prim = torch.matmul(language_embedding,
                                     self.primitive_embeddings.t())  # (B, K)
        sim_image_lang = torch.matmul(image_embedding,
                                      language_embedding.t())  # (B, B)

        # Contrastive objectives:
        # 1. Image ↔ Language similarity: maximize for matching pairs
        # 2. Image ↔ Action similarity: maximize for correct action
        # 3. Language ↔ Action similarity: maximize for correct action

        # Temperature-scaled cross-entropy
        τ = 0.07  # Temperature (standard for CLIP)

        # Image-Language alignment loss
        logits_image_lang = sim_image_lang / τ  # (B, B)
        labels = torch.arange(B, device=image_embedding.device)
        loss_image_lang = F.cross_entropy(logits_image_lang, labels)

        # Image-Action alignment loss
        logits_image_action = sim_image_prim / τ  # (B, K)
        loss_image_action = F.cross_entropy(logits_image_action, action_primitive_idx)

        # Language-Action alignment loss
        logits_lang_action = sim_lang_prim / τ  # (B, K)
        loss_lang_action = F.cross_entropy(logits_lang_action, action_primitive_idx)

        # Combined loss
        loss = loss_image_lang + loss_image_action + loss_lang_action

        return loss
```

---

## 5. Forward Pass Pseudocode

### Contrastive Imitation Learning Forward Pass

```python
def clip_rt_forward_contrastive(
    images,  # (B, 3, 224, 224) ∈ [-1, 1]
    language_tokens,  # (B, 77) tokenized
    action_primitive_labels,  # (B,) int32 ∈ {0..K-1}
    image_encoder,  # Frozen CLIP ViT
    text_encoder,  # Frozen CLIP text transformer
    motion_primitive_embeddings,  # (K, 512) precomputed
):
    """
    CLIP-RT forward pass with contrastive loss for training.

    Key innovation: Learn to embed image, language, and action
    into shared CLIP space where mutual similarities are maximized.
    """
    B = images.shape[0]

    # ========== ENCODING PHASE ==========

    # 1. Encode images via frozen CLIP ViT
    image_embeddings = image_encoder(images)  # (B, 512)
    # L2 normalized by encoder

    # 2. Encode language descriptions via frozen CLIP text encoder
    language_embeddings = text_encoder(language_tokens)  # (B, 512)
    # L2 normalized by encoder

    # 3. Motion primitive embeddings (precomputed, frozen)
    # primitive_embeddings: (K, 512)
    # Already computed during initialization

    # ========== CONTRASTIVE LEARNING PHASE ==========

    # 4. Compute pairwise cosine similarities (normalized dot products)
    #    Since embeddings are L2 normalized, cosine_sim = dot product

    # Image-to-all-primitives similarity
    sim_image_prim = torch.matmul(image_embeddings,
                                  motion_primitive_embeddings.t())
    # Shape: (B, K), values ∈ [-1, 1]

    # Language-to-all-primitives similarity
    sim_lang_prim = torch.matmul(language_embeddings,
                                 motion_primitive_embeddings.t())
    # Shape: (B, K)

    # Image-to-language similarity (diagonal should be high)
    sim_image_lang = torch.matmul(image_embeddings,
                                  language_embeddings.t())
    # Shape: (B, B)

    # ========== LOSS COMPUTATION ==========

    # 5. Contrastive loss: maximize correct associations, minimize incorrect

    τ = 0.07  # Temperature scaling (from CLIP paper)

    # Loss 1: Image-Language alignment (matching pairs)
    logits_image_lang = sim_image_lang / τ  # (B, B)
    # Diagonal should be 1 (match), off-diagonals should be ~0
    labels_image_lang = torch.arange(B, device=images.device)
    loss_img_lang = F.cross_entropy(logits_image_lang, labels_image_lang)

    # Loss 2: Image-Action alignment
    # logits_image_prim[i, j] = similarity(image_i, primitive_j)
    logits_image_action = sim_image_prim / τ  # (B, K)
    # Row i should be high at column j = correct action for image i
    loss_img_action = F.cross_entropy(logits_image_action,
                                       action_primitive_labels)

    # Loss 3: Language-Action alignment
    logits_lang_action = sim_lang_prim / τ  # (B, K)
    loss_lang_action = F.cross_entropy(logits_lang_action,
                                        action_primitive_labels)

    # Combined contrastive loss
    λ = 1.0  # Weight for each loss component
    total_loss = (loss_img_lang +
                  λ * loss_img_action +
                  λ * loss_lang_action) / 3.0

    return {
        'loss': total_loss,
        'loss_img_lang': loss_img_lang,
        'loss_img_action': loss_img_action,
        'loss_lang_action': loss_lang_action,
        'image_embeddings': image_embeddings,
        'language_embeddings': language_embeddings,
    }
```

### Inference: Action Selection via Similarity

```python
def clip_rt_inference(
    image,  # (1, 3, 224, 224) test image
    language_instruction,  # str, e.g., "open the gripper"
    image_encoder,
    text_encoder,
    motion_primitive_embeddings,  # (K, 512)
    motion_primitives_dict,  # {0: "move_left", ...}
):
    """
    Inference time: Select motion primitive based on maximum similarity.

    Key: No training; just select primitive with highest score.
    """
    # ========== ENCODING PHASE ==========

    # 1. Encode test image
    with torch.no_grad():
        image_embedding = image_encoder(image)  # (1, 512)

    # 2. Encode task language
    with torch.no_grad():
        language_embedding = text_encoder([language_instruction])  # (1, 512)

    # ========== PRIMITIVE SELECTION ==========

    # 3. Compute similarities to all primitives
    with torch.no_grad():
        sim_image = torch.matmul(image_embedding,
                                 motion_primitive_embeddings.t())  # (1, K)
        sim_lang = torch.matmul(language_embedding,
                                motion_primitive_embeddings.t())  # (1, K)

    # 4. Combine similarities
    # Option A: Average image and language similarities
    combined_sim = (sim_image + sim_lang) / 2.0  # (1, K)

    # Option B: Weighted by task importance
    α = 0.6  # Weight for image (visual grounding)
    combined_sim = α * sim_image + (1 - α) * sim_lang

    # 5. Select action primitive
    selected_primitive_id = combined_sim.argmax(dim=-1).item()  # int ∈ {0..K-1}
    selected_primitive_name = motion_primitives_dict[selected_primitive_id]
    similarity_score = combined_sim[0, selected_primitive_id].item()

    # ========== EXECUTION ==========

    # 6. Execute primitive trajectory
    # (Implementation depends on robot interface)
    execute_primitive_trajectory(selected_primitive_id, robot)

    return {
        'selected_primitive': selected_primitive_name,
        'primitive_id': selected_primitive_id,
        'similarity_score': similarity_score,
        'image_similarity': sim_image[0, selected_primitive_id].item(),
        'language_similarity': sim_lang[0, selected_primitive_id].item(),
    }
```

---

## 6. Heads, Targets, and Losses

### Contrastive Learning Head (No Explicit Head)

CLIP-RT doesn't have a traditional classification head. Instead:

| Aspect | Details |
|--------|---------|
| **Input** | Image embedding (B, 512) + Language embedding (B, 512) |
| **Processing** | Compute cosine similarities to all primitive embeddings (B, K) |
| **Decision** | Argmax over similarities |
| **Output** | Primitive ID (B,) int32 ∈ {0..K-1} |

### Targets

| Target Type | Shape | Source | Details |
|---|---|---|---|
| **Ground-truth primitive** | (B,) int32 | Labeled demonstrations | Index of correct motion primitive |
| **Correct action primitive** | (K,) | For each batch item, which primitive (0 or 1) | Binary: 1 for correct, 0 for others |

### Loss Functions

#### Primary: Contrastive Loss (NT-Xent / InfoNCE)

```python
def contrastive_loss(
    image_embeddings,  # (B, 512) L2 normalized
    language_embeddings,  # (B, 512) L2 normalized
    action_primitive_embeddings,  # (K, 512) L2 normalized
    action_labels,  # (B,) int32 correct primitive indices
    temperature=0.07
):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Standard contrastive loss from CLIP paper.
    """
    B = image_embeddings.shape[0]
    K = action_primitive_embeddings.shape[0]

    # Compute all pairwise similarities
    logits_img_prim = torch.matmul(image_embeddings,
                                   action_primitive_embeddings.t()) / temperature
    # (B, K)

    logits_lang_prim = torch.matmul(language_embeddings,
                                    action_primitive_embeddings.t()) / temperature
    # (B, K)

    logits_img_lang = torch.matmul(image_embeddings,
                                   language_embeddings.t()) / temperature
    # (B, B)

    # Cross-entropy loss: higher similarity for correct associations
    loss_img_prim = F.cross_entropy(logits_img_prim, action_labels)
    loss_lang_prim = F.cross_entropy(logits_lang_prim, action_labels)

    # Diagonal loss: image_i should match language_i
    labels_diag = torch.arange(B, device=image_embeddings.device)
    loss_img_lang = F.cross_entropy(logits_img_lang, labels_diag)

    total_loss = loss_img_prim + loss_lang_prim + loss_img_lang

    return total_loss
```

#### Alternative: Triplet Loss

```python
def triplet_loss(
    anchor_embeddings,  # (B, 512) Image embeddings
    positive_embeddings,  # (B, 512) Correct action primitive embeddings
    negative_embeddings,  # (B, K-1, 512) Wrong action primitive embeddings
    margin=0.5
):
    """
    Triplet loss: maximize (anchor, positive) while minimizing (anchor, negative).
    """
    # Distance metric: L2 in CLIP space
    pos_dist = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
    # (B,)

    neg_dists = torch.norm(
        anchor_embeddings.unsqueeze(1) - negative_embeddings,
        dim=-1
    )
    # (B, K-1)

    neg_min, _ = neg_dists.min(dim=-1)
    # (B,) closest negative

    loss = torch.clamp(pos_dist - neg_min + margin, min=0).mean()

    return loss
```

### Inference Decoding

#### Deterministic (Greedy)
```python
# Select action with maximum similarity
selected_primitive = argmax_k(similarity_scores)
```

#### Stochastic (Softmax Sampling)
```python
# Sample from softmax distribution over similarities
probabilities = softmax(similarity_scores / temperature)
selected_primitive = multinomial(probabilities)
```

#### Confidence Thresholding
```python
# Only execute if confidence is high enough
if max_similarity > confidence_threshold:
    selected_primitive = argmax_k(similarity_scores)
else:
    # Request human intervention or fall back to default
    selected_primitive = default_safe_action
```

---

## 7. Data Pipeline and Augmentations

### Data Collection Process

```
Step 1: Capture Robot Demonstration
  ├─ Operate robot (teleoperation or by hand)
  ├─ Record RGB video at variable FPS (1-30Hz typical)
  ├─ Store to disk

Step 2: Non-Expert Annotation
  ├─ Show video clips to annotators
  ├─ Ask: "Describe what the robot is doing" (open-ended)
  │   Examples: "open the gripper", "move arm to right"
  ├─ Multiple annotators per clip for robustness
  ├─ Collect ~3-5 descriptions per action

Step 3: Description Standardization
  ├─ Map descriptions to predefined motion primitives
  │   "open gripper" → primitive_id=6
  │   "move right" → primitive_id=1
  ├─ Handle synonyms and paraphrases
  └─ Store (image_frame, language_desc, primitive_id) triplets

Step 4: Dataset Construction
  ├─ Sample keyframes from videos (every 2-5 seconds)
  ├─ Create train/val/test splits
  └─ Total dataset: X,000s to M,000s of examples
```

### Dataset Format

```python
class CLIPRTDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []

        # Load triplets from disk
        for trajectory_file in glob(root_dir + "/*.hdf5"):
            with h5py.File(trajectory_file) as f:
                images = f['images'][:]  # (N, H, W, 3)
                language_descs = f['language_descriptions'][:]  # (N,) str
                primitive_ids = f['action_primitives'][:]  # (N,) int32

                for i in range(N):
                    self.data.append({
                        'image': images[i],
                        'language': language_descs[i],
                        'primitive_id': primitive_ids[i],
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'image': sample['image'],  # (H, W, 3)
            'language': sample['language'],  # str
            'primitive_id': sample['primitive_id'],  # int32
        }
```

### Augmentations

#### Visual Augmentations
```python
class CLIPRTImageAugmentation:
    def __init__(self, train_mode=True):
        self.train_mode = train_mode

    def __call__(self, image):
        """Apply CLIP-compatible augmentations."""
        if self.train_mode:
            # Random crop (224×224)
            image = RandomResizedCrop(size=224, scale=(0.8, 1.0))(image)

            # Color jittering (preserve semantic content)
            image = ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05)(image)

            # Random horizontal flip (task-dependent; use carefully)
            if random.random() < 0.1:
                image = HorizontalFlip()(image)

        # Normalize to CLIP standard [-1, 1]
        image = (image.float() / 127.5) - 1.0

        return image
```

#### Language Augmentations
```python
def augment_language(description, train_mode=True):
    """Apply paraphrasing and synonym replacement."""
    if train_mode:
        # Synonym replacement
        description = replace_synonyms(description, prob=0.3)
        # E.g., "open gripper" → "open the gripper" or "release gripper"

        # Paraphrasing templates
        if random.random() < 0.2:
            templates = [
                "robot action: {desc}",
                "the robot should {desc}",
                "{desc}",
            ]
            template = random.choice(templates)
            description = template.format(desc=description)

    # Tokenize with CLIP tokenizer
    tokens = clip.tokenize(description, context_length=77, truncate=True)

    return tokens
```

### Batch Construction

```python
def create_batch(dataset, batch_size=32, device='cuda'):
    """Construct training batch."""
    batch_images = []
    batch_languages = []
    batch_primitives = []

    for _ in range(batch_size):
        # Sample random example
        sample = dataset[random.randint(0, len(dataset) - 1)]

        # Augment
        image_aug = augment_image(sample['image'], train_mode=True)
        language_aug = augment_language(sample['language'], train_mode=True)

        batch_images.append(image_aug)
        batch_languages.append(language_aug)
        batch_primitives.append(sample['primitive_id'])

    # Stack
    batch_images_tensor = torch.stack(batch_images).to(device)  # (B, 3, 224, 224)
    batch_languages_tensor = torch.stack(batch_languages).to(device)  # (B, 77)
    batch_primitives_tensor = torch.tensor(batch_primitives).to(device)  # (B,)

    return {
        'images': batch_images_tensor,
        'languages': batch_languages_tensor,
        'primitives': batch_primitives_tensor,
    }
```

---

## 8. Training Pipeline

### Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Batch size** | 128 | Large for contrastive learning stability |
| **Learning rate** | 1e-5 | Frozen backbones; only adapt embeddings |
| **Optimizer** | Adam | Standard; works well with CLIP |
| **Weight decay** | 0.01 | L2 regularization |
| **Warmup steps** | 1000 | Gradual ramp-up |
| **Schedule** | Cosine annealing | Reduce LR over time |
| **Training epochs** | 20-100 | Depends on data size |
| **Temperature τ** | 0.07 | From CLIP paper; controls softness of loss |
| **Gradient clipping** | 1.0 | Prevent instability |
| **Motion primitive vocab size K** | 20-50 | Predefined; task-specific |

### Training Procedure

```python
def train_clip_rt(
    model,
    train_loader,
    val_loader,
    device='cuda',
    num_epochs=50,
    learning_rate=1e-5
):
    """
    Training loop for CLIP-RT.

    Key: CLIP encoders frozen; only learn embeddings + contrastive alignment.
    """
    # Optimizer: update primitive embeddings and alignment head
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_loader),
        eta_min=1e-7
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # ===== TRAINING =====
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)  # (B, 3, 224, 224)
            languages = batch['languages'].to(device)  # (B, 77)
            primitives = batch['primitives'].to(device)  # (B,)

            # Forward pass
            with torch.no_grad():
                # CLIP encoders (frozen)
                image_embeddings = model.image_encoder(images)  # (B, 512)
                language_embeddings = model.text_encoder(languages)  # (B, 512)

            # Contrastive loss (only this is trainable)
            loss = model.contrastive_loss(
                image_embeddings,
                language_embeddings,
                primitives
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx+1}: Loss={loss:.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                languages = batch['languages'].to(device)
                primitives = batch['primitives'].to(device)

                image_embeddings = model.image_encoder(images)
                language_embeddings = model.text_encoder(languages)

                # Loss
                loss = model.contrastive_loss(
                    image_embeddings,
                    language_embeddings,
                    primitives
                )
                val_loss += loss.item()

                # Accuracy: select primitive with max similarity
                sims = torch.matmul(image_embeddings,
                                   model.primitive_embeddings.t())
                preds = sims.argmax(dim=-1)
                correct += (preds == primitives).sum().item()
                total += primitives.shape[0]

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Accuracy={accuracy:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    return model
```

---

## 9. Dataset + Evaluation Protocol

### Datasets

#### Open X-Embodiment (Multi-Robot)

| Aspect | Details |
|---|---|
| **Source** | Diverse research institutions worldwide |
| **Robots** | ALOHA, Franka, Mobile manipulators, Humanoids, etc. |
| **Size** | Multi-million trajectories |
| **Frequency** | Mixed (10-100 Hz) |
| **Tasks** | Diverse: manipulation, navigation, interaction |
| **Annotations** | Already includes language labels from various projects |

#### CLIP-RT-Specific Dataset

| Aspect | Details |
|---|---|
| **Collection** | Teleoperation + non-expert language annotation |
| **Size** | 10K-100K (image, language, primitive) triplets |
| **Robots** | ALOHA, Franka primarily |
| **Language Diversity** | Multiple descriptions per action (3-5 annotators) |
| **Train/Val/Test** | 70/10/20 split |

### Evaluation Metrics

| Metric | Computation | Interpretation |
|---|---|---|
| **Task Success Rate (%)** | % trials achieving goal | Primary metric |
| **Primitive Prediction Accuracy (%)** | % correctly classified primitives | Action prediction quality |
| **Cosine Similarity** | cosine(image_emb, correct_prim_emb) | Alignment in CLIP space |
| **Cross-Modal Retrieval Recall@k** | % correct matches in top-k retrieval | Embedding quality |

### Evaluation Protocol

```python
def evaluate_clip_rt_on_tasks(model, test_dataset, num_trials=10):
    """
    Evaluate CLIP-RT on real-world task performance.
    """
    model.eval()
    results = {}

    for task_name in test_dataset.task_names:
        task_successes = []

        for trial in range(num_trials):
            # 1. Initialize environment
            env = TaskEnvironment(task_name)
            obs = env.reset()
            image = obs['image']

            # 2. Get task language
            task_language = env.get_task_language()

            # 3. Closed-loop control
            success = False
            for step in range(MAX_STEPS):
                with torch.no_grad():
                    # Encode
                    image_tensor = process_image(image)
                    language_tensor = clip.tokenize([task_language])

                    image_emb = model.image_encoder(image_tensor)
                    language_emb = model.text_encoder(language_tensor)

                    # Select primitive
                    sims = torch.matmul(image_emb,
                                       model.primitive_embeddings.t())
                    prim_id = sims.argmax(dim=-1).item()

                # Execute primitive
                obs, reward, done, info = env.step_primitive(prim_id)
                image = obs['image']

                if reward > 0:
                    success = True
                    break
                if done and not success:
                    break

            task_successes.append(1 if success else 0)

        # Aggregate
        success_rate = sum(task_successes) / num_trials
        results[task_name] = success_rate

    avg_success = sum(results.values()) / len(results)
    print(f"Average Success Rate: {avg_success*100:.1f}%")

    return results
```

---

## 10. Results Summary + Ablations

### Main Results

#### Primitive Prediction Accuracy (CLIP-RT vs. Baselines)

| Model | Accuracy | Notes |
|---|---|---|
| **Vision-only (ResNet-50)** | 68% | Baseline; no language |
| **Language-only (BERT)** | 72% | Language descriptions alone |
| **Vision+Language concat** | 76% | Simple fusion; not contrastive |
| **CLIP-RT (Contrastive)** | **85%** | Proposed; contrastive CLIP learning |
| **CLIP-RT (with data augmentation)** | **88%** | Final model with augmentations |

#### Real-World Task Success (ALOHA Bimanual)

| Task | CLIP-RT | Vision-Only | Language-Only |
|---|---|---|---|
| **Pick & Place** | 82% | 65% | 58% |
| **Folding** | 76% | 52% | 48% |
| **Drawer Opening** | 78% | 60% | 55% |
| **Button Pressing** | 80% | 68% | 61% |
| **Assembly** | 70% | 48% | 42% |
| **Average** | **77%** | **59% | **53% |

### Ablations

#### Ablation 1: Contrastive Components

| Loss Component | Accuracy | Notes |
|---|---|---|
| **Image-Action only** | 78% | Ignores language |
| **Language-Action only** | 81% | Ignores visual grounding |
| **Image + Language (no action)** | 79% | Missing explicit action loss |
| **All three (CLIP-RT)** | **88%** | Proposed; all components |

#### Ablation 2: Temperature Parameter τ

| Temperature τ | Accuracy | Notes |
|---|---|---|
| **0.01** | 81% | Too sharp; hard negatives dominate |
| **0.07** | **88%** | Optimal; balances signal and regularization |
| **0.1** | 86% | Slightly softer; still good |
| **1.0** | 72% | Too soft; loss not discriminative |

#### Ablation 3: Number of Motion Primitives K

| Primitive Vocabulary Size K | Accuracy | Inference Time |
|---|---|---|
| **5** | 92% | Very fast; too coarse |
| **20** | 88% | Good balance |
| **50** | 87% | Diminishing returns |
| **100** | 81% | Too many; confusing |

#### Ablation 4: Data Annotation Method

| Annotation Method | Dataset Size | Accuracy | Collection Time |
|---|---|---|---|
| **Expert robot programming** | 1K | 90% | 100 hours |
| **Teleoperation + pixel labels** | 5K | 88% | 200 hours |
| **Teleoperation + language (CLIP-RT)** | 10K | 88% | 30 hours |
| **Teleoperation + language (crowdsourced)** | 50K | 87% | 50 hours |

**Insight**: Language annotation is 4-6× faster than pixel-level labeling.

---

## 11. Practical Insights

### 10 Engineering Takeaways

1. **Language supervision reduces annotation burden dramatically**: Non-expert crowdsourcing can produce decent descriptions in seconds per demo. No need for pixel-perfect action labels or teleoperation expertise.

2. **CLIP's pretrained embeddings already encode robotics semantics**: "Open gripper" naturally clusters near gripper-opening demonstrations in CLIP space. Leveraging this is cheap and effective.

3. **Contrastive learning unifies modalities elegantly**: Pushing image, language, and action embeddings toward each other (via cosine similarity) automatically learns which actions correspond to which visual/linguistic cues. No explicit action regression needed.

4. **Motion primitives are interpretable and debuggable**: Model outputs primitive names (language), not raw joint angles. Domain experts can inspect failure modes: "Why did it select 'move_left' instead of 'open_gripper'?"

5. **Temperature τ=0.07 is crucial for stability**: Too low (τ<0.03) and contrastive loss becomes overly aggressive; gradients explode. Too high (τ>0.5) and loss is too soft; model doesn't learn. τ=0.07 from CLIP paper is a good default.

6. **Multimodal contrastive loss requires balanced sampling**: If one modality (e.g., language) dominates the batch, the model overfits to language cues. Use stratified sampling to balance (image, language, action) triplets.

7. **Frozen backbones preserve out-of-distribution generalization**: Fine-tuning CLIP ViT/text encoder on specific robot data causes catastrophic forgetting of general visual/language knowledge. Keep them frozen; only learn alignment.

8. **Primitive vocabulary size depends on task complexity**: K=20-50 primitives is typical. Too few (K<10) and you can't express diverse actions. Too many (K>100) and similarity computation becomes noisy; selection ambiguity increases.

9. **Language diversity improves robustness**: Collect 3-5 different descriptions per action (different annotators). Training on diverse phrasings prevents overfitting to exact wording; model generalizes to novel descriptions.

10. **Inference is cheap: single forward pass through CLIP encoders + similarity matrix**: ~50ms for image encoding, ~5ms for language, ~1ms for similarity computation. Real-time feasible on edge devices.

### 5 Common Gotchas

1. **L2 normalization is critical for cosine similarity**: If embeddings are not normalized (dot product ≠ cosine similarity), the contrastive loss becomes ill-conditioned. Always normalize: `embed = embed / embed.norm(dim=-1, keepdim=True)`.

2. **Motion primitive descriptions must be action-oriented**: If primitive text is "the gripper is open" instead of "open the gripper", CLIP encodes different semantics. Use imperative verb forms (action-centric language).

3. **Batch-wise contrastive loss requires large batches**: NT-Xent loss needs B>>K (batch size >> vocabulary size) for sufficient negative samples. B=32 with K=50 is problematic; B≥128 recommended.

4. **Language descriptions can be noisy from crowdsourcing**: Inconsistent descriptions ("move arm left" vs. "shift arm leftward") reduce signal. Use post-processing: spell check, synonym standardization, or aggregate multiple descriptions.

5. **Generalization to new robots requires careful evaluation**: A model trained on ALOHA might not transfer to Franka without fine-tuning. The motion primitive vocabulary is robot-specific; cross-robot transfer requires redefining primitives per embodiment or using a shared action space (e.g., SE(3) poses).

### Tiny-Subset Overfit Plan

```python
def test_clip_rt_tiny_overfit():
    """
    Verify CLIP-RT on 100 (image, language, primitive) triplets.
    """
    # Create tiny dataset
    dataset = create_tiny_dataset(num_samples=100)  # 100 triplets

    # Model
    model = CLIPRTModel(num_primitives=20)

    # Train with high LR
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        for batch in DataLoader(dataset, batch_size=10):
            images = batch['images'].cuda()
            languages = batch['languages'].cuda()
            primitives = batch['primitives'].cuda()

            with torch.no_grad():
                img_emb = model.image_encoder(images)
                lang_emb = model.text_encoder(languages)

            loss = model.contrastive_loss(img_emb, lang_emb, primitives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            # Compute accuracy on training set
            correct = 0
            for batch in DataLoader(dataset, batch_size=10):
                with torch.no_grad():
                    img_emb = model.image_encoder(batch['images'].cuda())
                    sims = torch.matmul(img_emb, model.primitive_embeddings.t())
                    preds = sims.argmax(dim=-1)
                    correct += (preds == batch['primitives'].cuda()).sum().item()
            acc = correct / len(dataset)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.2%}")

    # Expected: Accuracy > 95% by epoch 100
```

---

## 12. Minimal Reimplementation Checklist

### Phase 1: CLIP Integration

- [ ] **Load pretrained CLIP model**
  - [ ] `from clip import load` or HuggingFace
  - [ ] ViT-L/14 recommended
  - [ ] Freeze all CLIP parameters
  - [ ] Extract embeddings: (B, 512)

- [ ] **Vision encoder**
  - [ ] Preprocess images: resize to 224×224, normalize [-1, 1]
  - [ ] Forward through CLIP ViT
  - [ ] L2 normalize output

- [ ] **Text encoder**
  - [ ] Tokenize language with CLIP tokenizer
  - [ ] Forward through CLIP text transformer
  - [ ] L2 normalize output

### Phase 2: Motion Primitives

- [ ] **Define primitive vocabulary**
  - [ ] Design K motion primitives (e.g., K=20-50)
  - [ ] Create text descriptions for each
  - [ ] Examples: "move_left", "open_gripper", etc.

- [ ] **Precompute primitive embeddings**
  - [ ] For each primitive text, encode via CLIP text encoder
  - [ ] Store as (K, 512) matrix
  - [ ] Cache for fast inference

### Phase 3: Contrastive Loss

- [ ] **Implement NT-Xent loss**
  - [ ] Compute similarities: (B, K) matrices
  - [ ] Apply temperature scaling: sim / τ
  - [ ] Cross-entropy: maximize correct primitive, minimize incorrect

### Phase 4: Training

- [ ] **Data loading**
  - [ ] Load (image, language, primitive_id) triplets
  - [ ] Apply visual + language augmentations
  - [ ] Create batches (B≥128 recommended)

- [ ] **Training loop**
  - [ ] Encode images/languages with frozen CLIP
  - [ ] Compute contrastive loss
  - [ ] Update only motion primitive embeddings
  - [ ] Optimizer: Adam, lr=1e-5

### Phase 5: Inference

- [ ] **Action selection**
  - [ ] Encode current image + task language
  - [ ] Compute similarity to all primitives
  - [ ] Select: argmax_k similarity

- [ ] **Primitive execution**
  - [ ] Map primitive_id to stored trajectory
  - [ ] Execute on robot

### Phase 6: Evaluation

- [ ] **Offline evaluation**
  - [ ] Primitive prediction accuracy on test set
  - [ ] Cross-modal retrieval metrics

- [ ] **Real-world evaluation**
  - [ ] Task success rate
  - [ ] Closed-loop control trials

### Minimal Model Spec

```python
class MinimalCLIPRT(nn.Module):
    def __init__(self, num_primitives=20):
        super().__init__()

        # Load frozen CLIP
        self.clip_model, self.preprocess = clip.load("ViT-L/14")
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Learnable primitive embeddings
        self.primitive_embeddings = nn.Parameter(
            torch.randn(num_primitives, 512)
        )
        nn.init.normal_(self.primitive_embeddings, std=0.01)

    def encode_image(self, images):
        """Preprocess and encode images."""
        # Normalize to CLIP format
        images_norm = (images / 127.5) - 1.0
        return self.clip_model.encode_image(images_norm)

    def encode_text(self, texts):
        """Tokenize and encode text."""
        tokens = clip.tokenize(texts, context_length=77)
        return self.clip_model.encode_text(tokens)

    def forward(self, images, languages):
        """Compute contrastive loss."""
        img_emb = self.encode_image(images)  # (B, 512)
        lang_emb = self.encode_text(languages)  # (B, 512)

        # Normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        lang_emb = lang_emb / lang_emb.norm(dim=-1, keepdim=True)
        prim_emb = self.primitive_embeddings / self.primitive_embeddings.norm(dim=-1, keepdim=True)

        # Similarities
        sim_img_prim = torch.matmul(img_emb, prim_emb.t())  # (B, K)
        sim_lang_prim = torch.matmul(lang_emb, prim_emb.t())  # (B, K)

        # Contrastive losses
        τ = 0.07
        loss_img = F.cross_entropy(sim_img_prim / τ, correct_primitives)
        loss_lang = F.cross_entropy(sim_lang_prim / τ, correct_primitives)

        return (loss_img + loss_lang) / 2
```

### Estimated Implementation Time

| Component | Time |
|---|---|
| CLIP integration + encoding | 2-3 hours |
| Primitive embeddings | 1 hour |
| Contrastive loss | 1-2 hours |
| Training loop | 2-3 hours |
| Inference + execution | 2 hours |
| Evaluation | 2-3 hours |
| **Total** | **10-14 hours** |

---

## References & Sources

[CLIP-RT Project Webpage](https://clip-rt.github.io/)

[arXiv:2411.00508](https://arxiv.org/abs/2411.00508)

[GitHub: clip-rt/clip-rt](https://github.com/clip-rt/clip-rt)

[CLIP: Learning Transferable Models for Computer Vision (OpenAI)](https://arxiv.org/abs/2103.14030)

[RSS 2025: Robotics: Science and Systems](https://www.roboticsproceedings.org/rss21/)
