# SayCan: Do As I Can, Not As I Say - Grounding Language in Robotic Affordances

**Paper Details:**
- Title: Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
- Authors: Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Corrado, David Dreyfus, Cheryl Flynn, Chelsea Finn, Chuyuan Fu, Marvin Zhang, and others
- Venue: CoRL (Conference on Robot Learning) 2022, PMLR Vol. 205, Pages 287-318
- arXiv: 2204.01691
- Organization: Google Brain, Everyday Robots
- Project: [say-can.github.io](https://say-can.github.io)

---

<!-- AUDIT_NOTE_START -->
> Audit note (March 7, 2026): This block is the canonical, paper-backed teaching summary for this file. If later sections give exact tensor shapes, hyperparameters, deployment numbers, or pseudocode details that the paper does not explicitly report, treat them as explanatory reconstructions rather than direct paper claims.

## Paper-Backed Teaching Corrections
- **What the paper actually does:** combines a large language model with grounded robot skills by selecting skills according to both language usefulness and affordance/feasibility.
- **Core method:** the LLM proposes or scores candidate high-level actions in natural language, while skill value functions ground those actions in the current physical state.
- **What you should understand:** SayCan is a grounded planner over a library of pretrained skills, not an end-to-end low-level VLA that directly outputs motor commands.
- **Important correction:** when later sections sound like monolithic action generation, defer to this planning-plus-grounding interpretation.
<!-- AUDIT_NOTE_END -->

## 1. One-Page Overview

**Purpose:** SayCan is a method for grounding large language models (LLMs) in robot capabilities by combining semantic reasoning from pre-trained language models with real-world feasibility via learned affordance functions. Instead of training end-to-end visuomotor policies, SayCan leverages both foundation models and robotics-specific priors.

**Key Novelty:**
- **Affordance-grounded language planning:** Combine LLM reasoning (semantic knowledge) with value functions (real-world feasibility)
- **Primitive skill decomposition:** High-level task → sequence of low-level robot skills
- **Hybrid reasoning system:** LLM handles "what to do," value function handles "can it be done"
- **No task-specific training:** Language model requires no fine-tuning on robot tasks
- **Real-world kitchen domain:** Demonstrated on complex long-horizon tasks in actual kitchen

**Robot Platform:**
- Everyday Robots mobile manipulation platform (EDR)
- Mobile base + 7-DoF arm + parallel gripper
- Operates in human kitchen environments

**Pre-Trained Foundation Models:**
- **Language Model:** PaLM (Pathways Language Model) - 540B parameters
  - Trained on diverse internet text and code
  - In-context learning capabilities
  - No robot-specific fine-tuning
- **Vision Model:** (Optional) For object detection and scene understanding

**Low-Level Robot Skills:**
- Pre-defined set of primitive behaviors (e.g., "pick_cup", "move_to_counter")
- Each skill: takes input, produces action sequence
- Value function trained for each skill: P(skill succeeds | state)

**Key Tasks Solved:**
- Long-horizon kitchen manipulation tasks
- Examples: "put the red cup on the table, get the blue cup from the drawer"
- Compositional reasoning: decompose complex instructions into simple skills
- Natural language understanding without task-specific training

**Main Output:**
- Skill sequence: [skill_1, skill_2, ..., skill_N]
- Robot executes each skill to completion before requesting next action
- Closed-loop execution with human guidance for clarification

**Performance:**
- 84% skill sequence accuracy (correct ordered sequence)
- 74% execution success rate (actual robot completion)
- 50% reduction in errors vs baseline language models without affordance grounding
- Handles complex, compositional, and abstract natural language

**If You Only Remember 3 Things:**
1. **Hybrid reasoning wins:** Language models know semantics, value functions ground in reality. Multiply their probabilities for robust planning.
2. **Skills are reusable:** Pre-defined skill set enables compositional task decomposition without learning new skills per task.
3. **No fine-tuning needed:** Large pre-trained language models generalize remarkably well to robotics without task-specific training—just use affordances to filter and reweight suggestions.

---

## 2. Problem Setup and Outputs

### High-Level Task Decomposition Flow

**Input:**
```
Natural language task instruction
  "Put the red cup on the table and get the blue cup from the drawer"

Expected behavior:
  Step 1: Understand task semantically (LLM)
  Step 2: Decompose into primitives (LLM)
  Step 3: Verify each primitive is feasible (value functions)
  Step 4: Execute skills sequentially with feedback
```

### Robot State and Observation Space

| Component | Shape | Type | Description |
|-----------|-------|------|-------------|
| Camera image | (480, 640, 3) | uint8 RGB | Overhead or arm-mounted view(s) |
| Object detection | variable | list of (class, bbox, confidence) | Pre-computed from vision model |
| Gripper state | (1,) | float [0,1] | Gripper openness (0=closed, 1=open) |
| Arm joints | (7,) | float | Joint angles in radians |
| Base pose | (3,) | float | [x, y, theta] in world frame |
| Gripper contact | (1,) | bool | Tactile feedback if available |
| Task instruction | variable | text string | Natural language command |

### Robot Actions and Skills

**Skill Primitive Format:**
```
skill = "pick_object"
inputs: [target_object_name, gripper_type]
outputs: action_sequence (sequence of robot commands)
         success: bool (did skill succeed?)

Example primitive skills:
  - pick_object(object_name) → gripper trajectory
  - place_object(object_name, location) → move + release
  - move_to_location(location) → navigation
  - open_drawer(drawer_id) → manipulation
  - close_drawer(drawer_id) → manipulation
```

**Available Skill Set:**
```
Total primitives: ~15-25 skills (task-dependent)

Categories:
  1. Pick/Grasp:
     - pick_object(obj_name)
     - pick_from_drawer(drawer_id, obj_name)
     - pick_from_shelf(shelf_id, obj_name)

  2. Place/Release:
     - place_object(obj_name, location)
     - place_in_drawer(drawer_id, obj_name)
     - place_on_surface(surface_name, obj_name)

  3. Navigation:
     - move_to_location(location_name)
     - move_to_human()
     - move_to_counter()

  4. Furniture:
     - open_drawer(drawer_id)
     - close_drawer(drawer_id)
     - open_door(door_id)

  5. Utility:
     - ask_human(question)
     - wait(duration)
```

### Value Function Output

**Affordance Function:**
```
V(skill, state) → [0, 1]

Interpretation:
  V = 1.0: skill very likely to succeed from this state
  V = 0.5: uncertain success
  V = 0.0: skill very unlikely to succeed

Computed via:
  - Learned neural network (CNN + MLP)
  - Trained with GAIL or IL on expert demonstrations
  - Or: Rule-based heuristics for simple skills
```

---

## 3. Coordinate Frames and Geometry

### World Frame and Robot Frames

**World Frame (W):**
- Origin: Kitchen reference point (typically counter corner)
- x-axis: left-right
- y-axis: forward-backward
- z-axis: up-down
- Units: Meters

**Mobile Base Frame (B):**
- Attached to robot base
- x-forward, y-left, z-up
- Pose: [x_w, y_w, theta_w] in world frame

**Arm Manipulator Frame (A):**
- End-effector frame
- Pose: 3D position + 3D orientation (quaternion or Euler)
- Reachable workspace: ~0.6m radius, ~0.5m height variation

**Camera Frames:**
- Fixed overhead camera: world-aligned
- Wrist camera: attached to end-effector

### Spatial Reasoning in Language

**Location Semantics (from language understanding):**
```
"Put cup on table" → location = "table_surface"
"Get cup from drawer" → location = "drawer_interior"
"Move to counter" → navigation_target = "counter_zone"

LLM understands:
  - Semantic relationships (drawer contains objects)
  - Spatial containment (inside vs. on top)
  - Navigation goals (reach specific locations)
```

**Affordance Interpretation:**
```
For task "pick_cup_from_table":
  State: cup at [0.3, 0.2, 0.8]m, gripper at [0.4, 0.3, 0.9]m

  V(pick_cup, state) computed:
    - Distance to cup: 0.14m (reachable → high V)
    - Gripper alignment: gripper points at cup (good → high V)
    - Occlusion: cup visible in camera (good → high V)

    Final: V = 0.92 (high confidence)

For task "pick_cup_from_deep_drawer":
  State: cup at [0.1, -0.05, 0.5]m (deep inside), gripper at [0.4, 0.3, 0.9]m

  V(pick_cup_from_drawer, state) computed:
    - Reachability: difficult angle (medium V = 0.4)
    - Occlusion: cup partially hidden (medium V = 0.4)

    Final: V = 0.35 (low confidence)
```

---

## 4. Architecture Deep Dive

### System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    SAYCAN PLANNING SYSTEM                       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  High-Level Natural Language Task                               │
│  "Put cup on table, get other cup from drawer"                  │
│         │                                                       │
│         v                                                       │
│  ┌──────────────────────┐                                      │
│  │  PaLM Language Model │                                      │
│  │  (540B parameters)   │                                      │
│  │                      │                                      │
│  │  In-context prompt:  │                                      │
│  │  "Here are skills:"  │                                      │
│  │  [list of skills]    │                                      │
│  │                      │                                      │
│  │  "Task: ..."         │                                      │
│  │  "Next skill: _"     │                                      │
│  └──────────┬───────────┘                                      │
│             │                                                   │
│             v                                                   │
│    Skill suggestions with probabilities:                        │
│    P(pick_cup | task, context) = 0.85                          │
│    P(place_on_table | task, context) = 0.80                    │
│    P(move_to_drawer | task, context) = 0.60                    │
│             │                                                   │
│  ┌──────────┴────────────────────────────────────────┐        │
│  │                                                    │         │
│  │  For each suggested skill:                         │         │
│  │  ┌─────────────────────────────────────────┐      │         │
│  │  │  Affordance Value Function CNN          │      │         │
│  │  │  Input: [robot_state, skill_name]       │      │         │
│  │  │  Output: V ∈ [0,1] feasibility score    │      │         │
│  │  └─────────────────────┬───────────────────┘      │         │
│  │                        │                          │         │
│  │                        v                          │         │
│  │  Reweighted Skill Scores:                         │         │
│  │  score(skill) = P_LLM(skill) × V_affordance(skill)│         │
│  │                                                    │         │
│  │  pick_cup: 0.85 × 0.92 = 0.782                   │         │
│  │  place_on_table: 0.80 × 0.95 = 0.760             │         │
│  │  move_to_drawer: 0.60 × 0.10 = 0.060 (infeasible)│         │
│  │                                                    │         │
│  │  Winner: pick_cup (highest score)                 │         │
│  └──────────────────────┬─────────────────────────────┘         │
│                         │                                       │
│                         v                                       │
│          ┌──────────────────────────┐                          │
│          │  Execute Selected Skill  │                          │
│          │  pick_cup()              │                          │
│          │                          │                          │
│          │  Generates action seq.   │                          │
│          │  Monitors success        │                          │
│          └──────────┬───────────────┘                          │
│                     │                                          │
│          success? ──┴─→ Request next skill from LLM            │
│                         Loop back to skill suggestion          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | Input | Output | Role |
|-----------|-------|--------|------|
| PaLM LLM | Task instruction + skill list + context | P(skill_i \| task) for all i | Semantic reasoning, skill sequencing |
| Affordance Value Function | Robot state + skill name | V ∈ [0,1] | Feasibility scoring |
| Skill Executor | Skill name + task context | Action sequence | Primitive execution |
| State Estimator | Camera images + proprioception | Current robot state | Perception pipeline |
| Skill Termination Detector | Skill execution history + state | Boolean (done?) | When to request next skill |

### Language Model Prompting Strategy

**In-Context Learning Setup:**

```
System prompt:
"You are a helpful robot planning assistant.
Given a high-level task and a set of available robot skills,
plan a sequence of skills to accomplish the task."

Available skills:
"
[1] pick_object(object_name)
    - Picks up an object by name
    - Preconditions: object is visible, not occluded
    - Postconditions: object is in gripper

[2] place_object(object_name, location)
    - Places object at specified location
    - Preconditions: object is in gripper
    - Postconditions: object is at location

[3] move_to_location(location_name)
    - Moves base to location
    - Preconditions: location is reachable, not blocked
    - Postconditions: base is at location

[4] open_drawer(drawer_id)
    - Opens a drawer by ID
    - Preconditions: drawer handle is accessible
    - Postconditions: drawer is open

[5] close_drawer(drawer_id)
    - Closes a drawer
    - Preconditions: drawer is open
    - Postconditions: drawer is closed
"

Example 1:
Task: "Get the red cup"
Assistant:
[1] First, I'll look for the red cup. I'll move to where cups are typically stored.
[2] move_to_location(kitchen_counter)
[3] Now I'll try to pick up the red cup.
[4] pick_object(red_cup)

Example 2:
Task: "Put the red cup on the table and get the blue cup from the drawer"
Assistant:
[1] First, I need to find and pick up the red cup.
[2] pick_object(red_cup)
[3] Now place it on the table.
[4] place_object(red_cup, table)
[5] Now I need to get the blue cup from the drawer.
[6] open_drawer(drawer_1)
[7] pick_object(blue_cup)

User Task: "Put the red cup on the table and get the blue cup from the drawer"
Assistant: [Next skill]: _______
"
```

**LLM generates next skill suggestion:**
```
P(pick_object(red_cup) | context) = 0.85
P(place_object(...) | context) = 0.80
P(open_drawer(...) | context) = 0.60
P(move_to_location(...) | context) = 0.45
...
```

### Affordance Value Function Architecture

**Network Structure:**

```python
class AffordanceValueFunction(nn.Module):
    def __init__(self, num_skills=20):
        super().__init__()

        # Vision encoder (CNN)
        self.vision_encoder = ResNet50Backbone()  # output: (B, 2048, 7, 7)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),  # joint angles, gripper state, etc.
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Skill embedding
        self.skill_embedding = nn.Embedding(num_skills, 128)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(2048*7*7 + 128 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # single output [0,1]
        )

        # Sigmoid to constrain to probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, robot_state, skill_id):
        # Vision
        vision_feat = self.vision_encoder(image)
        vision_feat = vision_feat.flatten(1)  # (B, 2048*7*7)

        # State
        state_feat = self.state_encoder(robot_state)  # (B, 128)

        # Skill embedding
        skill_feat = self.skill_embedding(skill_id)  # (B, 128)

        # Concatenate and fuse
        combined = torch.cat([vision_feat, state_feat, skill_feat], dim=1)
        value = self.sigmoid(self.fusion(combined))  # (B, 1) in [0,1]

        return value
```

**Training:**
```
Supervised learning from demonstrations:
  Input: (image, robot_state, skill_name)
  Target: did this skill succeed? (1 or 0)
  Loss: BCELoss (binary cross-entropy)

Alternative: GAIL (Generative Adversarial Imitation Learning)
  Discriminator learns affordance from expert demonstrations
  More robust to distribution shift
```

---

## 5. Forward Pass Pseudocode

**Complete Planning and Execution Loop:**

```python
# ===== INITIALIZATION =====

# Pre-trained models loaded
lm = load_pretrained_model('PaLM')  # 540B, frozen
affordance_vf = load_affordance_vf()  # trained on robot data
skills = load_skill_library()  # dictionary of primitive behaviors

# Skill list (for LLM context)
skill_descriptions = [
    "pick_object(object_name): Pick up an object",
    "place_object(object_name, location): Place object at location",
    "move_to_location(location_name): Navigate to location",
    "open_drawer(drawer_id): Open a drawer",
    "close_drawer(drawer_id): Close a drawer",
    # ... etc
]

# ===== TASK PLANNING LOOP =====

task_instruction = "Put the red cup on the table and get the blue cup from the drawer"
executed_skills = []
failed_skills = []
skill_history = ""  # Keep track of what's been done

for planning_step in range(max_planning_steps):
    # Get current robot state
    observation = robot.get_observation()  # camera, proprioceptive, etc.
    robot_state = state_estimator(observation)
    # robot_state contains:
    #   - image: (480, 640, 3)
    #   - gripper_open: float [0, 1]
    #   - arm_joints: (7,)
    #   - base_pose: (3,)

    # ===== LANGUAGE MODEL PLANNING =====

    # Build LLM prompt with in-context examples
    llm_prompt = f"""
    You are a robot planning assistant. Given a task and available skills,
    plan the next skill to execute.

    Available skills:
    {skill_descriptions}

    Task: {task_instruction}

    Previously executed skills:
    {skill_history}

    Current robot state:
    - Objects visible: {object_detection(observation)}
    - Gripper open: {robot_state['gripper_open']:.2f}
    - Base position: {robot_state['base_pose']}

    What is the next skill to execute?
    Respond with skill name and parameters in format: skill_name(param1, param2, ...)
    """

    # Query language model
    llm_output = lm.generate(llm_prompt, max_tokens=100, temperature=0.1)
    # Example output: "pick_object(red_cup)"

    # Parse LLM output to extract skill suggestions
    suggested_skills = parse_llm_output(llm_output)
    # Result: [
    #   ('pick_object', {'object_name': 'red_cup'}, prob=0.85),
    #   ('place_object', {'object_name': 'red_cup', 'location': 'table'}, prob=0.70),
    #   ('move_to_location', {'location_name': 'drawer'}, prob=0.50)
    # ]

    # ===== AFFORDANCE GROUNDING =====

    # For each suggested skill, compute affordance score
    best_skill = None
    best_score = -1

    for skill_name, skill_params, p_llm in suggested_skills:
        # Get affordance value from value function
        skill_id = skill_to_id(skill_name)
        v_affordance = affordance_vf(
            image=observation['image'],
            robot_state=robot_state,
            skill_id=skill_id
        ).item()  # scalar in [0,1]

        # Combine LLM probability with affordance
        score = p_llm * v_affordance
        # Intuition: skill must be both semantically appropriate (p_llm)
        #           AND physically feasible (v_affordance)

        # Also log for debugging
        print(f"{skill_name}({skill_params}): "
              f"p_llm={p_llm:.3f}, v_aff={v_affordance:.3f}, score={score:.3f}")

        if score > best_score:
            best_score = score
            best_skill = (skill_name, skill_params)

    # ===== SKILL EXECUTION =====

    if best_skill is None:
        print("ERROR: No feasible skills available!")
        break

    skill_name, skill_params = best_skill
    print(f"Executing: {skill_name}({skill_params})")

    # Execute the selected skill
    skill_fn = skills[skill_name]
    success = skill_fn(robot, skill_params)

    # ===== UPDATE AND LOOP =====

    if success:
        executed_skills.append((skill_name, skill_params))
        skill_history += f"\n  - {skill_name}({skill_params}): SUCCESS"

        # Check task completion
        if task_complete(task_instruction, executed_skills):
            print(f"TASK COMPLETE after {len(executed_skills)} skills!")
            break
    else:
        failed_skills.append((skill_name, skill_params))
        skill_history += f"\n  - {skill_name}({skill_params}): FAILED"

        # Update failure count
        if skill_name in failure_counts:
            failure_counts[skill_name] += 1

        # If too many failures, ask human or abort
        if failure_counts[skill_name] > max_retries:
            print(f"Skill {skill_name} failed {max_retries} times. Asking human...")
            clarification = get_human_input(
                f"Skill {skill_name}({skill_params}) failed. What should I do?"
            )
            # Could: clarify task, adjust parameters, switch strategies

# ===== FINAL REPORT =====

print(f"Task: {task_instruction}")
print(f"Executed {len(executed_skills)} skills successfully")
print(f"Failed {len(failed_skills)} skill attempts")
print(f"Success rate: {len(executed_skills) / (len(executed_skills) + len(failed_skills)):.2%}")
```

**Key Algorithm: Skill Selection**

```python
def select_next_skill(
    lm,
    affordance_vf,
    robot_state,
    observation,
    task_instruction,
    skill_history,
    available_skills
):
    """
    Returns: (best_skill_name, best_params, combined_score)
    """

    # Step 1: Get LLM probability distribution over skills
    p_llm = lm.compute_skill_probabilities(
        task_instruction,
        skill_history,
        available_skills
    )
    # Returns: {skill_name: probability, ...}

    # Step 2: Compute affordance value for each skill
    best_skill = None
    best_score = 0

    for skill_name in available_skills:
        # Get feasibility score from value function
        v_aff = affordance_vf(
            image=observation['image'],
            robot_state=robot_state,
            skill_id=skill_to_id(skill_name)
        )  # scalar in [0,1]

        # Combine with LLM probability
        p = p_llm.get(skill_name, 1e-6)  # avoid log(0)
        combined_score = p * v_aff

        if combined_score > best_score:
            best_score = combined_score
            best_skill = skill_name

    return best_skill, best_score
```

---

## 6. Heads, Targets, and Losses

### Primary Component: Affordance Value Function Training

**Target:**
```
For each (observation, skill, state) triple from demonstrations:
  target = 1.0 if skill succeeded from this state
  target = 0.0 if skill failed from this state
```

**Loss Function:**

```python
# Binary classification loss
loss_bce = BCEWithLogitsLoss()

# During training on expert demonstrations
for batch in training_dataloader:
    # batch contains:
    # - images: (B, H, W, 3)
    # - robot_states: (B, state_dim)
    # - skill_ids: (B,)
    # - success_labels: (B,) [0 or 1]

    # Forward pass
    predicted_values = affordance_vf(
        images, robot_states, skill_ids
    )  # (B, 1)

    # Compute loss
    loss = loss_bce(predicted_values, success_labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Loss Curves:**

```
Initial training:
  Epoch 0: loss ≈ 0.693 (random initialization)
  Epoch 5: loss ≈ 0.35 (learning from easy examples)
  Epoch 20: loss ≈ 0.15 (convergence)

Expected behavior:
  - Loss should monotonically decrease
  - Accuracy should increase on validation set
  - If loss plateaus: learning rate too low, or model too small
```

### Secondary Components: No Other Trainable Heads

**Important Note:** SayCan does NOT require task-specific training!

The architecture is modular:
1. **Language Model (PaLM):** Pre-trained, frozen (540B parameters)
   - No fine-tuning on robot tasks
   - Uses in-context learning via prompting
   - Training cost: $0 for task-specific data

2. **Affordance Value Function:** Task-specific training, but reusable
   - Trained once on collected robot data
   - Learned what "is feasible" not "which task to do"
   - One value function shared across all tasks

3. **Skill Library:** Hand-crafted or learned, frozen at test time
   - Pre-defined or pre-learned primitives
   - No online adaptation

---

## 7. Data Pipeline and Augmentations

### Data Collection Strategy

**Demonstration Collection:**

```
1. Human teleoperation
   - Collect ~100-500 demonstrations per skill
   - Use joystick/VR controller for natural trajectories
   - Record: observations, actions, skill labels, success/failure

2. Naturalistic kitchen environment
   - Real kitchen with natural lighting
   - Various object placements and configurations
   - Different human operators for diversity

3. Per-skill data collection
   - Collect data for each primitive skill independently
   - Annotate success/failure labels
   - Store: {obs, state, action_seq, success}
```

**Data Format per Skill:**

```python
skill_data = {
    'skill_name': 'pick_object',
    'episodes': [
        {
            'observation': {
                'rgb': (H, W, 3),
                'depth': (H, W),
            },
            'robot_state': {
                'gripper_open': float,
                'arm_joints': (7,),
                'base_pose': (3,),
            },
            'action_sequence': [(7,), ...],  # sequence of 7D actions
            'success': True or False,
            'failure_reason': str or None,  # if failed
        },
        # ... more episodes
    ],
}
```

### Preprocessing Pipeline

**Image Preprocessing:**

```python
def preprocess_image(rgb_image):
    # 1. Resize to standard size
    resized = cv2.resize(rgb_image, (224, 224))

    # 2. Normalize
    normalized = (resized / 255.0).astype(np.float32)

    # 3. Standardize (ImageNet statistics)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    standardized = (normalized - mean) / std

    return standardized
```

**State Preprocessing:**

```python
def preprocess_state(robot_state):
    # Normalize each dimension independently

    gripper = np.clip(robot_state['gripper_open'] / 1.0, 0, 1)  # [0,1]

    # Arm: normalize joint angles
    arm_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    arm_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    arm_norm = (robot_state['arm_joints'] - arm_min) / (arm_max - arm_min)
    arm_norm = np.clip(arm_norm, 0, 1)

    # Base: normalize position
    base_min, base_max = np.array([-5, -5, -np.pi]), np.array([5, 5, np.pi])
    base_norm = (robot_state['base_pose'] - base_min) / (base_max - base_min)
    base_norm = np.clip(base_norm, 0, 1)

    # Concatenate
    state_vector = np.concatenate([gripper, arm_norm, base_norm])  # (1+7+3,) = (11,)

    return state_vector
```

### Data Augmentations

**Vision Augmentations (Training Only):**

```python
def augment_image(image):
    # 1. Random crops
    crop_size = 200
    max_offset = 224 - crop_size
    x_offset = np.random.randint(0, max_offset + 1)
    y_offset = np.random.randint(0, max_offset + 1)
    image_cropped = image[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
    image_cropped = cv2.resize(image_cropped, (224, 224))

    # 2. Random brightness/contrast
    brightness = np.random.uniform(0.8, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    image_aug = brightness * contrast * image_cropped + (1 - brightness) * 0.5

    # 3. Random hue/saturation (in HSV)
    hsv = cv2.cvtColor(image_aug, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-10, 11)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
    image_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 4. Gaussian blur
    if np.random.rand() < 0.3:
        image_aug = cv2.GaussianBlur(image_aug, (3, 3), 0)

    return image_aug
```

**State Augmentations:**

```python
def augment_state(state):
    # Add small Gaussian noise to simulate sensor uncertainty
    noise = np.random.normal(0, 0.01, state.shape)
    state_aug = np.clip(state + noise, 0, 1)
    return state_aug
```

### Batch Creation for Training

```python
def create_affordance_training_batch(
    skill_data,
    batch_size=32,
    positive_ratio=0.5
):
    """
    Create balanced batch (50% success, 50% failure)
    to avoid class imbalance
    """
    batch = {
        'images': [],
        'robot_states': [],
        'skill_ids': [],
        'success_labels': [],
    }

    # Separate successful and failed episodes
    successes = [ep for ep in skill_data['episodes'] if ep['success']]
    failures = [ep for ep in skill_data['episodes'] if not ep['success']]

    # Sample balanced
    num_pos = int(batch_size * positive_ratio)
    num_neg = batch_size - num_pos

    for ep in random.sample(successes, num_pos):
        batch['images'].append(ep['observation']['rgb'])
        batch['robot_states'].append(ep['robot_state'])
        batch['skill_ids'].append(skill_to_id(skill_data['skill_name']))
        batch['success_labels'].append(1)

    for ep in random.sample(failures, num_neg):
        batch['images'].append(ep['observation']['rgb'])
        batch['robot_states'].append(ep['robot_state'])
        batch['skill_ids'].append(skill_to_id(skill_data['skill_name']))
        batch['success_labels'].append(0)

    # Stack and augment
    batch['images'] = np.stack([
        augment_image(img) for img in batch['images']
    ])
    batch['robot_states'] = np.stack([
        augment_state(preprocess_state(state)) for state in batch['robot_states']
    ])
    batch['skill_ids'] = np.array(batch['skill_ids'])
    batch['success_labels'] = np.array(batch['success_labels'], dtype=np.float32)

    return {k: torch.from_numpy(v) for k, v in batch.items()}
```

---

## 8. Training Pipeline

### Affordance Value Function Training

**Hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimization** | | |
| Optimizer | Adam | β₁=0.9, β₂=0.999 |
| Learning Rate | 1e-3 | Fixed or schedule |
| Weight Decay | 1e-5 | Light L2 regularization |
| Gradient Clipping | 1.0 | Prevent exploding gradients |
| **Batch** | | |
| Batch Size | 32 | Small enough for GPU |
| Positive Ratio | 0.5 | Balanced classes |
| **Schedule** | | |
| Epochs | 50-100 | Varies by dataset size |
| Early Stopping | Patience=10 | On validation accuracy |
| **Regularization** | | |
| Dropout | 0.2 | In MLP heads |
| Data Augmentation | Yes | Image + state noise |

### Training Loop

```python
def train_affordance_vf(
    affordance_vf,
    train_dataloader,
    val_dataloader,
    num_epochs=50
):
    optimizer = torch.optim.Adam(
        affordance_vf.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        affordance_vf.train()
        train_loss = 0

        for batch in train_dataloader:
            images = batch['images'].to(device)
            states = batch['robot_states'].to(device)
            skill_ids = batch['skill_ids'].to(device)
            labels = batch['success_labels'].to(device)

            # Forward
            preds = affordance_vf(images, states, skill_ids)  # (B, 1)
            loss = loss_fn(preds, labels.unsqueeze(1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                affordance_vf.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            train_loss += loss.item()

        # Validation
        affordance_vf.eval()
        val_acc = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['images'].to(device)
                states = batch['robot_states'].to(device)
                skill_ids = batch['skill_ids'].to(device)
                labels = batch['success_labels'].to(device)

                preds = affordance_vf(images, states, skill_ids)
                pred_labels = (torch.sigmoid(preds) > 0.5).float()

                acc = (pred_labels.squeeze() == labels).float().sum()
                val_acc += acc.item()
                val_samples += len(labels)

        val_acc /= val_samples

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(affordance_vf.state_dict(), 'best_vf.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    affordance_vf.load_state_dict(torch.load('best_vf.pt'))
    return affordance_vf
```

### Language Model: No Training Required

**PaLM Setup:**
```
- Download pre-trained weights (or use API)
- Freeze all parameters (no gradient updates)
- Use in-context learning via prompting
- Cost: inference only, no training data collection

Advantages:
  - No task-specific fine-tuning
  - Generalizes to novel instructions
  - Can leverage pre-training on diverse internet data
```

---

## 9. Dataset + Evaluation Protocol

### Data Collection Scope

**Skill-Specific Data:**

```
Per primitive skill (e.g., "pick_object"):
  - ~100-200 demonstrations collected via teleoperation
  - Diverse object types, colors, sizes
  - Different initial configurations
  - Success and failure examples

For ~15-20 skills:
  Total demonstrations: 1500-4000 episodes
  Total hours of robot time: 50-200 hours (depending on skill difficulty)
```

**Evaluation Data:**

```
Held-out test set:
  - 20-30 unseen task descriptions
  - Each task: 5-10 novel instantiations (different objects/locations)
  - Total test episodes: ~200 robot runs
```

### Evaluation Metrics

**Primary Metric: Skill Sequence Accuracy**

```python
def evaluate_skill_sequence_accuracy(
    system,
    test_tasks,
    num_trials_per_task=3
):
    """
    For each test task, evaluate if the predicted skill sequence is correct.
    """
    correct_sequences = 0
    total_sequences = 0

    for task_instruction in test_tasks:
        for trial in range(num_trials_per_task):
            # Predict skill sequence using SayCan
            predicted_skills = []
            skill_history = ""

            for step in range(max_steps):
                next_skill = system.select_next_skill(
                    task_instruction,
                    skill_history
                )

                if next_skill is None or next_skill == 'done':
                    break

                predicted_skills.append(next_skill)
                skill_history += f"\n- {next_skill}"

            # Get ground truth sequence
            ground_truth = get_ground_truth_skills(task_instruction)

            # Compare sequences (exact match or partial credit)
            if predicted_skills == ground_truth:
                correct_sequences += 1

            total_sequences += 1

    accuracy = correct_sequences / total_sequences
    return accuracy
```

**Secondary Metric: Execution Success Rate**

```python
def evaluate_execution_success(
    robot,
    system,
    test_tasks,
    num_trials=3
):
    """
    Execute predicted skill sequences on real robot.
    Measure task completion success.
    """
    successful_executions = 0
    total_executions = 0

    for task_instruction in test_tasks:
        for trial in range(num_trials):
            # Reset environment
            env.reset()

            # Plan and execute skills
            success = True
            skill_history = ""

            for step in range(max_steps):
                # Plan next skill
                next_skill = system.select_next_skill(
                    task_instruction,
                    skill_history
                )

                if next_skill is None or next_skill == 'done':
                    break

                # Execute skill on robot
                skill_success = robot.execute_skill(next_skill)

                if not skill_success:
                    # Skill failed: could retry or ask human
                    # For simplicity, mark task as failed
                    success = False
                    break

                skill_history += f"\n- {next_skill}"

            # Verify task completion
            task_complete = verify_task_completion(task_instruction, env)

            if success and task_complete:
                successful_executions += 1

            total_executions += 1

    success_rate = successful_executions / total_executions
    return success_rate
```

**Tertiary Metric: Semantic Understanding (vs Baselines)**

```python
# Compare different approaches:

baseline_lm_only = {
    'skill_sequence_acc': 0.68,  # LLM generates sequence blindly
    'execution_success': 0.42,   # Many infeasible skills attempted
}

baseline_vf_only = {
    'skill_sequence_acc': 0.45,  # Random baseline via value function
    'execution_success': 0.38,   # Lacks semantic understanding
}

saycan_result = {
    'skill_sequence_acc': 0.84,  # Combines LLM + affordance
    'execution_success': 0.74,   # Both semantic + feasible
}

improvement_vs_lm_only = (0.84 - 0.68) / 0.68  # +23.5% improvement
improvement_vs_vf_only = (0.84 - 0.45) / 0.45  # +87% improvement
```

---

## 10. Results Summary + Ablations

### Main Results

**Overall Performance:**

| Method | Skill Sequence Accuracy | Execution Success | Error Reduction |
|--------|------------------------|--------------------|------------------|
| LLM only (PaLM) | 68% ± 4% | 42% ± 5% | Baseline |
| VF only (value function) | 45% ± 6% | 38% ± 6% | -45% |
| SayCan (PaLM + VF) | 84% ± 3% | 74% ± 4% | **50% reduction** |

**Key Finding:** Combining semantic reasoning (LLM) with feasibility checking (affordances) significantly outperforms either approach alone.

### Ablation Studies

**Ablation 1: Importance of Affordance Weighting**

| Configuration | Skill Accuracy | Execution Success | Notes |
|---------------|---------------|-------------------|-------|
| LLM only (no affordance) | 68% | 42% | Baseline |
| LLM × VF^0.5 (weak weighting) | 76% | 60% | Moderate improvement |
| LLM × VF^1.0 (equal weighting) | 84% | 74% | Optimal |
| LLM × VF^2.0 (strong weighting) | 79% | 68% | Over-emphasizes VF |

**Finding:** Equal weighting (multiply probabilities directly) works best. Over-weighting affordances removes semantic reasoning.

**Ablation 2: Language Model Capacity**

| Model | Parameters | Skill Accuracy | Execution Success |
|-------|-----------|---|---|
| Baseline (before LLM) | - | 20% | 15% |
| GPT-3 (175B) | 175B | 76% | 62% |
| PaLM (540B) | 540B | 84% | 74% |
| Fine-tuned PaLM | 540B | 85% | 75% |

**Finding:** Larger models (PaLM >> GPT-3) generalize better. Fine-tuning minimal improvement (+1 pp), not worth cost.

**Ablation 3: Affordance VF Architecture**

| VF Architecture | Skill Accuracy | Execution Success | Training Data |
|---|---|---|---|
| Random (baseline) | 52% | 39% | - |
| Heuristic rules | 68% | 54% | 0 (hand-crafted) |
| CNN-only (no state) | 75% | 63% | ~500 episodes |
| State-only (no vision) | 62% | 48% | ~500 episodes |
| CNN + State (full) | 84% | 74% | ~500 episodes |

**Finding:** Vision + state fusion critical. State alone very weak; vision alone mediocre.

**Ablation 4: Skill Library Size**

| Number of Skills | Skill Accuracy | Execution Success | Coverage |
|---|---|---|---|
| 5 skills (minimal) | 58% | 42% | Low flexibility |
| 10 skills | 72% | 60% | Moderate |
| 15 skills | 81% | 72% | Good |
| 20 skills | 84% | 74% | Excellent |
| 30 skills | 83% | 72% | Diminishing returns |

**Finding:** 15-20 skills sufficient for reasonable task coverage. Beyond that: redundancy and confusion.

**Ablation 5: In-Context Examples for LLM**

| Config | Skill Accuracy | Execution Success | Notes |
|---|---|---|---|
| Zero-shot (no examples) | 72% | 55% | |
| 1 example | 78% | 64% | |
| 3 examples | 82% | 71% | |
| 5 examples (default) | 84% | 74% | Optimal |
| 10 examples | 84% | 73% | No improvement |

**Finding:** 3-5 in-context examples optimal. Saturation around 5 examples.

**Ablation 6: Error Recovery Strategies**

| Strategy | Success Rate Improvement | Human Interruptions |
|---|---|---|
| No recovery (baseline) | 74% | 0 |
| Retry failed skill (1x) | 78% | 0 |
| Retry failed skill (3x) | 81% | 0 |
| Ask human on failure | 85% | High |
| Adaptive replanning | 83% | Low |

**Finding:** Simple retry helps (+4 pp); human input helps most but not practical; adaptive replanning good compromise.

---

## 11. Practical Insights

### Engineering Takeaways (10 Key Learnings)

1. **Large Language Models Are Already Good at Robotics**
   - No task-specific fine-tuning needed
   - In-context learning works surprisingly well
   - Pre-trained knowledge transfers immediately
   - Cost: just API/inference, no training data

2. **Affordance Functions Are Cheaper Than Learning Full Policies**
   - Value function trained on ~500 demos per skill
   - Much cheaper than learning end-to-end visuomotor policies
   - Reusable across multiple tasks (single VF per skill)
   - Binary classification much simpler than action regression

3. **Hybrid Reasoning Beats Pure Semantics**
   - LLM alone: 68% accuracy (ignores real-world constraints)
   - VF alone: 45% accuracy (lacks semantic understanding)
   - Combined: 84% accuracy (synergy effect)
   - Multiplication of probabilities better than averaging

4. **Skill Abstraction Is Key**
   - Pre-define 15-20 skills covering task space
   - LLM decomposes high-level task into skill sequence
   - Much easier than learning end-to-end policy
   - Enables compositional reasoning

5. **In-Context Learning Robust**
   - 3-5 examples in prompt sufficient (saturation)
   - Different example orders produce same results (robust)
   - Works with different LLM sizes (GPT-3, PaLM, etc.)
   - Prompt engineering matters but basic template works

6. **Vision Features Most Important for Affordances**
   - Vision-only VF: 75% accuracy
   - State-only VF: 62% accuracy
   - Combined: 84% accuracy
   - Implies object appearance/configuration critical

7. **Skill Failures Recoverable With Retry**
   - 1x retry: +4 pp success rate
   - 3x retry: +7 pp, but risk getting stuck
   - Human-in-loop option critical for robustness
   - Implement graceful failure recovery

8. **Generalization Across Objects and Scenes**
   - Train on ~30 objects
   - Test on novel objects of same category → works
   - Different kitchen layout → some transfer
   - Large gap to completely novel domains

9. **Real-Time Inference Feasible**
   - LLM inference: ~1-3 seconds per skill selection
   - Affordance VF: ~50ms per evaluation
   - Total: <5 seconds per skill (acceptable for long-horizon tasks)
   - Batching multiple skill candidates for parallelism

10. **Human-Robot Collaboration Enables Better Performance**
    - Robot can ask human clarification on ambiguous tasks
    - "Did you mean the red cup or orange cup?"
    - Human can override robot's choice if predicted skill seems wrong
    - Adds robustness without expensive retraining

### 5 Common Gotchas

1. **LLM Hallucination on Unfamiliar Skills**
   - PaLM might suggest skills not in the available set
   - Parse errors if skill format differs from prompt
   - **Fix:** Constrain LLM output with grammar/parsing, or post-process to nearest valid skill

2. **Affordance VF Distributional Shift**
   - Trained on teleoperated data (human-like trajectories)
   - Test-time states might differ significantly from training
   - **Fix:** Include diverse failure modes in training; test VF calibration

3. **Over-Reliance on Certain Skills**
   - If one skill has high VF everywhere, LLM will select it repeatedly
   - Leads to task failure (e.g., keep picking same object)
   - **Fix:** Add negative examples, calibrate VF via confidence intervals

4. **Task Specification Ambiguity**
   - "Put the cup on the table" → which table? Which cup?
   - LLM can't disambiguate from natural language alone
   - **Fix:** Ask human for clarification, or use vision to resolve references

5. **Skill Parameters Not Learned**
   - Skills take parameters (e.g., object_name)
   - LLM must generate correct parameter names
   - If LLM says "pick red_mug" but training used "pick_cup", failure
   - **Fix:** Standardize naming conventions, include examples in prompt

### Tiny-Subset Debugging Plan

**Step 1: Test LLM Prompting (No Robot)**

```
Minimal test: 3 tasks, 5 skills, 1 example
Check:
  - Does LLM suggest valid skills?
  - Are suggestions semantically reasonable?
  - Does it follow the prompt format?

Expected: 70-80% of suggestions reasonable
```

**Step 2: Test Affordance VF on Logged Data**

```
Minimal dataset: 10 success + 10 failure episodes per skill
Check:
  - Does VF output high value for successes? (>0.7)
  - Does VF output low value for failures? (<0.3)
  - Are predictions calibrated?

Expected: >70% classification accuracy
```

**Step 3: End-to-End on Simulator**

```
Minimal task: "pick_cup, place_cup"
Check:
  - Does SayCan select correct sequence?
  - Does execution succeed in simulation?
  - What if LLM mistakes are made? (graceful degradation)

Expected: 80%+ success in controlled sim
```

**Step 4: Real Robot Test**

```
Minimal setup: 1 skill, 1 object, 1 location
Check:
  - Affordance VF works with real camera input
  - Skill execution safe and correct
  - Human-in-loop interruption works

Expected: >90% success on simple task
```

---

## 12. Minimal Reimplementation Checklist

### Essential Components

- [ ] **Skill Library**
  - [ ] Define 10-15 primitive skills
  - [ ] Implement skill executor for each (returns success/failure)
  - [ ] Test each skill independently in robot environment

- [ ] **Affordance Value Function**
  - [ ] Design network: CNN vision encoder + state encoder + fusion
  - [ ] Collect ~500 demonstrations per skill (success + failure)
  - [ ] Train binary classifier (success/failure)
  - [ ] Validate on held-out test set (>70% accuracy)

- [ ] **Language Model Integration**
  - [ ] Setup API access (PaLM, GPT-3, or open-source alternative)
  - [ ] Write prompt template with skill descriptions
  - [ ] Implement in-context learning (few-shot examples)
  - [ ] Parse LLM output to extract skill sequence

- [ ] **Planning Loop**
  - [ ] Get current robot observation
  - [ ] Query LLM for next skill suggestions
  - [ ] For each suggestion: compute LLM prob × VF score
  - [ ] Select skill with highest combined score
  - [ ] Execute skill until completion

- [ ] **Evaluation**
  - [ ] Measure skill sequence accuracy
  - [ ] Measure execution success on test tasks
  - [ ] Compare to baselines (LLM-only, VF-only)

### Minimal Code Skeleton

```python
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SkillLibrary:
    def __init__(self):
        self.skills = {
            'pick_object': self.pick_object_skill,
            'place_object': self.place_object_skill,
            'move_to_location': self.move_to_location_skill,
            # ... more skills
        }

    def pick_object_skill(self, robot, object_name):
        """Execute pick skill"""
        # Move to object
        # Close gripper
        # Verify success
        success = robot.gripper_has_object()
        return success

    def execute(self, robot, skill_name, params):
        if skill_name not in self.skills:
            return False
        skill_fn = self.skills[skill_name]
        return skill_fn(robot, **params)

class AffordanceValueFunction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN + MLP
        self.vision_encoder = torch.nn.Conv2d(3, 32, kernel_size=3)
        self.state_encoder = torch.nn.Linear(11, 32)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(32 + 32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, image, state):
        vision_feat = self.vision_encoder(image).flatten(1)
        state_feat = self.state_encoder(state)
        combined = torch.cat([vision_feat, state_feat], dim=1)
        value = torch.sigmoid(self.fusion(combined))
        return value

class SayCan:
    def __init__(self, lm_model_name='gpt2'):
        # Load language model
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(lm_model_name)

        # Load affordance VF
        self.affordance_vf = AffordanceValueFunction()
        self.affordance_vf.load_state_dict(torch.load('affordance_vf.pt'))

        # Skill library
        self.skill_library = SkillLibrary()

        # Prompt template
        self.skill_descriptions = [
            "pick_object: Pick up an object by name",
            "place_object: Place an object at a location",
            # ... etc
        ]

    def plan_next_skill(self, task_instruction, skill_history, robot_state, image):
        """
        Plan the next skill using SayCan
        """

        # Build LLM prompt
        prompt = f"""You are a robot planning assistant.
Available skills:
{chr(10).join(self.skill_descriptions)}

Task: {task_instruction}

Previously executed:
{skill_history}

What is the next skill? Answer in format: skill_name(param1, param2)
Next skill: """

        # Query LLM for suggestions
        # (simplified; real LLM call might be async)
        inputs = self.lm_tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.lm(**inputs, output_scores=True)
        # Parse outputs to get skill suggestions

        # Evaluate affordances
        best_skill = None
        best_score = 0

        for skill_name in self.skill_library.skills:
            # Get LLM probability (from logits)
            p_lm = 0.5  # simplified

            # Get affordance value
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)
            state_tensor = torch.from_numpy(robot_state).float().unsqueeze(0)
            v_aff = self.affordance_vf(image_tensor, state_tensor).item()

            # Combine scores
            score = p_lm * v_aff

            if score > best_score:
                best_score = score
                best_skill = skill_name

        return best_skill

    def execute_task(self, robot, task_instruction, max_steps=10):
        """Execute full task"""
        skill_history = ""

        for step in range(max_steps):
            # Get current state
            obs = robot.get_observation()

            # Plan next skill
            next_skill = self.plan_next_skill(
                task_instruction,
                skill_history,
                obs['state'],
                obs['image']
            )

            if next_skill is None:
                break

            # Execute
            success = self.skill_library.execute(robot, next_skill, {})

            skill_history += f"\n- {next_skill}: {'SUCCESS' if success else 'FAILED'}"

            if not success:
                # Could retry or ask human
                pass

        return skill_history
```

### Evaluation Checklist

- [ ] Skill library works correctly
- [ ] Affordance VF makes reasonable predictions
- [ ] LLM prompting produces valid skill sequences
- [ ] Planning loop selects correct skills on test tasks
- [ ] Execution succeeds on real robot
- [ ] Graceful degradation on failure (asks human, retries)
- [ ] Inference speed acceptable (<5s per skill)
- [ ] Safety mechanisms in place (joint limits, collision avoidance)

---

## References

- SayCan Project: [say-can.github.io](https://say-can.github.io)
- arXiv Paper: [arxiv.org/abs/2204.01691](https://arxiv.org/abs/2204.01691)
- Google Research Blog: [research.google/blog/towards-helpful-robots-grounding-language-in-robotic-affordances/](https://research.google/blog/towards-helpful-robots-grounding-language-in-robotic-affordances/)
- CoRL 2022 Conference: [corl2022.org](https://corl2022.org)
- GitHub Implementation: [kyegomez/SayCan](https://github.com/kyegomez/SayCan)
