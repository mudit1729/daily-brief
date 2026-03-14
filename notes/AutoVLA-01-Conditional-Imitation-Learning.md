# End-to-End Driving via Conditional Imitation Learning

> **Authors:** Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, Alexey Dosovitskiy
> **Venue:** ICRA 2018 | **arXiv:** [1710.02410](https://arxiv.org/abs/1710.02410)
> **Citations:** ~1651 | **Impact:** ★★★★★ (foundational)
> **PDF:** [vladlen.info](https://vladlen.info/papers/conditional-imitation.pdf)

---

## 1. Problem & Motivation

End-to-end imitation learning policies map sensory input directly to control outputs, but are **uncontrollable** when perception alone doesn't determine the correct action. At intersections, the same visual scene can correspond to turning left, right, or going straight—causing oscillations and wrong turns. The policy has no mechanism to receive high-level intent.

## 2. Core Idea

**Condition** the imitation learning policy on a **high-level command** input (turn left / turn right / go straight / follow lane), making the learned policy responsive to navigation intent. The system acts as a "chauffeur" that handles low-level control while obeying turn-by-turn guidance.

## 3. Architecture & Method

```
Input: Front camera image + discrete command (4 options)
  │
  ├─ CNN Encoder (ResNet-style)
  │     │
  │     ▼
  │   Feature vector
  │     │
  └─ Command Conditioning ──┐
                             ▼
                    Branch Selection
                    ┌──────────────────┐
                    │ Branch: Left     │
                    │ Branch: Right    │
                    │ Branch: Straight │
                    │ Branch: Follow   │
                    └──────────────────┘
                             │
                             ▼
                    Controls (steer, gas, brake)
```

**Two conditioning architectures tested:**
- **Command input:** concatenate command embedding with CNN features → single FC head
- **Branched architecture:** separate FC heads per command, gated by command selection (works better)

**Training:** Supervised imitation learning on expert demonstrations. Data augmentation with noise injection to handle distribution shift.

## 4. Key Results

- **CARLA simulation:** Conditional model significantly outperforms unconditional end-to-end baselines at intersections
- **Real-world 1/5-scale truck:** Successfully trained to drive in residential environments following commands
- **Ablation:** Branched architecture > command-input architecture for handling multiple intents

## 5. Why Seminal

1. **First controllable end-to-end driving** via intent commands—widely reused paradigm
2. **Proto-language milestone:** discrete commands are a simplified "language" that resolves perceptual ambiguity
3. **Durable design pattern:** later VLA systems retain the separation between (a) language/intent interface and (b) low-level control generation
4. **Both sim + real demos:** validated in CARLA and on physical hardware

## 6. Limitations

- "Language" is a **predefined 4-word vocabulary**, not free-form natural language
- Inherits imitation-learning **distribution shift** and generalization issues
- Physical demo is limited to 1/5-scale, not full-sized autonomous driving
- No reasoning or explanation capability

## 7. Key Equations

**Conditional policy:**
$$\pi(a | o, c) = f_c(g(o))$$
where $o$ = observation (image), $c$ = command, $g$ = CNN encoder, $f_c$ = command-specific head

**Branched loss:**
$$\mathcal{L} = \sum_i \mathbb{1}[c_i = c] \| f_c(g(o_i)) - a_i^* \|^2$$

## 8. Connections to Later Work

- **→ LMDrive:** Upgrades discrete commands to natural language instructions
- **→ SimLingo:** Retains camera→control pattern but adds language-action alignment
- **→ DriveGPT4:** Same vision→control paradigm but adds language explanation output
- **→ ORION:** Keeps command-conditioned spirit but uses LLM reasoning tokens

## 9. Interview Quick-Hits

**Q: Why can't vanilla end-to-end imitation learning handle intersections?**
A: The policy sees identical visual inputs for left/right/straight turns—it's a multimodal output problem. Without intent conditioning, the model averages over all possible actions, causing oscillation or wrong turns.

**Q: What's the key insight of conditional imitation learning?**
A: Add a high-level command input to resolve the one-to-many mapping from perception to action. The "branched" architecture (separate heads per command) works best because each branch specializes in one type of maneuver.

**Q: How does this relate to modern VLA driving?**
A: It's the proto-VLA: replace the discrete command vocabulary with natural language, and you get instruction-following VLA like LMDrive. The command→action separation pattern persists even in 2025 models.

---

*VLA fit: Ambiguous (command vocabulary, not natural language). Foundational for the VLA trajectory in autonomous driving.*
