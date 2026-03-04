# PEFT: Parameter-Efficient Fine-Tuning

## Paper Overview

| | |
|---|---|
| **Concept** | Parameter-Efficient Fine-Tuning (PEFT) |
| **Context** | A family of techniques rather than a single paper |

**Key Papers:**
- "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al., 2019) -- Adapters
- "Prefix-Tuning" (Li & Liang, 2021)
- "LoRA" (Hu et al., 2021)
- "The Power of Scale for Parameter-Efficient Prompt Tuning" (Lester et al., 2021)

---

## Detailed Description

**Parameter-Efficient Fine-Tuning** (PEFT) refers to a collection of methods designed to adapt large pre-trained models to downstream tasks while updating only a small fraction of the model's parameters. PEFT has become crucial in the era of large language models, where full fine-tuning is computationally prohibitive.

### Core PEFT Techniques

#### 1. Adapter Layers (Houlsby et al., 2019)

- Insert small bottleneck layers between transformer blocks
- Structure: Down-project -> Non-linearity -> Up-project
- Only adapter parameters are trained; original weights frozen
- Adds minimal parameters (~0.5-8% of original model)

#### 2. Prefix Tuning (Li & Liang, 2021)

- Prepends learnable "prefix" vectors to each layer
- Prefixes are virtual tokens that influence attention
- Original model parameters remain frozen
- Only prefix parameters are optimized

#### 3. Prompt Tuning (Lester et al., 2021)

- Simplified version of prefix tuning
- Adds learnable *soft prompts* only to the input layer
- Extremely parameter-efficient
- Scales well with model size

#### 4. LoRA (Hu et al., 2021)

- Low-rank decomposition of weight updates
- Injects trainable rank decomposition matrices
- Zero inference latency after merging
- Highly effective with minimal parameters

#### 5. (IA)^3 -- Infused Adapter by Inhibiting and Amplifying Inner Activations (Liu et al., 2022)

- Learns vectors that scale activations
- Even more parameter-efficient than LoRA
- Rescales keys and values in attention, and feed-forward activations

#### 6. BitFit: Back-propagation of Bias terms (Zaken et al., 2021)

- Trains only bias terms in the model
- Extremely simple and parameter-efficient
- Surprisingly effective despite minimal changes

### Common Principles

1. **Freeze Pre-trained Weights:** Keep the majority of model parameters frozen
2. **Selective Updates:** Modify only specific, strategically chosen parameters
3. **Additive Modifications:** Add new trainable components rather than modifying existing ones
4. **Task-Specific Modules:** Create small, task-specific modules that can be swapped

### PEFT in Practice

The Hugging Face PEFT library provides unified implementations:

```python
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType
)

# Example: Using LoRA
config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,  # rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]  # which layers to adapt
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# Output: trainable params: 2,359,296 || all params: 11,245,895,680 || trainable%: 0.02%
```

---

## Pain Point Addressed

### Challenges with Full Fine-Tuning

1. **Computational Expense:**
   - Modern LLMs have billions to trillions of parameters
   - Full fine-tuning requires updating all parameters
   - GPU memory requirements are enormous (3-4x model size for gradients + optimizer states)
   - Training time is prohibitively long

2. **Storage Costs:**
   - Each task requires storing a complete model copy
   - For GPT-3 (175B params), each task needs ~350GB
   - Maintaining multiple task-specific models is impractical
   - Version control and distribution become challenging

3. **Deployment Complexity:**
   - Cannot efficiently serve multiple fine-tuned versions
   - Model switching requires loading entire models
   - High memory footprint in production
   - Scaling to many tasks is impractical

4. **Accessibility:**
   - Only well-funded organizations can afford full fine-tuning
   - Creates barriers for researchers and small companies
   - Limits experimentation and innovation

5. **Catastrophic Forgetting:**
   - Full fine-tuning can degrade general capabilities
   - Model forgets pre-trained knowledge
   - Trade-off between task performance and general ability

6. **Environmental Impact:**
   - Massive energy consumption for training
   - Large carbon footprint
   - Inefficient use of computational resources

7. **Overfitting Risk:**
   - Fine-tuning all parameters on small datasets often leads to overfitting
   - Regularization becomes more challenging
   - PEFT methods provide implicit regularization

---

## Novelty and Key Innovations

### Conceptual Innovations

1. **Paradigm Shift:**
   - Challenges the assumption that all parameters need updating
   - Demonstrates that task adaptation lives in a low-dimensional subspace
   - Shows that strategic parameter selection is more important than quantity

2. **Modularity:**
   - Enables plug-and-play task-specific modules
   - Base model remains universal and shared
   - Easy switching between tasks at runtime

3. **Democratization:**
   - Makes large model adaptation accessible to broader community
   - Enables fine-tuning on consumer hardware
   - Lowers barrier to entry for research and applications

4. **Efficient Multi-Task Learning:**
   - Single base model can support unlimited tasks
   - Task-specific adapters are small (KBs to MBs)
   - Enables "adapter hubs" for sharing task-specific modules

### Technical Innovations

1. **Adapter Layers:**
   - Introduced bottleneck architecture for parameter efficiency
   - Demonstrated that small capacity can capture task-specific information
   - Minimal impact on model performance

2. **Soft Prompts:**
   - Showed that continuous prompts outperform discrete prompts
   - Demonstrated prompt tuning scales with model size
   - Extremely parameter-efficient (0.001-0.1% of parameters)

3. **Low-Rank Hypothesis:**
   - Validated that weight updates have low intrinsic rank
   - Enabled methods like LoRA
   - Provided theoretical justification for PEFT

4. **Layer-Wise Adaptation:**
   - Identified which layers benefit most from adaptation
   - Showed that not all layers need equal treatment
   - Enabled targeted, efficient fine-tuning

5. **Zero-Latency Deployment:**
   - Methods like LoRA can be merged for inference
   - No runtime overhead compared to full fine-tuning
   - Best of both worlds: efficiency in training, speed in inference

### Comparison of PEFT Methods

| Method | Trainable % | Inference Overhead | Storage per Task | Performance |
|--------|-------------|-------------------|------------------|-------------|
| Full Fine-Tuning | 100% | None | Full model size | Baseline |
| Adapter Layers | 0.5-8% | Small | Small | Competitive |
| Prefix Tuning | 0.1-3% | Small | Small | Competitive |
| Prompt Tuning | 0.001-0.1% | None | Tiny | Good (better with scale) |
| LoRA | 0.01-1% | None (merged) | Small | Excellent |
| (IA)^3 | 0.001-0.01% | Small | Tiny | Very good |
| BitFit | 0.01-0.1% | None | Tiny | Moderate |

> **When to use which method:** Use Adapters for best performance-efficiency balance, LoRA for zero-latency inference and broad compatibility, Prompt Tuning for extreme parameter efficiency on large models, Prefix Tuning for generation tasks, (IA)^3 for maximum efficiency, and BitFit for simplicity when computational constraints are minimal.

---

## Implementation Concepts

While the repository doesn't have a specific PEFT implementation, here's a conceptual overview:

### 1. Adapter Implementation

```python
import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    """
    Bottleneck adapter layer that efficiently adapts pre-trained models.

    Implements a down-project -> activation -> up-project architecture
    with a residual connection. Only adapter parameters are trained while
    the base model weights remain frozen.

    Args:
        hidden_size: Dimension of the input/output
        adapter_size: Dimension of the bottleneck (typically 64 or smaller)
        dropout: Dropout rate for regularization
    """
    def __init__(self, hidden_size, adapter_size, dropout=0.1):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # Bottleneck: down -> activation -> up
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        return x + residual  # Residual connection
```

### 2. Prefix Tuning Implementation

```python
import torch
import torch.nn as nn

class PrefixTuning(nn.Module):
    """
    Prefix tuning method that prepends learnable virtual tokens to each layer.

    Maintains a set of learnable prefix embeddings for each transformer layer.
    These prefixes are optimized to influence the attention mechanism without
    modifying the base model weights.

    Note: The original paper uses an MLP reparameterization for stable training,
    which this simplified implementation omits for clarity.

    Args:
        prefix_length: Number of prefix tokens to add per layer
        num_layers: Number of transformer layers
        hidden_size: Dimension of hidden states
    """
    def __init__(self, prefix_length, num_layers, hidden_size):
        super().__init__()
        self.prefix_length = prefix_length
        # Learnable prefix for each layer
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_layers, prefix_length, hidden_size)
        )

    def forward(self, layer_idx, batch_size):
        # Get prefix for this layer
        prefix = self.prefix_embeddings[layer_idx]
        # Expand for batch
        return prefix.unsqueeze(0).expand(batch_size, -1, -1)
```

### 3. Prompt Tuning Implementation

```python
import torch
import torch.nn as nn

class PromptTuning(nn.Module):
    """
    Simplified prompt tuning that adds learnable soft prompts to input embeddings.

    Only optimizes continuous prompt embeddings prepended to the input layer,
    making this the most parameter-efficient PEFT method. Scales effectively
    with model size, showing better performance on larger models.

    Args:
        num_prompts: Number of soft prompt tokens to learn
        embedding_dim: Dimension of embedding space
    """
    def __init__(self, num_prompts, embedding_dim):
        super().__init__()
        # Learnable soft prompts
        self.soft_prompts = nn.Parameter(
            torch.randn(num_prompts, embedding_dim)
        )

    def forward(self, input_embeddings):
        batch_size = input_embeddings.size(0)
        # Expand prompts for batch
        prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        # Concatenate with input
        return torch.cat([prompts, input_embeddings], dim=1)
```

### 4. BitFit Implementation

```python
def apply_bitfit(model):
    """Freeze all parameters except bias terms"""
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
```

---

## Key Results

### General Findings

- PEFT methods match or exceed full fine-tuning on many tasks
- Parameter reduction: 100x to 10,000x fewer trainable parameters
- Storage reduction: Multiple tasks can be stored for the cost of one full model
- Training speed: 2-10x faster than full fine-tuning

### Task-Specific

- **Natural Language Understanding:** Adapters and LoRA achieve 95-100% of full fine-tuning performance
- **Generation Tasks:** Prefix tuning and LoRA excel, matching full fine-tuning
- **Few-Shot Learning:** Prompt tuning particularly effective with large models
- **Multi-Task:** PEFT enables efficient multi-task learning with shared base model

---

## Key Takeaways

> 1. **Not All Parameters Matter:** Strategic selection of parameters is more important than quantity
> 2. **Low-Dimensional Adaptation:** Task-specific adaptations exist in low-dimensional subspaces
> 3. **Modularity Wins:** Separating base knowledge from task-specific skills is powerful
> 4. **Scalability:** PEFT makes large model adaptation practical and accessible
> 5. **Implicit Regularization:** Constraining updates provides regularization, reducing overfitting
> 6. **Future-Proof:** As models grow larger, PEFT becomes increasingly essential

---

## Popular Libraries

- **Hugging Face PEFT:** Unified library for LoRA, Prefix Tuning, P-Tuning, Prompt Tuning, (IA)^3
- **Adapter-Transformers:** Specialized library for adapter-based methods
- **OpenDelta:** Microsoft's library for various delta-tuning methods

---

## Repository Reference

Note: The referenced repository doesn't have a specific PEFT implementation, but it includes LoRA which is one of the most popular PEFT techniques. See the LoRA markdown file for the implementation from this repository.

Repository: https://github.com/atullchaurasia/problem-solving
