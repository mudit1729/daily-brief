# GANs: Generative Adversarial Networks

## Paper Overview

**Authors:** Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
**Published:** 2014 (NeurIPS)
**Paper Link:** https://arxiv.org/abs/1406.2661

## Detailed Description

Generative Adversarial Networks (GANs) introduced a revolutionary framework for training generative models through an adversarial process. Two neural networks—a Generator and a Discriminator—compete against each other in a zero-sum game, where the generator learns to create realistic data and the discriminator learns to distinguish real from fake data.

### Core Architecture:

1. **Generator (G):**
   - Input: Random noise vector z ~ p(z) (typically Gaussian)
   - Output: Fake/generated samples G(z)
   - Goal: Fool the discriminator by generating realistic samples
   - Learns mapping from latent space to data space
   - Never sees real data directly

2. **Discriminator (D):**
   - Input: Either real samples x ~ p_data or fake samples G(z)
   - Output: Probability that input is real: D(x) ∈ [0, 1]
   - Goal: Correctly classify real vs fake samples
   - Acts as learned loss function for generator
   - Provides feedback signal to generator

### Training Process:

The training alternates between two steps:

1. **Train Discriminator:**
   - Sample mini-batch of real data: x ~ p_data
   - Sample mini-batch of noise: z ~ p(z)
   - Generate fake data: G(z)
   - Update D to maximize: log D(x) + log(1 - D(G(z)))
   - D tries to output 1 for real, 0 for fake

2. **Train Generator:**
   - Sample mini-batch of noise: z ~ p(z)
   - Generate fake data: G(z)
   - Update G to maximize: log D(G(z))
   - Equivalently, minimize: log(1 - D(G(z)))
   - G tries to make D output 1 for fake samples

### Mathematical Formulation:

**Minimax Objective:**
```
min_G max_D V(D,G) = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]
```

**Optimal Solution:**
- When D is optimal for a given G: D*(x) = p_data(x) / (p_data(x) + p_g(x))
- At equilibrium: p_g = p_data (generator perfectly mimics data distribution)
- Nash equilibrium: D(x) = 1/2 for all x

**Non-Saturating Objective (Practical):**
Instead of minimizing log(1 - D(G(z))), maximize log D(G(z)) for better gradients early in training.

### Key Properties:

1. **Game Theory Framework:**
   - Two-player minimax game
   - Generator and discriminator have opposing objectives
   - Equilibrium reached when generator matches data distribution

2. **Implicit Density:**
   - Unlike VAE, doesn't explicitly model p(x)
   - Learns to sample from distribution without density function
   - More flexible, can model complex distributions

3. **No Explicit Likelihood:**
   - Doesn't require likelihood computation
   - Avoids intractable partition functions
   - Enables training on complex, high-dimensional data

## Pain Point Addressed

### Limitations of Previous Generative Models:

1. **Restrictive Assumptions:**
   - Many generative models required specific distributional assumptions
   - Difficulty modeling complex, multi-modal distributions
   - Limited expressiveness for high-dimensional data

2. **Computational Intractability:**
   - **Boltzmann Machines:** Required expensive MCMC sampling
   - **Variational Methods:** Needed tractable lower bounds
   - **Likelihood-Based Models:** Partition function computation intractable
   - Scaling to high dimensions was challenging

3. **Blurry Generations:**
   - VAEs often produced blurry images
   - Maximum likelihood training tends to average over modes
   - Mean square error encourages blurriness

4. **Slow Sampling:**
   - Autoregressive models (e.g., PixelCNN) generate pixels sequentially
   - Extremely slow for high-resolution images
   - Impractical for real-time applications

5. **Limited Flexibility:**
   - Many models restricted to specific architectures
   - Difficulty incorporating advances in deep learning
   - Hard to scale with modern hardware

6. **Mode Collapse:**
   - While addressing other issues, GANs introduced new challenges
   - Can fail to capture all modes of the data distribution
   - Training instability

## Novelty of the Paper

### Revolutionary Innovations:

1. **Adversarial Training Framework:**
   - Completely new paradigm for training generative models
   - Framed generation as a two-player game
   - Discriminator acts as learned, adaptive loss function
   - Eliminated need for explicit density estimation

2. **Game-Theoretic Approach:**
   - Applied game theory to machine learning
   - Nash equilibrium as training objective
   - Theoretical guarantees under ideal conditions
   - Elegant mathematical formulation

3. **Backpropagation Only:**
   - No need for Markov chains or variational bounds
   - Simple backpropagation through both networks
   - Leverages standard deep learning tools
   - Scalable with modern hardware (GPUs)

4. **Sharp, Realistic Generations:**
   - Produces much sharper images than VAEs
   - Doesn't suffer from blurriness
   - Captures fine details well
   - Better perceptual quality

5. **Flexible Architecture:**
   - Works with any differentiable generator/discriminator
   - Can incorporate CNNs, ResNets, Transformers, etc.
   - Easy to extend and modify
   - Adapts to various data types

6. **Implicit Generative Models:**
   - Learns to sample without explicit density
   - More flexible than likelihood-based approaches
   - Can model arbitrarily complex distributions
   - No restricting assumptions on data distribution

7. **Fast Sampling:**
   - Single forward pass through generator
   - Parallel generation of samples
   - No sequential dependencies
   - Real-time generation possible

### Theoretical Contributions:

1. **Convergence Analysis:**
   - Proved that with optimal discriminator, minimizing V(G,D) is equivalent to minimizing Jensen-Shannon divergence
   - Showed global optimum is when p_g = p_data
   - Provided theoretical foundation for adversarial training

2. **Connection to Divergences:**
   - Later work showed GANs minimize various f-divergences
   - Different GAN variants correspond to different divergences
   - Rich theoretical framework emerged

### Impact:

- Spawned entire field of adversarial learning
- Influenced development of countless variants:
  - **DCGAN:** Deep Convolutional GAN
  - **WGAN:** Wasserstein GAN (better training stability)
  - **StyleGAN:** High-quality, controllable image synthesis
  - **CycleGAN:** Unpaired image-to-image translation
  - **BigGAN:** Large-scale high-fidelity image generation
  - **Pix2Pix:** Paired image-to-image translation
- Applications in art, design, data augmentation, drug discovery
- Influenced adversarial robustness research

## Implementation

Below is a PyTorch implementation of GAN from the repository:

```python
import torch
import torch.nn as nn
import os
import torchvision.utils as vutils
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    """
    Discriminator Network
    Classifies inputs as real or fake

    Args:
        in_features: Dimension of input (e.g., 784 for 28x28 images)
    """
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            # Input layer
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),  # LeakyReLU helps prevent dead neurons

            # Output layer
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output probability in [0, 1]
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    """
    Generator Network
    Generates fake samples from random noise

    Args:
        z_dim: Dimension of latent noise vector
        img_dim: Dimension of output image
    """
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            # Input layer
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),

            # Output layer
            nn.Linear(256, img_dim),
            nn.Tanh(),  # Output in [-1, 1] to match normalized data
        )

    def forward(self, x):
        return self.gen(x)


# HYPERPARAMETERS
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4  # Learning rate
z_dim = 64  # Dimension of noise vector
image_dim = 28 * 28 * 1  # 28x28 grayscale images (MNIST)
batch_size = 32
num_epochs = 50

# INITIALIZE MODELS
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)  # For consistent visualization

# DATA PREPARATION
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# OPTIMIZERS
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# LOSS FUNCTION
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

# TENSORBOARD SETUP
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

# CREATE OUTPUT DIRECTORY
os.makedirs("generated_images", exist_ok=True)

# TRAINING LOOP
# The training loop implements alternating optimization of the generator and discriminator.
# This is the core of GAN training:
# 1. Train discriminator to correctly classify real vs fake data
# 2. Train generator to fool the discriminator
# By alternating these steps, both networks improve together in an adversarial process.
# The .detach() on fake samples prevents backprop through generator during discriminator training.
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)  # Flatten images
        current_batch_size = real.shape[0]

        ### TRAIN DISCRIMINATOR: max log(D(x)) + log(1 - D(G(z))) ###

        # Generate fake data
        noise = torch.randn(current_batch_size, z_dim).to(device)
        fake = gen(noise)

        # Discriminator on real data
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        # Discriminator on fake data
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Combined discriminator loss
        lossD = (lossD_real + lossD_fake) / 2

        # Backpropagation for discriminator
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        ### TRAIN GENERATOR: min log(1 - D(G(z))) <-> max log(D(G(z))) ###

        # Discriminator's judgement on fake data
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))  # Want D to output 1

        # Backpropagation for generator
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # LOGGING
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} "
                f"Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            # Generate images with fixed noise for visualization
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                img_grid_fake = vutils.make_grid(fake, normalize=True)

                # Save generated images
                vutils.save_image(img_grid_fake, f"generated_images/fake_epoch_{epoch}.png")

                # Log to tensorboard
                writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
                step += 1
```

### Usage Example:

```python
# After training, generate new samples
def generate_samples(generator, num_samples=64, z_dim=64):
    """Generate new samples using trained generator"""
    generator.eval()
    with torch.no_grad():
        # Sample random noise
        z = torch.randn(num_samples, z_dim).to(device)

        # Generate fake images
        fake_images = generator(z).reshape(-1, 1, 28, 28)

        # Denormalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2

    return fake_images

# Generate 64 new MNIST-like digits
samples = generate_samples(gen, num_samples=64, z_dim=64)
```

## Training Challenges and Solutions:

### Common Issues:

1. **Mode Collapse:**
   - Generator produces limited variety
   - Solution: Minibatch discrimination, unrolled GANs, multiple discriminators

2. **Training Instability:**
   - Loss oscillates, doesn't converge
   - Solution: Careful learning rate tuning, WGAN, Spectral Normalization

3. **Vanishing Gradients:**
   - Discriminator too strong, generator gets no useful gradient
   - Solution: Non-saturating loss, label smoothing, noise injection

4. **Hyperparameter Sensitivity:**
   - Requires careful tuning
   - Solution: Experience, architecture search, gradient penalty

### Best Practices:

- Use LeakyReLU in discriminator
- Use batch normalization (except D input and G output)
- Avoid sparse gradients (ReLU, MaxPool in G)
- Use Adam optimizer
- Train D more than G initially
- Use label smoothing
- Add noise to discriminator inputs

## Key Results and Applications:

### Capabilities:

1. **Image Generation:** High-quality, realistic images
2. **Image-to-Image Translation:** Style transfer, domain adaptation
3. **Data Augmentation:** Generate synthetic training data
4. **Super-Resolution:** Enhance image quality
5. **Inpainting:** Fill missing regions in images
6. **Anomaly Detection:** Identify outliers

### Advantages:

- Sharp, realistic generations
- Fast sampling
- Flexible architecture
- No explicit density required
- Captures complex distributions

### Limitations:

- Training instability
- Mode collapse
- Difficult to evaluate
- Hyperparameter sensitive
- No direct way to encode data

## Key Takeaways:

1. **Adversarial training** is a powerful paradigm for generative modeling
2. **Game-theoretic framework** provides elegant theoretical foundation
3. **Sharp generations** superior to previous methods
4. **Fast sampling** enables real-time applications
5. **Flexible and scalable** architecture
6. **Training challenges** require careful engineering
7. Foundation for modern generative AI

## Repository Reference

Implementation source: https://github.com/atullchaurasia/problem-solving/tree/main/GANs
