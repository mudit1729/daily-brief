# VAE: Auto-Encoding Variational Bayes (Variational Autoencoder)

## Paper Overview

| | |
|---|---|
| **Authors** | Diederik P. Kingma, Max Welling |
| **Published** | 2013 (ICLR 2014) |
| **Paper Link** | https://arxiv.org/abs/1312.6114 |

---

## Detailed Description

The **Variational Autoencoder** (VAE) is a generative model that combines deep learning with variational inference. It learns to encode data into a latent space and decode it back, while ensuring the latent space has nice properties for generating new samples. VAEs are probabilistic models that learn the underlying probability distribution of the data.

### Key Components

1. **Encoder (Recognition Network):**
   - Maps input data x to latent space parameters
   - Outputs mean (mu) and variance (sigma^2) of latent distribution
   - Approximates posterior distribution q(z|x) ~ p(z|x)
   - Implemented as neural network

2. **Latent Space:**
   - Compressed representation of data
   - Assumed to follow Gaussian distribution: z ~ N(mu, sigma^2)
   - Enables sampling and interpolation
   - Typically much lower dimensional than input

3. **Reparameterization Trick:**
   - Enables backpropagation through stochastic sampling
   - `z = mu + sigma * epsilon`, where epsilon ~ N(0, 1)
   - Moves randomness to input, making operation differentiable
   - Critical innovation for training VAEs

4. **Decoder (Generative Network):**
   - Maps latent code z back to data space
   - Reconstructs original input
   - Models p(x|z)
   - Implemented as neural network

5. **Loss Function (ELBO):**
   - *Evidence Lower Bound* (ELBO) maximized during training
   - Two components:
     - **Reconstruction Loss:** How well can we reconstruct the input?
       - `L_recon = E[log p(x|z)]`
       - Often MSE for continuous data, BCE for binary
     - **KL Divergence:** How close is q(z|x) to prior p(z)?
       - `L_KL = KL[q(z|x) || p(z)]`
       - Regularizes latent space to be normally distributed
   - Total loss: `L = L_recon + beta * L_KL`

### Mathematical Framework

**Objective:**
```math
Maximize: log p(x) = log integral p(x|z) p(z) dz
Problem: Intractable integral
```

**Variational Inference:**
```math
Approximate p(z|x) with q(z|x) = N(mu(x), sigma^2(x))
Maximize ELBO: L = E_q[log p(x|z)] - KL[q(z|x) || p(z)]
```

**Reparameterization:**
```math
z = mu + sigma * epsilon, where epsilon ~ N(0,1)
Allows: grad_theta E_q[f(z)] = E_p[grad_theta f(z)]
```

**KL Divergence (Closed Form):**
```math
For q(z|x) = N(mu, sigma^2) and p(z) = N(0, 1):
KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

### VAE Variants

| Variant | Description |
|---------|-------------|
| **beta-VAE** | Introduces beta hyperparameter to control disentanglement |
| **Conditional VAE (CVAE)** | Conditions on labels for controlled generation |
| **VQ-VAE** | Uses discrete latent codes instead of continuous |
| **WAE** | Wasserstein AutoEncoders using different divergence |

---

## Pain Point Addressed

### Limitations of Traditional Autoencoders

1. **No Probabilistic Framework:**
   - Standard autoencoders don't model probability distributions
   - Difficult to generate new samples
   - Latent space is unstructured and discontinuous

2. **Poor Generative Capability:**
   - Traditional autoencoders only reconstruct seen data
   - Cannot reliably sample from latent space to generate new data
   - Latent codes may be sparse or have "holes"

3. **No Regularization of Latent Space:**
   - Latent representations can be arbitrary
   - Similar inputs might have very different latent codes
   - No smooth interpolation possible

4. **Overfitting:**
   - Without constraints, autoencoders can memorize training data
   - Latent space may not generalize well
   - No control over latent space structure

### Limitations of Earlier Generative Models

1. **Restricted Boltzmann Machines (RBMs):**
   - Difficult to train
   - Limited to specific architectures
   - Slow sampling process

2. **Variational Methods:**
   - Previously required complex mathematical derivations for each model
   - Not scalable with deep neural networks
   - Limited to simple models

3. **Lack of Scalability:**
   - Earlier generative models didn't scale to high-dimensional data
   - Couldn't handle complex datasets like images effectively

---

## Novelty of the Paper

### Key Innovations

1. **Reparameterization Trick:**
   - Elegant solution to backpropagation through stochastic nodes
   - Enables end-to-end training with stochastic latent variables
   - Transformed variational inference with neural networks
   - Simple yet powerful: `z = mu + sigma * epsilon`

2. **Scalable Variational Inference:**
   - Made variational inference practical for deep neural networks
   - Stochastic gradient descent for variational inference
   - Scalable to large datasets and high dimensions

3. **Unified Framework:**
   - Combines:
     - Deep learning (neural networks)
     - Probabilistic modeling (Bayesian inference)
     - Variational methods
   - End-to-end differentiable generative model

4. **Structured Latent Space:**
   - Enforces continuous, well-structured latent space
   - Enables smooth interpolation between samples
   - Allows meaningful latent arithmetic
   - KL regularization ensures latent codes follow known distribution

5. **Amortized Inference:**
   - Encoder amortizes the cost of inference
   - Single forward pass vs iterative inference
   - Shares parameters across all data points

6. **Practical Generative Model:**
   - Can generate new samples by sampling from prior: z ~ N(0,1)
   - Enables controlled generation through latent manipulation
   - Balances reconstruction quality and latent space structure

### Theoretical Contributions

1. Showed that ELBO can be efficiently optimized using SGD
2. Demonstrated that deep neural networks can approximate complex posteriors
3. Provided practical framework for learning latent variable models

### Impact

> VAEs became a foundation for modern generative models, influencing the development of GANs, Normalizing Flows, and Diffusion Models. They are widely used in semi-supervised learning, anomaly detection, drug discovery, image generation, text generation, and music synthesis.

---

## Implementation

Below is a PyTorch implementation of VAE from the repository:

```python
import torch
import torch.nn.functional as F
from torch import nn

class VariationalAutoEncoder(nn.Module):
    """
    Variational Autoencoder (VAE) implementation

    Args:
        input_dim: Dimension of input data (e.g., 784 for 28x28 images)
        h_dim: Hidden layer dimension (default: 200)
        z_dim: Latent space dimension (default: 20)
    """
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # ENCODER: Maps input to latent distribution parameters
        self.img_2hid = nn.Linear(input_dim, h_dim)   # Input to hidden
        self.hid_2mu = nn.Linear(h_dim, z_dim)        # Hidden to mean
        self.hid_2log_var = nn.Linear(h_dim, z_dim)   # Hidden to log-variance

        # DECODER: Maps latent code to reconstruction
        self.z_2hid = nn.Linear(z_dim, h_dim)        # Latent to hidden
        self.hid_2img = nn.Linear(h_dim, input_dim)  # Hidden to output

    def encode(self, x):
        """
        Encoder: q(z|x) - Approximate posterior

        Args:
            x: Input data

        Returns:
            mu: Mean of latent distribution
            log_var: Log-variance of latent distribution
        """
        # Hidden layer with ReLU activation
        h = F.relu(self.img_2hid(x))

        # Output mean and log-variance
        mu = self.hid_2mu(h)
        log_var = self.hid_2log_var(h)

        return mu, log_var

    def decode(self, z):
        """
        Decoder: p(x|z) - Likelihood

        Args:
            z: Latent code

        Returns:
            Reconstructed output (passed through sigmoid)
        """
        # Hidden layer with ReLU activation
        h = F.relu(self.z_2hid(z))

        # Output reconstruction with sigmoid activation
        # Sigmoid ensures output is in [0, 1] range
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        """
        Full forward pass: encode, sample, decode

        Args:
            x: Input data

        Returns:
            x_reconstructed: Reconstructed input
            mu: Mean of latent distribution
            log_var: Log-variance of latent distribution
        """
        # Encode input to get distribution parameters
        mu, log_var = self.encode(x)

        # Reparameterization trick: z = mu + sigma * epsilon, where epsilon ~ N(0,1)
        # We use log_var for numerical stability: sigma = exp(0.5 * log_var)
        # This avoids computing log on small positive numbers (numerically unstable)
        # and ensures sigma > 0 even if the network outputs negative log_var values
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z_new = mu + std * epsilon

        # Decode latent code to reconstruct input
        x_reconstructed = self.decode(z_new)

        return x_reconstructed, mu, log_var


# EXAMPLE USAGE
if __name__ == "__main__":
    # Create sample input: batch of 4 images (28x28 flattened)
    x = torch.rand(4, 28*28)

    # Initialize VAE
    vae = VariationalAutoEncoder(input_dim=784)

    # Forward pass
    x_reconstructed, mu, log_var = vae(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log-Var shape: {log_var.shape}")

    # Output:
    # Input shape: torch.Size([4, 784])
    # Reconstructed shape: torch.Size([4, 784])
    # Mu shape: torch.Size([4, 20])
    # Log-Var shape: torch.Size([4, 20])
```

### Training Code Example

```python
def vae_loss(x, x_recon, mu, log_var):
    """
    VAE loss function: ELBO = Reconstruction Loss + KL Divergence

    Args:
        x: Original input
        x_recon: Reconstructed input
        mu: Mean of latent distribution
        log_var: Log-variance of latent distribution

    Returns:
        Total VAE loss
    """
    # Reconstruction loss: Binary Cross-Entropy
    # Measures how well we reconstruct the input
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL Divergence: D_KL[q(z|x) || p(z)]
    # Measures how close our latent distribution is to standard normal
    # Closed form for N(mu, sigma^2) vs N(0, 1):
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # We use log_var directly (instead of computing torch.log(sigma**2))
    # for numerical stability: log(sigma^2) = 2*log(sigma) = 2*0.5*log_var = log_var
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss: ELBO (Evidence Lower Bound)
    return recon_loss + kl_divergence


# Training loop example
def train_vae(model, dataloader, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (data, _) in enumerate(dataloader):
            # Flatten images: (batch, 1, 28, 28) -> (batch, 784)
            data = data.view(-1, 784)

            # Forward pass
            x_recon, mu, log_var = model(data)

            # Compute loss
            loss = vae_loss(data, x_recon, mu, log_var)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')


# Generate new samples
def generate_samples(model, num_samples=16):
    """Generate new samples by sampling from prior"""
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, 20)  # z_dim = 20

        # Decode to generate samples
        samples = model.decode(z)

        # Reshape for visualization: (num_samples, 784) -> (num_samples, 28, 28)
        samples = samples.view(num_samples, 28, 28)

    return samples
```

---

## Key Results and Applications

### Capabilities

1. **Generation:** Sample new data by drawing z ~ N(0,1) and decoding
2. **Reconstruction:** Encode and decode data
3. **Interpolation:** Smooth transitions between samples in latent space
4. **Anomaly Detection:** Low reconstruction quality indicates anomalies
5. **Data Compression:** Efficient latent representation

### Advantages

- Principled probabilistic framework
- Smooth, continuous latent space
- Enables controlled generation
- Balances reconstruction and regularization
- Scalable training

### Limitations

- Often produces blurry images (compared to GANs)
- Balancing reconstruction vs KL divergence can be tricky
- Assumes Gaussian latent distribution (may not always be appropriate)
- Posterior collapse in some cases

---

## Key Takeaways

> 1. **Reparameterization trick** enables end-to-end training with stochastic variables
> 2. **ELBO objective** balances reconstruction quality and latent space structure
> 3. **Probabilistic framework** provides principled approach to generation
> 4. **Structured latent space** enables interpolation and controlled generation
> 5. **Scalable variational inference** made practical with neural networks
> 6. Foundation for modern generative modeling

---

## Repository Reference

Implementation source: https://github.com/atullchaurasia/problem-solving/tree/main/VAE
