# Deep-ML Medium Problems: Comprehensive Guide

A complete guide covering 5 medium-level problems with theory, implementations, and PyTorch integration.

## Table of Contents

- [Quick Reference Table](#quick-reference-table)
- [1. Matrix Transformation](#1-matrix-transformation)
- [2. Calculate Covariance Matrix](#2-calculate-covariance-matrix)
- [3. K-Means Clustering](#3-k-means-clustering)
- [4. Solve Linear Equations using Jacobi Method](#4-solve-linear-equations-using-jacobi-method)
- [5. Calculate Correlation Matrix](#5-calculate-correlation-matrix)

---

## Quick Reference Table

| # | Problem | Domain | Key Algorithm | Complexity | PyTorch Module |
|---|---------|--------|---------------|-----------|----------------|
| 1 | Matrix Transformation | Linear Algebra | Basis change via T^(-1)AS | O(n³) | `torch.linalg.solve()` |
| 2 | Covariance Matrix | Statistics | Σ(x-μ)(y-ν)/(n-1) | O(n²m) | `torch.cov()` |
| 3 | K-Means Clustering | Machine Learning | Iterative centroid update | O(nki) | Custom implementation |
| 4 | Jacobi Method | Numerical Methods | Iterative linear solver | O(n²i) | Custom implementation |
| 5 | Correlation Matrix | Statistics | Covariance / (σ_x σ_y) | O(n²m) | `torch.corrcoef()` |

**Legend:** n = features/dimensions, m = observations, k = clusters, i = iterations

---

# 1. Matrix Transformation

## Problem Statement

Transform matrix A using invertible matrices T and S: **A' = T⁻¹AS** (change of basis in linear algebra)

## Mathematical Theory

**Change of Basis:** Given invertible matrices T and S, the transformed matrix A' represents the same linear transformation in a different basis.

**Formula:** A' = T⁻¹AS where:
- T⁻¹ transforms from basis T to standard basis
- S transforms from standard basis to basis S
- Invertibility: det(M) ≠ 0

**Similarity Transform (S = T⁻¹):** A' = T⁻¹AT preserves determinant, trace, eigenvalues, and rank.

Code notation:
- `A_prime = inv(T) @ A @ S` is the basic form
- For numerical stability: `A_prime = solve(T, A @ S)`

## Implementation

```python
import numpy as np

def transform_matrix(A, T, S):
    """A' = T^(-1)AS. More stable using solve instead of explicit inverse."""
    A, T, S = map(lambda x: np.array(x, dtype=float), [A, T, S])

    det_T, det_S = np.linalg.det(T), np.linalg.det(S)
    if np.abs(det_T) < 1e-10 or np.abs(det_S) < 1e-10:
        raise ValueError(f"Non-invertible: det(T)={det_T:.2e}, det(S)={det_S:.2e}")

    # Solve TX = AS for X (more stable than T^(-1))
    return np.linalg.solve(T, A @ S).tolist()


def similarity_transform(A, T):
    """Similarity transformation preserves eigenvalues, determinant, trace."""
    return np.linalg.solve(T, A @ T)
```

## PyTorch Integration

```python
import torch

def transform_matrix_torch(A, T, S):
    """Transform using solve (numerically stable)."""
    AS = A @ S
    return torch.linalg.solve(T, AS)

# Batch operations
batch_A = torch.randn(32, 3, 3)
batch_T = torch.randn(32, 3, 3)
batch_S = torch.randn(32, 3, 3)
batch_A_prime = torch.linalg.solve(batch_T, batch_A @ batch_S)

# Similarity transformation for diagonalization
def diagonalize(A):
    """D = inv(V) @ A @ V where V contains eigenvectors, D is diagonal."""
    eigvals, eigvecs = torch.linalg.eig(A)
    D = torch.linalg.solve(eigvecs, A @ eigvecs)
    return D, eigvecs

# Invertibility check
def is_invertible(matrix, tol=1e-6):
    return torch.linalg.det(matrix).abs() > tol

# PCA via similarity transformation
def pca_via_transformation(X):
    """Covariance matrix C transformed to diagonal form."""
    X_centered = X - X.mean(dim=0)
    C = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(C)
    idx = torch.argsort(eigvals, descending=True)
    D = eigvecs[:, idx].T @ C @ eigvecs[:, idx]  # Diagonal matrix
    X_pca = X_centered @ eigvecs[:, idx]
    return X_pca, eigvals[idx], eigvecs[:, idx]

# Graph Convolutional Network normalization: A_tilde = inv(D^0.5) @ A @ inv(D^0.5)
def graph_conv_normalize(A_adj):
    """Normalize adjacency matrix by degree matrix."""
    D = torch.diag(A_adj.sum(dim=1))
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag()))
    return D_inv_sqrt @ A_adj @ D_inv_sqrt
```

**Best Practices:**
- Use `torch.linalg.solve()` instead of explicit inverse for numerical stability
- Always check determinant or condition number before assuming invertibility
- For large matrices, consider LU or Cholesky decomposition for efficiency

**Common Applications:**
- PCA (diagonalizing covariance matrix via eigendecomposition)
- Graph neural networks (normalizing adjacency matrices)
- Change of basis in physics and signal processing

---

# 2. Calculate Covariance Matrix

## Problem Statement

Compute the covariance matrix from feature vectors where each vector represents a feature with equal length observations.

## Mathematical Theory

**Covariance:** Cov(X, Y) = (1/(n-1)) Σ(xᵢ - x̄)(yᵢ - ȳ) where n-1 provides Bessel's correction.

**Covariance Matrix (p features, n observations):**
```
Sigma = (1/(n-1)) * (X - mean(X)) @ (X - mean(X)).T
```

**Properties:**
- Symmetric: Sigma[i,j] = Sigma[j,i]
- Positive semi-definite: x.T @ Sigma @ x >= 0
- Diagonal: variance; Off-diagonal: covariances

## Implementation

```python
import numpy as np

def calculate_covariance_matrix(vectors):
    """Manual covariance: Σ(x-μ)(y-μ)/(n-1)"""
    vectors = np.array(vectors, dtype=float)
    n_features, n_obs = vectors.shape[0], vectors.shape[1]

    means = [sum(v) / n_obs for v in vectors]
    cov_matrix = [[0.0] * n_features for _ in range(n_features)]

    for i in range(n_features):
        for j in range(i, n_features):
            cov = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
                     for k in range(n_obs)) / (n_obs - 1)
            cov_matrix[i][j] = cov_matrix[j][i] = cov

    return cov_matrix

def covariance_np(vectors):
    """Vectorized using NumPy."""
    n_obs = vectors.shape[1]
    means = np.mean(vectors, axis=1, keepdims=True)
    centered = vectors - means
    return (centered @ centered.T) / (n_obs - 1)

# Using np.cov (most efficient)
def covariance_builtin(vectors):
    return np.cov(vectors, rowvar=True)

def covariance_with_stats(vectors):
    """Compute covariance with additional statistics."""
    cov_matrix = np.cov(vectors, rowvar=True)
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
    return {
        'covariance': cov_matrix,
        'correlation': corr_matrix,
        'variances': np.diag(cov_matrix),
        'std_devs': std_devs,
        'means': np.mean(vectors, axis=1)
    }
```

## PyTorch Integration

```python
import torch

def calculate_covariance_torch(vectors):
    """Covariance with GPU support."""
    n_obs = vectors.shape[1]
    means = vectors.mean(dim=1, keepdim=True)
    centered = vectors - means
    return (centered @ centered.T) / (n_obs - 1)

# Built-in function (PyTorch 1.10+)
cov_matrix = torch.cov(vectors)

# Batch covariance
def batch_covariance(batch_vectors):
    """Compute covariance for each batch (32, 10, 100) -> (32, 10, 10)."""
    means = batch_vectors.mean(dim=2, keepdim=True)
    centered = batch_vectors - means
    return torch.bmm(centered, centered.transpose(1, 2)) / (batch_vectors.shape[2] - 1)

# Numerically stable with regularization
def stable_covariance(vectors, eps=1e-6):
    cov = torch.cov(vectors)
    return cov + eps * torch.eye(cov.shape[0])

# Mahalanobis distance: d = sqrt((x-mu).T @ inv(Sigma) @ (x-mu))
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    cov_inv = torch.linalg.inv(cov)
    return torch.sqrt(diff @ cov_inv @ diff.T)

# Shrinkage covariance (Ledoit-Wolf)
def shrinkage_covariance(vectors, shrinkage=0.1):
    """Shrinkage estimator improves condition number for high-dimensional data."""
    cov_sample = torch.cov(vectors)
    p = cov_sample.shape[0]
    trace = torch.trace(cov_sample)
    target = (trace / p) * torch.eye(p)
    return (1 - shrinkage) * cov_sample + shrinkage * target

# Multivariate Gaussian sampling using Cholesky
def sample_multivariate_gaussian(mean, cov, n_samples=1000):
    """Sample N(mu, Sigma) using x = mu + L @ z where L @ L.T = Sigma."""
    L = torch.linalg.cholesky(cov)
    z = torch.randn(n_samples, len(mean))
    return mean + (L @ z.T).T

# PCA from covariance eigendecomposition
def pca_from_covariance(vectors, n_components=None):
    cov = torch.cov(vectors)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    if n_components:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    mean = vectors.mean(dim=1, keepdim=True)
    centered = vectors - mean
    return {'projected': eigvecs.T @ centered, 'components': eigvecs, 'explained_variance': eigvals}

# Portfolio variance: sigma_p^2 = w.T @ Sigma @ w
def minimum_variance_portfolio(cov_matrix):
    """Find weights w minimizing portfolio variance."""
    cov_inv = torch.linalg.inv(cov_matrix)
    ones = torch.ones(cov_matrix.shape[0])
    return (cov_inv @ ones) / (ones @ cov_inv @ ones)
```

**Best Practices:**
- Always center data by subtracting the mean before computing covariance
- Use `torch.cov()` built-in for efficiency; avoid explicit matrix multiplication when possible
- Add regularization (eps term) for numerical stability in high dimensions
- Use shrinkage estimators (Ledoit-Wolf) when sample size is small relative to dimensions

**Common Applications:**
- PCA (eigendecomposition of covariance for dimensionality reduction)
- Multivariate Gaussian modeling and sampling
- Mahalanobis distance for anomaly detection
- Portfolio risk analysis in finance

---

# 3. K-Means Clustering

## Problem Statement

Partition n points into k clusters by iteratively: (1) assigning each point to nearest centroid, (2) updating centroids as mean of assigned points, (3) repeating until convergence.

## Mathematical Theory

**Objective:** Minimize within-cluster sum of squares (WCSS): J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²

**Centroid Update:** mu_i^(t+1) = (1/|C_i|) * Σ(x in C_i) x

**Convergence:** mu^(t+1) = mu^(t)

**Complexity:** O(n × k × i × d) where n=points, k=clusters, i=iterations, d=dims

**Properties:** Always converges to local minimum (not global); Sensitive to initialization

## Implementation

```python
import numpy as np
import math

def k_means(points, k, initial_centroids, max_iterations):
    """K-means with Euclidean distance."""
    def euclidean_distance(p1, p2):
        return math.sqrt(sum((p1[j] - p2[j])**2 for j in range(len(p1))))

    centroids = list(initial_centroids)

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]

        # Assignment step
        for point in points:
            distances = [euclidean_distance(point, c) for c in centroids]
            clusters[np.argmin(distances)].append(point)

        # Update step
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                dim = len(cluster[0])
                mean = tuple(round(sum(p[j] for p in cluster) / len(cluster), 4) for j in range(dim))
                new_centroids.append(mean)
            else:
                new_centroids.append(centroids[i])

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids

# NumPy vectorized
def k_means_np(points, k, initial_centroids, max_iterations):
    """Broadcasting: (n, 1, d) - (1, k, d) = (n, k, d)."""
    centroids = initial_centroids.copy()

    for _ in range(max_iterations):
        diff = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            points[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])
        new_centroids = np.round(new_centroids, 4)

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return centroids
```

## PyTorch Integration

```python
import torch

def k_means_torch(points, k, initial_centroids, max_iterations):
    """K-means with GPU support."""
    centroids = initial_centroids.clone()

    for _ in range(max_iterations):
        diff = points.unsqueeze(1) - centroids.unsqueeze(0)
        distances = torch.sqrt((diff ** 2).sum(dim=2))
        labels = torch.argmin(distances, dim=1)

        new_centroids = torch.stack([
            points[labels == i].mean(dim=0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])
        new_centroids = torch.round(new_centroids * 10000) / 10000

        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return centroids

# K-means++ initialization (better starting points)
def kmeans_plus_plus_init(points, k):
    """Choose centroid proportional to squared distance."""
    centroids = []
    idx = torch.randint(0, points.shape[0], (1,))
    centroids.append(points[idx])

    for _ in range(1, k):
        centroid_tensor = torch.stack(centroids).squeeze(1)
        diff = points.unsqueeze(1) - centroid_tensor.unsqueeze(0)
        distances = torch.sqrt((diff ** 2).sum(dim=2))
        min_distances = distances.min(dim=1)[0]

        probabilities = min_distances ** 2
        probabilities /= probabilities.sum()
        idx = torch.multinomial(probabilities, 1)
        centroids.append(points[idx])

    return torch.stack(centroids).squeeze(1)

# Soft k-means (differentiable with softmax)
def soft_k_means(points, k, initial_centroids, max_iterations, temperature=1.0):
    """Soft assignment for gradient flow."""
    centroids = initial_centroids.clone().requires_grad_(True)

    for _ in range(max_iterations):
        diff = points.unsqueeze(1) - centroids.unsqueeze(0)
        distances = (diff ** 2).sum(dim=2)
        assignment_weights = torch.softmax(-distances / temperature, dim=1)
        new_centroids = (assignment_weights.T @ points) / assignment_weights.sum(dim=0, keepdim=True).T

        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids

    return centroids

# Mini-batch k-means for large datasets
def mini_batch_k_means(points, k, initial_centroids, max_iterations, batch_size=100):
    """Update centroids using mini-batches instead of full dataset."""
    centroids = initial_centroids.clone()
    centroid_counts = torch.zeros(k)

    for _ in range(max_iterations):
        indices = torch.randperm(points.shape[0])[:batch_size]
        batch = points[indices]

        diff = batch.unsqueeze(1) - centroids.unsqueeze(0)
        distances = torch.sqrt((diff ** 2).sum(dim=2))
        labels = torch.argmin(distances, dim=1)

        for i in range(k):
            mask = labels == i
            if mask.any():
                count = mask.sum().float()
                centroid_counts[i] += count
                eta = 1.0 / centroid_counts[i]
                centroids[i] += eta * (batch[mask].sum(dim=0) - count * centroids[i])

    return centroids

# WCSS for elbow method
def elbow_method(points, max_k=10):
    """Compute WCSS for different k to find optimal k."""
    wcss_values = []
    for k in range(1, max_k + 1):
        centroids = kmeans_plus_plus_init(points, k)
        final_centroids = k_means_torch(points, k, centroids, max_iterations=100)

        diff = points.unsqueeze(1) - final_centroids.unsqueeze(0)
        distances = (diff ** 2).sum(dim=2)
        wcss = distances.min(dim=1)[0].sum().item()
        wcss_values.append(wcss)

    return wcss_values

# Silhouette score for cluster quality
def silhouette_score(points, labels, centroids):
    """Score from -1 (poor) to 1 (well separated)."""
    scores = []
    for i in range(points.shape[0]):
        same_cluster = points[labels == labels[i]]
        a = torch.norm(same_cluster - points[i], dim=1).mean() if len(same_cluster) > 1 else torch.tensor(0.0)

        b_vals = [torch.norm(points[labels == j] - points[i], dim=1).mean()
                 for j in range(centroids.shape[0]) if j != labels[i] and (labels == j).any()]
        b = min(b_vals) if b_vals else torch.tensor(0.0)

        s = (b - a) / max(a, b) if max(a, b) > 0 else torch.tensor(0.0)
        scores.append(s)

    return torch.tensor(scores).mean().item()

# Image compression
def image_compression_kmeans(image, k=16):
    """Compress by quantizing colors using k-means."""
    H, W, _ = image.shape
    pixels = image.reshape(-1, 3).float()

    centroids = kmeans_plus_plus_init(pixels, k)
    final_centroids = k_means_torch(pixels, k, centroids, max_iterations=50)

    diff = pixels.unsqueeze(1) - final_centroids.unsqueeze(0)
    labels = torch.argmin((diff ** 2).sum(dim=2), dim=1)
    compressed_pixels = final_centroids[labels]

    return compressed_pixels.reshape(H, W, 3), final_centroids

# Customer segmentation
def customer_segmentation(features, k=5):
    """Segment customers into k groups based on features."""
    features_norm = (features - features.mean(dim=0)) / features.std(dim=0)

    centroids = kmeans_plus_plus_init(features_norm, k)
    final_centroids = k_means_torch(features_norm, k, centroids, max_iterations=100)

    diff = features_norm.unsqueeze(1) - final_centroids.unsqueeze(0)
    labels = torch.argmin((diff ** 2).sum(dim=2), dim=1)

    segment_profiles = []
    for i in range(k):
        segment_features = features[labels == i]
        segment_profiles.append({
            'size': (labels == i).sum().item(),
            'mean_features': segment_features.mean(dim=0),
            'std_features': segment_features.std(dim=0)
        })

    return labels, segment_profiles
```

**Best Practices:**
- Use K-means++ initialization to avoid poor local minima
- Always normalize features before clustering if they have different scales
- Use elbow method or silhouette score to determine optimal k
- For large datasets, consider mini-batch k-means for memory efficiency

**Common Applications:**
- Customer segmentation in marketing analytics
- Image compression via color quantization
- Document clustering (after TF-IDF vectorization)
- Anomaly detection (points far from all centroids)

---

# 4. Solve Linear Equations using Jacobi Method

## Problem Statement

Solve Ax = b using Jacobi iteration: x_i^(k+1) = (1/a_ii)(b_i - Σⱼ≠ᵢ a_ij x_j^(k))

## Mathematical Theory

**Jacobi Iteration:** Decompose A = D + L + U (diagonal, lower, upper triangular).
```
x_new = inv(D) @ (b - (L + U) @ x_old)
```

**Convergence Conditions:**
1. Strict diagonal dominance: |a_ii| > Σⱼ≠ᵢ |a_ij|
2. Spectral radius rho(inv(D) @ (L + U)) < 1

**Error bound:** ||x^(k) - x*|| <= rho^k * ||x^(0) - x*||

| Aspect | Details |
|--------|---------|
| Advantages | Simple, parallelizable, memory efficient |
| Disadvantages | Slow convergence, requires diagonal dominance |
| Best for | Large sparse systems, parallel computing |
| Complexity | O(n²i) for n variables and i iterations |

## Implementation

```python
import numpy as np

def solve_jacobi(A, b, n_iterations):
    """Jacobi method: x_i = (1/a_ii) * (b_i - sum(a_ij * x_j for j≠i))."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    aii = np.diag(A)
    x = np.zeros(len(b))

    for _ in range(n_iterations):
        new_x = np.array([
            (b[i] - sum(A[i][j] * x[j] for j in range(len(b)) if j != i)) / aii[i]
            for i in range(len(b))
        ])
        x = np.round(new_x, 4)

    return x.tolist()

# Vectorized
def solve_jacobi_vectorized(A, b, n_iterations):
    """Vectorized using NumPy operations."""
    d_a = np.diag(A)
    nda = A - np.diag(d_a)
    x = np.zeros(len(b))

    for _ in range(n_iterations):
        x = np.round((b - nda @ x) / d_a, 4)

    return x.tolist()

# With convergence check
def solve_jacobi_adaptive(A, b, n_iterations, tolerance=1e-4):
    """Early stopping on convergence."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    aii = np.diag(A)
    x = np.zeros(len(b))

    for iteration in range(n_iterations):
        new_x = np.array([(b[i] - sum(A[i][j] * x[j] for j in range(len(b)) if j != i)) / aii[i]
                         for i in range(len(b))])
        new_x = np.round(new_x, 4)

        if max(abs(new_x[i] - x[i]) for i in range(len(b))) < tolerance:
            return new_x.tolist(), iteration + 1
        x = new_x

    return x.tolist(), n_iterations
```

## PyTorch Integration

```python
import torch

def solve_jacobi_torch(A, b, n_iterations):
    """PyTorch Jacobi method."""
    A, b = A.float(), b.float()

    D = torch.diag(torch.diag(A))
    L_plus_U = A - D
    D_inv = torch.diag(1.0 / torch.diag(D))

    x = torch.zeros_like(b)
    for _ in range(n_iterations):
        x = D_inv @ (b - L_plus_U @ x)
        x = torch.round(x * 10000) / 10000

    return x

# Batch Jacobi (multiple systems)
def batch_jacobi_torch(A_batch, b_batch, n_iterations):
    """Solve (batch_size, m, m) @ x = (batch_size, m)."""
    x = torch.zeros_like(b_batch)

    for _ in range(n_iterations):
        D = torch.diag_embed(torch.diagonal(A_batch, dim1=1, dim2=2))
        L_plus_U = A_batch - D
        D_diag = torch.diagonal(A_batch, dim1=1, dim2=2)
        D_inv = torch.diag_embed(1.0 / D_diag)

        x = torch.bmm(D_inv, b_batch.unsqueeze(2) - torch.bmm(L_plus_U, x.unsqueeze(2))).squeeze(2)
        x = torch.round(x * 10000) / 10000

    return x

# Convergence analysis
def convergence_analysis(A, b, n_iterations, tolerance=1e-4):
    """Track convergence: error and residuals."""
    A, b = A.float(), b.float()

    D = torch.diag(torch.diag(A))
    L_plus_U = A - D
    D_inv = torch.diag(1.0 / torch.diag(D))

    x = torch.zeros_like(b)
    x_true = torch.linalg.solve(A, b)
    errors, residuals = [], []

    for _ in range(n_iterations):
        x = D_inv @ (b - L_plus_U @ x)
        x = torch.round(x * 10000) / 10000

        error = torch.norm(x - x_true).item()
        residual = torch.norm(A @ x - b).item()
        errors.append(error)
        residuals.append(residual)

        if error < tolerance:
            break

    return {'solution': x, 'errors': errors, 'residuals': residuals, 'convergence_iterations': len(errors)}

# Spectral radius (determines convergence rate)
def spectral_radius_jacobi(A):
    """rho(inv(D) @ (L+U)) < 1 implies convergence."""
    A = A.float()
    D = torch.diag(torch.diag(A))
    D_inv = torch.diag(1.0 / torch.diag(D))
    M = torch.eye(A.shape[0]) - D_inv @ A
    eigvals = torch.linalg.eigvals(M)
    rho = torch.max(torch.abs(eigvals)).item()

    return {
        'spectral_radius': rho,
        'is_convergent': rho < 1.0,
        'convergence_rate': -torch.log(torch.tensor(rho)) if rho < 1.0 else float('inf')
    }

# Compare with direct solver
def compare_jacobi_direct(A, b, n_iterations=20):
    x_direct = torch.linalg.solve(A, b)
    x_jacobi = solve_jacobi_torch(A, b, n_iterations)
    error = torch.norm(x_jacobi - x_direct).item()
    relative_error = error / torch.norm(x_direct).item()

    return {'jacobi': x_jacobi, 'direct': x_direct, 'absolute_error': error, 'relative_error': relative_error}

# Applications: Sparse systems, preconditioning
def jacobi_2d_poisson(f, n_iterations=100, h=1.0):
    """Solve 2D Poisson equation (d²u/dx² + d²u/dy²) = f using 5-point stencil."""
    n = f.shape[0]
    u = torch.zeros_like(f)

    for _ in range(n_iterations):
        u_new = u.clone()
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                # u[i,j] = (1/4)(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - h²*f[i,j])
                u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - h**2 * f[i, j])
        u = u_new

    return u
```

**Best Practices:**
- Verify diagonal dominance before assuming convergence
- Use vectorized implementations (NumPy/PyTorch) for large n, as loops are slow
- Check convergence via residual norm ||Ax - b|| rather than solution difference
- Consider Gauss-Seidel variant (uses updated values) for faster convergence

**Common Applications:**
- Large sparse systems (where direct factorization is expensive)
- PDE solving via finite difference discretization
- Image processing and diffusion simulations
- Iterative refinement as a preconditioner for Krylov subspace methods

---

# 5. Calculate Correlation Matrix

## Problem Statement

Compute the correlation matrix measuring linear relationships: Corr(X, Y) = Cov(X, Y) / (σₓ σᵧ)

## Mathematical Theory

**Pearson Correlation:**
```
r_XY = Σ(x_i - mean_x)(y_i - mean_y) / sqrt(Σ(x_i - mean_x)^2 * Σ(y_i - mean_y)^2)
```

**Correlation Matrix (p variables):** R_ij = Cov_ij / (sigma_i * sigma_j) or R = inv(D^0.5) @ Sigma @ inv(D^0.5)

**Properties:**
- Diagonal: r_ii = 1 (perfect with itself)
- Symmetric: r_ij = r_ji
- Bounded: -1 <= r_ij <= 1
- Positive semi-definite
- Scale-invariant

| Correlation | Relationship |
|---|---|
| r ≈ 1 | Strong positive linear |
| r ≈ 0.5 | Moderate positive |
| r ≈ 0 | No linear relationship |
| r ≈ -0.5 | Moderate negative |
| r ≈ -1 | Strong negative linear |

## Implementation

```python
import numpy as np

def calculate_correlation_matrix(X):
    """Normalize covariance by standard deviations."""
    X = np.array(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    cov_matrix = np.cov(X.T, rowvar=True)
    std_devs = np.sqrt(np.diag(cov_matrix))
    return cov_matrix / np.outer(std_devs, std_devs)

# Optimized using np.corrcoef
def correlation_optimized(X):
    return np.corrcoef(X.T if X.ndim > 1 else X.reshape(1, -1))

# With p-values for significance
def correlation_with_pvalues(X):
    """P-value indicates probability correlation is due to chance."""
    from scipy import stats

    X = np.array(X, dtype=float)
    n_features = X.shape[1]
    corr_matrix = np.corrcoef(X.T)
    pvalues = np.zeros((n_features, n_features))
    n = X.shape[0]

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                pvalues[i, j] = 0
            else:
                r = corr_matrix[i, j]
                t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2 + 1e-10)
                pvalues[i, j] = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

    return {'correlation': corr_matrix, 'pvalues': pvalues, 'significant_pairs': np.argwhere(pvalues < 0.05)}
```

## PyTorch Integration

```python
import torch

def calculate_correlation_torch(X):
    """Manual computation: R = Cov / (sigma_i * sigma_j)."""
    X = X.float()
    if X.dim() == 1:
        X = X.unsqueeze(1)

    n_samples = X.shape[0]
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    std_devs = torch.sqrt(torch.diag(cov_matrix))

    return cov_matrix / torch.outer(std_devs, std_devs)

# Built-in (PyTorch 1.10+)
corr = torch.corrcoef(X.T) if X.shape[0] <= X.shape[1] else torch.corrcoef(X)

# Batch correlation
def batch_correlation_torch(X_batch):
    """(batch, n_samples, n_features) -> (batch, n_features, n_features)."""
    batch_size, n_samples, n_features = X_batch.shape
    X_centered = X_batch - X_batch.mean(dim=1, keepdim=True)

    cov_batch = torch.bmm(X_centered.transpose(1, 2), X_centered) / (n_samples - 1)
    std_batch = torch.sqrt(torch.diagonal(cov_batch, dim1=1, dim2=2))
    std_outer = torch.bmm(std_batch.unsqueeze(2), std_batch.unsqueeze(1))

    return cov_batch / std_outer

# Numerically stable (handles near-zero std devs)
def stable_correlation(X, eps=1e-8):
    X = X.float()
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    std = torch.sqrt(torch.diag(cov) + eps)
    corr = cov / torch.outer(std, std)
    return torch.clamp(corr, -1.0, 1.0)

# Statistical significance (t-statistic)
def correlation_statistical_tests(X):
    """t = r * sqrt(n-2) / sqrt(1-r²)."""
    X = X.float()
    n = X.shape[0]
    corr = calculate_correlation_torch(X)
    t_stat = corr * torch.sqrt(torch.tensor(n - 2)) / torch.sqrt(1 - corr**2 + 1e-10)
    return {'correlation': corr, 't_statistic': t_stat}

# PCA from correlation (useful for different-scale variables)
def pca_from_correlation(X, n_components=None):
    X = X.float()
    corr = calculate_correlation_torch(X)
    eigvals, eigvecs = torch.linalg.eigh(corr)
    idx = torch.argsort(eigvals, descending=True)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    if n_components:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    return {'components': eigvecs, 'explained_variance': eigvals, 'correlation_matrix': corr}

# Partial correlation (conditioned on other variables)
def partial_correlation(X, control_vars=None):
    """r_ij.k = (r_ij - r_ik * r_jk) / sqrt((1 - r_ik²)(1 - r_jk²))."""
    X = X.float()
    corr = calculate_correlation_torch(X)

    if control_vars is None:
        return corr

    partial_corr = corr.clone()
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            if i not in control_vars and j not in control_vars:
                r_ij = corr[i, j]
                for k in control_vars:
                    r_ik, r_jk = corr[i, k], corr[j, k]
                    numerator = r_ij - r_ik * r_jk
                    denominator = torch.sqrt((1 - r_ik**2) * (1 - r_jk**2) + 1e-10)
                    partial_corr[i, j] = numerator / denominator
                    partial_corr[j, i] = partial_corr[i, j]

    return partial_corr

# Detect multicollinearity (high correlations)
def detect_multicollinearity(X, threshold=0.9):
    corr = calculate_correlation_torch(X.float())
    highly_correlated = []

    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr[i, j].item()) > threshold:
                highly_correlated.append({'features': (i, j), 'correlation': corr[i, j].item()})

    return highly_correlated

# Portfolio risk analysis
def portfolio_correlation_risk(returns, weights):
    """Analyze portfolio risk using correlation structure."""
    corr = calculate_correlation_torch(returns.T)
    returns_centered = returns - returns.mean(dim=1, keepdim=True)
    cov = (returns_centered @ returns_centered.T) / (returns.shape[1] - 1)

    portfolio_var = weights @ cov @ weights
    portfolio_std = torch.sqrt(portfolio_var)

    return {
        'correlation_matrix': corr,
        'portfolio_variance': portfolio_var.item(),
        'portfolio_volatility': portfolio_std.item()
    }

# Heatmap visualization
def plot_correlation_heatmap(X, title="Correlation Matrix"):
    import matplotlib.pyplot as plt

    corr = calculate_correlation_torch(X.float()).numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Variables')
    ax.set_title(title)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            color = "white" if abs(corr[i, j]) >= 0.5 else "black"
            ax.text(j, i, f'{corr[i, j]:.2f}', ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    return fig
```

**Best Practices:**
- Always standardize/normalize features before computing correlations when they have different units
- Use `torch.corrcoef()` built-in for efficiency; manually computing correlation is slower
- Add numerical stability term (eps) when dividing by standard deviations near zero
- For high-dimensional data, consider robust correlation measures (Spearman, Kendall) for outlier resistance

**Common Applications:**
- Feature selection (removing highly correlated predictors to reduce multicollinearity)
- PCA via correlation matrix for variables with different scales
- Portfolio analysis (correlations between asset returns)
- Anomaly detection (unusual correlation structures)

---
