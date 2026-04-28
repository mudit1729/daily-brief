# Tesla ML Engineer: Tracking & Classical CV Technical Screen

Generated: April 27, 2026

Purpose: a targeted prep guide for Tesla autonomy interviews that cover classical computer vision and tracking algorithms. Covers Kalman filter, multi-object tracking, optical flow, HOG, SIFT/ORB, image pyramids, and camera calibration. Each section includes math, NumPy code, and interview Q&A.

Research alignment from the technical-screen article:

- Candidate-reported Tesla CV/ML screens include CNN tensor arithmetic, 2D convolution, Kalman-style tracking, HOG/classical CV, and log-derived metrics such as `has_collided` and radial TTC.
- Classical CV is usually a depth check after coding or CNN fundamentals. Keep answers short, equation-backed, and tied to failure modes.
- Do not present exact Tesla internal stack details as facts. Use Tesla-flavored examples to show autonomy reasoning, but phrase implementation details as "production autonomy systems" unless sourced by the interviewer.

---

## Kalman Filter

The Kalman filter is the most important topic in this space. Production autonomy systems use Kalman-style filters to smooth detector outputs, propagate object positions between frames, and fuse noisy measurements.

### State-Space Model

Two equations define the filter.

**Process model** (how the state evolves over time):

$$x_t = F \cdot x_{t-1} + B \cdot u_t + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$

**Measurement model** (what the sensor observes):

$$z_t = H \cdot x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

```text
x_t   : state vector at time t       (e.g., [x, y, vx, vy])
F     : state transition matrix       (physics / constant velocity)
B, u  : control input (often omitted in CV)
w_t   : process noise, covariance Q   (model uncertainty)
z_t   : measurement vector            (detector output: [x, y])
H     : measurement matrix            (maps state to observation space)
v_t   : measurement noise, covariance R (detector uncertainty)
```

### Predict Step

Before a new measurement arrives, propagate the estimate forward:

$$\hat{x}_{t|t-1} = F \cdot \hat{x}_{t-1|t-1}$$

$$P_{t|t-1} = F \cdot P_{t-1|t-1} \cdot F^T + Q$$

```text
P     : state covariance matrix       tracks our uncertainty
Q     : process noise covariance      grows P (we trust the model less)
```

### Update Step

When a measurement z_t arrives, compute the Kalman gain and correct:

$$K_t = P_{t|t-1} \cdot H^T \cdot (H \cdot P_{t|t-1} \cdot H^T + R)^{-1}$$

$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \cdot (z_t - H \cdot \hat{x}_{t|t-1})$$

$$P_{t|t} = (I - K_t \cdot H) \cdot P_{t|t-1}$$

```text
K_t                     : Kalman gain — how much we trust the measurement
(z_t - H * x_hat)       : innovation (residual between prediction and measurement)
(I - K*H) * P           : posterior covariance — shrinks after a good measurement
```

Key insight: when R is small (sensor is accurate), K is large and we trust the measurement. When Q is small (motion model is accurate), K is small and we trust the prediction.

### 2D Constant-Velocity Tracking Example

State: `[x, y, vx, vy]` — 4D. Measurement: `[x, y]` — 2D (detector bounding box center).

```python
import numpy as np

dt = 1.0  # time step (one frame)

# State transition: x_new = x + vx*dt, y_new = y + vy*dt, vx/vy constant
F = np.array([
    [1, 0, dt,  0],   # x  = x  + vx*dt
    [0, 1,  0, dt],   # y  = y  + vy*dt
    [0, 0,  1,  0],   # vx = vx
    [0, 0,  0,  1],   # vy = vy
], dtype=np.float64)   # shape (4, 4)

# Measurement matrix: we observe x and y only
H = np.array([
    [1, 0, 0, 0],     # z[0] = x
    [0, 1, 0, 0],     # z[1] = y
], dtype=np.float64)   # shape (2, 4)
```

### NumPy 1D Kalman Filter

A minimal, readable 1D implementation (~30 lines) to demonstrate the algorithm clearly.

```python
import numpy as np

def kalman_filter_1d(measurements, F=1.0, H=1.0, Q=1e-4, R=0.1, x0=0.0, P0=1.0):
    """
    1D Kalman filter for scalar position tracking.

    Args:
        measurements : array of noisy observations, shape (T,)
        F            : scalar state transition (1.0 = constant position)
        H            : scalar measurement mapping (1.0 = observe state directly)
        Q            : process noise variance (model uncertainty)
        R            : measurement noise variance (sensor uncertainty)
        x0, P0       : initial state and covariance

    Returns:
        estimates    : filtered state estimates, shape (T,)
    """
    measurements = np.asarray(measurements, dtype=np.float64)  # coerce input
    T = len(measurements)
    estimates = np.zeros(T)                                     # output buffer

    x = x0   # initial state estimate (scalar)
    P = P0   # initial covariance (scalar)

    for t in range(T):
        # --- Predict ---
        x_pred = F * x                    # propagate state
        P_pred = F * P * F + Q           # propagate covariance, add process noise

        # --- Update ---
        S = H * P_pred * H + R           # innovation covariance (scalar)
        K = P_pred * H / S               # Kalman gain
        innovation = measurements[t] - H * x_pred  # residual
        x = x_pred + K * innovation      # posterior state
        P = (1 - K * H) * P_pred        # posterior covariance

        estimates[t] = x

    return estimates

# Quick sanity check
rng = np.random.default_rng(42)
true_pos = np.linspace(0, 10, 50)                    # true trajectory
noisy = true_pos + rng.normal(0, 0.5, size=50)       # add sensor noise
filtered = kalman_filter_1d(noisy, Q=1e-3, R=0.25)   # run filter
rmse_raw = np.sqrt(np.mean((noisy - true_pos) ** 2))
rmse_kf  = np.sqrt(np.mean((filtered - true_pos) ** 2))
assert rmse_kf < rmse_raw  # filter should reduce noise
```

### When Kalman Assumptions Break

The standard Kalman filter requires:
- Linear dynamics (F, H are constant matrices)
- Gaussian noise (w_t, v_t are Gaussian)

When either breaks:

| Problem | Example | Solution |
|---|---|---|
| Non-linear state transition | Car turning, projectile motion | **EKF** — linearize F around current estimate via Jacobian |
| Mildly non-linear measurement | Bearing-only tracking | **UKF** — propagate sigma points through the true non-linear function |
| Highly non-linear, multi-modal | Abrupt maneuvers, occluded objects | **Particle filter** — represent distribution as weighted samples |

**EKF (Extended Kalman Filter)**: replace F with its Jacobian $\nabla F$ evaluated at $\hat{x}_{t-1}$. Works well for smooth non-linearities.

**UKF (Unscented Kalman Filter)**: uses deterministic sigma points to capture mean and covariance through non-linear transforms. More accurate than EKF for moderate non-linearities, no Jacobian needed.

**Particle filter**: represent $p(x_t | z_{1:t})$ as N weighted particles. Resamples particles according to likelihood. Handles arbitrary distributions. Cost scales with N (typically 100–1000 particles). Used when multi-modality is critical.

### Tesla Connection

Detector outputs (bounding boxes, 3D centroids from BEV) are noisy frame-to-frame. The Kalman filter:
1. Predicts where each object will be next frame given constant-velocity motion.
2. Associates new detections to predictions (via IoU or distance).
3. Updates the smooth trajectory estimate.
4. Provides uncertainty estimate for downstream planners.

Even with neural detectors, Kalman smoothing reduces jitter and handles missed detections gracefully.

---

## Multi-Object Tracking (MOT)

### SORT — Simple Online and Realtime Tracking

SORT is the workhorse MOT algorithm. Four steps per frame:

```text
1. DETECT  — run object detector on frame t → N bounding boxes
2. PREDICT — Kalman-predict each existing track's box to time t
3. ASSOCIATE — match detections to predicted boxes via Hungarian algorithm on IoU
4. UPDATE  — update matched tracks; create new tracks for unmatched detections;
              delete tracks lost for > max_age frames
```

Each track carries its own Kalman filter with state `[u, v, s, r, u_dot, v_dot, s_dot]` where `(u,v)` is box center, `s` is area, `r` is aspect ratio.

### DeepSORT Extension

DeepSORT adds an appearance embedding from a Re-ID CNN to SORT's motion model.

```text
Matching cost = alpha * (1 - IoU) + (1-alpha) * appearance_distance
```

Appearance distance is cosine distance between the detection's embedding and a gallery of recent embeddings for each track. This dramatically reduces **identity switches** when objects are occluded or cross paths.

### IoU-Only Tracking (Simplest Baseline)

```python
import numpy as np

def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])  # intersection left
    ya = max(box_a[1], box_b[1])  # intersection top
    xb = min(box_a[2], box_b[2])  # intersection right
    yb = min(box_a[3], box_b[3])  # intersection bottom

    inter = max(0, xb - xa) * max(0, yb - ya)  # intersection area
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])  # area of A
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # area of B
    union = area_a + area_b - inter               # union area
    return inter / (union + 1e-9)                 # IoU in [0, 1]

def iou_matrix(detections, tracks):
    """
    Compute IoU between all detection-track pairs.

    Args:
        detections : list of N boxes [x1, y1, x2, y2]
        tracks     : list of M boxes [x1, y1, x2, y2]

    Returns:
        iou        : array of shape (N, M)
    """
    N, M = len(detections), len(tracks)
    iou = np.zeros((N, M), dtype=np.float64)   # shape (N, M)
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracks):
            iou[i, j] = compute_iou(det, trk)
    return iou

def greedy_iou_assign(detections, tracks, iou_thresh=0.3):
    """
    Greedy (not optimal) assignment: match highest-IoU pairs first.
    Returns list of (det_idx, trk_idx) matched pairs.
    """
    cost = iou_matrix(detections, tracks)    # shape (N, M)
    matched = []
    used_det = set()
    used_trk = set()

    # Flatten and sort by IoU descending
    order = np.argsort(-cost.ravel())        # shape (N*M,)
    for idx in order:
        i, j = divmod(int(idx), len(tracks))
        if cost[i, j] < iou_thresh:
            break
        if i not in used_det and j not in used_trk:
            matched.append((i, j))
            used_det.add(i)
            used_trk.add(j)
    return matched
```

Note: greedy assignment is O(NM log(NM)) but suboptimal. SORT uses the Hungarian algorithm for global optimality.

### Hungarian Algorithm Intuition

The Hungarian algorithm solves the **linear assignment problem**: given a cost matrix C of shape (N, M), find a one-to-one matching that minimizes total cost.

```text
Complexity: O(n^3) with n = max(N, M)
In SORT: cost = 1 - IoU  (we minimize cost = we maximize IoU)
scipy.optimize.linear_sum_assignment(cost) returns (row_ind, col_ind)
```

For N=M=50 objects at 30 fps, O(n^3) = O(125,000) ops per frame — fast enough. For 500+ objects, approximate methods (auction algorithm, greedy) become preferable.

### Mahalanobis Distance Gating

Before assigning a detection to a track, check that the detection lies inside the Kalman filter's predicted uncertainty ellipsoid. This is the squared Mahalanobis distance between the innovation and zero:

$$d^2 = (z - H \hat{x})^T \, S^{-1} \, (z - H \hat{x}), \quad S = H P H^T + R$$

```text
S      : innovation covariance (already computed in the Kalman update)
d^2    : chi-squared distributed with k degrees of freedom (k = dim(z))
Gate   : reject association if d^2 > chi2_inv(0.95, k)
         For 2D measurement (k=2): threshold ≈ 5.99
         For 4D measurement (k=4): threshold ≈ 9.49
```

Mahalanobis is the right distance (not Euclidean) because it accounts for the fact that the filter is more uncertain about velocity than position, and uncertainty grows differently along different axes. DeepSORT combines a Mahalanobis motion gate with a cosine appearance gate.

### Track Lifecycle (Birth / Confirmation / Death)

A real MOT system does not promote every detection to a permanent track. Standard lifecycle:

```text
Tentative → Confirmed → Lost → Deleted

- Birth      : an unmatched detection spawns a Tentative track.
- Confirmed  : after N consecutive matches (typically N=3), promote to Confirmed.
- Coasting   : if a Confirmed track misses a frame, Kalman-predict only (no update),
               increment miss counter.
- Lost       : after max_age frames without a match (typically 30 at 30 fps = 1 s),
               mark as Lost.
- Deletion   : Lost tracks past the re-ID window are deleted.
- Tentative tracks die immediately on first miss (prevents one-frame-detector-noise
  from creating ghost tracks).
```

This hysteresis prevents flicker: a one-frame detector blip never produces a track, and a one-frame missed detection never kills a real track.

### MOT Metrics

**MOTA (Multiple Object Tracking Accuracy)**:

$$\text{MOTA} = 1 - \frac{\sum_t (\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}$$

Can be negative when errors exceed ground-truth count. Most intuitive metric.

**IDF1 (ID F1 Score)**:

$$\text{IDF1} = \frac{2 \cdot \text{IDTP}}{2 \cdot \text{IDTP} + \text{IDFP} + \text{IDFN}}$$

Measures how long a tracker maintains correct identities. Better at capturing Re-ID quality than MOTA.

**Identity Switch (IDSW)**: an existing track is assigned the wrong ID — typically happens on occlusion, crossing paths, or detector failure.

### ByteTrack

ByteTrack (2022) improves on SORT by using **both high- and low-confidence detections**:

```text
Step 1: Associate high-confidence detections (score > 0.5) to tracks via IoU.
Step 2: Associate low-confidence detections to unmatched tracks from step 1.
Step 3: Initialize new tracks from remaining unmatched high-conf detections.
```

Low-confidence detections often correspond to partially occluded objects. Using them prevents track loss during occlusion without adding noisy false positives. ByteTrack achieves state-of-the-art MOTA on MOT17/MOT20.

---

## Optical Flow Basics

Optical flow estimates per-pixel motion between consecutive frames — how much each pixel moved.

### Brightness Constancy Assumption

$$I(x, y, t) = I(x + dx, \; y + dy, \; t + dt)$$

Expanding with a Taylor series and dividing by dt:

$$I_x \cdot u + I_y \cdot v + I_t = 0$$

```text
I_x, I_y : spatial image gradients
I_t       : temporal gradient (frame difference)
u, v      : unknown flow at pixel (x, y)
```

This is one equation with two unknowns — the **aperture problem**: you cannot determine flow from a single pixel; you need a neighbourhood or global constraint.

### Lucas-Kanade (Sparse, Local)

Assumes flow is constant within a small window W. Stacks the constraint equation for all pixels in W into a least-squares system:

$$\begin{bmatrix} u \\ v \end{bmatrix} = (A^T A)^{-1} A^T b$$

where A contains `[I_x, I_y]` rows for each pixel in W, and b contains `-I_t`. Solves well when the structure tensor $A^T A$ is well-conditioned (i.e., the window has texture in both directions — corners work, edges do not).

Typically applied at keypoints (Harris corners) and run in a pyramid for large motions.

### Horn-Schunck (Dense, Global)

Adds a global smoothness regularizer over the entire image:

$$E = \iint (I_x u + I_y v + I_t)^2 \, dx \, dy + \lambda \iint (|\nabla u|^2 + |\nabla v|^2) \, dx \, dy$$

Minimized via iterative updates. Produces dense flow but over-smooths at motion boundaries. Sensitive to large displacements.

### Aperture Problem Intuition

Looking at a moving edge through a small aperture, you can only measure the component of motion perpendicular to the edge. Motion parallel to the edge is invisible. You need a corner or texture in multiple orientations to recover full 2D flow.

### Modern Methods

- **FlowNet / FlowNet2** (2015-2017): first CNNs trained end-to-end on synthetic optical flow data. Used correlation layers.
- **RAFT** (2020): recurrent architecture with 4D cost volumes. State-of-the-art on Sintel and KITTI. Key idea: iteratively update flow estimates at full resolution using GRUs.

### Use Cases in Autonomy

```text
Ego-motion estimation : integrate flow over the full frame to estimate camera translation/rotation
Dense scene flow      : per-point 3D motion in camera frame; used for moving object segmentation
Stereo                : horizontal flow between rectified stereo pair → disparity → depth
Long-range tracking   : flow as a prior for feature correspondence across frames
```

---

## HOG (Histogram of Oriented Gradients)

HOG was the dominant pedestrian detection feature before deep learning. Still conceptually important.

### Algorithm

**Step 1 — Gradient computation**

Dalal & Triggs found the simple 1-D centred-difference kernel works best for HOG (Sobel actually performs slightly worse here):

$$G_x = \begin{bmatrix} -1 & 0 & +1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 \\ 0 \\ +1 \end{bmatrix}$$

For reference, the 3×3 Sobel kernels (used in many other CV pipelines) are:

$$S_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

Magnitude and orientation:

$$m = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan(G_y / G_x)$$

```python
import numpy as np

def compute_gradients(img):
    """
    Compute gradient magnitude and orientation.

    Args:
        img : grayscale image, shape (H, W), dtype float32

    Returns:
        mag   : gradient magnitude, shape (H, W)
        theta : orientation in [0, 180) degrees, shape (H, W)
    """
    img = img.astype(np.float32)
    Gx = np.zeros_like(img)             # horizontal gradient
    Gy = np.zeros_like(img)             # vertical gradient

    # Sobel-like via finite difference on interior pixels
    Gx[:, 1:-1] = img[:, 2:] - img[:, :-2]    # central diff in x
    Gy[1:-1, :] = img[2:, :] - img[:-2, :]    # central diff in y

    mag   = np.sqrt(Gx**2 + Gy**2)            # magnitude, shape (H, W)
    theta = np.degrees(np.arctan2(Gy, Gx))    # orientation, shape (H, W)
    theta = theta % 180                        # unsigned, [0, 180)
    return mag, theta
```

**Step 2 — Cell histogram**

Divide image into cells (e.g., 8x8 pixels). For each cell, vote magnitude-weighted gradient orientations into 9 bins (0–180° in 20° increments).

```python
def cell_histogram(mag_cell, theta_cell, n_bins=9):
    """
    Build HOG histogram for one cell.

    Args:
        mag_cell   : magnitude values in cell, shape (8, 8)
        theta_cell : orientation values in cell, shape (8, 8)  [0, 180)
        n_bins     : number of orientation bins

    Returns:
        hist       : histogram, shape (n_bins,)
    """
    bin_width = 180.0 / n_bins                         # 20 degrees per bin
    bin_idx = (theta_cell / bin_width).astype(int)     # shape (8, 8)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)          # guard edge case
    hist = np.zeros(n_bins, dtype=np.float32)          # output histogram
    np.add.at(hist, bin_idx.ravel(), mag_cell.ravel()) # magnitude voting
    return hist  # shape (n_bins,)
```

**Step 3 — Block normalization**

Group cells into overlapping 2x2 blocks. Concatenate histograms from 4 cells → 36-d vector. Normalize with L2-Hys (L2-norm, clip to 0.2, renormalize):

```text
Block descriptor = [hist_c0, hist_c1, hist_c2, hist_c3]  shape (36,)
v_norm           = v / (||v||_2 + eps)
v_clipped        = clip(v_norm, 0, 0.2)
v_final          = v_clipped / (||v_clipped||_2 + eps)
```

**Step 4 — Final descriptor**

Slide the block window over all cell positions in the detection window (64x128 for pedestrians). Concatenate all block descriptors. Standard result: 3780-d vector for 64x128 window.

Feed into a linear SVM for classification (Dalal & Triggs, CVPR 2005).

### Why HOG Worked

Gradients are largely invariant to small illumination changes. The orientation histogram is robust to precise localization within a cell. Block normalization handles local contrast variation. The pedestrian silhouette produces consistent gradient patterns that HOG captures well.

### HOG vs Learned Conv Features

| Property | HOG | Conv Features |
|---|---|---|
| Learning | Hand-designed | Learned from data |
| Invariance | Contrast, small shifts | Learned via augmentation |
| Compute | Fast (vectorized) | GPU-heavy |
| Interpretability | High | Low |
| Performance (now) | Outdated | State of the art |

---

## SIFT and ORB

### SIFT (Scale-Invariant Feature Transform)

SIFT detects and describes keypoints that are invariant to scale and rotation.

```text
1. Scale-space extrema    : build Gaussian pyramid; detect DoG (Difference of Gaussians)
                             extrema across scale and space
2. Keypoint localization  : reject low-contrast and edge-response points using the
                             Hessian eigenvalue ratio
3. Orientation assignment : compute dominant gradient orientation in neighborhood;
                             make descriptor rotation-invariant
4. Descriptor             : 16x16 neighborhood split into 4x4 sub-cells; 8-bin histogram
                             per sub-cell → 128-d L2-normalized vector
```

SIFT descriptors are highly distinctive and robust to affine distortion, blur, and illumination. Patented (expired 2020). Still the gold standard for image registration.

### ORB (Oriented FAST and Rotated BRIEF)

ORB is a free, fast alternative designed for real-time use.

```text
Keypoints : FAST corner detector + Harris score to select top N
Orientation: intensity centroid method for rotation invariance
Descriptor : rBRIEF (steered BRIEF) — binary 256-bit descriptor (compare pixel pairs)
Matching   : Hamming distance (fast on hardware with XOR popcount)
```

ORB is ~100x faster than SIFT. Descriptors are binary, so matching is extremely fast. Less distinctive than SIFT for large viewpoint changes.

### When to Use These

```text
SLAM                : ORB-SLAM uses ORB for real-time feature-based SLAM
Image registration  : SIFT for medical imaging, panorama stitching, satellite alignment
Loop closure        : bag-of-visual-words with SIFT or ORB descriptors
Camera calibration  : checkerboard corner matching across views
Classical pipelines : anywhere you need correspondence without a GPU
```

Modern deep-learning alternatives: SuperPoint (learned keypoints + descriptors), LoFTR (transformer-based dense matching).

---

## Image Pyramids and Multi-Scale Detection

### Gaussian Pyramid

Repeatedly apply a Gaussian blur and subsample by 2x:

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def gaussian_pyramid(img, levels=4):
    """
    Build a Gaussian pyramid.

    Args:
        img    : input image, shape (H, W) or (H, W, C)
        levels : number of pyramid levels

    Returns:
        pyramid: list of arrays, each half the spatial resolution of the previous
    """
    pyramid = [img.astype(np.float32)]     # level 0 = original
    for _ in range(levels - 1):
        blurred = gaussian_filter(pyramid[-1], sigma=1.0)  # anti-aliasing
        # Subsample by 2 in spatial dims
        if blurred.ndim == 2:
            downsampled = blurred[::2, ::2]               # shape (H/2, W/2)
        else:
            downsampled = blurred[::2, ::2, :]            # shape (H/2, W/2, C)
        pyramid.append(downsampled)
    return pyramid  # list of length `levels`
```

### Laplacian Pyramid

Stores the residual detail lost at each downsample step. Used for image blending and compression.

```text
L_i = G_i - upsample(G_{i+1})
```

Where `upsample` expands the smaller image back to full resolution. The Laplacian pyramid is perfectly reconstructible: sum all levels to recover the original.

### Why Scale Matters for Detection

Objects appear at different sizes in the image depending on distance. A pedestrian at 5 m might span 200 px; at 50 m it spans 20 px. A fixed-size detector or filter works only on one scale. Solutions:

```text
Image pyramid         : run detector at each pyramid level (scale the image, fixed detector)
Feature pyramid       : build multi-scale features inside the network (FPN, BiFPN)
Anchor-based          : use anchors of different sizes at each location
Anchor-free           : predict scale directly (FCOS, CenterPoint)
```

BEV detection reduces the image-scale problem by projecting features into a fixed bird's-eye-view grid where scale is more controlled, but camera-based detectors still need multi-scale handling before or during the BEV transform.

---

## Camera Calibration

### Intrinsic Parameters

The intrinsic matrix K maps 3D camera-frame coordinates to 2D pixel coordinates:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

```text
f_x, f_y : focal lengths in pixels (f_x = f / pixel_width_in_mm)
c_x, c_y : principal point (optical axis intersection with image plane, usually near center)
```

Distortion coefficients model lens imperfections:

```text
Radial   : k1, k2, (k3)   — barrel or pincushion distortion
Tangential: p1, p2         — lens not perfectly parallel to image sensor
```

### Extrinsic Parameters

The extrinsic transform (R, t) maps world coordinates to camera coordinates:

$$X_{cam} = R \cdot X_{world} + t$$

R is a 3x3 rotation matrix; t is a 3x1 translation vector.

### Full Projection Pipeline

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \cdot [R \;|\; t] \cdot \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

```text
X_world : 4D homogeneous world point                  shape (4,)
[R|t]   : 3x4 extrinsic matrix                       shape (3, 4)
K       : 3x3 intrinsic matrix                        shape (3, 3)
Result  : homogeneous image coords → divide by z to get (u, v)
```

```python
import numpy as np

def project_points(X_world, K, R, t):
    """
    Project 3D world points to image pixels.

    Args:
        X_world : world points, shape (N, 3)
        K       : intrinsic matrix, shape (3, 3)
        R       : rotation matrix, shape (3, 3)
        t       : translation vector, shape (3,)

    Returns:
        uv      : pixel coordinates, shape (N, 2)
    """
    # Transform to camera frame
    X_cam = (R @ X_world.T).T + t           # shape (N, 3)

    # Perspective divide
    X_norm = X_cam / X_cam[:, 2:3]         # normalize by depth, shape (N, 3)

    # Apply intrinsics
    uvw = (K @ X_norm.T).T                 # shape (N, 3)
    return uvw[:, :2]                      # return (u, v), shape (N, 2)
```

### Stereo and Depth from Disparity

Given a calibrated stereo pair with baseline B (meters between cameras) and shared focal length f (pixels):

$$Z = \frac{f \cdot B}{d}$$

```text
Z   : depth in meters
f   : focal length in pixels
B   : stereo baseline in meters (e.g., 0.12 m for a typical stereo rig)
d   : disparity in pixels (horizontal pixel difference for the same 3D point)
```

Disparity is large for close objects and small for distant objects. At 50 m with f=1000 px and B=0.12 m, d = 1000 * 0.12 / 50 = 2.4 px — very small; noise in d causes large depth uncertainty at range.

### Why Calibration Drift Matters in Autonomy

```text
Lane misalignment   : a 0.5° pitch error at 40 m projects camera-detected lane to wrong ground plane
Depth error         : intrinsic shift of 2 px changes depth estimates by several percent at range
BEV fusion error    : camera features projected into a shared world/BEV frame become misaligned
Aux sensor fusion   : any auxiliary sensor projection through [R|t] becomes misregistered
```

A production camera stack should treat calibration as uncertain. Extrinsics can drift from vibration, thermal changes, or service events, so the system should either re-estimate calibration online or propagate calibration uncertainty through the tracking pipeline.

---

## Top Asked Questions

### 1. Walk me through the Kalman filter predict and update steps.

Question:

```text
Explain the Kalman filter from scratch. What equations govern each step?
```

Answer:

```text
Predict:
  x_pred = F * x_prev         — propagate state through motion model
  P_pred = F * P_prev * F^T + Q  — propagate covariance, grow by process noise Q

Update (when measurement z arrives):
  K = P_pred * H^T * (H * P_pred * H^T + R)^{-1}  — Kalman gain
  x = x_pred + K * (z - H * x_pred)               — correct with innovation
  P = (I - K * H) * P_pred                         — shrink covariance

Key intuition: K interpolates between our prediction and the measurement.
When R is small (trusted sensor), K is large: lean on measurement.
When Q is small (trusted model), K is small: lean on prediction.
```

Follow-up: what is the innovation? The innovation `z - H*x_pred` is the surprise — how far the measurement deviates from what we expected. A large innovation suggests either a bad prediction or an outlier measurement (gating: reject if innovation exceeds chi-squared threshold).

### 2. How would you track 50 objects across 30 fps video?

Answer:

```text
Use SORT or a similar Kalman + Hungarian algorithm pipeline.

Per-frame (33 ms budget, must finish in < ~5 ms):
1. Run detector to get N new detections (bounding boxes).
2. Kalman-predict all M existing track boxes to current timestamp.
3. Compute IoU matrix (N x M). Use Hungarian algorithm to solve assignment.
4. Update matched tracks' Kalman filters with new detections.
5. Increment miss count for unmatched tracks; delete if miss > max_age.
6. Create new tracks for unmatched detections.

At 50 objects, Hungarian is O(50^3) = 125K ops — trivially fast.
For a real system, add appearance re-ID (DeepSORT) if identity switches are costly.
```

### 3. When does the Kalman filter fail and what do you switch to?

Answer:

```text
Failures:
- Non-linear dynamics: car making a U-turn, spinning objects → EKF (linearize) or UKF (sigma points)
- Non-Gaussian noise: heavy-tailed sensor noise, clutter → particle filter or robust M-estimators
- Multi-modal distribution: object could be in one of two places after occlusion → particle filter
- Highly occluded scenes: the measurement model H*x breaks if the sensor can't see the object at all
  → handle with gating + track coasting

Switch to:
- EKF for mildly non-linear motion (standard in robotics)
- UKF for moderate non-linearity without a Jacobian
- Particle filter for highly non-linear, multi-modal, or complex likelihood scenarios
```

### 4. Explain HOG features.

Answer:

```text
HOG computes a descriptor for an image patch by:
1. Compute gradient magnitude and orientation at every pixel.
2. Divide patch into small cells (8x8 px). Bin gradients by orientation (9 bins, 0–180°),
   weighted by magnitude. Result: one 9-d histogram per cell.
3. Group cells into overlapping blocks (2x2 cells). Normalize each block's 36-d vector
   with L2-Hys (clip at 0.2, re-normalize) to handle contrast variation.
4. Concatenate all block descriptors into the final vector (~3780-d for 64x128).

Feed into a linear SVM. Captures human silhouette structure robustly.
Limitation: not rotation-invariant; expensive at test time compared to a CNN layer.
```

### 5. How does SORT work and what does DeepSORT add?

Answer:

```text
SORT: detect → Kalman-predict tracks → Hungarian-assign on IoU → update matched tracks
DeepSORT adds:
- A Re-ID CNN extracts a 128-d appearance embedding for each detection.
- Association uses a combined cost: motion (Mahalanobis distance) + appearance (cosine distance).
- A gallery stores the last k embeddings per track.
- Appearance matching allows re-identification after long occlusions that break IoU association.
Result: far fewer identity switches in crowded scenes. Cost: one extra CNN forward pass per detection.
```

### 6. What's the difference between dense and sparse optical flow?

Answer:

```text
Sparse: compute flow at a small set of keypoints (e.g., Harris corners).
  - Lucas-Kanade is the canonical method.
  - Fast, robust, used for feature tracking, ego-motion estimation.
  - Fails at textureless regions (nothing to track).

Dense: compute flow at every pixel.
  - Horn-Schunck, FlowNet, RAFT.
  - Needed for scene flow, video segmentation, structure-from-motion.
  - Much slower; RAFT needs a GPU.

Tesla use: sparse flow is adequate for ego-motion from camera; dense scene flow
is used for moving-object segmentation from monocular or stereo video.
```

### 7. How do you handle identity switches in MOT?

Answer:

```text
Prevention:
- Add appearance Re-ID embedding (DeepSORT): match on look not just position.
- Use Mahalanobis-gated association: only associate if prediction uncertainty is small.
- Track coasting: keep track alive for N frames without a detection (don't immediately delete).
- ByteTrack: use low-confidence detections to re-link tracks lost during occlusion.

Recovery:
- Re-ID gallery: compare lost track's last embedding to new detections after it reappears.
- Track merging: if two tracks overlap significantly, merge (occlusion handling).

Measurement:
- IDSW count in MOTA, or IDF1 score (measures long-term ID consistency).
```

### 8. Why is camera calibration critical for autonomy?

Answer:

```text
Every pixel-to-world mapping depends on calibration. Errors compound:
- Lane detection: a small principal-point or extrinsic error misprojects lane lines on the ground plane.
- Depth from stereo: depth Z = f*B/d; error in focal length, baseline, or disparity creates depth error.
- Multi-camera fusion: drift misaligns features projected into a shared BEV/world frame.
- Generic robotics sensor fusion: any auxiliary sensor projection through [R|t] will misregister if calibration drifts.
- Planner safety margins: incorrect depth or position -> incorrect following distance -> safety violation.

Production systems re-estimate calibration online (vehicle-motion-based extrinsic calibration)
and propagate calibration uncertainty into downstream confidence estimates.
```

---

### 9. Given a CNN layer, compute output shape and learnable parameters.

Question:

```text
Input: [B, 3, 224, 224].
Layer: Conv2d(C_in=3, C_out=64, kernel=7, stride=2, padding=3, dilation=1, bias=True).
What is the output shape and parameter count?
```

Answer:

```text
H_out = floor((H + 2P - D*(K - 1) - 1) / S + 1)
      = floor((224 + 6 - 1*(7 - 1) - 1) / 2 + 1)
      = 112

W_out = 112
Output shape = [B, 64, 112, 112]

Params = C_out * (C_in/groups) * K_h * K_w + C_out
       = 64 * 3 * 7 * 7 + 64
       = 9472
```

Follow-ups:

- Pooling has no learnable parameters.
- BatchNorm has two learnable parameters per channel: gamma and beta. Running mean/var are buffers, not learned parameters.
- Grouped convolution uses `C_in / groups` channels per filter.

### 10. Code a valid 2D convolution/cross-correlation.

Question:

```text
Implement valid 2D filtering for one grayscale image and one kernel.
```

Answer:

```python
from typing import List


def conv2d_valid(image: List[List[float]], kernel: List[List[float]]) -> List[List[float]]:
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    out_h, out_w = h - kh + 1, w - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("kernel must fit inside image")

    out = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
    for i in range(out_h):
        for j in range(out_w):
            acc = 0.0
            for r in range(kh):
                for c in range(kw):
                    acc += image[i + r][j + c] * kernel[r][c]
            out[i][j] = acc
    return out
```

Interview note: most deep-learning libraries implement cross-correlation, not flipped-kernel mathematical convolution. Clarify if the interviewer wants the kernel flipped.

### 11. Compute radial TTC from positions and velocities.

Question:

```text
Given ego/object positions and velocities, compute a fast time-to-collision screening metric.
```

Answer:

```python
from math import inf, sqrt
from typing import Tuple


def radial_ttc(
    ego_pos: Tuple[float, float],
    ego_vel: Tuple[float, float],
    obj_pos: Tuple[float, float],
    obj_vel: Tuple[float, float],
    collision_radius: float = 0.0,
) -> float:
    rx = obj_pos[0] - ego_pos[0]
    ry = obj_pos[1] - ego_pos[1]
    rvx = obj_vel[0] - ego_vel[0]
    rvy = obj_vel[1] - ego_vel[1]

    dist = sqrt(rx * rx + ry * ry)
    if dist <= collision_radius:
        return 0.0
    if dist == 0.0:
        return 0.0

    closing_speed = -(rx * rvx + ry * rvy) / dist
    if closing_speed <= 0.0:
        return inf
    return (dist - collision_radius) / closing_speed
```

Follow-up: radial TTC assumes constant relative velocity and reduces geometry to a radial closing metric. For production-grade collision labels, compare oriented boxes or swept volumes and slice failures by scenario.

---

## Common Pitfalls

- **Assuming constant-velocity motion**: objects accelerate, brake, and turn. A CV model's error grows quickly during maneuvers; keep Q large enough or switch to an adaptive model (IMMKF — Interacting Multiple Models).

- **Fixing R too low**: setting measurement noise R too small makes the filter over-trust detections, amplifying detector noise frame-to-frame. Tune R to match actual detector variance, not theoretical sensor specs.

- **IoU association breaks on occlusion**: when two objects overlap, IoU drops and the track can be lost or swapped. Mitigation: use appearance embeddings, maintain a separate occlusion state, or run ByteTrack-style two-stage association.

- **HOG is not rotation-invariant**: the standard HOG descriptor changes if the object rotates. Rotating objects (turning pedestrians, vehicles at sharp angles) require either training on augmented data or using SIFT/ORB descriptors that explicitly handle rotation.

- **Forgetting to normalize block descriptors**: raw cell histograms are contrast-dependent; without L2-Hys block normalization, HOG performance degrades under lighting changes.

- **Ignoring calibration drift**: assuming static calibration parameters after manufacturing. In production, thermal expansion, vibration, and mechanical wear shift intrinsics and extrinsics; online re-calibration is required.

- **Hungarian algorithm on the wrong cost sign**: IoU is a similarity (higher is better); the Hungarian algorithm minimizes cost. Passing IoU directly instead of `1 - IoU` or `-IoU` will produce the worst assignment instead of the best.

- **Particle filter degeneracy**: without resampling, most particles accumulate zero weight after a few steps. Always resample when effective sample size $N_{eff} = 1 / \sum w_i^2$ drops below N/2.

---

## Tesla Connection

In autonomy stacks, classical tracking algorithms are not a relic; they remain useful production tools around learned perception. A simplified architecture looks like this:

```text
Cameras / perception sensors
       |
   Neural detector or BEV network -> object detections per frame
       |
   Tracker          (Kalman filter per object, Hungarian/ByteTrack association)
       |
   Smoothed tracks  (position, velocity, uncertainty for each object)
       |
   Planner          (trajectory optimization using track predictions)
```

Even though detectors are deep neural networks, Kalman-style filtering can remain useful immediately after detection for several reasons. First, detectors are noisy: they miss objects for 1-3 frames during occlusion, and their boxes jitter frame-to-frame. A filter smooths this jitter and coasts through missed detections using motion assumptions. Second, the planner needs velocity and uncertainty, not just position. Third, the covariance matrix `P` gives a usable uncertainty estimate for gating and risk-aware downstream logic. Classical CV knowledge - HOG for quick prototyping, optical flow for ego-motion sanity checks, camera calibration for the projection pipeline, multi-scale pyramids for BEV feature alignment - remains useful because teams still debug geometry, tracking, and data quality around neural models.

---

## Quick Reference Cheatsheet

```text
Kalman predict : x_pred = F*x;  P_pred = F*P*F' + Q
Kalman update  : K = P_pred*H'*inv(H*P_pred*H'+R);  x = x_pred + K*(z - H*x_pred);  P = (I-K*H)*P_pred
SORT           : detect → Kalman predict → Hungarian (1-IoU) → update / birth / death
DeepSORT add   : appearance embedding, cosine distance in assignment cost
ByteTrack add  : two-stage matching with low-confidence detections
HOG            : gradient → cell histograms (9 bins) → block L2-Hys normalization → SVM
SIFT           : DoG extrema → orientation → 128-d histogram descriptor (scale+rotation invariant)
ORB            : FAST + Harris score → rBRIEF binary descriptor (fast, free)
Optical flow   : LK sparse (local LS), HS dense (global smoothness), RAFT deep
Projection     : uv ~ K * [R|t] * X_world
Stereo depth   : Z = f * B / d
```
