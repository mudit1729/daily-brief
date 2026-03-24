# Computer Vision Cheatsheet — Scale Robotics Interview

> Dense reference for last-minute review. Covers projective geometry, epipolar geometry, 3D reconstruction, SLAM, detection, tracking, and more.

---

## 0. HIGH-YIELD INTERVIEW ANSWERS

### If Asked To Compare Things Quickly
```
F vs E vs H:
- F: uncalibrated cameras, pixel coordinates, general 3D scene
- E: calibrated cameras, normalized coordinates, E = [t]ₓR
- H: direct plane-to-plane mapping; valid for planar scenes or pure camera rotation

Triangulation vs PnP vs Bundle Adjustment:
- Triangulation: known camera poses + 2D matches → recover 3D point
- PnP: known 3D points + 2D observations → recover camera pose
- Bundle Adjustment: jointly refine camera poses + 3D points

Monocular vs Stereo vs RGB-D:
- Monocular: cheapest, widest deployment, scale ambiguous
- Stereo: metric depth from disparity, fails on textureless/reflective regions
- RGB-D: direct depth, but limited range / sunlight issues / missing depth

RANSAC in one line:
- Sample minimal set → fit model → count inliers → keep best → refit on all inliers
```

### Good Short Explanations
- If the scene is mostly planar, estimate a homography before reaching for `F` or `E`.
- Pure rotation gives a valid homography but makes translation/depth recovery ill-conditioned.
- Small baseline hurts triangulation because rays intersect at shallow angles.
- Always validate geometry visually: epipolar lines, reprojection error, track consistency.

---

## 1. CAMERA MODEL & CALIBRATION

### Pinhole Camera Model
```
             [fx  0  cx]
K (intrinsic) = [ 0  fy  cy]    (3x3 upper triangular)
             [ 0   0   1]

fx, fy = focal length in pixels (fx = f/px, fy = f/py)
cx, cy = principal point (usually ≈ image center)
```

### Full Projection (World → Pixel)
```
s * [u, v, 1]ᵀ = K [R | t] [X, Y, Z, 1]ᵀ

s = scale factor (depth)
[R|t] = extrinsic matrix (3x4), R = rotation, t = translation
P = K[R|t] = projection matrix (3x4)
```

### Lens Distortion
```
Normalized image coords: (x, y),  r² = x² + y²

Radial:
  x_rad = x(1 + k1*r² + k2*r⁴ + k3*r⁶)
  y_rad = y(1 + k1*r² + k2*r⁴ + k3*r⁶)

Tangential:
  x' = x_rad + 2*p1*x*y + p2*(r² + 2x²)
  y' = y_rad + p1*(r² + 2y²) + 2*p2*x*y

Distortion coefficients: (k1, k2, p1, p2, k3)
```

### Calibration
- Use checkerboard/charuco board
- Need ≥ 10-15 images at varied angles
- Cover image center + corners, and vary distance to board
- `cv2.calibrateCamera()` → returns K, distortion coeffs, R, t per image
- Reprojection error < 0.5 px is good, but low error alone is not enough if views lack diversity

---

## 2. HOMOGENEOUS COORDINATES & TRANSFORMATIONS

### Homogeneous Coordinates
```
2D point: (x, y) → [x, y, 1]ᵀ  or  [wx, wy, w]ᵀ
3D point: (X, Y, Z) → [X, Y, Z, 1]ᵀ

Point at infinity: [x, y, 0]ᵀ (2D), [X, Y, Z, 0]ᵀ (3D)
```

### 2D Transformations (in homogeneous coords)
```
Euclidean (3 DOF):   [R  t] (rotation + translation)
                     [0  1]

Similarity (4 DOF):  [sR t] (+ uniform scale)
                     [0  1]

Affine (6 DOF):      [A  t] (A is 2x2 invertible)
                     [0  1]
                     Preserves: parallel lines, ratios of areas

Projective (8 DOF):  [A  t]  (general 3x3 non-singular)
                     [vᵀ λ]
                     Preserves: straight lines, cross-ratio
```

### Homography (H)
```
Maps points between two planes (or two views of a planar scene):
x' = H * x    (H is 3x3, 8 DOF, up to scale)

Each point correspondence gives 2 equations.
Need ≥ 4 point correspondences (DLT algorithm).
Use RANSAC for robustness.
```

### DLT (Direct Linear Transform)
```
For x = (x, y, 1)ᵀ and x' = (u, v, 1)ᵀ:

[ -x  -y  -1   0   0   0   u*x  u*y  u ] h = 0
[  0   0   0  -x  -y  -1   v*x  v*y  v ] h = 0

Stack all points → Ah = 0 → solve via SVD, h = last column of V
Normalize points first for better numerical stability.
```

---

## 3. EPIPOLAR GEOMETRY

### Setup
```
Two cameras C, C' viewing point X.
x, x' = projections in respective images.
Baseline = line joining C and C'.
Epipoles e, e' = projection of one camera center onto the other image.
Epipolar plane = plane through X, C, C'.
Epipolar lines = intersection of epipolar plane with image planes.
```

### Fundamental Matrix (F)
```
x'ᵀ F x = 0    (Epipolar constraint)

F is 3x3, rank 2, 7 DOF
F encodes epipolar geometry in PIXEL coordinates

Properties:
- Fᵀe' = 0  (e' is left null space of F)
- Fe = 0    (e is right null space of F)
- l' = Fx   (epipolar line in image 2)
- l = Fᵀx'  (epipolar line in image 1)
```

### Essential Matrix (E)
```
x̂'ᵀ E x̂ = 0    (in NORMALIZED camera coordinates, x̂ = K⁻¹x)

E = [t]ₓ R    where [t]ₓ is the skew-symmetric matrix of t

E = K'ᵀ F K    (relationship between E and F)

Properties:
- 5 DOF (3 rotation + 2 translation direction, scale ambiguity)
- Two equal singular values: SVD(E) → diag(σ, σ, 0)

Decompose E → 4 possible (R, t), only 1 has points in front of both cameras.
```

### 8-Point Algorithm (for F)
```
1. Collect ≥ 8 point correspondences
2. Normalize points (Hartley normalization: zero mean, √2 avg distance)
3. Build matrix A from x'ᵀFx = 0 → Af = 0
4. SVD of A → f = last column of V
5. Reshape f to 3x3 → F̃
6. Enforce rank-2: SVD(F̃) = UDVᵀ, set smallest singular value to 0
7. F = U diag(σ1,σ2,0) Vᵀ
8. Denormalize: F = T'ᵀ F̃ T
```

### 5-Point Algorithm (for E)
```
- Minimal solver: 5 points → up to 10 solutions
- Used inside RANSAC
- Better for calibrated cameras
```

---

## 4. 3D RECONSTRUCTION

### Triangulation
```
Given x = P*X, x' = P'*X:
Solve: x × (PX) = 0  for each view → linear system AX = 0

     [ x*p3ᵀ - p1ᵀ ]
A =  [ y*p3ᵀ - p2ᵀ ]     (stack for all views)
     [ x'*p3'ᵀ - p1'ᵀ]
     [ y'*p3'ᵀ - p2'ᵀ]

SVD → X = last column of V
```

### Stereo Vision
```
Rectified stereo pair:
depth Z = f * B / d

f = focal length (pixels)
B = baseline (distance between cameras)
d = disparity = x_left - x_right

Disparity map → Dense depth estimation
Block matching / Semi-Global Matching (SGM) / Deep stereo (RAFT-Stereo)
```

### Structure from Motion (SfM)
```
Pipeline:
1. Feature detection (SIFT, SuperPoint)
2. Feature matching (nearest neighbor + ratio test, SuperGlue)
3. Geometric verification (F/E estimation + RANSAC)
4. Incremental reconstruction:
   a. Initialize with 2-view reconstruction
   b. PnP for new camera (add images one by one)
   c. Triangulate new points
   d. Bundle Adjustment (minimize reprojection error)
5. Dense reconstruction (MVS)
```

### Bundle Adjustment
```
min Σᵢ Σⱼ vᵢⱼ * ‖xᵢⱼ - π(Cⱼ, Xᵢ)‖²

xᵢⱼ = observed 2D point
π(Cⱼ, Xᵢ) = projection of 3D point Xᵢ by camera Cⱼ
vᵢⱼ = visibility indicator

Optimize over: camera parameters {Cⱼ} AND 3D points {Xᵢ}
Sparse Jacobian structure → Schur complement trick
Levenberg-Marquardt optimization
```

### PnP (Perspective-n-Point)
```
Given: n 3D-2D correspondences + K
Find: camera pose [R|t]

- P3P: minimal solver (3 points → up to 4 solutions), use in RANSAC
- EPnP: efficient O(n) for n ≥ 4
- PnP + RANSAC → cv2.solvePnPRansac()
```

---

## 5. SLAM (Simultaneous Localization and Mapping)

### Visual SLAM Pipeline
```
1. Front-end (Tracking):
   - Feature extraction & matching (ORB-SLAM) or direct methods (LSD-SLAM, DSO)
   - Motion estimation (PnP / essential matrix)
   - Keyframe selection

2. Back-end (Optimization):
   - Graph-based optimization (poses as nodes, constraints as edges)
   - Bundle adjustment (local & global)
   - Loop closure detection (DBoW2 / NetVLAD)

3. Map Management:
   - Sparse map (feature points) or Dense map (depth)
   - Map point culling, keyframe culling
```

### Key SLAM Systems
```
ORB-SLAM3:    Feature-based, monocular/stereo/RGB-D, IMU fusion
LSD-SLAM:     Direct (photometric error), semi-dense
DSO:          Direct Sparse Odometry, photometrically calibrated
RTAB-Map:     RGB-D SLAM with appearance-based loop closure
NICE-SLAM:    Neural implicit representations
```

### Pose Graph Optimization
```
Minimize: Σ ‖zᵢⱼ - (xᵢ⁻¹ ∘ xⱼ)‖²_Ωᵢⱼ

zᵢⱼ = relative pose measurement between nodes i and j
Ωᵢⱼ = information matrix (inverse covariance)
Solved with: g2o, GTSAM, Ceres Solver
Lie algebra (se(3)) for smooth optimization on manifold
```

### ICP (Iterative Closest Point)
```
Aligns two point clouds:
1. Find closest point correspondences
2. Compute optimal R, t (SVD of cross-covariance)
3. Apply transform
4. Repeat until convergence

Variants: point-to-plane ICP (faster convergence),
          GICP (generalized), colored ICP
```

---

## 6. POINT CLOUD PROCESSING

### Representations
```
Point Cloud:   {(x,y,z)} unordered set of 3D points
Voxel Grid:    3D grid, each cell occupied/empty + features
BEV (Bird's Eye View): top-down 2D projection
Range Image:   2D image where pixel value = depth (for LiDAR)
Mesh:          Vertices + Faces (triangles)
```

### PointNet / PointNet++
```
PointNet:
- Per-point MLP → max-pool → global feature
- T-Net for input/feature alignment (learned transformation)
- Permutation invariant via symmetric function (max pool)

PointNet++:
- Hierarchical: Set Abstraction layers
  1. Sampling (FPS - farthest point sampling)
  2. Grouping (ball query / kNN)
  3. PointNet on each local group
- Multi-scale grouping (MSG) for varying density
```

### 3D Object Detection
```
PointPillars:  Point cloud → pillars → 2D pseudo-image → 2D detection head
VoxelNet:      Voxelize → per-voxel PointNet → 3D conv → detection
CenterPoint:   Detect centers in BEV → refine with point features
SECOND:        Sparse 3D convolutions on voxels
PV-RCNN:       Point-Voxel fusion
```

---

## 7. OBJECT DETECTION (2D)

### Key Architectures
```
Two-stage:
  Faster R-CNN: RPN → RoI Pooling → classification + bbox regression
  Cascade R-CNN: Multi-stage refinement with increasing IoU thresholds

One-stage:
  YOLO (v5/v7/v8): Grid-based, anchor-free (recent versions)
  SSD: Multi-scale feature maps
  RetinaNet: FPN + Focal Loss
  FCOS: Fully convolutional, anchor-free, predicts distance to box edges

Transformer-based:
  DETR: CNN backbone → Transformer encoder-decoder → bipartite matching
  Deformable DETR: Deformable attention (faster convergence)
  RT-DETR: Real-time DETR
```

### Anchor Boxes
```
Predefined boxes at each spatial location:
- Multiple scales and aspect ratios
- Network predicts offsets (dx, dy, dw, dh) from anchors
- Anchor-free methods: predict center + distances to edges
```

### Non-Maximum Suppression (NMS)
```
1. Sort detections by confidence score
2. Pick highest score box, add to output
3. Remove all boxes with IoU > threshold (e.g., 0.5) with picked box
4. Repeat until no boxes remain

Variants: Soft-NMS (decay scores instead of remove),
          DIoU-NMS, Matrix NMS
```

### Metrics

#### IoU (Intersection over Union)
```
IoU = Area(A ∩ B) / Area(A ∪ B)
    = Area(A ∩ B) / (Area(A) + Area(B) - Area(A ∩ B))
```

#### GIoU (Generalized IoU)
```
GIoU = IoU - |C \ (A ∪ B)| / |C|
C = smallest enclosing box of A and B
Range: [-1, 1]
```

#### DIoU / CIoU
```
DIoU = IoU - ρ²(b, b_gt) / c²
ρ = Euclidean distance between centers
c = diagonal of smallest enclosing box

CIoU = IoU - ρ²(b, b_gt)/c² - αv
v = (4/π²)(arctan(w_gt/h_gt) - arctan(w/h))²
α = v / ((1-IoU) + v)
```

#### mAP (mean Average Precision)
```
For each class:
1. Rank detections by confidence
2. Match to ground truth (IoU > threshold)
3. Compute precision-recall curve
4. AP = area under P-R curve (interpolated)

mAP = mean of AP over all classes

COCO mAP: average over IoU thresholds [0.5:0.05:0.95]
Pascal VOC mAP@0.5: single IoU threshold = 0.5
```

#### Precision & Recall
```
Precision = TP / (TP + FP)    "Of detections, how many correct?"
Recall    = TP / (TP + FN)    "Of ground truth, how many found?"

TP = detection matched to GT with IoU > threshold
FP = detection not matched (or duplicate)
FN = GT not matched by any detection
```

### Feature Pyramid Network (FPN)
```
Bottom-up: C2→C3→C4→C5 (backbone features, decreasing resolution)
Top-down:  P5→P4→P3→P2 (upsampled + lateral connections from Cᵢ)
Lateral connection: 1x1 conv to match channels + element-wise add
Each Pᵢ → detection head (shared weights)
```

---

## 8. MULTI-OBJECT TRACKING (MOT)

### Paradigm: Tracking-by-Detection
```
1. Detect objects per frame
2. Associate detections across frames
3. Manage track lifecycle (birth, death, re-identification)
```

### SORT (Simple Online Realtime Tracking)
```
1. Kalman Filter: predict next state [x, y, s, r, ẋ, ẏ, ṡ]
   (x,y = center, s = area, r = aspect ratio)
2. Hungarian Algorithm: match predictions to detections (IoU cost)
3. Update matched tracks, create new / delete lost
```

### DeepSORT
```
SORT + Re-identification (ReID) features:
- Appearance descriptor (CNN embedding per detection)
- Association cost = λ * motion_cost + (1-λ) * appearance_cost
- Cascade matching: prioritize recently seen tracks
- Mahalanobis distance for motion gating
```

### MOT Metrics
```
MOTA = 1 - (FN + FP + IDSW) / GT
  FN = false negatives, FP = false positives, IDSW = identity switches

MOTP = Σ dₜ / Σ cₜ     (average overlap of matched pairs)

IDF1 = 2 * IDTP / (2*IDTP + IDFP + IDFN)
  "How well identities are preserved"

HOTA = √(DetA * AssA)   (newer metric, balances detection & association)
```

---

## 9. HAND / BODY TRACKING

### Body Pose Estimation
```
Top-Down:    Detect person → estimate joints per person (HRNet, SimpleBaseline)
Bottom-Up:   Detect all joints → group into people (OpenPose, HigherHRNet)

Heatmap-based: predict K heatmaps (one per joint), argmax = joint location
Regression-based: directly predict (x,y) coordinates

OpenPose: Two branches — Part Affinity Fields (PAFs) + confidence maps
  PAFs encode limb direction for bottom-up grouping
```

### Hand Pose Estimation
```
21 keypoints per hand (4 per finger + wrist)

MANO model: parametric hand model
  - Shape params β (10-dim)
  - Pose params θ (15 joints × 3 rot = 45-dim + 3 global rot)
  - Output: 778 vertices mesh

MediaPipe Hands: palm detection → hand landmark model (21 3D landmarks)
```

### SMPL / SMPL-X (Body Model)
```
SMPL: body shape β (10) + pose θ (72 = 24 joints × 3)
  → 6890 vertices mesh
  M(β, θ) = W(T(β,θ), J(β), θ, W)
  T = template + shape blend shapes + pose blend shapes
  W = linear blend skinning

SMPL-X: SMPL + hands (MANO) + face (FLAME)
```

---

## 10. VIDEO PROCESSING & ACTION RECOGNITION

### Optical Flow
```
Brightness constancy: I(x,y,t) = I(x+u, y+v, t+1)
Taylor expansion → Ix*u + Iy*v + It = 0  (aperture problem)

Lucas-Kanade: assume constant flow in local window → solve overdetermined system
Horn-Schunck: global smoothness constraint

Deep optical flow: FlowNet → RAFT (Recurrent All-Pairs Field Transforms)
RAFT: all-pairs correlation volume + GRU iterative updates
```

### Video Understanding Architectures
```
Two-Stream:     RGB stream + optical flow stream → late fusion
I3D:            Inflate 2D convs to 3D (3D kernels on video)
SlowFast:       Slow pathway (low FPS, rich semantics) + Fast pathway (high FPS, motion)
                Lateral connections between pathways
TimeSformer:    Divided space-time attention on video tokens
Video Swin:     3D shifted window attention
VideoMAE:       Masked autoencoder for video (self-supervised pretraining)
```

### Temporal Modeling
```
3D Conv:       (C, T, H, W) → captures spatiotemporal patterns
(2+1)D Conv:   Separate spatial (1×H×W) and temporal (T×1×1) convolutions
               More parameters-efficient, easier to optimize
Temporal Shift: Shift part of channels along time dimension (TSM) — zero compute cost
```

---

## 11. DEPTH ESTIMATION

### Monocular Depth
```
Self-supervised: image reconstruction loss + smoothness loss
  - Photometric loss: SSIM + L1 between warped and target image
  - Monodepth2, MiDaS, DPT (transformer-based), Depth Anything

Scale ambiguity: monocular depth is up to scale
  → resolve with known object size, IMU, or metric depth training
```

### Multi-View Stereo (MVS)
```
1. Plane Sweep: sweep depth planes, compute photometric consistency
2. Cost Volume: store matching costs at each pixel × depth hypothesis
3. Regularization: 3D conv or GRU to smooth cost volume
4. Depth map: argmin along depth dimension

Key methods: MVSNet, Vis-MVSNet, CasMVSNet (coarse-to-fine)
```

---

## 12. NEURAL RADIANCE FIELDS (NeRF) & 3D RECONSTRUCTION

### NeRF
```
F(x, y, z, θ, φ) → (RGB, σ)
MLP maps 3D position + viewing direction → color + density

Volume rendering:
C(r) = Σᵢ Tᵢ (1 - exp(-σᵢδᵢ)) cᵢ
Tᵢ = exp(-Σⱼ<ᵢ σⱼδⱼ)    (transmittance)

Positional encoding: γ(p) = [sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
```

### 3D Gaussian Splatting
```
Represent scene as millions of 3D Gaussians:
Each Gaussian: position μ, covariance Σ, opacity α, color (SH coefficients)

Rasterization-based rendering (not ray marching) → real-time
Differentiable: optimize Gaussians via gradient descent on rendering loss
Adaptive density control: split/clone/prune Gaussians during training
```

---

## 13. MODEL OPTIMIZATION FOR EDGE DEPLOYMENT

### Techniques
```
Quantization:       FP32 → FP16/INT8 (post-training or quantization-aware training)
Pruning:            Remove low-magnitude weights (structured / unstructured)
Knowledge Distill.: Train small "student" from large "teacher"
TensorRT:           NVIDIA optimizer for inference (layer fusion, kernel auto-tuning)
ONNX Runtime:       Cross-platform inference optimization
Mobile architectures: MobileNet (depthwise separable conv), EfficientNet, ShuffleNet
```

### Depthwise Separable Convolution
```
Standard conv: Cᵢₙ × Cₒᵤₜ × K × K  params
Depthwise separable:
  1. Depthwise: Cᵢₙ × 1 × K × K  (one filter per input channel)
  2. Pointwise: Cᵢₙ × Cₒᵤₜ × 1 × 1

Cost reduction: 1/Cₒᵤₜ + 1/K²
```

---

## 14. FAILURE MODES & DEBUGGING

### Common Failure Modes
```
Calibration:
- Poor corner coverage → unstable intrinsics
- Rolling shutter / blur → bad reprojection error

Epipolar geometry:
- Repeated texture → wrong matches
- Tiny baseline or pure rotation → unstable translation/depth
- Dynamic objects → violate rigid-scene assumption

Stereo / depth:
- Textureless, reflective, transparent, occluded regions break matching

Tracking:
- Missed detections → fragmented tracks
- Similar appearance + long occlusion → ID switches

ICP / point cloud alignment:
- Poor initialization, partial overlap, symmetric geometry → local minima
```

### Strong Practical Answers
- "I would inspect residuals and inlier masks, not just the final scalar metric."
- "I would visualize matches, epipolar lines, and reprojections before changing the model."
- "I would gate associations with motion/geometry first, then appearance."
- "If metric scale matters, monocular alone is not enough without extra information."

---

## 15. KEY FORMULAS QUICK REFERENCE

```
Projection:     s x = K [R | t] X

Epipolar:       x'ᵀ F x = 0
Essential:      E = [t]ₓ R

Stereo depth:   Z = fB / d

Rodrigues:     R = I + sin(θ)[k]ₓ + (1-cos(θ))[k]ₓ²
               (axis-angle k,θ → rotation matrix)

Quaternion:    q = [w, x, y, z],  ‖q‖ = 1
               Rotation: p' = q p q*

SE(3):         [R  t]  ∈ SE(3)  (rigid body transformation)
               [0  1]

Cross product  [a]ₓ = [ 0  -a3  a2]
matrix:               [ a3  0  -a1]
                      [-a2  a1   0]

Reprojection   e = Σᵢ ‖xᵢ - π(P Xᵢ)‖²
error:         lower is better, but inspect outliers not just mean
```
