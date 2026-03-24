# Computer Vision Interview Revision Handbook

> Dense revision material for ML/CV engineer, perception engineer, applied scientist, and research engineer interviews.

## Legend
- `[HIGH-YIELD]`: top 20% topics that generate disproportionate interview value.
- `Must know`: expected in most ML/CV interviews.
- `Good to know`: common follow-up material.
- `Bonus`: mostly for research-heavy or senior roles.
- `[COMMON TRAP]`: answer pattern that often gets candidates rejected.
- `[GOOD INTERVIEW LINE]`: phrasing that signals technical maturity.

---

## 1. Executive Overview

### 1.1 One-Page High-Level Summary
- Computer Vision interviews test whether you can move cleanly from raw pixels to a reliable system: representation, geometry, learning, metrics, failure analysis, and deployment tradeoffs.
- Classical CV is still relevant because it teaches invariances, image structure, correspondences, camera geometry, and why deep models behave the way they do.
- Deep CV interviews focus less on memorizing architectures and more on whether you understand:
  - why a model class is appropriate,
  - what the outputs mean,
  - how it is trained,
  - how it fails,
  - how it is evaluated,
  - how it behaves under production constraints.
- Geometry questions separate candidates who can build perception systems from candidates who only fine-tune models. You should be comfortable with projection, calibration, homography, epipolar geometry, PnP, triangulation, stereo, and bundle adjustment.
- Task formulation matters. Interviewers expect precise distinctions between:
  - classification vs detection vs segmentation,
  - semantic vs instance vs panoptic segmentation,
  - tracking vs optical flow,
  - stereo depth vs monocular depth,
  - 2D detection vs 3D detection vs BEV perception.
- Metrics matter as much as models. Candidates often know architectures but fail to justify the correct metric under class imbalance, localization error, calibration requirements, or long-tail distributions.
- Data quality is often the dominant factor in CV performance. Strong candidates talk about label quality, leakage, sampling, augmentation, ablations, and failure taxonomy before proposing bigger models.
- Production CV interviews care about latency, throughput, batching, memory, model compression, robustness, and monitoring. A good answer is rarely "use a larger model."

### 1.2 What Interviewers Usually Test
- First-principles understanding: convolution, feature extraction, calibration, metric definitions.
- Ability to distinguish adjacent concepts cleanly.
- Practical engineering intuition: what breaks in the real world and why.
- Model-task fit: why this architecture, loss, augmentation, and metric.
- Debugging skill: when a model underperforms, where do you look first.
- Communication quality: can you explain geometry and system tradeoffs without hand-waving.
- Project ownership: did you just train the model, or did you improve the whole pipeline.

### 1.3 The 80/20 Topics [HIGH-YIELD]
- `[HIGH-YIELD] Must know` image representation, convolution, filtering, gradients, edge detection, histogram methods.
- `[HIGH-YIELD] Must know` SIFT/ORB intuition, feature matching, RANSAC, registration.
- `[HIGH-YIELD] Must know` camera model, intrinsics/extrinsics, calibration, distortion, homography, essential vs fundamental matrix, triangulation, PnP.
- `[HIGH-YIELD] Must know` CNN basics, receptive field, residual connections, normalization, augmentation, transfer learning.
- `[HIGH-YIELD] Must know` detection, segmentation, tracking, depth estimation: outputs, losses, metrics, failure modes.
- `[HIGH-YIELD] Must know` IoU, mAP, precision/recall, F1, ROC vs PR, calibration.
- `[HIGH-YIELD] Must know` class imbalance, overfitting, data leakage, noisy labels, error analysis, ablations.
- `[HIGH-YIELD] Must know` latency vs accuracy, quantization, distillation, edge deployment tradeoffs.
- `Good to know` ViTs, DETR, open-vocabulary detection, foundation segmentation, self-supervised learning.
- `Bonus` NeRF, diffusion details, synthetic data pipelines, VLMs in perception.

### 1.4 What To Memorize vs What To Understand
- Memorize:
  - projection equation,
  - IoU, precision, recall, F1,
  - essential vs fundamental matrix distinction,
  - stereo depth equation,
  - attention equation,
  - convolution output size formula,
  - detection and segmentation metrics.
- Understand:
  - why homography fails on non-planar scenes,
  - why CE is better than MSE for classification,
  - why class imbalance breaks naive training,
  - why one-stage detectors trade some accuracy for speed,
  - why monocular depth is ambiguous up to scale,
  - why calibration can look numerically good but still be bad.

### 1.5 Strong Interview Meta-Positioning
- `[GOOD INTERVIEW LINE] "I try to separate data problems, optimization problems, metric problems, and modeling problems before changing the architecture."`
- `[GOOD INTERVIEW LINE] "For perception work, I care about uncertainty, calibration, and failure taxonomy, not just aggregate accuracy."`
- `[GOOD INTERVIEW LINE] "If geometry is available, I prefer to use it explicitly rather than asking a model to relearn it from scratch."`

---

## 2. Full Topic Map For Computer Vision Interviews

## 2.A Classical Computer Vision

### Image Representation [HIGH-YIELD] Must know
- What it is: images as discrete sampled signals; grayscale, RGB, HSV, LAB; pixel intensities as arrays or tensors.
- Why it matters: every filter, augmentation, and network assumes a representation and dynamic range.
- Where used: preprocessing, normalization, color-based segmentation, photometric augmentation.
- Common interview questions:
  - Why use HSV instead of RGB?
  - What changes when an image is normalized to `[0,1]` vs standardized?
  - Why can color space conversion improve robustness?
- `[COMMON TRAP]` Treating images as abstract tensors without discussing sampling, quantization, illumination, or channel semantics.

### Filtering And Convolution [HIGH-YIELD] Must know
- What it is: applying a local kernel to compute weighted combinations of neighboring pixels.
- Why it matters: smoothing, denoising, derivatives, edge detection, and the core operation in CNNs.
- Where used: Gaussian blur, Sobel filters, Laplacian, image sharpening, feature extraction.
- Common interview questions:
  - Difference between correlation and convolution?
  - Why use separable filters?
  - What does kernel size or stride change?
- `[COMMON TRAP]` Saying convolution is only "feature extraction" without explaining locality, translation equivariance, or boundary handling.

### Edge Detection [HIGH-YIELD] Must know
- What it is: detecting strong intensity gradients that often correspond to object boundaries.
- Why it matters: edges are a compact structural cue for segmentation, registration, and shape analysis.
- Where used: Canny edge detection, contour extraction, Hough transforms, classical tracking.
- Common interview questions:
  - Why does Canny use non-maximum suppression and hysteresis?
  - Why is Gaussian smoothing applied before derivatives?
  - Difference between Sobel and Laplacian?
- `[COMMON TRAP]` Ignoring noise sensitivity and the fact that gradients respond to texture and illumination, not just true object boundaries.

### Corners And Features [HIGH-YIELD] Must know
- What it is: points with strong intensity variation in multiple directions; useful as stable correspondences.
- Why it matters: corners are better than edges for matching because edges are ambiguous along the tangent direction.
- Where used: Harris, Shi-Tomasi, SLAM front-ends, tracking, image stitching.
- Common interview questions:
  - Why are corners easier to match than edges?
  - What does the structure tensor measure?
  - Why does Harris respond poorly to scale changes?
- `[COMMON TRAP]` Describing corners only visually and not explaining the eigenvalue intuition.

### SIFT / SURF / ORB [HIGH-YIELD] Must know
- What it is: local feature detectors and descriptors with varying invariance and speed tradeoffs.
- Why it matters: robust matching, registration, localization, and classical retrieval still depend on local descriptors.
- Where used: panorama stitching, visual odometry, place recognition, structure from motion.
- Common interview questions:
  - Why is SIFT robust to scale and rotation?
  - Why is ORB faster?
  - When would you prefer ORB to SIFT?
- `[COMMON TRAP]` Saying ORB is "better because it is faster" without discussing binary descriptors, invariance, and match quality.

### Template Matching Must know
- What it is: sliding a template over an image and scoring similarity.
- Why it matters: simplest form of detection and a useful baseline.
- Where used: industrial inspection, UI automation, controlled environments.
- Common interview questions:
  - Why does template matching fail under scale, rotation, illumination, or deformation?
  - Why use normalized cross-correlation?
- `[COMMON TRAP]` Proposing template matching in settings with pose variation or clutter.

### Hough Transform Good to know
- What it is: voting in parameter space for shapes such as lines or circles.
- Why it matters: robust detection when boundaries are fragmented.
- Where used: lane detection, line extraction, circle detection, document analysis.
- Common interview questions:
  - How does line Hough transform map an image point to parameter space?
  - Why is it robust to missing edge fragments?
- `[COMMON TRAP]` Not explaining the duality between image space and parameter space.

### Image Pyramids Must know
- What it is: multi-scale representations formed by repeated smoothing and downsampling.
- Why it matters: scale-space reasoning and coarse-to-fine processing.
- Where used: SIFT, optical flow, multi-scale detection, segmentation backbones.
- Common interview questions:
  - Why do you smooth before downsampling?
  - Why does pyramid matching help with large motion?
- `[COMMON TRAP]` Ignoring aliasing during subsampling.

### Optical Flow [HIGH-YIELD] Must know
- What it is: dense or sparse apparent motion field between frames.
- Why it matters: motion estimation, stabilization, tracking, odometry, and video understanding.
- Where used: Lucas-Kanade, Horn-Schunck, RAFT, action analysis.
- Common interview questions:
  - What is the brightness constancy assumption?
  - What is the aperture problem?
  - Difference between optical flow and object tracking?
- `[COMMON TRAP]` Treating flow as actual 3D motion rather than image-plane apparent motion.

### Image Registration [HIGH-YIELD] Must know
- What it is: aligning images or modalities through geometric transformation.
- Why it matters: stitching, medical imaging, remote sensing, multi-view reconstruction.
- Where used: homography estimation, affine registration, deformable registration.
- Common interview questions:
  - When is homography sufficient?
  - Difference between feature-based and direct registration?
  - Why use RANSAC?
- `[COMMON TRAP]` Assuming a single global transform always exists.

### Background Subtraction Good to know
- What it is: separating moving foreground from a relatively static scene.
- Why it matters: simple motion segmentation baseline.
- Where used: surveillance, traffic analytics, people counting.
- Common interview questions:
  - Why does it fail with dynamic backgrounds or lighting changes?
  - Difference between frame differencing and Gaussian mixture models?
- `[COMMON TRAP]` Ignoring camera motion and shadow artifacts.

### Morphology Must know
- What it is: shape operations based on structuring elements, such as erosion and dilation.
- Why it matters: cleaning masks, filling holes, separating connected regions.
- Where used: postprocessing segmentation masks, OCR, defect detection.
- Common interview questions:
  - Difference between erosion and dilation?
  - What are opening and closing used for?
- `[COMMON TRAP]` Not linking morphology to binary structure and connectivity.

### Thresholding And Histogram Methods Must know
- What it is: converting intensities to labels using global or adaptive rules; histogram equalization modifies contrast.
- Why it matters: quick segmentation baselines and photometric normalization.
- Where used: OCR, industrial inspection, low-resource pipelines.
- Common interview questions:
  - When does Otsu work well?
  - Difference between global and adaptive thresholding?
  - Why can histogram equalization hurt?
- `[COMMON TRAP]` Ignoring illumination nonuniformity.

## 2.B Geometry And Multi-View Vision

### Pinhole Camera Model [HIGH-YIELD] Must know
- What it is: ideal perspective projection from 3D world coordinates to 2D image coordinates.
- Why it matters: basis of calibration, pose estimation, stereo, and SLAM.
- Key equation: `s x = K [R | t] X`.
- Interview framing: explain projection as scaling by inverse depth after rigid transform into camera coordinates.
- `[COMMON TRAP]` Mixing up camera coordinates and world coordinates.

### Camera Intrinsics And Extrinsics [HIGH-YIELD] Must know
- What it is:
  - intrinsics: focal lengths, principal point, skew;
  - extrinsics: camera pose relative to world or world relative to camera.
- Why it matters: any 3D-to-2D reasoning depends on separating calibration from pose.
- Common interview questions:
  - What changes if you resize the image?
  - What do extrinsics mean physically?
- `[COMMON TRAP]` Saying translation is the camera position directly without being clear about the coordinate convention.

### Projection And Backprojection [HIGH-YIELD] Must know
- What it is: projection maps 3D to 2D; backprojection maps a pixel to a ray in 3D.
- Why it matters: triangulation, depth estimation, ray casting, NeRF-style reasoning.
- Common interview questions:
  - Why can’t one pixel define a unique 3D point without depth?
  - How do you backproject a depth pixel to 3D?
- `[COMMON TRAP]` Forgetting that backprojection gives a ray unless metric depth is known.

### Homogeneous Coordinates Must know
- What it is: projective representation that makes perspective transforms linear.
- Why it matters: enables elegant formulation of translation, projection, homography, and points at infinity.
- Common interview questions:
  - Why use homogeneous coordinates?
  - What does the scale ambiguity mean?
- `[COMMON TRAP]` Treating `[x, y, 1]` and `[2x, 2y, 2]` as different points.

### Coordinate Transforms Must know
- What it is: mapping between world, camera, image, body, LiDAR, and map frames.
- Why it matters: robotics and autonomy interviews expect disciplined frame reasoning.
- Common interview questions:
  - How do you chain transforms?
  - How do you invert an SE(3) transform?
- `[COMMON TRAP]` Losing track of source frame vs destination frame.

### Camera Calibration [HIGH-YIELD] Must know
- What it is: estimating intrinsics and distortion from known patterns or self-calibration.
- Why it matters: bad calibration corrupts everything downstream.
- Where used: robotics, AR/VR, stereo rigs, measurement systems.
- Common interview questions:
  - How many views do you need and why should they vary?
  - What does reprojection error measure?
- `[COMMON TRAP]` Focusing only on low reprojection error and not on coverage or deployment conditions.

### Distortion Must know
- What it is: deviation from ideal pinhole projection, usually radial and tangential.
- Why it matters: edges of wide-FOV images can be severely warped.
- Common interview questions:
  - What is radial distortion?
  - Why undistort before geometry?
- `[COMMON TRAP]` Ignoring distortion in feature matching or stereo.

### Homography [HIGH-YIELD] Must know
- What it is: projective mapping between image planes for a planar scene or pure camera rotation.
- Why it matters: stitching, registration, plane tracking, IPM, document rectification.
- Common interview questions:
  - When is homography valid?
  - Why does it fail on non-planar scenes?
  - How many correspondences are needed?
- `[COMMON TRAP]` Using homography to explain arbitrary 3D scenes.

### Epipolar Geometry [HIGH-YIELD] Must know
- What it is: geometric relationship between two views of the same rigid scene.
- Why it matters: constrains matching from 2D search to epipolar lines.
- Key equation: `x'^T F x = 0`.
- Common interview questions:
  - What are epipoles and epipolar lines?
  - Why does this reduce correspondence search?
- `[COMMON TRAP]` Memorizing equations without geometric meaning.

### Essential Matrix Vs Fundamental Matrix [HIGH-YIELD] Must know
- What it is:
  - `F`: uncalibrated relation in pixel coordinates,
  - `E`: calibrated relation in normalized coordinates.
- Why it matters: interviewers use this to test whether you understand calibrated vs uncalibrated geometry.
- Common interview questions:
  - When do you estimate `F` vs `E`?
  - Why is `E = [t]_x R`?
- `[COMMON TRAP]` Saying they are interchangeable.

### Triangulation [HIGH-YIELD] Must know
- What it is: recovering a 3D point from multiple rays/views.
- Why it matters: central to stereo, SfM, VO, SLAM.
- Common interview questions:
  - Why is small baseline bad?
  - Why does noisy correspondence create large depth error far away?
- `[COMMON TRAP]` Ignoring uncertainty growth with distance.

### Stereo Vision [HIGH-YIELD] Must know
- What it is: estimating depth from disparity between rectified left/right images.
- Key equation: `Z = fB / d`.
- Why it matters: one of the most common depth interview topics.
- Common interview questions:
  - Why rectify?
  - Why does depth explode when disparity is small?
- `[COMMON TRAP]` Forgetting occlusions and low-texture failures.

### Pose Estimation / PnP [HIGH-YIELD] Must know
- What it is: estimating camera pose from 3D-2D correspondences.
- Why it matters: localization, AR, SfM, robotics.
- Common interview questions:
  - What is P3P?
  - Why use RANSAC with PnP?
  - What is needed besides correspondences?
- `[COMMON TRAP]` Confusing PnP with triangulation.

### Bundle Adjustment Good to know
- What it is: joint nonlinear optimization over poses and 3D structure to minimize reprojection error.
- Why it matters: the accuracy backbone of SfM and SLAM.
- Common interview questions:
  - Why is it nonlinear?
  - Why is it expensive?
  - Why is sparse structure important?
- `[COMMON TRAP]` Describing BA as a small postprocessing step rather than the global refinement stage.

### Visual Odometry Must know
- What it is: estimating egomotion from sequential visual observations.
- Why it matters: robotics, autonomy, AR.
- Common interview questions:
  - Feature-based vs direct VO?
  - What causes drift?
- `[COMMON TRAP]` Ignoring scale ambiguity in monocular VO.

### SLAM Basics Good to know
- What it is: simultaneous localization and mapping with loop closure and map optimization.
- Why it matters: common in robotics interviews.
- Common interview questions:
  - Difference between VO and SLAM?
  - What is loop closure?
  - What is pose graph optimization?
- `[COMMON TRAP]` Saying SLAM is just "VO plus mapping" without discussing loop closure or consistency.

## 2.C Deep Learning For Computer Vision

### CNN Basics [HIGH-YIELD] Must know
- What it is: weight sharing + local receptive fields for spatial pattern learning.
- Why used: inductive bias for images, parameter efficiency, translation equivariance.
- Tradeoffs: strong locality prior, but less global context than transformer-based models.
- Common interview questions:
  - Why do CNNs work well on images?
  - What does weight sharing buy you?
- `[COMMON TRAP]` Saying CNNs are invariant to translation; they are mostly equivariant, with approximate invariance after pooling.

### Convolution, Stride, Padding, Dilation [HIGH-YIELD] Must know
- Concept:
  - stride downsamples,
  - padding controls output size/border effects,
  - dilation expands receptive field without extra parameters.
- Why used: receptive field control and efficiency.
- Tradeoffs: more stride loses spatial detail; more dilation can create gridding artifacts.
- Common interview questions:
  - Output shape formula?
  - Same vs valid padding?
  - Why use dilated convs in segmentation?
- `[COMMON TRAP]` Confusing parameter count with activation memory.

### Receptive Field [HIGH-YIELD] Must know
- Concept: the input region that can influence one output feature.
- Why used: determines available context for classification, detection, segmentation.
- Tradeoffs: theoretical receptive field can exceed effective receptive field.
- Common interview questions:
  - Why does receptive field matter for small vs large objects?
  - How do skip connections/FPN help?
- `[COMMON TRAP]` Assuming deeper always means practically enough context.

### Pooling Must know
- Concept: local aggregation for downsampling or invariance.
- Why used: reduces resolution and noise sensitivity.
- Tradeoffs: max pooling preserves strong activations; average pooling preserves smoother statistics.
- Common interview questions:
  - Why global average pooling before classification?
  - Why is pooling less central in some modern architectures?
- `[COMMON TRAP]` Ignoring the information loss.

### Residual Connections [HIGH-YIELD] Must know
- Concept: learn residual function `F(x)` and add input skip.
- Why used: stabilizes optimization in deeper networks.
- Tradeoffs: better gradient flow but not a free fix for bad architecture or data.
- Common interview questions:
  - Why do residual networks train deeper models more easily?
  - Identity mapping vs projection shortcut?
- `[COMMON TRAP]` Saying residuals "prevent vanishing gradients completely."

### Normalization [HIGH-YIELD] Must know
- Concept: normalize activations or features to stabilize optimization.
- Why used: smoother optimization and less sensitivity to initialization.
- Tradeoffs:
  - BatchNorm depends on batch statistics,
  - LayerNorm is batch-independent,
  - GroupNorm helps with small batch sizes.
- Common interview questions:
  - Why does BN struggle with tiny batches?
  - Why is LN standard in transformers?
- `[COMMON TRAP]` Treating all normalization layers as interchangeable.

### Transfer Learning [HIGH-YIELD] Must know
- Concept: initialize from pretrained weights and fine-tune or freeze.
- Why used: improves data efficiency and convergence.
- Tradeoffs: domain mismatch can limit gains.
- Common interview questions:
  - When would you freeze the backbone?
  - What if pretrained data is very different?
- `[COMMON TRAP]` Assuming transfer learning always helps equally.

### Regularization And Augmentation [HIGH-YIELD] Must know
- Concept: reduce overfitting by changing data, targets, or optimization behavior.
- Why used: generalization.
- Tradeoffs: too much augmentation can shift distribution or hurt label validity.
- Common interview questions:
  - Why does MixUp help?
  - When does random crop break labels?
  - What augmentations are dangerous for detection or keypoints?
- `[COMMON TRAP]` Recommending augmentations without checking label geometry.

### Loss Functions [HIGH-YIELD] Must know
- Concept: objective functions for classification, localization, segmentation, metric learning.
- Why used: shape the optimization landscape to match the task.
- Common interview questions:
  - CE vs BCE?
  - L1 vs Smooth L1 for box regression?
  - Dice vs CE for segmentation?
- `[COMMON TRAP]` Choosing a loss because it is popular rather than because it matches data/metric imbalance.

### Class Imbalance [HIGH-YIELD] Must know
- Concept: skewed label distribution across classes or spatial negatives vs positives.
- Why used: detection and segmentation heavily suffer from imbalance.
- Methods: reweighting, focal loss, resampling, hard negative mining, threshold tuning.
- Common interview questions:
  - Why does accuracy fail here?
  - When use focal loss?
- `[COMMON TRAP]` Using naive oversampling without discussing overfitting or calibration.

### Feature Pyramids [HIGH-YIELD] Must know
- Concept: multi-scale feature hierarchy for objects of different sizes.
- Why used: detection and segmentation need both semantics and resolution.
- Common interview questions:
  - Why is FPN effective?
  - Why are small objects hard without high-resolution features?
- `[COMMON TRAP]` Not connecting FPN to receptive field and scale variation.

### Attention In Vision Must know
- Concept: data-dependent weighted aggregation across tokens or spatial positions.
- Why used: global context, long-range interactions, multimodal alignment.
- Tradeoffs: quadratic attention cost in vanilla form; windowed or sparse attention reduces compute.
- Common interview questions:
  - Why might attention help compared with convolution?
  - What is self-attention vs cross-attention?
- `[COMMON TRAP]` Saying attention replaces all inductive biases with no downside.

### ViT Basics [HIGH-YIELD] Good to know
- Concept: split image into patches, embed them, apply transformer blocks.
- Why used: strong scaling behavior and global context.
- Tradeoffs: weaker locality prior, usually wants more data or stronger pretraining.
- Common interview questions:
  - Why do ViTs usually need more pretraining data?
  - How do positional embeddings matter?
- `[COMMON TRAP]` Claiming ViTs are strictly better than CNNs in all regimes.

### DETR Intuition Good to know
- Concept: object detection as set prediction with bipartite matching.
- Why used: removes anchors and NMS in the core formulation.
- Tradeoffs: slower convergence; variants like Deformable DETR improve it.
- Common interview questions:
  - Why does DETR avoid NMS?
  - Why is Hungarian matching used?
- `[COMMON TRAP]` Missing the set-prediction viewpoint.

## 2.D Core CV Tasks

### Image Classification [HIGH-YIELD] Must know
- Problem definition: assign one or more labels to an image.
- Common models: ResNet, EfficientNet, ConvNeXt, ViT.
- Input/output: image tensor to logits or probabilities.
- Losses: cross-entropy, BCE for multi-label, label smoothing.
- Metrics: accuracy, top-k, precision/recall/F1, calibration.
- Failure modes: class imbalance, shortcut learning, spurious correlations, domain shift.
- Practical interview questions:
  - What if classes are highly imbalanced?
  - When use top-5 instead of top-1?

### Object Detection [HIGH-YIELD] Must know
- Problem definition: classify and localize objects with bounding boxes.
- Common models: Faster R-CNN, RetinaNet, YOLO, FCOS, DETR.
- Input/output: image to class scores plus boxes; sometimes objectness.
- Losses: classification + box regression + IoU/GIoU/DIoU/CIoU variants.
- Metrics: IoU, AP, mAP, per-class AP.
- Failure modes: duplicate boxes, missed small objects, localization bias, long-tail classes.
- Practical interview questions:
  - One-stage vs two-stage?
  - Anchor-based vs anchor-free?
  - Why is mAP@[.5:.95] harder than mAP@0.5?

### Semantic Segmentation [HIGH-YIELD] Must know
- Problem definition: per-pixel class prediction.
- Common models: FCN, U-Net, DeepLab, SegFormer.
- Input/output: image to dense class map.
- Losses: per-pixel CE, Dice, focal, Lovasz, boundary losses.
- Metrics: pixel accuracy, mIoU, Dice.
- Failure modes: boundary quality, minority classes, class confusion, resolution loss.
- Practical interview questions:
  - Why is mIoU preferred over accuracy?
  - Why do skip connections help?

### Instance Segmentation Must know
- Problem definition: detect and segment each object instance separately.
- Common models: Mask R-CNN, YOLACT, SOLO, Mask2Former.
- Input/output: per-instance mask plus class and box.
- Losses: detection losses + mask loss.
- Metrics: mask AP, box AP.
- Failure modes: crowded scenes, occlusion, mask leakage.
- Practical interview questions:
  - How is it different from semantic segmentation?
  - Why is instance segmentation harder?

### Panoptic Segmentation Good to know
- Problem definition: unify semantic "stuff" and instance "things."
- Common models: Panoptic FPN, Mask2Former.
- Input/output: every pixel gets a semantic class and, for thing classes, an instance ID.
- Metrics: PQ, SQ, RQ.
- Failure modes: merging/splitting errors, stuff-thing confusion.
- Practical interview questions:
  - Why is panoptic useful for autonomy and scene understanding?

### Keypoint Detection Must know
- Problem definition: localize semantic points such as joints or landmarks.
- Common models: heatmap-based CNNs, HRNet, stacked hourglass, transformer heads.
- Input/output: heatmaps or coordinate regression.
- Losses: L2/MSE on heatmaps, coordinate regression losses, OKS-aware losses.
- Metrics: PCK, OKS, AP for pose benchmarks.
- Failure modes: occlusion, left-right ambiguity, low resolution.
- Practical interview questions:
  - Heatmap vs direct regression?
  - Why is quantization error relevant?

### Pose Estimation Must know
- Problem definition: estimate body, hand, face, or 6D object pose.
- Common models: top-down pose, bottom-up pose, PnP-based 6D pose estimators.
- Input/output: keypoints, skeleton, or 6D pose.
- Losses: heatmap losses, reprojection loss, pose parameter losses.
- Metrics: PCK, OKS, ADD/ADD-S for object pose.
- Failure modes: occlusion, self-occlusion, symmetries.
- Practical interview questions:
  - Top-down vs bottom-up human pose?
  - How do symmetries affect 6D pose metrics?

### Object Tracking [HIGH-YIELD] Must know
- Problem definition: maintain object identities over time.
- Common models: SORT, DeepSORT, ByteTrack, FairMOT, transformer trackers.
- Input/output: trajectories with IDs.
- Losses: detection loss plus sometimes re-identification or association loss.
- Metrics: MOTA, IDF1, HOTA, ID switches.
- Failure modes: occlusion, missed detections, identity switches, camera motion.
- Practical interview questions:
  - Why can a better detector improve tracking more than a fancier tracker?
  - Motion model vs appearance model?

### Optical Flow Must know
- Problem definition: estimate dense pixel-wise motion.
- Common models: Lucas-Kanade, Horn-Schunck, FlowNet, PWC-Net, RAFT.
- Input/output: dense 2D displacement field.
- Losses: endpoint error, robust photometric terms, multi-scale losses.
- Metrics: EPE, Fl-all, outlier rate.
- Failure modes: occlusions, motion boundaries, low texture, lighting changes.
- Practical interview questions:
  - Why is flow different from tracking?
  - Why are occlusions difficult?

### Depth Estimation [HIGH-YIELD] Must know
- Problem definition: infer per-pixel distance or inverse depth.
- Common models: stereo cost-volume models, monocular depth networks, RGB-D pipelines.
- Input/output: depth map or disparity map.
- Losses: L1/L2/inverse depth, scale-invariant losses, photometric reprojection.
- Metrics: Abs Rel, RMSE, `δ < 1.25`.
- Failure modes: scale ambiguity, textureless surfaces, reflective objects, invalid depth.
- Practical interview questions:
  - Stereo vs monocular depth?
  - Why is monocular depth ambiguous up to scale?

### 3D Detection Good to know
- Problem definition: detect objects in 3D space using LiDAR, camera, or fusion.
- Common models: PointPillars, SECOND, PV-RCNN, CenterPoint, BEVFusion.
- Input/output: 3D boxes with class, position, size, yaw, velocity.
- Losses: classification, center/offset, box regression, orientation.
- Metrics: BEV AP, 3D AP, NDS depending on dataset.
- Failure modes: sparse points, occlusion, truncation, sensor misalignment.
- Practical interview questions:
  - Why is BEV popular for autonomous driving?
  - Camera-only vs LiDAR-based 3D detection?

### BEV Perception Good to know
- Problem definition: represent scene in bird’s-eye view for detection, occupancy, motion, planning.
- Common models: BEVFormer, Lift-Splat-Shoot, BEVDepth, occupancy networks.
- Input/output: BEV feature map, occupancy grid, objects, lanes, free space.
- Losses: detection, segmentation, occupancy, temporal consistency.
- Metrics: task-specific AP/IoU plus planning-relevant measures.
- Failure modes: calibration drift, long-range uncertainty, temporal inconsistency.
- Practical interview questions:
  - Why is BEV more useful than image-plane reasoning for planning?

### Multimodal Perception Good to know
- Problem definition: fuse camera, LiDAR, radar, text, audio, map, or IMU signals.
- Common models: late fusion, cross-attention fusion, BEV fusion, VLM-based grounding.
- Input/output: task dependent; often more robust scene understanding.
- Losses: task losses plus alignment or contrastive losses.
- Metrics: task specific plus robustness under missing modalities.
- Failure modes: synchronization, calibration, missing-modality failure, conflicting sensor evidence.
- Practical interview questions:
  - Early vs late fusion?
  - What if one sensor degrades?

## 2.E Metrics And Evaluation

### Accuracy Must know
- Definition: fraction of correct predictions.
- When to use: balanced multi-class classification with equal error costs.
- Limitations: misleading under imbalance or asymmetric costs.
- Easy interview explanation: "Accuracy tells me how often I am right overall, but not what kinds of mistakes I make."

### Precision Must know
- Definition: `TP / (TP + FP)`.
- When to use: false positives are costly.
- Limitations: can be high while recall is poor.
- Easy explanation: "Of the things I predicted positive, how many were actually positive?"

### Recall Must know
- Definition: `TP / (TP + FN)`.
- When to use: false negatives are costly.
- Limitations: can be high with many false positives.
- Easy explanation: "Of all true positives, how many did I recover?"

### F1 Must know
- Definition: harmonic mean of precision and recall.
- When to use: balance between precision and recall matters.
- Limitations: hides class-level asymmetry and threshold sensitivity.
- Easy explanation: "F1 is useful when both missing positives and raising false alarms matter."

### ROC / PR Curves [HIGH-YIELD] Must know
- Definition:
  - ROC plots TPR vs FPR;
  - PR plots precision vs recall.
- When to use:
  - ROC for balanced settings,
  - PR for strong imbalance.
- Limitations: ROC can look optimistic on rare positives.
- Easy explanation: "PR is more honest when positives are rare."

### IoU [HIGH-YIELD] Must know
- Definition: overlap over union between predicted and target regions or boxes.
- When to use: localization quality in detection/segmentation.
- Limitations: insensitive to calibration or class confidence.
- Easy explanation: "IoU answers how much the predicted region overlaps the true region."

### mAP [HIGH-YIELD] Must know
- Definition: mean average precision across classes, often across IoU thresholds.
- When to use: detection and instance segmentation.
- Limitations: harder to interpret operationally; can hide category-specific weakness.
- Easy explanation: "mAP rewards both correct ranking by confidence and sufficiently accurate localization."

### Confusion Matrix Must know
- Definition: counts of predicted vs true classes.
- When to use: multi-class error analysis.
- Limitations: not enough alone when thresholding or calibration matters.
- Easy explanation: "It tells me which classes the model confuses, not just how often."

### Calibration [HIGH-YIELD] Good to know
- Definition: alignment between predicted confidence and empirical correctness.
- When to use: safety-critical or decision-making settings.
- Limitations: a model can have good accuracy and poor calibration.
- Easy explanation: "If the model says 0.9 confidence, it should be correct about 90% of the time."

### Tracking Metrics Good to know
- Definition: MOTA, IDF1, HOTA, ID switches, mostly balancing detection and identity consistency.
- When to use: multi-object tracking.
- Limitations: MOTA can hide identity quality; HOTA is richer but less intuitive.
- Easy explanation: "Tracking is not just detecting objects, it is preserving identity over time."

### Segmentation Metrics Must know
- Definition: mIoU, Dice, pixel accuracy, boundary F-score.
- When to use: semantic and medical segmentation.
- Limitations: pixel accuracy is weak under severe background dominance.
- Easy explanation: "mIoU is stronger than accuracy because it penalizes overlap quality per class."

### Depth Metrics Good to know
- Definition: Abs Rel, RMSE, Sq Rel, log RMSE, threshold accuracies such as `δ < 1.25`.
- When to use: depth estimation.
- Limitations: aggregate errors can miss catastrophic far-range behavior.
- Easy explanation: "Different metrics emphasize absolute error, relative scale, or multiplicative tolerance."

## 2.F Data And Training

### Annotation Quality [HIGH-YIELD] Must know
- What it is: consistency and correctness of labels.
- Why it matters: label noise often upper-bounds performance.
- Interview questions:
  - How do you assess annotation quality?
  - What label issues matter more for boxes than masks?
- `[COMMON TRAP]` Blaming architecture before checking labels.

### Noisy Labels Must know
- What it is: wrong, ambiguous, incomplete, or inconsistent labels.
- Why it matters: especially common in detection, segmentation, long-tail datasets.
- Interview questions:
  - How would you detect noisy labels?
  - Would you relabel or use robust loss?
- `[COMMON TRAP]` Treating all noisy labels the same; missing missing-label bias in detection.

### Train / Val / Test Splits [HIGH-YIELD] Must know
- What it is: strict separation for model fitting, tuning, and final evaluation.
- Why it matters: leakage is a silent killer.
- Interview questions:
  - How do you split video data?
  - Why not random split near-duplicate frames?
- `[COMMON TRAP]` Randomly splitting correlated scenes or time-adjacent samples.

### Data Leakage [HIGH-YIELD] Must know
- What it is: train-time access to information that will not exist at inference.
- Why it matters: can create unrealistically high metrics.
- Interview questions:
  - Give examples of leakage in CV.
  - How can augmentations or preprocessing cause leakage?
- `[COMMON TRAP]` Thinking leakage only means train/test duplication.

### Augmentation Strategies [HIGH-YIELD] Must know
- What it is: transformations that preserve semantics while broadening training distribution.
- Why it matters: core generalization lever.
- Interview questions:
  - Which augmentations are valid for detection, segmentation, or pose?
  - Why can heavy augmentation hurt?
- `[COMMON TRAP]` Applying image-level augmentations that invalidate geometric labels.

### Sampling And Long-Tail Distributions Must know
- What it is: class frequency imbalance and rare examples underrepresented in batches.
- Why it matters: models overfit dominant classes.
- Interview questions:
  - Reweighting vs resampling?
  - Why can oversampling rare classes hurt calibration?
- `[COMMON TRAP]` Ignoring batch construction in long-tail problems.

### Hard Negative Mining Must know
- What it is: focusing training on confusing negative examples.
- Why it matters: especially useful in detection and metric learning.
- Interview questions:
  - Why are easy negatives not enough?
  - What are risks of overly aggressive hard mining?
- `[COMMON TRAP]` Mining examples that are actually mislabeled or unlearnable.

### Overfitting / Underfitting [HIGH-YIELD] Must know
- What it is:
  - overfitting: train improves, val degrades,
  - underfitting: both poor.
- Why it matters: candidate must debug training behavior quickly.
- Interview questions:
  - What would you change first?
- `[COMMON TRAP]` Recommending more model capacity for overfitting.

### Debugging Training [HIGH-YIELD] Must know
- What it is: systematic diagnosis of optimization, data, label, and implementation issues.
- Why it matters: practical interviews value debugging more than architecture trivia.
- Interview questions:
  - What would you do if loss does not go down?
  - How do you test the training pipeline?
- `[COMMON TRAP]` Not suggesting "overfit one small batch" early.

### Error Analysis And Ablation Studies [HIGH-YIELD] Must know
- What it is: structured breakdown of failure modes and controlled experiments.
- Why it matters: separates real engineering from random experimentation.
- Interview questions:
  - How do you decide the next experiment?
  - What makes an ablation credible?
- `[COMMON TRAP]` Reporting one global metric without slice analysis.

## 2.G Systems / Production CV

### Latency Vs Accuracy Tradeoff [HIGH-YIELD] Must know
- What it is: operating point between quality and real-time constraints.
- Why it matters: most deployed systems are constrained.
- Interview questions:
  - How do you optimize for 30 FPS?
  - When would you accept lower mAP?
- `[COMMON TRAP]` Focusing only on model FLOPs and ignoring preprocessing/postprocessing.

### Memory Bottlenecks Must know
- What it is: activations, batch size, feature maps, sequence length, and intermediate buffers can dominate memory.
- Why it matters: memory often limits deployment more than compute.
- Interview questions:
  - Why do high-resolution detectors use so much memory?
  - Params vs activations?
- `[COMMON TRAP]` Assuming fewer parameters always means lower runtime memory.

### Throughput And Batching Must know
- What it is: images per second vs single-sample latency.
- Why it matters: cloud inference may optimize throughput, edge may optimize latency.
- Interview questions:
  - Why does batching help throughput but hurt latency?
- `[COMMON TRAP]` Treating latency and throughput as the same metric.

### Preprocessing And Postprocessing Must know
- What it is: resize, normalize, color conversion, NMS, thresholding, tracking association, smoothing.
- Why it matters: production regressions often come from non-model code.
- Interview questions:
  - What if train and inference preprocessing do not match?
  - Why can NMS dominate latency?
- `[COMMON TRAP]` Talking only about the neural network.

### Quantization / Pruning / Distillation [HIGH-YIELD] Must know
- What it is: compression and acceleration methods.
- Why it matters: common interview topic for deployment roles.
- Interview questions:
  - When do you choose quantization vs distillation?
  - Why can INT8 hurt small-object detection?
- `[COMMON TRAP]` Treating compression as accuracy-free.

### Edge Deployment Good to know
- What it is: deployment under tight compute, thermal, memory, and power constraints.
- Why it matters: robotics, mobile, AR, automotive, medical devices.
- Interview questions:
  - What changes for on-device inference?
  - How do you handle intermittent connectivity?
- `[COMMON TRAP]` Assuming server-side solutions transfer directly.

### Failure Analysis In Production [HIGH-YIELD] Must know
- What it is: systematic analysis of real-world bad cases.
- Why it matters: production failures are rarely random.
- Interview questions:
  - How do you categorize failures?
  - What logs or artifacts do you store?
- `[COMMON TRAP]` Only checking aggregate metrics and not collecting difficult slices.

### Monitoring Drift And Robustness Good to know
- What it is: distribution shift detection, confidence monitoring, calibration, sensor health.
- Why it matters: long-lived systems degrade silently.
- Interview questions:
  - How would you monitor a deployed detector?
  - How do you detect domain drift without labels?
- `[COMMON TRAP]` Assuming high offline validation means stable field performance.

## 2.H Modern CV Trends

### Vision Transformers [HIGH-YIELD] Good to know
- Interview focus: when ViTs outperform CNNs, why pretraining matters, memory/sequence cost, local vs global inductive bias.
- `[COMMON TRAP]` Over-selling ViTs without discussing data and compute requirements.

### Foundation Vision Models Good to know
- Interview focus: large pretrained backbones, generalization, promptability, transfer efficiency, adaptation costs.
- `[COMMON TRAP]` Talking about them like product buzzwords instead of model families and deployment tradeoffs.

### Multimodal Models / VLMs Good to know
- Interview focus: image-text alignment, grounding, retrieval, open-vocabulary recognition, prompt sensitivity, hallucination risk.
- `[COMMON TRAP]` Assuming language supervision solves detection reliability by default.

### Open-Vocabulary Detection Good to know
- Interview focus: text-conditioned detection, CLIP-style pretraining, label-space flexibility vs localization quality.
- `[COMMON TRAP]` Ignoring long-tail coverage and threshold calibration.

### Segmentation Foundation Models Good to know
- Interview focus: promptable segmentation, zero-shot interaction, human-in-the-loop labeling acceleration.
- `[COMMON TRAP]` Confusing interactive segmentation with fully autonomous production segmentation.

### Diffusion Relevance To CV Bonus
- Interview focus: denoising as generative modeling, representation learning, data synthesis, editing, inverse problems.
- `[COMMON TRAP]` Going too deep into generative math when the role is perception-focused.

### Self-Supervised Learning Good to know
- Interview focus: contrastive vs masked modeling, better representation learning under limited labels, transfer performance.
- `[COMMON TRAP]` Not explaining why SSL matters in data-scarce settings.

### Synthetic Data Good to know
- Interview focus: domain randomization, sim-to-real gap, annotation scalability, distribution mismatch.
- `[COMMON TRAP]` Assuming synthetic data automatically replaces real data.

### Foundation Models For Robotics / Autonomy Bonus
- Interview focus: embodied grounding, temporal consistency, safety, map/context fusion, action relevance.
- `[COMMON TRAP]` Talking about general intelligence instead of task reliability.

---

## 3. Topic-By-Topic Revision Notes

## 3.1 Convolution And Filtering [HIGH-YIELD]
- Definition: convolution applies a local kernel across an image to aggregate spatial neighborhoods.
- Intuition: each output pixel is a question about a local pattern such as smoothness, contrast, orientation, or texture.
- Key equations:
  - 2D convolution: `(I * K)(u,v) = Σ_i Σ_j I(u-i, v-j) K(i,j)`
  - CNN output size: `floor((H + 2P - D(K-1) - 1)/S + 1)`
- Interview answer version: "Convolution combines locality, parameter sharing, and translation-equivariant filtering. In classical CV I use it for smoothing or derivatives; in CNNs the kernels are learned."
- Deeper explanation version: discuss separability, frequency response, aliasing, padding, stride, and how stacked small kernels approximate larger receptive fields.
- Practical examples: Gaussian blur before Canny; Sobel for gradients; first CNN layer learning edge-like kernels.
- Common follow-up questions:
  - Why blur before gradient estimation?
  - Why is max pooling not the same as stride?
  - What is the difference between theoretical and effective receptive field?
- Common traps:
  - `[COMMON TRAP]` Forgetting that large stride causes aliasing and localization loss.
  - `[COMMON TRAP]` Calling convolution invariant rather than equivariant.
- Compare with adjacent topics:
  - filtering vs attention: fixed local weights vs input-dependent global weights;
  - convolution vs template matching: learned local pattern detector vs fixed similarity search.

## 3.2 Keypoints, Descriptors, And Matching [HIGH-YIELD]
- Definition: detect repeatable points and compute descriptors that can be matched across views.
- Intuition: good keypoints are stable under viewpoint and photometric change; good descriptors are distinctive but compact.
- Key equations:
  - structure tensor for corner reasoning,
  - ratio test for nearest-neighbor matching in descriptor space.
- Interview answer version: "I use keypoints for correspondence. Corners are good because they have strong variation in two directions. SIFT gives robust float descriptors; ORB is much faster with binary descriptors."
- Deeper explanation version: scale-space extrema, orientation assignment, histogram descriptors, Hamming vs L2 distance, and geometric verification with RANSAC.
- Practical examples: panorama stitching, pose estimation from matched landmarks, loop closure candidates.
- Common follow-ups:
  - Why is an edge ambiguous for matching?
  - Why is descriptor matching not enough without geometry?
- Common traps:
  - `[COMMON TRAP]` Ignoring outlier rejection after descriptor matching.
  - `[COMMON TRAP]` Claiming ORB has the same robustness as SIFT in difficult matching.
- Compare with adjacent topics:
  - local features vs dense descriptors;
  - handcrafted descriptors vs learned descriptors like SuperPoint/SuperGlue pipelines.

## 3.3 Optical Flow Vs Tracking [HIGH-YIELD]
- Definition:
  - optical flow: pixel motion field,
  - tracking: object identity persistence over time.
- Intuition: flow is local motion; tracking is object-level temporal association.
- Key equations:
  - brightness constancy: `I_x u + I_y v + I_t = 0`.
- Interview answer version: "Optical flow estimates apparent motion for pixels. Tracking typically starts with detections and solves association over time, often with motion and appearance cues."
- Deeper explanation version: aperture problem, occlusion handling, motion models, re-identification embeddings, association costs.
- Practical examples: stabilization uses flow; multi-object tracking in retail or traffic uses detection + association.
- Common follow-ups:
  - Why does flow fail at occlusions?
  - Why can better detection help tracking more than a better association model?
- Common traps:
  - `[COMMON TRAP]` Equating flow with object tracking.
  - `[COMMON TRAP]` Ignoring camera motion in tracking discussion.
- Compare with adjacent topics:
  - sparse flow vs dense flow;
  - detection-based tracking vs joint detection-tracking models.

## 3.4 Camera Model And Calibration [HIGH-YIELD]
- Definition: estimate camera intrinsics and distortion to relate 3D geometry to pixels.
- Intuition: calibration tells you how the sensor projects rays; pose tells you where the camera is.
- Key equations:
  - `s x = K [R|t] X`
  - radial distortion polynomial.
- Interview answer version: "Intrinsics describe the camera itself; extrinsics describe its pose relative to a frame. Calibration estimates intrinsics and distortion so geometry downstream is meaningful."
- Deeper explanation version: principal point, skew, normalized coordinates, reprojection error, calibration target coverage, real-world pitfalls like autofocus or rolling shutter.
- Practical examples: stereo rigs, robot cameras, industrial metrology.
- Common follow-ups:
  - Why can low reprojection error still be misleading?
  - What happens if you resize images after calibration?
- Common traps:
  - `[COMMON TRAP]` Mixing world-to-camera and camera-to-world transforms.
  - `[COMMON TRAP]` Ignoring distortion when discussing feature matching or stereo.
- Compare with adjacent topics:
  - calibration vs pose estimation;
  - pinhole ideal model vs real lens model.

## 3.5 Homography Vs Fundamental Vs Essential Matrix [HIGH-YIELD]
- Definition:
  - homography: planar projective transform,
  - fundamental matrix: uncalibrated two-view epipolar geometry,
  - essential matrix: calibrated two-view geometry.
- Intuition: homography explains a plane; `F`/`E` explain rigid 3D geometry between cameras.
- Key equations:
  - `x' ~ Hx`
  - `x'^T F x = 0`
  - `x'^T E x = 0`, `E = [t]_x R`
- Interview answer version: "If the scene is planar or motion is pure rotation, homography is appropriate. For general two-view geometry I estimate `F` or `E` depending on whether cameras are calibrated."
- Deeper explanation version: epipoles, rank constraints, scale ambiguity in translation, degenerate motion, Hartley normalization, RANSAC.
- Practical examples: panorama stitching with `H`; relative pose from calibrated matches with `E`.
- Common follow-ups:
  - Why does homography work for pure rotation?
  - Why can `E` recover translation only up to scale?
- Common traps:
  - `[COMMON TRAP]` Using homography for arbitrary non-planar scenes.
  - `[COMMON TRAP]` Saying `F` and `E` are the same with different notation.
- Compare with adjacent topics:
  - homography vs affine transform;
  - `E`/`F` estimation vs PnP.

## 3.6 Triangulation, Stereo, PnP, Bundle Adjustment [HIGH-YIELD]
- Definition:
  - triangulation: 2D correspondences + poses -> 3D point,
  - stereo: dense disparity -> depth,
  - PnP: 3D points + 2D observations -> pose,
  - bundle adjustment: jointly refine poses and 3D structure.
- Intuition: these are inverse problems with different knowns and unknowns.
- Key equations:
  - stereo depth: `Z = fB/d`
  - BA objective: minimize reprojection error over poses and 3D points.
- Interview answer version: "Triangulation and PnP are dual in a sense: one solves for 3D given pose, the other solves for pose given 3D."
- Deeper explanation version: cheirality, baseline, uncertainty, outlier rejection, nonlinear refinement, sparse Jacobians.
- Practical examples: AR marker pose, SfM pipelines, stereo depth for robotics.
- Common follow-ups:
  - Why is small disparity unreliable far away?
  - Why use RANSAC with PnP?
  - Why is BA expensive but valuable?
- Common traps:
  - `[COMMON TRAP]` Forgetting scale ambiguity in monocular systems.
  - `[COMMON TRAP]` Treating triangulated points as accurate without discussing pose/match uncertainty.
- Compare with adjacent topics:
  - PnP vs ICP;
  - stereo vs monocular depth;
  - BA vs pose graph optimization.

## 3.7 Detection Pipelines [HIGH-YIELD]
- Definition: localize and classify objects, usually via one-stage or two-stage pipelines.
- Intuition: detection requires both recognition and localization under scale variation and imbalance.
- Key equations:
  - IoU,
  - focal loss,
  - AP as area under precision-recall curve.
- Interview answer version: "Two-stage detectors usually buy accuracy with proposal refinement; one-stage detectors optimize speed with dense prediction. Feature pyramids help with scale variation."
- Deeper explanation version: anchors vs anchor-free, objectness, box parameterization, label assignment, NMS, DETR set prediction.
- Practical examples: retail shelf detection, traffic perception, defect localization.
- Common follow-ups:
  - Why is localization harder for small objects?
  - Why does NMS create recall tradeoffs?
  - Why can anchor design matter?
- Common traps:
  - `[COMMON TRAP]` Discussing only the backbone and ignoring assignment, loss, and postprocessing.
  - `[COMMON TRAP]` Using accuracy as the main detection metric.
- Compare with adjacent topics:
  - classification vs detection;
  - detection vs instance segmentation;
  - one-stage vs two-stage vs DETR.

## 3.8 Segmentation Family [HIGH-YIELD]
- Definition:
  - semantic segmentation labels each pixel,
  - instance segmentation separates object instances,
  - panoptic segmentation unifies both.
- Intuition: richer scene understanding than boxes, but more annotation cost and harder boundaries.
- Key equations:
  - IoU,
  - Dice coefficient.
- Interview answer version: "The main difference is whether instance identity matters. Semantic segmentation merges same-class objects; instance segmentation separates them; panoptic combines stuff and things."
- Deeper explanation version: encoder-decoder design, skip connections, resolution recovery, mask heads, transformer decoders.
- Practical examples: drivable area, medical masks, inventory counting.
- Common follow-ups:
  - Why is pixel accuracy weak for segmentation?
  - Why use Dice for class imbalance?
- Common traps:
  - `[COMMON TRAP]` Confusing semantic and instance segmentation outputs.
  - `[COMMON TRAP]` Ignoring boundary quality and small-region failure.
- Compare with adjacent topics:
  - detection vs segmentation;
  - semantic vs panoptic; U-Net vs DeepLab vs Mask2Former.

## 3.9 Tracking [HIGH-YIELD]
- Definition: maintain identities across frames.
- Intuition: track management is as important as detection quality.
- Key metrics:
  - MOTA, IDF1, HOTA, ID switches.
- Interview answer version: "Modern tracking-by-detection systems often combine detection confidence, motion prediction, gating, and appearance embeddings. Identity stability is the hard part."
- Deeper explanation version: Kalman filtering, Hungarian assignment, ReID embeddings, ByteTrack low-score association, occlusion handling.
- Practical examples: sports analytics, warehouse monitoring, autonomous perception.
- Common follow-ups:
  - Why does association fail after long occlusions?
  - What is motion gating?
- Common traps:
  - `[COMMON TRAP]` Talking only about object trajectories and ignoring ID preservation.
  - `[COMMON TRAP]` Ignoring detector confidence threshold effects.
- Compare with adjacent topics:
  - tracking vs optical flow;
  - online vs offline tracking;
  - appearance-based vs motion-based association.

## 3.10 CNNs, FPNs, And Normalization [HIGH-YIELD]
- Definition: core design elements of many vision backbones.
- Intuition: local features get progressively more semantic; FPN recovers multi-scale utility; normalization stabilizes training.
- Key equations:
  - BN: `y = gamma * (x - mu) / sqrt(var + eps) + beta`.
- Interview answer version: "CNN backbones build hierarchical features. FPN mixes low-resolution semantics with high-resolution detail, which is why it helps small objects. Normalization choice depends on batch regime and architecture."
- Deeper explanation version: residual bottlenecks, depthwise convs, ConvNeXt modernization, BN vs GN in detection.
- Practical examples: ResNet-FPN detector, U-Net decoder skips, ConvNeXt backbone in segmentation.
- Common follow-ups:
  - Why does GroupNorm help in detection?
  - What if batch size is 2 per GPU?
- Common traps:
  - `[COMMON TRAP]` Choosing BN in a tiny-batch segmentation setup without discussion.
- Compare with adjacent topics:
  - FPN vs dilated backbone;
  - BN vs LN vs GN.

## 3.11 ViTs, Attention, And DETR [HIGH-YIELD]
- Definition: transformer-based approaches for image representation and detection.
- Intuition: attention lets any token look at any other token, giving flexible long-range context.
- Key equations:
  - `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
- Interview answer version: "ViTs trade CNN locality bias for global token mixing. They scale well with data and pretraining. DETR frames detection as set prediction rather than dense anchor classification."
- Deeper explanation version: patch embeddings, positional encodings, self-attention cost, window attention, cross-attention in decoders, Hungarian matching.
- Practical examples: ViT fine-tuning on classification; Deformable DETR in detection; SAM-like promptable segmentation.
- Common follow-ups:
  - Why is DETR convergence slower?
  - Why do ViTs tend to need more data?
- Common traps:
  - `[COMMON TRAP]` Claiming attention is always better for images.
  - `[COMMON TRAP]` Missing the role of positional information.
- Compare with adjacent topics:
  - CNN vs ViT;
  - DETR vs RetinaNet/Faster R-CNN.

## 3.12 Data, Metrics, And Debugging [HIGH-YIELD]
- Definition: the decision-making layer around models.
- Intuition: most real gains come from better data and better failure analysis.
- Key equations:
  - precision/recall/F1,
  - ECE concept,
  - confusion matrix interpretation.
- Interview answer version: "Before changing the model, I verify the split, labels, preprocessing, metric, and failure slices. Then I run controlled ablations."
- Deeper explanation version: leakage pathways, long-tail failure, threshold selection, calibration, slice-based evaluation, error buckets.
- Practical examples: false positives at night, small-object miss rate, mask boundary failures, cross-camera drift.
- Common follow-ups:
  - How would you debug if mAP improved but product performance worsened?
  - What slices would you inspect?
- Common traps:
  - `[COMMON TRAP]` Reporting one averaged metric without class, scale, or environment slices.
  - `[COMMON TRAP]` Confusing model confidence with calibration.
- Compare with adjacent topics:
  - offline validation vs online monitoring;
  - aggregate metric vs operational quality.

---

## 4. Whiteboard And Coding Round Preparation

### 4.1 Convolution Implementation Intuition [HIGH-YIELD]
- What they may ask:
  - implement 2D convolution,
  - derive output shape,
  - explain stride/padding,
  - optimize naive loops.
- What a good answer looks like:
  - explicitly states input shape `(B, C, H, W)` or `(H, W)`,
  - handles padding carefully,
  - distinguishes cross-correlation from mathematical convolution,
  - comments on time complexity.
- Common coding mistakes:
  - off-by-one output size errors,
  - forgetting channel dimension,
  - mixing up batch and channel axes,
  - applying padding after instead of before indexing.

### 4.2 IoU Computation [HIGH-YIELD]
- What they may ask:
  - implement IoU for two boxes or vectorized IoU for `N x M` boxes.
- What a good answer looks like:
  - computes intersection with clamped positive width/height,
  - handles non-overlap cleanly,
  - clarifies coordinate format `[x1, y1, x2, y2]`.
- Common coding mistakes:
  - negative intersection area,
  - using center-width-height format accidentally,
  - forgetting epsilon in division.

### 4.3 NMS [HIGH-YIELD]
- What they may ask:
  - implement greedy NMS,
  - explain Soft-NMS,
  - discuss batched NMS.
- What a good answer looks like:
  - sort by score,
  - iteratively keep highest score and suppress overlaps,
  - notes class-wise NMS when relevant.
- Common coding mistakes:
  - not sorting,
  - suppressing across different classes incorrectly,
  - failing on empty input.

### 4.4 Data Augmentation Pipeline
- What they may ask:
  - design augmentations for classification, detection, segmentation, keypoints.
- What a good answer looks like:
  - distinguishes photometric vs geometric transforms,
  - updates labels for spatial transforms,
  - avoids label-breaking augmentations.
- Common coding mistakes:
  - flipping images without flipping keypoints,
  - resizing boxes incorrectly,
  - applying normalization inconsistently between train and inference.

### 4.5 Image Transforms And Coordinate Systems
- What they may ask:
  - resize an image and update boxes,
  - crop and map points,
  - convert between normalized and pixel coordinates.
- What a good answer looks like:
  - clearly defines coordinate convention,
  - tracks scale factors and offsets,
  - handles integer rounding policy consciously.
- Common coding mistakes:
  - mixing width/height order,
  - assuming origin at bottom-left instead of top-left,
  - forgetting that interpolation changes label semantics for masks.

### 4.6 Attention Basics For Vision
- What they may ask:
  - implement scaled dot-product attention,
  - explain mask shape,
  - compare self-attention and cross-attention.
- What a good answer looks like:
  - states shapes for `Q, K, V`,
  - explains scaling by `sqrt(d_k)`,
  - handles causal or padding mask correctly.
- Common coding mistakes:
  - wrong transpose dimension,
  - applying mask after softmax,
  - forgetting that attention output keeps value dimension.

### 4.7 Writing Loss Functions
- What they may ask:
  - implement Dice loss, focal loss, contrastive loss.
- What a good answer looks like:
  - uses logits correctly,
  - explains reduction and numerical stability,
  - checks target format.
- Common coding mistakes:
  - applying softmax twice,
  - using BCE with integer class labels,
  - flattening tensors incorrectly for segmentation.

### 4.8 Debugging Shape Mismatches [HIGH-YIELD]
- What they may ask:
  - debug failing training code or tensor shape error.
- What a good answer looks like:
  - prints shapes at stage boundaries,
  - writes down expected shape per layer,
  - checks batch dimension first,
  - checks channel-last vs channel-first assumptions.
- Common coding mistakes:
  - blindly reshaping to make code run,
  - flattening away spatial dimensions before loss,
  - ignoring broadcasting bugs.

### 4.9 Training Loop Bugs [HIGH-YIELD]
- What they may ask:
  - identify why model does not learn.
- What a good answer looks like:
  - checks `model.train()` vs `eval()`,
  - checks optimizer zeroing,
  - checks loss decreases on one batch,
  - verifies labels and preprocessing,
  - checks learning rate and gradient norms.
- Common coding mistakes:
  - forgetting `zero_grad`,
  - wrong label dtype,
  - not moving targets to device,
  - validation under train mode.

### 4.10 Inference Pipeline Issues
- What they may ask:
  - why offline validation is good but production output is bad.
- What a good answer looks like:
  - compares train-time and inference-time preprocessing,
  - checks image resize policy, aspect ratio, normalization, thresholds, NMS, batching behavior.
- Common coding mistakes:
  - mismatched color space,
  - forgetting to undo normalization,
  - different interpolation,
  - inconsistent score threshold or NMS threshold.

### 4.11 Useful Coding-Round Habits
- `[GOOD INTERVIEW LINE] "I’ll start with the tensor shapes so we can agree on the contract before I code."`
- `[GOOD INTERVIEW LINE] "I’ll write the correct version first, then vectorize if there’s time."`
- `[GOOD INTERVIEW LINE] "For detection or segmentation code, I want to be explicit about coordinate format and label updates."`

---

## 5. Project Deep-Dive Preparation

### 5.1 Reusable Framework For Explaining A CV Project [HIGH-YIELD]
- Problem statement:
  - What exact task were you solving?
  - Why was it hard?
  - What was the operational constraint?
- Data:
  - Dataset size, source, annotation type, class balance, sensor setup.
  - What was noisy or missing?
- Model choice:
  - Why this architecture rather than obvious alternatives?
  - What inductive bias or system tradeoff justified it?
- Losses:
  - What losses did you use and why do they match the task?
- Metrics:
  - Which offline metrics mattered?
  - Which online or business metric mattered?
- Training strategy:
  - Initialization, augmentation, sampling, schedule, validation plan.
- Bottlenecks:
  - Data bottleneck, model bottleneck, optimization bottleneck, system bottleneck.
- Failure modes:
  - Small objects, long tail, occlusion, night scenes, shiny surfaces, calibration error, missing labels.
- Debugging steps:
  - How did you isolate the main issue?
  - What ablations changed your mind?
- What improved results:
  - Specific interventions and measured lift.
- Productionization / deployment concerns:
  - Latency, memory, model compression, calibration monitoring, fallback logic.
- Tradeoffs:
  - What did you sacrifice to achieve the final result?
- What you would do next:
  - Next highest-value experiments if you had more time.

### 5.2 Strong Project Story Template
- Context: one sentence on user or system need.
- Baseline: what existed before.
- Gap: what was failing and how you proved it.
- Approach: model + data + systems choices.
- Result: concrete metric improvements and operational effect.
- Reflection: what was learned and what remains unsolved.

### 5.3 Fifteen Strong Interrogation Questions For Project Deep Dives

#### 1. Why did you choose this task formulation?
- Weak answer: "It is what people usually use."
- Strong answer: "The decision was tied to output requirements. Boxes were insufficient because downstream needed pixel-level free-space boundaries, so semantic segmentation was the right abstraction."

#### 2. Why this model family and not the obvious alternative?
- Weak answer: "It gave the best benchmark score."
- Strong answer: "We needed high recall on small objects under a 20 ms latency budget, so we used a one-stage detector with an FPN rather than a heavier two-stage model."

#### 3. What was the hardest failure mode?
- Weak answer: "Generalization."
- Strong answer: "Night scenes with reflective surfaces caused false positives and low-confidence misses because training data underrepresented those photometric conditions."

#### 4. How did you know the problem was in the data and not the model?
- Weak answer: "We suspected data issues."
- Strong answer: "Slice analysis showed large degradation in a specific capture domain, and manual audit revealed annotation inconsistency. Architecture changes alone gave little gain."

#### 5. What was your evaluation metric and why?
- Weak answer: "We used mAP because everyone uses it."
- Strong answer: "We used mAP offline, but recall at a fixed precision threshold was more aligned with the operational cost of misses."

#### 6. How did you debug training when metrics plateaued?
- Weak answer: "We tuned hyperparameters."
- Strong answer: "We first overfit a small subset, verified label alignment, inspected gradient norms, then ran ablations on augmentation strength and sampling."

#### 7. What did your ablations show?
- Weak answer: "A few changes improved performance."
- Strong answer: "Most of the gain came from label cleanup and class-balanced sampling; architecture swaps mattered less than expected."

#### 8. Where did the biggest lift come from?
- Weak answer: "Using a better backbone."
- Strong answer: "The biggest lift came from fixing label noise and adding hard-negative mining around visually similar failure cases."

#### 9. How did you handle class imbalance?
- Weak answer: "We used focal loss."
- Strong answer: "We combined focal loss with class-aware sampling and threshold analysis, because focal loss alone improved recall but hurt calibration on minority classes."

#### 10. What were the latency bottlenecks?
- Weak answer: "The model was large."
- Strong answer: "Preprocessing resize and NMS were a surprising part of the end-to-end latency; compression helped the backbone, but postprocessing required separate optimization."

#### 11. How did you validate robustness?
- Weak answer: "The validation set was diverse."
- Strong answer: "We defined environment slices for lighting, weather, occlusion, and camera position, then compared per-slice degradation to the main metric."

#### 12. What would you do if deployed performance drifted?
- Weak answer: "Retrain the model."
- Strong answer: "I would first identify whether drift came from sensor calibration, upstream preprocessing, environment shift, or class prior shift before scheduling relabeling and retraining."

#### 13. What part of the project are you least satisfied with?
- Weak answer: "Nothing major."
- Strong answer: "Calibration and uncertainty estimation remained weak, so the model confidence did not map well to decision quality in rare conditions."

#### 14. What did you personally own?
- Weak answer: "I worked on the model."
- Strong answer: "I owned data audit, training pipeline, metric design, and the detector-threshold tuning used for deployment."

#### 15. What would you do next with two more months?
- Weak answer: "Try larger models."
- Strong answer: "I would prioritize targeted relabeling for the top failure slices, improve calibration, and add a lightweight temporal consistency module before considering model scaling."

### 5.4 What Weak Answers Usually Look Like
- Benchmark recitation with no dataset or failure discussion.
- Architecture name-dropping without justification.
- No slice-based evaluation.
- No mention of data quality or annotation noise.
- No concrete bottlenecks.
- No tradeoff discussion.
- No understanding of what actually improved the system.

### 5.5 What Strong Answers Usually Look Like
- Clear task definition and constraint framing.
- Data quality discussion before architecture excitement.
- Specific, measured interventions.
- Honest description of failure modes.
- Good metric choice and calibration awareness.
- Practical deployment reasoning.

---

## 6. Common Interview Questions

## 6.1 Beginner Questions

1. What is convolution in images?
- Ideal answer outline: local weighted aggregation over neighborhoods; useful for edge, texture, and pattern extraction; parameter sharing gives efficiency.
- Interviewer is testing: first-principles understanding.
- Optional follow-up: What is the difference between convolution and correlation?

2. Why do we use padding in CNNs?
- Ideal answer outline: preserve spatial size, control border behavior, avoid shrinking too quickly.
- Interviewer is testing: shape reasoning.
- Optional follow-up: What is "same" padding?

3. What does an edge detector compute?
- Ideal answer outline: intensity gradient magnitude/direction; edges correspond to strong local changes.
- Interviewer is testing: classical CV basics.
- Optional follow-up: Why smooth before computing derivatives?

4. Why are corners better than edges for matching?
- Ideal answer outline: corners vary in two directions; edges are ambiguous along the tangent direction.
- Interviewer is testing: feature intuition.
- Optional follow-up: What is the structure tensor?

5. SIFT vs ORB?
- Ideal answer outline: SIFT is more robust and float-based; ORB is faster, binary, and cheaper to match.
- Interviewer is testing: practical tradeoff thinking.
- Optional follow-up: When would you choose ORB?

6. What is optical flow?
- Ideal answer outline: apparent pixel motion between frames; dense or sparse; not the same as object tracking.
- Interviewer is testing: motion understanding.
- Optional follow-up: What is the aperture problem?

7. Difference between classification, detection, and segmentation?
- Ideal answer outline: image label vs boxes + labels vs per-pixel labels.
- Interviewer is testing: task formulation clarity.
- Optional follow-up: Instance vs semantic segmentation?

8. Precision vs recall?
- Ideal answer outline: precision penalizes false positives; recall penalizes false negatives.
- Interviewer is testing: metric literacy.
- Optional follow-up: Which matters more in medical screening?

9. What is IoU?
- Ideal answer outline: overlap over union for boxes or masks; measures localization quality.
- Interviewer is testing: detection/segmentation basics.
- Optional follow-up: Why is IoU better than raw overlap area?

10. Why use data augmentation?
- Ideal answer outline: improve generalization, expose model to plausible variation, reduce overfitting.
- Interviewer is testing: training intuition.
- Optional follow-up: Which augmentations are unsafe for detection?

11. How do you know a model is overfitting?
- Ideal answer outline: training improves while validation stagnates or worsens; gap grows.
- Interviewer is testing: debugging maturity.
- Optional follow-up: What would you change first?

12. What is transfer learning?
- Ideal answer outline: initialize from pretrained weights and fine-tune or freeze for a target task.
- Interviewer is testing: practical ML understanding.
- Optional follow-up: When does pretraining help less?

13. Why does BatchNorm help?
- Ideal answer outline: stabilizes optimization, smooths training, reduces sensitivity to initialization.
- Interviewer is testing: optimization understanding.
- Optional follow-up: Why can it fail with tiny batches?

14. What is NMS?
- Ideal answer outline: greedy suppression of overlapping detections after sorting by score.
- Interviewer is testing: detection pipeline understanding.
- Optional follow-up: What is Soft-NMS?

15. Semantic vs instance segmentation?
- Ideal answer outline: semantic gives per-pixel class; instance separates individual objects of the same class.
- Interviewer is testing: output-space precision.
- Optional follow-up: What is panoptic segmentation?

## 6.2 Intermediate Questions

16. What is receptive field and why does it matter?
- Ideal answer outline: input region affecting an output feature; controls context available to predictions.
- Interviewer is testing: architecture intuition.
- Optional follow-up: Effective vs theoretical receptive field?

17. Why does FPN help detection?
- Ideal answer outline: combines high-resolution detail with semantically strong deeper features for multi-scale objects.
- Interviewer is testing: detection backbone understanding.
- Optional follow-up: Why are small objects hard?

18. How do you handle class imbalance?
- Ideal answer outline: reweighting, resampling, focal loss, threshold tuning, data collection.
- Interviewer is testing: practical training strategy.
- Optional follow-up: Why is accuracy misleading here?

19. What problem does focal loss solve?
- Ideal answer outline: down-weights easy negatives, focuses learning on hard examples in dense detection.
- Interviewer is testing: loss-task fit.
- Optional follow-up: When can focal loss hurt?

20. Anchor-based vs anchor-free detection?
- Ideal answer outline: anchors use predefined priors; anchor-free predicts centers or keypoints directly.
- Interviewer is testing: detector design understanding.
- Optional follow-up: Which is easier to tune?

21. One-stage vs two-stage detectors?
- Ideal answer outline: one-stage is faster and simpler; two-stage often gives stronger accuracy and proposal refinement.
- Interviewer is testing: speed-accuracy tradeoff reasoning.
- Optional follow-up: Why might two-stage help small objects?

22. Dice loss vs cross-entropy for segmentation?
- Ideal answer outline: Dice helps overlap quality and imbalance; CE gives stable pixel-wise optimization; often combined.
- Interviewer is testing: segmentation loss design.
- Optional follow-up: Why is Dice common in medical imaging?

23. Which tracking metric would you report and why?
- Ideal answer outline: use IDF1/HOTA for identity quality, MOTA for detection-heavy summary, depending on goal.
- Interviewer is testing: metric-task alignment.
- Optional follow-up: Why is MOTA insufficient alone?

24. Why is mAP more informative than accuracy for detection?
- Ideal answer outline: accuracy ignores localization and confidence ranking; mAP captures both ranking and IoU quality.
- Interviewer is testing: evaluation maturity.
- Optional follow-up: What does mAP@[.5:.95] emphasize?

25. What is calibration and why do you care?
- Ideal answer outline: confidence should match empirical correctness; critical for thresholding and safety.
- Interviewer is testing: reliability thinking.
- Optional follow-up: How would you improve calibration?

26. What is hard negative mining?
- Ideal answer outline: select confusing negatives to sharpen discrimination, especially in detection and retrieval.
- Interviewer is testing: practical optimization strategy.
- Optional follow-up: What is the risk if your negatives are mislabeled?

27. What makes an ablation study credible?
- Ideal answer outline: controlled changes, enough runs, clear baseline, fixed protocol, interpretable conclusions.
- Interviewer is testing: scientific rigor.
- Optional follow-up: Why are coupled changes hard to interpret?

28. How can data leakage happen in CV?
- Ideal answer outline: near-duplicate frames across splits, same scene across train/test, leaked preprocessing statistics, human annotation artifacts.
- Interviewer is testing: evaluation hygiene.
- Optional follow-up: How would you split video data correctly?

29. How should augmentation differ between classification and detection?
- Ideal answer outline: classification mostly preserves label; detection requires updating boxes; some crops invalidate objects.
- Interviewer is testing: label-preservation reasoning.
- Optional follow-up: What about keypoints or masks?

30. Top-down vs bottom-up human pose estimation?
- Ideal answer outline: top-down detects people then keypoints; bottom-up detects joints then groups; tradeoff is speed vs crowded-scene behavior.
- Interviewer is testing: task decomposition understanding.
- Optional follow-up: Which scales better in crowded scenes?

## 6.3 Advanced Questions

31. Why does DETR avoid anchors and NMS?
- Ideal answer outline: formulates detection as set prediction with one-to-one matching between predictions and ground truth.
- Interviewer is testing: transformer-based detection understanding.
- Optional follow-up: Why is Hungarian matching used?

32. Why do ViTs usually want more pretraining data than CNNs?
- Ideal answer outline: weaker built-in locality and translation biases; rely more on data to learn image structure.
- Interviewer is testing: architectural inductive bias understanding.
- Optional follow-up: What changes with strong self-supervised pretraining?

33. Why can self-supervised learning help CV?
- Ideal answer outline: learns transferable representations from unlabeled data, especially useful when labels are scarce or expensive.
- Interviewer is testing: representation learning intuition.
- Optional follow-up: Contrastive vs masked-image modeling?

34. What are the main limitations of open-vocabulary detection?
- Ideal answer outline: localization may lag closed-set detectors; calibration and long-tail precision can be weak; prompt sensitivity matters.
- Interviewer is testing: modern trend realism.
- Optional follow-up: How would you evaluate open-vocabulary performance?

35. Why is monocular depth ambiguous up to scale?
- Ideal answer outline: a single image does not uniquely determine metric distance without additional priors or sensors.
- Interviewer is testing: geometric reasoning.
- Optional follow-up: How can scale be recovered?

36. Why is BEV perception attractive for autonomous driving?
- Ideal answer outline: aligns with planning and occupancy reasoning, fuses multiple cameras naturally, normalizes perspective effects.
- Interviewer is testing: perception-system intuition.
- Optional follow-up: What are the weaknesses of BEV lifting?

37. What happens when multi-camera calibration drifts?
- Ideal answer outline: cross-view fusion degrades, geometry becomes inconsistent, 3D/BEV performance collapses in structured ways.
- Interviewer is testing: systems + geometry awareness.
- Optional follow-up: How would you detect it online?

38. How would you reason about synthetic data for CV?
- Ideal answer outline: useful for scale and annotation, but domain gap must be managed through realism, randomization, and validation on real data.
- Interviewer is testing: data strategy maturity.
- Optional follow-up: When is synthetic data most effective?

39. How would you model uncertainty in CV?
- Ideal answer outline: predictive confidence, calibration, ensembles, MC dropout, heteroscedastic outputs, uncertainty-aware thresholds.
- Interviewer is testing: reliability thinking.
- Optional follow-up: Aleatoric vs epistemic uncertainty?

40. What are the risks in multi-task CV training?
- Ideal answer outline: gradient conflict, task imbalance, metric coupling, optimization instability, harmful shared representation.
- Interviewer is testing: advanced modeling judgment.
- Optional follow-up: How would you weight losses?

## 6.4 Systems / Production Questions

41. Difference between latency and throughput?
- Ideal answer outline: latency is per-request time; throughput is work per unit time; batching can improve throughput while harming latency.
- Interviewer is testing: deployment literacy.
- Optional follow-up: Which matters more on-device?

42. What part of a CV pipeline can dominate runtime besides the model?
- Ideal answer outline: resize, decode, color conversion, NMS, tracking association, serialization, data movement.
- Interviewer is testing: end-to-end thinking.
- Optional follow-up: How would you profile it?

43. What does quantization buy you and what can it hurt?
- Ideal answer outline: lower memory and faster inference; may hurt small-object sensitivity, calibration, or edge cases.
- Interviewer is testing: compression realism.
- Optional follow-up: PTQ vs QAT?

44. Quantization vs pruning vs distillation?
- Ideal answer outline: quantization changes numeric precision; pruning removes weights/structure; distillation transfers behavior to a smaller model.
- Interviewer is testing: model-compression understanding.
- Optional follow-up: Which would you try first on edge hardware?

45. Why is batching not always helpful?
- Ideal answer outline: helps utilization and throughput but increases waiting time and memory; can violate real-time constraints.
- Interviewer is testing: deployment tradeoff reasoning.
- Optional follow-up: What about dynamic batching?

46. How would you monitor drift in production?
- Ideal answer outline: confidence distributions, embedding shift, slice metrics, sensor health, calibration metrics, manual audit loops.
- Interviewer is testing: lifecycle ownership.
- Optional follow-up: How do you monitor without labels?

47. What changes when deploying to edge devices?
- Ideal answer outline: tighter power, memory, thermal, and runtime budgets; less tolerance for heavy preprocessing and large postprocessing.
- Interviewer is testing: hardware-aware design.
- Optional follow-up: How would you simplify the pipeline?

48. Why can train-time vs inference-time preprocessing mismatch be catastrophic?
- Ideal answer outline: model learns one input distribution but sees another; color/order/resize errors can silently destroy performance.
- Interviewer is testing: practical debugging.
- Optional follow-up: How would you verify parity?

49. How would you handle production failures after a model release?
- Ideal answer outline: reproduce, slice failures, inspect logs/artifacts, separate data shift from software regressions, roll back if needed.
- Interviewer is testing: operational maturity.
- Optional follow-up: What artifacts would you log?

50. How do you decide whether to trade accuracy for latency?
- Ideal answer outline: based on operational cost, failure type, fallback options, and where the system bottleneck actually is.
- Interviewer is testing: product-oriented judgment.
- Optional follow-up: What metric would you optimize for?

## 6.5 Geometry Questions

51. Explain the pinhole camera model.
- Ideal answer outline: 3D point transformed into camera frame, projected by perspective division, then mapped with intrinsics.
- Interviewer is testing: core geometry fluency.
- Optional follow-up: What does focal length in pixels mean?

52. Intrinsics vs extrinsics?
- Ideal answer outline: intrinsics describe internal camera geometry; extrinsics describe pose relative to a frame.
- Interviewer is testing: disciplined terminology.
- Optional follow-up: If I resize the image, what changes?

53. What is backprojection?
- Ideal answer outline: map pixel to a 3D ray, or to a 3D point if depth is available.
- Interviewer is testing: ray geometry.
- Optional follow-up: How do you backproject a depth image?

54. Why use homogeneous coordinates?
- Ideal answer outline: represent perspective mappings and translations linearly up to scale.
- Interviewer is testing: projective geometry understanding.
- Optional follow-up: What does a point at infinity mean?

55. How would you calibrate a camera?
- Ideal answer outline: capture multiple views of a known pattern across pose/scale, solve for intrinsics/distortion, validate reprojection error and coverage.
- Interviewer is testing: practical calibration knowledge.
- Optional follow-up: Why is coverage important?

56. What is radial distortion?
- Ideal answer outline: nonlinear displacement increasing with distance from the principal point, often barrel or pincushion.
- Interviewer is testing: lens-model awareness.
- Optional follow-up: Why does it matter for matching?

57. When is homography valid?
- Ideal answer outline: planar scenes or pure camera rotation between views.
- Interviewer is testing: geometric model selection.
- Optional follow-up: Why does it fail for general 3D scenes?

58. Essential vs fundamental matrix?
- Ideal answer outline: `F` is in pixel coordinates for uncalibrated cameras; `E` is in normalized coordinates for calibrated cameras.
- Interviewer is testing: calibrated vs uncalibrated reasoning.
- Optional follow-up: Why is `E = [t]_x R`?

59. What is an epipolar line?
- Ideal answer outline: the locus in one image where the correspondence of a point in the other image must lie.
- Interviewer is testing: geometry intuition.
- Optional follow-up: What is the epipole?

60. Why normalize points in the 8-point algorithm?
- Ideal answer outline: improves numerical conditioning and stability.
- Interviewer is testing: algorithmic maturity.
- Optional follow-up: What rank constraint do we enforce on `F`?

61. Why does triangulation get worse for distant points?
- Ideal answer outline: disparity becomes tiny, rays intersect at shallow angles, so pixel noise causes large depth uncertainty.
- Interviewer is testing: uncertainty intuition.
- Optional follow-up: Why does baseline matter?

62. Why rectify stereo pairs?
- Ideal answer outline: make epipolar lines horizontal so matching reduces to a 1D search.
- Interviewer is testing: stereo fundamentals.
- Optional follow-up: What if calibration is wrong?

63. What problem does PnP solve?
- Ideal answer outline: recover camera pose from 3D-2D correspondences and known intrinsics.
- Interviewer is testing: pose-estimation clarity.
- Optional follow-up: Why use RANSAC?

64. What is bundle adjustment?
- Ideal answer outline: nonlinear optimization over poses and 3D points to minimize reprojection error.
- Interviewer is testing: SfM/SLAM understanding.
- Optional follow-up: Why is sparse structure important?

65. Visual odometry vs SLAM?
- Ideal answer outline: VO estimates motion sequentially and drifts; SLAM adds mapping and loop closure to maintain consistency.
- Interviewer is testing: robotics/perception differentiation.
- Optional follow-up: What is pose graph optimization?

## 6.6 Deep Learning Questions

66. Why do CNNs work so well on images?
- Ideal answer outline: locality, translation equivariance, parameter sharing, hierarchical features.
- Interviewer is testing: first-principles deep learning understanding.
- Optional follow-up: When might ViTs do better?

67. What do stride, padding, and dilation do?
- Ideal answer outline: stride downsamples, padding controls borders/output size, dilation expands receptive field.
- Interviewer is testing: architectural literacy.
- Optional follow-up: What causes gridding artifacts?

68. Why do residual connections help optimization?
- Ideal answer outline: easier gradient flow, identity path, learning residuals is easier than full mappings in deep nets.
- Interviewer is testing: optimization intuition.
- Optional follow-up: Projection shortcut vs identity shortcut?

69. BatchNorm vs LayerNorm?
- Ideal answer outline: BN uses batch statistics and works well in CNNs with decent batch size; LN is per sample/token and standard in transformers.
- Interviewer is testing: normalization choice.
- Optional follow-up: Why prefer GroupNorm in detection?

70. Why can MixUp or CutMix help?
- Ideal answer outline: regularize decision boundaries, reduce overconfidence, improve robustness.
- Interviewer is testing: augmentation reasoning.
- Optional follow-up: When are these bad for localization tasks?

71. What does label smoothing do?
- Ideal answer outline: softens targets, reduces overconfidence, may improve calibration and generalization.
- Interviewer is testing: loss regularization understanding.
- Optional follow-up: When can it hurt?

72. BCE vs softmax cross-entropy?
- Ideal answer outline: BCE for independent multi-label outputs; softmax CE for mutually exclusive classes.
- Interviewer is testing: output-space correctness.
- Optional follow-up: What if the dataset has partial labels?

73. When would you use Dice or focal loss?
- Ideal answer outline: Dice for overlap and heavy imbalance in segmentation; focal for dense imbalance in detection/classification.
- Interviewer is testing: loss selection maturity.
- Optional follow-up: Can you combine them?

74. Attention vs convolution in vision?
- Ideal answer outline: attention gives dynamic long-range mixing; convolution gives strong locality bias and efficiency.
- Interviewer is testing: modern architecture tradeoffs.
- Optional follow-up: Why are hybrid models common?

75. Why do positional encodings matter in ViTs?
- Ideal answer outline: self-attention alone is permutation-equivariant; positions restore spatial structure.
- Interviewer is testing: transformer fundamentals.
- Optional follow-up: Learned vs sinusoidal vs rotary?

76. What is cross-attention and where is it used in CV?
- Ideal answer outline: query from one source attends to keys/values from another; used in DETR, multimodal fusion, segmentation decoders.
- Interviewer is testing: attention mechanics.
- Optional follow-up: Self-attention vs cross-attention?

77. When would you freeze a backbone vs fine-tune end-to-end?
- Ideal answer outline: freeze with small data or limited compute; fine-tune when domain gap is large and enough data exists.
- Interviewer is testing: transfer learning judgment.
- Optional follow-up: Layer-wise LR decay?

78. How do you handle long-tail class distributions?
- Ideal answer outline: resampling, reweighting, focal loss, class-aware metrics, data collection, threshold tuning.
- Interviewer is testing: practical robustness thinking.
- Optional follow-up: What about calibration?

79. Why are small objects hard for detectors?
- Ideal answer outline: few pixels, low signal, aggressive downsampling, label noise, NMS interactions.
- Interviewer is testing: detection failure intuition.
- Optional follow-up: How can FPN help?

80. A model is not learning. What do you check first?
- Ideal answer outline: can it overfit one batch, are labels correct, is preprocessing correct, is LR reasonable, are gradients nonzero.
- Interviewer is testing: debugging ability.
- Optional follow-up: What if train improves but val stays flat?

## 6.7 Project-Based Questions

81. Tell me about a CV project you are proud of.
- Ideal answer outline: problem, constraints, data, model, metric, bottleneck, improvement, outcome.
- Interviewer is testing: ownership and communication.
- Optional follow-up: What specifically did you own?

82. What was the main bottleneck in your project?
- Ideal answer outline: identify whether it was data, model, optimization, or systems; show evidence.
- Interviewer is testing: diagnosis ability.
- Optional follow-up: How did you prove it?

83. Why did you choose your evaluation metric?
- Ideal answer outline: connect metric to task and operational cost.
- Interviewer is testing: metric alignment.
- Optional follow-up: What metric did stakeholders care about?

84. What data issue hurt your project most?
- Ideal answer outline: label noise, imbalance, domain shift, near-duplicates, sparse annotations, etc.
- Interviewer is testing: data awareness.
- Optional follow-up: How did you fix or quantify it?

85. Why did you choose that model?
- Ideal answer outline: architecture justified by data scale, latency, and error profile, not just benchmark popularity.
- Interviewer is testing: model-task fit.
- Optional follow-up: What obvious alternative did you reject?

86. Describe a failure mode and how you addressed it.
- Ideal answer outline: specific slice, root-cause reasoning, intervention, quantitative effect.
- Interviewer is testing: debugging depth.
- Optional follow-up: Did the fix hurt anything else?

87. What production constraint most influenced your design?
- Ideal answer outline: latency, memory, reliability, sensor setup, or annotation budget tied to design choice.
- Interviewer is testing: engineering realism.
- Optional follow-up: What tradeoff did you accept?

88. What did your ablations show?
- Ideal answer outline: controlled experiments that changed project direction.
- Interviewer is testing: scientific rigor.
- Optional follow-up: Which result surprised you?

89. What would you do next if you had more time?
- Ideal answer outline: prioritized next steps based on the biggest remaining failure modes.
- Interviewer is testing: judgment and roadmap thinking.
- Optional follow-up: Why is that higher priority than a bigger model?

90. What did you personally own end-to-end?
- Ideal answer outline: explicit components you designed, implemented, debugged, or deployed.
- Interviewer is testing: ownership honesty.
- Optional follow-up: What was not your responsibility?

## 6.8 Research-Style Open-Ended Questions

91. When should you prefer explicit geometry over end-to-end learning?
- Ideal answer outline: when structure is known, data is limited, or reliability matters; combine geometry and learning when possible.
- Interviewer is testing: modeling philosophy.
- Optional follow-up: Give a perception example.

92. If you had infinite labeled data, would CNNs still matter?
- Ideal answer outline: data can reduce the need for inductive bias, but compute, efficiency, and deployment still make CNN biases valuable.
- Interviewer is testing: nuanced thinking about scaling.
- Optional follow-up: What role do hybrid models play?

93. How would you evaluate open-world perception?
- Ideal answer outline: closed-set metrics are insufficient; need open-set robustness, calibration, unknown detection, long-tail coverage.
- Interviewer is testing: evaluation sophistication.
- Optional follow-up: What benchmark would you design?

94. SSL vs supervised learning when labels are scarce?
- Ideal answer outline: SSL helps with representation learning, then task-specific fine-tuning uses fewer labels.
- Interviewer is testing: modern pretraining reasoning.
- Optional follow-up: What if unlabeled data comes from a different domain?

95. If annotation is expensive, where would you spend budget?
- Ideal answer outline: targeted relabeling on high-error slices, rare classes, boundary-quality cases, ambiguous labels.
- Interviewer is testing: data-economics judgment.
- Optional follow-up: Would active learning help?

96. How would you build a robust perception stack for robotics?
- Ideal answer outline: calibrated sensors, strong data pipeline, geometry where possible, uncertainty-aware outputs, monitoring, fallback logic.
- Interviewer is testing: systems thinking.
- Optional follow-up: What failures are safety-critical?

97. How would you use synthetic data responsibly?
- Ideal answer outline: target rare or expensive-to-label cases, validate on real data, monitor domain gap, avoid overtrusting simulation.
- Interviewer is testing: realistic data strategy.
- Optional follow-up: Domain randomization vs photorealism?

98. What makes a foundation model useful in perception rather than just impressive?
- Ideal answer outline: transferability, grounding, calibration, efficiency, controllability, and integration with task constraints.
- Interviewer is testing: practical view of foundation models.
- Optional follow-up: What failure modes worry you?

99. How would you move from closed-set detection to open-vocabulary detection safely?
- Ideal answer outline: staged rollout, threshold and calibration analysis, human audit, long-tail evaluation, fallback classes.
- Interviewer is testing: product-minded adaptation of research ideas.
- Optional follow-up: How would you handle prompt sensitivity?

100. Design a perception system for rare but safety-critical events.
- Ideal answer outline: identify rare-event data strategy, uncertainty handling, robust metrics, fallback rules, simulation, active collection loop.
- Interviewer is testing: end-to-end design under ambiguity.
- Optional follow-up: Which metric would you optimize?

---

## 7. Fast Revision Cheat Sheets

## 7.1 Last-Minute 1-Hour Revision Sheet

### 0-10 Minutes: Geometry [HIGH-YIELD]
- Projection equation: `s x = K [R|t] X`
- Intrinsics vs extrinsics.
- Homography vs essential vs fundamental matrix.
- Triangulation vs PnP.
- Stereo depth: `Z = fB/d`.

### 10-20 Minutes: Core Tasks And Metrics [HIGH-YIELD]
- Detection: IoU, AP, mAP, NMS, one-stage vs two-stage.
- Segmentation: semantic vs instance vs panoptic, mIoU, Dice.
- Tracking: MOTA vs IDF1 vs HOTA.
- Depth: Abs Rel, RMSE, `δ < 1.25`.

### 20-30 Minutes: CNN / Transformer Basics
- convolution, stride, padding, dilation, receptive field.
- residual connections, BatchNorm vs LayerNorm.
- attention equation, ViT intuition, DETR set prediction.

### 30-40 Minutes: Data And Training
- class imbalance, noisy labels, leakage, augmentation validity.
- overfit one batch, slice-based error analysis, ablations.

### 40-50 Minutes: Systems And Deployment
- latency vs throughput,
- preprocessing/postprocessing bottlenecks,
- quantization vs pruning vs distillation,
- production drift and monitoring.

### 50-60 Minutes: Project Story
- problem, data, model, metric, failure mode, debugging, what improved, tradeoff, what next.

## 7.2 Last-Minute 30-Minute Revision Sheet
- `[HIGH-YIELD]` Convolution, receptive field, BN/LN, residuals.
- `[HIGH-YIELD]` Detection pipeline: boxes, IoU, NMS, mAP.
- `[HIGH-YIELD]` Segmentation family + mIoU/Dice.
- `[HIGH-YIELD]` Camera model, calibration, homography, `E` vs `F`, stereo depth, PnP.
- `[HIGH-YIELD]` Tracking vs flow.
- `[HIGH-YIELD]` Class imbalance, augmentation, leakage, calibration.
- `[HIGH-YIELD]` Latency vs accuracy, compression, failure analysis.
- `[HIGH-YIELD]` Two project stories with quantified results.

## 7.3 Top Formulas Sheet
- Projection: `s x = K [R|t] X`
- Homography: `x' ~ Hx`
- Epipolar constraint: `x'^T F x = 0`
- Essential matrix: `E = [t]_x R`
- Stereo depth: `Z = fB/d`
- IoU: `|A ∩ B| / |A ∪ B|`
- Precision: `TP / (TP + FP)`
- Recall: `TP / (TP + FN)`
- F1: `2PR / (P + R)`
- Dice: `2|A ∩ B| / (|A| + |B|)`
- Cross-entropy: `-Σ y log p`
- Focal loss: `-(1-p_t)^γ log p_t`
- Attention: `softmax(QK^T / sqrt(d_k)) V`
- Conv output size: `floor((H + 2P - D(K-1) - 1)/S + 1)`

## 7.4 Top Metrics Sheet
- Classification:
  - balanced: accuracy,
  - imbalanced: precision/recall/F1, PR-AUC,
  - confidence-sensitive: ECE, Brier score.
- Detection:
  - localization + ranking: AP/mAP,
  - debugging: per-class AP, AP by object size.
- Segmentation:
  - overlap: mIoU, Dice,
  - weak metric: pixel accuracy under heavy background.
- Tracking:
  - identity: IDF1,
  - combined: HOTA,
  - older summary: MOTA.
- Depth:
  - relative error: Abs Rel,
  - absolute scale: RMSE,
  - tolerance: `δ < 1.25`.

## 7.5 Top Pitfalls Sheet
- `[COMMON TRAP]` Using accuracy for detection or long-tail classification.
- `[COMMON TRAP]` Confusing homography with general two-view geometry.
- `[COMMON TRAP]` Saying `F` and `E` are interchangeable.
- `[COMMON TRAP]` Forgetting data leakage in video or multi-camera splits.
- `[COMMON TRAP]` Recommending bigger models before checking labels.
- `[COMMON TRAP]` Ignoring preprocessing and postprocessing in latency discussion.
- `[COMMON TRAP]` Using augmentations that invalidate boxes, masks, or keypoints.
- `[COMMON TRAP]` Reporting one metric without slice analysis.
- `[COMMON TRAP]` Discussing confidence without calibration.
- `[COMMON TRAP]` Calling optical flow and tracking the same thing.

## 7.6 Things To Say Clearly In Interviews
- `[GOOD INTERVIEW LINE] "I want to choose the output representation first, because that determines the right model family and metric."`
- `[GOOD INTERVIEW LINE] "For imbalanced problems, I would not trust accuracy; I would look at precision-recall behavior and class-wise metrics."`
- `[GOOD INTERVIEW LINE] "In geometry problems, I first clarify which frame each quantity lives in."`
- `[GOOD INTERVIEW LINE] "For CV deployment, I profile the full pipeline, not just the network."`
- `[GOOD INTERVIEW LINE] "If geometry is valid, I prefer to use it as structure rather than asking a network to rediscover it."`
- `[GOOD INTERVIEW LINE] "I separate whether the issue is data quality, optimization, model capacity, or system integration."`

---

## 8. Comparison Tables

### 8.1 SIFT vs ORB

| Method | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| SIFT | scale- and rotation-invariant float descriptor | robust matching, strong invariance | slower, larger descriptor, patent history | stitching, SfM, hard matching | choose when robustness matters more than speed |
| ORB | FAST keypoints + rotated BRIEF binary descriptor | fast, cheap matching with Hamming distance | less robust than SIFT/SURF under severe variation | real-time robotics, embedded matching | choose when runtime is tight |

### 8.2 Classical CV vs Deep Learning CV

| Paradigm | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| Classical CV | handcrafted operators and geometry-driven algorithms | interpretable, data-efficient, geometry-aware | brittle under appearance variation | calibration, registration, controlled pipelines | still essential when structure is known |
| Deep Learning CV | learned features and end-to-end task models | strong performance under high variability | data-hungry, harder to debug, can be opaque | recognition, detection, segmentation at scale | best answers combine learned models with explicit structure |

### 8.3 One-Stage vs Two-Stage Detectors

| Detector Type | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| One-stage | dense prediction directly from feature maps | fast, simple pipeline, real-time friendly | can be weaker on hard localization/small objects | edge or real-time perception | strong speed-accuracy option |
| Two-stage | proposals first, then refine/classify | strong accuracy and proposal refinement | heavier, slower | offline or accuracy-heavy detection | useful when recall and localization quality dominate |

### 8.4 Semantic vs Instance vs Panoptic Segmentation

| Task | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| Semantic | per-pixel class only | simple dense understanding | cannot separate same-class instances | road scene parsing, tissue labeling | use when instance identity is irrelevant |
| Instance | per-instance masks | separates individual objects | harder, more annotation cost | counting, manipulation, retail | needed when object identity matters |
| Panoptic | unify stuff + things | comprehensive scene understanding | more complex evaluation and modeling | autonomy, robotics | best when both scene layout and instances matter |

### 8.5 CNN vs ViT

| Backbone | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| CNN | local weight-shared convolutions | efficient, strong locality bias, data-efficient | less direct global context | many production vision systems | excellent default under moderate data and tight compute |
| ViT | patch tokens with transformer blocks | strong scaling, global context, flexible pretraining | more data/compute hungry, attention cost | large-scale pretraining, modern foundation models | know why it wins and where it is overkill |

### 8.6 Essential vs Fundamental Matrix

| Matrix | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| Fundamental `F` | uncalibrated two-view geometry in pixel coordinates | works without known intrinsics | less directly tied to pose decomposition | uncalibrated matching, stereo geometry | use when cameras are not calibrated |
| Essential `E` | calibrated two-view geometry in normalized coordinates | directly related to `R` and `t` direction | needs intrinsics; scale still ambiguous | calibrated VO, SfM initialization | use when calibration is known |

### 8.7 Homography vs Fundamental Matrix

| Model | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| Homography | planar or pure-rotation image mapping | simple, direct warp between images | invalid for general non-planar scenes | panorama, document rectification | check planarity or pure rotation first |
| Fundamental Matrix | general rigid two-view relation | handles non-planar 3D scenes | does not give direct image warp everywhere | epipolar matching, stereo constraints | use for general two-view geometry |

### 8.8 RANSAC vs Least Squares

| Method | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| Least Squares | minimize residuals over all points | efficient under low-noise inlier data | brittle to outliers | calibration refinement, BA | best after outlier rejection |
| RANSAC | robust model fitting from random minimal samples | handles high outlier rates | stochastic, threshold sensitive | homography, `F/E`, PnP | use when correspondences contain outliers |

### 8.9 Batch Norm vs Layer Norm

| Norm | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| BatchNorm | normalize using batch statistics | very effective in CNNs with decent batch size | unstable with tiny batches, train/infer mismatch issues | image classification backbones | great default for CNNs if batch regime supports it |
| LayerNorm | normalize per sample/token | batch-size independent, standard for transformers | different inductive behavior from BN | ViTs, NLP-style models | best when batch independence matters |

### 8.10 Anchor-Based vs Anchor-Free Detectors

| Type | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| Anchor-based | predict offsets from predefined anchors | intuitive priors, mature ecosystem | anchor tuning complexity, many negatives | RetinaNet, Faster R-CNN variants | useful when priors are well understood |
| Anchor-free | predict centers/keypoints/distances directly | simpler label assignment in many setups | still needs careful design and thresholds | FCOS, CenterNet, modern YOLO variants | often easier to maintain and tune |

### 8.11 Optical Flow vs Tracking

| Task | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| Optical Flow | dense or sparse pixel motion | rich local motion field | no object identity, struggles at occlusion | stabilization, motion estimation | motion is not identity |
| Tracking | identity-preserving object trajectories | object-level temporal consistency | depends heavily on detector and association | surveillance, traffic, sports | tracking needs detection + association |

### 8.12 Stereo vs Monocular Depth

| Method | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| Stereo | depth from disparity across calibrated views | metric geometry, explicit triangulation | needs multiple views and calibration; fails on low texture | robotics, AV, industrial depth | stronger geometry, more hardware/setup cost |
| Monocular | depth from one image using learned priors | cheap hardware, simple capture | scale ambiguity, harder generalization | mobile vision, single-camera setups | useful but less reliable metrically |

### 8.13 Quantization vs Pruning vs Distillation

| Method | Definition | Advantages | Disadvantages | Use Cases | Interview Takeaway |
|---|---|---|---|---|---|
| Quantization | reduce numeric precision | memory/runtime gains, deployment friendly | accuracy drop if poorly calibrated | edge inference | often first deployment lever |
| Pruning | remove weights or structures | can reduce compute and params | hardware gains depend on structure support | model compression | structured pruning is more practical than unstructured |
| Distillation | train smaller student from teacher | often preserves accuracy better | extra training complexity | compact deployment models | strong when you can afford retraining |

---

## 9. Role-Specific Adaptation

### 9.1 General CV Engineer
- Topics that matter more:
  - `[HIGH-YIELD]` CNNs, detection, segmentation, metrics, augmentation, debugging, deployment basics.
- Likely questions:
  - detection tradeoffs, metric choice, overfitting/debugging, model deployment, project bottlenecks.
- Project angles to emphasize:
  - data quality, metric design, clear production impact, speed-accuracy tradeoff.

### 9.2 Autonomous Driving / Perception
- Topics that matter more:
  - `[HIGH-YIELD]` calibration, stereo/depth, tracking, 3D detection, BEV, multi-sensor fusion, robustness, long-tail safety cases.
- Likely questions:
  - camera/LiDAR fusion, BEV motivation, online calibration drift, false positives vs false negatives, weather and night failures.
- Project angles to emphasize:
  - safety-critical error analysis, slice metrics, latency constraints, sensor synchronization, calibration discipline.

### 9.3 Robotics Vision
- Topics that matter more:
  - `[HIGH-YIELD]` geometry, PnP, VO, SLAM basics, hand-eye calibration, depth, tracking, uncertainty.
- Likely questions:
  - frame transforms, mapping vs localization, monocular scale ambiguity, calibration pipelines, failure recovery.
- Project angles to emphasize:
  - coordinate frames, state estimation, real-time constraints, edge deployment, closed-loop system behavior.

### 9.4 Medical Imaging
- Topics that matter more:
  - segmentation, Dice/mIoU, label uncertainty, imbalance, calibration, 3D volumetric models, weak supervision.
- Likely questions:
  - Dice vs CE, annotation ambiguity, class imbalance, sensitivity vs specificity, domain shift across sites.
- Project angles to emphasize:
  - clinical metric alignment, annotation protocol, uncertainty, slice/volume consistency, false negative cost.

### 9.5 Multimodal / VLM Interview
- Topics that matter more:
  - vision-language alignment, open-vocabulary recognition, grounding, calibration, hallucination, retrieval, multimodal fusion.
- Likely questions:
  - CLIP-style learning, prompt sensitivity, zero-shot evaluation, open-vocabulary detection tradeoffs.
- Project angles to emphasize:
  - data curation, alignment objectives, safety/grounding, evaluation beyond benchmark accuracy.

### 9.6 Edge Deployment CV Roles
- Topics that matter more:
  - latency, memory, batching, quantization, pruning, distillation, runtime profiling, preprocessing/postprocessing optimization.
- Likely questions:
  - INT8 vs FP16, bottleneck profiling, deployment toolchains, throughput vs latency, thermal or power constraints.
- Project angles to emphasize:
  - end-to-end profiling, compression strategy, hardware-aware design, fallback logic, robustness under constrained compute.

---

## Closing Priorities
- If time is short, master Sections 1, 2.B, 2.C, 2.D, 2.E, 4, and 5 first.
- If the role is perception/robotics-heavy, over-index on geometry, tracking, depth, BEV, and calibration.
- If the role is applied-science-heavy, over-index on metrics, ablations, error analysis, and model-selection tradeoffs.
- If the role is production-heavy, over-index on data quality, latency, compression, calibration, and drift monitoring.
