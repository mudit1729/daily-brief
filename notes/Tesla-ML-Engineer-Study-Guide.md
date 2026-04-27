# Tesla ML Engineer Interview Guide - Dense Print Version

Use this for printing. It compresses the full guide into interview-ready notes. Keep the full guide for source links, longer explanations, and extra examples.

## 0. What To Optimize For

Primary signal: clean Python + ML fundamentals + production/autonomy tradeoffs + concrete project ownership.

Recruiter note priority: technical background discussion -> ML fundamentals/problem solving -> Python coding.

Do not over-index on exotic papers. Tesla-style interviews tend to reward builders: correct code, clear debugging, edge deployment awareness, data quality thinking, and crisp discussion of failure modes.

High-ROI prep order:

1. Python/NumPy coding: arrays, indexing, binary search, sliding windows, IoU/NMS, trajectory distances, softmax/CE, attention.
2. ML fundamentals: bias/variance, overfitting, metrics, imbalance, calibration, leakage, noisy labels, shift, non-convergence.
3. Deep learning/PyTorch: train/eval, autograd, CE inputs, Conv2D shapes, attention, AMP, checkpointing, debugging.
4. Autonomy reasoning: perception, prediction, edge inference, latency, monitoring, closed-loop evaluation.
5. Project deep dive: numbers, ownership, tradeoffs, failure cases, what you personally did.

Likely loop:

| Round | What they test | Your stance |
|---|---|---|
| Recruiter | Fit, role match, why Tesla, project overview | concise, specific, mission aligned |
| Technical screen | Python, ML fundamentals, project probing | write correct code, explain tradeoffs |
| Domain expert | CV/autonomy/deployment/debugging | reason from data -> model -> eval -> deploy |
| Panel/onsite | coding, ML design, project deep dive, behavioral | concrete examples, no hand-waving |

## 1. Python Coding: Core Recipes

Interview coding standard: clarify input shapes, state assumptions, handle empty/edge cases, write simple correct code, test with 1-2 examples, discuss complexity.

### 1.1 Must-Know Implementations

```python
import math
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy_from_logits(logits, y):
    logits = np.asarray(logits, dtype=np.float64)   # [N, C]
    y = np.asarray(y, dtype=np.int64)               # [N]
    z = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    return -np.mean(log_probs[np.arange(len(y)), y])


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(cm, eps=1e-12):
    tp = np.diag(cm).astype(np.float64)
    precision = tp / (cm.sum(axis=0) + eps)
    recall = tp / (cm.sum(axis=1) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def top_k_hardest(losses, k):
    losses = np.asarray(losses)
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, len(losses))
    idx = np.argpartition(-losses, k - 1)[:k]
    return idx[np.argsort(-losses[idx])]


def first_crossing(xs, threshold):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] >= threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo if lo < len(xs) else -1


def waypoint_suffix_distances(points):
    p = np.asarray(points, dtype=np.float64)        # [N, 2] or [N, D]
    if len(p) == 0:
        return np.array([], dtype=np.float64)
    if len(p) == 1:
        return np.array([0.0])
    seg = np.linalg.norm(p[1:] - p[:-1], axis=1)    # [N-1]
    out = np.zeros(len(p), dtype=np.float64)
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def braking_decision(speed, distance, max_decel, reaction_time=0.0, margin=0.0):
    if max_decel <= 0:
        raise ValueError("max_decel must be positive")
    speed = max(float(speed), 0.0)
    distance = float(distance)
    stopping_distance = speed * reaction_time + speed * speed / (2 * max_decel)
    required_decel = speed * speed / (2 * max(distance, 1e-9))
    should_brake = stopping_distance + margin >= distance
    return should_brake, required_decel


def iou_matrix(boxes1, boxes2):
    b1 = np.asarray(boxes1, dtype=np.float64)       # [N, 4], xyxy
    b2 = np.asarray(boxes2, dtype=np.float64)       # [M, 4]
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.prod(np.clip(b1[:, 2:] - b1[:, :2], 0, None), axis=1)
    area2 = np.prod(np.clip(b2[:, 2:] - b2[:, :2], 0, None), axis=1)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-12)


def nms(boxes, scores, iou_threshold=0.5):
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[[i]], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep


def conv2d_output_shape(h, w, kh, kw, stride=1, padding=0, dilation=1):
    def pair(v):
        return v if isinstance(v, tuple) else (v, v)
    sh, sw = pair(stride)
    ph, pw = pair(padding)
    dh, dw = pair(dilation)
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return oh, ow


def conv2d_forward_nchw(x, weight, bias=None, stride=1, padding=0):
    def pair(v):
        return v if isinstance(v, tuple) else (v, v)
    x = np.asarray(x, dtype=np.float64)             # [N, Cin, H, W]
    w = np.asarray(weight, dtype=np.float64)         # [Cout, Cin, Kh, Kw]
    sh, sw = pair(stride)
    ph, pw = pair(padding)
    n, cin, h, ww = x.shape
    cout, cin2, kh, kw = w.shape
    assert cin == cin2
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    oh, ow = conv2d_output_shape(h, ww, kh, kw, stride, padding)
    out = np.zeros((n, cout, oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    if bias is not None:
        out += np.asarray(bias, dtype=np.float64)[None, :, None, None]
    return out


def scaled_dot_product_attention(q, k, v, mask=None):
    # q [B,H,Lq,D], k [B,H,Lk,D], v [B,H,Lk,Dv], mask broadcastable to [B,H,Lq,Lk]
    scores = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e30)      # True means keep
    weights = softmax(scores, axis=-1)
    return weights @ v, weights
```

### 1.2 Coding Pitfalls

| Problem | Key idea | Pitfalls |
|---|---|---|
| Softmax | subtract row max | overflow, wrong axis |
| Cross entropy | log-softmax then gather true class | passing probabilities into CE that expects logits |
| Confusion matrix | rows true, cols predicted | mixing row/column convention |
| Top-k | `argpartition` then sort selected | `argpartition` is not sorted |
| Binary search | maintain first valid index | off-by-one, no crossing |
| Waypoint suffix | segment lengths, reverse cumulative sum | returning prefix distance instead |
| Braking | stopping distance `v^2/(2a)` plus reaction distance | sign of decel, zero distance |
| IoU | intersection / union | coordinate convention, zero-area boxes |
| NMS | sort by score, suppress high IoU | class-agnostic vs per-class NMS |
| Conv2D | DL libraries use cross-correlation | output shape, padding, stride |
| Attention | `softmax(QK^T/sqrt(d))V` | mask convention, shape broadcast |

### 1.3 Tesla-Flavored Coding Questions

1. Given waypoints `[N,2]`, return remaining path distance from each waypoint. Answer: compute segment norms, reverse cumsum, last is zero.
2. Given speed, obstacle distance, max decel, decide brake/no brake. Answer: brake if `v^2/(2a) + v*t_reaction + margin >= distance`.
3. Given losses, return top-k hardest examples. Answer: `argpartition(-losses, k-1)` then sort selected.
4. Given `y_true/y_pred`, compute confusion matrix and per-class precision/recall/F1.
5. Implement vectorized IoU for two sets of boxes.
6. Implement NMS. Discuss per-class NMS and threshold tradeoff.
7. Implement stable softmax and CE from logits.
8. Implement scaled dot-product attention with a mask.
9. Implement Conv2D output shape; optionally forward pass.
10. Find first timestamp where signal crosses threshold. Use binary search only if signal is monotonic.

## 2. ML Fundamentals: Fast Answer Bank

### Bias and Variance

High bias: train and validation both poor. Causes: weak model, insufficient features, excessive regularization, undertraining. Fix: stronger model, better features, train longer, reduce regularization.

High variance: train good, validation poor. Causes: overfit, leakage in train process, insufficient data, model too large. Fix: more data, augmentation, regularization, early stopping, smaller model, better split.

Tesla-style framing: diagnose by slice, not just aggregate. A detector can look fine overall and fail on night pedestrians or construction zones.

### Overfitting

Symptoms: train loss decreases, validation loss increases; train metric much better than val/test. Fixes: data augmentation, dropout, weight decay, early stopping, smaller model, more diverse data, label cleanup, hard negative mining, cross-validation/group split.

### Metrics

| Metric | Use | Failure mode |
|---|---|---|
| Accuracy | balanced classes, equal costs | dangerous under imbalance |
| Precision | false positives costly | can miss too many positives |
| Recall | false negatives costly | can create many false alarms |
| F1 | single threshold, balance P/R | hides threshold curve |
| ROC-AUC | ranking with balanced-ish data | can look good under heavy imbalance |
| PR-AUC | rare positives | more informative for detection/rare events |
| Calibration | probability quality | high confidence wrong predictions |

Pedestrian detection: prefer recall-focused metric under safety constraints, but track precision/false positives because unnecessary braking or alerts matter. Use slice metrics for night/rain/occlusion/small objects.

### Imbalanced Data

Tools: class weights, focal loss, oversampling, undersampling, hard negative mining, threshold tuning, better sampling by scenario, PR-AUC, per-slice recall. Accuracy is dangerous because majority class can dominate.

Focal loss idea: down-weight easy examples, focus on hard/misclassified examples. Useful for dense detection with many easy background negatives.

### Calibration

Calibrated model: among predictions with confidence 0.8, about 80% are correct. Diagnose with reliability diagram/ECE. Fix with temperature scaling, Platt scaling, isotonic regression, better validation set, avoid overconfident training. Calibration matters for planning and risk-aware decisions.

### Data Leakage

Common autonomy leakage: split by frame instead of clip/trip/location; future frames in features; duplicate scenes across splits; labels derived from future information; train/val overlap through near-identical fleet clips. Fix: split by clip/session/geography/time/hardware, deduplicate, audit features.

### Noisy Labels

Approaches: label audits, robust losses, label smoothing, bootstrapping/teacher models, confident learning, downweight suspicious examples, train on clean seed then expand. Do not blindly trust aggregate metrics when label quality differs by slice.

### Distribution Shift

Examples: day/night, rain/snow, construction, new camera, new geography, rare object types. Detect with slice metrics, embedding drift, failure clustering, uncertainty/confidence drift, data quality monitors. Fix via targeted data collection, active learning, augmentation, domain adaptation, model/hardware recalibration.

### Non-Convergence

Debug order: overfit one tiny batch -> check labels/dtypes/shapes -> inspect loss inputs -> lower LR -> check data normalization -> gradient norms -> NaNs/infs -> train/eval mode -> frozen params -> optimizer receives params -> compare baseline.

## 3. Deep Learning and PyTorch

### 3.1 Core Concepts

Logistic regression gradient: for sigmoid + binary CE, `dL/dz = p - y`; `dw = X.T@(p-y)/N`, `db = mean(p-y)`.

Cross entropy: combines log-softmax and negative log-likelihood. In PyTorch `CrossEntropyLoss` expects raw logits `[N,C]` and class indices `[N]`, not softmax probabilities.

Numerical stability: subtract max before softmax; use `BCEWithLogitsLoss` instead of sigmoid + BCE; use log-sum-exp.

Learning rate too high: divergence, oscillation, NaNs. Too low: slow progress, apparent plateau. Use LR finder, scheduler, gradient norm checks.

Vanishing/exploding gradients: deep nets/RNNs/poor initialization. Fix with residuals, normalization, better init, gradient clipping, gated architectures, shorter sequences.

BatchNorm vs LayerNorm:

| Norm | Normalizes over | Common use | Gotchas |
|---|---|---|---|
| BatchNorm | batch/spatial stats per channel | CNNs | train/eval mode, small batch instability |
| LayerNorm | features within sample | Transformers/RNNs | independent of batch size |

Dropout: active in train, disabled in eval. Forgetting `model.eval()` breaks validation/inference; forgetting `model.train()` after eval disables dropout during training.

Adam vs SGD vs AdamW: Adam adapts per-parameter steps and is easier to tune; SGD+momentum can generalize well in vision; AdamW decouples weight decay and is default for Transformers.

Weight decay vs L2: equivalent for plain SGD in many setups; AdamW decouples decay from adaptive gradient update, usually preferred.

Init: Xavier for tanh/sigmoid-ish activations; He/Kaiming for ReLU-family.

### 3.2 PyTorch Skeletons

```python
# Training loop
model.train()
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional
    optimizer.step()

# Evaluation loop
model.eval()
total_loss, correct, n = 0.0, 0, 0
with torch.inference_mode():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)

# Attention in PyTorch
def attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return w @ v, w
```

### 3.3 CNN and Detection

Conv2D output:

`H_out = floor((H + 2P - D*(K-1) - 1)/S + 1)`

Conv2D parameters:

`params = C_out * (C_in/groups) * K_h * K_w + C_out if bias`

Receptive field: input region that affects an output. Larger via depth, kernel size, stride, dilation. Too much stride hurts small objects.

NMS: keep high-score box, suppress lower-score boxes with IoU above threshold. Soft-NMS decays scores instead of hard removing.

Detection metrics: AP is area under precision-recall curve for a class at an IoU threshold. mAP averages AP over classes and often IoU thresholds. AP50 is easier than AP75/COCO mAP. mAP can hide safety failures if rare slices are underrepresented.

Small-object detection: higher input resolution, FPN/multi-scale features, anchors/priors tuned to small boxes, less aggressive stride, better labels, oversample rare/small examples, evaluate by size bucket.

### 3.4 Deployment and Speed

If model is accurate but 3x too slow:

1. Profile first: preprocessing, CPU/GPU transfer, kernels, memory bandwidth, batch size.
2. Use smaller architecture, lower resolution, early exits, ROI processing.
3. Quantize FP32 -> FP16/INT8; calibrate INT8 carefully.
4. Prune or distill.
5. Fuse ops, export to ONNX/TensorRT-style runtime, use static shapes where possible.
6. Validate accuracy by slice after optimization.

FP32 vs FP16 vs INT8:

| Type | Benefit | Risk |
|---|---|---|
| FP32 | most stable | slower, memory-heavy |
| FP16/BF16 | faster on accelerators, less memory | overflow/underflow, numerics |
| INT8 | high throughput | calibration error, accuracy loss |

## 4. Transformers and Attention

Q/K/V: queries ask what each token wants; keys describe what each token offers; values are content mixed by attention weights.

Scaled dot-product attention:

`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + mask) V`

Why divide by `sqrt(d_k)`: dot-product variance grows with dimension; scaling prevents saturated softmax and tiny gradients.

Masks: padding mask hides padded tokens; causal mask hides future tokens; cross-attention mask can hide invalid source positions. Always state mask convention.

Self-attention vs cross-attention: self uses same sequence for Q/K/V; cross uses Q from target and K/V from source.

Position matters: vanilla attention is permutation-invariant without positional information. Options: sinusoidal, learned, relative, RoPE.

Long sequence cost: attention is O(L^2) memory/compute. Reduce via windowed attention, sparse attention, linear attention, downsampling, recurrence/memory, chunking, FlashAttention-style kernels.

Attention collapse: attention becomes uniform or too peaky. Debug temperature/scale, mask bugs, initialization, LR, normalization, entropy, gradient flow, data leakage.

Tesla relevance: temporal camera features, route/map context, trajectory prediction, occupancy/perception fusion can all involve sequence/spatial attention.

## 5. Autonomy and Tesla-Domain Reasoning

### 5.1 Perception

Lane detection answer template:

1. Define output: segmentation, polylines, lane graph, or occupancy representation.
2. Data: diverse fleet clips, camera calibration, weather, lane types, construction, night, glare.
3. Labels: human labels + auto-labels + QA; split by clip/location/time, not frame.
4. Model: CNN/Transformer backbone, multi-scale features, temporal context if needed.
5. Loss: segmentation CE/Dice/Focal, polyline regression, topology losses.
6. Metrics: IoU/F1, lane boundary error, topology accuracy, temporal stability, slice metrics.
7. Deployment: latency, memory, quantization, camera calibration drift.
8. Monitoring: false positives/negatives by scenario, drift, intervention/failure mining.

Pedestrian at night:

- Data: night/rain/glare/occlusion/small pedestrian slices.
- Model: better low-light augmentation, temporal context, multi-scale features.
- Loss/sampling: focal loss, oversample rare night positives, hard negative mining.
- Eval: recall at fixed FP rate, distance buckets, occlusion buckets, small-object buckets.
- Deployment: track false braking risk and calibration, not just recall.

False positives in construction zones:

- Cluster failures by scene/object/context.
- Check label ambiguity and class taxonomy.
- Mine hard negatives: cones, signs, workers, barriers, unusual lane paint.
- Add context/temporal features if single-frame confusion dominates.
- Evaluate before/after on both construction and normal scenes to avoid regression.

Camera calibration drift:

- Symptoms: systematic localization error, lane/box misalignment, worse at image edges.
- Detect with calibration monitors, reprojection residuals, temporal consistency.
- Fix with calibration pipeline, robust augmentation, camera-specific validation, hardware split.

### 5.2 Prediction and Planning

Trajectory prediction output: multi-modal future trajectories with probabilities. Multi-modality matters because agents can turn, stop, yield, merge, or continue.

Metrics:

| Metric | Meaning |
|---|---|
| ADE | average displacement error |
| FDE | final displacement error |
| minADE/minFDE | best of K predicted modes |
| Miss rate | no predicted trajectory close enough |
| NLL | probability quality |
| Collision/rule metrics | closed-loop safety relevance |

Open-loop metrics can lie: a trajectory close to logged human behavior may not be best under a changed ego action. Closed-loop eval tests interaction, compounding errors, comfort, safety, and rule compliance.

Imitation learning vs RL vs MPC:

| Method | Strength | Weakness |
|---|---|---|
| Imitation learning | scalable from logs, stable | covariate shift, copies biases |
| RL | optimizes long-horizon objective | reward hacking, sample inefficiency, safety |
| MPC | interpretable constraints/control | depends on model/cost, can be brittle |

Reward for speed-limit following: reward progress and legal speed, penalize speeding, harsh braking, jerk, collisions, lane violations. Avoid rewarding only speed; it can create unsafe behavior.

### 5.3 Edge Inference and Monitoring

Real-time constraints: latency budget, memory footprint, power/thermal limits, deterministic behavior, fallback behavior, sensor synchronization.

Monitor after deployment: slice metrics, confidence calibration, rare failure clusters, latency, memory, hardware-specific behavior, weather/geography drift, false positive/negative examples, intervention-triggered clips.

High confidence but wrong: calibration failure or distribution shift. Investigate slices, labels, uncertainty, embedding neighbors, hard negatives, and threshold policy. Never trust confidence alone for safety-critical decisions.

## 6. ML System Design Templates

### 6.1 Lane Detection Training Pipeline

Goal: detect lane boundaries/topology robustly across weather, lighting, geography, and construction.

Pipeline:

1. Data collection: fleet clips across scenarios; prioritize failures and rare slices.
2. Splitting: by clip/trip/location/time/hardware, not frame.
3. Labeling: human labels, auto-labels, QA, ontology versioning.
4. Training: multi-scale model, augmentations, temporal context, balanced sampling.
5. Evaluation: aggregate + slice metrics, temporal stability, topology, regression tests.
6. Deployment: export, quantize, profile, latency/memory gates.
7. Monitoring: collect failures, cluster, label, retrain, compare old/new.

Tradeoffs to state: segmentation vs polyline vs lane graph; latency vs resolution; recall vs false positives; auto-label scale vs label noise.

### 6.2 Fleet-Scale Failure Mining System

Goal: find rare useful clips from massive fleet data.

Architecture:

1. Triggers: model uncertainty, disagreement, planner intervention, hard brake, human takeover, near-miss heuristics, novel embeddings.
2. On-vehicle filtering: lightweight rules/embeddings; upload only useful clips.
3. Dedup: perceptual hashes, embedding clustering, GPS/time grouping.
4. Prioritize: rarity, severity, model uncertainty, coverage gaps, business/safety impact.
5. Label/auto-label: route to human or auto-label pipeline; QA hard slices.
6. Train/eval: add to training set; hold out a slice-specific regression set.
7. Monitor: measure failure recurrence and avoid overfitting to mined clips.

Risks: biased sampling toward current model blind spots, duplicate data, privacy/storage limits, label bottleneck, confirmation bias.

### 6.3 Edge Pedestrian Detector

Goal: high recall under latency/memory constraints.

Design:

1. Dataset: day/night/rain/snow, occlusion, distance buckets, small pedestrians, unusual poses.
2. Model: efficient backbone + FPN + detection head; maybe temporal smoothing.
3. Loss: focal/class-balanced loss; box regression loss; hard negative mining.
4. Metrics: recall at fixed FP rate, AP by size/distance/lighting, calibration, latency.
5. Optimization: FP16/INT8, TensorRT-style export, op fusion, smaller input if needed.
6. Safety validation: scenario regression set, false braking analysis, closed-loop replay if available.

### 6.4 Active Learning System

Selection:

- Uncertainty sampling: high entropy/low margin.
- Disagreement sampling: ensemble or teacher/student disagreement.
- Diversity sampling: avoid labeling near-duplicates.
- Failure sampling: interventions, planner conflicts, high loss.

Guardrails: maintain random baseline sample, deduplicate, stratify by slice, track labeling budget, validate on untouched holdout.

### 6.5 Model Debugging Pipeline

1. Define failure exactly: false positive/false negative/class confusion/localization error.
2. Slice: weather, time, geography, camera, distance, occlusion, object size.
3. Inspect labels and inputs.
4. Cluster embeddings for common patterns.
5. Compare model versions.
6. Add targeted data/labels/augmentations.
7. Retrain with regression tests.
8. Deploy only if aggregate and critical slices do not regress.

## 7. Project Deep Dive Script

Prepare one flagship project in this exact structure:

1. Problem: what was broken or needed.
2. Input/output: shapes, data source, label type.
3. Constraints: latency, memory, data size, reliability, deployment target.
4. Baseline: simplest working approach and metric.
5. Model: architecture and why.
6. Alternatives rejected: why not simpler/larger/different.
7. Loss and metrics: what optimized vs what monitored.
8. Data: collection, cleaning, splitting, leakage prevention.
9. Failure cases: 3 concrete examples.
10. Debugging: tools, ablations, slice analysis.
11. Improvements: numbers before/after.
12. Deployment: how shipped, monitored, rolled back or validated.
13. Ownership: what you personally built.
14. Lessons: what you would do differently.

Good phrasing:

"I owned the data pipeline and training loop, including split design, model ablations, and failure analysis. The biggest issue was not the architecture; it was that validation was contaminated by near-duplicate clips. After splitting by session and adding a night/rain slice, aggregate AP dropped but became honest. We then mined hard negatives and recovered X points without regressing latency."

Bad phrasing:

"We used a Transformer because it is state of the art." Say why it matched the data, constraints, and failure mode.

## 8. Behavioral Answers

### Why Tesla

"I am interested in ML systems where prediction quality, latency, and deployment constraints all matter at once. Tesla's work sits in the physical world, so the model is not just optimizing a dashboard metric. It has to survive edge deployment, distribution shift, and real-time decisions. That combination is the kind of ML engineering I want to work on."

### Why This Role

Tie your background to the role:

- I like building production ML, not only training notebooks.
- I am comfortable debugging data, models, metrics, and deployment constraints.
- My strongest work is where system constraints force practical tradeoffs.

### STAR Story Bank

Prepare 5 stories:

| Story | Must include |
|---|---|
| Hard technical problem | concrete bug/failure, your diagnosis, result |
| Ambiguous requirements | how you scoped, aligned, shipped |
| Model failed | detection, root cause, fix, monitoring |
| Disagreement | technical evidence, tradeoff, outcome |
| Time pressure | what you cut, what you protected, result |

Behavioral rule: one sentence situation, one sentence task, most time on actions/results. Use numbers.

## 9. Mock Questions With Dense Answers

### Coding

1. Waypoint suffix distances: segment distances then reverse cumulative sum. Last point has 0 remaining distance.
2. Automatic braking: compute stopping distance `v^2/(2a)` plus reaction distance and margin; brake if exceeds obstacle distance.
3. NMS: sort by score, keep best, remove boxes with IoU above threshold, repeat. Mention per-class NMS.
4. Stable softmax: subtract max along axis, exponentiate, normalize.
5. Attention: scores `QK^T/sqrt(d)`, apply mask, softmax, multiply by V.

### ML Fundamentals

6. Train loss down, val loss up: overfitting or leakage. Add regularization/augmentation, early stop, inspect split, collect data, slice metrics.
7. High precision low recall pedestrian detector: lower threshold, adjust loss/class weights/focal, mine false negatives, add relevant data, evaluate false positive cost.
8. Model works in California but fails in snow: distribution shift. Build snow slice, inspect labels/sensors, mine data, augment, retrain, monitor snow performance.
9. PR-AUC vs ROC-AUC: PR-AUC better for rare positives because it focuses on precision/recall for positive class.
10. Calibration: predicted probabilities match empirical correctness; fix with temperature scaling or recalibration set.

### CV and DL

11. mAP: mean AP over classes/IoU thresholds; can hide rare safety failures and poor calibration.
12. Small object detection: improve resolution/FPN/anchors/sampling/labels; track AP by size and distance.
13. BatchNorm train/eval: train uses batch stats and updates running stats; eval uses running stats.
14. Dropout train/eval: train randomly drops activations; eval uses full network.
15. AdamW: decoupled weight decay, common for Transformers, easier to tune than SGD.

### Transformers

16. Why `sqrt(d_k)`: prevents dot products from growing too large and saturating softmax.
17. Causal mask: blocks future tokens.
18. Cross-attention: Q from target, K/V from source.
19. Long sequences: O(L^2); use sparse/windowed/downsampled/efficient attention.
20. Attention collapse: check mask, scale, LR, init, normalization, entropy, data bugs.

### Autonomy/System Design

21. Fleet data engine: triggers -> upload -> dedup -> prioritize -> label -> train -> eval -> deploy -> monitor.
22. Open-loop metrics can lie: logged future changes under new ego behavior; closed-loop tests interactions and compounding errors.
23. Accurate but too slow: profile, optimize architecture/resolution, quantize, fuse/export, validate slices.
24. Auto-labeling safely: use confidence/consistency checks, human QA, holdout labels, versioning, slice validation.
25. Closed-loop eval metrics: collision, intervention, rule violation, comfort, progress, near misses, route completion.

### Behavioral

26. Weakness answer: choose non-core weakness, show active improvement and evidence.
27. Disagreement: explain technical evidence and tradeoff, not personality.
28. Ambiguous task: define success metric, constraints, owners, and first shippable baseline.
29. Failed model: say how you detected it, root cause, fix, prevention.
30. Ownership: be explicit about what you personally built versus what the team did.

## 10. Formula Sheet

Softmax:

`softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`

Cross entropy from logits:

`CE = -log(exp(z_y) / sum_j exp(z_j)) = -z_y + logsumexp(z)`

Binary CE:

`BCE = -[y log(p) + (1-y) log(1-p)]`

Sigmoid gradient with BCE:

`dL/dz = sigmoid(z) - y`

Precision/recall/F1:

`precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2PR/(P+R)`

IoU:

`IoU = area(intersection) / area(union)`

Conv2D:

`out = floor((in + 2p - d(k-1) - 1)/s + 1)`

Conv params:

`C_out * (C_in/groups) * K_h * K_w + bias`

Attention:

`softmax(QK^T / sqrt(d_k) + mask)V`

Stopping distance:

`d = v*t_reaction + v^2/(2a)`

Trajectory errors:

`ADE = mean_t ||pred_t - gt_t||`, `FDE = ||pred_T - gt_T||`

Calibration:

`confidence p should be correct about p fraction of the time`

## 11. Red Flags To Avoid

- Saying "accuracy" for imbalanced detection without precision/recall/PR-AUC.
- Splitting video/time-series data by frame.
- Optimizing aggregate mAP without safety-critical slice metrics.
- Passing softmax probabilities into PyTorch `CrossEntropyLoss`.
- Forgetting `model.eval()` for validation/inference.
- Saying confidence equals correctness.
- Ignoring latency/memory when discussing vehicle deployment.
- Discussing a project without numbers.
- Saying "we" for everything and never clarifying your ownership.
- Claiming auto-labels are safe without QA and holdout validation.

## 12. Last 48-Hour Plan

Day -2:

1. Implement: softmax, CE, confusion matrix, IoU, NMS, waypoint suffix, braking, attention.
2. Review: bias/variance, metrics, imbalance, calibration, leakage, distribution shift.
3. Prepare one system design: fleet failure mining or edge pedestrian detector.
4. Write project deep dive in 10 bullets with numbers.

Day -1:

1. Timed coding: 2 Python, 1 NumPy, 1 PyTorch-style loop.
2. Mock explain: mAP, PR-AUC, closed-loop vs open-loop, quantization.
3. Practice behavioral answers out loud.
4. Create 3 failure cases from your main project and how you debugged each.

Interview day:

1. Ask clarifying questions before coding.
2. State shapes and edge cases.
3. Write simple correct code first.
4. Test manually.
5. Explain complexity and tradeoffs.
6. For design questions, always cover data -> labels -> model -> eval -> deploy -> monitor.

## 13. One-Page Mental Model

For every ML question, answer in this order:

1. Data: what are inputs, labels, splits, quality risks?
2. Model: what architecture and why?
3. Loss: what objective maps to the task?
4. Metrics: what aggregate and slice metrics matter?
5. Debugging: what failures and ablations would you inspect?
6. Deployment: what latency/memory/hardware constraints apply?
7. Monitoring: how do you catch drift and regressions?

For every autonomy question, add:

1. Physical constraints.
2. Rare safety-critical slices.
3. Temporal behavior.
4. Closed-loop effects.
5. Confidence/calibration.
6. Edge inference limits.

Final signal to project: "I can build ML systems that work outside a notebook."
