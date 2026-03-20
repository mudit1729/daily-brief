# ML & CV Implementation Questions — PyTorch Solutions

Interview-ready implementations for 85 coding questions spanning computer vision and ML/PyTorch fundamentals.

---

## Part 1: CV Implementation Questions (35 questions)

### 1. IoU Between Two Boxes

Given two bounding boxes in `[x1, y1, x2, y2]` format, compute their Intersection over Union.

```python
import torch

def iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return (inter_area / union_area).item() if union_area > 0 else 0.0

# --- test ---
a = torch.tensor([0.0, 0.0, 10.0, 10.0])
b = torch.tensor([5.0, 5.0, 15.0, 15.0])
print(f"IoU = {iou(a, b):.4f}")  # 25/175 ≈ 0.1429
```

### 2. Vectorized IoU (N x M)

Compute the IoU matrix between N predicted boxes and M ground-truth boxes without Python loops.

```python
import torch

def pairwise_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Vectorized IoU between pred (N, 4) and gt (M, 4).
    Returns (N, M) IoU matrix.  Boxes in [x1, y1, x2, y2].
    """
    # Broadcast: (N,1,2) vs (1,M,2)
    inter_tl = torch.max(pred[:, None, :2], gt[None, :, :2])  # (N, M, 2)
    inter_br = torch.min(pred[:, None, 2:], gt[None, :, 2:])  # (N, M, 2)

    inter_wh = (inter_br - inter_tl).clamp(min=0)             # (N, M, 2)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]        # (N, M)

    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])  # (N,)
    area_gt   = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])          # (M,)

    union = area_pred[:, None] + area_gt[None, :] - inter_area
    return inter_area / union.clamp(min=1e-6)

# --- test ---
pred = torch.tensor([[0,0,10,10],[5,5,15,15]], dtype=torch.float)
gt   = torch.tensor([[0,0,10,10],[10,10,20,20]], dtype=torch.float)
print(pairwise_iou(pred, gt))
# row 0: IoU with itself = 1.0, IoU with non-overlapping ≈ 0.0
```

### 3. Non-Max Suppression (NMS)

Given boxes and their confidence scores, greedily keep the highest-scoring box and remove all boxes that overlap it above an IoU threshold.

```python
import torch

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    """
    boxes:  (N, 4) in [x1, y1, x2, y2]
    scores: (N,)
    Returns: indices of kept boxes.
    """
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        # IoU of top box with the rest
        rest = order[1:]
        tl = torch.max(boxes[i, :2], boxes[rest, :2])
        br = torch.min(boxes[i, 2:], boxes[rest, 2:])
        inter = (br - tl).clamp(min=0).prod(dim=1)

        area_i    = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter).clamp(min=1e-6)

        order = rest[iou <= iou_thresh]

    return torch.tensor(keep, dtype=torch.long)

# --- test ---
boxes  = torch.tensor([[0,0,10,10],[1,1,11,11],[20,20,30,30]], dtype=torch.float)p
scores = torch.tensor([0.9, 0.85, 0.7])
print(nms(boxes, scores, 0.5))  # tensor([0, 2]) — middle box suppressed
```

### 4. Batched NMS (Per-Class)

Run NMS independently per class by offsetting box coordinates so boxes from different classes never overlap.

```python
import torch

def batched_nms(
    boxes: torch.Tensor,   # (N, 4)
    scores: torch.Tensor,  # (N,)
    classes: torch.Tensor, # (N,) integer class ids
    iou_thresh: float = 0.5,
) -> torch.Tensor:
    """Per-class NMS using the coordinate-offset trick."""
    # Offset each class's boxes so they can't intersect across classes
    max_coord = boxes.max()
    offsets = classes.float() * (max_coord + 1.0)
    offset_boxes = boxes + offsets[:, None]

    return nms(offset_boxes, scores, iou_thresh)

# reuse nms() from Q3

# --- test ---
boxes   = torch.tensor([[0,0,10,10],[1,1,11,11],[0,0,10,10]], dtype=torch.float)
scores  = torch.tensor([0.9, 0.85, 0.8])
classes = torch.tensor([0, 0, 1])
print(batched_nms(boxes, scores, classes, 0.5))  # tensor([0, 2]) — keeps one per class
```

### 5. Box Format Conversion: xyxy <-> cxcywh

Convert between corner format `[x1, y1, x2, y2]` and center format `[cx, cy, w, h]`.

```python
import torch

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """(N,4) [x1,y1,x2,y2] -> [cx,cy,w,h]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(N,4) [cx,cy,w,h] -> [x1,y1,x2,y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    half_w, half_h = w / 2, h / 2
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)

# --- test (round-trip) ---
orig = torch.tensor([[10, 20, 50, 80], [0, 0, 100, 100]], dtype=torch.float)
converted = xyxy_to_cxcywh(orig)
restored  = cxcywh_to_xyxy(converted)
print(converted)  # [30,50,40,60], [50,50,100,100]
assert torch.allclose(orig, restored)
```

### 6. Box Clipping to Image Boundaries

Clamp box coordinates so they lie within `[0, 0, W, H]`.

```python
import torch

def clip_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Clip (N,4) boxes [x1,y1,x2,y2] to image boundaries."""
    clipped = boxes.clone()
    clipped[:, 0::2] = clipped[:, 0::2].clamp(0, img_w)   # x coords
    clipped[:, 1::2] = clipped[:, 1::2].clamp(0, img_h)   # y coords
    return clipped

# --- test ---
boxes = torch.tensor([[-5, -10, 50, 80], [90, 90, 120, 130]], dtype=torch.float)
print(clip_boxes(boxes, img_w=100, img_h=100))
# tensor([[ 0,  0, 50, 80],
#         [90, 90,100,100]])
```

### 7. Area Computation for Bounding Boxes

Compute area for a batch of boxes, returning zero for degenerate boxes.

```python
import torch

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Area of (N,4) boxes [x1,y1,x2,y2]. Degenerate boxes get area 0."""
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return w * h

# --- test ---
boxes = torch.tensor([
    [0, 0, 10, 10],   # area 100
    [5, 5, 5, 5],     # degenerate, area 0
    [0, 0, 3, 7],     # area 21
], dtype=torch.float)
print(box_area(boxes))  # tensor([100.,  0., 21.])
```

### 8. Pairwise Distance / Similarity for Image Embeddings

Compute cosine similarity and Euclidean distance matrices between two sets of embeddings.

```python
import torch
import torch.nn.functional as F

def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between (N, D) and (M, D) -> (N, M)."""
    a_norm = F.normalize(a, dim=1)  # (N, D)
    b_norm = F.normalize(b, dim=1)  # (M, D)
    return a_norm @ b_norm.T

def euclidean_distance_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between (N, D) and (M, D) -> (N, M)."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    aa = (a * a).sum(dim=1, keepdim=True)       # (N, 1)
    bb = (b * b).sum(dim=1, keepdim=True).T      # (1, M)
    dist_sq = aa + bb - 2.0 * (a @ b.T)
    return dist_sq.clamp(min=0).sqrt()

# --- test ---
a = torch.randn(3, 128)
b = torch.randn(5, 128)
cos_sim = cosine_similarity_matrix(a, b)   # (3, 5) in [-1, 1]
euc_dst = euclidean_distance_matrix(a, b)  # (3, 5) >= 0
print(f"Cosine sim shape: {cos_sim.shape}, Euclidean dist shape: {euc_dst.shape}")
# Self-similarity diagonal should be 1.0
self_sim = cosine_similarity_matrix(a, a)
print(f"Self-sim diagonal: {self_sim.diag()}")  # ≈ [1, 1, 1]
```

### 9. Precision, Recall, F1 for Binary Classification

Compute metrics from raw predictions and labels using only PyTorch.

```python
import torch

def binary_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    preds:   (N,) float probabilities or logits
    targets: (N,) binary {0, 1}
    Returns: precision, recall, f1
    """
    pred_labels = (preds >= threshold).long()

    tp = ((pred_labels == 1) & (targets == 1)).sum().float()
    fp = ((pred_labels == 1) & (targets == 0)).sum().float()
    fn = ((pred_labels == 0) & (targets == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1.item()

# --- test ---
preds   = torch.tensor([0.9, 0.8, 0.4, 0.3, 0.7, 0.1])
targets = torch.tensor([1,   1,   0,   0,   0,   1  ])
p, r, f = binary_metrics(preds, targets)
# TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3
print(f"Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
```

### 10. mAP@IoU for a Tiny Object-Detection Dataset

Compute mean Average Precision at a given IoU threshold across classes.

```python
import torch

def compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    """Compute AP using the 11-point interpolation (PASCAL VOC style)."""
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        p = precisions[mask].max().item() if mask.any() else 0.0
        ap += p
    return ap / 11.0

def mean_ap(predictions: list, ground_truths: list, iou_thresh: float = 0.5) -> float:
    """
    predictions:   list of dicts {boxes: (K,4), scores: (K,), labels: (K,)}
    ground_truths: list of dicts {boxes: (J,4), labels: (J,)}
    All boxes in [x1,y1,x2,y2].  Returns scalar mAP.
    """
    # Gather all class ids
    all_labels = set()
    for gt in ground_truths:
        all_labels.update(gt["labels"].tolist())

    aps = []
    for cls in sorted(all_labels):
        # Collect all predictions and GT for this class across images
        det_scores, det_tp = [], []
        n_gt_total = 0

        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            gt_mask   = gt["labels"] == cls
            pred_mask = pred["labels"] == cls
            gt_boxes   = gt["boxes"][gt_mask]
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]

            n_gt_total += gt_boxes.shape[0]
            matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

            # Sort predictions by score descending
            order = pred_scores.argsort(descending=True)
            for idx in order:
                det_scores.append(pred_scores[idx].item())
                if gt_boxes.numel() == 0:
                    det_tp.append(0)
                    continue
                # IoU with all GT boxes
                box = pred_boxes[idx]
                tl = torch.max(box[:2], gt_boxes[:, :2])
                br = torch.min(box[2:], gt_boxes[:, 2:])
                inter = (br - tl).clamp(min=0).prod(dim=1)
                area_p = (box[2]-box[0]) * (box[3]-box[1])
                area_g = (gt_boxes[:,2]-gt_boxes[:,0]) * (gt_boxes[:,3]-gt_boxes[:,1])
                iou = inter / (area_p + area_g - inter).clamp(min=1e-6)

                best_iou, best_j = iou.max(dim=0)
                if best_iou >= iou_thresh and not matched[best_j]:
                    det_tp.append(1)
                    matched[best_j] = True
                else:
                    det_tp.append(0)

        if n_gt_total == 0:
            continue

        # Sort by score and compute precision/recall curve
        sorted_idx = sorted(range(len(det_scores)), key=lambda i: -det_scores[i])
        tp_cumsum = torch.tensor([det_tp[i] for i in sorted_idx]).cumsum(0).float()
        fp_cumsum = torch.arange(1, len(sorted_idx)+1).float() - tp_cumsum

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall    = tp_cumsum / n_gt_total
        aps.append(compute_ap(recall, precision))

    return sum(aps) / len(aps) if aps else 0.0

# --- test ---
preds = [
    {"boxes": torch.tensor([[0,0,10,10],[20,20,30,30],[0,0,12,12]], dtype=torch.float),
     "scores": torch.tensor([0.9, 0.8, 0.4]),
     "labels": torch.tensor([0, 1, 0])},
]
gts = [
    {"boxes": torch.tensor([[0,0,10,10],[20,20,30,30]], dtype=torch.float),
     "labels": torch.tensor([0, 1])},
]
print(f"mAP@0.5 = {mean_ap(preds, gts, 0.5):.4f}")  # 1.0 (both perfectly detected)
```

### 11. Confusion Matrix for Multi-Class Classification

Build a C x C confusion matrix where entry (i, j) counts predictions of class j when truth is class i.

```python
import torch

def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    preds:   (N,) predicted class indices
    targets: (N,) ground-truth class indices
    Returns: (C, C) confusion matrix — row = true, col = predicted
    """
    assert preds.shape == targets.shape
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm

def confusion_matrix_vectorized(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Same result, no Python loop — uses linear index trick."""
    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)

# --- test ---
preds   = torch.tensor([0, 1, 2, 1, 0, 2, 2, 0])
targets = torch.tensor([0, 1, 2, 2, 0, 1, 2, 1])
cm = confusion_matrix_vectorized(preds, targets, num_classes=3)
print(cm)
# Row=true, Col=pred
# Class 0: 2 correct
# Class 1: 1 correct, 1 predicted as 0, 1 predicted as 2
# Class 2: 2 correct, 1 predicted as 1
per_class_acc = cm.diag().float() / cm.sum(dim=1).float()
print(f"Per-class accuracy: {per_class_acc}")
```

### 12. Dice Loss for Segmentation

Implement Dice loss operating on soft predictions (after sigmoid) and binary masks.

```python
import torch

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    logits:  (N, 1, H, W) raw model output (pre-sigmoid)
    targets: (N, 1, H, W) binary ground truth {0, 1}
    Returns: scalar Dice loss = 1 - Dice coefficient
    """
    probs = torch.sigmoid(logits)

    # Flatten spatial dims per sample: (N, H*W)
    probs_flat  = probs.view(probs.shape[0], -1)
    target_flat = targets.view(targets.shape[0], -1)

    intersection = (probs_flat * target_flat).sum(dim=1)
    cardinality  = probs_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()

# --- test ---
N, H, W = 2, 8, 8
# Perfect prediction: loss should be near 0
targets = torch.randint(0, 2, (N, 1, H, W)).float()
perfect_logits = targets * 20 - 10   # large positive where target=1, large negative where 0
print(f"Dice loss (perfect): {dice_loss(perfect_logits, targets):.6f}")  # ≈ 0

# Random prediction: loss should be higher
random_logits = torch.randn(N, 1, H, W)
print(f"Dice loss (random):  {dice_loss(random_logits, targets):.4f}")
```

### 13. IoU Loss / Jaccard Score for Segmentation Masks

Compute Intersection over Union between predicted and ground-truth binary segmentation masks, and derive a differentiable IoU loss.

```python
import torch
import torch.nn as nn

def iou_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute IoU (Jaccard) score for binary segmentation masks.
    
    Args:
        preds: (N, H, W) float tensor of predicted probabilities or binary masks
        targets: (N, H, W) float tensor of ground-truth binary masks
        smooth: smoothing constant to avoid division by zero
    
    Returns:
        Scalar mean IoU score across the batch.
    """
    # Flatten spatial dims: (N, H*W)
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def iou_loss(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """IoU loss = 1 - IoU. Differentiable when preds are sigmoid outputs."""
    return 1.0 - iou_score(preds, targets, smooth)


# --- demo ---
if __name__ == "__main__":
    logits = torch.randn(4, 256, 256)
    masks = torch.randint(0, 2, (4, 256, 256), dtype=torch.float32)

    preds = torch.sigmoid(logits)
    score = iou_score(preds, masks)
    loss = iou_loss(preds, masks)
    print(f"IoU score: {score:.4f}, IoU loss: {loss:.4f}")
```

### 14. Focal Loss for Classification or Detection

Focal loss down-weights well-classified examples so the model focuses on hard negatives. Used in RetinaNet and similar one-stage detectors.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: balancing factor per class (scalar or tensor of shape (C,))
        gamma: focusing parameter; gamma=0 recovers standard CE
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw scores (pre-softmax) for multi-class,
                    or (N,) raw scores (pre-sigmoid) for binary
            targets: (N,) integer class labels
        """
        if logits.dim() == 1 or logits.size(1) == 1:
            # Binary focal loss
            logits = logits.view(-1)
            targets_float = targets.float()
            p = torch.sigmoid(logits)
            # p_t = p for positive, 1-p for negative
            p_t = p * targets_float + (1 - p) * (1 - targets_float)
            alpha_t = self.alpha * targets_float + (1 - self.alpha) * (1 - targets_float)
            # Use BCE with logits for numerical stability, then weight
            bce = F.binary_cross_entropy_with_logits(logits, targets_float, reduction="none")
            focal_weight = alpha_t * (1 - p_t) ** self.gamma
            loss = focal_weight * bce
        else:
            # Multi-class focal loss
            ce = F.cross_entropy(logits, targets, reduction="none")  # (N,)
            p = F.softmax(logits, dim=1)  # (N, C)
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
            focal_weight = self.alpha * (1 - p_t) ** self.gamma
            loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# --- demo ---
if __name__ == "__main__":
    # Multi-class
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    fl = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"Focal loss (multi-class): {fl(logits, targets):.4f}")

    # Binary
    logits_bin = torch.randn(8)
    targets_bin = torch.randint(0, 2, (8,))
    print(f"Focal loss (binary): {fl(logits_bin, targets_bin):.4f}")
```

### 15. Weighted Cross-Entropy for Class Imbalance

Assign per-class weights inversely proportional to class frequency so rare classes contribute more to the loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency weights from a label tensor."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    # Avoid div-by-zero for classes not present
    counts = counts.clamp(min=1.0)
    weights = 1.0 / counts
    # Normalize so weights sum to num_classes (keeps loss scale stable)
    weights = weights * num_classes / weights.sum()
    return weights


def weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: (N, C) unnormalized scores
        targets: (N,) integer class labels
        weights: (C,) per-class weights
    """
    return F.cross_entropy(logits, targets, weight=weights)


# --- demo ---
if __name__ == "__main__":
    num_classes = 4
    # Simulate imbalanced dataset: class 0 is rare
    targets = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    logits = torch.randn(len(targets), num_classes)

    weights = compute_class_weights(targets, num_classes)
    print(f"Class weights: {weights}")

    loss_weighted = weighted_cross_entropy(logits, targets, weights)
    loss_plain = F.cross_entropy(logits, targets)
    print(f"Weighted CE: {loss_weighted:.4f}, Plain CE: {loss_plain:.4f}")
```

### 16. Image Resize + Bounding Box Coordinate Remapping

When you resize an image, bounding boxes must be scaled by the same factors.

```python
import torch
import torch.nn.functional as F

def resize_image_and_boxes(image: torch.Tensor, boxes: torch.Tensor,
                           target_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resize image and remap bounding boxes accordingly.
    
    Args:
        image: (C, H, W) image tensor
        boxes: (N, 4) bounding boxes in [x_min, y_min, x_max, y_max] pixel coords
        target_size: (new_H, new_W)
    
    Returns:
        resized_image: (C, new_H, new_W)
        remapped_boxes: (N, 4) boxes scaled to new image dimensions
    """
    _, orig_h, orig_w = image.shape
    new_h, new_w = target_size

    # Resize image: F.interpolate expects (N, C, H, W)
    resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                            mode="bilinear", align_corners=False).squeeze(0)

    # Scale factors
    sx = new_w / orig_w
    sy = new_h / orig_h

    # Remap boxes: multiply x-coords by sx, y-coords by sy
    scale = torch.tensor([sx, sy, sx, sy], dtype=boxes.dtype, device=boxes.device)
    remapped = boxes * scale

    return resized, remapped


# --- demo ---
if __name__ == "__main__":
    img = torch.randn(3, 480, 640)
    boxes = torch.tensor([[100.0, 50.0, 300.0, 200.0],
                          [400.0, 300.0, 600.0, 450.0]])

    resized_img, new_boxes = resize_image_and_boxes(img, boxes, (240, 320))
    print(f"Original image: {img.shape} -> Resized: {resized_img.shape}")
    print(f"Original boxes:\n{boxes}")
    print(f"Remapped boxes:\n{new_boxes}")
```

### 17. Horizontal Flip Augmentation with Box Transformation

Flip the image left-right and mirror box x-coordinates around the image center.

```python
import torch

def horizontal_flip(image: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Horizontally flip an image and transform bounding boxes.
    
    Args:
        image: (C, H, W) image tensor
        boxes: (N, 4) in [x_min, y_min, x_max, y_max] pixel coordinates
    
    Returns:
        flipped_image, flipped_boxes
    """
    _, h, w = image.shape

    # Flip image along width axis
    flipped_image = image.flip(dims=[2])

    # Mirror x-coordinates: new_x = W - old_x
    # x_min becomes W - x_max, x_max becomes W - x_min
    flipped_boxes = boxes.clone()
    flipped_boxes[:, 0] = w - boxes[:, 2]  # new x_min = W - old x_max
    flipped_boxes[:, 2] = w - boxes[:, 0]  # new x_max = W - old x_min
    # y coordinates unchanged

    return flipped_image, flipped_boxes


# --- demo ---
if __name__ == "__main__":
    img = torch.randn(3, 100, 200)
    boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0],
                          [150.0, 30.0, 190.0, 80.0]])

    flipped_img, flipped_boxes = horizontal_flip(img, boxes)
    print(f"Original boxes:\n{boxes}")
    print(f"Flipped boxes:\n{flipped_boxes}")
    # Verify: box widths and heights should be preserved
    orig_wh = boxes[:, 2:] - boxes[:, :2]
    flip_wh = flipped_boxes[:, 2:] - flipped_boxes[:, :2]
    print(f"Sizes preserved: {torch.allclose(orig_wh, flip_wh)}")
```

### 18. Random Crop with Box Update

Crop a random region from the image and discard/clip boxes that fall outside.

```python
import torch

def random_crop_with_boxes(image: torch.Tensor, boxes: torch.Tensor,
                           labels: torch.Tensor, crop_size: tuple[int, int],
                           min_visibility: float = 0.3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random crop an image and update bounding boxes.
    Boxes partially inside are clipped; boxes mostly outside are dropped.
    
    Args:
        image: (C, H, W)
        boxes: (N, 4) [x_min, y_min, x_max, y_max]
        labels: (N,) class labels for each box
        crop_size: (crop_H, crop_W)
        min_visibility: minimum fraction of original box area that must remain
    
    Returns:
        cropped_image, surviving_boxes (in crop coords), surviving_labels
    """
    _, h, w = image.shape
    crop_h, crop_w = crop_size

    # Random top-left corner
    top = torch.randint(0, h - crop_h + 1, (1,)).item()
    left = torch.randint(0, w - crop_w + 1, (1,)).item()

    # Crop image
    cropped = image[:, top:top + crop_h, left:left + crop_w]

    # Shift boxes to crop coordinate system
    shifted = boxes.clone()
    shifted[:, 0] -= left
    shifted[:, 1] -= top
    shifted[:, 2] -= left
    shifted[:, 3] -= top

    # Clip to crop boundaries
    clipped = shifted.clone()
    clipped[:, 0] = clipped[:, 0].clamp(0, crop_w)
    clipped[:, 1] = clipped[:, 1].clamp(0, crop_h)
    clipped[:, 2] = clipped[:, 2].clamp(0, crop_w)
    clipped[:, 3] = clipped[:, 3].clamp(0, crop_h)

    # Compute surviving area vs original area
    orig_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    clipped_w = clipped[:, 2] - clipped[:, 0]
    clipped_h = clipped[:, 3] - clipped[:, 1]
    clipped_area = clipped_w * clipped_h

    # Keep boxes with positive area and sufficient visibility
    valid = (clipped_w > 0) & (clipped_h > 0) & (clipped_area / orig_area >= min_visibility)

    return cropped, clipped[valid], labels[valid]


# --- demo ---
if __name__ == "__main__":
    img = torch.randn(3, 400, 600)
    boxes = torch.tensor([[50., 50., 200., 200.],
                          [300., 100., 500., 350.],
                          [550., 350., 590., 390.]])
    labels = torch.tensor([0, 1, 2])

    cropped, new_boxes, new_labels = random_crop_with_boxes(img, boxes, labels, (300, 400))
    print(f"Cropped image: {cropped.shape}")
    print(f"Surviving boxes ({len(new_boxes)}/{len(boxes)}):\n{new_boxes}")
    print(f"Surviving labels: {new_labels}")
```

### 19. Letterbox Resize (YOLO-style)

Resize the image to fit within a target size while preserving aspect ratio, padding the remainder with a fill value.

```python
import torch
import torch.nn.functional as F

def letterbox_resize(image: torch.Tensor, target_size: int = 640,
                     fill_value: float = 0.5) -> tuple[torch.Tensor, dict]:
    """
    Letterbox resize: scale to fit target_size x target_size preserving
    aspect ratio, then pad the shorter dimension.
    
    Args:
        image: (C, H, W) image tensor
        target_size: square output dimension
        fill_value: padding fill value (0.5 = gray for [0,1] images)
    
    Returns:
        letterboxed: (C, target_size, target_size)
        meta: dict with scale, pad offsets for inverse-mapping detections
    """
    _, h, w = image.shape

    # Scale factor: fit the longer side to target_size
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize with aspect ratio preserved
    resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                            mode="bilinear", align_corners=False).squeeze(0)

    # Compute padding (center the image)
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    # F.pad format: (left, right, top, bottom)
    letterboxed = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant", value=fill_value)

    meta = {"scale": scale, "pad_left": pad_left, "pad_top": pad_top,
            "new_h": new_h, "new_w": new_w}
    return letterboxed, meta


def map_boxes_from_letterbox(boxes: torch.Tensor, meta: dict) -> torch.Tensor:
    """Map detections from letterbox coords back to original image coords."""
    boxes = boxes.clone().float()
    boxes[:, 0] -= meta["pad_left"]
    boxes[:, 1] -= meta["pad_top"]
    boxes[:, 2] -= meta["pad_left"]
    boxes[:, 3] -= meta["pad_top"]
    boxes /= meta["scale"]
    return boxes


# --- demo ---
if __name__ == "__main__":
    img = torch.randn(3, 720, 1280)  # landscape HD image
    lb, meta = letterbox_resize(img, target_size=640)
    print(f"Input: {img.shape} -> Letterboxed: {lb.shape}")
    print(f"Meta: {meta}")

    # Simulated detection in letterbox space
    det_boxes = torch.tensor([[100., 180., 300., 350.]])
    orig_boxes = map_boxes_from_letterbox(det_boxes, meta)
    print(f"Detection in letterbox: {det_boxes}")
    print(f"Mapped to original: {orig_boxes}")
```

### 20. Sliding-Window Inference on a Large Image

Run a model on overlapping patches and stitch results together, handling overlap by averaging.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def sliding_window_inference(image: torch.Tensor, model: nn.Module,
                             window_size: int = 256, stride: int = 128,
                             num_classes: int = 1) -> torch.Tensor:
    """
    Run a segmentation model on overlapping windows and average predictions.
    
    Args:
        image: (C, H, W) input image
        model: segmentation model that maps (B, C, window, window) -> (B, num_classes, window, window)
        window_size: patch size
        stride: step between patches (< window_size for overlap)
        num_classes: number of output channels
    
    Returns:
        output: (num_classes, H, W) averaged prediction map
    """
    c, h, w = image.shape
    device = image.device

    # Accumulators
    output_sum = torch.zeros(num_classes, h, w, device=device)
    count = torch.zeros(1, h, w, device=device)

    # Collect all patches and their positions
    patches, positions = [], []
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image[:, y:y + window_size, x:x + window_size]
            patches.append(patch)
            positions.append((y, x))

    # Handle right/bottom edges if image isn't evenly divisible
    # (slide window from the edge inward)
    if (h - window_size) % stride != 0:
        for x in range(0, w - window_size + 1, stride):
            patches.append(image[:, h - window_size:h, x:x + window_size])
            positions.append((h - window_size, x))
    if (w - window_size) % stride != 0:
        for y in range(0, h - window_size + 1, stride):
            patches.append(image[:, y:y + window_size, w - window_size:w])
            positions.append((y, w - window_size))

    # Batch inference
    batch = torch.stack(patches)  # (num_patches, C, ws, ws)
    with torch.no_grad():
        preds = model(batch)  # (num_patches, num_classes, ws, ws)

    # Accumulate
    for idx, (y, x) in enumerate(positions):
        output_sum[:, y:y + window_size, x:x + window_size] += preds[idx]
        count[:, y:y + window_size, x:x + window_size] += 1.0

    # Average overlapping regions
    output = output_sum / count.clamp(min=1.0)
    return output


# --- demo ---
if __name__ == "__main__":
    # Dummy segmentation model (identity-like)
    model = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1), nn.Sigmoid())
    model.eval()

    large_image = torch.randn(3, 512, 512)
    result = sliding_window_inference(large_image, model, window_size=256, stride=128, num_classes=1)
    print(f"Input: {large_image.shape}, Output: {result.shape}")
```

### 21. Patch Extraction from an Image Tensor

Extract non-overlapping or strided patches using `unfold`, the standard PyTorch approach.

```python
import torch

def extract_patches(image: torch.Tensor, patch_size: int,
                    stride: int | None = None) -> torch.Tensor:
    """
    Extract patches from an image tensor using unfold.
    
    Args:
        image: (C, H, W) image tensor
        patch_size: height and width of each square patch
        stride: step between patches (defaults to patch_size for non-overlapping)
    
    Returns:
        patches: (num_patches_h, num_patches_w, C, patch_size, patch_size)
    """
    if stride is None:
        stride = patch_size

    c, h, w = image.shape

    # unfold(dim, size, step) -> adds a new dim at the end
    # Unfold along H, then along W
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    # Shape: (C, n_h, n_w, patch_size, patch_size)

    # Rearrange to (n_h, n_w, C, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()

    return patches


def reconstruct_from_patches(patches: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """
    Reconstruct image from non-overlapping patches (stride == patch_size).
    
    Args:
        patches: (n_h, n_w, C, ph, pw)
        image_size: (H, W) original image size
    
    Returns:
        image: (C, H, W)
    """
    n_h, n_w, c, ph, pw = patches.shape
    # Rearrange: (C, n_h, n_w, ph, pw) -> (C, n_h*ph, n_w*pw)
    patches = patches.permute(2, 0, 3, 1, 4)  # (C, n_h, ph, n_w, pw)
    image = patches.reshape(c, n_h * ph, n_w * pw)
    # Crop to original size if needed
    h, w = image_size
    return image[:, :h, :w]


# --- demo ---
if __name__ == "__main__":
    img = torch.randn(3, 224, 224)
    patches = extract_patches(img, patch_size=16)
    print(f"Image: {img.shape} -> Patches: {patches.shape}")
    # (14, 14, 3, 16, 16) = 196 patches, like a ViT

    reconstructed = reconstruct_from_patches(patches, (224, 224))
    print(f"Reconstructed: {reconstructed.shape}")
    print(f"Perfect reconstruction: {torch.allclose(img, reconstructed)}")

    # Overlapping patches
    overlapping = extract_patches(img, patch_size=32, stride=16)
    print(f"Overlapping patches (32, stride 16): {overlapping.shape}")
```

### 22. Connected Components on a Binary Mask

BFS-based connected component labeling implemented purely with Python/PyTorch tensors.

```python
import torch
from collections import deque

def connected_components(mask: torch.Tensor, connectivity: int = 4) -> tuple[torch.Tensor, int]:
    """
    Label connected components in a binary mask using BFS.
    
    Args:
        mask: (H, W) binary tensor (0 or 1)
        connectivity: 4 or 8 (neighbor connectivity)
    
    Returns:
        labels: (H, W) integer tensor where each component has a unique ID (1-indexed)
        num_components: total number of connected components
    """
    h, w = mask.shape
    labels = torch.zeros_like(mask, dtype=torch.int32)
    visited = torch.zeros_like(mask, dtype=torch.bool)

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # 8-connectivity
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

    component_id = 0
    mask_bool = mask.bool()

    for i in range(h):
        for j in range(w):
            if mask_bool[i, j] and not visited[i, j]:
                component_id += 1
                # BFS from (i, j)
                queue = deque([(i, j)])
                visited[i, j] = True
                labels[i, j] = component_id

                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask_bool[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            labels[ny, nx] = component_id
                            queue.append((ny, nx))

    return labels, component_id


# --- demo ---
if __name__ == "__main__":
    mask = torch.zeros(8, 8, dtype=torch.uint8)
    # Two separate blobs
    mask[1:3, 1:3] = 1
    mask[5:7, 5:7] = 1
    # A diagonal pixel (connected only under 8-connectivity)
    mask[3, 3] = 1

    labels_4, n4 = connected_components(mask, connectivity=4)
    labels_8, n8 = connected_components(mask, connectivity=8)

    print(f"Mask:\n{mask}")
    print(f"\n4-connected: {n4} components\n{labels_4}")
    print(f"\n8-connected: {n8} components\n{labels_8}")
```

### 23. Morphological Dilation and Erosion on Binary Masks

Implement dilation and erosion using convolution with a structuring element.

```python
import torch
import torch.nn.functional as F

def dilate(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """
    Morphological dilation: a pixel is 1 if ANY neighbor within the
    structuring element is 1. Implemented via max-pooling.
    
    Args:
        mask: (H, W) binary tensor
        kernel_size: size of the square structuring element
        iterations: number of times to apply dilation
    """
    pad = kernel_size // 2
    result = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    for _ in range(iterations):
        result = F.max_pool2d(result, kernel_size, stride=1, padding=pad)

    return (result.squeeze(0).squeeze(0) > 0.5).to(mask.dtype)


def erode(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """
    Morphological erosion: a pixel is 1 only if ALL neighbors within the
    structuring element are 1. Erosion = complement of dilation of complement.
    Equivalently, use -max_pool(-mask).
    """
    pad = kernel_size // 2
    # Negate, dilate (max_pool), negate
    result = (-mask.float()).unsqueeze(0).unsqueeze(0)

    for _ in range(iterations):
        result = F.max_pool2d(result, kernel_size, stride=1, padding=pad)

    return (-result.squeeze(0).squeeze(0) > 0.5).to(mask.dtype)


def opening(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Erosion followed by dilation. Removes small foreground noise."""
    return dilate(erode(mask, kernel_size), kernel_size)


def closing(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Dilation followed by erosion. Fills small holes."""
    return erode(dilate(mask, kernel_size), kernel_size)


# --- demo ---
if __name__ == "__main__":
    # Create a mask with a small hole and a small noise blob
    mask = torch.zeros(16, 16, dtype=torch.uint8)
    mask[4:12, 4:12] = 1   # main square
    mask[7, 7] = 0          # small hole inside
    mask[14, 14] = 1        # isolated noise pixel

    dilated = dilate(mask, kernel_size=3)
    eroded = erode(mask, kernel_size=3)
    opened = opening(mask, kernel_size=3)
    closed = closing(mask, kernel_size=3)

    print(f"Original foreground pixels: {mask.sum().item()}")
    print(f"After dilation: {dilated.sum().item()}")
    print(f"After erosion: {eroded.sum().item()}")
    print(f"After opening (removes noise): {opened.sum().item()}")
    print(f"After closing (fills holes): {closed.sum().item()}")
```

### 24. Implement Sobel Edge Detection

Given an image tensor, apply Sobel filters to compute gradient magnitude for edge detection.

```python
import torch
import torch.nn.functional as F

def sobel_edge_detection(image: torch.Tensor) -> torch.Tensor:
    """
    Args:
        image: (H, W) or (1, 1, H, W) grayscale image tensor, values in [0, 1]
    Returns:
        magnitude: (H, W) gradient magnitude
    """
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=image.dtype).reshape(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=image.dtype).reshape(1, 1, 3, 3)

    gx = F.conv2d(image, sobel_x, padding=1)  # (1, 1, H, W)
    gy = F.conv2d(image, sobel_y, padding=1)

    magnitude = torch.sqrt(gx ** 2 + gy ** 2).squeeze()
    return magnitude

# Demo
img = torch.rand(64, 64)
edges = sobel_edge_detection(img)
print(f"Input: {img.shape} -> Edges: {edges.shape}, range [{edges.min():.3f}, {edges.max():.3f}]")
```

### 25. Implement Canny-like Gradient Magnitude + Thresholding Core Steps

Compute gradient magnitude and direction via Sobel, then apply non-maximum suppression along gradient direction and double thresholding with hysteresis.

```python
import torch
import torch.nn.functional as F

def canny_core(image: torch.Tensor, low: float = 0.1, high: float = 0.3) -> torch.Tensor:
    """
    Args:
        image: (H, W) grayscale tensor in [0, 1]
        low, high: thresholds for hysteresis
    Returns:
        edges: (H, W) binary edge map
    """
    img = image.unsqueeze(0).unsqueeze(0).float()

    # Step 1: Gaussian blur (5x5 approx)
    k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
    gauss = (k.unsqueeze(1) @ k.unsqueeze(0)).float()
    gauss = (gauss / gauss.sum()).reshape(1, 1, 5, 5)
    img = F.conv2d(img, gauss, padding=2)

    # Step 2: Sobel gradients
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).reshape(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).reshape(1,1,3,3)
    gx = F.conv2d(img, sx, padding=1).squeeze()
    gy = F.conv2d(img, sy, padding=1).squeeze()

    mag = torch.sqrt(gx**2 + gy**2)
    mag = mag / (mag.max() + 1e-8)  # normalize to [0, 1]
    angle = torch.atan2(gy, gx)

    # Step 3: Non-maximum suppression (quantize angle to 4 directions)
    H, W = mag.shape
    nms = mag.clone()
    # Quantize angles to 0, 45, 90, 135 degrees
    angle_deg = (angle * 180 / 3.14159) % 180  # map to [0, 180)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle_deg[i, j].item()
            # Determine neighbors along gradient direction
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                n1, n2 = mag[i, j-1], mag[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = mag[i-1, j+1], mag[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = mag[i-1, j], mag[i+1, j]
            else:
                n1, n2 = mag[i-1, j-1], mag[i+1, j+1]

            if mag[i, j] < n1 or mag[i, j] < n2:
                nms[i, j] = 0

    # Step 4: Double threshold + hysteresis
    strong = nms >= high
    weak = (nms >= low) & (nms < high)

    # Simple hysteresis: weak pixels adjacent to strong become strong
    edges = strong.clone()
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if weak[i, j] and strong[i-1:i+2, j-1:j+2].any():
                edges[i, j] = True

    return edges.float()

# Demo
img = torch.rand(32, 32)
edges = canny_core(img, low=0.05, high=0.15)
print(f"Canny edges: {edges.shape}, edge pixels: {edges.sum().item():.0f}")
```

### 26. Implement 2D Convolution from Scratch

Implement 2D convolution using only basic PyTorch tensor operations (no nn.Conv2d or F.conv2d).

```python
import torch

def conv2d_scratch(
    input: torch.Tensor,   # (N, C_in, H, W)
    weight: torch.Tensor,  # (C_out, C_in, kH, kW)
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    N, C_in, H, W = input.shape
    C_out, C_in_k, kH, kW = weight.shape
    assert C_in == C_in_k

    # Pad input
    if padding > 0:
        input = torch.nn.functional.pad(input, [padding]*4)
        H, W = H + 2*padding, W + 2*padding

    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    # im2col: extract all patches as columns
    # Shape: (N, C_in, kH, kW, H_out, W_out)
    cols = input.unfold(2, kH, stride).unfold(3, kW, stride)
    # Reshape to (N, C_in*kH*kW, H_out*W_out)
    cols = cols.contiguous().reshape(N, C_in * kH * kW, H_out * W_out)

    # Reshape weight to (C_out, C_in*kH*kW)
    w = weight.reshape(C_out, -1)

    # Matrix multiply: (C_out, C_in*kH*kW) x (N, C_in*kH*kW, H_out*W_out)
    out = torch.bmm(w.unsqueeze(0).expand(N, -1, -1), cols)  # (N, C_out, H_out*W_out)
    out = out.reshape(N, C_out, H_out, W_out)

    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    return out

# Demo
x = torch.randn(2, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
b = torch.randn(16)
out = conv2d_scratch(x, w, b, stride=1, padding=1)
print(f"Input: {x.shape} -> Output: {out.shape}")  # (2, 16, 8, 8)

# Verify against PyTorch
ref = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
print(f"Max error vs F.conv2d: {(out - ref).abs().max().item():.6f}")
```

### 27. Implement Max-Pooling from Scratch

Implement 2D max pooling using only tensor indexing operations.

```python
import torch

def maxpool2d_scratch(
    input: torch.Tensor,  # (N, C, H, W)
    kernel_size: int = 2,
    stride: int = 2,
) -> torch.Tensor:
    N, C, H, W = input.shape
    k = kernel_size
    H_out = (H - k) // stride + 1
    W_out = (W - k) // stride + 1

    # Use unfold to extract sliding windows
    # unfold(dim, size, step) -> adds a new dim at the end
    patches = input.unfold(2, k, stride).unfold(3, k, stride)
    # patches shape: (N, C, H_out, W_out, k, k)

    # Max over the last two dims (the pooling window)
    out = patches.contiguous().reshape(N, C, H_out, W_out, k * k).max(dim=-1).values
    return out

# Demo
x = torch.randn(1, 3, 8, 8)
out = maxpool2d_scratch(x, kernel_size=2, stride=2)
print(f"Input: {x.shape} -> Pooled: {out.shape}")  # (1, 3, 4, 4)

# Verify
ref = torch.nn.functional.max_pool2d(x, 2, 2)
print(f"Max error vs F.max_pool2d: {(out - ref).abs().max().item():.6f}")
```

### 28. Implement RoI Pooling / RoI Align Intuition Version

Given a feature map and region proposals, extract fixed-size features per RoI using bilinear sampling (RoI Align style).

```python
import torch

def roi_align(
    feature_map: torch.Tensor,  # (1, C, H, W)
    rois: torch.Tensor,         # (num_rois, 4) in [x1, y1, x2, y2] on feature map coords
    output_size: int = 7,
    sampling_points: int = 2,
) -> torch.Tensor:
    """Simplified RoI Align: bilinear interpolation at regular sample points."""
    _, C, H, W = feature_map.shape
    num_rois = rois.shape[0]
    out = torch.zeros(num_rois, C, output_size, output_size)

    for idx in range(num_rois):
        x1, y1, x2, y2 = rois[idx]
        roi_h = (y2 - y1).item()
        roi_w = (x2 - x1).item()
        bin_h = roi_h / output_size
        bin_w = roi_w / output_size

        for i in range(output_size):
            for j in range(output_size):
                # Sample points within this bin
                vals = []
                for sy in range(sampling_points):
                    for sx in range(sampling_points):
                        # Compute sample coordinate (center of sub-bin)
                        y = y1.item() + bin_h * (i + (sy + 0.5) / sampling_points)
                        x = x1.item() + bin_w * (j + (sx + 0.5) / sampling_points)

                        # Bilinear interpolation
                        x0, y0 = int(x), int(y)
                        x1c = min(x0 + 1, W - 1)
                        y1c = min(y0 + 1, H - 1)
                        x0 = max(x0, 0)
                        y0 = max(y0, 0)

                        wa = (x1c - x) * (y1c - y)
                        wb = (x - x0) * (y1c - y)
                        wc = (x1c - x) * (y - y0)
                        wd = (x - x0) * (y - y0)

                        val = (wa * feature_map[0, :, y0, x0] +
                               wb * feature_map[0, :, y0, x1c] +
                               wc * feature_map[0, :, y1c, x0] +
                               wd * feature_map[0, :, y1c, x1c])
                        vals.append(val)

                out[idx, :, i, j] = torch.stack(vals).mean(dim=0)

    return out

# Demo
fmap = torch.randn(1, 256, 14, 14)
rois = torch.tensor([[1.5, 1.5, 10.3, 10.3],
                      [0.0, 0.0, 7.0, 7.0]])
pooled = roi_align(fmap, rois, output_size=7)
print(f"Feature map: {fmap.shape}, RoIs: {rois.shape} -> Pooled: {pooled.shape}")
# (2, 256, 7, 7)
```

### 29. Implement Anchor Generation for Different Scales and Aspect Ratios

Generate anchor boxes at every spatial position of a feature map for given scales and aspect ratios.

```python
import torch

def generate_anchors(
    feature_size: tuple,      # (H, W) of feature map
    stride: int = 16,         # pixels per feature map cell
    scales: list = [64, 128, 256],
    aspect_ratios: list = [0.5, 1.0, 2.0],
) -> torch.Tensor:
    """
    Returns:
        anchors: (H * W * num_anchors_per_cell, 4) in [x1, y1, x2, y2] format
                 in original image coordinates.
    """
    H, W = feature_size
    num_per_cell = len(scales) * len(aspect_ratios)

    # Base anchors centered at origin for each (scale, ratio) pair
    base_anchors = []
    for s in scales:
        for r in aspect_ratios:
            # area = s^2, w/h = r => w = s*sqrt(r), h = s/sqrt(r)
            w = s * (r ** 0.5)
            h = s / (r ** 0.5)
            base_anchors.append([-w/2, -h/2, w/2, h/2])
    base_anchors = torch.tensor(base_anchors, dtype=torch.float32)  # (K, 4)

    # Grid of center positions
    shift_y = (torch.arange(H, dtype=torch.float32) + 0.5) * stride
    shift_x = (torch.arange(W, dtype=torch.float32) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')

    shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)  # (H, W, 4)
    shifts = shifts.reshape(-1, 1, 4)          # (H*W, 1, 4)
    base_anchors = base_anchors.reshape(1, -1, 4)  # (1, K, 4)

    all_anchors = (shifts + base_anchors).reshape(-1, 4)  # (H*W*K, 4)
    return all_anchors

# Demo
anchors = generate_anchors((3, 3), stride=16, scales=[64, 128], aspect_ratios=[0.5, 1.0, 2.0])
print(f"Anchors: {anchors.shape}")  # (3*3*6, 4) = (54, 4)
print(f"First cell anchors:\n{anchors[:6]}")
```

### 30. Implement Anchor Matching to GT Boxes Using IoU Thresholds

Assign each anchor a label (positive/negative/ignore) based on IoU overlap with ground-truth boxes.

```python
import torch

def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes [x1,y1,x2,y2]."""
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)

def match_anchors_to_gt(
    anchors: torch.Tensor,   # (N, 4)
    gt_boxes: torch.Tensor,  # (M, 4)
    pos_thresh: float = 0.7,
    neg_thresh: float = 0.3,
) -> tuple:
    """
    Returns:
        labels: (N,) — 1=positive, 0=negative, -1=ignore
        matched_gt_idx: (N,) — index of matched GT box for each anchor
    """
    iou_matrix = compute_iou_matrix(anchors, gt_boxes)  # (N, M)

    max_iou_per_anchor, matched_gt_idx = iou_matrix.max(dim=1)  # (N,)

    labels = torch.full((anchors.shape[0],), -1, dtype=torch.long)

    # Negative: max IoU < neg_thresh
    labels[max_iou_per_anchor < neg_thresh] = 0

    # Positive: max IoU >= pos_thresh
    labels[max_iou_per_anchor >= pos_thresh] = 1

    # Also mark the best anchor for each GT as positive (ensures every GT has a match)
    best_anchor_per_gt = iou_matrix.argmax(dim=0)  # (M,)
    labels[best_anchor_per_gt] = 1
    matched_gt_idx[best_anchor_per_gt] = torch.arange(gt_boxes.shape[0])

    return labels, matched_gt_idx

# Demo
anchors = torch.tensor([[0,0,50,50],[30,30,80,80],[100,100,150,150],[10,10,60,60]], dtype=torch.float32)
gt_boxes = torch.tensor([[5,5,55,55],[35,35,85,85]], dtype=torch.float32)

labels, matched = match_anchors_to_gt(anchors, gt_boxes)
print(f"Labels: {labels}")       # positive/negative/ignore per anchor
print(f"Matched GT: {matched}")  # which GT each anchor maps to
```

### 31. Implement Bbox Regression Target Encoding/Decoding

Encode offsets from anchor to GT box, and decode predicted offsets back to boxes.

```python
import torch

def encode_bbox_targets(
    anchors: torch.Tensor,   # (N, 4) [x1, y1, x2, y2]
    gt_boxes: torch.Tensor,  # (N, 4) matched GT for each anchor
) -> torch.Tensor:
    """Encode GT boxes as regression targets relative to anchors (Faster R-CNN style)."""
    # Convert to center format
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w = anchors[:, 2] - anchors[:, 0]
    a_h = anchors[:, 3] - anchors[:, 1]

    g_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    g_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    g_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    g_h = gt_boxes[:, 3] - gt_boxes[:, 1]

    tx = (g_cx - a_cx) / a_w
    ty = (g_cy - a_cy) / a_h
    tw = torch.log(g_w / a_w)
    th = torch.log(g_h / a_h)

    return torch.stack([tx, ty, tw, th], dim=1)

def decode_bbox_predictions(
    anchors: torch.Tensor,     # (N, 4)
    predictions: torch.Tensor, # (N, 4) [tx, ty, tw, th]
) -> torch.Tensor:
    """Decode predicted offsets back to [x1, y1, x2, y2] boxes."""
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w = anchors[:, 2] - anchors[:, 0]
    a_h = anchors[:, 3] - anchors[:, 1]

    pred_cx = predictions[:, 0] * a_w + a_cx
    pred_cy = predictions[:, 1] * a_h + a_cy
    pred_w = torch.exp(predictions[:, 2]) * a_w
    pred_h = torch.exp(predictions[:, 3]) * a_h

    x1 = pred_cx - pred_w / 2
    y1 = pred_cy - pred_h / 2
    x2 = pred_cx + pred_w / 2
    y2 = pred_cy + pred_h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)

# Demo: round-trip test
anchors = torch.tensor([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=torch.float32)
gt_boxes = torch.tensor([[15, 12, 55, 48], [110, 105, 210, 195]], dtype=torch.float32)

targets = encode_bbox_targets(anchors, gt_boxes)
decoded = decode_bbox_predictions(anchors, targets)
print(f"Targets: {targets}")
print(f"Decoded matches GT: {torch.allclose(decoded, gt_boxes, atol=1e-4)}")
```

### 32. Implement Hard Negative Mining for Detection Examples

Select the hardest negative samples (highest classification loss) to balance positive/negative ratio during training.

```python
import torch

def hard_negative_mining(
    cls_loss: torch.Tensor,  # (N,) per-anchor classification loss
    labels: torch.Tensor,    # (N,) 1=pos, 0=neg, -1=ignore
    neg_pos_ratio: int = 3,
) -> torch.Tensor:
    """
    Select hard negatives: keep top-k negatives by loss where k = neg_pos_ratio * num_positives.

    Returns:
        mask: (N,) bool tensor — True for selected positives and hard negatives
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    num_pos = pos_mask.sum().item()
    num_neg_to_keep = int(neg_pos_ratio * num_pos)
    num_neg_to_keep = min(num_neg_to_keep, neg_mask.sum().item())

    # Set non-negative losses to 0 so they're never selected
    neg_loss = cls_loss.clone()
    neg_loss[~neg_mask] = 0.0

    # Sort negatives by loss descending, keep top-k
    _, neg_indices = neg_loss.sort(descending=True)
    hard_neg_mask = torch.zeros_like(labels, dtype=torch.bool)
    hard_neg_mask[neg_indices[:num_neg_to_keep]] = True

    # Final mask: positives + selected hard negatives
    selected = pos_mask | hard_neg_mask
    return selected

# Demo
torch.manual_seed(42)
num_anchors = 1000
labels = torch.full((num_anchors,), 0, dtype=torch.long)   # mostly negatives
labels[:20] = 1    # 20 positives
labels[20:30] = -1 # 10 ignored

cls_loss = torch.rand(num_anchors)  # simulated per-anchor loss

mask = hard_negative_mining(cls_loss, labels, neg_pos_ratio=3)
print(f"Positives: {(labels == 1).sum().item()}")
print(f"Selected: {mask.sum().item()} (expect ~80: 20 pos + 60 neg)")
print(f"Selected positives: {(mask & (labels == 1)).sum().item()}")
print(f"Selected negatives: {(mask & (labels == 0)).sum().item()}")
```

### 33. Implement a Mini Dataloader That Returns Image, Boxes, Labels

A simple dataset and manual batching loop for object detection data.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class MiniDetectionDataset(Dataset):
    """Synthetic detection dataset for demonstration."""

    def __init__(self, num_images: int = 20, img_size: int = 224, max_objects: int = 5):
        self.num_images = num_images
        self.img_size = img_size
        self.max_objects = max_objects

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Simulate an RGB image
        image = torch.rand(3, self.img_size, self.img_size)

        # Random number of objects per image
        num_obj = torch.randint(1, self.max_objects + 1, (1,)).item()

        # Random boxes: ensure x2 > x1, y2 > y1
        xy1 = torch.rand(num_obj, 2) * (self.img_size * 0.7)
        wh = torch.rand(num_obj, 2) * (self.img_size * 0.3) + 10
        xy2 = (xy1 + wh).clamp(max=self.img_size)
        boxes = torch.cat([xy1, xy2], dim=1)  # (num_obj, 4)

        # Random class labels (e.g., 5 classes)
        labels = torch.randint(0, 5, (num_obj,))

        return image, boxes, labels

# Demo
dataset = MiniDetectionDataset(num_images=10)
img, boxes, labels = dataset[0]
print(f"Image: {img.shape}, Boxes: {boxes.shape}, Labels: {labels.shape}")

# Simple manual batching (no custom collate yet)
for i in range(0, len(dataset), 4):
    batch = [dataset[j] for j in range(i, min(i + 4, len(dataset)))]
    images = torch.stack([b[0] for b in batch])
    print(f"Batch images: {images.shape}, num_boxes per image: {[b[1].shape[0] for b in batch]}")
```

### 34. Implement a Custom Collate Function for Variable Number of Boxes per Image

Handle detection batches where each image has a different number of bounding boxes.

```python
import torch
from torch.utils.data import DataLoader

def detection_collate_fn(batch):
    """
    Custom collate for detection: stack images, keep boxes/labels as lists.

    Args:
        batch: list of (image, boxes, labels) tuples
    Returns:
        images: (N, C, H, W) stacked tensor
        targets: list of dicts, each with 'boxes' and 'labels' tensors
    """
    images = torch.stack([item[0] for item in batch])

    targets = []
    for _, boxes, labels in batch:
        targets.append({
            'boxes': boxes,     # (num_obj_i, 4)
            'labels': labels,   # (num_obj_i,)
        })

    return images, targets

# Alternative: pad boxes to max count in batch (useful for some architectures)
def detection_collate_padded(batch):
    """Pad boxes/labels to the max object count in the batch."""
    images = torch.stack([item[0] for item in batch])
    max_obj = max(item[1].shape[0] for item in batch)

    padded_boxes = torch.zeros(len(batch), max_obj, 4)
    padded_labels = torch.full((len(batch), max_obj), -1, dtype=torch.long)  # -1 = padding
    num_objects = torch.zeros(len(batch), dtype=torch.long)

    for i, (_, boxes, labels) in enumerate(batch):
        n = boxes.shape[0]
        padded_boxes[i, :n] = boxes
        padded_labels[i, :n] = labels
        num_objects[i] = n

    return images, padded_boxes, padded_labels, num_objects

# Demo with MiniDetectionDataset from Q33
from torch.utils.data import Dataset

class MiniDetectionDataset(Dataset):
    def __init__(self, n=10, size=64):
        self.n, self.size = n, size
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        img = torch.rand(3, self.size, self.size)
        num_obj = torch.randint(1, 6, (1,)).item()
        xy1 = torch.rand(num_obj, 2) * 40
        boxes = torch.cat([xy1, xy1 + torch.rand(num_obj, 2) * 20 + 5], dim=1)
        labels = torch.randint(0, 3, (num_obj,))
        return img, boxes, labels

dataset = MiniDetectionDataset()

# List-based collate
loader = DataLoader(dataset, batch_size=4, collate_fn=detection_collate_fn)
for images, targets in loader:
    print(f"Batch: {images.shape}, targets: {[t['boxes'].shape for t in targets]}")
    break

# Padded collate
loader2 = DataLoader(dataset, batch_size=4, collate_fn=detection_collate_padded)
for images, boxes, labels, counts in loader2:
    print(f"Padded batch: images={images.shape}, boxes={boxes.shape}, counts={counts}")
    break
```

### 35. Implement Per-Class NMS + Score Thresholding Pipeline End-to-End

Full post-processing pipeline: filter by score, apply NMS per class, and collect final detections.

```python
import torch

def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-6)

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Greedy NMS. Returns indices of kept boxes."""
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = compute_iou_matrix(boxes[i:i+1], boxes[remaining]).squeeze(0)
        mask = ious <= iou_thresh
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long)

def per_class_nms_pipeline(
    boxes: torch.Tensor,       # (N, 4)
    class_scores: torch.Tensor, # (N, num_classes) — softmax/sigmoid scores
    score_thresh: float = 0.05,
    iou_thresh: float = 0.5,
    max_detections: int = 100,
) -> dict:
    """
    Full detection post-processing pipeline.

    Returns:
        dict with 'boxes' (K, 4), 'scores' (K,), 'labels' (K,)
    """
    num_classes = class_scores.shape[1]

    all_boxes, all_scores, all_labels = [], [], []

    for cls_id in range(num_classes):
        scores = class_scores[:, cls_id]

        # Step 1: Score thresholding
        mask = scores > score_thresh
        if mask.sum() == 0:
            continue

        cls_boxes = boxes[mask]
        cls_scores = scores[mask]

        # Step 2: Per-class NMS
        keep = nms(cls_boxes, cls_scores, iou_thresh)

        all_boxes.append(cls_boxes[keep])
        all_scores.append(cls_scores[keep])
        all_labels.append(torch.full((keep.numel(),), cls_id, dtype=torch.long))

    if len(all_boxes) == 0:
        return {'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0), 'labels': torch.zeros(0, dtype=torch.long)}

    all_boxes = torch.cat(all_boxes)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    # Step 3: Keep top-k across all classes
    if all_scores.numel() > max_detections:
        topk = all_scores.topk(max_detections).indices
        all_boxes = all_boxes[topk]
        all_scores = all_scores[topk]
        all_labels = all_labels[topk]

    return {'boxes': all_boxes, 'scores': all_scores, 'labels': all_labels}

# Demo
torch.manual_seed(0)
num_proposals = 500
num_classes = 5

boxes = torch.rand(num_proposals, 2) * 200
boxes = torch.cat([boxes, boxes + torch.rand(num_proposals, 2) * 80 + 20], dim=1)
scores = torch.rand(num_proposals, num_classes) * 0.3  # mostly low scores
# Inject some high-confidence detections
scores[:10, 0] = torch.rand(10) * 0.5 + 0.5
scores[5:15, 2] = torch.rand(10) * 0.5 + 0.5

results = per_class_nms_pipeline(boxes, scores, score_thresh=0.3, iou_thresh=0.5)
print(f"Final detections: {results['boxes'].shape[0]}")
print(f"  Boxes:  {results['boxes'].shape}")
print(f"  Scores: {results['scores'][:5]}")
print(f"  Labels: {results['labels'][:5]}")

# Per-class breakdown
for c in range(num_classes):
    count = (results['labels'] == c).sum().item()
    if count > 0:
        print(f"  Class {c}: {count} detections")
```

---

## Part 2: ML / PyTorch Implementation Questions (50 questions)

### 1. Linear Regression from Scratch

Fit a linear model y = Xw + b using PyTorch tensors and manual gradient descent (no nn.Module).

```python
import torch

def linear_regression(X_train, y_train, lr=0.01, epochs=1000):
    """Linear regression from scratch using gradient descent.
    Args:
        X_train: (N, D) feature matrix
        y_train: (N,) targets
    Returns:
        w: (D,) weight vector, b: scalar bias
    """
    N, D = X_train.shape
    w = torch.zeros(D, requires_grad=False)
    b = torch.zeros(1)

    for _ in range(epochs):
        # Forward
        y_pred = X_train @ w + b          # (N,)
        residual = y_pred - y_train        # (N,)
        loss = (residual ** 2).mean()

        # Gradients (analytical)
        dw = (2.0 / N) * (X_train.T @ residual)   # (D,)
        db = (2.0 / N) * residual.sum()

        # Update
        w -= lr * dw
        b -= lr * db

    return w, b

# --- demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.randn(200, 3)
    w_true = torch.tensor([1.0, -2.0, 3.0])
    y = X @ w_true + 0.5 + 0.1 * torch.randn(200)

    w, b = linear_regression(X, y, lr=0.05, epochs=500)
    print(f"Learned w={w.tolist()}, b={b.item():.4f}")
```

### 2. Logistic Regression from Scratch

Binary classification with sigmoid and manual gradient descent.

```python
import torch

def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))

def logistic_regression(X, y, lr=0.1, epochs=1000):
    """Logistic regression with manual gradients.
    Args:
        X: (N, D) features
        y: (N,) binary labels in {0, 1}
    Returns:
        w: (D,), b: scalar
    """
    N, D = X.shape
    w = torch.zeros(D)
    b = torch.zeros(1)

    for _ in range(epochs):
        logits = X @ w + b                       # (N,)
        p = sigmoid(logits)                       # (N,)

        # BCE loss (for monitoring)
        # loss = -( y*log(p) + (1-y)*log(1-p) ).mean()

        # Gradient of BCE w.r.t. logits: (p - y)
        err = p - y                               # (N,)
        dw = (1.0 / N) * (X.T @ err)
        db = (1.0 / N) * err.sum()

        w -= lr * dw
        b -= lr * db

    return w, b

def predict(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).long()

# --- demo ---
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(300, 2)
    y = ((X[:, 0] + X[:, 1]) > 0).float()
    w, b = logistic_regression(X, y)
    acc = (predict(X, w, b) == y.long()).float().mean()
    print(f"Training accuracy: {acc:.2%}")
```

### 3. Softmax Regression for Multi-Class Classification

Multinomial logistic regression with manual softmax + cross-entropy gradients.

```python
import torch

def softmax(logits):
    """Numerically stable softmax. logits: (N, C)"""
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=1, keepdim=True)

def softmax_regression(X, y, num_classes, lr=0.1, epochs=500):
    """
    Args:
        X: (N, D), y: (N,) integer class labels
    Returns:
        W: (D, C), b: (C,)
    """
    N, D = X.shape
    C = num_classes
    W = torch.zeros(D, C)
    b = torch.zeros(C)

    # One-hot encode targets
    Y_onehot = torch.zeros(N, C)
    Y_onehot.scatter_(1, y.unsqueeze(1), 1.0)

    for _ in range(epochs):
        logits = X @ W + b                # (N, C)
        probs = softmax(logits)            # (N, C)

        # Gradient: (1/N) * X^T (probs - onehot)
        diff = probs - Y_onehot            # (N, C)
        dW = (1.0 / N) * (X.T @ diff)     # (D, C)
        db = (1.0 / N) * diff.sum(dim=0)   # (C,)

        W -= lr * dW
        b -= lr * db

    return W, b

# --- demo ---
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(300, 4)
    y = torch.randint(0, 3, (300,))
    W, b = softmax_regression(X, y, num_classes=3)
    preds = (X @ W + b).argmax(dim=1)
    print(f"Accuracy: {(preds == y).float().mean():.2%}")
```

### 4. Cross-Entropy Loss Manually

Numerically stable cross-entropy from raw logits.

```python
import torch

def cross_entropy_loss(logits, targets):
    """
    Args:
        logits: (N, C) raw scores (unnormalized)
        targets: (N,) integer class labels
    Returns:
        scalar mean loss
    """
    N, C = logits.shape
    # Log-sum-exp trick for numerical stability
    max_logits = logits.max(dim=1, keepdim=True).values
    log_sum_exp = max_logits.squeeze(1) + torch.log(
        torch.exp(logits - max_logits).sum(dim=1)
    )
    # -log P(correct class) = -logit[correct] + log_sum_exp
    correct_logits = logits[torch.arange(N), targets]
    loss = (-correct_logits + log_sum_exp).mean()
    return loss

# --- verify against PyTorch ---
if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    ours = cross_entropy_loss(logits, targets)
    ref = torch.nn.functional.cross_entropy(logits, targets)
    print(f"Ours: {ours:.6f}  Reference: {ref:.6f}  Match: {torch.allclose(ours, ref)}")
```

### 5. Binary Cross-Entropy Manually

BCE from raw logits with numerical stability.

```python
import torch

def binary_cross_entropy(logits, targets):
    """Numerically stable BCE from logits.
    Args:
        logits: (N,) raw scores
        targets: (N,) labels in {0, 1}
    Returns:
        scalar mean loss
    """
    # Stable form: max(logit, 0) - logit*y + log(1 + exp(-|logit|))
    loss = (
        torch.clamp(logits, min=0)
        - logits * targets
        + torch.log(1.0 + torch.exp(-torch.abs(logits)))
    )
    return loss.mean()

# --- verify ---
if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(10)
    targets = torch.randint(0, 2, (10,)).float()
    ours = binary_cross_entropy(logits, targets)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    print(f"Ours: {ours:.6f}  Reference: {ref:.6f}  Match: {torch.allclose(ours, ref)}")
```

### 6. Mean Squared Error Loss Manually

```python
import torch

def mse_loss(predictions, targets):
    """
    Args:
        predictions: (N,) or (N, D)
        targets: same shape as predictions
    Returns:
        scalar mean squared error
    """
    return ((predictions - targets) ** 2).mean()

# --- verify ---
if __name__ == "__main__":
    torch.manual_seed(0)
    pred = torch.randn(16, 3)
    targ = torch.randn(16, 3)
    ours = mse_loss(pred, targ)
    ref = torch.nn.functional.mse_loss(pred, targ)
    print(f"Ours: {ours:.6f}  Reference: {ref:.6f}  Match: {torch.allclose(ours, ref)}")
```

### 7. L2 Regularization in the Loss

Add weight decay penalty to any base loss.

```python
import torch
import torch.nn as nn

def loss_with_l2(model, base_loss_val, weight_decay=1e-4):
    """Add L2 regularization to a computed loss.
    Args:
        model: nn.Module whose parameters to penalize
        base_loss_val: scalar tensor (e.g., cross-entropy)
        weight_decay: lambda coefficient
    Returns:
        total loss = base_loss + (lambda/2) * sum(||w||^2)
    """
    l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
    return base_loss_val + (weight_decay / 2.0) * l2_penalty

# --- standalone version (no nn.Module) ---
def l2_loss_manual(y_pred, y_true, weights, weight_decay=1e-4):
    """MSE + L2 from raw tensors."""
    mse = ((y_pred - y_true) ** 2).mean()
    l2 = sum(w.pow(2).sum() for w in weights)
    return mse + (weight_decay / 2.0) * l2

# --- demo ---
if __name__ == "__main__":
    model = nn.Linear(10, 1)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    base = nn.functional.mse_loss(model(x), y)
    total = loss_with_l2(model, base, weight_decay=0.01)
    print(f"Base loss: {base:.4f}  With L2: {total:.4f}")
```

### 8. SGD Optimizer from Scratch

```python
import torch

class ManualSGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= self.lr * p.grad

# --- demo ---
if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.randn(3, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = ManualSGD([w, b], lr=0.1)

    X = torch.randn(100, 3)
    y = X @ torch.tensor([1., -1., 2.]) + 0.5

    for epoch in range(200):
        opt.zero_grad()
        loss = ((X @ w + b - y) ** 2).mean()
        loss.backward()
        opt.step()

    print(f"w={w.data.tolist()}, b={b.item():.4f}")
```

### 9. Momentum SGD from Scratch

```python
import torch

class MomentumSGD:
    """SGD with momentum."""

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        # Velocity buffers (initialized to zero)
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        with torch.no_grad():
            for p, v in zip(self.params, self.velocities):
                if p.grad is None:
                    continue
                # v_t = momentum * v_{t-1} + grad
                v.mul_(self.momentum).add_(p.grad)
                # w_t = w_{t-1} - lr * v_t
                p -= self.lr * v

# --- demo ---
if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.randn(3, requires_grad=True)
    opt = MomentumSGD([w], lr=0.01, momentum=0.9)
    for _ in range(100):
        opt.zero_grad()
        loss = (w ** 2).sum()
        loss.backward()
        opt.step()
    print(f"w converged to ~0: {w.data.tolist()}")
```

### 10. Adam / AdamW Core Update Rule from Scratch

```python
import torch
import math

class ManualAdamW:
    """Adam with decoupled weight decay (AdamW).
    Set weight_decay=0 for vanilla Adam.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]  # 1st moment
        self.v = [torch.zeros_like(p) for p in self.params]  # 2nd moment

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad

                # Decoupled weight decay (AdamW style)
                if self.weight_decay != 0:
                    p.mul_(1 - self.lr * self.weight_decay)

                # Update biased moments
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Parameter update
                p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

# --- demo ---
if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.randn(5, requires_grad=True)
    opt = ManualAdamW([w], lr=0.01, weight_decay=0.0)  # vanilla Adam
    for _ in range(300):
        opt.zero_grad()
        loss = (w ** 2).sum()
        loss.backward()
        opt.step()
    print(f"w converged to ~0: {w.data.tolist()}")
```

### 11. Gradient Clipping

Clip by global norm and by value.

```python
import torch
import torch.nn as nn

def clip_grad_norm_(parameters, max_norm):
    """Clip gradients by global L2 norm (in-place).
    Args:
        parameters: iterable of tensors with .grad
        max_norm: maximum allowed L2 norm
    Returns:
        total_norm before clipping
    """
    params = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(
        sum(p.grad.pow(2).sum() for p in params)
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.mul_(clip_coef)
    return total_norm

def clip_grad_value_(parameters, clip_value):
    """Clip each gradient element to [-clip_value, clip_value]."""
    for p in parameters:
        if p.grad is not None:
            p.grad.clamp_(-clip_value, clip_value)

# --- demo ---
if __name__ == "__main__":
    model = nn.Linear(10, 2)
    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    norm_before = clip_grad_norm_(model.parameters(), max_norm=1.0)
    norm_after = torch.sqrt(
        sum(p.grad.pow(2).sum() for p in model.parameters())
    )
    print(f"Norm before: {norm_before:.4f}  After: {norm_after:.4f}")
```

### 12. Learning-Rate Decay Scheduler

Step decay, exponential decay, and cosine annealing.

```python
import torch
import math

class StepLRScheduler:
    """Multiply LR by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            for pg in self.optimizer.param_groups:
                pg['lr'] *= self.gamma

class CosineAnnealingScheduler:
    """Cosine annealing from initial LR to eta_min."""

    def __init__(self, optimizer, T_max, eta_min=0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.epoch = 0

    def step(self):
        self.epoch += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = self.eta_min + (base_lr - self.eta_min) * (
                1 + math.cos(math.pi * self.epoch / self.T_max)
            ) / 2

# --- demo ---
if __name__ == "__main__":
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingScheduler(opt, T_max=100, eta_min=1e-5)
    for epoch in range(100):
        scheduler.step()
    print(f"Final LR: {opt.param_groups[0]['lr']:.6f}")
```

### 13. A Full PyTorch Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total

# --- demo ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cpu")
    X = torch.randn(500, 10)
    y = torch.randint(0, 3, (500,))
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1:02d}  Loss: {loss:.4f}  Acc: {acc:.2%}")
```

### 14. Validation Loop with torch.no_grad()

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Run validation and return loss + accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total

# Usage inside a training script:
# val_loss, val_acc = evaluate(model, val_loader, criterion, device)
# print(f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2%}")
```

### 15. Early Stopping

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# --- usage in training loop ---
# early_stop = EarlyStopping(patience=5)
# for epoch in range(max_epochs):
#     train_one_epoch(...)
#     val_loss, val_acc = evaluate(...)
#     if early_stop(val_loss):
#         print(f"Early stopping at epoch {epoch}")
#         break
```

### 16. Checkpoint Save/Load for Model + Optimizer

```python
import torch
import torch.nn as nn

def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Load checkpoint. Returns epoch and best_val_loss."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'], ckpt['best_val_loss']

# --- demo ---
if __name__ == "__main__":
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Save
    save_checkpoint("ckpt.pt", model, optimizer, epoch=5, best_val_loss=0.32)

    # Load into fresh model
    model2 = nn.Linear(10, 2)
    opt2 = torch.optim.Adam(model2.parameters())
    ep, best = load_checkpoint("ckpt.pt", model2, opt2)
    print(f"Resumed from epoch {ep}, best val loss {best}")

    # Verify weights match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
    print("Weights match!")
```

### 17. Custom Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    """Custom dataset that loads features and labels from tensors.
    Replace __init__ with file I/O (csv, images, etc.) as needed.
    """

    def __init__(self, features, labels, transform=None):
        """
        Args:
            features: (N, D) tensor or numpy array
            labels: (N,) tensor or numpy array
            transform: optional callable applied to each feature vector
        """
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# --- demo ---
if __name__ == "__main__":
    X = torch.randn(100, 8)
    y = torch.randint(0, 3, (100,))

    dataset = CSVDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)

    for batch_x, batch_y in loader:
        print(f"Batch shapes: X={batch_x.shape}, y={batch_y.shape}")
        break  # just show first batch
```

### 18. Custom collate_fn for variable-length sequences

Given a list of variable-length sequences (e.g., from a Dataset), write a collate function that pads them to the same length and returns a batched tensor along with the original lengths.

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch: list[torch.Tensor], pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        batch: list of 1-D tensors with different lengths
        pad_value: value used for padding
    Returns:
        padded: (B, max_len) padded tensor
        lengths: (B,) original lengths
    """
    lengths = torch.tensor([seq.size(0) for seq in batch])
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_value)
    return padded, lengths

# --- demo ---
seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9])]
padded, lengths = collate_fn(seqs)
print(padded)   # tensor([[1,2,3,0],[4,5,0,0],[6,7,8,9]])
print(lengths)  # tensor([3, 2, 4])
```

### 19. Padding + attention mask creation for sequences

Given a batch of token-id lists of varying length, pad them and produce a binary attention mask (1 = real token, 0 = pad).

```python
import torch

def pad_and_mask(token_lists: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        token_lists: list of B token-id lists, each of different length
        pad_id: id used for padding
    Returns:
        input_ids:      (B, max_len) padded token ids
        attention_mask:  (B, max_len) binary mask, 1 for real tokens
    """
    lengths = [len(t) for t in token_lists]
    max_len = max(lengths)

    input_ids = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(token_lists), max_len, dtype=torch.long)

    for i, (tokens, l) in enumerate(zip(token_lists, lengths)):
        input_ids[i, :l] = torch.tensor(tokens)
        attention_mask[i, :l] = 1

    return input_ids, attention_mask

# --- demo ---
ids, mask = pad_and_mask([[10, 20, 30], [40, 50]])
print(ids)   # tensor([[10,20,30],[40,50,0]])
print(mask)  # tensor([[1,1,1],[1,1,0]])
```

### 20. Batch normalization from scratch

Implement batch normalization for a (B, D) input during training and inference, with learnable gamma/beta.

```python
import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D)
        Returns:
            out: (B, D) normalized tensor
        """
        if self.training:
            mean = x.mean(dim=0)                       # (D,)
            var = x.var(dim=0, unbiased=False)          # (D,)
            # update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# --- demo ---
bn = BatchNorm(4)
x = torch.randn(8, 4)
out = bn(x)
print(out.shape)  # torch.Size([8, 4])
print(out.mean(dim=0).detach())  # ≈ beta (0)
```

### 21. Layer normalization from scratch

Implement layer normalization over the last dimension for an input of shape (B, ..., D).

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D) — normalizes over the last dimension
        Returns:
            out: same shape as x
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# --- demo ---
ln = LayerNorm(16)
x = torch.randn(2, 8, 16)
out = ln(x)
print(out.shape)  # torch.Size([2, 8, 16])
```

### 22. Dropout from scratch

Implement dropout using a random mask, correctly scaling activations during training and passing through during eval.

```python
import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # Bernoulli mask: 1 with prob (1-p), 0 with prob p
        mask = (torch.rand_like(x) > self.p).float()
        # inverted dropout: scale by 1/(1-p) so expectation is unchanged at test time
        return x * mask / (1 - self.p)

# --- demo ---
drop = Dropout(0.3)
x = torch.ones(2, 10)
print(drop(x))        # some values zeroed, rest scaled by 1/0.7
drop.eval()
print(drop(x))        # all ones (identity at eval)
```

### 23. 2-layer MLP in PyTorch

Implement a standard 2-hidden-layer MLP with ReLU activations and optional dropout.

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            logits: (B, output_dim)
        """
        return self.net(x)

# --- demo ---
model = MLP(784, 256, 10, dropout=0.1)
x = torch.randn(32, 784)
print(model(x).shape)  # torch.Size([32, 10])
```

### 24. Manual backprop for a 2-layer MLP

Implement forward and backward passes for a 2-layer MLP (input -> hidden -> output with ReLU and cross-entropy) using only tensor operations — no autograd.

```python
import torch
import torch.nn.functional as F

def manual_mlp_forward_backward(
    X: torch.Tensor,        # (B, D_in)
    Y: torch.Tensor,        # (B,) class labels
    W1: torch.Tensor,       # (D_in, H)
    b1: torch.Tensor,       # (H,)
    W2: torch.Tensor,       # (H, D_out)
    b2: torch.Tensor,       # (D_out,)
) -> tuple[float, dict[str, torch.Tensor]]:
    B = X.size(0)

    # --- forward ---
    z1 = X @ W1 + b1                          # (B, H)
    a1 = z1.clamp(min=0)                       # ReLU
    z2 = a1 @ W2 + b2                         # (B, D_out)

    # softmax + cross-entropy loss
    logits = z2 - z2.max(dim=-1, keepdim=True).values
    exp_logits = logits.exp()
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)  # (B, D_out)
    loss = -torch.log(probs[torch.arange(B), Y] + 1e-9).mean()

    # --- backward ---
    # d_loss/d_z2 for softmax + CE
    dz2 = probs.clone()
    dz2[torch.arange(B), Y] -= 1
    dz2 /= B                                   # (B, D_out)

    dW2 = a1.T @ dz2                           # (H, D_out)
    db2 = dz2.sum(dim=0)                       # (D_out,)

    da1 = dz2 @ W2.T                           # (B, H)
    dz1 = da1 * (z1 > 0).float()              # ReLU backward

    dW1 = X.T @ dz1                            # (D_in, H)
    db1 = dz1.sum(dim=0)                       # (H,)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return loss.item(), grads

# --- demo & verification against autograd ---
torch.manual_seed(0)
B, D_in, H, D_out = 4, 8, 16, 3
X = torch.randn(B, D_in)
Y = torch.randint(0, D_out, (B,))
W1 = torch.randn(D_in, H, requires_grad=True)
b1 = torch.zeros(H, requires_grad=True)
W2 = torch.randn(H, D_out, requires_grad=True)
b2 = torch.zeros(D_out, requires_grad=True)

loss_manual, grads = manual_mlp_forward_backward(X, Y, W1.data, b1.data, W2.data, b2.data)

# autograd reference
loss_auto = F.cross_entropy(X @ W1 + b1, Y)  # shortcut won't match; do full pass
z1 = X @ W1 + b1; a1 = z1.clamp(min=0); loss_auto = F.cross_entropy(a1 @ W2 + b2, Y)
loss_auto.backward()
print(f"Manual loss: {loss_manual:.4f}, Autograd loss: {loss_auto.item():.4f}")
print(f"dW2 close: {torch.allclose(grads['dW2'], W2.grad, atol=1e-5)}")
```

### 25. ReLU, sigmoid, tanh — forward and backward

Implement the three activations and their derivatives using only tensor ops.

```python
import torch

# --- ReLU ---
def relu_forward(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=0)

def relu_backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    return grad_output * (x > 0).float()

# --- Sigmoid ---
def sigmoid_forward(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + (-x).exp())

def sigmoid_backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    s = sigmoid_forward(x)
    return grad_output * s * (1 - s)

# --- Tanh ---
def tanh_forward(x: torch.Tensor) -> torch.Tensor:
    return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

def tanh_backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    t = tanh_forward(x)
    return grad_output * (1 - t ** 2)

# --- verification ---
x = torch.randn(5, requires_grad=True)

for name, fwd, bwd in [("ReLU", relu_forward, relu_backward),
                         ("Sigmoid", sigmoid_forward, sigmoid_backward),
                         ("Tanh", tanh_forward, tanh_backward)]:
    y = fwd(x)
    y.sum().backward()
    manual = bwd(x.data, torch.ones_like(x))
    print(f"{name} grad match: {torch.allclose(x.grad, manual, atol=1e-5)}")
    x.grad = None
```

### 26. Embedding lookup from token ids

Implement an embedding table and look up vectors for a batch of token ids (equivalent to nn.Embedding).

```python
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) long tensor of token indices
        Returns:
            embeddings: (B, T, embed_dim)
        """
        return self.weight[token_ids]

# --- demo ---
emb = Embedding(vocab_size=1000, embed_dim=64)
ids = torch.tensor([[1, 42, 999], [0, 7, 3]])
out = emb(ids)
print(out.shape)  # torch.Size([2, 3, 64])
# Verify it's differentiable
out.sum().backward()
print(emb.weight.grad.shape)  # torch.Size([1000, 64])
```

### 27. Sinusoidal positional encoding

Implement the fixed sinusoidal positional encoding from "Attention Is All You Need."

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                          # (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()       # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                            # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # even dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims

        self.register_buffer('pe', pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) token embeddings
        Returns:
            x + positional encoding, same shape
        """
        return x + self.pe[:, :x.size(1)]

# --- demo ---
pe = SinusoidalPositionalEncoding(d_model=64)
x = torch.randn(2, 10, 64)
print(pe(x).shape)  # torch.Size([2, 10, 64])
```

### 28. Scaled dot-product attention

Implement scaled dot-product attention with optional mask.

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,    # (B, T_q, d_k)
    K: torch.Tensor,    # (B, T_k, d_k)
    V: torch.Tensor,    # (B, T_k, d_v)
    mask: torch.Tensor | None = None,  # broadcastable to (B, T_q, T_k)
) -> torch.Tensor:
    """
    Returns:
        output: (B, T_q, d_v)
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, T_q, T_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)                   # (B, T_q, T_k)
    return weights @ V                                     # (B, T_q, d_v)

# --- demo ---
B, T, d = 2, 5, 16
Q = K = V = torch.randn(B, T, d)
out = scaled_dot_product_attention(Q, K, V)
print(out.shape)  # torch.Size([2, 5, 16])
```

### 29. Multi-head attention

Implement multi-head attention by splitting Q, K, V into heads, applying scaled dot-product attention per head, and concatenating.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            Q, K, V: (B, T, d_model)
            mask: (B, 1, 1, T_k) or (B, 1, T_q, T_k), broadcastable
        Returns:
            output: (B, T_q, d_model)
        """
        B, T_q, _ = Q.shape

        # project and reshape to (B, num_heads, T, d_k)
        q = self.W_q(Q).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # attention scores
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, H, T_q, T_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        # combine heads
        attn = weights @ v                                    # (B, H, T_q, d_k)
        attn = attn.transpose(1, 2).contiguous().view(B, T_q, -1)  # (B, T_q, d_model)
        return self.W_o(attn)

# --- demo ---
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
print(mha(x, x, x).shape)  # torch.Size([2, 10, 64])
```

### 30. Causal masking for decoder attention

Create a causal (lower-triangular) mask that prevents attending to future positions, suitable for autoregressive decoding.

```python
import torch
import torch.nn.functional as F
import math

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Returns:
        mask: (1, 1, T, T) boolean mask — True where attention is ALLOWED.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Scaled dot-product attention with causal mask.
    Args:
        Q, K, V: (B, T, d_k)
    Returns:
        output: (B, T, d_k)
    """
    T = Q.size(1)
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    causal_mask = create_causal_mask(T).to(Q.device)        # (1, 1, T, T)
    scores = scores.masked_fill(~causal_mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return weights @ V

# --- demo ---
x = torch.randn(2, 6, 32)
out = causal_attention(x, x, x)
print(out.shape)           # torch.Size([2, 6, 32])
print(create_causal_mask(4).squeeze())
# tensor([[1, 0, 0, 0],
#         [1, 1, 0, 0],
#         [1, 1, 1, 0],
#         [1, 1, 1, 1]])
```

### 31. Self-attention block with projection matrices

Implement a self-attention block: project input to Q, K, V, compute attention, project output, with residual + layer norm.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)   # fused Q, K, V projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: optional (B, 1, T, T) or broadcastable
        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        residual = x
        x = self.norm(x)   # pre-norm

        # fused QKV
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attn = weights @ v                                        # (B, H, T, d_k)
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)

        return residual + self.out_proj(attn)

# --- demo ---
block = SelfAttentionBlock(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
print(block(x).shape)  # torch.Size([2, 10, 64])
```

### 32. Transformer encoder block

Implement a full transformer encoder block: self-attention + feed-forward, each with residual connections and layer norm (pre-norm style).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # feed-forward
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _self_attn(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.W_qkv(x).view(B, T, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, T, d_k)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))

        out = weights @ v  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: optional attention mask
        Returns:
            (B, T, d_model)
        """
        x = x + self.dropout(self._self_attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# --- demo ---
enc = TransformerEncoderBlock(d_model=64, num_heads=8, d_ff=256)
x = torch.randn(2, 10, 64)
print(enc(x).shape)  # torch.Size([2, 10, 64])
```

### 33. Transformer decoder block

Implement a transformer decoder block: masked self-attention, cross-attention over encoder output, and feed-forward — all with pre-norm residual connections.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # masked self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_out = nn.Linear(d_model, d_model)

        # cross-attention
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_kv = nn.Linear(d_model, 2 * d_model)
        self.cross_out = nn.Linear(d_model, d_model)

        # feed-forward
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _mha(self, q, k, v, mask=None):
        """Generic multi-head attention. q/k/v already projected, shape (B, T, d_model)."""
        B = q.size(0)
        q = q.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        out = self.dropout(F.softmax(scores, dim=-1)) @ v
        return out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)

    def forward(
        self, x: torch.Tensor, enc_out: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_dec, d_model) decoder input
            enc_out: (B, T_enc, d_model) encoder output
            causal_mask: (1, 1, T_dec, T_dec) causal mask
            cross_mask: optional mask for cross-attention
        Returns:
            (B, T_dec, d_model)
        """
        # 1) masked self-attention
        h = self.norm1(x)
        qkv = self.self_qkv(h).chunk(3, dim=-1)
        x = x + self.dropout(self.self_out(self._mha(*qkv, mask=causal_mask)))

        # 2) cross-attention (Q from decoder, K/V from encoder)
        h = self.norm2(x)
        q = self.cross_q(h)
        k, v = self.cross_kv(enc_out).chunk(2, dim=-1)
        x = x + self.dropout(self.cross_out(self._mha(q, k, v, mask=cross_mask)))

        # 3) feed-forward
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x

# --- demo ---
dec = TransformerDecoderBlock(d_model=64, num_heads=8, d_ff=256)
enc_out = torch.randn(2, 20, 64)
dec_in = torch.randn(2, 10, 64)
causal = torch.tril(torch.ones(10, 10)).bool().unsqueeze(0).unsqueeze(0)
print(dec(dec_in, enc_out, causal_mask=causal).shape)  # torch.Size([2, 10, 64])
```

### 34. Top-k sampling from logits

Implement top-k sampling: zero out all logits outside the top-k, re-normalize, and sample.

```python
import torch

def top_k_sample(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Args:
        logits: (B, vocab_size) raw logits from the model
        k: number of top tokens to keep
        temperature: softmax temperature (lower = more greedy)
    Returns:
        sampled_ids: (B,) sampled token indices
    """
    logits = logits / temperature

    # keep only top-k values, set rest to -inf
    top_k_values, _ = logits.topk(k, dim=-1)                  # (B, k)
    threshold = top_k_values[:, -1].unsqueeze(-1)              # (B, 1)
    logits = logits.masked_fill(logits < threshold, float('-inf'))

    probs = torch.softmax(logits, dim=-1)                      # (B, vocab_size)
    sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
    return sampled_ids

# --- demo ---
torch.manual_seed(42)
logits = torch.randn(3, 50000)  # batch of 3, vocab 50k
tokens = top_k_sample(logits, k=50, temperature=0.8)
print(tokens)        # 3 sampled token ids
print(tokens.shape)  # torch.Size([3])

# verify only top-k tokens can be sampled
logits_single = torch.randn(1, 100)
top_vals, top_idx = logits_single.topk(5)
samples = torch.tensor([top_k_sample(logits_single, k=5).item() for _ in range(1000)])
assert all(s in top_idx[0] for s in samples.unique()), "Sampled outside top-k!"
print("All samples within top-k ✓")
```

### 35. Top-p / Nucleus Sampling

Given a logits tensor, implement nucleus sampling: sort probabilities in descending order, compute the cumulative sum, zero out tokens whose cumulative probability exceeds threshold p, re-normalize, and sample.

```python
import torch
import torch.nn.functional as F

def top_p_sample(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """Top-p (nucleus) sampling from logits of shape (batch, vocab)."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)                          # (B, V)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)

    # Mask tokens beyond the nucleus (keep at least one token)
    mask = cumulative - sorted_probs > p                       # shift right by one
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)     # re-normalize

    # Sample from the filtered distribution
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1)
    token_ids = sorted_idx.gather(dim=-1, index=sampled_sorted_idx)      # (B, 1)
    return token_ids.squeeze(-1)                                         # (B,)

# --- demo ---
logits = torch.randn(2, 1000)
print(top_p_sample(logits, p=0.9))
```

### 36. Beam Search Decoding

Implement beam search for an autoregressive model. At each step, expand every beam by the full vocabulary, keep the top-k (beam_width) cumulative log-probability sequences, and return the best completed sequence.

```python
import torch
import torch.nn.functional as F

def beam_search(model_fn, encoder_out, bos_id: int, eos_id: int,
                max_len: int = 50, beam_width: int = 5) -> list[int]:
    """
    model_fn(token_ids: (beam, seq), encoder_out) -> logits (beam, seq, vocab)
    Returns the best decoded sequence as a list of token ids.
    """
    device = encoder_out.device
    # Each beam: (log_prob, token_id_list)
    seqs = torch.full((beam_width, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(beam_width, device=device)           # cumulative log-probs
    scores[1:] = -float('inf')                                # only one live beam initially

    for _ in range(max_len):
        logits = model_fn(seqs, encoder_out)                  # (beam, seq, V)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)   # (beam, V)
        vocab_size = log_probs.size(-1)

        # Expand: score every (beam, token) pair
        candidates = scores.unsqueeze(1) + log_probs          # (beam, V)
        candidates = candidates.view(-1)                      # (beam * V)

        top_scores, top_idx = candidates.topk(beam_width)     # (beam_width,)
        beam_idx = top_idx // vocab_size
        token_idx = top_idx % vocab_size

        # Assemble new beams
        seqs = torch.cat([seqs[beam_idx],
                          token_idx.unsqueeze(1)], dim=1)     # (beam, seq+1)
        scores = top_scores

        # If best beam ended with EOS, stop early
        if seqs[0, -1].item() == eos_id:
            break

    best = seqs[0].tolist()
    if best[0] == bos_id:
        best = best[1:]
    if best and best[-1] == eos_id:
        best = best[:-1]
    return best

# --- demo (mock model) ---
vocab_size = 20
def mock_model(token_ids, enc_out):
    B, S = token_ids.shape
    return torch.randn(B, S, vocab_size)

enc = torch.zeros(1)
print(beam_search(mock_model, enc, bos_id=0, eos_id=1, max_len=10, beam_width=3))
```

### 37. Temperature Scaling on Logits

Apply temperature scaling: divide logits by a temperature parameter before softmax. T < 1 sharpens the distribution (more confident); T > 1 flattens it (more random).

```python
import torch
import torch.nn.functional as F

def temperature_scale(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Scale logits by temperature and return probabilities."""
    assert temperature > 0, "Temperature must be positive"
    return F.softmax(logits / temperature, dim=-1)

# --- demo ---
logits = torch.tensor([[2.0, 1.0, 0.1]])
for t in [0.5, 1.0, 2.0]:
    print(f"T={t}: {temperature_scale(logits, t)}")
```

### 38. Cosine Similarity Between Embeddings

Compute pairwise cosine similarity between two batches of embeddings using dot product over L2-normalized vectors.

```python
import torch
import torch.nn.functional as F

def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity. a: (N, D), b: (M, D) -> (N, M)."""
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    return a_norm @ b_norm.T

# --- demo ---
a = torch.randn(3, 128)
b = torch.randn(5, 128)
sim = cosine_similarity_matrix(a, b)
print(sim.shape)  # (3, 5)
print(sim)
```

### 39. K-Means from Scratch

Implement Lloyd's algorithm: initialize centroids randomly from the data, assign each point to its nearest centroid, recompute centroids as cluster means, and repeat.

```python
import torch

def kmeans(X: torch.Tensor, k: int, max_iters: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    X: (N, D) data points.
    Returns: centroids (k, D), assignments (N,)
    """
    N, D = X.shape
    # Random initialization: pick k data points
    idx = torch.randperm(N)[:k]
    centroids = X[idx].clone()                                 # (k, D)

    for _ in range(max_iters):
        # Assign each point to nearest centroid
        dists = torch.cdist(X, centroids)                      # (N, k)
        assignments = dists.argmin(dim=1)                      # (N,)

        # Recompute centroids
        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = X[mask].mean(dim=0)
            else:
                new_centroids[c] = centroids[c]                # keep old if empty

        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return centroids, assignments

# --- demo ---
X = torch.cat([torch.randn(50, 2) + torch.tensor([5.0, 5.0]),
               torch.randn(50, 2) + torch.tensor([-5.0, -5.0])])
centroids, labels = kmeans(X, k=2)
print("Centroids:", centroids)
print("Label counts:", labels.bincount())
```

### 40. kNN Classifier from Scratch

Classify query points by majority vote among the k nearest training points, using pairwise Euclidean distance.

```python
import torch

def knn_classify(X_train: torch.Tensor, y_train: torch.Tensor,
                 X_query: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    X_train: (N, D), y_train: (N,) int labels, X_query: (M, D).
    Returns predicted labels (M,).
    """
    dists = torch.cdist(X_query, X_train)                     # (M, N)
    _, topk_idx = dists.topk(k, largest=False, dim=1)         # (M, k)
    topk_labels = y_train[topk_idx]                           # (M, k)

    # Majority vote
    num_classes = int(y_train.max().item()) + 1
    votes = torch.zeros(X_query.size(0), num_classes, device=X_query.device)
    votes.scatter_add_(1, topk_labels.long(),
                       torch.ones_like(topk_labels, dtype=votes.dtype))
    return votes.argmax(dim=1)

# --- demo ---
X_train = torch.randn(100, 2)
y_train = (X_train[:, 0] > 0).long()
X_test = torch.randn(10, 2)
preds = knn_classify(X_train, y_train, X_test, k=5)
print("Predictions:", preds)
```

### 41. PCA Using SVD

Center the data, compute the truncated SVD, and project onto the top-k principal components.

```python
import torch

def pca(X: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    X: (N, D). Returns: projected (N, n_components), components (n_components, D).
    """
    mean = X.mean(dim=0)
    X_centered = X - mean

    # Economy SVD: X = U S V^T, columns of V are principal directions
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:n_components]                             # (n_components, D)
    projected = X_centered @ components.T                      # (N, n_components)
    return projected, components

# --- demo ---
X = torch.randn(200, 50)
proj, comps = pca(X, n_components=5)
print("Projected shape:", proj.shape)   # (200, 5)
print("Components shape:", comps.shape) # (5, 50)
```

### 42. Pairwise Euclidean Distance Matrix Efficiently

Compute all-pairs Euclidean distances using the expansion ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T, avoiding explicit O(N^2 D) loops.

```python
import torch

def pairwise_euclidean(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """X: (N, D), Y: (M, D) -> distances (N, M)."""
    XX = (X * X).sum(dim=1, keepdim=True)                     # (N, 1)
    YY = (Y * Y).sum(dim=1, keepdim=True)                     # (M, 1)
    dists_sq = XX + YY.T - 2.0 * X @ Y.T                     # (N, M)
    return dists_sq.clamp(min=0.0).sqrt()                     # clamp for numerical safety

# --- verify against torch.cdist ---
A = torch.randn(100, 64)
B = torch.randn(80, 64)
ours = pairwise_euclidean(A, B)
ref = torch.cdist(A, B)
print("Max error:", (ours - ref).abs().max().item())           # ~1e-6
```

### 43. Masked Loss (Ignore Padded Positions)

Compute cross-entropy loss only over non-padding positions using a boolean mask derived from the pad token id.

```python
import torch
import torch.nn.functional as F

def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         pad_id: int) -> torch.Tensor:
    """
    logits:  (B, T, V) raw scores.
    targets: (B, T)    ground-truth token ids.
    Returns scalar loss averaged over non-pad tokens.
    """
    # Flatten for F.cross_entropy
    B, T, V = logits.shape
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction='none'
    )                                                          # (B*T,)
    loss_per_token = loss_per_token.view(B, T)

    mask = (targets != pad_id).float()                         # 1 where real, 0 where pad
    masked_loss = (loss_per_token * mask).sum() / mask.sum()
    return masked_loss

# --- demo ---
B, T, V, pad_id = 2, 10, 50, 0
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
targets[:, -3:] = pad_id                                       # last 3 positions are padding
print("Masked loss:", masked_cross_entropy(logits, targets, pad_id).item())
```

### 44. Teacher Forcing in a Toy Seq2Seq Loop

Train a minimal encoder-decoder where the decoder receives the ground-truth previous token at each step (teacher forcing) instead of its own prediction.

```python
import torch
import torch.nn as nn

class ToySeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """src, tgt: (B, T) token ids. Returns logits (B, T_tgt, V)."""
        _, enc_hidden = self.encoder(self.embed(src))          # enc_hidden: (1, B, H)
        # Teacher forcing: feed ground-truth tokens shifted right
        dec_input = self.embed(tgt[:, :-1])                    # (B, T-1, H)
        dec_out, _ = self.decoder(dec_input, enc_hidden)       # (B, T-1, H)
        return self.head(dec_out)                               # (B, T-1, V)

# --- training loop ---
V, H, B, T = 20, 32, 4, 8
model = ToySeq2Seq(V, H)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    src = torch.randint(2, V, (B, T))
    tgt = torch.cat([torch.zeros(B, 1, dtype=torch.long),     # BOS = 0
                     src.flip(1)], dim=1)                      # reverse-copy task

    logits = model(src, tgt)                                   # (B, T, V)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, V), tgt[:, 1:].reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step {step}  loss {loss.item():.4f}")
```

### 45. Weighted Sampler for Imbalanced Classes

Build a WeightedRandomSampler that oversamples minority classes so each class is drawn with equal probability per epoch.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def make_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """Create a sampler that balances classes by inverse frequency."""
    class_counts = labels.bincount().float()                   # (C,)
    class_weights = 1.0 / class_counts                         # inverse frequency
    sample_weights = class_weights[labels]                     # per-sample weight

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),                               # draw full epoch
        replacement=True
    )

# --- demo: imbalanced dataset (90/10 split) ---
X = torch.randn(1000, 16)
y = torch.cat([torch.zeros(900, dtype=torch.long),
               torch.ones(100, dtype=torch.long)])

sampler = make_weighted_sampler(y)
loader = DataLoader(TensorDataset(X, y), batch_size=64, sampler=sampler)

# Verify balance
all_labels = torch.cat([batch[1] for batch in loader])
print("Class distribution in epoch:", all_labels.bincount())   # ~50/50
```

### 46. Label Smoothing

Replace hard one-hot targets with smoothed targets: (1 - alpha) on the correct class and alpha / (C - 1) on the rest.

```python
import torch
import torch.nn.functional as F

def label_smoothing_loss(logits: torch.Tensor, targets: torch.Tensor,
                         alpha: float = 0.1) -> torch.Tensor:
    """
    logits: (B, C), targets: (B,) class indices.
    Returns scalar smoothed cross-entropy loss.
    """
    C = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)                  # (B, C)

    # Build smooth target distribution
    smooth = torch.full_like(log_probs, alpha / (C - 1))
    smooth.scatter_(1, targets.unsqueeze(1), 1.0 - alpha)

    loss = -(smooth * log_probs).sum(dim=-1).mean()
    return loss

# --- demo ---
logits = torch.randn(8, 10)
targets = torch.randint(0, 10, (8,))
print("Smoothed loss:", label_smoothing_loss(logits, targets, alpha=0.1).item())
print("Hard loss:    ", F.cross_entropy(logits, targets).item())
```

### 47. Gradient Accumulation Over Micro-Batches

Simulate a large effective batch by accumulating gradients over several smaller forward/backward passes before stepping the optimizer.

```python
import torch
import torch.nn as nn

model = nn.Linear(64, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

ACCUM_STEPS = 4                                                # effective batch = 4 * micro_batch

# Generate micro-batches
micro_batches = [(torch.randn(8, 64), torch.randint(0, 10, (8,)))
                 for _ in range(ACCUM_STEPS)]

optimizer.zero_grad()
total_loss = 0.0

for i, (x, y) in enumerate(micro_batches):
    logits = model(x)
    loss = loss_fn(logits, y) / ACCUM_STEPS                    # normalize by accum steps
    loss.backward()                                            # gradients accumulate
    total_loss += loss.item()

optimizer.step()                                               # single update
optimizer.zero_grad()

print(f"Accumulated loss: {total_loss:.4f}")
print(f"Effective batch size: {8 * ACCUM_STEPS}")
```

### 48. Mixed Precision Training with torch.autocast + GradScaler

Use automatic mixed precision (AMP) to run forward passes in float16 while keeping the master weights in float32 for stability.

```python
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(256, 512), nn.ReLU(),
    nn.Linear(512, 10)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()                                         # loss scaling for fp16

for step in range(100):
    x = torch.randn(32, 256, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    with autocast(device_type=device.type, dtype=torch.float16):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

    optimizer.zero_grad()
    scaler.scale(loss).backward()                             # scaled backward
    scaler.step(optimizer)                                    # unscale + step
    scaler.update()                                           # adjust scale factor

    if step % 25 == 0:
        print(f"step {step}  loss {loss.item():.4f}")
```

### 49. Custom Autograd Function in PyTorch

Define a custom autograd Function with explicit forward and backward passes. Example: a "clipped ReLU" that clamps gradients in the backward pass (straight-through estimator style).

```python
import torch
from torch.autograd import Function

class ClippedReLU(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, max_val: float = 6.0):
        ctx.save_for_backward(x)
        ctx.max_val = max_val
        return x.clamp(min=0.0, max=max_val)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        # Gradient is 1 where 0 < x < max_val, else 0
        mask = (x > 0) & (x < ctx.max_val)
        return grad_output * mask.float(), None                # None for max_val

# Convenience wrapper
def clipped_relu(x, max_val=6.0):
    return ClippedReLU.apply(x, max_val)

# --- demo ---
x = torch.randn(5, requires_grad=True)
y = clipped_relu(x)
y.sum().backward()
print("Input: ", x.data)
print("Output:", y.data)
print("Grad:  ", x.grad)
```

### 50. Manual Parameter Update Loop Without torch.optim

Implement SGD with momentum manually: compute gradients via autograd, then update parameters in a `torch.no_grad()` block, and zero gradients afterward.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10)
)
loss_fn = nn.CrossEntropyLoss()

lr = 0.01
momentum = 0.9

# Velocity buffers (one per parameter)
velocities = [torch.zeros_like(p) for p in model.parameters()]

for step in range(200):
    x = torch.randn(16, 32)
    y = torch.randint(0, 10, (16,))

    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    # Manual SGD + momentum update
    with torch.no_grad():
        for param, vel in zip(model.parameters(), velocities):
            vel.mul_(momentum).add_(param.grad)                # v = m*v + grad
            param.sub_(lr * vel)                               # w = w - lr * v
            param.grad.zero_()                                 # clear grads

    if step % 50 == 0:
        print(f"step {step}  loss {loss.item():.4f}")
```

---

## Highest-Yield Shortlist

If you want the most interview-relevant 15, do these first:

1. IoU (Q1)
2. Vectorized IoU (Q2)
3. NMS (Q3)
4. Resize/crop/flip with bbox remapping (Q16-18)
5. Dice loss / Focal loss (Q12, Q14)
6. Detection collate function (Q34)
7. Logistic regression from scratch (ML Q2)
8. SGD / AdamW from scratch (ML Q8-10)
9. PyTorch training loop (ML Q13)
10. Custom Dataset + collate_fn (ML Q17-18)
11. BatchNorm / LayerNorm (ML Q20-21)
12. Positional encoding (ML Q27)
13. Scaled dot-product attention (ML Q28)
14. Multi-head attention (ML Q29)
15. Gradient clipping / masked loss (ML Q11, Q43)
