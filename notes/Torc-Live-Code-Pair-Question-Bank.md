# Torc Robotics - Live Code Pair Question Bank (With Answers)

Tailored for **Computer Vision Engineer — 3D Object Detection** at Torc Robotics.

---

# Python Questions (22)

## High-Priority

### 1) Merge overlapping time windows from multiple sensors

**Theory:** Sort intervals by start time, then greedily extend or close the current window. O(n log n) from the sort.

```python
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    counts = [1]  # track how many originals merged
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:  # <= handles touching intervals
            merged[-1][1] = max(merged[-1][1], e)
            counts[-1] += 1
        else:
            merged.append([s, e])
            counts.append(1)
    return merged, counts

# Test
intervals = [(1,5),(2,4),(7,10),(9,12)]
print(merge_intervals(intervals))
# ([[1,5],[7,12]], [2, 2])
```

**Edge cases:** empty list, single interval, all overlapping, touching intervals `(1,5),(5,8)`.

---

### 2) Compute 2D IoU between two axis-aligned boxes

**Theory:** IoU = Intersection Area / Union Area. Union = A + B − Intersection. Clamp overlap dims to 0.

```python
def iou(a, b):
    # boxes: [x1, y1, x2, y2]
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# N×M IoU matrix
def iou_matrix(boxes_a, boxes_b):
    return [[iou(a, b) for b in boxes_b] for a in boxes_a]

# Test
print(iou([0,0,2,2], [1,1,3,3]))  # 1/7 ≈ 0.143
```

**Follow-up — invalid boxes:** Check `x2 < x1` or `y2 < y1` → return 0 or raise.

---

### 3) Implement Non-Maximum Suppression (NMS) in pure Python

**Theory:** Greedy: pick highest-score box, suppress all boxes with IoU > threshold, repeat. O(n²) worst case.

```python
def nms(boxes, scores, iou_thresh=0.5):
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        best = idxs.pop(0)
        keep.append(best)
        idxs = [i for i in idxs if iou(boxes[best], boxes[i]) < iou_thresh]
    return keep

# Class-wise NMS
def class_nms(boxes, scores, classes, iou_thresh=0.5):
    keeps = []
    for cls in set(classes):
        cls_idxs = [i for i, c in enumerate(classes) if c == cls]
        cls_boxes = [boxes[i] for i in cls_idxs]
        cls_scores = [scores[i] for i in cls_idxs]
        local_keep = nms(cls_boxes, cls_scores, iou_thresh)
        keeps.extend(cls_idxs[k] for k in local_keep)
    return keeps
```

**Tie handling:** Stable sort preserves original order when scores are equal.

---

### 4) Greedy tracking association across two frames

**Theory:** For each detection pair across frames, compute IoU. Greedily assign highest-IoU pairs above threshold. O(N·M) for N×M detections.

```python
def greedy_associate(dets_t, dets_t1, iou_thresh=0.3):
    # dets: list of [x1,y1,x2,y2]
    pairs = []
    for i, a in enumerate(dets_t):
        for j, b in enumerate(dets_t1):
            score = iou(a, b)
            if score >= iou_thresh:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)

    matched, used_t, used_t1 = [], set(), set()
    for score, i, j in pairs:
        if i not in used_t and j not in used_t1:
            matched.append((i, j, score))
            used_t.add(i)
            used_t1.add(j)

    unmatched_t = [i for i in range(len(dets_t)) if i not in used_t]
    unmatched_t1 = [j for j in range(len(dets_t1)) if j not in used_t1]
    return matched, unmatched_t, unmatched_t1
```

**Greedy vs Hungarian:** Greedy is faster O(NM log NM) but suboptimal. Hungarian gives global optimum in O(n³) — use `scipy.optimize.linear_sum_assignment`.

---

### 5) Sensor packet synchronization within tolerance

**Theory:** Two-pointer on sorted streams. Advance the pointer whose timestamp is behind. Match when |diff| ≤ delta. O(n+m).

```python
def sync_sensors(cam_ts, lidar_ts, delta):
    i, j = 0, 0
    pairs, unmatched_cam, unmatched_lidar = [], [], []
    used_lidar = set()
    while i < len(cam_ts) and j < len(lidar_ts):
        diff = cam_ts[i] - lidar_ts[j]
        if abs(diff) <= delta:
            pairs.append((cam_ts[i], lidar_ts[j]))
            used_lidar.add(j)
            i += 1
            j += 1
        elif diff < 0:
            unmatched_cam.append(cam_ts[i])
            i += 1
        else:
            if j not in used_lidar:
                unmatched_lidar.append(lidar_ts[j])
            j += 1
    unmatched_cam.extend(cam_ts[i:])
    unmatched_lidar.extend(lidar_ts[k] for k in range(j, len(lidar_ts)) if k not in used_lidar)
    return pairs, unmatched_cam, unmatched_lidar

# Test
print(sync_sensors([1.0,2.0,5.0], [1.1,2.3,4.9], delta=0.2))
```

**Three sensors:** Sync first two, then sync result with third, or use a priority-queue-based approach.

---

### 6) Sliding-window label smoothing

**Theory:** For each index, take the majority class in a window of size k centered at that index. Use `collections.Counter`. O(n·k).

```python
from collections import Counter

def smooth_labels(preds, k):
    n = len(preds)
    half = k // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        cnt = Counter(preds[lo:hi])
        result.append(cnt.most_common(1)[0][0])
    return result

# Weighted version — weight by recency
def smooth_labels_weighted(preds, confs, k, conf_thresh=0.3):
    n = len(preds)
    half = k // 2
    result = []
    for i in range(n):
        lo, hi = max(0, i-half), min(n, i+half+1)
        votes = {}
        for j in range(lo, hi):
            if confs[j] >= conf_thresh:
                weight = 1.0 + (1.0 - abs(j-i)/half)  # recency weight
                votes[preds[j]] = votes.get(preds[j], 0) + weight
        result.append(max(votes, key=votes.get) if votes else preds[i])
    return result
```

---

### 7) Balanced mini-batch sampler

**Theory:** Group samples by (class, slice), cycle through groups round-robin. Oversample rare groups by repeating.

```python
from collections import defaultdict
import random

def balanced_sampler(samples, batch_size, seed=42):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for idx, (cls, slc) in enumerate(samples):
        buckets[(cls, slc)].append(idx)
    # shuffle each bucket
    for k in buckets:
        rng.shuffle(buckets[k])

    # round-robin over buckets
    keys = sorted(buckets.keys())
    pointers = {k: 0 for k in keys}
    batch = []
    while len(batch) < batch_size:
        for k in keys:
            if len(batch) >= batch_size:
                break
            p = pointers[k] % len(buckets[k])  # wrap = oversampling
            batch.append(buckets[k][p])
            pointers[k] = p + 1
    return batch

# Test
samples = [('car','night'),('car','rain'),('ped','night'),('ped','night'),('truck','rain')]
print(balanced_sampler(samples, 4))
```

---

### 8) Label audit summary — precision / recall per class

**Theory:** Precision = TP/(TP+FP), Recall = TP/(TP+FN). Build confusion counts from y_true vs y_pred.

```python
from collections import defaultdict

def audit_metrics(y_true, y_pred, confidences=None):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    classes = set(y_true) | set(y_pred)
    metrics = {}
    for c in sorted(classes):
        prec = tp[c] / (tp[c]+fp[c]) if (tp[c]+fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c]+fn[c]) if (tp[c]+fn[c]) > 0 else 0.0
        metrics[c] = {'precision': round(prec,3), 'recall': round(rec,3),
                       'tp': tp[c], 'fp': fp[c], 'fn': fn[c]}
    # Worst class by false negatives
    worst = max(classes, key=lambda c: fn[c])
    return metrics, worst

# Test
y_t = ['car','car','ped','ped','truck']
y_p = ['car','ped','ped','truck','truck']
print(audit_metrics(y_t, y_p))
```

---

## Medium-Priority

### 9) Top-K hardest samples by disagreement

**Theory:** Disagreement = class mismatch, confidence gap, or IoU gap between two teacher models. Sort descending, return top k.

```python
def topk_hardest(teacher1, teacher2, k, mode='conf_gap'):
    # Each teacher output: list of {'class': str, 'conf': float, 'box': [...]}
    disagreements = []
    for i, (t1, t2) in enumerate(zip(teacher1, teacher2)):
        if mode == 'class_mismatch':
            score = 0 if t1['class'] == t2['class'] else 1
        elif mode == 'conf_gap':
            score = abs(t1['conf'] - t2['conf'])
        elif mode == 'iou_gap':
            score = 1.0 - iou(t1['box'], t2['box'])
        disagreements.append((score, i))
    disagreements.sort(reverse=True)
    return [idx for _, idx in disagreements[:k]]
```

---

### 10) Deduplicate labels by track ID and timestamp

**Theory:** Group by `(track_id, timestamp)`, keep the one with highest score (tie-break by sensor count).

```python
def dedup_labels(records):
    # record: {'track_id', 'timestamp', 'score', 'n_sensors', 'geometry'}
    groups = {}
    discarded = []
    for r in records:
        key = (r['track_id'], r['timestamp'])
        if key not in groups:
            groups[key] = r
        else:
            existing = groups[key]
            if (r['score'] > existing['score'] or
                (r['score'] == existing['score'] and r.get('n_sensors',0) > existing.get('n_sensors',0))):
                discarded.append(existing)
                groups[key] = r
            else:
                discarded.append(r)
    return list(groups.values()), discarded
```

---

### 11) Majority vote with weighted teachers

**Theory:** Sum weights per class across teachers. Abstain if margin between top two < threshold.

```python
def weighted_vote(predictions, weights, abstain_margin=0.1):
    # predictions: list of class labels per teacher, weights: list of floats
    votes = {}
    for pred, w in zip(predictions, weights):
        if pred is None:
            continue  # handle missing predictions
        votes[pred] = votes.get(pred, 0) + w
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda x: -x[1])
    if len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) < abstain_margin:
        return None  # abstain
    return ranked[0][0]
```

---

### 12) Compute Expected Calibration Error (ECE)

**Theory:** Partition predictions into confidence bins. ECE = Σ (|bin| / N) · |acc(bin) − conf(bin)|. Measures how well predicted confidence matches actual accuracy.

```python
def compute_ece(confidences, correct, n_bins=10):
    bin_bounds = [(i/n_bins, (i+1)/n_bins) for i in range(n_bins)]
    ece = 0.0
    n = len(confidences)
    per_bin = []
    for lo, hi in bin_bounds:
        mask = [(lo <= c < hi) if hi < 1.0 else (lo <= c <= hi)
                for c in confidences]
        bin_confs = [c for c, m in zip(confidences, mask) if m]
        bin_accs  = [a for a, m in zip(correct, mask) if m]
        if not bin_confs:
            per_bin.append((0, 0, 0))
            continue
        avg_conf = sum(bin_confs) / len(bin_confs)
        avg_acc  = sum(bin_accs) / len(bin_accs)
        ece += (len(bin_confs) / n) * abs(avg_acc - avg_conf)
        per_bin.append((avg_acc, avg_conf, len(bin_confs)))
    return ece, per_bin

# Why it matters: In auto-labeling, ECE tells you if a model's 0.9 confidence
# truly means 90% accuracy — critical for setting pseudo-label acceptance thresholds.
```

---

### 13) Run-length encode class spans

**Theory:** Iterate once, track current class. On change, emit span. O(n).

```python
def rle_spans(seq, confs=None):
    if not seq:
        return []
    spans = []
    start, cur = 0, seq[0]
    for i in range(1, len(seq)):
        if seq[i] != cur:
            mean_conf = (sum(confs[start:i])/len(confs[start:i])) if confs else None
            spans.append({'class': cur, 'start': start, 'end': i-1, 'conf': mean_conf})
            start, cur = i, seq[i]
    mean_conf = (sum(confs[start:])/len(confs[start:])) if confs else None
    spans.append({'class': cur, 'start': start, 'end': len(seq)-1, 'conf': mean_conf})
    return spans

# Merge short noisy spans (< min_len) into neighbors
def merge_short_spans(spans, min_len=2):
    result = [spans[0]]
    for s in spans[1:]:
        span_len = s['end'] - s['start'] + 1
        if span_len < min_len:
            result[-1]['end'] = s['end']  # absorb into previous
        else:
            result.append(s)
    return result
```

---

### 14) Find contiguous invalid calibration ranges

**Theory:** Track state transitions: when we enter `False`, record start. When we exit, record end.

```python
def find_invalid_ranges(healthy, min_len=1):
    ranges = []
    start = None
    for i, h in enumerate(healthy):
        if not h and start is None:
            start = i
        elif h and start is not None:
            if (i - start) >= min_len:
                ranges.append((start, i - 1))
            start = None
    if start is not None and (len(healthy) - start) >= min_len:
        ranges.append((start, len(healthy) - 1))

    longest = max(ranges, key=lambda r: r[1]-r[0]) if ranges else None
    return ranges, longest
```

---

### 15) Build a mini event logger

**Theory:** Store events in a list. Use bisect for efficient time-range queries. Bounded memory via deque.

```python
from collections import deque
import bisect

class EventLogger:
    def __init__(self, max_events=10000):
        self.events = deque(maxlen=max_events)
        self.timestamps = deque(maxlen=max_events)  # for bisect

    def append(self, timestamp, level, message):
        self.events.append((timestamp, level, message))
        self.timestamps.append(timestamp)

    def query_time_range(self, t_start, t_end):
        lo = bisect.bisect_left(list(self.timestamps), t_start)
        hi = bisect.bisect_right(list(self.timestamps), t_end)
        return [self.events[i] for i in range(lo, hi)]

    def query_level(self, level):
        return [(t, l, m) for t, l, m in self.events if l == level]

    def latest_k(self, k):
        return list(self.events)[-k:]
```

**Thread-safe:** Wrap mutations with `threading.Lock()`.

---

### 16) Nearest map element lookup

**Theory:** Brute force O(N·M). Optimize with KD-tree for O(N log M).

```python
import math

def nearest_lane(objects, lane_centers, max_dist=float('inf')):
    # objects, lane_centers: lists of (x, y)
    assignments = []
    for ox, oy in objects:
        best_d, best_idx = float('inf'), -1
        for j, (lx, ly) in enumerate(lane_centers):
            d = math.hypot(ox-lx, oy-ly)
            if d < best_d:
                best_d, best_idx = d, j
        if best_d <= max_dist:
            assignments.append((best_idx, best_d))
        else:
            assignments.append((None, best_d))
    return assignments

# Optimized: from scipy.spatial import cKDTree
# tree = cKDTree(lane_centers)
# dists, idxs = tree.query(objects)  → O(N log M)
```

---

### 17) Streaming top-K by confidence

**Theory:** Min-heap of size k. Push new items; if heap size > k, pop smallest. O(n log k).

```python
import heapq

def streaming_topk(stream, k):
    heap = []  # min-heap of (conf, detection)
    for det in stream:
        if len(heap) < k:
            heapq.heappush(heap, (det['conf'], det))
        elif det['conf'] > heap[0][0]:
            heapq.heapreplace(heap, (det['conf'], det))
    return sorted(heap, key=lambda x: -x[0])

# Per-class: use dict of heaps
def streaming_topk_per_class(stream, k):
    heaps = {}
    for det in stream:
        cls = det['class']
        if cls not in heaps:
            heaps[cls] = []
        h = heaps[cls]
        if len(h) < k:
            heapq.heappush(h, (det['conf'], det))
        elif det['conf'] > h[0][0]:
            heapq.heapreplace(h, (det['conf'], det))
    return {c: sorted(h, key=lambda x: -x[0]) for c, h in heaps.items()}
```

---

### 18) Parse and aggregate annotation review actions

```python
from collections import defaultdict

def aggregate_reviews(logs):
    # log: {'action': str, 'annotator': str, 'class': str}
    total = defaultdict(int)
    per_annotator = defaultdict(lambda: defaultdict(int))
    per_class = defaultdict(lambda: defaultdict(int))

    for log in logs:
        a, ann, cls = log['action'], log['annotator'], log['class']
        total[a] += 1
        per_annotator[ann][a] += 1
        per_class[cls][a] += 1

    # Edit rate per class
    edit_rates = {}
    for cls, counts in per_class.items():
        t = sum(counts.values())
        edit_rates[cls] = counts.get('edit', 0) / t if t > 0 else 0

    # Flag outlier annotators by reject rate
    reject_rates = {}
    for ann, counts in per_annotator.items():
        t = sum(counts.values())
        reject_rates[ann] = counts.get('reject', 0) / t if t > 0 else 0
    mean_rr = sum(reject_rates.values()) / len(reject_rates) if reject_rates else 0
    outliers = [ann for ann, rr in reject_rates.items() if rr > mean_rr * 1.5]

    return dict(total), dict(per_annotator), edit_rates, outliers
```

---

## Stretch

### 19) Implement a small LRU cache

**Theory:** Hash map for O(1) lookup + doubly-linked list for O(1) eviction. Python shortcut: `OrderedDict`.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)  # evict LRU

# Both get and put are O(1).
# Expiration: store (value, expiry_ts), check on get.
```

---

### 20) Compare two label versions and produce a diff

**Theory:** Use object ID as key. Set difference for added/removed. Compare fields for changed.

```python
def label_diff(old_labels, new_labels, id_key='track_id'):
    old_map = {l[id_key]: l for l in old_labels}
    new_map = {l[id_key]: l for l in new_labels}

    added   = [new_map[k] for k in set(new_map) - set(old_map)]
    removed = [old_map[k] for k in set(old_map) - set(new_map)]

    changed = []
    for k in set(old_map) & set(new_map):
        diffs = {}
        for field in ['class', 'score', 'geometry']:
            if old_map[k].get(field) != new_map[k].get(field):
                diffs[field] = {'old': old_map[k].get(field), 'new': new_map[k].get(field)}
        if diffs:
            changed.append({'id': k, 'changes': diffs})

    return {'added': added, 'removed': removed, 'changed': changed}
```

---

### 21) Implement simple DBSCAN-style clustering from scratch

**Theory:** For each unvisited point, find neighbors within `eps`. If ≥ `min_pts`, expand cluster via BFS. Otherwise mark as noise. O(n²) brute-force.

```python
import math
from collections import deque

def dbscan(points, eps, min_pts):
    n = len(points)
    labels = [-1] * n  # -1 = noise
    visited = [False] * n
    cluster_id = 0

    def neighbors(i):
        return [j for j in range(n)
                if math.hypot(points[i][0]-points[j][0], points[i][1]-points[j][1]) <= eps]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors(i)
        if len(nbrs) < min_pts:
            continue  # noise
        labels[i] = cluster_id
        queue = deque(nbrs)
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                j_nbrs = neighbors(j)
                if len(j_nbrs) >= min_pts:
                    queue.extend(j_nbrs)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels

# Point cloud use: cluster LiDAR returns to separate objects before detection.
# Optimize with KD-tree for neighbor queries → O(n log n).
```

---

### 22) Compute 3D box volume and axis-aligned IoU

**Theory:** Direct extension of 2D IoU to 3 dimensions. Clamp overlap per axis.

```python
def volume_3d(box):
    # box: [x1,y1,z1, x2,y2,z2]
    return ((box[3]-box[0]) * (box[4]-box[1]) * (box[5]-box[2]))

def iou_3d(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1]); iz1 = max(a[2], b[2])
    ix2 = min(a[3], b[3]); iy2 = min(a[4], b[4]); iz2 = min(a[5], b[5])
    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1); id_ = max(0, iz2-iz1)
    inter = iw * ih * id_
    union = volume_3d(a) + volume_3d(b) - inter
    return inter / union if union > 0 else 0.0

# BEV (Bird's Eye View) IoU: project to XY plane, use 2D IoU.
# Useful when height is less important (e.g., vehicle detection).
def iou_bev(a, b):
    return iou([a[0],a[1],a[3],a[4]], [b[0],b[1],b[3],b[4]])

# Oriented 3D boxes: Need Shapely or custom polygon intersection on BEV,
# then multiply by height overlap. Much harder — common in nuScenes/Waymo evals.
```

---

# PyTorch Questions (10)

## High-Priority

### 1) Write a custom `Dataset` for object detection

**Theory:** `__getitem__` returns (image_tensor, target_dict). Boxes are `float32`, labels are `int64`. Variable number of boxes per image — cannot stack naively.

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DetectionDataset(Dataset):
    def __init__(self, image_paths, annotations, transform=None):
        # annotations[i] = {'boxes': [[x1,y1,x2,y2],...], 'labels': [int,...]}
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)  # [C, H, W] float32

        ann = self.annotations[idx]
        target = {
            'boxes':    torch.tensor(ann['boxes'], dtype=torch.float32),   # [N, 4]
            'labels':   torch.tensor(ann['labels'], dtype=torch.int64),    # [N]
            'image_id': torch.tensor([idx]),
        }
        return img, target

# Augmentations: apply SAME spatial transform to both image and boxes
# (e.g., Albumentations with bbox_params).
```

---

### 2) Write a custom `collate_fn` for variable-length targets

**Theory:** Images can be stacked (same size). Targets must stay as a list of dicts since box counts differ.

```python
def detection_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])  # [B, C, H, W]
    targets = [item[1] for item in batch]               # list of dicts
    return images, targets

# If image sizes vary: either resize in Dataset, or pad + return size masks.
# Bug: torch.stack(targets) → crashes because tensors have different shapes.

# Usage:
# loader = DataLoader(dataset, batch_size=4, collate_fn=detection_collate_fn)
```

---

### 3) Vectorized IoU in PyTorch

**Theory:** Broadcast boxes1 `[N,1,4]` vs boxes2 `[1,M,4]` → pairwise ops on `[N,M]`. Clamp for zero overlap.

```python
def vectorized_iou(boxes1, boxes2):
    # boxes1: [N, 4], boxes2: [M, 4] — format [x1, y1, x2, y2]
    b1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    b2 = boxes2.unsqueeze(0)  # [1, M, 4]

    ix1 = torch.max(b1[..., 0], b2[..., 0])  # [N, M]
    iy1 = torch.max(b1[..., 1], b2[..., 1])
    ix2 = torch.min(b1[..., 2], b2[..., 2])
    iy2 = torch.min(b1[..., 3], b2[..., 3])

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)  # [N, M]

    area1 = (b1[..., 2]-b1[..., 0]) * (b1[..., 3]-b1[..., 1])  # [N, 1]
    area2 = (b2[..., 2]-b2[..., 0]) * (b2[..., 3]-b2[..., 1])  # [1, M]

    union = area1 + area2 - inter  # [N, M]
    return inter / union.clamp(min=1e-6)  # [N, M]

# Vectorized: ~100x faster than Python loops for N,M > 100.
```

---

### 4) Implement NMS in PyTorch

**Theory:** Same greedy logic, but use tensor ops for IoU computation. Still iterative outer loop.

```python
def torch_nms(boxes, scores, iou_thresh=0.5):
    # boxes: [N, 4], scores: [N]
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = vectorized_iou(boxes[i:i+1], boxes[remaining])[0]  # [len(remaining)]
        mask = ious < iou_thresh
        order = remaining[mask]
    return torch.tensor(keep, dtype=torch.long)

# Validate: torch.testing.assert_close(my_keep, torchvision.ops.nms(boxes, scores, thresh))
# Class-wise: run per unique class, concat results.
```

---

### 5) Write a minimal training loop

**Theory:** Order matters: zero_grad → forward → loss → backward → step. `model.train()` enables dropout/BN updates; `model.eval()` disables them.

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, targets).item()
    return total_loss / len(loader)

# Full loop:
# for epoch in range(n_epochs):
#     train_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
#     val_loss = validate(model, val_loader, loss_fn, device)
```

---

## Medium-Priority

### 6) Freeze backbone, train head only

**Theory:** Set `requires_grad = False` on backbone params. Only pass head params to optimizer.

```python
# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Only optimize head parameters
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)

# Unfreeze later for fine-tuning:
# for param in model.backbone.parameters():
#     param.requires_grad = True
# optimizer.add_param_group({'params': model.backbone.parameters(), 'lr': 1e-5})

# Pitfall: If frozen params are in optimizer, their gradients are None → wastes memory
# and optimizer state. Always filter: filter(lambda p: p.requires_grad, model.parameters())
```

---

### 7) Implement focal loss for binary classification

**Theory:** FL(p_t) = −α_t (1 − p_t)^γ log(p_t). Down-weights easy examples, focuses on hard ones. When γ=0, reduces to BCE. Key for class imbalance in detection.

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    # logits: [B], targets: [B] ∈ {0, 1}
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)  # prob of true class
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()

# Why focal loss helps: In object detection, ~99% of anchors are background (easy negatives).
# Focal loss reduces their contribution, letting the model focus on hard positives/negatives.
```

---

### 8) Compute masked loss

**Theory:** Flatten predictions and targets, apply boolean mask to select valid positions, compute loss only on those.

```python
def masked_cross_entropy(preds, targets, mask):
    # preds: [B, T, C], targets: [B, T], mask: [B, T] (bool)
    B, T, C = preds.shape

    # Flatten
    preds_flat   = preds.reshape(-1, C)    # [B*T, C]
    targets_flat = targets.reshape(-1)     # [B*T]
    mask_flat    = mask.reshape(-1)        # [B*T]

    if not mask_flat.any():
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    # Select valid positions
    valid_preds   = preds_flat[mask_flat]    # [V, C]
    valid_targets = targets_flat[mask_flat]  # [V]

    return F.cross_entropy(valid_preds, valid_targets)

# Alternative: use ignore_index
# targets[~mask] = -100
# F.cross_entropy(preds.reshape(-1,C), targets.reshape(-1), ignore_index=-100)
```

---

### 9) Debug a device / dtype mismatch

**Theory:** All tensors in an operation must be on the same device. `CrossEntropyLoss` expects targets as `int64` (Long), not float. Mixed precision: use `torch.cuda.amp`.

```python
# BUGGY CODE:
# model = Model().cuda()
# images = torch.randn(4, 3, 224, 224)           # CPU!
# targets = torch.tensor([0,1,1,0], dtype=torch.float32)  # wrong dtype!
# loss = F.cross_entropy(model(images), targets)  # device + dtype error

# FIXED:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
images = torch.randn(4, 3, 224, 224).to(device)           # same device
targets = torch.tensor([0,1,1,0], dtype=torch.int64).to(device)  # Long for CE
loss = F.cross_entropy(model(images), targets)

# Precision notes:
# - float32: default, safe for training
# - float16: faster on GPU, but can overflow → use AMP autocast
# - bfloat16: wider range than fp16, good for training (A100+)
```

---

### 10) Implement a tiny MLP as an `nn.Module`

**Theory:** Define layers in `__init__` so they're registered (params tracked). `forward` defines computation graph.

```python
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):  # x: [B, in_dim]
        return self.net(x)  # [B, out_dim]

# Test
model = TinyMLP(10, 64, 3)
x = torch.randn(8, 10)
print(model(x).shape)  # [8, 3]

# Why __init__ not forward: Layers in __init__ register parameters with the module.
# Creating layers in forward would re-init weights every call and break training.
# BatchNorm: normalizes per-feature across the batch → requires B > 1 at train time.
# Dropout: randomly zeros neurons during training (model.train()), disabled at eval.
```

---

# Best bets for Torc interview (3D Object Detection focus)

1. **Python IoU** (2D → extend to 3D and BEV)
2. **Python NMS** (core detection post-processing)
3. **Greedy tracking** (frame-to-frame association in perception)
4. **Sensor sync** (camera-LiDAR fusion alignment)
5. **Audit metrics / balanced sampler** (training data quality)
6. **PyTorch Dataset** (detection data pipeline)
7. **PyTorch collate_fn** (variable-length batching)
8. **PyTorch vectorized IoU** (broadcasting mastery)
9. **PyTorch training loop** (fundamentals)
10. **PyTorch masked loss** (padded sequence handling)

---

# Interview habits that score points

- Clarify input/output before writing code.
- State assumptions about box format, tensor shapes, and edge cases.
- Start with a correct simple version, then optimize.
- Write one tiny test before trying to be clever.
- In PyTorch, say shapes out loud at each step.
- For detection questions, mention empty boxes, invalid boxes, and class-aware handling.
- For training questions, mention `model.train()`, `model.eval()`, device placement, and dtype.

---

# Fast self-check before the interview

You're ready if you can do all of these without notes:

- Explain IoU and NMS cleanly (2D and 3D)
- Implement a `Dataset` and `collate_fn`
- Write a minimal training loop from memory
- Debug a shape mismatch without panicking
- Use masking and broadcasting comfortably
- Explain time complexity of your Python solutions
- Extend 2D box math to 3D and BEV for autonomous driving context
