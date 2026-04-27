# NumPy ML Interview Guide

Generated: April 27, 2026

Purpose: a practical NumPy guide for ML Engineer interviews. It starts with a cheatsheet, then gives high-frequency NumPy interview questions with answers, code, complexity, and common follow-ups.

Source basis:

- Official NumPy docs for technical behavior: quickstart, broadcasting, indexing, copies/views, `einsum`, `linalg.solve`, random `Generator`, and `sliding_window_view`.
- Your local DeepML/ML coding notes for problem style: covariance, correlation, k-means, convolution, softmax, attention, and ML metric implementations.
- Public interview-prep search as weak signal for question themes: broadcasting/vectorization, stable softmax, pairwise similarity, Conv2D, and ML-from-scratch coding.

## NumPy Cheatsheet

### Interview Mindset

Say shapes out loud.

```text
Input: X shape [N, D], y shape [N]
Output: logits shape [N, C]
Reduce over axis=-1, keepdims=True so broadcasting works.
```

Default pattern:

```text
1. Clarify shape and dtype.
2. Write a correct loop or formula mentally.
3. Convert loops into broadcasting, matrix multiply, indexing, or reductions.
4. Add a tiny sanity test.
5. State time and memory complexity.
```

### Imports and Display

```python
import numpy as np  # import dependency

np.set_printoptions(precision=4, suppress=True)  # format printed arrays
```

### Array Creation

```python
a = np.array([1, 2, 3], dtype=np.float64)  # create typed array
z = np.zeros((3, 4), dtype=np.float32)  # allocate zeros
o = np.ones((2, 3))  # allocate ones
e = np.empty((2, 3))          # uninitialized memory
r = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
l = np.linspace(0, 1, 5)      # 5 evenly spaced values
I = np.eye(3)  # create identity matrix
```

Interview notes:

- Prefer `linspace` over float-step `arange` when you need an exact count.
- Set dtype explicitly when numerical stability matters.
- `empty` is fast but contains arbitrary memory; initialize before use.

### Shape and Metadata

```python
x.shape       # tuple of axis lengths
x.ndim        # number of dimensions
x.size        # total element count
x.dtype       # dtype
x.itemsize    # bytes per element
```

Common ML shapes:

```text
X:      [N, D]          N examples, D features
images: [B, H, W, C]    TensorFlow-style image batch
images: [B, C, H, W]    PyTorch-style image batch
boxes:  [N, 4]          x1, y1, x2, y2
seq:    [B, T, D]       batch, time, feature
```

### Reshape and Axis Operations

```python
x.reshape(2, 3)  # reshape array
x.reshape(-1, 3)        # infer first dimension
x.ravel()               # view when possible
x.flatten()             # copy
x.T                     # reverse axes for 2D
x.transpose(0, 2, 1)  # reorder axes
np.swapaxes(x, 1, 2)  # swap two axes
np.moveaxis(x, -1, 1)   # NHWC -> NCHW
np.expand_dims(x, axis=0)  # insert new axis
x[:, None, :]           # insert axis for broadcasting
np.squeeze(x, axis=0)  # remove size-one axis
```

Pitfall:

- `reshape` works when the memory layout permits or can create a copy.
- Do not assume every reshape is a cheap view.

### Concatenation and Stacking

```python
np.concatenate([a, b], axis=0)  # join along existing axis
np.stack([a, b], axis=0)        # create new axis
np.vstack([a, b])  # stack vertically
np.hstack([a, b])  # stack horizontally
```

Example:

```python
a = np.zeros((2, 3))  # allocate zeros
b = np.ones((2, 3))  # allocate ones
np.concatenate([a, b], axis=0).shape  # (4, 3)
np.stack([a, b], axis=0).shape        # (2, 2, 3)
```

### Indexing

Basic slicing often returns a view:

```python
x[2:5]  # slice or select values
x[:, 1:3]  # slice or select values
x[::-1]  # slice or select values
```

Advanced indexing usually returns a copy:

```python
x[[0, 2, 4]]  # slice or select values
x[x > 0]  # slice or select values
x[rows, cols]  # slice or select values
```

Boolean masks:

```python
mask = scores > 0.5  # build boolean mask
selected = x[mask]  # slice or select values
```

Index helpers:

```python
np.where(mask, a, b)  # select by condition
np.nonzero(mask)  # find true indices
np.argmax(scores, axis=-1)  # find max index
np.argsort(scores)  # sort indices
np.argpartition(scores, kth=3)  # partition top values
np.take_along_axis(values, indices, axis=1)  # gather by indices
```

Interview pitfall:

- `x[mask] = value` modifies `x`.
- `y = x[mask]` is a copy; modifying `y` does not modify `x`.

### Broadcasting Rules

NumPy compares shapes from right to left. Dimensions are compatible when:

```text
same size OR one of them is 1
```

Examples:

```text
[N, D]    + [D]       -> [N, D]
[N, 1, D] - [1, M, D] -> [N, M, D]
[B, T, D] * [1, 1, D] -> [B, T, D]
```

Use `keepdims=True` to preserve axes for broadcasting:

```python
x_centered = x - x.mean(axis=0, keepdims=True)  # compute mean
row_normed = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)  # compute vector norm
```

Memory warning:

- Broadcasting itself can be lazy conceptually, but the result of an operation may allocate the full broadcasted array.
- Pairwise arrays like `[N, M, D]` can be too large.

### Reductions

```python
x.sum()  # sum values
x.sum(axis=0)  # sum values
x.mean(axis=1, keepdims=True)  # compute mean
x.max(axis=-1)  # take maximum
x.argmax(axis=-1)  # find max index
x.cumsum(axis=0)  # cumulative sum
x.prod(axis=-1)  # multiply values
np.all(mask, axis=1)  # check all true
np.any(mask, axis=1)  # check any true
```

Axis rule:

```text
axis=0 collapses rows/down the first dimension.
axis=1 collapses columns/across the second dimension.
axis=-1 collapses the last dimension.
```

### Elementwise Ops and Ufuncs

```python
np.exp(x)  # exponentiate values
np.log(x)  # take natural log
np.sqrt(x)  # take square root
np.maximum(x, 0)  # take maximum
np.minimum(x, 1)  # take minimum
np.clip(x, 0, 1)  # clip value range
np.abs(x)  # take absolute value
np.sign(x)  # get element signs
np.isnan(x)  # detect NaNs
np.isfinite(x)  # detect finite values
```

Use `np.where` for vectorized conditionals:

```python
huber = np.where(np.abs(e) <= delta, 0.5 * e**2, delta * (np.abs(e) - 0.5 * delta))  # select by condition
```

### Matrix and Tensor Operations

```python
A @ B                         # matrix multiply
np.matmul(A, B)  # matrix multiply
np.dot(a, b)                   # dot for vectors, matmul-like for 2D
np.outer(a, b)  # compute outer product
np.einsum("nd,md->nm", X, Y)   # pairwise dot products
np.tensordot(a, b, axes=([1,2], [0,1]))  # contract over axes
```

Common `einsum` patterns:

```python
np.einsum("ij,jk->ik", A, B)       # matrix multiply
np.einsum("nd,md->nm", X, Y)       # pairwise dot
np.einsum("btd,df->btf", X, W)     # batched linear
np.einsum("bhld,bhsd->bhls", Q, K) # attention scores
np.einsum("ii->i", A)              # diagonal
np.einsum("ij->", A)               # sum all
```

Interview advice:

- Use `@` when possible.
- Use `einsum` when it makes dimensions clearer or avoids reshaping.
- Be ready to translate `einsum` into shapes verbally.

### Linear Algebra

```python
np.linalg.norm(x, axis=-1)  # compute vector norm
np.linalg.solve(A, b)       # solve Ax=b
np.linalg.lstsq(A, b, rcond=None)  # solve least squares
np.linalg.inv(A)            # avoid unless you need explicit inverse
np.linalg.det(A)  # compute determinant
np.linalg.cond(A)  # estimate conditioning
np.linalg.eig(A)  # eigendecompose matrix
np.linalg.eigh(S)           # symmetric/Hermitian
np.linalg.svd(X, full_matrices=False)  # compute SVD
```

Rule:

```text
Use solve(A, b), not inv(A) @ b.
```

`solve` is more numerically stable and directly solves the system.

### Random Numbers

Modern NumPy pattern:

```python
rng = np.random.default_rng(42)  # create random generator
x = rng.normal(size=(100, 10))  # sample normal values
idx = rng.choice(100, size=16, replace=False)  # sample indices
perm = rng.permutation(100)  # generate permutation
```

Interview notes:

- Prefer `default_rng` over legacy global random state.
- Pass seed for reproducible tests.

### Sorting and Top-K

```python
order = np.argsort(scores)          # ascending
desc = np.argsort(-scores)          # descending
topk_unsorted = np.argpartition(-scores, k - 1)[:k]  # partition top values
topk_sorted = topk_unsorted[np.argsort(-scores[topk_unsorted])]  # sort indices
```

Complexity:

- Full sort: O(n log n)
- Top-k with `argpartition`: expected O(n), then O(k log k) to sort selected items

### Numerical Stability

Stable softmax:

```python
def softmax(x, axis=-1):  # define softmax
    z = x - np.max(x, axis=axis, keepdims=True)  # take maximum
    ez = np.exp(z)  # exponentiate values
    return ez / np.sum(ez, axis=axis, keepdims=True)  # return result
```

Stable logsumexp:

```python
def logsumexp(x, axis=-1, keepdims=False):  # define logsumexp
    m = np.max(x, axis=axis, keepdims=True)  # take maximum
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))  # sum values
    return out if keepdims else np.squeeze(out, axis=axis)  # return result
```

Stability rules:

- Add epsilon to denominators: `x / (norm + 1e-12)`.
- Use logits directly for cross entropy.
- Avoid `np.log(softmax(x))` if you can compute log-softmax/logsumexp directly.
- Be mindful of integer division and dtype promotion.

### Copies, Views, and Memory

```python
y = x.view()       # view if possible
y = x.copy()       # independent copy
np.shares_memory(x, y)  # check shared storage
x.flags  # inspect memory layout
```

Common memory blowups:

```python
diff = X[:, None, :] - Y[None, :, :]  # shape [N, M, D]
```

If `N*M*D` is huge, compute in chunks.

### Sliding Windows

```python
from numpy.lib.stride_tricks import sliding_window_view  # import specific APIs

windows = sliding_window_view(x, window_shape=3)  # make rolling windows
```

Warning:

- Window views can create huge logical arrays.
- For simple rolling sums, prefix sums may be cheaper.

### High-Yield ML Snippets

Standardize features:

```python
mu = X.mean(axis=0, keepdims=True)  # compute mean
sigma = X.std(axis=0, keepdims=True)  # compute feature std
X_std = (X - mu) / (sigma + 1e-12)  # standardize features
```

One-hot:

```python
Y = np.eye(num_classes)[y]  # create identity matrix
```

Pairwise squared distances:

```python
d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)  # sum values
```

Memory-friendlier pairwise squared distances:

```python
d2 = (X**2).sum(axis=1, keepdims=True) + (Y**2).sum(axis=1)[None, :] - 2 * X @ Y.T  # sum values
d2 = np.maximum(d2, 0.0)  # take maximum
```

Cosine similarity:

```python
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # compute vector norm
Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)  # compute vector norm
sim = Xn @ Yn.T  # matrix multiply
```

## Top Asked NumPy Questions

### 1. Explain Broadcasting

Question:

```text
What happens when X has shape [N, 1, D] and Y has shape [1, M, D]?
```

Answer:

```text
NumPy compares dimensions from right to left.
D matches D, 1 broadcasts to M, and N broadcasts against 1.
The result has shape [N, M, D].
```

Code:

```python
X = np.zeros((5, 1, 3))  # allocate zeros
Y = np.ones((1, 7, 3))  # allocate ones
Z = X - Y  # broadcast pairwise differences
assert Z.shape == (5, 7, 3)  # validate assumption
```

Follow-up:

- This is useful for pairwise differences, but it can allocate a large `[N, M, D]` result.

### 2. Compute Pairwise Cosine Similarity

Question:

```text
Given X [N, D] and Y [M, D], compute cosine similarity [N, M] without Python loops.
```

Answer:

```python
def pairwise_cosine(X, Y, eps=1e-12):  # define pairwise_cosine
    X = np.asarray(X, dtype=np.float64)  # coerce to array
    Y = np.asarray(Y, dtype=np.float64)  # coerce to array
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)  # normalize X rows
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + eps)  # normalize Y rows
    return Xn @ Yn.T  # return result
```

Complexity:

```text
Time: O(NMD)
Memory: O(NM) for output
```

Follow-up:

- For huge `N` or `M`, compute in chunks.

### 3. Stable Softmax

Question:

```text
Implement softmax for a 2D logits matrix along rows.
```

Answer:

```python
def softmax_rows(logits):  # define softmax_rows
    logits = np.asarray(logits, dtype=np.float64)  # coerce to array
    z = logits - logits.max(axis=1, keepdims=True)  # take maximum
    exp_z = np.exp(z)  # exponentiate values
    return exp_z / exp_z.sum(axis=1, keepdims=True)  # return result
```

Why subtract max:

```text
Softmax is invariant to adding/subtracting the same constant from every logit.
Subtracting the max prevents exp overflow.
```

### 4. Cross Entropy From Logits

Question:

```text
Implement mean cross entropy from logits and integer class labels.
```

Answer:

```python
def cross_entropy_from_logits(logits, y):  # define cross_entropy_from_logits
    logits = np.asarray(logits, dtype=np.float64)  # coerce to array
    y = np.asarray(y, dtype=np.int64)  # coerce to array
    m = logits.max(axis=1, keepdims=True)  # take maximum
    logsumexp = np.log(np.exp(logits - m).sum(axis=1)) + m.squeeze(1)  # compute stable logsumexp
    correct = logits[np.arange(logits.shape[0]), y]  # gather true-class logits
    return float(np.mean(logsumexp - correct))  # return result
```

Common mistake:

- Computing `np.log(softmax(logits))` directly can underflow.

### 5. Confusion Matrix, Precision, Recall, F1

Question:

```text
Given y_true and y_pred, compute a confusion matrix and per-class metrics.
```

Answer:

```python
def confusion_matrix(y_true, y_pred, num_classes):  # define confusion_matrix
    y_true = np.asarray(y_true, dtype=np.int64)  # coerce to array
    y_pred = np.asarray(y_pred, dtype=np.int64)  # coerce to array
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # allocate zeros
    np.add.at(cm, (y_true, y_pred), 1)  # accumulate class counts
    return cm  # return result


def precision_recall_f1(cm, eps=1e-12):  # define precision_recall_f1
    tp = np.diag(cm)  # read true positives
    fp = cm.sum(axis=0) - tp  # count false positives
    fn = cm.sum(axis=1) - tp  # count false negatives
    precision = tp / (tp + fp + eps)  # compute precision
    recall = tp / (tp + fn + eps)  # compute recall
    f1 = 2 * precision * recall / (precision + recall + eps)  # compute F1
    return precision, recall, f1  # return result
```

Interview note:

- Row = true class, column = predicted class. State your convention.

### 6. One-Hot Encode Labels

Question:

```text
Convert integer labels [N] into one-hot matrix [N, C].
```

Answer:

```python
def one_hot(y, num_classes):  # define one_hot
    y = np.asarray(y, dtype=np.int64)  # coerce to array
    out = np.zeros((y.size, num_classes), dtype=np.float32)  # allocate zeros
    out[np.arange(y.size), y] = 1.0  # create range values
    return out  # return result
```

Short version:

```python
np.eye(num_classes, dtype=np.float32)[y]  # create identity matrix
```

### 7. Top-K Hardest Examples

Question:

```text
Given losses [N], return indices of top-k largest losses.
```

Answer:

```python
def top_k_indices(losses, k):  # define top_k_indices
    losses = np.asarray(losses)  # coerce to array
    if k <= 0:  # branch on condition
        return np.array([], dtype=np.int64)  # return result
    k = min(k, losses.size)  # inspect element count
    idx = np.argpartition(-losses, k - 1)[:k]  # partition top values
    return idx[np.argsort(-losses[idx])]  # return result
```

Follow-up:

- Use this for hard-example mining, but inspect high-loss examples for label noise.

### 8. Pairwise Euclidean Distance

Question:

```text
Compute squared Euclidean distances between X [N, D] and Y [M, D].
```

Answer:

```python
def pairwise_sq_dist(X, Y):  # define pairwise_sq_dist
    X = np.asarray(X, dtype=np.float64)  # coerce to array
    Y = np.asarray(Y, dtype=np.float64)  # coerce to array
    d2 = (X**2).sum(axis=1, keepdims=True)  # sum values
    d2 = d2 + (Y**2).sum(axis=1)[None, :] - 2 * X @ Y.T  # sum values
    return np.maximum(d2, 0.0)  # return result
```

Why this version:

- Avoids materializing `[N, M, D]`.

### 9. Vectorized IoU

Question:

```text
Given boxes A [N, 4] and B [M, 4], compute pairwise IoU [N, M].
```

Answer:

```python
def pairwise_iou(boxes_a, boxes_b, eps=1e-12):  # define pairwise_iou
    a = np.asarray(boxes_a, dtype=np.float64)  # coerce to array
    b = np.asarray(boxes_b, dtype=np.float64)  # coerce to array
    tl = np.maximum(a[:, None, :2], b[None, :, :2])  # intersection top-left
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])  # intersection bottom-right
    wh = np.maximum(br - tl, 0.0)  # clamp intersection size
    inter = wh[..., 0] * wh[..., 1]  # compute intersection area
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)  # compute box A area
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)  # compute box B area
    union = area_a[:, None] + area_b[None, :] - inter  # compute union area
    return inter / (union + eps)  # return result
```

Clarify:

- Coordinates are continuous `[x1, y1, x2, y2]`.
- If using inclusive pixel boxes, area formula may add 1.

### 10. Non-Maximum Suppression

Question:

```text
Implement NMS from boxes and scores.
```

Answer:

```python
def nms(boxes, scores, iou_thresh):  # define nms
    boxes = np.asarray(boxes, dtype=np.float64)  # coerce to array
    scores = np.asarray(scores, dtype=np.float64)  # coerce to array
    order = np.argsort(-scores)  # sort indices
    keep = []  # track kept indices
    while order.size:  # loop until done
        i = order[0]  # choose highest-score box
        keep.append(i)  # track kept indices
        if order.size == 1:  # branch on condition
            break  # stop loop
        ious = pairwise_iou(boxes[[i]], boxes[order[1:]])[0]  # score overlaps with rest
        order = order[1:][ious <= iou_thresh]  # suppress high-overlap boxes
    return np.array(keep, dtype=np.int64)  # return result
```

Complexity:

- O(N^2) simple implementation.

Follow-up:

- Class-aware NMS applies NMS per class.
- Soft-NMS decays scores instead of hard suppression.

### 11. Waypoint Suffix Distances

Question:

```text
Given ordered 2D waypoints [N, 2], return remaining distance from each waypoint to the end.
```

Answer:

```python
def suffix_distances(points):  # define suffix_distances
    p = np.asarray(points, dtype=np.float64)  # coerce to array
    seg = np.linalg.norm(p[1:] - p[:-1], axis=1)  # compute vector norm
    out = np.zeros(p.shape[0], dtype=np.float64)  # allocate zeros
    out[:-1] = np.cumsum(seg[::-1])[::-1]  # cumulative sum
    return out  # return result
```

Batched version:

```python
def suffix_distances_batched(points):  # define suffix_distances_batched
    p = np.asarray(points, dtype=np.float64)  # coerce to array
    seg = np.linalg.norm(p[:, 1:, :] - p[:, :-1, :], axis=-1)  # compute vector norm
    out = np.zeros(p.shape[:2], dtype=np.float64)  # allocate zeros
    out[:, :-1] = np.cumsum(seg[:, ::-1], axis=1)[:, ::-1]  # cumulative sum
    return out  # return result
```

### 12. Sliding Window Threshold

Question:

```text
Find all starts where readings stay above threshold for k consecutive frames.
```

Answer:

```python
def above_threshold_windows(values, threshold, k):  # define above_threshold_windows
    values = np.asarray(values)  # coerce to array
    if k <= 0 or k > values.size:  # branch on condition
        return []  # return result
    above = values >= threshold  # mark threshold hits
    prefix = np.concatenate([[0], np.cumsum(above)])  # join existing axis
    counts = prefix[k:] - prefix[:-k]  # count hits per window
    return np.where(counts == k)[0].tolist()  # return result
```

Why prefix sums:

- O(N) time, O(N) memory.
- Avoids explicit window loop.

### 13. Conv2D Output Shape and Parameter Count

Question:

```text
Compute Conv2D output shape and parameter count.
```

Answer:

```python
def conv2d_output_shape(h, w, kernel_size, stride=1, padding=0, dilation=1):  # define conv2d_output_shape
    def pair(x):  # define pair
        return x if isinstance(x, tuple) else (x, x)  # normalize to pair

    kh, kw = pair(kernel_size)  # unpack kernel size
    sh, sw = pair(stride)  # unpack stride
    ph, pw = pair(padding)  # unpack padding
    dh, dw = pair(dilation)  # unpack dilation
    out_h = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1  # compute output height
    out_w = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1  # compute output width
    return out_h, out_w  # return result


def conv2d_param_count(c_in, c_out, kernel_size, groups=1, bias=True):  # define conv2d_param_count
    def pair(x):  # define pair
        return x if isinstance(x, tuple) else (x, x)  # normalize to pair

    kh, kw = pair(kernel_size)  # unpack kernel size
    params = c_out * (c_in // groups) * kh * kw  # count filter weights
    return params + (c_out if bias else 0)  # return result
```

### 14. Conv2D Forward Pass

Question:

```text
Implement NCHW Conv2D forward pass using NumPy.
```

Answer:

```python
def conv2d_forward(x, weight, bias=None, stride=1, padding=0):  # define conv2d_forward
    x = np.asarray(x, dtype=np.float64)  # coerce to array
    weight = np.asarray(weight, dtype=np.float64)  # coerce to array

    def pair(v):  # define pair
        return v if isinstance(v, tuple) else (v, v)  # normalize to pair

    sh, sw = pair(stride)  # unpack stride
    ph, pw = pair(padding)  # unpack padding
    n, c_in, h, w = x.shape  # inspect shape
    c_out, c_w, kh, kw = weight.shape  # inspect shape
    assert c_in == c_w  # validate assumption

    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))  # pad array
    out_h = (h + 2 * ph - kh) // sh + 1  # compute output height
    out_w = (w + 2 * pw - kw) // sw + 1  # compute output width
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.float64)  # allocate zeros

    for i in range(out_h):  # iterate over items
        for j in range(out_w):  # iterate over items
            patch = x_pad[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]  # extract input window
            out[:, :, i, j] = np.tensordot(  # compute filter responses
                patch,  # pass input window
                weight,  # pass filter bank
                axes=([1, 2, 3], [1, 2, 3]),  # sum over Cin/Kh/Kw
            )  # finish contraction

    if bias is not None:  # branch on condition
        out += np.asarray(bias, dtype=np.float64)[None, :, None, None]  # coerce to array
    return out  # return result
```

Explain:

- This is cross-correlation, matching deep learning libraries.
- Complexity is O(N * C_out * H_out * W_out * C_in * K_h * K_w).

### 15. MaxPool2D Forward

Question:

```text
Implement max pooling for NCHW input.
```

Answer:

```python
def maxpool2d_forward(x, kernel_size=2, stride=2):  # define maxpool2d_forward
    x = np.asarray(x)  # coerce to array
    n, c, h, w = x.shape  # inspect shape
    def pair(v):  # define pair
        return v if isinstance(v, tuple) else (v, v)  # normalize to pair

    kh, kw = pair(kernel_size)  # unpack kernel size
    sh, sw = pair(stride)  # unpack stride
    out_h = (h - kh) // sh + 1  # compute output height
    out_w = (w - kw) // sw + 1  # compute output width
    out = np.empty((n, c, out_h, out_w), dtype=x.dtype)  # allocate uninitialized array
    for i in range(out_h):  # iterate over items
        for j in range(out_w):  # iterate over items
            patch = x[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]  # extract pooling window
            out[:, :, i, j] = patch.max(axis=(2, 3))  # take maximum
    return out  # return result
```

Follow-up:

- For backward pass, route gradient only to argmax locations, handling ties by a chosen convention.

### 16. Scaled Dot-Product Attention

Question:

```text
Implement attention using NumPy.
```

Answer:

```python
def attention(q, k, v, mask=None):  # define attention
    d = q.shape[-1]  # read key dimension
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)  # compute scaled scores
    if mask is not None:  # branch on condition
        scores = np.where(mask, scores, -np.inf)  # select by condition
    weights = softmax(scores, axis=-1)  # normalize attention weights
    return weights @ v, weights  # return result
```

Shapes:

```text
q: [B, H, Lq, D]
k: [B, H, Lk, D]
v: [B, H, Lk, Dv]
weights: [B, H, Lq, Lk]
out: [B, H, Lq, Dv]
```

### 17. Covariance and Correlation Matrix

Question:

```text
Compute covariance and correlation for data X [N, D].
```

Answer:

```python
def covariance_matrix(X):  # define covariance_matrix
    X = np.asarray(X, dtype=np.float64)  # coerce to array
    Xc = X - X.mean(axis=0, keepdims=True)  # compute mean
    return Xc.T @ Xc / (X.shape[0] - 1)  # return result


def correlation_matrix(X, eps=1e-12):  # define correlation_matrix
    cov = covariance_matrix(X)  # compute covariance
    std = np.sqrt(np.diag(cov))  # take square root
    return cov / (std[:, None] * std[None, :] + eps)  # return result
```

### 18. PCA With SVD

Question:

```text
Implement PCA projection to k dimensions.
```

Answer:

```python
def pca_project(X, k):  # define pca_project
    X = np.asarray(X, dtype=np.float64)  # coerce to array
    mu = X.mean(axis=0, keepdims=True)  # compute mean
    Xc = X - mu  # center features
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)  # compute SVD
    components = vt[:k]  # keep top components
    return Xc @ components.T, components, mu  # return result
```

Why SVD:

- Stable and avoids explicitly forming covariance when not needed.

### 19. K-Means

Question:

```text
Implement k-means clustering in NumPy.
```

Answer:

```python
def kmeans(X, k, num_iters=50, seed=0):  # define kmeans
    X = np.asarray(X, dtype=np.float64)  # coerce to array
    rng = np.random.default_rng(seed)  # create random generator
    centers = X[rng.choice(X.shape[0], size=k, replace=False)].copy()  # inspect shape

    for _ in range(num_iters):  # iterate over items
        d2 = pairwise_sq_dist(X, centers)  # set d2
        labels = d2.argmin(axis=1)  # assign nearest center
        new_centers = centers.copy()  # start updated centers
        for c in range(k):  # iterate over items
            mask = labels == c  # build boolean mask
            if np.any(mask):  # branch on condition
                new_centers[c] = X[mask].mean(axis=0)  # compute mean
        if np.allclose(new_centers, centers):  # branch on condition
            break  # stop loop
        centers = new_centers  # accept new centers
    return labels, centers  # return result
```

Follow-up:

- Empty clusters need handling.
- K-means++ improves initialization.

### 20. Logistic Regression Gradient Step

Question:

```text
Implement one gradient descent step for binary logistic regression.
```

Answer:

```python
def sigmoid(z):  # define sigmoid
    z = np.asarray(z, dtype=np.float64)  # coerce to array
    out = np.empty_like(z)  # allocate uninitialized array
    pos = z >= 0  # identify stable branch
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))  # exponentiate values
    exp_z = np.exp(z[~pos])  # exponentiate values
    out[~pos] = exp_z / (1.0 + exp_z)  # write output slice
    return out  # return result


def logistic_step(X, y, w, b, lr=0.1):  # define logistic_step
    X = np.asarray(X, dtype=np.float64)  # coerce to array
    y = np.asarray(y, dtype=np.float64)  # coerce to array
    z = X @ w + b  # matrix multiply
    p = sigmoid(z)  # predict probabilities
    err = p - y  # compute logit gradient
    dw = X.T @ err / X.shape[0]  # inspect shape
    db = err.mean()  # compute mean
    w = w - lr * dw  # update weights
    b = b - lr * db  # update bias
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))  # compute mean
    return w, b, loss  # return result
```

Explain:

- For sigmoid plus binary cross entropy, derivative with respect to logit is `p - y`.

### 21. Numerical Gradient Check

Question:

```text
How do you verify a hand-coded gradient?
```

Answer:

```python
def numerical_grad(f, x, eps=1e-5):  # define numerical_grad
    x = x.astype(np.float64).copy()  # work on float copy
    grad = np.zeros_like(x)  # allocate zeros
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])  # iterate array indices
    while not it.finished:  # loop until done
        idx = it.multi_index  # get current index
        old = x[idx]  # save original value
        x[idx] = old + eps  # slice or select values
        f_pos = f(x)  # evaluate positive perturbation
        x[idx] = old - eps  # slice or select values
        f_neg = f(x)  # evaluate negative perturbation
        x[idx] = old  # slice or select values
        grad[idx] = (f_pos - f_neg) / (2 * eps)  # store gradient value
        it.iternext()  # advance iterator
    return grad  # return result
```

Follow-up:

- Use central difference.
- Compare relative error, not just absolute error.

### 22. Chunked Pairwise Computation

Question:

```text
Pairwise similarity is too large for memory. What do you do?
```

Answer:

```python
def pairwise_cosine_chunked(X, Y, chunk=1024):  # define pairwise_cosine_chunked
    outs = []  # collect chunk outputs
    for start in range(0, X.shape[0], chunk):  # iterate over items
        outs.append(pairwise_cosine(X[start:start + chunk], Y))  # append chunk result
    return np.concatenate(outs, axis=0)  # return result
```

Explain:

- Same total compute, lower peak memory.

### 23. `np.vectorize` vs Vectorization

Question:

```text
Is np.vectorize a performance optimization?
```

Answer:

```text
No. np.vectorize is mostly a convenience wrapper around a Python function.
Real NumPy vectorization uses ufuncs, broadcasting, reductions, BLAS-backed matmul, or compiled routines.
```

Example:

```python
# Good vectorized code
y = np.maximum(x, 0)  # take maximum

# Usually not a speedup
f = np.vectorize(lambda t: max(t, 0))  # wrap Python function
y = f(x)  # apply wrapper elementwise
```

### 24. Basic vs Advanced Indexing

Question:

```text
What is the difference between x[:, 1:3] and x[:, [1, 2]]?
```

Answer:

```text
x[:, 1:3] uses basic slicing and usually returns a view.
x[:, [1, 2]] uses advanced indexing and returns a copy.
```

Why interviewers care:

- It affects memory, mutation behavior, and performance.

### 25. Batched Gather With `take_along_axis`

Question:

```text
Given scores [B, C], gather the score of each target class y [B].
```

Answer:

```python
def gather_target_scores(scores, y):  # define gather_target_scores
    scores = np.asarray(scores)  # coerce to array
    y = np.asarray(y)  # coerce to array
    return np.take_along_axis(scores, y[:, None], axis=1).squeeze(1)  # return result
```

Equivalent:

```python
scores[np.arange(scores.shape[0]), y]  # create range values
```

## NumPy Interview Red Flags

- Not knowing which axis you are reducing.
- Forgetting `keepdims=True` and breaking broadcasting.
- Using unstable softmax/cross entropy.
- Materializing huge `[N, M, D]` arrays without checking memory.
- Mixing NHWC and NCHW silently.
- Calling `np.linalg.inv(A) @ b` instead of `np.linalg.solve(A, b)`.
- Assuming advanced indexing returns a view.
- Using `np.vectorize` as a performance claim.
- Ignoring dtype and integer division issues.
- Not testing on a tiny shape.

## 3-Day NumPy Drill Plan

Day 1:

- Broadcasting rules.
- Pairwise cosine/distance.
- Softmax and cross entropy.
- Confusion matrix/F1.

Day 2:

- IoU/NMS.
- Conv2D output shape.
- Conv2D forward.
- MaxPool forward.

Day 3:

- Attention.
- Covariance/correlation/PCA.
- K-means.
- Logistic gradient step.
- One full timed mock.

## Source Notes

Official NumPy:

- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy copies and views](https://numpy.org/doc/stable/user/basics.copies.html)
- [NumPy einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
- [NumPy linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)
- [NumPy random Generator](https://numpy.org/doc/stable/reference/random/generator.html)
- [NumPy sliding_window_view](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html)

Local notes used:

- `/Users/muditj/Desktop/Projects/daily-brief/notes/DeepML-Medium.md`
- `/Users/muditj/Desktop/Projects/daily-brief/notes/DeepML-Hard.md`
- `/Users/muditj/Desktop/Projects/daily-brief/notes/Cheatsheet-03-ML-Coding.md`

Interview-theme search signals:

- Public question banks and prep articles repeatedly emphasize vectorization, broadcasting, stable softmax, pairwise similarity, Conv2D, and ML metrics. Treat these as question-theme signals, not technical authorities.
